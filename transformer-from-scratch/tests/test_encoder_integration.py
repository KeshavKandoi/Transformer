from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from scripts.prepare_data import prepare
from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.embeddings import TransformerEmbedding
from src.model.encoder import TransformerEncoder
from src.model.multi_head_attention import MultiHeadAttention


def test_real_batch_encoder_memory_checkpoint_and_cross_attention(tmp_path: Path) -> None:
    torch.manual_seed(42)
    data_path = ROOT / "configs/data_config.yaml"
    data = DataConfig.load(data_path)
    config = ModelConfig.load(ROOT / "configs/model_config.yaml")
    processed, artifacts = tmp_path / "processed", tmp_path / "tokenizer"
    prepare(Namespace(
        config=data_path, input=ROOT / "data/raw/sample_parallel.csv",
        processed_dir=processed, tokenizer_dir=artifacts,
    ))
    tokenizer = BilingualBPETokenizer.load(artifacts)
    pairs, _ = read_parallel_csv(processed / "train.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(dataset, batch_size=data.batch_size, collate_fn=ParallelCollator(tokenizer.pad_id))
    batch = next(iter(loader))
    source = batch["encoder_input_ids"]
    target = batch["decoder_input_ids"]
    assert source.size(1) != target.size(1)
    encoder = TransformerEncoder(tokenizer.vocab_size, tokenizer.pad_id, config).eval()
    memory, maps = encoder(source, batch["encoder_attention_mask"], return_attention_weights=True)
    b, s = source.shape
    assert memory.shape == (b, s, config.d_model)
    assert maps.shape == (b, config.num_encoder_layers, config.num_heads, s, s)
    valid = source.ne(tokenizer.pad_id)
    assert not valid.all()
    assert torch.count_nonzero(memory[~valid]) == 0
    assert torch.count_nonzero(maps.masked_select(~valid[:, None, None, None, :])) == 0
    assert torch.isfinite(memory).all() and torch.isfinite(maps).all()
    torch.testing.assert_close(maps.sum(-1), torch.ones_like(maps[..., 0]))
    with torch.no_grad():
        torch.testing.assert_close(encoder(source), memory)
        positional = encoder.embedding.position_encoding
        token_vectors = encoder.embedding.token_embedding(source)
        positioned = positional(token_vectors)
        torch.testing.assert_close(encoder.embedding(source)[valid], positioned[valid])
        assert not torch.allclose(positioned[valid], token_vectors[valid])
        checkpoint = tmp_path / "encoder.pt"
        torch.save(encoder.state_dict(), checkpoint)
        restored = TransformerEncoder(tokenizer.vocab_size, tokenizer.pad_id, config).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        actual_memory, actual_maps = restored(source, return_attention_weights=True)
        torch.testing.assert_close(actual_memory, memory)
        torch.testing.assert_close(actual_maps, maps)

    decoder_embedding = TransformerEmbedding(
        tokenizer.vocab_size, config.d_model, config.max_seq_len, config.dropout, tokenizer.pad_id,
    ).eval()
    queries = decoder_embedding(target)
    cross_attention = MultiHeadAttention(config.d_model, config.num_heads, config.attention_dropout).eval()
    context, weights = cross_attention(
        queries, memory, batch["cross_attention_mask"], return_attention_weights=True,
    )
    assert context.shape == (*target.shape, config.d_model)
    assert weights.shape == (b, config.num_heads, target.size(1), s)
    assert torch.isfinite(context).all() and torch.isfinite(weights).all()
    assert torch.count_nonzero(weights.masked_select(~valid[:, None, None, :])) == 0
    memory.retain_grad()
    context.square().mean().backward()
    parameter = encoder.embedding.token_embedding.embedding.weight
    for gradient in (memory.grad, parameter.grad):
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No CUDA or MPS device available",
)
def test_encoder_accelerator_smoke() -> None:
    device = "cuda" if torch.cuda.is_available() else "mps"
    config = ModelConfig(d_model=16, d_ff=32)
    encoder = TransformerEncoder(32, 0, config).eval()
    ids = torch.tensor([[4, 5, 6, 0]])
    expected = encoder(ids)
    output = encoder.to(device)(ids.to(device))
    torch.testing.assert_close(output.cpu(), expected, atol=1e-5, rtol=1e-5)
    output[..., 0].square().mean().backward()
    weight = encoder.embedding.token_embedding.embedding.weight
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
