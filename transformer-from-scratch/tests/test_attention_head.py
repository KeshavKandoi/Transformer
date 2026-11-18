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
from src.model.attention_head import SingleAttentionHead
from src.model.config import ModelConfig
from src.model.embeddings import TransformerEmbedding


def test_independent_projections_and_shapes() -> None:
    head = SingleAttentionHead(6, 3)
    inputs = torch.randn(2, 4, 6)
    memory = torch.randn(2, 5, 6)
    projections = (head.Wq, head.Wk, head.Wv)
    assert len({linear.weight.data_ptr() for linear in projections}) == 3
    assert all(linear.bias is None and linear.weight.requires_grad for linear in projections)

    query, key, value = head.project_qkv(inputs, memory)
    assert query.shape == (2, 1, 4, 3)
    assert key.shape == value.shape == (2, 1, 5, 3)
    for projected, linear, original in zip((query, key, value), projections, (inputs, memory, memory)):
        torch.testing.assert_close(projected.squeeze(1), original @ linear.weight.T)

    context, weights = head(inputs, memory, return_weights=True)
    assert context.shape == (2, 4, 3)
    assert weights.shape == (2, 1, 4, 5)


def test_state_dict_preserves_evaluation_output(tmp_path: Path) -> None:
    head = SingleAttentionHead(6, 3, dropout=0.5).eval()
    inputs = torch.randn(2, 4, 6)
    expected = head(inputs)
    path = tmp_path / "attention.pt"
    torch.save(head.state_dict(), path)
    restored = SingleAttentionHead(6, 3, dropout=0.5).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(inputs), expected, atol=0, rtol=0)


def test_invalid_head_inputs() -> None:
    head = SingleAttentionHead(6, 3)
    inputs = torch.zeros(2, 4, 6)
    with pytest.raises(ValueError, match="query_input must have shape"):
        head(inputs[0])
    with pytest.raises(ValueError, match="key_value_input must have shape"):
        head(inputs, torch.zeros(2, 4, 7))
    with pytest.raises(ValueError, match="same batch size"):
        head(inputs, inputs[:1])
    with pytest.raises(TypeError, match="floating-point"):
        head(inputs.long())


def test_head_dimension_and_older_model_config(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text("model: {d_model: 24, num_heads: 4}\n", encoding="utf-8")
    model = ModelConfig.load(path)
    assert model.head_dim == 6
    assert model.attention_dropout == 0.1
    with pytest.raises(ValueError, match="divisible"):
        _ = ModelConfig(d_model=24, num_heads=5).head_dim
    with pytest.raises(ValueError, match="num_heads"):
        _ = ModelConfig(num_heads=0).head_dim


def test_real_batch_all_attention_modes_and_gradients(tmp_path: Path) -> None:
    torch.manual_seed(42)
    data_path = ROOT / "configs/data_config.yaml"
    processed, artifacts = tmp_path / "processed", tmp_path / "tokenizer"
    prepare(Namespace(
        config=data_path,
        input=ROOT / "data/raw/sample_parallel.csv",
        processed_dir=processed,
        tokenizer_dir=artifacts,
    ))
    data = DataConfig.load(data_path)
    model = ModelConfig.load(ROOT / "configs/model_config.yaml")
    tokenizer = BilingualBPETokenizer.load(artifacts)
    pairs, _ = read_parallel_csv(processed / "train.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(dataset, batch_size=data.batch_size, collate_fn=ParallelCollator(tokenizer.pad_id))
    batch = next(iter(loader))

    embedding_args = (tokenizer.vocab_size, model.d_model, model.max_seq_len, model.dropout, tokenizer.pad_id)
    encoder = TransformerEmbedding(*embedding_args).eval()
    decoder = TransformerEmbedding(*embedding_args).eval()
    encoder_x = encoder(batch["encoder_input_ids"])
    decoder_x = decoder(batch["decoder_input_ids"])
    assert encoder_x.size(1) != decoder_x.size(1)
    cases = (
        (encoder_x, None, batch["encoder_attention_mask"]),
        (decoder_x, None, batch["decoder_attention_mask"]),
        (decoder_x, encoder_x, batch["cross_attention_mask"]),
    )
    for query, memory, mask in cases:
        head = SingleAttentionHead(model.d_model, model.head_dim, model.attention_dropout).eval()
        output, weights = head(query, memory, mask, return_weights=True)
        key_length = query.size(1) if memory is None else memory.size(1)
        assert output.shape == (*query.shape[:2], model.head_dim)
        assert weights.shape == (query.size(0), 1, query.size(1), key_length)
        allowed = mask.expand_as(weights)
        assert torch.count_nonzero(weights[~allowed]) == 0
        torch.testing.assert_close(weights.sum(-1), allowed.any(-1).float())

        parameters = tuple(head.parameters())
        embedding_inputs = (query,) if memory is None else (query, memory)
        gradients = torch.autograd.grad(output.square().mean(), (*parameters, *embedding_inputs),
                                        retain_graph=True)
        assert all(torch.isfinite(grad).all() and torch.count_nonzero(grad) for grad in gradients)

    output, _ = head(decoder_x, encoder_x, batch["cross_attention_mask"], return_weights=True)
    output.square().mean().backward()
    for embedding in (encoder, decoder):
        grad = embedding.token_embedding.embedding.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)
        assert torch.count_nonzero(grad[tokenizer.pad_id]) == 0
