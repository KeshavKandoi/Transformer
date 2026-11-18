from argparse import Namespace
from pathlib import Path

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
from src.model.multi_head_attention import MultiHeadAttention


def test_real_batch_shapes_masks_and_gradients_for_three_independent_modules(tmp_path: Path) -> None:
    config_path = ROOT / "configs/data_config.yaml"
    data = DataConfig.load(config_path)
    model = ModelConfig.load(ROOT / "configs/model_config.yaml")
    processed, artifacts = tmp_path / "processed", tmp_path / "tokenizer"
    prepare(Namespace(
        config=config_path,
        input=ROOT / "data/raw/sample_parallel.csv",
        processed_dir=processed,
        tokenizer_dir=artifacts,
    ))
    tokenizer = BilingualBPETokenizer.load(artifacts)
    pairs, _ = read_parallel_csv(processed / "train.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(dataset, batch_size=data.batch_size, collate_fn=ParallelCollator(tokenizer.pad_id))
    batch = next(iter(loader))

    torch.manual_seed(data.seed)
    embedding_args = (tokenizer.vocab_size, model.d_model, model.max_seq_len, model.dropout, tokenizer.pad_id)
    encoder = TransformerEmbedding(*embedding_args).eval()
    decoder = TransformerEmbedding(*embedding_args).eval()
    source = encoder(batch["encoder_input_ids"])
    target = decoder(batch["decoder_input_ids"])
    assert source.size(1) != target.size(1)

    modules = [
        MultiHeadAttention(model.d_model, model.num_heads, model.attention_dropout).eval()
        for _ in range(3)
    ]
    parameters = [parameter for module in modules for parameter in module.parameters()]
    assert len({parameter.untyped_storage().data_ptr() for parameter in parameters}) == len(parameters)
    cases = (
        (source, None, batch["encoder_attention_mask"]),
        (target, None, batch["decoder_attention_mask"]),
        (target, source, batch["cross_attention_mask"]),
    )
    losses = []
    for module, (query, memory, mask) in zip(modules, cases):
        output, weights = module(query, memory, mask, return_attention_weights=True)
        key_length = query.size(1) if memory is None else memory.size(1)
        assert output.shape == query.shape
        assert weights.shape == (query.size(0), model.num_heads, query.size(1), key_length)
        assert torch.isfinite(output).all() and torch.isfinite(weights).all()
        allowed = mask.expand_as(weights)
        assert torch.count_nonzero(weights[~allowed]) == 0
        torch.testing.assert_close(weights.sum(-1), allowed.any(-1).to(weights.dtype))

        loss = output.square().mean()
        inputs = (query,) if memory is None else (query, memory)
        gradients = torch.autograd.grad(loss, inputs, retain_graph=True)
        assert all(torch.isfinite(gradient).all() and torch.count_nonzero(gradient) for gradient in gradients)
        losses.append(loss)

    torch.stack(losses).sum().backward()
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad)
    for embedding in (encoder, decoder):
        gradient = embedding.token_embedding.embedding.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
        assert torch.count_nonzero(gradient[tokenizer.pad_id]) == 0
