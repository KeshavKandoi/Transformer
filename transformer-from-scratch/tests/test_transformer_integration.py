import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from scripts.inspect_full_transformer import masked_cross_entropy
from scripts.prepare_data import prepare
from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer


def test_all_real_splits_default_width_and_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(42)
    config_path = ROOT / "configs/data_config.yaml"
    data = DataConfig.load(config_path)
    config = ModelConfig.load(ROOT / "configs/model_config.yaml")
    config.validate_sequence_lengths(data.max_source_length, data.max_target_length)
    processed, artifacts = tmp_path / "processed", tmp_path / "tokenizer"
    prepare(Namespace(
        config=config_path, input=ROOT / "data/raw/sample_parallel.csv",
        processed_dir=processed, tokenizer_dir=artifacts,
    ))
    tokenizer = BilingualBPETokenizer.load(artifacts)
    metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    assert tokenizer.vocab_size == metadata["vocab_size"]
    assert tokenizer.pad_id == metadata["special_token_ids"]["<pad>"]
    args = (tokenizer.vocab_size, tokenizer.pad_id, config)
    model = EncoderDecoderTransformer(*args).eval()
    for split in ("train", "val", "test"):
        pairs, _ = read_parallel_csv(processed / f"{split}.csv")
        dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
        loader = DataLoader(dataset, batch_size=data.batch_size, collate_fn=ParallelCollator(tokenizer.pad_id))
        batch = next(iter(loader))
        labels = batch.pop("decoder_target_ids")
        with torch.no_grad():
            details = model(**batch, return_diagnostics=True)
            assert details.encoder_memory.shape == (*batch["encoder_input_ids"].shape, config.d_model)
            assert details.decoder_hidden_states.shape == (*labels.shape, config.d_model)
            assert details.logits.shape == (*labels.shape, metadata["vocab_size"])
            assert torch.isfinite(details.logits).all()
            torch.testing.assert_close(model(**batch), details.logits)

        if split == "train":
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            weight = model.output_projection.projection.weight
            before = weight.detach().clone()
            loss = masked_cross_entropy(model(**batch), labels, tokenizer.pad_id)
            assert torch.isfinite(loss)
            loss.backward()
            assert weight.grad is not None and torch.count_nonzero(weight.grad)
            optimizer.step()
            assert not torch.equal(weight, before)
            model.zero_grad(set_to_none=True)

    path = tmp_path / "transformer.pt"
    torch.save(model.state_dict(), path)
    restored = EncoderDecoderTransformer(*args).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch), model(**batch))


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No CUDA or MPS device available",
)
def test_accelerator_forward_backward() -> None:
    device = "cuda" if torch.cuda.is_available() else "mps"
    config = ModelConfig(d_model=16, d_ff=32, num_encoder_layers=1, num_decoder_layers=1)
    model = EncoderDecoderTransformer(32, 0, config).to(device)
    source = torch.tensor([[4, 5, 0]], device=device)
    target = torch.tensor([[2, 7]], device=device)
    logits = model(source, target)
    assert logits.device.type == device and torch.isfinite(logits).all()
    masked_cross_entropy(logits, torch.tensor([[7, 3]], device=device), 0).backward()
    gradient = model.encoder.embedding.token_embedding.embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)


def test_tiny_fixed_batch_optimization() -> None:
    torch.manual_seed(42)
    config = ModelConfig(
        d_model=16, d_ff=32, dropout=0.0, attention_dropout=0.0,
        num_encoder_layers=1, num_decoder_layers=1,
    )
    model = EncoderDecoderTransformer(32, 0, config)
    source = torch.tensor([[4, 5, 6, 0], [7, 8, 9, 10]])
    target = torch.tensor([[2, 11, 12], [2, 13, 0]])
    labels = torch.tensor([[11, 12, 3], [13, 3, 0]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    with torch.no_grad():
        initial = masked_cross_entropy(model(source, target), labels, 0).item()
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        masked_cross_entropy(model(source, target), labels, 0).backward()
        optimizer.step()
    with torch.no_grad():
        final = masked_cross_entropy(model(source, target), labels, 0).item()
    print(f"Fixed batch cross-entropy after eight AdamW updates: {initial:.6f} -> {final:.6f}")
    assert final < initial
