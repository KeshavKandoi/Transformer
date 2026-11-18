"""Inspect embeddings and gradient flow on a real Phase 1 batch."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig, data_parser
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.embeddings import TransformerEmbedding


def preview(values: Tensor) -> list[float]:
    return [round(value, 6) for value in values[:10].detach().tolist()]


def inspect_side(label: str, module: TransformerEmbedding, ids: Tensor) -> None:
    module.eval()
    weight = module.token_embedding.embedding.weight
    positions = module.position_encoding.encoding
    is_pad = ids.eq(module.pad_id)
    row, position = (~is_pad).nonzero()[0].tolist()
    token_id = ids[row, position].item()
    with torch.no_grad():
        output = module(ids)
        scaled = module.token_embedding(ids)[row, position]
        torch.testing.assert_close(output[row, position], scaled + positions[position])
        padded_count = int(is_pad.sum())
        if padded_count and torch.count_nonzero(output[is_pad]).item():
            raise ValueError(f"{label} padded outputs must be zero")

    print(f"\n{label}")
    print(f"  Input IDs shape: {list(ids.shape)}")
    print(f"  First example IDs (up to 24): {ids[0, :24].tolist()}")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Raw embedding for token {token_id}: {preview(weight[token_id])}")
    print(f"  Scaled embedding: {preview(scaled)}")
    print(f"  Position 0: {preview(positions[0])}")
    if positions.size(0) > 1:
        print(f"  Position 1: {preview(positions[1])}")
    print(f"  Combined representation: {preview(output[row, position])}")
    print("  Evaluation formula check: passed")
    print(f"  Zero padded outputs verified: {padded_count} positions")
    print(f"  Embedding weights require gradients: {weight.requires_grad}")
    print(f"  Positional buffer requires gradients: {positions.requires_grad}")

    module(ids).sum().backward()
    gradient = weight.grad
    if gradient is None:
        raise ValueError(f"{label} embedding weights did not receive gradients")
    valid_norm = gradient[token_id].norm().item()
    padding_norm = gradient[module.pad_id].norm().item()
    if not valid_norm > 0 or padding_norm != 0 or positions.grad is not None:
        raise ValueError(f"{label} gradient-flow check failed")
    print(f"  Gradient norms (valid token, padding): {valid_norm:.6f}, {padding_norm:.6f}")


def inspect(args: argparse.Namespace) -> None:
    data = DataConfig.load(args.config)
    model = ModelConfig.load(args.model_config)
    model.validate_sequence_lengths(data.max_source_length, data.max_target_length)
    tokenizer = BilingualBPETokenizer.load(args.tokenizer_dir)
    pairs, _ = read_parallel_csv(args.processed_dir / f"{args.split}.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(
        dataset,
        batch_size=data.batch_size,
        collate_fn=ParallelCollator(tokenizer.pad_id),
        generator=torch.Generator().manual_seed(data.seed),
    )
    batch = next(iter(loader))

    torch.manual_seed(data.seed)
    embedding_args = (tokenizer.vocab_size, model.d_model, model.max_seq_len, model.dropout, tokenizer.pad_id)
    encoder_embedding = TransformerEmbedding(*embedding_args)
    decoder_embedding = TransformerEmbedding(*embedding_args)
    print(f"Actual vocabulary size: {tokenizer.vocab_size}")
    print(f"Padding ID: {tokenizer.pad_id}")
    print(f"Sample English: {pairs[0].source}")
    print(f"Sample Hindi: {pairs[0].target}")
    inspect_side("Encoder", encoder_embedding, batch["encoder_input_ids"])
    inspect_side("Decoder", decoder_embedding, batch["decoder_input_ids"])


def main() -> None:
    parser = data_parser(__doc__ or "")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model_config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        inspect(parser.parse_args())
    except FileNotFoundError as error:
        logging.error("%s. For missing Phase 1 artifacts, run scripts/prepare_data.py.", error)
        raise SystemExit(1) from error
    except (OSError, ValueError, AssertionError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
