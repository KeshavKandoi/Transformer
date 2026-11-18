"""Show real token IDs, their decoded text, and the masks for one batch."""

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
from src.data.config import DataConfig, data_parser
from src.data.dataset import ParallelDataset
from src.data.masks import causal_attention_mask
from src.data.tokenizer import BilingualBPETokenizer


def print_mask(label: str, mask: Tensor) -> None:
    print(f"{label} [{', '.join(map(str, mask.shape))}]")
    for row in mask.tolist():
        print(" ".join("T" if allowed else "F" for allowed in row))


def inspect(args: argparse.Namespace) -> None:
    config = DataConfig.load(args.config)
    splits = {name: read_parallel_csv(args.processed_dir / f"{name}.csv")[0]
              for name in ("train", "val", "test")}
    tokenizer = BilingualBPETokenizer.load(args.tokenizer_dir)
    dataset = ParallelDataset(
        splits[args.split], tokenizer, config.max_source_length, config.max_target_length
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=ParallelCollator(tokenizer.pad_id),
        generator=torch.Generator().manual_seed(config.seed),
    )
    batch = next(iter(loader))
    print("Dataset sizes:", ", ".join(f"{name}={len(pairs)}" for name, pairs in splits.items()))
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample English: {dataset.pairs[0].source}")
    print(f"Sample Hindi:   {dataset.pairs[0].target}")
    for name in ("encoder_input_ids", "decoder_input_ids", "decoder_target_ids"):
        ids = batch[name][0].tolist()
        print(f"{name}: {ids}")
        print(f"  Decoded content: {tokenizer.decode(ids)!r}")

    print("\nBatch tensor shapes")
    for name, tensor in batch.items():
        print(f"  {name}: {list(tensor.shape)} ({tensor.dtype})")

    print("\nT = allowed key, F = blocked key. Queries are rows in attention matrices.")
    print_mask("Encoder key padding mask (all examples)", batch["encoder_attention_mask"][:, 0, 0])
    print_mask("Decoder key padding mask (all examples)", batch["decoder_input_ids"].ne(tokenizer.pad_id))
    print_mask("Causal mask", causal_attention_mask(batch["decoder_input_ids"].size(1)))
    print_mask("Decoder attention mask (first example)", batch["decoder_attention_mask"][0, 0])
    print_mask("Cross-attention mask (first example)", batch["cross_attention_mask"][0, 0])


def main() -> None:
    parser = data_parser(__doc__ or "")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        inspect(parser.parse_args())
    except FileNotFoundError as error:
        logging.error("%s. Run scripts/prepare_data.py with matching output directories.", error)
        raise SystemExit(1) from error
    except (OSError, ValueError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
