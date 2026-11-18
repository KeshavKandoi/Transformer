"""Clean and split a parallel corpus, then train its bilingual tokenizer."""

import argparse
import logging
import sys
from itertools import chain
from pathlib import Path

# Support direct execution with python scripts/prepare_data.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import read_parallel_csv, split_pairs, write_parallel_csv
from src.data.config import ROOT, DataConfig, data_parser
from src.data.prepared import write_preparation_manifest
from src.data.tokenizer import BilingualBPETokenizer


def prepare(args: argparse.Namespace) -> None:
    config = DataConfig.load(args.config)
    pairs, stats = read_parallel_csv(args.input)
    splits = split_pairs(pairs, config.seed, (config.train_ratio, config.val_ratio, config.test_ratio))
    train_texts = chain.from_iterable((pair.source, pair.target) for pair in splits["train"])
    tokenizer = BilingualBPETokenizer.train(train_texts, config.vocab_size)
    logging.info(
        "Read %d rows, kept %d pairs, dropped %d invalid and %d duplicate rows",
        stats.rows, len(pairs), stats.invalid, stats.duplicates,
    )
    for split, split_rows in splits.items():
        path = args.processed_dir / f"{split}.csv"
        write_parallel_csv(path, split_rows)
        logging.info(
            "%s: %d pairs, %d distinct sources -> %s",
            split, len(split_rows), len({pair.source for pair in split_rows}), path,
        )
    tokenizer.save(args.tokenizer_dir)
    write_preparation_manifest(args.processed_dir, args.tokenizer_dir, args.input, config)
    logging.info("Saved vocabulary of %d tokens to %s", tokenizer.vocab_size, args.tokenizer_dir)


def main() -> None:
    parser = data_parser(__doc__ or "")
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/sample_parallel.csv")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        prepare(parser.parse_args())
    except (OSError, ValueError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
