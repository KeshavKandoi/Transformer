"""Import an explicitly selected local English–Hindi parallel CSV."""

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from itertools import chain
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import read_parallel_csv, split_groups, write_parallel_csv
from src.data.config import ROOT, DataConfig
from src.data.corpus import CorpusPolicy, filter_pairs, leakage_groups
from src.data.prepared import file_digest, write_preparation_manifest
from src.data.tokenizer import BilingualBPETokenizer

DEMO = ROOT / "data/raw/sample_parallel.csv"


def import_corpus(
    source: Path, destination: Path, config: DataConfig, policy: CorpusPolicy,
    *, name: str, license_id: str | None, provenance: str, demo: bool = False,
) -> dict:
    if demo and source.resolve() != DEMO.resolve():
        raise ValueError("Demo mode is reserved for the bundled sample")
    if not name.strip() or not provenance.strip() or (not demo and not (license_id or "").strip()):
        raise ValueError("A corpus name, license and provenance are required")
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}. Choose a fresh experiment directory.")
    input_digest = file_digest(source)
    pairs, raw_stats = read_parallel_csv(source)
    pairs, rejections = filter_pairs(pairs, policy)
    groups = leakage_groups(pairs, policy.near_threshold)
    if len(groups) < 3:
        raise ValueError("Need at least three independent pairs or duplicate clusters after filtering")
    splits = split_groups(groups, config.seed, (config.train_ratio, config.val_ratio, config.test_ratio))
    texts = chain.from_iterable((pair.source, pair.target) for pair in splits["train"])
    tokenizer = BilingualBPETokenizer.train(texts, config.vocab_size)
    limits = (config.max_source_length, config.max_target_length)
    max_tokens: dict[str, dict[str, int]] = {}
    too_long: dict[str, int] = {}
    for split, rows in splits.items():
        maxima = [0, 0]
        for pair in rows:
            lengths = [len(tokenizer.encode(text)) + 1 for text in (pair.source, pair.target)]
            maxima = [max(old, new) for old, new in zip(maxima, lengths, strict=True)]
            if any(length > limit for length, limit in zip(lengths, limits, strict=True)):
                too_long[split] = too_long.get(split, 0) + 1
        max_tokens[split] = dict(zip(("source", "target"), maxima, strict=True))
    if too_long:
        raise ValueError(
            f"Tokenized rows exceed sequence limits {limits}: {too_long}. "
            "Reduce the import length filters or raise the data and model sequence limits."
        )
    if file_digest(source) != input_digest:
        raise ValueError("The input file changed during import")
    metadata = {
        "name": name, "license": license_id, "provenance": provenance, "demo": demo,
        "language_identifier": "langid 1.1.6 (bundled profiles)",
        "policy": asdict(policy), "rows": asdict(raw_stats), "rejected": rejections,
        "kept": len(pairs), "groups": len(groups), "largest_group": max(map(len, groups)),
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "vocab_size": tokenizer.vocab_size, "maximum_tokens_including_boundary": max_tokens,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staging = Path(temp) / "corpus"
        processed, artifacts = staging / "processed", staging / "tokenizer"
        for split, rows in splits.items():
            write_parallel_csv(processed / f"{split}.csv", rows)
        tokenizer.save(artifacts)
        write_preparation_manifest(processed, artifacts, source, config, corpus=metadata, input_sha256=input_digest)
        staging.rename(destination)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_mode = parser.add_mutually_exclusive_group(required=True)
    input_mode.add_argument("--input", type=Path, help="Local UTF-8 CSV with source and target columns")
    input_mode.add_argument("--demo", action="store_true", help="Use the bundled smoke data explicitly")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment_data.yaml")
    parser.add_argument("--name")
    parser.add_argument("--license", dest="license_id", help="License identifier or URL for the selected corpus")
    parser.add_argument("--provenance", help="Publisher/release and source URL or local acquisition record")
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--max-word-ratio", type=float, default=4.0)
    parser.add_argument("--near-threshold", type=float, default=0.8)
    parser.add_argument("--language-min-letters", type=int, default=25)
    parser.add_argument("--language-confidence", type=float, default=0.95)
    args = parser.parse_args()
    if not args.demo and not all((args.name, args.license_id, args.provenance)):
        parser.error("--input requires --name, --license and --provenance")
    try:
        stats = import_corpus(
            DEMO if args.demo else args.input, args.output, DataConfig.load(args.config),
            CorpusPolicy(
                args.max_chars, args.max_words, args.max_word_ratio, args.near_threshold,
                args.language_min_letters, args.language_confidence,
            ),
            name="bundled-40-pair-smoke" if args.demo else args.name,
            license_id=None if args.demo else args.license_id,
            provenance="Repository bundled sample; smoke-only" if args.demo else args.provenance,
            demo=args.demo,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Import failed: {error}\n")


if __name__ == "__main__":
    main()
