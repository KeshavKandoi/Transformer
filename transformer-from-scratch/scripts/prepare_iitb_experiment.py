"""Prepare official IITB splits, with test > validation > train leakage priority."""

import argparse
import json
import os
import random
import re
import sys
import tempfile
from dataclasses import asdict
from itertools import chain
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_iitb_dataset import DATASET, FILES, LICENSE, PUBLISHER
from src.data.cleaning import TranslationPair, read_parallel_csv, write_parallel_csv
from src.data.config import ROOT, DataConfig
from src.data.corpus import CorpusPolicy, filter_pairs, leakage_groups
from src.data.prepared import file_digest, write_preparation_manifest
from src.data.tokenizer import BilingualBPETokenizer

SPLIT_FILES = {"train": FILES["train"], "val": FILES["validation"], "test": FILES["test"]}
PRIORITY = ("test", "val", "train")


def verify_source(source_dir: Path) -> dict:
    metadata = json.loads((source_dir / "source_metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("dataset") != DATASET or metadata.get("publisher") != PUBLISHER
        or metadata.get("license") != LICENSE or not metadata.get("license_warning")
        or not re.fullmatch(r"[a-f0-9]{40}", metadata.get("revision", ""))
        or metadata.get("source_url") != f"https://huggingface.co/datasets/{DATASET}"
    ):
        raise ValueError("IITB source identity, exact revision or license provenance is missing or incompatible")
    checksums = {name: file_digest(source_dir / name) for name in SPLIT_FILES.values()}
    if metadata.get("files") != checksums or metadata.get("selected_train_sha256") != checksums[FILES["train"]]:
        raise ValueError("IITB source CSV checksums do not match acquisition metadata")
    return metadata


def remove_leakage(
    splits: dict[str, list[TranslationPair]], threshold: float,
) -> tuple[dict[str, list[TranslationPair]], dict]:
    # Track occurrences separately, including identical full pairs in different splits.
    aliases = {}
    unique = {}
    for split in PRIORITY:
        for pair in splits[split]:
            aliases.setdefault(pair, set()).add(split)
            unique[pair] = None
    groups = leakage_groups(list(unique), threshold)
    keep = {name: set() for name in PRIORITY}
    cross_groups = 0
    for group in groups:
        present = set().union(*(aliases[pair] for pair in group))
        if len(present) > 1:
            cross_groups += 1
        winner = next(name for name in PRIORITY if name in present)
        keep[winner].update(pair for pair in group if winner in aliases[pair])
    retained = {name: [pair for pair in rows if pair in keep[name]] for name, rows in splits.items()}
    return retained, {
        "groups": len(groups), "multi_pair_groups": sum(len(group) > 1 for group in groups),
        "largest_group_distinct_pairs": max(map(len, groups), default=0),
        "cross_split_groups": cross_groups, "remaining_cross_split_groups": 0,
        "removed": {name: len(splits[name]) - len(retained[name]) for name in splits},
        "priority": list(PRIORITY), "trigram_jaccard": threshold,
    }


def require_nonempty(splits: dict[str, list[TranslationPair]]) -> None:
    empty = [name for name, rows in splits.items() if not rows]
    if empty:
        raise ValueError(f"Filtering left empty official splits: {', '.join(empty)}")


def train_bpe(
    splits: dict[str, list[TranslationPair]], config: DataConfig,
) -> tuple[BilingualBPETokenizer, dict[str, list[TranslationPair]], dict[str, int], int]:
    retained = dict(splits)
    rejected = dict.fromkeys(splits, 0)

    def fits(pair: TranslationPair, tokenizer: BilingualBPETokenizer) -> bool:
        return (len(tokenizer.encode(pair.source)) < config.max_source_length
                and len(tokenizer.encode(pair.target)) < config.max_target_length)

    # Refit if the training vocabulary included a row that must be quarantined.
    for passes in range(1, 21):
        tokenizer = BilingualBPETokenizer.train(
            chain.from_iterable((row.source, row.target) for row in retained["train"]), config.vocab_size,
        )
        eligible = [pair for pair in retained["train"] if fits(pair, tokenizer)]
        removed = len(retained["train"]) - len(eligible)
        if not removed:
            break
        rejected["train"] += removed
        retained["train"] = eligible
        require_nonempty(retained)
    else:
        raise ValueError("Training token-length filtering did not stabilize in 20 passes; increase sequence limits")

    for name in ("val", "test"):
        eligible = [pair for pair in retained[name] if fits(pair, tokenizer)]
        rejected[name] = len(retained[name]) - len(eligible)
        retained[name] = eligible
    require_nonempty(retained)
    return tokenizer, retained, rejected, passes


def describe_lengths(rows: list[TranslationPair], tokenizer: BilingualBPETokenizer) -> dict:
    result = {}
    for side in ("source", "target"):
        texts = [getattr(pair, side) for pair in rows]
        result[side] = {}
        for unit, values in (
            ("characters", [len(text) for text in texts]),
            ("tokens_with_boundary", [len(tokenizer.encode(text)) + 1 for text in texts]),
        ):
            ordered = sorted(values)
            result[side][unit] = {
                label: ordered[round((len(ordered) - 1) * fraction)]
                for label, fraction in (("min", 0), ("p50", 0.5), ("p95", 0.95), ("max", 1))
            }
    return result


def prepare(
    source_dir: Path, output: Path, config: DataConfig, policy: CorpusPolicy,
    *, train_limit: int | None = None, min_chars: int = 1,
) -> dict:
    if min_chars < 1 or min_chars > policy.max_chars or (train_limit is not None and train_limit < 1):
        raise ValueError("Invalid minimum length or train limit")
    if any((output / item).exists() for item in ("processed", "tokenizer")):
        raise ValueError("Prepared output exists; choose a new experiment directory")
    source_metadata = verify_source(source_dir)
    splits, stats = {}, {}
    for name, filename in SPLIT_FILES.items():
        rows, csv_stats = read_parallel_csv(source_dir / filename)
        expected = (source_metadata["selected_train_rows"] if name == "train"
                    else source_metadata["official_split_sizes"]["validation" if name == "val" else "test"])
        if csv_stats.rows != expected:
            raise ValueError(f"Unexpected source row count for {name}")
        minimum = [pair for pair in rows if min(len(pair.source), len(pair.target)) >= min_chars]
        side_flags: dict[str, int] = {}
        splits[name], rejected = filter_pairs(minimum, policy, side_flags=side_flags)
        stats[name] = {
            **asdict(csv_stats), "quality_rejected": rejected, "side_quality_flags": side_flags,
            "below_minimum": len(rows) - len(minimum), "quality_kept": len(splits[name]),
        }
    splits, leakage = remove_leakage(splits, policy.near_threshold)
    require_nonempty(splits)
    before_limit = len(splits["train"])
    if train_limit is not None and train_limit < before_limit:
        selection = sorted(random.Random(config.seed).sample(range(before_limit), train_limit))
        splits["train"] = [splits["train"][index] for index in selection]
    tokenizer, retained, rejected, passes = train_bpe(splits, config)

    for name in retained:
        stats[name].update({
            "post_bpe_quarantined": rejected[name], "kept": len(retained[name]),
            "coverage": len(retained[name]) / stats[name]["rows"],
            "lengths": describe_lengths(retained[name], tokenizer),
        })
    for text in ("We are friends.", "हम दोस्त हैं।"):
        if tokenizer.decode(tokenizer.encode(text)) != text:
            raise ValueError("Tokenizer round-trip failed")
    # Omit access timestamps so an equivalent offline reconstruction has the same identity.
    provenance = {key: value for key, value in source_metadata.items() if key not in ("retrieved_at_utc", "requested_revision")}
    metadata = {
        "source": provenance, "policy": asdict(policy), "minimum_chars": min_chars,
        "splits": stats, "leakage": leakage, "train_limit": train_limit, "train_before_limit": before_limit,
        "bpe_training_split": "train", "bpe_passes": passes, "vocab_size": tokenizer.vocab_size,
        "vocab_target": config.vocab_size, "token_limits": {"source": config.max_source_length, "target": config.max_target_length},
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        staging = Path(scratch)
        processed, artifacts = staging / "processed", staging / "tokenizer"
        for name, rows in retained.items():
            write_parallel_csv(processed / f"{name}.csv", rows)
        tokenizer.save(artifacts)
        metadata["bpe_training_csv_sha256"] = file_digest(processed / "train.csv")
        # Deterministic length-stratified sample for private manual alignment review.
        with (processed / "alignment_audit.jsonl").open("w", encoding="utf-8") as stream:
            for name, rows in retained.items():
                ordered = sorted(rows, key=lambda row: (len(row.source), row.source, row.target))
                indices = sorted({round((len(ordered) - 1) * index / 19) for index in range(20)})
                for index in indices:
                    stream.write(json.dumps({"split": name, **asdict(ordered[index])}, ensure_ascii=False) + "\n")
        write_preparation_manifest(processed, artifacts, source_dir / FILES["train"], config,
                                   corpus=metadata, input_sha256=source_metadata["selected_train_sha256"])
        os.rename(artifacts, output / "tokenizer")
        try:
            os.rename(processed, output / "processed")
        except BaseException:
            os.rename(output / "tokenizer", artifacts)
            raise
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/experiment_data.yaml")
    parser.add_argument("--train-limit", type=int, help="Optional seeded smoke subset after decontamination")
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--max-word-ratio", type=float, default=4.0)
    parser.add_argument("--near-threshold", type=float, default=0.8)
    parser.add_argument("--language-min-letters", type=int, default=25)
    parser.add_argument("--language-confidence", type=float, default=0.95)
    args = parser.parse_args()
    try:
        result = prepare(
            args.source_dir, args.output, DataConfig.load(args.data_config),
            CorpusPolicy(args.max_chars, args.max_words, args.max_word_ratio, args.near_threshold,
                         args.language_min_letters, args.language_confidence),
            train_limit=args.train_limit, min_chars=args.min_chars,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"IITB preparation failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
