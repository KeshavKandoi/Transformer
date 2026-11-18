"""Acquire the pinned IITB parallel Parquet files without executing remote code."""

import argparse
import csv
import json
import platform
import random
import re
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import TranslationPair, write_parallel_csv
from src.data.prepared import file_digest

DATASET = "cfilt/iitb-english-hindi"
REVISION = "e104eb47979d570d3df7af4912da85f83efde6fc"
PUBLISHER = "Center for Indian Language Technology (CFILT), IIT Bombay"
PUBLISHER_URL = "https://www.cfilt.iitb.ac.in/iitb_parallel/"
LICENSE = "CC BY-NC 4.0"
WARNING = "The publisher says material compiled from other sources remains under its respective licenses. Noncommercial local research only; redistribution of data and checkpoints is not assumed."
FILES = {"train": "train_selected.csv", "validation": "validation_official.csv", "test": "test_official.csv"}


def resolve_revision(dataset: str, revision: str, offline: bool) -> str:
    if dataset != DATASET:
        raise ValueError(f"Only {DATASET} is supported by this importer")
    if re.fullmatch(r"[a-f0-9]{40}", revision):
        return revision
    if offline:
        raise ValueError("Offline acquisition requires an exact 40-character revision SHA")
    info = HfApi().dataset_info(dataset, revision=revision, token=False)
    if info.gated or info.private:
        raise ValueError("Dataset access requires publisher authorization or acknowledgement; this command uses public access")
    if not info.sha or not re.fullmatch(r"[a-f0-9]{40}", info.sha):
        raise ValueError("Could not resolve an exact dataset revision")
    return info.sha


def convert_record(record: dict) -> TranslationPair:
    value = record.get("translation")
    if not isinstance(value, dict) or any(not isinstance(value.get(key), str) for key in ("en", "hi")):
        raise ValueError("Invalid dataset record: translation.en and translation.hi must be strings")
    return TranslationPair(value["en"], value["hi"])


def parquet_rows(file: pq.ParquetFile) -> Iterable[TranslationPair]:
    for batch in file.iter_batches(batch_size=4096, columns=["translation"]):
        for record in batch.to_pylist():
            yield convert_record(record)


def validate_schema(file: pq.ParquetFile, split: str) -> None:
    schema = file.schema_arrow
    field = schema.field("translation").type if "translation" in schema.names else None
    if field is None or not pa.types.is_struct(field):
        raise ValueError(f"{split} must have a translation struct")
    for name in ("en", "hi"):
        if field.get_field_index(name) < 0:
            raise ValueError(f"{split} is missing translation.{name}")
        dtype = field.field(name).type
        if not (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)):
            raise ValueError(f"{split} translation.{name} must be a string")


def reservoir(rows: Iterable[TranslationPair], count: int, seed: int) -> tuple[list[TranslationPair], int]:
    if count < 1:
        raise ValueError("train-samples must be positive")
    rng = random.Random(seed)
    selected: list[tuple[int, TranslationPair]] = []
    seen = 0
    for index, row in enumerate(rows):
        seen = index + 1
        if index < count:
            selected.append((index, row))
        else:
            slot = rng.randrange(seen)
            if slot < count:
                selected[slot] = (index, row)
    if seen < count:
        raise ValueError(f"Requested {count} training rows, but only {seen} exist")
    return [row for _, row in sorted(selected)], seen


def acquire(dataset: str, revision: str, count: int, seed: int, output: Path, *, offline: bool, force: bool) -> dict:
    if count < 1:
        raise ValueError("train-samples must be positive")
    if output.exists() and (
        not force or output.is_symlink() or not (output / "source_metadata.json").is_file()
        or set(path.name for path in output.iterdir()) != {*FILES.values(), "source_metadata.json"}
    ):
        raise ValueError(f"Output exists: {output}. Choose a new path or use --force on a complete source directory.")
    sha = resolve_revision(dataset, revision, offline)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Acquiring official Parquet splits at {sha} (train stream sampled to {count:,})", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        staging = Path(scratch) / "source"
        staging.mkdir()
        split_sizes = {}
        source_files = {}
        cached = {}
        for split in FILES:
            remote = f"data/{split}-00000-of-00001.parquet"
            path = Path(hf_hub_download(
                dataset, remote, repo_type="dataset", revision=sha, token=False, local_files_only=offline,
                cache_dir=output.parent.parent / ".iitb-hub-cache",
            ))
            with pq.ParquetFile(path) as file:
                validate_schema(file, split)
                size = file.metadata.num_rows
                print(f"{split}: {size:,} rows, translation.en and translation.hi", file=sys.stderr, flush=True)
                if size < 1:
                    raise ValueError(f"Published {split} is empty")
            split_sizes[split] = size
            cached[split] = path
            source_files[split] = {"path": remote, "sha256": file_digest(path)}
        for split, destination in FILES.items():
            with pq.ParquetFile(cached[split]) as file:
                rows = parquet_rows(file)
                if split == "train":
                    sample, seen = reservoir(rows, count, seed)
                    if seen != split_sizes[split]:
                        raise ValueError("Training row count differs from Parquet metadata")
                    write_parallel_csv(staging / destination, sample)
                else:
                    with (staging / destination).open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(("source", "target"))
                        seen = 0
                        for row in rows:
                            writer.writerow((row.source, row.target))
                            seen += 1
                    if seen != split_sizes[split]:
                        raise ValueError(f"{split} row count differs from Parquet metadata")

        checksums = {name: file_digest(staging / name) for name in FILES.values()}
        metadata = {
            "dataset": dataset, "dataset_name": "IIT Bombay English–Hindi Parallel Corpus",
            "publisher": PUBLISHER, "source_url": f"https://huggingface.co/datasets/{dataset}",
            "publisher_url": PUBLISHER_URL, "publisher_version": "3.1",
            "revision": sha, "requested_revision": revision, "license": LICENSE, "license_warning": WARNING,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "official_split_sizes": split_sizes,
            "selected_train_rows": count, "selected_train_sha256": checksums[FILES["train"]],
            "selection": "Algorithm R reservoir over published train in Parquet order, output sorted by original row index",
            "seed": seed, "files": checksums, "published_files": source_files,
            "versions": {"python": platform.python_version(), "pyarrow": version("pyarrow"),
                         "huggingface_hub": version("huggingface-hub")},
        }
        (staging / "source_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        backup = Path(scratch) / "previous"
        if output.exists():
            output.rename(backup)
        try:
            staging.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = acquire(args.dataset, args.revision, args.train_samples, args.seed, args.output,
                         offline=args.offline, force=args.force)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"IITB acquisition failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
