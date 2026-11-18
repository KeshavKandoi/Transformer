"""Read, clean and split UTF-8 parallel text."""

import csv
import random
import unicodedata
from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path


@dataclass(frozen=True)
class TranslationPair:
    source: str
    target: str


@dataclass(frozen=True)
class CleaningStats:
    rows: int
    invalid: int
    duplicates: int


def _clean(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _valid(text: str) -> bool:
    return bool(text) and not any(
        unicodedata.category(char) == "Cc" and char not in "\t\r\n"
        or char == "\ufffd"
        for char in text
    )


def read_parallel_csv(path: Path) -> tuple[list[TranslationPair], CleaningStats]:
    pairs: dict[TranslationPair, None] = {}
    rows = invalid = duplicates = 0
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames or []
            if not {"source", "target"}.issubset(fields) or len(fields) != len(set(fields)):
                raise ValueError(f"{path} needs unambiguous source and target columns")
            for row in reader:
                rows += 1
                source = _clean(row["source"] or "")
                target = _clean(row["target"] or "")
                pair = TranslationPair(source, target)
                if None in row or None in row.values() or not (_valid(source) and _valid(target)):
                    invalid += 1
                elif pair in pairs:
                    duplicates += 1
                else:
                    pairs[pair] = None
    except (csv.Error, UnicodeError) as error:
        raise ValueError(f"Cannot read {path} as a UTF-8 CSV: {error}") from error
    if not pairs:
        raise ValueError(f"{path} contains no valid translation pairs")
    return list(pairs), CleaningStats(rows, invalid, duplicates)


def split_pairs(
    pairs: list[TranslationPair], seed: int, ratios: tuple[float, float, float]
) -> dict[str, list[TranslationPair]]:
    """Keep all translations of each normalized source in the same split."""
    if not all(isfinite(ratio) and ratio > 0 for ratio in ratios) or not isclose(sum(ratios), 1):
        raise ValueError("Split ratios must be positive and add up to 1")
    grouped: dict[str, list[TranslationPair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.source, []).append(pair)
    groups = list(grouped.values())
    names = ("train", "val", "test")
    if len(groups) < len(names):
        raise ValueError("A valid split needs at least three distinct English sources")
    random.Random(seed).shuffle(groups)
    # Stable sorting preserves seeded tie-breaking and places large groups first.
    groups.sort(key=len, reverse=True)
    splits: dict[str, list[TranslationPair]] = {name: [] for name in names}
    goals = dict(zip(names, (len(pairs) * ratio for ratio in ratios), strict=True))
    for index, group in enumerate(groups):
        empty = [name for name in names if not splits[name]]
        candidates = empty if len(groups) - index == len(empty) else names
        chosen = max(candidates, key=lambda name: goals[name] - len(splits[name]))
        splits[chosen].extend(group)
    return splits


def write_parallel_csv(path: Path, pairs: list[TranslationPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source", "target"))
        writer.writerows((pair.source, pair.target) for pair in pairs)
