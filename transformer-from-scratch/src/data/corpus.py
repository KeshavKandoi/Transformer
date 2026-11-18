"""Quality filters and deterministic duplicate grouping for English–Hindi CSVs."""

import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from langid.langid import LanguageIdentifier, model as language_profiles

from .cleaning import TranslationPair


@dataclass(frozen=True)
class CorpusPolicy:
    max_chars: int = 400
    max_words: int = 80
    max_word_ratio: float = 4.0
    near_threshold: float = 0.8
    language_min_letters: int = 25
    language_confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.max_chars < 1 or self.max_words < 1 or not 1 <= self.max_word_ratio < float("inf"):
            raise ValueError("Length limits must be positive and max_word_ratio must be finite and at least one")
        if not 0 < self.near_threshold <= 1:
            raise ValueError("near_threshold must be in (0, 1]")
        if self.language_min_letters < 1 or not 0 < self.language_confidence <= 1:
            raise ValueError("Language check requires a positive letter minimum and confidence in (0, 1]")


def comparison_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(
        char if char.isalnum() or unicodedata.category(char).startswith("M") else " "
        for char in normalized
    ).split())


def _dominant_script(text: str, script: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and 5 * sum(
        script in unicodedata.name(char, "") for char in letters
    ) >= 4 * len(letters)


@lru_cache(maxsize=1)
def _language_identifier() -> LanguageIdentifier:
    return LanguageIdentifier.from_modelstring(language_profiles, norm_probs=True)


def _language_mismatch(text: str, expected: str, policy: CorpusPolicy) -> bool:
    if sum(char.isalpha() for char in text) < policy.language_min_letters:
        return False
    language, confidence = _language_identifier().classify(text)
    return language != expected and confidence >= policy.language_confidence


def filter_pairs(
    pairs: list[TranslationPair], policy: CorpusPolicy, *, side_flags: dict[str, int] | None = None,
) -> tuple[list[TranslationPair], dict[str, int]]:
    accepted: dict[tuple[str, str], TranslationPair] = {}
    rejected: Counter[str] = Counter()

    def reject(reason: str, source_bad: bool, target_bad: bool) -> bool:
        if not (source_bad or target_bad):
            return False
        rejected[reason] += 1
        if side_flags is not None:
            for side, bad in (("source", source_bad), ("target", target_bad)):
                if bad:
                    key = f"{side}_{reason}"
                    side_flags[key] = side_flags.get(key, 0) + 1
        return True

    def invalid_format(text: str) -> bool:
        return any(unicodedata.category(char) == "Cf" and char not in "\u200c\u200d" for char in text)

    for original in pairs:
        pair = TranslationPair(" ".join(original.source.split()), " ".join(original.target.split()))
        if reject("unicode_format", invalid_format(pair.source), invalid_format(pair.target)):
            continue
        if reject("script", not _dominant_script(pair.source, "LATIN"), not _dominant_script(pair.target, "DEVANAGARI")):
            continue
        source_words, target_words = len(pair.source.split()), len(pair.target.split())
        if reject("length",
                  len(pair.source) > policy.max_chars or source_words > policy.max_words,
                  len(pair.target) > policy.max_chars or target_words > policy.max_words):
            continue
        if max(source_words, target_words) / min(source_words, target_words) > policy.max_word_ratio:
            rejected["length_ratio"] += 1
            continue
        if reject("language", _language_mismatch(pair.source, "en", policy),
                  _language_mismatch(pair.target, "hi", policy)):
            continue
        key = (comparison_key(pair.source), comparison_key(pair.target))
        previous = accepted.get(key)
        if previous is not None:
            rejected["normalized_duplicate"] += 1
            if (pair.source, pair.target) >= (previous.source, previous.target):
                continue
        accepted[key] = pair
    return [accepted[key] for key in sorted(accepted)], dict(sorted(rejected.items()))


def _grams(key: str) -> set[str]:
    padded = f"^{key}$"
    return {padded[index:index + 3] for index in range(len(padded) - 2)}


def _link_near_duplicates(keys: list[str], threshold: Fraction, union) -> None:
    # Equal normalized sentences share one index even when their other side differs.
    representatives: dict[str, int] = {}
    for index, key in enumerate(keys):
        union(index, representatives.setdefault(key, index))
    grams = {index: _grams(key) for key, index in representatives.items()}
    frequencies = Counter(gram for features in grams.values() for gram in features)
    postings: dict[str, list[int]] = defaultdict(list)
    numerator, denominator = threshold.numerator, threshold.denominator
    for index in sorted(grams, key=lambda item: (len(grams[item]), item)):
        features = grams[index]
        size = len(features)
        minimum = (numerator * size + denominator - 1) // denominator
        # Jaccard matches share a gram in the rarity-ordered prefixes.
        prefix = sorted(features, key=lambda gram: (frequencies[gram], gram))[:size - minimum + 1]
        candidates = {other for gram in prefix for other in postings[gram] if len(grams[other]) >= minimum}
        for other in sorted(candidates):
            overlap = len(features & grams[other])
            if overlap * denominator >= numerator * (size + len(grams[other]) - overlap):
                union(index, other)
        for gram in prefix:
            postings[gram].append(index)


def leakage_groups(pairs: list[TranslationPair], threshold: float) -> list[list[TranslationPair]]:
    """Connect exact and trigram-near duplicates on either side before splitting."""
    parents = list(range(len(pairs)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    cutoff = Fraction(str(threshold))
    if not 0 < cutoff <= 1:
        raise ValueError("Near-duplicate threshold must be in (0, 1]")
    for side in ("source", "target"):
        _link_near_duplicates([comparison_key(getattr(pair, side)) for pair in pairs], cutoff, union)
    grouped: dict[int, list[TranslationPair]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        grouped[root(index)].append(pair)
    return list(grouped.values())
