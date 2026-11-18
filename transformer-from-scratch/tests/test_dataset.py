from itertools import combinations
from pathlib import Path

import pytest
import torch

from src.data.cleaning import CleaningStats, TranslationPair, read_parallel_csv, split_pairs, write_parallel_csv
from src.data.collator import ParallelCollator
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer


def test_reproducible_disjoint_splits(tmp_path: Path) -> None:
    path = tmp_path / "parallel.csv"
    pairs = [TranslationPair(f"Source {i}", f"पाठ {i}") for i in range(30)]
    pairs.append(TranslationPair("Source 0", "एक और पाठ"))
    write_parallel_csv(path, pairs)
    loaded, _ = read_parallel_csv(path)
    ratios = (0.8, 0.1, 0.1)

    splits = split_pairs(loaded, 42, ratios)
    assert splits == split_pairs(loaded, 42, ratios)
    assert splits != split_pairs(loaded, 13, ratios)
    assert sorted(map(len, splits.values())) == [3, 3, 25]
    assert set().union(*map(set, splits.values())) == set(pairs)
    for first, second in combinations(splits.values(), 2):
        assert set(first).isdisjoint(second)
        assert {pair.source for pair in first}.isdisjoint(pair.source for pair in second)


def test_cleaning(tmp_path: Path) -> None:
    path = tmp_path / "parallel.csv"
    path.write_text(
        "source,target\n cafe\u0301 , नमस्ते \ncafé,नमस्ते\nEmpty, \nShort\nExtra,पाठ,x\nBad,a\x00\n",
        encoding="utf-8",
    )
    assert read_parallel_csv(path) == ([TranslationPair("café", "नमस्ते")], CleaningStats(6, 4, 1))


@pytest.mark.parametrize(("contents", "message"), [
    (b"", "source and target columns"),
    (b"english,hindi\nhello,test\n", "source and target columns"),
    (b"source,target\nhello,\n", "no valid translation pairs"),
    (b'source,target\n"hello,test\n', "UTF-8 CSV"),
    (b"source,target\nhello,\xff\n", "UTF-8 CSV"),
])
def test_bad_csv(tmp_path: Path, contents: bytes, message: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match=message):
        read_parallel_csv(path)


def test_too_few_distinct_sources() -> None:
    with pytest.raises(ValueError, match="three distinct English sources"):
        split_pairs([TranslationPair("Hello", target) for target in ("नमस्ते", "प्रणाम", "नमस्कार")],
                    42, (0.8, 0.1, 0.1))


@pytest.mark.parametrize("limit", [1, 4, 128])
def test_shift_and_truncation(limit: int) -> None:
    pair = TranslationPair("We are friends", "हम दोस्त हैं")
    tokenizer = BilingualBPETokenizer.train(["Unrelated training text"], 260)
    example = ParallelDataset([pair], tokenizer, limit, limit)[0]
    decoder = example["decoder_input_ids"]
    labels = example["decoder_target_ids"]
    assert example["encoder_input_ids"] == [*tokenizer.encode(pair.source)[:limit - 1], tokenizer.eos_id]
    assert decoder == [tokenizer.bos_id, *tokenizer.encode(pair.target)[:limit - 1]]
    assert len(decoder) == len(labels)
    assert labels == [*decoder[1:], tokenizer.eos_id]
    assert tokenizer.eos_id not in decoder
    assert tokenizer.bos_id not in labels


def test_dynamic_padding_and_tensor_shapes() -> None:
    tokenizer = BilingualBPETokenizer.train(["Hello"], 260)
    pairs = [TranslationPair("A long source", "ह"), TranslationPair("Hi", "हम दोस्त हैं")]
    dataset = ParallelDataset(pairs, tokenizer, 128, 128)
    examples = [dataset[0], dataset[1]]
    batch = ParallelCollator(tokenizer.pad_id)(examples)
    source_length = len(examples[0]["encoder_input_ids"])
    target_length = len(examples[1]["decoder_input_ids"])
    assert source_length != target_length
    for key, length in (
        ("encoder_input_ids", source_length),
        ("decoder_input_ids", target_length),
        ("decoder_target_ids", target_length),
    ):
        tensor = batch[key]
        assert tensor.shape == (2, length)
        assert tensor.dtype == torch.long
        for index, example in enumerate(examples):
            ids = example[key]
            assert tensor[index].tolist() == ids + [tokenizer.pad_id] * (length - len(ids))
    assert batch["encoder_attention_mask"].shape == (2, 1, 1, source_length)
    assert batch["decoder_attention_mask"].shape == (2, 1, target_length, target_length)
    assert batch["cross_attention_mask"].shape == (2, 1, target_length, source_length)
