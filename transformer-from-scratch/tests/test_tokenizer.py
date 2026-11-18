import json
import unicodedata
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from src.data.tokenizer import BilingualBPETokenizer, SPECIAL_TOKENS


@pytest.mark.parametrize("sentence", [
    "We are friends.\tHello!",
    "हम दोस्त हैं। क्या आप ठीक हैं?",
    "Rare characters: 🦋 ऋ",
    "Literal <pad> and <eos>",
    "  cafe\u0301  ",
])
def test_round_trip(sentence: str) -> None:
    tokenizer = BilingualBPETokenizer.train(["We are friends", "हम दोस्त हैं"], 350)
    ids = tokenizer.encode(sentence)
    assert not {tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id}.intersection(ids)
    assert tokenizer.decode(ids) == unicodedata.normalize("NFC", sentence)


def test_special_ids_and_save_load(tmp_path: Path) -> None:
    tokenizer = BilingualBPETokenizer.train(["Hello", "नमस्ते"])
    tokenizer.save(tmp_path)
    loaded = BilingualBPETokenizer.load(tmp_path)
    raw = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    expected = {token: raw.token_to_id(token) for token in SPECIAL_TOKENS}
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert list(expected.values()) == [loaded.pad_id, loaded.unk_id, loaded.bos_id, loaded.eos_id]
    assert len(set(expected.values())) == len(SPECIAL_TOKENS)
    assert metadata == {"vocab_size": loaded.vocab_size, "special_token_ids": expected}
    for sentence in ("Hello नमस्ते", "Literal <bos>"):
        assert loaded.encode(sentence) == tokenizer.encode(sentence)
        assert loaded.decode(loaded.encode(sentence)) == sentence
    assert loaded.decode([loaded.pad_id, loaded.bos_id, loaded.eos_id]) == ""


def test_prepare_trains_on_training_split_only(tmp_path: Path) -> None:
    from argparse import Namespace
    from itertools import chain

    from scripts.prepare_data import prepare
    from src.data.cleaning import TranslationPair, read_parallel_csv, write_parallel_csv

    source = tmp_path / "parallel.csv"
    processed = tmp_path / "processed"
    artifacts = tmp_path / "tokenizer"
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    pairs = [
        TranslationPair(f"Source {i}", target)
        for i, target in enumerate(("धन्यवाद", "नमस्ते", "दोस्त", "बारिश", "खिड़की"))
    ]
    write_parallel_csv(source, pairs)
    prepare(Namespace(config=config, input=source, processed_dir=processed, tokenizer_dir=artifacts))

    train, _ = read_parallel_csv(processed / "train.csv")
    expected = BilingualBPETokenizer.train(
        chain.from_iterable((pair.source, pair.target) for pair in train)
    )
    all_splits = BilingualBPETokenizer.train(
        chain.from_iterable((pair.source, pair.target) for pair in pairs)
    )
    actual = BilingualBPETokenizer.load(artifacts)
    assert actual.vocab_size == expected.vocab_size < all_splits.vocab_size
    for pair in pairs:
        assert actual.encode(pair.source) == expected.encode(pair.source)
        assert actual.encode(pair.target) == expected.encode(pair.target)
