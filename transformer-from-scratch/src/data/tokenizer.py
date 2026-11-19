"""Shared byte-level BPE for English and Hindi."""

import json
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
MIN_VOCAB_SIZE = len(pre_tokenizers.ByteLevel.alphabet()) + len(SPECIAL_TOKENS)


class BilingualBPETokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = (
            self._required_id(token) for token in SPECIAL_TOKENS
        )

    def _required_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required token {token}")
        return token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 8000) -> "BilingualBPETokenizer":
        if vocab_size < MIN_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {MIN_VOCAB_SIZE}")
        corpus = (text for text in texts if text.strip())
        first = next(corpus, None)
        if first is None:
            raise ValueError("Tokenizer training requires non-empty text")
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(chain((first,), corpus), trainer=trainer)
        # Treat literal marker spellings in corpus text as ordinary bytes.
        tokenizer.encode_special_tokens = True
        return cls(tokenizer)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(directory / "tokenizer.json"))
        metadata = {
            "vocab_size": self.vocab_size,
            "special_token_ids": {token: self._required_id(token) for token in SPECIAL_TOKENS},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "BilingualBPETokenizer":
        path = directory / "tokenizer.json"
        if not path.is_file():
            raise ValueError(f"Tokenizer missing at {path}. Run scripts/prepare_data.py first.")
        try:
            tokenizer = Tokenizer.from_file(str(path))
        except Exception as error:
            raise ValueError(f"Cannot load tokenizer from {path}: {error}") from error
        tokenizer.encode_special_tokens = True
        return cls(tokenizer)

    def id_to_token(self, token_id: int) -> str:
        """Return the vocabulary spelling before ByteLevel decoding."""
        token = self._tokenizer.id_to_token(token_id)
        if token is None:
            raise ValueError(f"Unknown tokenizer ID {token_id}")
        return token

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
