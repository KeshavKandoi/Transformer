"""Construct sequences for encoder-decoder teacher forcing."""

from collections.abc import Sequence
from typing import TypedDict

from torch.utils.data import Dataset

from .cleaning import TranslationPair
from .tokenizer import BilingualBPETokenizer


class SequenceExample(TypedDict):
    encoder_input_ids: list[int]
    decoder_input_ids: list[int]
    decoder_target_ids: list[int]


class ParallelDataset(Dataset[SequenceExample]):
    def __init__(
        self,
        pairs: Sequence[TranslationPair],
        tokenizer: BilingualBPETokenizer,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        if min(max_source_length, max_target_length) < 1:
            raise ValueError("Maximum sequence lengths must be at least 1")
        self.pairs = tuple(pairs)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> SequenceExample:
        pair = self.pairs[index]
        source = self.tokenizer.encode(pair.source)[: self.max_source_length - 1]
        target = self.tokenizer.encode(pair.target)[: self.max_target_length - 1]
        # Teacher forcing supplies the gold prefix. At each position the label
        # is the next token, which the causal mask hides from that query.
        return {
            "encoder_input_ids": [*source, self.tokenizer.eos_id],
            "decoder_input_ids": [self.tokenizer.bos_id, *target],
            "decoder_target_ids": [*target, self.tokenizer.eos_id],
        }
