"""Dynamically right-pad and mask a batch of parallel examples."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .dataset import SequenceExample
from .masks import cross_attention_mask, decoder_attention_mask, encoder_attention_mask


class ParallelCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, examples: Sequence[SequenceExample]) -> dict[str, Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        encoder = self._pad([example["encoder_input_ids"] for example in examples])
        decoder = self._pad([example["decoder_input_ids"] for example in examples])
        targets = self._pad([example["decoder_target_ids"] for example in examples])
        return {
            "encoder_input_ids": encoder,
            "decoder_input_ids": decoder,
            "decoder_target_ids": targets,
            "encoder_attention_mask": encoder_attention_mask(encoder, self.pad_id),
            "decoder_attention_mask": decoder_attention_mask(decoder, self.pad_id),
            "cross_attention_mask": cross_attention_mask(encoder, decoder.size(1), self.pad_id),
        }

    def _pad(self, sequences: list[list[int]]) -> Tensor:
        return pad_sequence(
            [torch.tensor(sequence, dtype=torch.long) for sequence in sequences],
            batch_first=True,
            padding_value=self.pad_id,
        )
