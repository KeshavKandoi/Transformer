"""Trainable, scaled token lookup."""

import math

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int) -> None:
        super().__init__()
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        if type(d_model) is not int or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        if type(padding_idx) is not int or not 0 <= padding_idx < vocab_size:
            raise ValueError("padding_idx must be an integer in [0, vocab_size)")

        # PyTorch initializes the table randomly and zeros the padding row.
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, T]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must have dtype torch.int32 or torch.int64")
        return self.embedding(token_ids) * self.scale
