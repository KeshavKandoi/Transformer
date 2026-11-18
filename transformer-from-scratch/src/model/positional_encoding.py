"""Fixed sinusoidal positions from the original Transformer."""

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        if type(d_model) is not int or d_model < 1 or d_model % 2:
            raise ValueError("d_model must be a positive even integer")
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")

        positions = torch.arange(max_seq_len, dtype=torch.float32)[:, None]
        even_dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
        inverse_frequencies = torch.exp(-math.log(10_000) * even_dimensions / d_model)
        angles = positions * inverse_frequencies
        encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=1)
        self.register_buffer("encoding", encoding)

    def forward(self, embeddings: Tensor) -> Tensor:
        max_seq_len, d_model = self.encoding.shape
        if embeddings.ndim != 3 or embeddings.size(-1) != d_model:
            raise ValueError(f"embeddings must have shape [B, T, {d_model}]")
        if not embeddings.is_floating_point():
            raise TypeError("embeddings must have a floating-point dtype")
        length = embeddings.size(1)
        if length > max_seq_len:
            raise ValueError(f"Sequence length {length} exceeds max_seq_len {max_seq_len}")
        if embeddings.device != self.encoding.device:
            raise ValueError("Move positional encoding to the same device as the embeddings")
        return embeddings + self.encoding[:length].to(dtype=embeddings.dtype)
