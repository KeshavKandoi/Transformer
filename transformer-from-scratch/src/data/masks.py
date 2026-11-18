"""Attention masks for [batch, heads, queries, keys]. True allows attention."""

import torch
from torch import Tensor


def encoder_attention_mask(encoder_ids: Tensor, pad_id: int) -> Tensor:
    """Allow non-padding source keys. Shape [B, 1, 1, S]."""
    return encoder_ids.ne(pad_id)[:, None, None, :]


def causal_attention_mask(length: int, device: torch.device | None = None) -> Tensor:
    """Allow the current and earlier positions. Shape [T, T]."""
    return torch.ones((length, length), dtype=torch.bool, device=device).tril()


def decoder_attention_mask(decoder_ids: Tensor, pad_id: int) -> Tensor:
    """Combine causality with non-padding decoder keys. Shape [B, 1, T, T]."""
    causal = causal_attention_mask(decoder_ids.size(1), decoder_ids.device)
    return causal & decoder_ids.ne(pad_id)[:, None, None, :]


def cross_attention_mask(encoder_ids: Tensor, target_length: int, pad_id: int) -> Tensor:
    """Allow each decoder query to see non-padding source keys. Shape [B, 1, T, S]."""
    return encoder_attention_mask(encoder_ids, pad_id).expand(-1, -1, target_length, -1)
