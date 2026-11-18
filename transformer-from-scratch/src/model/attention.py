"""Scaled dot-product attention with boolean allowed-key masks."""

import math

import torch
from torch import Tensor, nn


def scaled_attention_scores(query: Tensor, key: Tensor) -> Tensor:
    """Compute unmasked logits from projected queries and keys."""
    if query.dtype in (torch.float16, torch.bfloat16):
        # Keep the dot products and softmax in float32 for low-precision inputs.
        with torch.autocast(device_type=query.device.type, enabled=False):
            return (query.float() @ key.float().transpose(-2, -1)) / math.sqrt(query.size(-1))
    return (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return context and, if requested, pre-dropout probabilities."""
        if any(tensor.ndim != 4 for tensor in (query, key, value)):
            raise ValueError("Q, K and V must have shape [B, H, sequence, features]")
        if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
            raise ValueError("Q, K and V must have the same batch and head dimensions")
        if query.size(-1) != key.size(-1) or query.size(-1) == 0:
            raise ValueError("Q and K must have the same positive feature dimension")
        if key.size(-2) != value.size(-2) or key.size(-2) == 0:
            raise ValueError("K and V must have the same positive sequence length")
        if not query.is_floating_point() or query.dtype != key.dtype or query.dtype != value.dtype:
            raise TypeError("Q, K and V must share a floating-point dtype")
        if query.device != key.device or query.device != value.device:
            raise ValueError("Q, K and V must be on the same device")

        score_shape = (*query.shape[:-1], key.size(-2))
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must have dtype torch.bool (True allows attention)")
            if attention_mask.device != query.device:
                raise ValueError("attention_mask must be on the same device as Q, K and V")
            if (
                attention_mask.ndim > len(score_shape)
                or any(
                    size not in (1, expected)
                    for size, expected in zip(reversed(attention_mask.shape), reversed(score_shape))
                )
            ):
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} must broadcast "
                    f"to score shape {score_shape}"
                )

        scores = scaled_attention_scores(query, key)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, -torch.inf)
            # Softmax of an entirely -inf row would produce NaNs.
            allowed_rows = torch.atleast_1d(attention_mask).any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~allowed_rows, 0.0)

        probabilities = scores.softmax(dim=-1)
        if attention_mask is not None:
            probabilities = probabilities.masked_fill(~attention_mask, 0.0)
        context = self.dropout(probabilities).to(dtype=value.dtype) @ value
        return (context, probabilities) if return_weights else context
