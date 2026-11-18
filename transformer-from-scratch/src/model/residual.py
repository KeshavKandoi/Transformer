"""Post-norm residual addition and optional query-padding handling."""

import torch
from torch import Tensor, nn

from .layer_norm import LayerNormalization


class ResidualAddNorm(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1, epsilon: float = 1e-5) -> None:
        super().__init__()
        if type(dropout) not in (int, float) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be a number in [0, 1]")
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model, epsilon)

    def forward(
        self, x: Tensor, sublayer_output: Tensor, token_valid_mask: Tensor | None = None
    ) -> Tensor:
        """Apply post-norm and optionally clear invalid [B, T] query positions."""
        if x.shape != sublayer_output.shape:
            raise ValueError("x and sublayer_output must have identical shapes")
        if token_valid_mask is not None:
            if x.ndim != 3 or token_valid_mask.shape != x.shape[:2]:
                raise ValueError("token_valid_mask must have shape [B, T] for input [B, T, d_model]")
            if token_valid_mask.dtype != torch.bool:
                raise TypeError("token_valid_mask must have dtype torch.bool")
            if token_valid_mask.device != x.device:
                raise ValueError("token_valid_mask must be on the same device as x")
        output = self.norm(x + self.dropout(sublayer_output))
        if token_valid_mask is not None:
            output = output.masked_fill(~token_valid_mask.unsqueeze(-1), 0.0)
        return output
