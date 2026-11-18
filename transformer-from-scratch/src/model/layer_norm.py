"""Token-wise layer normalization with learned affine parameters."""

import math

import torch
from torch import Tensor, nn


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int = 256, epsilon: float = 1e-5) -> None:
        super().__init__()
        if type(d_model) is not int or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.d_model = d_model
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0 or x.size(-1) != self.d_model:
            raise ValueError(f"LayerNorm expects a last dimension of size {self.d_model}")
        if not x.is_floating_point():
            raise TypeError("LayerNorm expects floating-point input")
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        centered = values - values.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.epsilon)
        output = self.gamma.to(values.dtype) * normalized + self.beta.to(values.dtype)
        return output.to(x.dtype)
