"""Multi-head attention built from independent Phase 3 attention heads."""

import torch
from torch import Tensor, nn

from .attention_head import SingleAttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if type(d_model) is not int or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        if type(num_heads) is not int or num_heads < 1:
            raise ValueError("num_heads must be a positive integer")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if type(dropout) not in (int, float) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be a number in [0, 1]")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.heads = nn.ModuleList(
            SingleAttentionHead(d_model, self.head_dim, dropout) for _ in range(num_heads)
        )
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        query_input: Tensor,
        key_value_input: Tensor | None = None,
        attention_mask: Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return [B, Tq, d_model] and optionally pre-dropout [B, H, Tq, Tk] probabilities."""
        mask = attention_mask
        if mask is not None:
            memory = query_input if key_value_input is None else key_value_input
            if query_input.ndim != 3 or memory.ndim != 3:
                raise ValueError("Attention inputs must have shape [B, sequence, d_model]")
            score_shape = (query_input.size(0), self.num_heads, query_input.size(1), memory.size(1))
            try:
                mask = torch.broadcast_to(mask, score_shape)
            except RuntimeError as error:
                raise ValueError(
                    f"attention_mask shape {tuple(mask.shape)} must broadcast to {score_shape}"
                ) from error

        contexts: list[Tensor] = []
        weights: list[Tensor] = []
        for index, head in enumerate(self.heads):
            head_mask = None if mask is None else mask[:, index:index + 1]
            result = head(query_input, key_value_input, head_mask, return_weights=return_attention_weights)
            if isinstance(result, tuple):
                context, probabilities = result
                weights.append(probabilities)
            else:
                context = result
            contexts.append(context)

        output = self.output_projection(torch.cat(contexts, dim=-1))
        if return_attention_weights:
            return output, torch.cat(weights, dim=1)
        return output
