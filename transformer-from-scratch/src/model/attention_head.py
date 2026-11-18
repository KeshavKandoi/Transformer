"""Independent Q, K and V projections for a single attention head."""

from torch import Tensor, nn

from .attention import ScaledDotProductAttention


class SingleAttentionHead(nn.Module):
    def __init__(self, d_model: int, head_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if type(d_model) is not int or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        if type(head_dim) is not int or head_dim < 1:
            raise ValueError("head_dim must be a positive integer")
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, head_dim, bias=False)
        self.Wk = nn.Linear(d_model, head_dim, bias=False)
        self.Wv = nn.Linear(d_model, head_dim, bias=False)
        self.attention = ScaledDotProductAttention(dropout)

    def project_qkv(
        self, query_input: Tensor, key_value_input: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project [B, sequence, d_model] inputs and add the singleton head axis."""
        if key_value_input is None:
            key_value_input = query_input
        for name, tensor in (("query_input", query_input), ("key_value_input", key_value_input)):
            if tensor.ndim != 3 or tensor.size(-1) != self.d_model:
                raise ValueError(f"{name} must have shape [B, sequence, {self.d_model}]")
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
        if query_input.size(0) != key_value_input.size(0):
            raise ValueError("query_input and key_value_input must have the same batch size")
        return (
            self.Wq(query_input).unsqueeze(1),
            self.Wk(key_value_input).unsqueeze(1),
            self.Wv(key_value_input).unsqueeze(1),
        )

    def forward(
        self,
        query_input: Tensor,
        key_value_input: Tensor | None = None,
        attention_mask: Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        query, key, value = self.project_qkv(query_input, key_value_input)
        result = self.attention(query, key, value, attention_mask, return_weights=return_weights)
        if isinstance(result, tuple):
            context, probabilities = result
            return context.squeeze(1), probabilities
        return result.squeeze(1)
