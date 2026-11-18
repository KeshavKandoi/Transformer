"""Trainable decoder-to-vocabulary projection."""

from torch import Tensor, nn


class VocabularyProjection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False) -> None:
        super().__init__()
        if type(d_model) is not int or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        if type(bias) is not bool:
            raise TypeError("bias must be a boolean")
        self.projection = nn.Linear(d_model, vocab_size, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project [B, T, d_model] states to raw [B, T, vocab_size] logits."""
        width = self.projection.in_features
        if hidden_states.ndim != 3 or hidden_states.size(-1) != width:
            raise ValueError(f"hidden_states must have shape [B, T, {width}]")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must have a floating-point dtype")
        return self.projection(hidden_states)
