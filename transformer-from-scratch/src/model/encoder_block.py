"""One bidirectional Transformer encoder block in post-norm order."""

from torch import Tensor, nn

from .feed_forward import PositionwiseFeedForward
from .multi_head_attention import MultiHeadAttention
from .residual import ResidualAddNorm


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        *,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, attention_dropout)
        self.attention_residual = ResidualAddNorm(d_model, dropout, layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ffn_residual = ResidualAddNorm(d_model, dropout, layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        token_valid_mask: Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return [B, S, d_model] and optionally pre-dropout [B, H, S, S] maps."""
        result = self.self_attention(
            x, attention_mask=attention_mask, return_attention_weights=return_attention_weights,
        )
        attention_output = result[0] if isinstance(result, tuple) else result
        y = self.attention_residual(x, attention_output, token_valid_mask)
        z = self.ffn_residual(y, self.feed_forward(y), token_valid_mask)
        if isinstance(result, tuple):
            return z, result[1]
        return z
