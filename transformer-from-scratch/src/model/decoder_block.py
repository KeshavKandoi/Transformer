"""One post-norm Transformer decoder block."""

from torch import Tensor, nn

from ..data.masks import causal_attention_mask
from .feed_forward import PositionwiseFeedForward
from .multi_head_attention import MultiHeadAttention
from .residual import ResidualAddNorm


class DecoderBlock(nn.Module):
    def __init__(
        self, d_model: int = 256, num_heads: int = 4, d_ff: int = 1024,
        dropout: float = 0.1, *, attention_dropout: float = 0.1, layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, attention_dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, attention_dropout)
        self.self_attention_residual = ResidualAddNorm(d_model, dropout, layer_norm_eps)
        self.cross_attention_residual = ResidualAddNorm(d_model, dropout, layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ffn_residual = ResidualAddNorm(d_model, dropout, layer_norm_eps)

    def forward(
        self, decoder_x: Tensor, encoder_memory: Tensor, decoder_attention_mask: Tensor,
        cross_attention_mask: Tensor, token_valid_mask: Tensor | None = None,
        *, return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Return states and optionally self- and cross-attention probabilities."""
        if decoder_x.ndim != 3:
            raise ValueError("decoder_x must have shape [B, T, d_model]")
        causal = causal_attention_mask(decoder_x.size(1), decoder_x.device)
        if decoder_attention_mask.dtype != causal.dtype:
            raise TypeError("decoder_attention_mask must have dtype torch.bool")
        if decoder_attention_mask.device != decoder_x.device:
            raise ValueError("decoder_attention_mask must be on the same device as decoder_x")
        try:
            self_mask = decoder_attention_mask & causal
        except RuntimeError as error:
            raise ValueError("decoder_attention_mask must broadcast to the self-attention scores") from error
        self_result = self.masked_self_attention(
            decoder_x, attention_mask=self_mask, return_attention_weights=return_attention_weights,
        )
        self_output = self_result[0] if isinstance(self_result, tuple) else self_result
        y = self.self_attention_residual(decoder_x, self_output, token_valid_mask)
        cross_result = self.cross_attention(
            y, encoder_memory, cross_attention_mask, return_attention_weights=return_attention_weights,
        )
        cross_output = cross_result[0] if isinstance(cross_result, tuple) else cross_result
        z = self.cross_attention_residual(y, cross_output, token_valid_mask)
        output = self.ffn_residual(z, self.feed_forward(z), token_valid_mask)
        if isinstance(self_result, tuple) and isinstance(cross_result, tuple):
            return output, self_result[1], cross_result[1]
        return output
