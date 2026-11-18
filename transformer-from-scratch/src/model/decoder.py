"""Transformer decoder stack and its teacher-forced token-ID entry point."""

import torch
from torch import Tensor, nn

from ..data.masks import decoder_attention_mask as make_decoder_mask
from .config import ModelConfig
from .decoder_block import DecoderBlock
from .embeddings import TransformerEmbedding


def _validate_mask(mask: Tensor, name: str, shape: tuple[int, ...], device: torch.device) -> None:
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if mask.device != device:
        raise ValueError(f"{name} must be on the same device as decoder_input_ids")
    try:
        broadcast_shape = torch.broadcast_shapes(mask.shape, shape)
    except RuntimeError as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error
    if broadcast_shape != shape:
        raise ValueError(f"{name} must broadcast to {shape}")


class DecoderStack(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = ModelConfig() if config is None else config
        config.validate_sublayer_settings()
        depth = config.num_decoder_layers
        if type(depth) is not int or depth < 1:
            raise ValueError("num_decoder_layers must be a positive integer")
        self.layers = nn.ModuleList(
            DecoderBlock(
                config.d_model, config.num_heads, config.d_ff, config.dropout,
                attention_dropout=config.attention_dropout, layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(depth)
        )

    def forward(
        self, x: Tensor, encoder_memory: Tensor, decoder_attention_mask: Tensor,
        cross_attention_mask: Tensor, token_valid_mask: Tensor | None = None,
        *, return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Return states and optionally both maps in [B, layer, head, query, key] order."""
        self_maps: list[Tensor] = []
        cross_maps: list[Tensor] = []
        for layer in self.layers:
            result = layer(
                x, encoder_memory, decoder_attention_mask, cross_attention_mask, token_valid_mask,
                return_attention_weights=return_attention_weights,
            )
            if isinstance(result, tuple):
                x, self_weights, cross_weights = result
                self_maps.append(self_weights)
                cross_maps.append(cross_weights)
            else:
                x = result
        if return_attention_weights:
            return x, torch.stack(self_maps, dim=1), torch.stack(cross_maps, dim=1)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else config
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.embedding = TransformerEmbedding(
            vocab_size, self.config.d_model, self.config.max_seq_len, self.config.dropout, pad_id,
        )
        self.stack = DecoderStack(self.config)

    def forward(
        self, decoder_input_ids: Tensor, encoder_memory: Tensor,
        decoder_attention_mask: Tensor | None = None, cross_attention_mask: Tensor | None = None,
        *, return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Decode shifted IDs; the cross-attention mask must describe source key validity."""
        if decoder_input_ids.ndim != 2 or not decoder_input_ids.numel():
            raise ValueError("decoder_input_ids must have shape [B, T] with nonempty axes")
        if decoder_input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("decoder_input_ids must have dtype torch.int32 or torch.int64")
        if ((decoder_input_ids < 0) | (decoder_input_ids >= self.vocab_size)).any():
            raise ValueError("decoder_input_ids must be within the decoder vocabulary")
        valid = decoder_input_ids.ne(self.pad_id)
        if not valid.any(dim=-1).all():
            raise ValueError("Each decoder example must have at least one non-padding token")
        if (
            encoder_memory.ndim != 3 or encoder_memory.size(1) == 0
            or encoder_memory.size(-1) != self.config.d_model
        ):
            raise ValueError("encoder_memory must have shape [B, nonempty S, d_model]")
        batch, target_length = decoder_input_ids.shape
        if encoder_memory.size(0) != batch:
            raise ValueError("encoder_memory and decoder_input_ids must have the same batch size")
        device = decoder_input_ids.device
        if encoder_memory.device != device:
            raise ValueError("encoder_memory must be on the same device as decoder_input_ids")

        heads = self.config.num_heads
        self_mask = make_decoder_mask(decoder_input_ids, self.pad_id)
        if decoder_attention_mask is not None:
            _validate_mask(decoder_attention_mask, "decoder_attention_mask",
                           (batch, heads, target_length, target_length), device)
            self_mask = self_mask & decoder_attention_mask
        if cross_attention_mask is None:
            raise ValueError("cross_attention_mask is required to identify allowed encoder keys")
        _validate_mask(cross_attention_mask, "cross_attention_mask",
                       (batch, heads, target_length, encoder_memory.size(1)), device)

        return self.stack(
            self.embedding(decoder_input_ids), encoder_memory, self_mask, cross_attention_mask, valid,
            return_attention_weights=return_attention_weights,
        )
