"""Stacked post-norm encoder and its source-token entry point."""

import torch
from torch import Tensor, nn

from ..data.masks import encoder_attention_mask
from .config import ModelConfig
from .embeddings import TransformerEmbedding
from .encoder_block import EncoderBlock


class EncoderStack(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = ModelConfig() if config is None else config
        config.validate_sublayer_settings()
        depth = config.num_encoder_layers
        if type(depth) is not int or depth < 1:
            raise ValueError("num_encoder_layers must be a positive integer")
        self.layers = nn.ModuleList(
            EncoderBlock(
                config.d_model, config.num_heads, config.d_ff, config.dropout,
                attention_dropout=config.attention_dropout, layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(depth)
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        token_valid_mask: Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return memory and optionally maps with axes [B, layer, head, query, key]."""
        maps: list[Tensor] = []
        for layer in self.layers:
            result = layer(
                x, attention_mask, token_valid_mask,
                return_attention_weights=return_attention_weights,
            )
            if isinstance(result, tuple):
                x, weights = result
                maps.append(weights)
            else:
                x = result
        if return_attention_weights:
            return x, torch.stack(maps, dim=1)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else config
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.embedding = TransformerEmbedding(
            vocab_size, self.config.d_model, self.config.max_seq_len, self.config.dropout, pad_id,
        )
        self.stack = EncoderStack(self.config)

    def forward(
        self,
        source_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Encode [B, S] IDs into [B, S, d_model] memory with zero padded queries."""
        if source_ids.ndim != 2 or source_ids.size(0) == 0 or source_ids.size(1) == 0:
            raise ValueError("source_ids must have shape [B, S] with nonempty batch and sequence axes")
        if source_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("source_ids must have dtype torch.int32 or torch.int64")
        if ((source_ids < 0) | (source_ids >= self.vocab_size)).any():
            raise ValueError("source_ids must be within the encoder vocabulary")

        key_mask = encoder_attention_mask(source_ids, self.pad_id)
        valid = key_mask[:, 0, 0]
        if not valid.any(dim=-1).all():
            raise ValueError("Each source example must have at least one non-padding token")
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must have dtype torch.bool")
            if attention_mask.device != source_ids.device:
                raise ValueError("attention_mask must be on the same device as source_ids")
            # Validate against the actual score axes before intersecting the masks.
            batch, length = source_ids.shape
            score_shape = (batch, self.config.num_heads, length, length)
            try:
                shape = torch.broadcast_shapes(attention_mask.shape, score_shape)
            except RuntimeError as error:
                raise ValueError(f"attention_mask must broadcast to {score_shape}") from error
            if shape != score_shape:
                raise ValueError(f"attention_mask must broadcast to {score_shape}")
            key_mask = key_mask & attention_mask

        x = self.embedding(source_ids)
        return self.stack(x, key_mask, valid, return_attention_weights=return_attention_weights)
