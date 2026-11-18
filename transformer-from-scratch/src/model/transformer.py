"""Teacher-forced encoder–decoder Transformer with a vocabulary head."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.masks import cross_attention_mask as make_cross_attention_mask
from .config import ModelConfig
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder
from .output_projection import VocabularyProjection


@dataclass(frozen=True)
class TransformerDiagnostics:
    """Live intermediate tensors. Attention axes are [batch, layer, head, query, key]."""

    logits: Tensor
    encoder_memory: Tensor
    decoder_hidden_states: Tensor
    encoder_attention_maps: Tensor
    decoder_self_attention_maps: Tensor
    decoder_cross_attention_maps: Tensor


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else config
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.output_projection = VocabularyProjection(
            self.config.d_model, vocab_size, bias=self.config.output_projection_bias,
        )
        self.encoder = TransformerEncoder(vocab_size, pad_id, self.config)
        self.decoder = TransformerDecoder(vocab_size, pad_id, self.config)

    def forward(
        self,
        encoder_input_ids: Tensor,
        decoder_input_ids: Tensor,
        encoder_attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | TransformerDiagnostics:
        """Use already shifted decoder IDs and return logits or explicit diagnostics."""
        if encoder_input_ids.ndim != 2:
            raise ValueError("encoder_input_ids must have shape [B, S]")
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must have shape [B, T]")
        if encoder_input_ids.size(0) != decoder_input_ids.size(0):
            raise ValueError("Encoder and decoder input IDs must have the same batch size")
        if encoder_input_ids.device != decoder_input_ids.device:
            raise ValueError("Encoder and decoder input IDs must be on the same device")
        if encoder_input_ids.device != self.output_projection.projection.weight.device:
            raise ValueError("Input IDs must be on the same device as model parameters")

        cross_mask = make_cross_attention_mask(encoder_input_ids, decoder_input_ids.size(1), self.pad_id)
        if cross_attention_mask is not None:
            if cross_attention_mask.dtype != torch.bool:
                raise TypeError("cross_attention_mask must have dtype torch.bool")
            if cross_attention_mask.device != cross_mask.device:
                raise ValueError("cross_attention_mask must be on the same device as input IDs")
            try:
                cross_mask = cross_mask & cross_attention_mask
            except RuntimeError as error:
                raise ValueError("cross_attention_mask must broadcast to [B, heads, T, S]") from error

        encoded = self.encoder(
            encoder_input_ids, encoder_attention_mask, return_attention_weights=return_diagnostics,
        )
        memory = encoded[0] if isinstance(encoded, tuple) else encoded
        decoded = self.decoder(
            decoder_input_ids, memory, decoder_attention_mask, cross_mask,
            return_attention_weights=return_diagnostics,
        )
        hidden = decoded[0] if isinstance(decoded, tuple) else decoded
        logits = self.output_projection(hidden)
        if isinstance(encoded, tuple) and isinstance(decoded, tuple):
            _, self_maps, cross_maps = decoded
            return TransformerDiagnostics(
                logits, memory, hidden, encoded[1], self_maps, cross_maps,
            )
        return logits
