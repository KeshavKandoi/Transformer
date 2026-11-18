"""Model settings and compatibility checks for the Transformer."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    max_seq_len: int = 128
    dropout: float = 0.1
    num_heads: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 1024
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    normalization_style: str = "post_norm"
    ffn_activation: str = "relu"
    output_projection_bias: bool = False

    def validate_sublayer_settings(self) -> None:
        if self.normalization_style != "post_norm":
            raise ValueError("model.normalization_style must be post_norm")
        if self.ffn_activation != "relu":
            raise ValueError("model.ffn_activation must be relu")

    @property
    def head_dim(self) -> int:
        if type(self.d_model) is not int or self.d_model < 1:
            raise ValueError("model.d_model must be a positive integer")
        if type(self.num_heads) is not int or self.num_heads < 1:
            raise ValueError("model.num_heads must be a positive integer")
        if self.d_model % self.num_heads:
            raise ValueError("model.d_model must be divisible by model.num_heads")
        return self.d_model // self.num_heads

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        try:
            values = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(values, dict) or not isinstance(values.get("model"), dict):
                raise ValueError("Expected a YAML mapping with a model section")
            return cls(**values["model"])
        except (TypeError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"Invalid configuration in {path}: {error}") from error

    def validate_sequence_lengths(self, max_source_length: int, max_target_length: int) -> None:
        if type(self.max_seq_len) is not int or self.max_seq_len < 1:
            raise ValueError("model.max_seq_len must be a positive integer")
        if self.max_seq_len < max(max_source_length, max_target_length):
            raise ValueError(
                f"model.max_seq_len ({self.max_seq_len}) must cover both max_source_length "
                f"({max_source_length}) and max_target_length ({max_target_length})"
            )
