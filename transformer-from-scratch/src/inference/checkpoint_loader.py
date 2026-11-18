"""Restore trained Transformer weights and verify the exact tokenizer mapping."""

from dataclasses import asdict
from pathlib import Path

import torch

from ..data.prepared import load_tokenizer_artifact
from ..data.tokenizer import BilingualBPETokenizer
from ..model.config import ModelConfig
from ..model.transformer import EncoderDecoderTransformer
from ..training.checkpoint import load_checkpoint


def load_translation_model(
    checkpoint_path: Path, tokenizer_dir: Path, device: torch.device,
    expected_config: ModelConfig | None = None,
) -> tuple[EncoderDecoderTransformer, BilingualBPETokenizer]:
    payload = load_checkpoint(checkpoint_path)
    if type(payload["global_step"]) is not int or payload["global_step"] < 1:
        raise ValueError("The checkpoint must record at least one optimizer update")
    tokenizer, metadata = load_tokenizer_artifact(tokenizer_dir)
    if payload["tokenizer_metadata"] != metadata:
        raise ValueError("Checkpoint tokenizer vocabulary, special IDs or token mapping are incompatible")
    try:
        config = ModelConfig(**payload["model_config"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint model configuration: {error}") from error
    if expected_config is not None and asdict(expected_config) != asdict(config):
        raise ValueError("Checkpoint and requested model architecture are incompatible")
    model = EncoderDecoderTransformer(tokenizer.vocab_size, tokenizer.pad_id, config)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Checkpoint weights do not match the saved model architecture: {error}") from error
    return model.to(device).eval(), tokenizer
