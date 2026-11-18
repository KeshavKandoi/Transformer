"""Versioned state-dictionary checkpoints with atomic replacement."""

import os
import pickle
import tempfile
from pathlib import Path

import torch

CHECKPOINT_VERSION = 1
REQUIRED_KEYS = {
    "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    "epoch", "global_step", "best_validation_loss", "early_stopping_state",
    "training_config", "model_config", "tokenizer_metadata", "data_identity", "rng_state",
}


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, ValueError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Cannot read checkpoint {path}: {error}") from error
    if (
        not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION
        or not REQUIRED_KEYS.issubset(payload)
    ):
        raise ValueError(f"Unsupported or incomplete training checkpoint: {path}")
    return payload


def validate_checkpoint(payload: dict, model_config: dict, tokenizer_metadata: dict, data_identity: dict) -> None:
    for key, current, label in (
        ("model_config", model_config, "model architecture"),
        ("tokenizer_metadata", tokenizer_metadata, "tokenizer vocabulary, special IDs or token mapping"),
        ("data_identity", data_identity, "prepared dataset or sequence limits"),
    ):
        if payload[key] != current:
            raise ValueError(f"Checkpoint is incompatible with the selected {label}")


class CheckpointManager:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, name: str, payload: dict) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / name
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                torch.save({**payload, "version": CHECKPOINT_VERSION}, stream)
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination

    def save_epoch(self, payload: dict, *, best: bool, snapshot: bool) -> None:
        if best:
            self.save("best.pt", payload)
        if snapshot:
            self.save(f"epoch-{payload['epoch']:04d}.pt", payload)
        self.save("latest.pt", payload)
