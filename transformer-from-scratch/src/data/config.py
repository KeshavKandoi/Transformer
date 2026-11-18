"""Shared configuration and command-line paths for Phase 1."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    seed: int = 42
    batch_size: int = 8
    max_source_length: int = 128
    max_target_length: int = 128
    vocab_size: int = 8000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a non-negative 63-bit integer")
        for key in ("batch_size", "max_source_length", "max_target_length", "vocab_size"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("train_ratio", "val_ratio", "test_ratio"):
            if type(getattr(self, key)) not in (int, float):
                raise ValueError(f"{key} must be numeric")

    @classmethod
    def load(cls, path: Path) -> "DataConfig":
        try:
            values = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("Expected a YAML mapping")
            return cls(**values)
        except (TypeError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"Invalid configuration in {path}: {error}") from error


def data_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data_config.yaml")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "artifacts/tokenizer")
    return parser
