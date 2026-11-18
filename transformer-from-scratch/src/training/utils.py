"""Device selection, reproducible data loading and the model batch interface."""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..data.cleaning import read_parallel_csv
from ..data.collator import ParallelCollator
from ..data.config import ROOT, DataConfig
from ..data.dataset import ParallelDataset
from ..data.prepared import SPLITS, load_tokenizer_artifact, preparation_settings, prepared_fingerprints
from ..data.tokenizer import BilingualBPETokenizer


def select_device(requested: str) -> torch.device:
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if requested == "auto":
        return torch.device(next(name for name, enabled in available.items() if enabled))
    if not available.get(requested, False):
        raise ValueError(f"Requested device {requested!r} is unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def capture_rng(device: torch.device) -> dict:
    state = {"python": random.getstate(), "cpu": torch.get_rng_state(), "device_type": device.type}
    if device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["device_type"] != device.type:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["accelerator"].cpu(), device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["accelerator"].cpu())


@dataclass
class PreparedData:
    tokenizer: BilingualBPETokenizer
    tokenizer_metadata: dict
    data_identity: dict
    datasets: dict[str, ParallelDataset]

    def loader(self, split: str, batch_size: int, seed: int, *, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.datasets[split], batch_size=batch_size, shuffle=shuffle,
            collate_fn=ParallelCollator(self.tokenizer.pad_id),
            generator=torch.Generator().manual_seed(seed),
        )


def load_prepared_data(
    processed_dir: Path, tokenizer_dir: Path, config: DataConfig, *, splits: tuple[str, ...] = SPLITS,
) -> PreparedData:
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing {manifest_path}. Run scripts/prepare_data.py with the selected paths.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprints = prepared_fingerprints(processed_dir, tokenizer_dir)
    expected = {"version": 1, "preparation_settings": preparation_settings(config), **fingerprints}
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("Prepared splits, tokenizer or data settings do not match. Rerun scripts/prepare_data.py.")
    tokenizer, metadata = load_tokenizer_artifact(tokenizer_dir, fingerprints["tokenizer_sha256"])
    datasets = {}
    sources: set[str] = set()
    for name in splits:
        pairs, stats = read_parallel_csv(processed_dir / f"{name}.csv")
        if stats.invalid or stats.duplicates:
            raise ValueError(f"Prepared {name} split contains invalid or duplicate pairs")
        split_sources = {pair.source for pair in pairs}
        if sources & split_sources:
            raise ValueError(f"Prepared {name} split shares source sentences with another split")
        sources.update(split_sources)
        datasets[name] = ParallelDataset(pairs, tokenizer, config.max_source_length, config.max_target_length)
    identity = {**manifest, "max_source_length": config.max_source_length, "max_target_length": config.max_target_length}
    return PreparedData(tokenizer, metadata, identity, datasets)


def forward_batch(model: nn.Module, batch: dict[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    inputs = {key: value.to(device) for key, value in batch.items()}
    targets = inputs.pop("decoder_target_ids")
    return model(**inputs), targets


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/data_config.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model_config.yaml")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "artifacts/tokenizer")
