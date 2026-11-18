import json

import pytest
import torch

from src.training.checkpoint import REQUIRED_KEYS, CheckpointManager, load_checkpoint, validate_checkpoint
from src.training.config import EarlyStoppingConfig, TrainingConfig


def test_latest_best_snapshots_and_metadata(tmp_path, trainer_factory, monkeypatch) -> None:
    manager = CheckpointManager(tmp_path / "checkpoints")
    metrics_path = tmp_path / "metrics.jsonl"
    config = TrainingConfig(epochs=5, checkpoint_frequency=2, early_stopping=EarlyStoppingConfig(patience=2))
    trainer = trainer_factory(config, checkpoint_manager=manager, metrics_path=metrics_path)
    losses = iter([3.0, 2.0, 2.2, 2.1])
    monkeypatch.setattr(trainer, "validate", lambda: {"loss": next(losses), "perplexity_status": "finite"})
    history = trainer.fit()
    assert len(history) == trainer.epoch == 4
    best = load_checkpoint(manager.directory / "best.pt")
    latest = load_checkpoint(manager.directory / "latest.pt")
    assert best["epoch"] == 2 and best["best_validation_loss"] == 2.0
    assert latest["epoch"] == 4 and latest["global_step"] == 12
    assert latest["best_validation_loss"] == 2.0
    assert REQUIRED_KEYS <= latest.keys()
    assert latest["training_config"]["epochs"] == 5
    assert latest["tokenizer_metadata"] == trainer.tokenizer_metadata
    assert latest["model_config"] == trainer.model_config
    assert len(latest["optimizer_state_dict"]["state"]) > 0
    assert latest["scheduler_state_dict"]["last_epoch"] == latest["global_step"]
    assert latest["early_stopping_state"]["bad_checks"] == 2
    assert (manager.directory / "epoch-0002.pt").is_file()
    assert not (manager.directory / "epoch-0001.pt").exists()
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2, 3, 4]
    assert [row["global_step"] for row in rows] == [3, 6, 9, 12]


def test_compatibility_includes_architecture_and_token_mapping(trainer_factory) -> None:
    trainer = trainer_factory(TrainingConfig())
    payload = trainer.state_dict()
    for key, changed, pattern in (
        ("model_config", {"width": 6}, "model architecture"),
        ("tokenizer_metadata", {"vocab_size": 10}, "tokenizer"),
        ("tokenizer_metadata", {**trainer.tokenizer_metadata, "tokenizer_sha256": "new-map"}, "token mapping"),
        ("data_identity", {"split": "other"}, "dataset"),
    ):
        with pytest.raises(ValueError, match=pattern):
            validate_checkpoint({**payload, key: changed}, trainer.model_config,
                                trainer.tokenizer_metadata, trainer.data_identity)


def test_missing_corrupt_and_atomic_replacement(tmp_path, monkeypatch, trainer_factory) -> None:
    path = tmp_path / "latest.pt"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_checkpoint(path)
    path.write_bytes(b"This is not a checkpoint")
    with pytest.raises(ValueError, match="Cannot read checkpoint"):
        load_checkpoint(path)
    manager = CheckpointManager(tmp_path)
    payload = trainer_factory(TrainingConfig()).state_dict()
    manager.save(path.name, payload)

    def fail_save(value, stream):
        stream.write(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        manager.save(path.name, {**payload, "epoch": 1})
    assert load_checkpoint(path)["epoch"] == 0
    assert list(tmp_path.iterdir()) == [path]
