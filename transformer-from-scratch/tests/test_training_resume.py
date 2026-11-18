import random
from dataclasses import replace

import pytest
import torch

from src.training.checkpoint import CheckpointManager, load_checkpoint
from src.training.config import SchedulerConfig, TrainingConfig
from src.training.utils import seed_everything


def test_epoch_boundary_resume_matches_uninterrupted_with_dropout(tmp_path, trainer_factory) -> None:
    config = TrainingConfig(epochs=3, gradient_accumulation_steps=2, scheduler=SchedulerConfig(warmup_steps=3))
    seed_everything(config.seed)
    continuous = trainer_factory(config)
    expected_history = continuous.fit()
    expected_random = random.random()
    expected_rng = torch.rand(4)
    expected_optimizer = continuous.optimizer.state_dict()

    seed_everything(config.seed)
    manager = CheckpointManager(tmp_path)
    interrupted = trainer_factory(replace(config, epochs=1), checkpoint_manager=manager)
    interrupted.fit()
    payload = load_checkpoint(tmp_path / "latest.pt")
    assert payload["epoch"] == 1 and payload["global_step"] == 2

    seed_everything(1000)
    resumed = trainer_factory(config)
    resumed.restore(payload)
    assert resumed.optimizer.param_groups[0]["lr"] == config.optimizer.learning_rate
    history = resumed.fit()
    assert [entry["epoch"] for entry in history] == [2, 3]
    for actual, expected in zip(history, expected_history[1:]):
        for key in ("epoch", "global_step", "train", "validation", "best_validation_loss"):
            assert actual[key] == expected[key]
    assert resumed.scheduler.state_dict() == continuous.scheduler.state_dict()
    for key, state in resumed.optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, expected_optimizer["state"][key][name], rtol=0, atol=0)
    for key, tensor in resumed.model.state_dict().items():
        torch.testing.assert_close(tensor, continuous.model.state_dict()[key], rtol=0, atol=0)
    assert random.random() == expected_random
    torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)


def test_resume_rejects_incompatible_training_settings(tmp_path, trainer_factory) -> None:
    manager = CheckpointManager(tmp_path)
    first = trainer_factory(TrainingConfig(epochs=1), checkpoint_manager=manager)
    first.fit()
    changed = trainer_factory(TrainingConfig(epochs=2, gradient_accumulation_steps=3))
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        changed.restore(load_checkpoint(tmp_path / "latest.pt"))
