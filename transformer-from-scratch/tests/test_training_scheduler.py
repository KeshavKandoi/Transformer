import math

import pytest
import torch

from src.training.config import SchedulerConfig
from src.training.scheduler import build_scheduler


def test_first_update_warmup_boundary_decay_and_resume() -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.03)
    config = SchedulerConfig(warmup_steps=3)
    scheduler = build_scheduler(optimizer, config)
    used = []
    for _ in range(5):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert used == pytest.approx([0.01, 0.02, 0.03, 0.03 * math.sqrt(3 / 4), 0.03 * math.sqrt(3 / 5)])
    restored = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.03)
    restored_scheduler = build_scheduler(restored, config)
    restored.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())
    assert restored.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    restored.step()
    restored_scheduler.step()
    assert restored.param_groups[0]["lr"] == pytest.approx(0.03 * math.sqrt(3 / 7))


def test_constant_and_invalid_warmup() -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.03)
    scheduler = build_scheduler(optimizer, SchedulerConfig(type="constant"))
    for _ in range(4):
        assert optimizer.param_groups[0]["lr"] == 0.03
        optimizer.step()
        scheduler.step()
    with pytest.raises(ValueError, match="warmup_steps"):
        SchedulerConfig(warmup_steps=0)
