"""Learning rates indexed by completed optimizer updates."""

import math
from dataclasses import replace

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .config import SchedulerConfig


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LambdaLR:
    def multiplier(completed_updates: int) -> float:
        if config.type == "constant":
            return 1.0
        step = completed_updates + 1
        warmup = config.warmup_steps
        return step / warmup if step <= warmup else math.sqrt(warmup / step)

    # LambdaLR sets the rate for update 1 at construction. Step after each update.
    return LambdaLR(optimizer, multiplier)


def resolve_schedule(config: SchedulerConfig, total_updates: int) -> SchedulerConfig:
    """Resolve fractional warmup once and persist the originally planned horizon."""
    if total_updates < 1:
        raise ValueError("A training schedule needs at least one planned optimizer update")
    if config.warmup_fraction is None or config.planned_updates is not None:
        return config
    warmup = min(max(1, total_updates - 1), math.ceil(total_updates * config.warmup_fraction))
    return replace(config, warmup_steps=warmup, planned_updates=total_updates)
