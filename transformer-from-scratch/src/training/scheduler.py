"""Learning rates indexed by completed optimizer updates."""

import math

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
