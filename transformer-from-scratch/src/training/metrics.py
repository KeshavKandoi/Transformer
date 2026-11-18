"""Token-weighted metrics for teacher-forced predictions."""

import math
from dataclasses import dataclass

from torch import Tensor


@dataclass
class TokenMetrics:
    loss_sum: float = 0.0
    token_count: int = 0
    correct: int = 0

    def add(self, loss_sum: Tensor, count: int, logits: Tensor, targets: Tensor, pad_id: int) -> None:
        valid = targets.ne(pad_id)
        self.correct += int((logits.detach().argmax(-1).eq(targets) & valid).sum().item())
        self.loss_sum += float(loss_sum.detach().item())
        self.token_count += count

    def summary(self) -> dict[str, float | int | str | None]:
        if not self.token_count:
            raise ValueError("Metrics need at least one valid target token")
        mean = self.loss_sum / self.token_count
        if not math.isfinite(mean):
            raise FloatingPointError("Non-finite mean token loss")
        try:
            perplexity = math.exp(mean)
            status = "finite"
        except OverflowError:
            perplexity = None
            status = "overflow"
        return {
            "loss": mean, "accuracy": self.correct / self.token_count,
            "valid_tokens": self.token_count, "perplexity": perplexity, "perplexity_status": status,
        }
