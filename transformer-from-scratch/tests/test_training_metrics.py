import math

import pytest
import torch

from src.training.metrics import TokenMetrics


def test_weighted_loss_accuracy_and_perplexity() -> None:
    metrics = TokenMetrics()
    metrics.add(torch.tensor(2.), 1, torch.tensor([[[0., 3.], [0., 3.]]]), torch.tensor([[1, 0]]), 0)
    metrics.add(torch.tensor(9.), 3, torch.tensor([[[3., 0.], [3., 0.], [0., 3.]]]), torch.tensor([[1, 1, 1]]), 0)
    result = metrics.summary()
    assert result == metrics.summary()
    assert result["valid_tokens"] == 4 and result["accuracy"] == 0.5
    assert result["loss"] == 2.75 and result["perplexity"] == math.exp(2.75)


def test_overflow_and_empty_metrics() -> None:
    overflow = TokenMetrics(loss_sum=1000, token_count=1).summary()
    assert overflow["perplexity"] is None and overflow["perplexity_status"] == "overflow"
    with pytest.raises(ValueError, match="valid target"):
        TokenMetrics().summary()
