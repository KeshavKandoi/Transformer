import math

import pytest
import torch

from src.training.loss import masked_cross_entropy


def test_valid_token_sum_and_manual_loss(training_batches) -> None:
    batch = training_batches[0]
    targets = batch["decoder_target_ids"]
    assert targets[0, 0] == batch["decoder_input_ids"][0, 1] and targets[0, 1] == 3
    logits = torch.zeros(1, 3, 9, requires_grad=True)
    loss, count = masked_cross_entropy(logits, targets, 0)
    assert count == 2 and loss.ndim == 0
    torch.testing.assert_close(loss, torch.tensor(2 * math.log(9)))
    (loss / count).backward()
    assert torch.count_nonzero(logits.grad[0, 2]) == 0
    assert torch.count_nonzero(logits.grad[0, 1]) > 0


def test_label_smoothing_keeps_padding_excluded() -> None:
    logits = torch.tensor([[[0., math.log(2), 0.], [40., 0., -40.]]], requires_grad=True)
    loss, count = masked_cross_entropy(logits, torch.tensor([[1, 0]]), 0, 0.3)
    expected = (0.7 * math.log(2)) + 0.3 * (math.log(2) + 2 * math.log(4)) / 3
    torch.testing.assert_close(loss, torch.tensor(expected))
    assert count == 1
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 1]) == 0


@pytest.mark.parametrize(("targets", "error", "message"), [
    (torch.zeros(1, 2, dtype=torch.long), ValueError, "non-padding"),
    (torch.ones(1, 3, dtype=torch.long), ValueError, "matching"),
    (torch.ones(1, 2), TypeError, "torch.long"),
    (torch.tensor([[3, 4]]), ValueError, "within the vocabulary"),
    (torch.tensor([[-1, 1]]), ValueError, "within the vocabulary"),
])
def test_invalid_loss_inputs(targets, error, message) -> None:
    with pytest.raises(error, match=message):
        masked_cross_entropy(torch.zeros(1, 2, 4), targets, 0)
