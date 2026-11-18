"""Token-summed cross-entropy for training and weighted reporting."""

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_cross_entropy(
    logits: Tensor, targets: Tensor, pad_id: int, label_smoothing: float = 0.0,
) -> tuple[Tensor, int]:
    """Return summed cross-entropy and valid target count. Reject all-padding batches."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("Expected logits [B, T, V] and matching targets [B, T]")
    if not logits.is_floating_point() or targets.dtype != torch.long:
        raise TypeError("Expected floating-point logits and torch.long targets")
    if targets.device != logits.device:
        raise ValueError("Targets and logits must be on the same device")
    vocab_size = logits.size(-1)
    if type(pad_id) is not int or not 0 <= pad_id < vocab_size:
        raise ValueError("pad_id must be within the vocabulary")
    if ((targets < 0) | (targets >= vocab_size)).any():
        raise ValueError("Targets must be within the vocabulary")
    count = int(targets.ne(pad_id).sum().item())
    if not count:
        raise ValueError("Cross-entropy needs at least one non-padding target")
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1),
        ignore_index=pad_id, reduction="sum", label_smoothing=label_smoothing,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite cross-entropy")
    return loss, count
