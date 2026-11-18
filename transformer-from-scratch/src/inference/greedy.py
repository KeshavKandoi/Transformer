"""Greedy decoding and shared prefix growth for token-by-token selection."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

NextLogits = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class TokenStep:
    token_id: int
    candidates: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class Hypothesis:
    token_ids: tuple[int, ...]
    reached_eos: bool
    log_probability: float | None = None
    steps: tuple[TokenStep, ...] = ()


def suppress_tokens(scores: Tensor, forbidden: Sequence[int]) -> Tensor:
    scores = scores.clone()
    scores[..., list(forbidden)] = -torch.inf
    return scores


def token_step(scores: Tensor, token_id: int) -> TokenStep:
    indices = scores.argsort(descending=True, stable=True)[:5]
    candidates = tuple(
        (index, score) for index, score in zip(indices.tolist(), scores[indices].tolist())
        if score != -float("inf")
    )
    return TokenStep(token_id, candidates)


def decode_tokens(
    next_logits: NextLogits, prefixes: Tensor, *, eos_id: int, pad_id: int,
    max_new_tokens: int, select: Callable[[Tensor], Tensor], trace: bool = False,
) -> list[Hypothesis]:
    batch_size = prefixes.size(0)
    forbidden = (pad_id, int(prefixes[0, 0].item()))
    finished = [False] * batch_size
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    steps: list[list[TokenStep]] = [[] for _ in range(batch_size)]
    for _ in range(max_new_tokens):
        scores = suppress_tokens(next_logits(prefixes), forbidden)
        selected = select(scores)
        appended = []
        for row, token in enumerate(selected.tolist()):
            if finished[row]:
                appended.append(pad_id)
                continue
            appended.append(token)
            generated[row].append(token)
            if trace:
                steps[row].append(token_step(scores[row], token))
            finished[row] = token == eos_id
        if all(finished):
            break
        prefixes = torch.cat((prefixes, prefixes.new_tensor(appended)[:, None]), dim=1)
    return [
        Hypothesis(tuple(ids), eos, steps=tuple(row_steps))
        for ids, eos, row_steps in zip(generated, finished, steps)
    ]


def greedy_decode(
    next_logits: NextLogits, prefixes: Tensor, *, eos_id: int, pad_id: int,
    max_new_tokens: int, trace: bool = False,
) -> list[Hypothesis]:
    """Decode a batch of BOS prefixes, stopping each row independently."""
    return decode_tokens(
        next_logits, prefixes, eos_id=eos_id, pad_id=pad_id, max_new_tokens=max_new_tokens,
        select=lambda scores: scores.argmax(dim=-1), trace=trace,
    )
