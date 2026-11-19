"""Non-cached beam search over one source sentence."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .greedy import Hypothesis, NextLogits, TokenStep, suppress_tokens, token_step


@dataclass(frozen=True)
class _Beam:
    ids: tuple[int, ...]
    log_probability: float
    steps: tuple[TokenStep, ...] = ()


def beam_search_decode(
    next_logits: NextLogits, prefix: Tensor, *, eos_id: int, pad_id: int,
    max_new_tokens: int, beam_width: int = 4, length_penalty: float = 0.6, trace: bool = False,
) -> Hypothesis:
    """Batch active prefixes for a single source; never expand a completed prefix."""
    if prefix.shape != (1, 1):
        raise ValueError("Beam search needs one source and one BOS token")
    if type(beam_width) is not int or beam_width < 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("beam_width and max_new_tokens must be positive integers")
    if not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError("length_penalty must be finite and non-negative")
    bos_id = int(prefix.item())
    active = [_Beam((), 0.0)]
    completed: list[_Beam] = []

    def rank(beam: _Beam) -> float:
        if beam.log_probability == 0:
            return math.inf
        # Order by the negative log magnitude of the normalized negative score.
        # This is equivalent to dividing by the length penalty without overflow.
        return length_penalty * math.log((5 + len(beam.ids)) / 6) - math.log(-beam.log_probability)

    for _ in range(max_new_tokens):
        prefixes = prefix.new_tensor([(bos_id, *candidate.ids) for candidate in active])
        logits = next_logits(prefixes).float()
        # Score the original distribution before excluding control tokens or pruning.
        log_probs = suppress_tokens(logits.log_softmax(-1), (pad_id, bos_id))
        accumulated = log_probs + log_probs.new_tensor([beam.log_probability for beam in active])[:, None]
        vocab_size = logits.size(-1)
        selected = accumulated.flatten().argsort(descending=True, stable=True)[:beam_width]
        next_active = []
        for index, score in zip(selected.tolist(), accumulated.flatten()[selected].tolist()):
            if score == -float("inf"):
                continue
            parent, token = divmod(index, vocab_size)
            beam = active[parent]
            steps = (*beam.steps, token_step(
                suppress_tokens(logits[parent], (pad_id, bos_id)), token, logits[parent], eos_id,
            )) if trace else ()
            candidate = _Beam((*beam.ids, token), score, steps)
            if token == eos_id:
                completed.append(candidate)
            else:
                next_active.append(candidate)
        completed = sorted(completed, key=rank, reverse=True)[:beam_width]
        active = next_active
        if not active:
            break

    best = max(completed or active, key=rank)
    return Hypothesis(best.ids, bool(best.ids and best.ids[-1] == eos_id), best.log_probability, best.steps)
