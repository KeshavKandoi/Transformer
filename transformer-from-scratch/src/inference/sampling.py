"""Temperature, top-k and nucleus sampling in that order."""

import math

import torch
from torch import Tensor


def sampling_probabilities(
    logits: Tensor, *, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
) -> Tensor:
    if type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if type(top_k) is not int or not 0 <= top_k <= logits.size(-1):
        raise ValueError("top_k must be between 0 (disabled) and vocabulary size")
    if type(top_p) not in (float, int) or not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1")
    if logits.isnan().any() or logits.isposinf().any() or not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError("Each sampling row must have a finite candidate and no NaN or positive infinity")

    values = logits.cpu().to(torch.float64)
    scores = (values - values.amax(-1, keepdim=True)) / temperature
    indices = scores.argsort(dim=-1, descending=True, stable=True)
    sorted_scores = scores.gather(-1, indices)
    positions = torch.arange(indices.size(-1))
    if top_k:
        sorted_scores = sorted_scores.masked_fill(positions >= top_k, -torch.inf)
    probabilities = sorted_scores.softmax(-1)
    if top_p < 1:
        cumulative = probabilities.cumsum(-1)
        mass_before = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), dim=-1)
        probabilities = probabilities.masked_fill((mass_before >= top_p) & (positions > 0), 0)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return torch.zeros_like(probabilities).scatter(-1, indices, probabilities)


def sample_next_token(
    logits: Tensor, *, temperature: float, top_k: int, top_p: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    probabilities = sampling_probabilities(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1).to(logits.device)
