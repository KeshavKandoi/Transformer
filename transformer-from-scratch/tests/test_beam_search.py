import math

import pytest
import torch

from src.inference.beam_search import beam_search_decode
from src.inference.greedy import greedy_decode

PAD, BOS, EOS = 0, 1, 2


def test_beam_preserves_finished_candidates_and_scores_original_distribution() -> None:
    calls = []

    def logits(prefixes):
        calls.append(prefixes.tolist())
        rows = []
        for prefix in prefixes.tolist():
            if len(prefix) == 1:
                probability = [0.01, 0.01, 0.02, 0.51, 0.45]
            elif prefix[-1] == 3:
                probability = [0.31, 0.31, 0.31, 0.04, 0.03]
            else:
                probability = [0.01, 0.01, 0.96, 0.01, 0.01]
            rows.append(probability)
        return torch.tensor(rows).log()

    prefix = torch.tensor([[BOS]])
    result = beam_search_decode(logits, prefix, eos_id=EOS, pad_id=PAD, max_new_tokens=4, beam_width=2, trace=True)
    assert calls == [[[BOS]], [[BOS, 3], [BOS, 4]]]
    assert result.token_ids == (4, EOS)
    assert result.reached_eos and result.log_probability == pytest.approx(math.log(0.45 * 0.96))
    assert [step.token_id for step in result.steps] == [4, EOS]


def test_length_penalty_can_prefer_a_longer_completed_candidate() -> None:
    def logits(prefixes):
        rows = [[0.01, 0.01, 0.51, 0.47] if len(prefix) == 1 else [0.001, 0.001, 0.997, 0.001]
                for prefix in prefixes.tolist()]
        return torch.tensor(rows).log()

    prefix = torch.tensor([[BOS]])
    options = dict(eos_id=EOS, pad_id=PAD, max_new_tokens=3, beam_width=2)
    assert beam_search_decode(logits, prefix, length_penalty=0, **options).token_ids == (EOS,)
    assert beam_search_decode(logits, prefix, length_penalty=2, **options).token_ids == (3, EOS)


def test_width_one_matches_greedy_and_returns_unfinished_sequence_at_limit() -> None:
    def logits(prefixes):
        return torch.tensor([[7.0, 8., 0., 3., 2.]]).expand(prefixes.size(0), -1)

    prefix = torch.tensor([[BOS]])
    options = dict(eos_id=EOS, pad_id=PAD, max_new_tokens=3)
    beam = beam_search_decode(logits, prefix, beam_width=1, **options)
    greedy = greedy_decode(logits, prefix, **options)[0]
    assert beam.token_ids == greedy.token_ids == (3, 3, 3)
    assert not beam.reached_eos
    assert beam.log_probability == pytest.approx(3 * logits(prefix).log_softmax(-1)[0, 3].item(), rel=1e-6)
