import torch

from src.inference.greedy import greedy_decode

PAD, BOS, EOS = 0, 1, 2


def test_greedy_grows_bos_prefix_and_stops_rows_independently() -> None:
    calls = []

    def next_logits(prefix):
        calls.append(prefix.clone())
        scores = torch.zeros(2, 6)
        scores[:, PAD] = 100
        scores[:, BOS] = 90
        scores[0, EOS] = 4
        scores[1, 4 if prefix.size(1) == 1 else EOS] = 4
        return scores

    result = greedy_decode(next_logits, torch.full((2, 1), BOS), eos_id=EOS, pad_id=PAD,
                           max_new_tokens=5, trace=True)
    assert [call.tolist() for call in calls] == [[[BOS], [BOS]], [[BOS, EOS], [BOS, 4]]]
    assert [hypothesis.token_ids for hypothesis in result] == [(EOS,), (4, EOS)]
    assert all(hypothesis.reached_eos for hypothesis in result)
    assert result[1].steps[0].candidates[0] == (4, 4.0)


def test_greedy_is_deterministic_at_generation_limit() -> None:
    def next_logits(prefix):
        return torch.tensor([[0., 1., 2., 3.]]).expand(prefix.size(0), -1)

    options = dict(eos_id=EOS, pad_id=PAD, max_new_tokens=3)
    prefix = torch.tensor([[BOS]])
    result = greedy_decode(next_logits, prefix, **options)
    assert result == greedy_decode(next_logits, prefix, **options)
    assert result[0].token_ids == (3, 3, 3) and not result[0].reached_eos
