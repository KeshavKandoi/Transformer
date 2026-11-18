import pytest
import torch

from src.inference.sampling import sample_next_token, sampling_probabilities


def test_temperature_top_k_then_top_p() -> None:
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float64).log()
    torch.testing.assert_close(sampling_probabilities(logits, temperature=2), (logits / 2).softmax(-1))
    torch.testing.assert_close(
        sampling_probabilities(logits, top_k=2, top_p=0.6),
        torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        sampling_probabilities(logits, top_p=0.6),
        torch.tensor([[0.625, 0.375, 0., 0.]], dtype=torch.float64),
    )


def test_seed_reproducibility_and_degenerate_logits() -> None:
    logits = torch.tensor([[1., 0.5, -torch.inf]]).expand(32, -1)
    options = dict(temperature=0.7, top_k=2, top_p=1.0)
    first = sample_next_token(logits, generator=torch.Generator().manual_seed(7), **options)
    second = sample_next_token(logits, generator=torch.Generator().manual_seed(7), **options)
    assert torch.equal(first, second) and set(first.tolist()) == {0, 1}
    near = sampling_probabilities(torch.tensor([[1e30, -1e30, -torch.inf]]), temperature=1e-300, top_p=0.01)
    torch.testing.assert_close(near, torch.tensor([[1., 0., 0.]], dtype=torch.float64))


@pytest.mark.parametrize("options", [
    {"temperature": 0}, {"temperature": float("inf")}, {"top_k": -1},
    {"top_k": 4}, {"top_p": 0}, {"top_p": 1.01},
])
def test_invalid_sampling_configuration(options) -> None:
    with pytest.raises(ValueError):
        sampling_probabilities(torch.zeros(1, 3), **options)


@pytest.mark.parametrize("logits", [[-torch.inf, -torch.inf], [0., torch.nan], [0., torch.inf]])
def test_invalid_sampling_logits(logits) -> None:
    with pytest.raises(ValueError, match="finite candidate"):
        sampling_probabilities(torch.tensor([logits]))
