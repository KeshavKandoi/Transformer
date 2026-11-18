import pytest
import torch

from src.data.masks import cross_attention_mask, decoder_attention_mask, encoder_attention_mask
from src.model.attention import ScaledDotProductAttention


def test_phase_one_masks() -> None:
    source = torch.tensor([[1, 2, 0], [3, 4, 5]])
    target = torch.tensor([[6, 7, 0, 0], [6, 8, 9, 0]])
    encoder = torch.zeros(2, 1, 3, 2)
    decoder = torch.zeros(2, 1, 4, 2)
    attention = ScaledDotProductAttention()
    cases = (
        (encoder, encoder, encoder_attention_mask(source, 0)),
        (decoder, decoder, decoder_attention_mask(target, 0)),
        (decoder, encoder, cross_attention_mask(source, target.size(1), 0)),
    )
    for queries, keys, mask in cases:
        _, weights = attention(queries, keys, keys, mask, return_weights=True)
        # Equal logits give uniform weights over exactly the allowed keys.
        allowed = mask.expand_as(weights)
        torch.testing.assert_close(weights, allowed / allowed.sum(-1, keepdim=True))
        assert torch.count_nonzero(weights[~allowed]) == 0

    _, causal = attention(decoder, decoder, decoder, cases[1][2], return_weights=True)
    torch.testing.assert_close(causal[:, :, 0, 0], torch.ones(2, 1))
    assert torch.count_nonzero(causal.triu(1)) == 0


def test_fully_masked_rows_have_zero_context_and_gradients() -> None:
    query = torch.ones(1, 1, 2, 2, requires_grad=True)
    key = torch.tensor([[[[1., 0.], [0., 0.]]]], requires_grad=True)
    value = torch.tensor([[[[1.], [3.]]]], requires_grad=True)
    mask = torch.tensor([[False, False], [True, True]])
    output, weights = ScaledDotProductAttention()(query, key, value, mask, return_weights=True)
    assert torch.isfinite(output).all() and torch.isfinite(weights).all()
    assert torch.count_nonzero(output[..., 0, :]) == 0
    assert torch.count_nonzero(weights[..., 0, :]) == 0
    torch.testing.assert_close(weights[..., 1, :].sum(), torch.tensor(1.))

    output.sum().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
    assert torch.count_nonzero(query.grad[..., 0, :]) == 0


@pytest.mark.parametrize("shape", [(1, 1, 1, 1, 3), (3, 1, 2, 3), (2, 3, 2, 3), (2, 1, 2, 2)])
def test_mask_cannot_alter_score_shape(shape: tuple[int, ...]) -> None:
    query = torch.ones(2, 1, 2, 4)
    keys = torch.ones(2, 1, 3, 4)
    with pytest.raises(ValueError, match="broadcast"):
        ScaledDotProductAttention()(query, keys, keys, torch.ones(shape, dtype=torch.bool))


def test_mask_requires_boolean_dtype() -> None:
    inputs = torch.ones(2, 1, 3, 4)
    with pytest.raises(TypeError, match="torch.bool"):
        ScaledDotProductAttention()(inputs, inputs, inputs, torch.ones(3, 3))
