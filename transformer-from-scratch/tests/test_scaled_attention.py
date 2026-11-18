import math

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from src.model.attention import ScaledDotProductAttention, scaled_attention_scores


def test_manual_dot_product_scaling_softmax_and_value_mix() -> None:
    query = key = torch.eye(2).reshape(1, 1, 2, 2)
    value = torch.tensor([[[[1., 2.], [3., 4.]]]])
    logits = torch.tensor([[[[1., 0.], [0., 1.]]]]) / math.sqrt(2)
    torch.testing.assert_close(scaled_attention_scores(query, key), logits)

    high = math.exp(1 / math.sqrt(2)) / (1 + math.exp(1 / math.sqrt(2)))
    low = 1 - high
    context, weights = ScaledDotProductAttention()(query, key, value, return_weights=True)
    torch.testing.assert_close(weights, torch.tensor([[[[high, low], [low, high]]]]))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 1, 2))
    expected = [[high + 3 * low, 2 * high + 4 * low], [low + 3 * high, 2 * low + 4 * high]]
    torch.testing.assert_close(context, torch.tensor([[expected]]))


@pytest.mark.parametrize("masked", [False, True])
def test_pytorch_reference_with_multiple_heads_and_unequal_dimensions(masked: bool) -> None:
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(2, 3, 4, 5, generator=generator)
    key = torch.randn(2, 3, 7, 5, generator=generator)
    value = torch.randn(2, 3, 7, 6, generator=generator)
    mask = torch.rand(2, 1, 4, 7, generator=generator) > 0.5 if masked else None
    if mask is not None:
        mask[0, :, 0] = False
    output = ScaledDotProductAttention(0.3).eval()(query, key, value, mask)
    expected = scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0)
    assert output.shape == (2, 3, 4, 6)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected)


def test_dropout_changes_context_but_returns_pre_dropout_weights() -> None:
    inputs = torch.eye(2).reshape(1, 1, 2, 2)
    module = ScaledDotProductAttention(dropout=1.0)
    context, weights = module(inputs, inputs, inputs, return_weights=True)
    assert torch.count_nonzero(context) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 1, 2))

    module.eval()
    first, evaluation_weights = module(inputs, inputs, inputs, return_weights=True)
    torch.testing.assert_close(first, weights)
    torch.testing.assert_close(weights, evaluation_weights)
    torch.testing.assert_close(first, module(inputs, inputs, inputs), rtol=0, atol=0)


@pytest.mark.parametrize(("query", "key", "value", "message"), [
    ((2, 3, 4), (2, 1, 2, 4), (2, 1, 2, 8), "shape"),
    ((2, 1, 3, 4), (1, 1, 2, 4), (2, 1, 2, 8), "batch and head"),
    ((2, 1, 3, 4), (2, 1, 2, 5), (2, 1, 2, 8), "feature"),
    ((2, 1, 3, 4), (2, 1, 2, 4), (2, 1, 3, 8), "sequence length"),
])
def test_invalid_qkv_shapes(
    query: tuple[int, ...], key: tuple[int, ...], value: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ScaledDotProductAttention()(torch.zeros(query), torch.zeros(key), torch.zeros(value))


def test_invalid_tensor_types() -> None:
    inputs = torch.zeros(1, 1, 1, 2)
    with pytest.raises(TypeError, match="floating-point dtype"):
        ScaledDotProductAttention()(inputs, inputs.double(), inputs)


def test_low_precision_logits_do_not_overflow_float16() -> None:
    query = torch.tensor([[[[512., 0.]]]], dtype=torch.float16)
    key = torch.tensor([[[[512., 0.], [0., 512.]]]], dtype=torch.float16)
    value = torch.tensor([[[[1.], [2.]]]], dtype=torch.float16)
    output, weights = ScaledDotProductAttention()(query, key, value, return_weights=True)
    torch.testing.assert_close(output, value[:, :, :1])
    torch.testing.assert_close(weights, torch.tensor([[[[1., 0.]]]]))


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No CUDA or MPS device available",
)
def test_accelerator_agrees_with_cpu() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    inputs = torch.randn(2, 2, 3, 4)
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool).tril()
    module = ScaledDotProductAttention().eval()
    cpu = module(inputs, inputs, inputs, mask)
    device_inputs = inputs.to(device)
    accelerated = module(device_inputs, device_inputs, device_inputs, mask.to(device))
    torch.testing.assert_close(accelerated.cpu(), cpu, atol=1e-5, rtol=1e-5)
