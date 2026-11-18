import pytest
import torch

from src.data.masks import cross_attention_mask, decoder_attention_mask, encoder_attention_mask
from src.model.multi_head_attention import MultiHeadAttention


def test_phase_one_masks_broadcast_across_every_head() -> None:
    source_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    target_ids = torch.tensor([[6, 7, 0, 0], [6, 8, 9, 0]])
    source = torch.zeros(2, 3, 16)
    target = torch.zeros(2, 4, 16)
    module = MultiHeadAttention(16, 4).eval()

    source_keys = source_ids.ne(0)[:, None, None, :]
    target_keys = target_ids.ne(0)[:, None, None, :]
    positions = torch.arange(target_ids.size(1))
    causal = positions[:, None] >= positions[None, :]
    cases = (
        (source, None, encoder_attention_mask(source_ids, 0), source_keys),
        (target, None, decoder_attention_mask(target_ids, 0), target_keys & causal),
        (target, source, cross_attention_mask(source_ids, target.size(1), 0), source_keys),
    )
    for query, memory, mask, expected_allowed in cases:
        context, weights = module(query, memory, mask, return_attention_weights=True)
        expected_allowed = expected_allowed.expand_as(weights)
        expected = expected_allowed / expected_allowed.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, expected)
        assert torch.count_nonzero(weights[~expected_allowed]) == 0
        assert torch.isfinite(context).all()


def test_fully_masked_rows_remain_zero_through_output_projection() -> None:
    module = MultiHeadAttention(16, 4).eval()
    query, memory = torch.randn(2, 3, 16), torch.randn(2, 5, 16)
    mask = torch.ones(2, 4, 3, 5, dtype=torch.bool)
    mask[:, :, 0] = False
    mask[:, 1, 1] = False
    output, weights = module(query, memory, mask, return_attention_weights=True)

    assert torch.isfinite(output).all() and torch.isfinite(weights).all()
    assert torch.count_nonzero(output[:, 0]) == 0
    assert torch.count_nonzero(output[:, 1:]) > 0
    assert torch.count_nonzero(weights[~mask]) == 0
    torch.testing.assert_close(weights.sum(-1), mask.any(-1).to(weights.dtype))


@pytest.mark.parametrize("shape", [(2, 3, 3, 3), (4, 4), (1, 2, 4, 3, 3)])
def test_invalid_mask_shapes(shape: tuple[int, ...]) -> None:
    module = MultiHeadAttention(16, 4)
    with pytest.raises(ValueError, match="broadcast"):
        module(torch.zeros(2, 3, 16), attention_mask=torch.ones(shape, dtype=torch.bool))


def test_invalid_mask_dtype() -> None:
    module = MultiHeadAttention(16, 4)
    with pytest.raises(TypeError, match="torch.bool"):
        module(torch.zeros(2, 3, 16), attention_mask=torch.ones(3, 3))
