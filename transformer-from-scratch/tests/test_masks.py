import torch

from src.data.masks import (
    causal_attention_mask,
    cross_attention_mask,
    decoder_attention_mask,
    encoder_attention_mask,
)


def test_full_causal_triangle() -> None:
    mask = causal_attention_mask(4)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_attention_masks_block_padding_and_future_keys() -> None:
    pad_id = 99
    source = torch.tensor([[4, 5, pad_id], [6, 7, 8]])
    target = torch.tensor([[1, 2, pad_id, pad_id], [1, 2, 3, 4]])
    encoder = encoder_attention_mask(source, pad_id)
    decoder = decoder_attention_mask(target, pad_id)
    cross = cross_attention_mask(source, target.size(1), pad_id)

    assert encoder.dtype == decoder.dtype == cross.dtype == torch.bool
    assert encoder[:, 0, 0].tolist() == [[True, True, False], [True, True, True]]
    assert decoder[0, 0].tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, False, False],
        [True, True, False, False],
    ]
    assert torch.equal(decoder[1, 0], causal_attention_mask(4))
    assert decoder.any(dim=-1).all()
    assert cross[0, 0].tolist() == [[True, True, False]] * 4
    assert cross[1, 0].all()


def test_masks_work_with_scaled_dot_product_attention() -> None:
    from torch.nn.functional import scaled_dot_product_attention

    source = torch.tensor([[10, 11, 0]])
    target = torch.tensor([[12, 13, 0, 0]])
    for mask, query_length, values, expected in (
        (encoder_attention_mask(source, 0), 3, [2., 4., 100.], [3., 3., 3.]),
        (decoder_attention_mask(target, 0), 4, [2., 4., 100., 100.], [2., 3., 3., 3.]),
        (cross_attention_mask(source, 4, 0), 4, [2., 4., 100.], [3., 3., 3., 3.]),
    ):
        # Zero scores make each row the mean of the allowed values.
        queries = torch.zeros(1, 2, query_length, 1)
        keys = torch.zeros(1, 2, len(values), 1)
        value_tensor = torch.tensor(values).view(1, 1, -1, 1).expand_as(keys)
        result = scaled_dot_product_attention(queries, keys, value_tensor, attn_mask=mask)
        torch.testing.assert_close(result, torch.tensor(expected).view(1, 1, -1, 1).expand_as(result))
