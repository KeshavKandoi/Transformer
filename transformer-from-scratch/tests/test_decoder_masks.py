import pytest
import torch

from src.data.masks import cross_attention_mask, decoder_attention_mask
from src.model.config import ModelConfig
from src.model.decoder import TransformerDecoder


def small_decoder() -> TransformerDecoder:
    return TransformerDecoder(32, 0, ModelConfig(d_model=16, d_ff=32)).eval()


def test_changing_multiple_future_tokens_preserves_prefix_in_every_layer() -> None:
    torch.manual_seed(42)
    model = small_decoder()
    ids = torch.tensor([[2, 5, 6, 7, 8], [2, 5, 9, 10, 11]])
    memory = torch.randn(1, 3, 16).expand(2, -1, -1)
    # A supplied permissive mask must not remove the canonical causal mask.
    output, self_maps, cross_maps = model(
        ids, memory, torch.ones(5, 5, dtype=torch.bool), torch.ones(5, 3, dtype=torch.bool),
        return_attention_weights=True,
    )
    torch.testing.assert_close(output[0, :2], output[1, :2], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(self_maps[0, :, :, :2], self_maps[1, :, :, :2])
    torch.testing.assert_close(cross_maps[0, :, :, :2], cross_maps[1, :, :, :2])
    assert not torch.allclose(output[0, 2:], output[1, 2:])
    assert torch.count_nonzero(self_maps.triu(1)) == 0
    torch.testing.assert_close(self_maps.sum(-1), torch.ones_like(self_maps[..., 0]))


def test_target_and_source_padding_and_fully_masked_query() -> None:
    torch.manual_seed(42)
    model = small_decoder()
    ids = torch.tensor([[2, 5, 6]])
    source_ids = torch.tensor([[4, 7, 0, 0, 0]])
    memory = torch.randn(1, 5, 16)
    base = model(ids, memory, cross_attention_mask=cross_attention_mask(source_ids, 3, 0))
    padded = torch.nn.functional.pad(ids, (0, 3))
    mask = cross_attention_mask(source_ids, 6, 0).clone()
    output, self_maps, cross_maps = model(padded, memory, cross_attention_mask=mask, return_attention_weights=True)
    torch.testing.assert_close(output[:, :3], base, atol=2e-6, rtol=1e-5)
    assert torch.count_nonzero(output[:, 3:]) == 0
    assert torch.count_nonzero(self_maps[..., 3:]) == 0
    assert torch.count_nonzero(cross_maps[..., 2:]) == 0
    mask[:, :, 1] = False
    self_mask = decoder_attention_mask(padded, 0)
    self_mask[:, :, 1] = False
    result, self_maps, cross_maps = model(padded, memory, self_mask, mask, return_attention_weights=True)
    assert all(torch.isfinite(tensor).all() for tensor in (result, self_maps, cross_maps))
    assert torch.count_nonzero(self_maps[..., 1, :]) == torch.count_nonzero(cross_maps[..., 1, :]) == 0


def test_swapped_masks_and_missing_cross_mask_raise() -> None:
    ids, source_ids = torch.tensor([[2, 5, 0]]), torch.tensor([[4, 6, 7, 8, 0]])
    memory = torch.zeros(1, 5, 16)
    model = small_decoder()
    self_mask, cross_mask = decoder_attention_mask(ids, 0), cross_attention_mask(source_ids, 3, 0)
    with pytest.raises(ValueError, match="decoder_attention_mask must broadcast"):
        model(ids, memory, cross_mask, self_mask)
    with pytest.raises(ValueError, match="cross_attention_mask must broadcast"):
        model(ids, memory, self_mask, self_mask)
    with pytest.raises(ValueError, match="cross_attention_mask is required"):
        model(ids, memory)
    with pytest.raises(TypeError, match="torch.bool"):
        model(ids, memory, self_mask, cross_mask.float())


@pytest.mark.parametrize(("ids", "error", "message"), [
    (torch.tensor([[0, 0]]), ValueError, "non-padding"),
    (torch.empty(1, 0, dtype=torch.long), ValueError, "nonempty"),
    (torch.tensor([2, 5]), ValueError, "shape"),
    (torch.ones(1, 2), TypeError, "dtype"),
    (torch.tensor([[32]]), ValueError, "vocabulary"),
    (torch.ones(1, 129, dtype=torch.long), ValueError, "exceeds max_seq_len"),
])
def test_invalid_decoder_ids(ids: torch.Tensor, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        small_decoder()(ids, torch.zeros(1, 2, 16), cross_attention_mask=torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize("shape", [(1, 0, 16), (1, 4, 8), (2, 4, 16)])
def test_invalid_encoder_memory(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="encoder_memory"):
        small_decoder()(torch.tensor([[2, 5]]), torch.zeros(shape))
