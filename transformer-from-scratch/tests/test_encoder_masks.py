import pytest
import torch

from src.model.config import ModelConfig
from src.model.encoder import TransformerEncoder


def small_encoder() -> TransformerEncoder:
    return TransformerEncoder(32, 3, ModelConfig(d_model=16, d_ff=32, max_seq_len=128)).eval()


def test_default_mask_is_bidirectional_and_right_padding_invariant() -> None:
    torch.manual_seed(42)
    encoder = small_encoder()
    ids = torch.tensor([[4, 5, 6], [7, 8, 9]])
    padded = torch.nn.functional.pad(ids, (0, 5), value=encoder.pad_id)
    memory, maps = encoder(ids, return_attention_weights=True)
    padded_memory, padded_maps = encoder(padded, return_attention_weights=True)
    torch.testing.assert_close(padded_memory[:, :3], memory, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(padded_maps[..., :3, :3], maps, atol=2e-6, rtol=1e-5)
    assert maps[..., 0, 2].gt(0).all()
    assert torch.count_nonzero(padded_maps[..., 3:]) == 0
    assert torch.count_nonzero(padded_memory[:, 3:]) == 0


def test_supplied_mask_cannot_unblock_padding_and_can_fully_mask_a_query() -> None:
    encoder = small_encoder()
    ids = torch.tensor([[4, 5, 3]])
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[0] = False
    memory, maps = encoder(ids, mask, return_attention_weights=True)
    assert torch.count_nonzero(maps[..., 0, :]) == 0
    assert torch.count_nonzero(maps[..., 2]) == 0
    assert torch.count_nonzero(memory[:, 2]) == 0
    assert torch.isfinite(memory).all() and torch.isfinite(maps).all()


@pytest.mark.parametrize(("ids", "error", "message"), [
    (torch.tensor([[4, 5], [3, 3]]), ValueError, "non-padding"),
    (torch.empty(1, 0, dtype=torch.long), ValueError, "nonempty"),
    (torch.empty(0, 2, dtype=torch.long), ValueError, "nonempty"),
    (torch.ones(2, dtype=torch.long), ValueError, "shape"),
    (torch.ones(1, 2), TypeError, "dtype"),
    (torch.tensor([[32]]), ValueError, "vocabulary"),
    (torch.tensor([[-1]]), ValueError, "vocabulary"),
    (torch.ones(1, 129, dtype=torch.long), ValueError, "exceeds max_seq_len"),
])
def test_invalid_source_ids(ids: torch.Tensor, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        small_encoder()(ids)


def test_invalid_explicit_attention_masks() -> None:
    encoder = small_encoder()
    ids = torch.tensor([[4, 5, 3]])
    with pytest.raises(ValueError, match="broadcast"):
        encoder(ids, torch.ones(2, 4, 3, 3, dtype=torch.bool))
    with pytest.raises(TypeError, match="torch.bool"):
        encoder(ids, torch.ones(3, 3))


@pytest.mark.parametrize(("batch", "length"), [(1, 128), (8, 17)])
def test_max_length_and_padding_heavy_batches(batch: int, length: int) -> None:
    ids = torch.full((batch, length), 3)
    ids[:, :2] = torch.tensor([7, 8])
    with torch.no_grad():
        memory = small_encoder()(ids)
    assert memory.shape == (batch, length, 16)
    assert torch.isfinite(memory).all()
    assert torch.count_nonzero(memory[:, 2:]) == 0
