import math

import pytest
import torch

from src.model.token_embedding import TokenEmbedding


def test_lookup_shape_scaling_and_random_initialization() -> None:
    torch.manual_seed(42)
    module = TokenEmbedding(vocab_size=7, d_model=6, padding_idx=3)
    ids = torch.tensor([[0, 3, 1], [1, 0, 3]], dtype=torch.int32)
    weight = module.embedding.weight
    output = module(ids)

    assert weight.shape == (7, 6)
    assert weight.requires_grad
    assert torch.count_nonzero(weight[3]) == 0
    assert not torch.equal(weight[0], weight[1])
    assert output.shape == (2, 3, 6)
    torch.testing.assert_close(output, weight[ids.long()] * math.sqrt(6))


@pytest.mark.parametrize("args", [(0, 6, 0), (7, 0, 0), (7, 6, -1), (7, 6, 7), (7, 6, True)])
def test_invalid_configuration(args: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="vocab_size|d_model|padding_idx"):
        TokenEmbedding(*args)


def test_invalid_input() -> None:
    module = TokenEmbedding(7, 6, 3)
    with pytest.raises(ValueError, match=r"shape \[B, T\]"):
        module(torch.zeros(3, dtype=torch.long))
    with pytest.raises(TypeError, match="dtype"):
        module(torch.zeros(1, 3))
