import math

import torch

from src.model.decoder_block import DecoderBlock


def test_cross_attention_uses_decoder_queries_and_encoder_keys_and_values() -> None:
    block = DecoderBlock(4, 2, 8).eval()
    module = block.cross_attention
    with torch.no_grad():
        for index, head in enumerate(module.heads, start=1):
            base = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
            head.Wq.weight.copy_(base * index)
            head.Wk.weight.copy_(base.flip(1) / index)
            head.Wv.weight.copy_(base.flip(0) + index)
        module.output_projection.weight.copy_(torch.arange(16).reshape(4, 4) / 20)
    query_input = torch.tensor([[[1., 0., 1., 0.], [0., 1., 0., 1.]]])
    memory = torch.tensor([[[0., 1., 0., 0.], [0.5, 0., 0., 1.], [2., 2., 2., 2.]]])
    mask = torch.tensor([[[[True, True, False], [True, True, False]]]])
    output, maps = module(query_input, memory, mask, return_attention_weights=True)

    expected_maps, contexts = [], []
    for head in module.heads:
        query = query_input @ head.Wq.weight.T
        key = memory @ head.Wk.weight.T
        value = memory @ head.Wv.weight.T
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(2)
        assert scores.shape == (1, 2, 3)
        weights = scores.masked_fill(~mask[:, 0], -torch.inf).softmax(dim=-1)
        expected_maps.append(weights)
        contexts.append(weights @ value)
    torch.testing.assert_close(maps, torch.stack(expected_maps, dim=1))
    torch.testing.assert_close(output, torch.cat(contexts, dim=-1) @ module.output_projection.weight.T)

    changed = memory.clone()
    changed[:, 2] = -100
    torch.testing.assert_close(module(query_input, changed, mask), output)
    changed[:, 0] = -5
    assert not torch.allclose(module(query_input, changed, mask), output)
