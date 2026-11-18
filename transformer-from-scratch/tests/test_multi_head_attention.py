from pathlib import Path

import pytest
import torch

from src.model.multi_head_attention import MultiHeadAttention


def test_default_heads_have_independent_projections_and_expected_parameter_count() -> None:
    module = MultiHeadAttention()
    assert (module.d_model, module.num_heads, module.head_dim) == (256, 4, 64)
    assert len(module.heads) == 4
    weights = [layer.weight for head in module.heads for layer in (head.Wq, head.Wk, head.Wv)]
    assert all(weight.shape == (64, 256) and weight.requires_grad for weight in weights)
    assert len({id(weight) for weight in weights}) == 12
    assert len({weight.untyped_storage().data_ptr() for weight in weights}) == 12
    assert module.output_projection.bias is None
    assert sum(parameter.numel() for parameter in module.parameters()) == 262_144


@pytest.mark.parametrize("cross_attention", [False, True])
def test_matches_manual_head_concatenation_and_projection(cross_attention: bool) -> None:
    torch.manual_seed(42)
    module = MultiHeadAttention(d_model=16, num_heads=4).eval()
    query = torch.randn(2, 3, 16)
    memory = torch.randn(2, 5, 16) if cross_attention else None
    key_length = memory.size(1) if memory is not None else query.size(1)
    mask = torch.rand(2, 4, query.size(1), key_length) > 0.25
    results = [
        head(query, memory, mask[:, index:index + 1], return_weights=True)
        for index, head in enumerate(module.heads)
    ]
    contexts, probabilities = zip(*results)
    expected = module.output_projection(torch.cat(contexts, dim=-1))
    out, weights = module(query, memory, mask, return_attention_weights=True)

    assert out.shape == (2, 3, 16)
    assert weights.shape == (2, 4, 3, key_length)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(weights, torch.cat(probabilities, dim=1))
    torch.testing.assert_close(module(query, memory, mask), expected)


def test_dropout_is_applied_once_per_head_and_disabled_in_eval() -> None:
    torch.manual_seed(42)
    module = MultiHeadAttention(16, 4, dropout=0.5)
    inputs = torch.randn(2, 4, 16)
    torch.manual_seed(7)
    output, weights = module(inputs, return_attention_weights=True)

    torch.manual_seed(7)
    expected = []
    for head in module.heads:
        query, key, value = head.project_qkv(inputs)
        probs = ((query @ key.transpose(-2, -1)) / module.head_dim**0.5).softmax(-1)
        context = torch.nn.functional.dropout(probs, p=0.5, training=True) @ value
        expected.append(context.squeeze(1))
    torch.testing.assert_close(output, module.output_projection(torch.cat(expected, dim=-1)))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 4))

    module.eval()
    eval_output, eval_weights = module(inputs, return_attention_weights=True)
    assert not torch.allclose(output, eval_output)
    torch.testing.assert_close(weights, eval_weights)
    torch.testing.assert_close(module(inputs), eval_output, rtol=0, atol=0)


def test_save_and_load_preserves_evaluation_outputs(tmp_path: Path) -> None:
    module = MultiHeadAttention(8, 4).eval()
    inputs = torch.randn(2, 4, 8)
    path = tmp_path / "multi_head.pt"
    torch.save(module.state_dict(), path)
    restored = MultiHeadAttention(8, 4).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(inputs), module(inputs), rtol=0, atol=0)


@pytest.mark.parametrize(("args", "message"), [
    ((0, 4, 0.1), "d_model"),
    ((256, 0, 0.1), "num_heads"),
    ((10, 4, 0.1), "divisible"),
    ((256, 4, -0.1), "dropout"),
    ((256, 4, 1.1), "dropout"),
    ((256, 4, float("nan")), "dropout"),
])
def test_invalid_configuration(args: tuple[int, int, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MultiHeadAttention(*args)


def test_invalid_input_shapes() -> None:
    module = MultiHeadAttention(8, 4)
    query = torch.zeros(2, 3, 8)
    with pytest.raises(ValueError, match="shape"):
        module(query[0], attention_mask=torch.ones(3, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="key_value_input must have shape"):
        module(query, torch.zeros(2, 4, 6))
    with pytest.raises(ValueError, match="same batch size"):
        module(query, query[:1])


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No CUDA or MPS device available",
)
def test_accelerator_agrees_with_cpu() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    module = MultiHeadAttention(16, 4).eval()
    query, memory = torch.randn(2, 3, 16), torch.randn(2, 5, 16)
    mask = torch.rand(2, 1, 3, 5) > 0.3
    expected = module(query, memory, mask)
    accelerated = module.to(device)(query.to(device), memory.to(device), mask.to(device))
    torch.testing.assert_close(accelerated.cpu(), expected, rtol=1e-5, atol=1e-5)
