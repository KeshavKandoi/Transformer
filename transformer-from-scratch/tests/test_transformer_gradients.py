import torch

from scripts.inspect_full_transformer import masked_cross_entropy
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer


def test_cross_entropy_reaches_both_stacks_and_adamw_updates_parameters() -> None:
    torch.manual_seed(42)
    model = EncoderDecoderTransformer(32, 0, ModelConfig(d_model=16, d_ff=32))
    source = torch.tensor([[4, 5, 6, 0], [9, 10, 11, 12]])
    target = torch.tensor([[2, 7, 8], [2, 12, 0]])
    labels = torch.tensor([[7, 8, 3], [12, 3, 0]])
    details = model(source, target, return_diagnostics=True)
    details.encoder_memory.retain_grad()
    parameters = dict(model.named_parameters())
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3)
    masked_cross_entropy(details.logits, labels, 0).backward()
    assert details.encoder_memory.grad is not None
    assert torch.isfinite(details.encoder_memory.grad).all() and torch.count_nonzero(details.encoder_memory.grad)
    for name, parameter in parameters.items():
        gradient = parameter.grad
        assert parameter.requires_grad and gradient is not None, name
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient), name
    optimizer.step()
    assert all(not torch.equal(parameter, before[name]) for name, parameter in parameters.items())
    for module in (model.encoder, model.decoder):
        weight = module.embedding.token_embedding.embedding.weight
        assert torch.count_nonzero(weight.grad[0]) == torch.count_nonzero(weight[0]) == 0
