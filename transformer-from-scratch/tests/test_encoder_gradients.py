import torch

from src.model.config import ModelConfig
from src.model.encoder import TransformerEncoder


def test_all_layers_receive_gradients_and_adamw_updates_parameters() -> None:
    torch.manual_seed(42)
    config = ModelConfig(d_model=16, d_ff=32, dropout=0.2, attention_dropout=0.1)
    encoder = TransformerEncoder(32, 0, config)
    ids = torch.tensor([[4, 7, 8, 0], [9, 10, 11, 12]])
    valid = ids.ne(0)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    parameters = dict(encoder.named_parameters())
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    target = torch.randn(*ids.shape, config.d_model)[valid]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        memory = encoder(ids)
        (memory[valid] - target).square().mean().backward()
        for name, parameter in parameters.items():
            gradient = parameter.grad
            assert gradient is not None, name
            assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient), name
        assert torch.count_nonzero(parameters["embedding.token_embedding.embedding.weight"].grad[0]) == 0
        optimizer.step()
    assert all(not torch.equal(parameter, before[name]) for name, parameter in parameters.items())
    assert torch.count_nonzero(parameters["embedding.token_embedding.embedding.weight"][0]) == 0
    encoder.eval()
    with torch.no_grad():
        torch.testing.assert_close(encoder(ids), encoder(ids), rtol=0, atol=0)
