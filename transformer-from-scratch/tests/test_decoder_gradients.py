import torch

from src.data.masks import cross_attention_mask
from src.model.config import ModelConfig
from src.model.decoder import TransformerDecoder
from src.model.encoder import TransformerEncoder


def test_repeated_train_backward_and_adamw_through_encoder_memory() -> None:
    torch.manual_seed(42)
    config = ModelConfig(d_model=16, d_ff=32, dropout=0.2)
    encoder, decoder = TransformerEncoder(32, 0, config), TransformerDecoder(32, 0, config)
    source = torch.tensor([[7, 8, 9, 0], [9, 10, 11, 12]])
    target = torch.tensor([[2, 5, 6], [2, 8, 0]])
    cross_mask = cross_attention_mask(source, target.size(1), 0)
    valid = target.ne(0)
    target_vectors = torch.randn(*target.shape, config.d_model)[valid]
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        memory = encoder(source)
        memory.retain_grad()
        states = decoder(target, memory, cross_attention_mask=cross_mask)
        (states[valid] - target_vectors).square().mean().backward()
        for gradient in (memory.grad, *(parameter.grad for parameter in parameters)):
            assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
        optimizer.step()
    assert all(not torch.equal(parameter, old) for parameter, old in zip(parameters, before))
    for module in (encoder, decoder):
        weight = module.embedding.token_embedding.embedding.weight
        assert torch.count_nonzero(weight.grad[0]) == torch.count_nonzero(weight[0]) == 0
