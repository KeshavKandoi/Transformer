from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from scripts.prepare_data import prepare
from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig
from src.data.dataset import ParallelDataset
from src.data.masks import cross_attention_mask
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.decoder import TransformerDecoder
from src.model.encoder import TransformerEncoder


def test_real_batch_forward_backward_memory_sensitivity_and_reload(tmp_path: Path) -> None:
    torch.manual_seed(42)
    config_path = ROOT / "configs/data_config.yaml"
    data = DataConfig.load(config_path)
    config = ModelConfig.load(ROOT / "configs/model_config.yaml")
    assert config.num_decoder_layers == config.num_encoder_layers == 4
    processed, artifacts = tmp_path / "processed", tmp_path / "tokenizer"
    prepare(Namespace(
        config=config_path, input=ROOT / "data/raw/sample_parallel.csv",
        processed_dir=processed, tokenizer_dir=artifacts,
    ))
    tokenizer = BilingualBPETokenizer.load(artifacts)
    pairs, _ = read_parallel_csv(processed / "train.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(dataset, batch_size=data.batch_size, collate_fn=ParallelCollator(tokenizer.pad_id))
    batch = next(iter(loader))
    source, target = batch["encoder_input_ids"], batch["decoder_input_ids"]
    assert source.size(1) != target.size(1)
    assert (target[:, 0] == tokenizer.bos_id).all()
    assert torch.equal(target[:, 1:][target[:, 1:].ne(tokenizer.pad_id)],
                       batch["decoder_target_ids"][:, :-1][target[:, 1:].ne(tokenizer.pad_id)])

    args = (tokenizer.vocab_size, tokenizer.pad_id, config)
    encoder, decoder = TransformerEncoder(*args).eval(), TransformerDecoder(*args).eval()
    memory = encoder(source, batch["encoder_attention_mask"])
    memory.retain_grad()
    masks = (batch["decoder_attention_mask"], batch["cross_attention_mask"])
    states, self_maps, cross_maps = decoder(target, memory, *masks, return_attention_weights=True)
    b, t = target.shape
    s = source.size(1)
    assert states.shape == (b, t, config.d_model)
    assert self_maps.shape == (b, 4, config.num_heads, t, t)
    assert cross_maps.shape == (b, 4, config.num_heads, t, s)
    for probabilities, mask in zip((self_maps, cross_maps), masks):
        assert torch.isfinite(probabilities).all()
        assert torch.count_nonzero(probabilities.masked_select(~mask.unsqueeze(1))) == 0
        torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))
    valid = target.ne(tokenizer.pad_id)
    assert torch.isfinite(states).all() and torch.count_nonzero(states[~valid]) == 0

    with torch.no_grad():
        source_valid = source.ne(tokenizer.pad_id)
        changed = memory.clone()
        changed[~source_valid] = 30
        torch.testing.assert_close(decoder(target, changed, *masks), states)
        changed[source_valid] += torch.randn_like(changed[source_valid])
        assert not torch.allclose(decoder(target, changed, *masks)[valid], states[valid])

        prefix = [tokenizer.bos_id, *tokenizer.encode("हम")]
        first = torch.tensor([[*prefix, *tokenizer.encode(" दोस्त हैं")]])
        second = torch.tensor([[*prefix, *tokenizer.encode(" कुछ और")]])
        first_states = decoder(first, memory[:1], cross_attention_mask=cross_attention_mask(
            source[:1], first.size(1), tokenizer.pad_id,
        ))
        second_states = decoder(second, memory[:1], cross_attention_mask=cross_attention_mask(
            source[:1], second.size(1), tokenizer.pad_id,
        ))
        torch.testing.assert_close(first_states[:, :len(prefix)], second_states[:, :len(prefix)])

    parameters = dict(
        list(("encoder." + name, p) for name, p in encoder.named_parameters())
        + list(("decoder." + name, p) for name, p in decoder.named_parameters())
    )
    before = {name: p.detach().clone() for name, p in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3)
    (states[valid] - torch.randn_like(states[valid])).square().mean().backward()
    assert memory.grad is not None and torch.isfinite(memory.grad).all() and torch.count_nonzero(memory.grad)
    for name, parameter in parameters.items():
        grad = parameter.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad), name
    optimizer.step()
    assert all(not torch.equal(p, before[name]) for name, p in parameters.items())

    path = tmp_path / "encoder_decoder.pt"
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    saved_encoder, saved_decoder = TransformerEncoder(*args).eval(), TransformerDecoder(*args).eval()
    saved_encoder.load_state_dict(checkpoint["encoder"])
    saved_decoder.load_state_dict(checkpoint["decoder"])
    assert saved_decoder.config == config
    with torch.no_grad():
        original = decoder(target, encoder(source), *masks, return_attention_weights=True)
        restored = saved_decoder(target, saved_encoder(source), *masks, return_attention_weights=True)
        for actual, expected in zip(restored, original):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("source_length", "target_length"), [(1, 1), (3, 128), (128, 4)])
def test_short_max_length_and_padding_heavy_single_batches(source_length: int, target_length: int) -> None:
    config = ModelConfig(d_model=16, d_ff=32)
    encoder, decoder = TransformerEncoder(32, 0, config).eval(), TransformerDecoder(32, 0, config).eval()
    source, target = torch.zeros(1, source_length, dtype=torch.long), torch.zeros(1, target_length, dtype=torch.long)
    source[:, 0], target[:, 0] = 4, 2
    with torch.no_grad():
        memory = encoder(source)
        output = decoder(target, memory, cross_attention_mask=cross_attention_mask(source, target_length, 0))
    assert output.shape == (1, target_length, config.d_model)
    assert torch.isfinite(output).all() and torch.count_nonzero(output[:, 1:]) == 0


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No CUDA or MPS device available",
)
def test_accelerator_smoke() -> None:
    device = "cuda" if torch.cuda.is_available() else "mps"
    config = ModelConfig(d_model=16, d_ff=32)
    encoder = TransformerEncoder(32, 0, config).to(device).eval()
    decoder = TransformerDecoder(32, 0, config).to(device).eval()
    source = torch.tensor([[4, 5, 0]], device=device)
    target = torch.tensor([[2, 7]], device=device)
    output = decoder(target, encoder(source), cross_attention_mask=cross_attention_mask(source, 2, 0))
    assert output.device.type == device and torch.isfinite(output).all()
    output[..., 0].square().mean().backward()
    grad = encoder.embedding.token_embedding.embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)
