"""Inspect the teacher-forced encoder-to-decoder pipeline on a real batch."""

import argparse
import io
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig, data_parser
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.decoder import TransformerDecoder
from src.model.encoder import TransformerEncoder


def inspect(args: argparse.Namespace) -> None:
    data = DataConfig.load(args.config)
    config = ModelConfig.load(args.model_config)
    config.validate_sequence_lengths(data.max_source_length, data.max_target_length)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")

    tokenizer = BilingualBPETokenizer.load(args.tokenizer_dir)
    pairs, _ = read_parallel_csv(args.processed_dir / f"{args.split}.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(
        dataset, batch_size=data.batch_size, collate_fn=ParallelCollator(tokenizer.pad_id),
        generator=torch.Generator().manual_seed(data.seed),
    )
    batch = {key: value.to(device) for key, value in next(iter(loader)).items()}
    source, target = batch["encoder_input_ids"], batch["decoder_input_ids"]
    source_mask = batch["encoder_attention_mask"]
    self_mask, cross_mask = batch["decoder_attention_mask"], batch["cross_attention_mask"]
    valid = target.ne(tokenizer.pad_id)

    torch.manual_seed(data.seed)
    encoder = TransformerEncoder(tokenizer.vocab_size, tokenizer.pad_id, config).to(device).eval()
    decoder = TransformerDecoder(tokenizer.vocab_size, tokenizer.pad_id, config).to(device).eval()
    print(f"{args.split} batch on {device} | vocabulary {tokenizer.vocab_size} | padding ID {tokenizer.pad_id}")
    print(f"Source: {pairs[0].source} | target: {pairs[0].target}")
    for label, ids in (("Source", source), ("Shifted decoder input", target)):
        print(f"  {label} shape {list(ids.shape)}, first row preview {ids[0, :24].tolist()}")
    memory = encoder(source, source_mask)
    memory.retain_grad()
    print(f"Encoder memory {list(memory.shape)}")

    hooks = [decoder.embedding.register_forward_hook(
        lambda _, inputs, result: print(f"Decoder embeddings {list(result.shape)}")
    )]
    for index, layer in enumerate(decoder.stack.layers, start=1):
        def show_layer(_, inputs, result, index=index):
            states, self_weights, cross_weights = result
            if not torch.isfinite(states).all() or torch.count_nonzero(states[~valid]):
                raise ValueError(f"Block {index} has non-finite states or nonzero padded queries")
            print(f"  Block {index} states {list(states.shape)}, self {list(self_weights.shape)}, cross {list(cross_weights.shape)}")
        hooks.append(layer.register_forward_hook(show_layer))
    try:
        states, self_maps, cross_maps = decoder(
            target, memory, self_mask, cross_mask, return_attention_weights=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    print(f"Decoder states {list(states.shape)}, self maps {list(self_maps.shape)}, cross maps {list(cross_maps.shape)}")
    for label, maps, mask in (("Causal self", self_maps, self_mask), ("Cross", cross_maps, cross_mask)):
        allowed = mask.unsqueeze(1)
        if not torch.isfinite(maps).all() or torch.count_nonzero(maps.masked_select(~allowed)):
            raise ValueError(f"{label} maps must be finite with zero probability on blocked keys")
        expected = allowed.any(-1).expand_as(maps[..., 0]).to(maps.dtype)
        torch.testing.assert_close(maps.sum(-1), expected)
        print(f"{label}: blocked keys have zero probability. First example and head, up to 3 × 4 entries per layer")
        for index, matrix in enumerate(maps[0, :, 0, :3, :4].detach().tolist(), start=1):
            print(f"  Block {index} " + " | ".join(" ".join(f"{value:.3f}" for value in row) for row in matrix))
    print(f"Finite states, {int((~valid).sum())} padded decoder query vectors at zero")

    for label, module in (("Decoder embedding", decoder.embedding), ("Decoder stack", decoder.stack),
                          ("Decoder total", decoder), ("Encoder total", encoder)):
        print(f"{label} trainable parameters: {sum(p.numel() for p in module.parameters() if p.requires_grad):,}")
    all_parameters = [parameter for layer in decoder.stack.layers for parameter in layer.parameters()]
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    if len({p.untyped_storage().data_ptr() for p in parameters}) != len(parameters):
        raise ValueError("Parameter storage is shared unexpectedly")
    print(f"Verified independent storage for all {len(all_parameters)} decoder-stack parameter tensors")

    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    before = [parameter.detach().clone() for parameter in parameters]
    loss = (states[valid] - torch.randn_like(states[valid])).square().mean()
    loss.backward()
    for label, module in (("Encoder", encoder), ("Decoder embedding", decoder.embedding),
                          *((f"Decoder block {i}", layer) for i, layer in enumerate(decoder.stack.layers, 1))):
        norms = []
        for name, parameter in module.named_parameters():
            grad = parameter.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
                raise ValueError(f"Expected finite nonzero gradients for {label} {name}")
            norms.append(grad.norm().item())
        print(f"{label}: {len(norms)} gradient tensors, norms [{min(norms):.6f}, {max(norms):.6f}]")
    if memory.grad is None or not torch.isfinite(memory.grad).all() or not torch.count_nonzero(memory.grad):
        raise ValueError("Expected gradients through encoder memory")
    print(f"Encoder memory gradient norm {memory.grad.norm().item():.6f}")
    optimizer.step()
    changed = sum(not torch.equal(parameter, old) for parameter, old in zip(parameters, before))
    if changed != len(parameters):
        raise ValueError(f"Expected all parameter tensors to update, got {changed}/{len(parameters)}")
    print(f"AdamW updated {changed}/{len(parameters)} parameter tensors on the temporary objective")

    checkpoint = io.BytesIO()
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    copy_encoder = TransformerEncoder(tokenizer.vocab_size, tokenizer.pad_id, config).to(device).eval()
    copy_decoder = TransformerDecoder(tokenizer.vocab_size, tokenizer.pad_id, config).to(device).eval()
    copy_encoder.load_state_dict(saved["encoder"])
    copy_decoder.load_state_dict(saved["decoder"])
    with torch.no_grad():
        expected = decoder(target, encoder(source, source_mask), self_mask, cross_mask, return_attention_weights=True)
        actual = copy_decoder(target, copy_encoder(source, source_mask), self_mask, cross_mask, return_attention_weights=True)
        for restored, original in zip(actual, expected):
            torch.testing.assert_close(restored, original)
    print("State dictionaries reproduced decoder states and both attention maps")


def main() -> None:
    parser = data_parser(__doc__ or "")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model_config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        inspect(parser.parse_args())
    except (OSError, ValueError, AssertionError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
