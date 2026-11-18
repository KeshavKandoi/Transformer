"""Inspect encoder memory, attention maps, gradients and serialization."""

import argparse
import io
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig, data_parser
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
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
    batch = next(iter(loader))
    ids = batch["encoder_input_ids"].to(device)
    key_mask = batch["encoder_attention_mask"].to(device)
    valid = ids.ne(tokenizer.pad_id)

    torch.manual_seed(data.seed)
    encoder = TransformerEncoder(tokenizer.vocab_size, tokenizer.pad_id, config).to(device).eval()
    print(f"{args.split} batch on {device} | vocabulary {tokenizer.vocab_size} | padding ID {tokenizer.pad_id}")
    print(f"Source IDs {list(ids.shape)}, {len(encoder.stack.layers)} blocks, {config.num_heads} heads per block")
    for row in ids[:3].tolist():
        print(f"  IDs (first 24) {row[:24]} -> {tokenizer.decode(row)}")

    def show_shape(label: str) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor | tuple[Tensor, Tensor]], None]:
        def hook(_: nn.Module, inputs: tuple[Tensor, ...], result: Tensor | tuple[Tensor, Tensor]) -> None:
            output = result[0] if isinstance(result, tuple) else result
            if not torch.isfinite(output).all() or torch.count_nonzero(output[~valid]):
                raise ValueError(f"{label} needs finite outputs and zero padded queries")
            print(f"  {label} {list(inputs[0].shape)} -> {list(output.shape)}")
        return hook

    hooks = [encoder.embedding.register_forward_hook(show_shape("Embedding"))]
    hooks.extend(
        block.register_forward_hook(show_shape(f"Block {index}"))
        for index, block in enumerate(encoder.stack.layers, start=1)
    )
    try:
        memory, maps = encoder(ids, key_mask, return_attention_weights=True)
    finally:
        for handle in hooks:
            handle.remove()
    print(f"Encoder memory {list(memory.shape)}, attention maps {list(maps.shape)}")
    if not torch.isfinite(maps).all() or torch.count_nonzero(maps.masked_select(~valid[:, None, None, None, :])):
        raise ValueError("Expected finite attention maps with zero probability on padded keys")
    torch.testing.assert_close(maps.sum(-1), torch.ones_like(maps[..., 0]))
    print(f"  Finite output, padded keys blocked, {int((~valid).sum())} padded query vectors zero")
    size = min(4, ids.size(1))
    print(f"  First example and head, first {size} query rows and key columns")
    for index, matrix in enumerate(maps[0, :, 0, :size, :size].detach().tolist(), start=1):
        print(f"  Layer {index}")
        for row in matrix:
            print("    " + " ".join(f"{value:.3f}" for value in row))

    groups = {"Embedding": encoder.embedding, "Stack": encoder.stack, "Total": encoder}
    for label, module in groups.items():
        count = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        print(f"{label} trainable parameters: {count:,}")
    block_parameters = [parameter for block in encoder.stack.layers for parameter in block.parameters()]
    if len({parameter.untyped_storage().data_ptr() for parameter in block_parameters}) != len(block_parameters):
        raise ValueError("Encoder blocks must have independent parameter storage")
    print("Encoder block parameters are independent")

    parameters = dict(encoder.named_parameters())
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3)
    loss = (memory[valid] - torch.randn_like(memory[valid])).square().mean()
    loss.backward()
    norms: dict[str, float] = {}
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
            raise ValueError(f"Expected a finite, nonzero gradient for {name}")
        norms[name] = gradient.norm().item()
    print(f"Temporary objective {loss.item():.6f}")
    print(f"  Embedding table gradient norm {norms['embedding.token_embedding.embedding.weight']:.6f}")
    for index in range(config.num_encoder_layers):
        prefix = f"stack.layers.{index}."
        layer_norms = [norm for name, norm in norms.items() if name.startswith(prefix)]
        print(f"  Block {index + 1}: {len(layer_norms)} gradients, norm range [{min(layer_norms):.6f}, {max(layer_norms):.6f}]")
    optimizer.step()
    changed = sum(not torch.equal(parameter, before[name]) for name, parameter in parameters.items())
    if changed != len(parameters):
        raise ValueError(f"Expected all parameter tensors to update, got {changed}/{len(parameters)}")
    if torch.count_nonzero(encoder.embedding.token_embedding.embedding.weight[tokenizer.pad_id]):
        raise ValueError("The embedding padding row must remain zero after optimization")
    print(f"AdamW updated {changed}/{len(parameters)} parameter tensors. Embedding padding row stayed zero.")

    checkpoint = io.BytesIO()
    torch.save(encoder.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = TransformerEncoder(tokenizer.vocab_size, tokenizer.pad_id, config).to(device).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with torch.no_grad():
        expected, expected_maps = encoder(ids, key_mask, return_attention_weights=True)
        actual, actual_maps = restored(ids, key_mask, return_attention_weights=True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_maps, expected_maps)
    print("State dictionary round-trip reproduced memory and attention maps")


def main() -> None:
    parser = data_parser(__doc__ or "")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model_config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        inspect(parser.parse_args())
    except FileNotFoundError as error:
        logging.error("%s. For missing Phase 1 artifacts, run scripts/prepare_data.py.", error)
        raise SystemExit(1) from error
    except (OSError, ValueError, AssertionError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
