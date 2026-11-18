"""Inspect Phase 5 components on one English–Hindi translation batch."""

import argparse
import logging
import sys
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
from src.model.embeddings import TransformerEmbedding
from src.model.feed_forward import PositionwiseFeedForward
from src.model.layer_norm import LayerNormalization
from src.model.multi_head_attention import MultiHeadAttention
from src.model.residual import ResidualAddNorm


def inspect(args: argparse.Namespace) -> None:
    data = DataConfig.load(args.config)
    model = ModelConfig.load(args.model_config)
    model.validate_sequence_lengths(data.max_source_length, data.max_target_length)
    model.validate_sublayer_settings()
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
    attention_mask = batch["encoder_attention_mask"].to(device)
    valid = ids.ne(tokenizer.pad_id)

    torch.manual_seed(data.seed)
    embedding = TransformerEmbedding(
        tokenizer.vocab_size, model.d_model, model.max_seq_len, model.dropout, tokenizer.pad_id,
    )
    attention = MultiHeadAttention(model.d_model, model.num_heads, model.attention_dropout)
    attention_residual = ResidualAddNorm(model.d_model, model.dropout, model.layer_norm_eps)
    ffn = PositionwiseFeedForward(model.d_model, model.d_ff, model.dropout)
    ffn_residual = ResidualAddNorm(model.d_model, model.dropout, model.layer_norm_eps)
    components = nn.ModuleDict({
        "embedding": embedding, "attention": attention, "attention_residual": attention_residual,
        "ffn": ffn, "ffn_residual": ffn_residual,
    }).to(device).eval()

    print(f"{args.split} batch on {device} | vocabulary {tokenizer.vocab_size} | padding ID {tokenizer.pad_id}")
    print(f"Source IDs {list(ids.shape)}, target IDs {list(batch['decoder_input_ids'].shape)}")
    print(f"First source: {pairs[0].source}")
    example = torch.tensor([2., 4., 6.], device=device)
    example_norm = LayerNormalization(3, model.layer_norm_eps).to(device)
    print(f"LayerNorm example: {example.tolist()} -> {example_norm(example).detach().tolist()}")
    print(f"Initial gamma (first 4): {attention_residual.norm.gamma[:4].tolist()}")
    print(f"Initial beta (first 4): {attention_residual.norm.beta[:4].tolist()}")

    def show_shape(layer: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        print(f"  FFN {type(layer).__name__}: {list(inputs[0].shape)} -> {list(output.shape)}")

    x = embedding(ids)
    a = attention(x, attention_mask=attention_mask)
    y = attention_residual(x, a, valid)
    print(f"  Embedding {list(x.shape)}, attention {list(a.shape)}, first post-norm {list(y.shape)}")
    hooks = [layer.register_forward_hook(show_shape) for layer in (ffn.linear1, ffn.activation, ffn.linear2)]
    try:
        f = ffn(y)
    finally:
        for hook in hooks:
            hook.remove()
    z = ffn_residual(y, f, valid)
    print(f"  FFN output {list(f.shape)}, final post-norm {list(z.shape)}")
    for label, values in (("Before first norm", x + a), ("After first norm", y), ("After final norm", z)):
        token = values[valid][0].detach()
        print(f"  {label}: mean {token.mean().item():.6f}, population variance {token.var(unbiased=False).item():.6f}")

    if not all(torch.isfinite(tensor).all() for tensor in (x, a, y, f, z)):
        raise ValueError("Non-finite values in the pipeline")
    if any(torch.count_nonzero(tensor[~valid]) for tensor in (x, y, z)):
        raise ValueError("Padded query positions must be zero after embedding and both post-norms")
    with torch.no_grad():
        if any(module.training for module in components.modules() if isinstance(module, nn.Dropout)):
            raise ValueError("Dropout should be disabled during evaluation")
        replay_x = embedding(ids)
        replay_y = attention_residual(replay_x, attention(replay_x, attention_mask=attention_mask), valid)
        torch.testing.assert_close(ffn_residual(replay_y, ffn(replay_y), valid), z)
    print(f"  Finite activations, deterministic eval, {int((~valid).sum())} padded queries kept at zero")

    parameters = dict(components.named_parameters())
    if not all(parameter.requires_grad for parameter in parameters.values()):
        raise ValueError("Expected all registered parameters to be trainable")
    optimizer = torch.optim.SGD(parameters.values(), lr=0.01)
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    # A random regression target avoids a constant objective on normalized rows.
    loss = (z[valid] - torch.randn_like(z[valid])).square().mean()
    loss.backward()
    norms = []
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
            raise ValueError(f"Expected a finite, nonzero gradient for {name}")
        norms.append(gradient.norm().item())
    optimizer.step()
    changed = sum(not torch.equal(parameter, before[name]) for name, parameter in parameters.items())
    if changed != len(parameters):
        raise ValueError(f"Expected all parameter tensors to update, got {changed}/{len(parameters)}")
    if torch.count_nonzero(embedding.token_embedding.embedding.weight[tokenizer.pad_id]):
        raise ValueError("Padding row changed after the optimizer step")
    print(f"Temporary objective {loss.item():.6f}")
    print(f"  {len(parameters)} trainable tensors with gradient norms in [{min(norms):.6f}, {max(norms):.6f}]")
    print(f"  SGD updated {changed}/{len(parameters)} parameter tensors. Padding row stayed zero.")


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
