"""Inspect logits, masked cross-entropy, gradients and a checkpoint round-trip."""

import argparse
import io
import logging
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig, data_parser
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer, TransformerDiagnostics


def masked_cross_entropy(logits: Tensor, labels: Tensor, pad_id: int) -> Tensor:
    """Calculate a mean over non-padding labels for the Phase 8 smoke test."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [B, T, V] and matching labels [B, T]")
    if labels.dtype != torch.long:
        raise TypeError("Labels must have dtype torch.long")
    vocab_size = logits.size(-1)
    if not 0 <= pad_id < vocab_size or ((labels < 0) | (labels >= vocab_size)).any():
        raise ValueError("Labels and the padding ID must be within the vocabulary")
    if not labels.ne(pad_id).any():
        raise ValueError("Cross-entropy needs at least one non-padding target")
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=pad_id)


def inspect(args: argparse.Namespace) -> None:
    data = DataConfig.load(args.config)
    config = ModelConfig.load(args.model_config)
    config.validate_sequence_lengths(data.max_source_length, data.max_target_length)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    device = torch.device(args.device)
    tokenizer = BilingualBPETokenizer.load(args.tokenizer_dir)
    pairs, _ = read_parallel_csv(args.processed_dir / f"{args.split}.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(
        dataset, batch_size=data.batch_size, collate_fn=ParallelCollator(tokenizer.pad_id),
        generator=torch.Generator().manual_seed(data.seed),
    )
    batch = {key: value.to(device) for key, value in next(iter(loader)).items()}
    labels = batch.pop("decoder_target_ids")
    torch.manual_seed(data.seed)
    model = EncoderDecoderTransformer(tokenizer.vocab_size, tokenizer.pad_id, config).to(device).eval()
    print(f"{args.split} split on {device} | vocabulary {tokenizer.vocab_size} | pad ID {tokenizer.pad_id}")
    print(f"Width {config.d_model}, FFN {config.d_ff}, heads {config.num_heads}, layers {config.num_encoder_layers} encoder / {config.num_decoder_layers} decoder")
    print(f"Source {list(batch['encoder_input_ids'].shape)}, decoder input {list(batch['decoder_input_ids'].shape)}, labels {list(labels.shape)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    with torch.no_grad():
        details = model(**batch, return_diagnostics=True)
        assert isinstance(details, TransformerDiagnostics)
        print(f"Encoder memory {list(details.encoder_memory.shape)}, decoder states {list(details.decoder_hidden_states.shape)}, logits {list(details.logits.shape)}")
        for title, weights, mask_key in (
            ("Encoder", details.encoder_attention_maps, "encoder_attention_mask"),
            ("Decoder self", details.decoder_self_attention_maps, "decoder_attention_mask"),
            ("Decoder cross", details.decoder_cross_attention_maps, "cross_attention_mask"),
        ):
            allowed = batch[mask_key].unsqueeze(1)
            if not torch.isfinite(weights).all() or torch.count_nonzero(weights.masked_select(~allowed)):
                raise ValueError(f"Invalid {title} attention probabilities")
            expected = allowed.any(-1).expand_as(weights[..., 0]).to(weights.dtype)
            torch.testing.assert_close(weights.sum(-1), expected)
            print(f"{title} maps {list(weights.shape)}, blocked keys have zero probability")
        row, position = labels.ne(tokenizer.pad_id).nonzero()[0].tolist()
        scores, token_ids = details.logits[row, position].topk(min(5, tokenizer.vocab_size))
        print("Untrained predictions for the first valid target position")
        for token_id, score in zip(token_ids.tolist(), scores.tolist()):
            print(f"  ID {token_id:>3} logit {score:+.6f} decoded {tokenizer.decode([token_id], skip_special_tokens=False)!r}")
        sample = details.logits[row, :6].argmax(-1).tolist()
        print(f"  First position-wise argmax IDs {sample} -> {tokenizer.decode(sample)!r}")

    logits = model(**batch)
    torch.testing.assert_close(logits, details.logits)
    loss = masked_cross_entropy(logits, labels, tokenizer.pad_id)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss) or loss.ndim or not loss.requires_grad:
        raise ValueError("Expected finite logits and a differentiable scalar loss")
    print(f"Initial cross-entropy {loss.item():.6f} over {int(labels.ne(tokenizer.pad_id).sum())} valid labels")
    parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3)
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    loss.backward()
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
            raise ValueError(f"Expected finite nonzero gradient for {name}")
    for name in ("encoder.embedding.token_embedding.embedding.weight",
                 "decoder.stack.layers.0.cross_attention.heads.0.Wq.weight",
                 "output_projection.projection.weight"):
        print(f"  Gradient norm {name}: {parameters[name].grad.norm().item():.6f}")
    optimizer.step()
    changed = sum(not torch.equal(parameter, before[name]) for name, parameter in parameters.items())
    if changed != len(parameters):
        raise ValueError(f"AdamW only updated {changed}/{len(parameters)} parameter tensors")
    print(f"AdamW updated {changed}/{len(parameters)} parameter tensors")

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = EncoderDecoderTransformer(tokenizer.vocab_size, tokenizer.pad_id, config).to(device).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch), model(**batch))
    print("State dictionary round-trip reproduced logits")


def main() -> None:
    parser = data_parser(__doc__ or "")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model_config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        inspect(parser.parse_args())
    except (OSError, ValueError, TypeError, AssertionError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
