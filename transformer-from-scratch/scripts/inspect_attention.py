"""Inspect self-attention and cross-attention on a real translation batch."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig, data_parser
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.attention import scaled_attention_scores
from src.model.attention_head import SingleAttentionHead
from src.model.config import ModelConfig
from src.model.embeddings import TransformerEmbedding


def inspect_head(
    label: str,
    head: SingleAttentionHead,
    query_input: Tensor,
    mask: Tensor,
    key_value_input: Tensor | None = None,
) -> Tensor:
    context, probabilities = head(query_input, key_value_input, mask, return_weights=True)
    allowed = mask.expand_as(probabilities)
    if not torch.isfinite(context).all() or not torch.isfinite(probabilities).all():
        raise ValueError(f"{label} produced non-finite values")
    if torch.count_nonzero(probabilities[~allowed]):
        raise ValueError(f"{label} assigned probability to blocked positions")
    torch.testing.assert_close(probabilities.sum(-1), allowed.any(-1).to(probabilities.dtype))

    with torch.no_grad():
        query, key, value = head.project_qkv(query_input, key_value_input)
        scores = scaled_attention_scores(query, key)
    rows, columns = min(4, scores.size(-2)), min(8, scores.size(-1))
    print(f"\n{label}")
    print(f"  Q {list(query.shape)}, K {list(key.shape)}, V {list(value.shape)}")
    print(f"  Context {list(context.shape)}, probabilities {list(probabilities.shape)}")
    print(f"  First example, first {rows} queries and {columns} keys")
    print("  Scaled scores before masking -> allowed mask -> probabilities")
    score_rows = scores[0, 0, :rows, :columns].tolist()
    mask_rows = allowed[0, 0, :rows, :columns].tolist()
    probability_rows = probabilities[0, 0, :rows, :columns].detach().tolist()
    for score_row, mask_row, probability_row in zip(score_rows, mask_rows, probability_rows):
        scores_text = " ".join(f"{score:8.2f}" for score in score_row)
        mask_text = " ".join("T" if position else "F" for position in mask_row)
        probability_text = " ".join(f"{probability:.3f}" for probability in probability_row)
        print(f"    {scores_text}  |  {mask_text}  |  {probability_text}")
    print(f"  Finite values, correct probability sums, {int((~allowed).sum())} blocked entries at zero")
    return context.square().mean()


def check_gradient(label: str, tensor: Tensor) -> None:
    gradient = tensor.grad
    if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
        raise ValueError(f"{label} needs a finite, nonzero gradient")
    print(f"  {label} gradient norm: {gradient.norm().item():.6f}")


def inspect(args: argparse.Namespace) -> None:
    data = DataConfig.load(args.config)
    model = ModelConfig.load(args.model_config)
    model.validate_sequence_lengths(data.max_source_length, data.max_target_length)
    head_dim = model.head_dim
    tokenizer = BilingualBPETokenizer.load(args.tokenizer_dir)
    pairs, _ = read_parallel_csv(args.processed_dir / f"{args.split}.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(
        dataset,
        batch_size=data.batch_size,
        collate_fn=ParallelCollator(tokenizer.pad_id),
        generator=torch.Generator().manual_seed(data.seed),
    )
    batch = next(iter(loader))

    torch.manual_seed(data.seed)
    embedding_args = (tokenizer.vocab_size, model.d_model, model.max_seq_len, model.dropout, tokenizer.pad_id)
    encoder_embedding = TransformerEmbedding(*embedding_args).eval()
    decoder_embedding = TransformerEmbedding(*embedding_args).eval()
    encoder_x = encoder_embedding(batch["encoder_input_ids"])
    decoder_x = decoder_embedding(batch["decoder_input_ids"])
    encoder_x.retain_grad()
    decoder_x.retain_grad()

    print(f"Vocabulary {tokenizer.vocab_size}, padding ID {tokenizer.pad_id}, single-head width {head_dim}")
    print(f"First translation pair: {pairs[0].source} -> {pairs[0].target}")
    print(f"Encoder embeddings {list(encoder_x.shape)}, decoder embeddings {list(decoder_x.shape)}")
    print("Each row is one query. T allows a key; F blocks it. Evaluation mode disables dropout.")

    cases = (
        ("Encoder self-attention", encoder_x, "encoder_attention_mask", None),
        ("Decoder causal self-attention", decoder_x, "decoder_attention_mask", None),
        ("Decoder-to-encoder cross-attention", decoder_x, "cross_attention_mask", encoder_x),
    )
    heads = []
    losses = []
    for label, query_input, mask_key, key_value_input in cases:
        head = SingleAttentionHead(model.d_model, head_dim, model.attention_dropout).eval()
        heads.append((label, head))
        losses.append(inspect_head(label, head, query_input, batch[mask_key], key_value_input))

    torch.stack(losses).sum().backward()
    print("\nGradient check")
    for label, head in heads:
        for name, parameter in head.named_parameters():
            check_gradient(f"{label} {name}", parameter)
    check_gradient("Encoder embeddings", encoder_x)
    check_gradient("Decoder embeddings", decoder_x)
    check_gradient("Encoder token table", encoder_embedding.token_embedding.embedding.weight)
    check_gradient("Decoder token table", decoder_embedding.token_embedding.embedding.weight)


def main() -> None:
    parser = data_parser(__doc__ or "")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model_config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
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
