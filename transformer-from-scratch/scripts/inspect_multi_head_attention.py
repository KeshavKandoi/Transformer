"""Inspect multi-head attention on an English–Hindi translation batch."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inspect_attention import check_gradient
from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig, data_parser
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.embeddings import TransformerEmbedding
from src.model.multi_head_attention import MultiHeadAttention


def inspect_attention(
    label: str,
    module: MultiHeadAttention,
    inputs: Tensor,
    mask: Tensor,
    memory: Tensor | None,
) -> Tensor:
    output, weights = module(inputs, memory, mask, return_attention_weights=True)
    allowed = mask.expand_as(weights)
    if not torch.isfinite(output).all() or not torch.isfinite(weights).all():
        raise ValueError(f"{label} produced NaNs or infinities")
    if torch.count_nonzero(weights[~allowed]):
        raise ValueError(f"{label} assigned probability to a blocked key")
    torch.testing.assert_close(weights.sum(-1), allowed.any(-1).to(weights.dtype))

    with torch.no_grad():
        query, key, value = module.heads[0].project_qkv(inputs, memory)

    print(f"\n{label}")
    print(f"  Per head Q {list(query.shape)}, K {list(key.shape)}, V {list(value.shape)}")
    print(f"  Concatenated {[*inputs.shape[:2], module.num_heads * query.size(-1)]}")
    print(f"  Projected {list(output.shape)}, probabilities {list(weights.shape)}")
    print(f"  Attention parameters: {sum(parameter.numel() for parameter in module.parameters()):,}")
    print(f"  All {int((~allowed).sum())} blocked entries are zero. No NaNs or infinities.")
    rows, cols = min(4, weights.size(-2)), min(6, weights.size(-1))
    print(f"  First sample, first {rows} queries and {cols} keys per head. '-' marks a blocked key.")
    for index in range(module.num_heads):
        print(f"  Head {index + 1}")
        probabilities = weights[0, index, :rows, :cols].detach().tolist()
        mask_rows = allowed[0, index, :rows, :cols].tolist()
        for row, mask_row in zip(probabilities, mask_rows):
            print("    " + " ".join(f"{p:.3f}" if valid else "    -" for p, valid in zip(row, mask_row)))
    return output.square().mean()


def inspect(args: argparse.Namespace) -> None:
    data = DataConfig.load(args.config)
    model = ModelConfig.load(args.model_config)
    model.validate_sequence_lengths(data.max_source_length, data.max_target_length)
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

    print(f"Vocabulary size {tokenizer.vocab_size}, padding ID {tokenizer.pad_id}")
    print(f"Embedding width {model.d_model}, {model.num_heads} heads of width {model.head_dim}")
    print(f"Encoder inputs {list(encoder_x.shape)}, decoder inputs {list(decoder_x.shape)}")
    print(f"First pair: {pairs[0].source} -> {pairs[0].target}")
    print("Rows represent queries and columns represent keys. Evaluation disables attention dropout.")

    cases = (
        ("Encoder self-attention", encoder_x, "encoder_attention_mask", None),
        ("Decoder causal self-attention", decoder_x, "decoder_attention_mask", None),
        ("Decoder-to-encoder cross-attention", decoder_x, "cross_attention_mask", encoder_x),
    )
    modules = []
    losses = []
    for label, inputs, mask_key, memory in cases:
        module = MultiHeadAttention(model.d_model, model.num_heads, model.attention_dropout).eval()
        modules.append((label, module))
        losses.append(inspect_attention(label, module, inputs, batch[mask_key], memory))

    torch.stack(losses).sum().backward()
    print("\nGradient check")
    for label, module in modules:
        for name, parameter in module.named_parameters():
            check_gradient(f"{label} {name}", parameter)
    check_gradient("Encoder representations", encoder_x)
    check_gradient("Decoder representations", decoder_x)
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
    except (OSError, ValueError, TypeError, AssertionError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
