"""Compare greedy and beam translation on individual or prepared source examples."""

import argparse
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.translate import add_model_arguments, resolved_config
from src.data.cleaning import read_parallel_csv
from src.data.config import ROOT
from src.inference import InferenceConfig, TransformerTranslator


def inspect_text(translator: TransformerTranslator, text: str, config: InferenceConfig, top_candidates: bool) -> None:
    print(f"English: {text}")
    for strategy in ("greedy", "beam"):
        start = time.perf_counter()
        result = translator.generate([text], replace(config, strategy=strategy), trace=top_candidates)[0]
        elapsed = time.perf_counter() - start
        hypothesis = result.hypothesis
        if strategy == "greedy":
            print(f"  Encoder IDs: {list(result.source_ids)} | memory: {result.encoder_memory_shape}")
            print(f"  Initial decoder prefix: {list(result.initial_prefix)}")
        print(f"  {strategy} Hindi: {result.text!r}")
        print(f"  IDs: {list(hypothesis.token_ids)} | EOS: {hypothesis.reached_eos} | tokens: {len(hypothesis.token_ids)} | {elapsed:.4f}s")
        for index, step in enumerate(hypothesis.steps, start=1):
            candidates = ", ".join(f"{token}:{score:+.4f}" for token, score in step.candidates)
            print(f"    step {index}: selected {step.token_id}, candidates (ID:logit) {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--text")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--examples-per-split", type=int, default=0)
    parser.add_argument("--top-candidates", action="store_true")
    args = parser.parse_args()
    try:
        if args.examples_per_split < 0:
            raise ValueError("examples-per-split cannot be negative")
        config = resolved_config(args, ("device", "max_new_tokens", "beam_width", "length_penalty"))
        translator = TransformerTranslator.from_checkpoint(args.checkpoint, args.tokenizer_dir, config.device)
        model = translator.model
        print(f"Checkpoint: {args.checkpoint} | device: {translator.device} | vocabulary: {translator.tokenizer.vocab_size}")
        print(f"Model: {asdict(model.config)}")
        before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        if args.text is not None or not args.examples_per_split:
            inspect_text(translator, args.text if args.text is not None else "We are friends", config, args.top_candidates)
        if args.examples_per_split:
            for split in ("train", "val", "test"):
                pairs, _ = read_parallel_csv(args.processed_dir / f"{split}.csv")
                print(f"\n{split} examples ({'training' if split == 'train' else 'held out'})")
                for pair in pairs[:args.examples_per_split]:
                    print(f"Reference Hindi: {pair.target}")
                    inspect_text(translator, pair.source, config, args.top_candidates)
        if not all(torch.equal(before[name], parameter.detach().cpu()) for name, parameter in model.named_parameters()):
            raise ValueError("Generation changed model parameters")
        print("All next-token logits passed finite checks. Model parameters stayed unchanged.")
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
