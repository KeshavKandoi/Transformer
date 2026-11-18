"""Translate English sentences with a trained local Transformer checkpoint."""

import argparse
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import ROOT
from src.inference import InferenceConfig, TransformerTranslator, TranslationResult


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "artifacts/tokenizer")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference_config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--length-penalty", type=float)


def resolved_config(args: argparse.Namespace, names: tuple[str, ...]) -> InferenceConfig:
    config = InferenceConfig.load(args.config)
    return replace(config, **{
        name: getattr(args, name) for name in names if getattr(args, name) is not None
    })


def display_text(text: str) -> str:
    """Keep terminal controls visible without executing them."""
    return "".join(f"\\x{ord(char):02x}" if unicodedata.category(char) == "Cc" else char for char in text)


def print_translation(translator: TransformerTranslator, text: str, config: InferenceConfig, diagnostics: bool) -> None:
    result = translator.translate(text, config, diagnostics=diagnostics)
    print(f"English: {display_text(text)}")
    translation = result.text if isinstance(result, TranslationResult) else result
    print(f"Hindi: {display_text(translation)}")
    if isinstance(result, TranslationResult):
        hypothesis = result.hypothesis
        print(f"EOS reached: {hypothesis.reached_eos} | Generated tokens: {len(hypothesis.token_ids)}")
        print(f"Raw generated IDs: {list(hypothesis.token_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--text")
    modes.add_argument("--interactive", action="store_true")
    parser.add_argument("--strategy", choices=("greedy", "beam", "sampling"))
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    errors = (OSError, ValueError, TypeError, RuntimeError, FloatingPointError)
    try:
        config = resolved_config(args, (
            "device", "strategy", "max_new_tokens", "beam_width", "length_penalty",
            "temperature", "top_k", "top_p", "seed",
        ))
        if args.text is not None and not args.text.strip():
            raise ValueError("English input must be a nonempty string")
        translator = TransformerTranslator.from_checkpoint(args.checkpoint, args.tokenizer_dir, config.device)
        if args.interactive:
            while True:
                try:
                    text = input("English (Ctrl-D to exit): ")
                except EOFError:
                    break
                try:
                    print_translation(translator, text, config, args.diagnostics)
                except errors as error:
                    print(f"Translation failed: {error}", file=sys.stderr)
        else:
            print_translation(translator, args.text, config, args.diagnostics)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except errors as error:
        print(f"Translation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
