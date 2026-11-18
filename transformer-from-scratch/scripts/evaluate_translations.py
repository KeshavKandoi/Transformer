"""Score generated greedy and beam translations on a held-out split."""

import argparse
import csv
import json
import sys
import tempfile
import time
import unicodedata
from dataclasses import asdict
from itertools import groupby
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import read_parallel_csv
from src.data.config import ROOT, DataConfig
from src.data.prepared import file_digest
from src.inference import InferenceConfig, TransformerTranslator
from src.training.checkpoint import experiment_fingerprints, load_checkpoint, validate_checkpoint
from src.training.utils import load_prepared_data


def score_translations(hypotheses: list[str], references: list[str]) -> dict:
    """Single-reference corpus chrF2 and international-tokenized corpus BLEU."""
    if not hypotheses or len(hypotheses) != len(references):
        raise ValueError("Need equal nonempty hypothesis and reference lists")
    output = {}
    for name, metric in (("chrf", CHRF()), ("bleu", BLEU(tokenize="intl", effective_order=True))):
        result = metric.corpus_score(hypotheses, [references])
        output[name] = {"score": result.score, "signature": str(metric.get_signature())}
    return output


def export_curves(metrics_path: Path, destination: Path, checkpoint: dict) -> None:
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoint_epoch = checkpoint["epoch"]
    selected = [row for row in rows if row["epoch"] <= checkpoint_epoch]
    if not selected or selected[-1]["epoch"] != checkpoint_epoch:
        raise ValueError("Training metrics must cover the checkpoint epoch")
    expected = experiment_fingerprints(checkpoint)
    last = selected[-1]
    if last["global_step"] != checkpoint["global_step"] or last.get("fingerprints") != expected:
        raise ValueError("Training metric record does not match the selected checkpoint")
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "global_step", "train_loss", "train_accuracy", "validation_loss",
                         "validation_accuracy", "validation_perplexity", "learning_rate", "gradient_norm"))
        for row in selected:
            validation = row["validation"] or {}
            train = row["train"]
            writer.writerow((row["epoch"], row["global_step"], train["loss"], train["accuracy"],
                             validation.get("loss"), validation.get("accuracy"), validation.get("perplexity"),
                             train["learning_rate"], train["gradient_norm"]))


def prediction_flags(text: str) -> dict[str, bool]:
    words = text.split()
    return {
        "empty": not text.strip(),
        "repetition": any(sum(1 for _ in repeat) >= 4 for char, repeat in groupby(text) if not char.isspace())
        or any(sum(1 for _ in repeat) >= 3 for _, repeat in groupby(words)),
        "control": any(unicodedata.category(char) == "Cc" or
                       (unicodedata.category(char) == "Cf" and char not in "\u200c\u200d") for char in text),
        "replacement_character": "\ufffd" in text,
        "devanagari_letters": any(char.isalpha() and "DEVANAGARI" in unicodedata.name(char, "") for char in text),
    }


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}; choose a fresh directory")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch-size and limit must be positive")
    checkpoint = load_checkpoint(args.checkpoint)
    data = load_prepared_data(args.processed_dir, args.tokenizer_dir, DataConfig.load(args.data_config),
                              splits=(args.split,))
    translator = TransformerTranslator.from_checkpoint(args.checkpoint, args.tokenizer_dir, args.device)
    validate_checkpoint(checkpoint, asdict(translator.model.config), data.tokenizer_metadata, data.data_identity)
    pairs, _ = read_parallel_csv(args.processed_dir / f"{args.split}.csv")
    if args.limit is not None and args.limit < len(pairs) and getattr(args, "stratified", False):
        ordered = sorted(pairs, key=lambda pair: (len(pair.source), pair.source, pair.target))
        indices = [round((len(ordered) - 1) * i / max(1, args.limit - 1)) for i in range(args.limit)]
        pairs = [ordered[index] for index in indices]
    else:
        pairs = pairs[:args.limit]
    references = [pair.target for pair in pairs]
    corpus = checkpoint["data_identity"].get("corpus", {})
    official = corpus.get("splits", {}).get(args.split)
    prepared_count = len(data.datasets[args.split])
    original_count = official["rows"] if official else prepared_count
    report = {
        "split": args.split, "examples": len(pairs), "full_split": len(pairs) == prepared_count,
        "official_split_examples": original_count, "prepared_examples": prepared_count,
        "official_coverage": len(pairs) / original_count,
        "excluded_by_preparation": original_count - prepared_count,
        "omitted_by_evaluation_limit": prepared_count - len(pairs),
        "dataset_revision": corpus.get("source", {}).get("revision"),
        "preparation_stats": official, "leakage_removed": corpus.get("leakage", {}).get("removed", {}).get(args.split),
        "model_config": asdict(translator.model.config), "tokenizer_sha256": data.tokenizer_metadata["tokenizer_sha256"],
        "checkpoint_sha256": file_digest(args.checkpoint), "experiment_fingerprints": experiment_fingerprints(checkpoint),
        "device": str(translator.device), "batch_size": args.batch_size, "strategies": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staging = Path(temp) / "evaluation"
        staging.mkdir()
        for strategy in ("greedy", "beam"):
            config = InferenceConfig(strategy=strategy, max_new_tokens=args.max_new_tokens, beam_width=args.beam_width,
                                     length_penalty=args.length_penalty, device=args.device)
            hypotheses = []
            completed = 0
            flag_counts: dict[str, int] = {}
            started = time.perf_counter()
            with (staging / f"{strategy}.jsonl").open("w", encoding="utf-8") as stream:
                for start in range(0, len(pairs), args.batch_size):
                    rows = pairs[start:start + args.batch_size]
                    results = translator.generate([row.source for row in rows], config)
                    for row, result in zip(rows, results, strict=True):
                        hypotheses.append(result.text)
                        completed += result.hypothesis.reached_eos
                        flags = prediction_flags(result.text)
                        for flag, present in flags.items():
                            flag_counts[flag] = flag_counts.get(flag, 0) + int(present)
                        stream.write(json.dumps({
                            "source": row.source, "reference": row.target, "prediction": result.text,
                            "reached_eos": result.hypothesis.reached_eos,
                            "generated_tokens": len(result.hypothesis.token_ids), "flags": flags,
                        }, ensure_ascii=False) + "\n")
            seconds = time.perf_counter() - started
            report["strategies"][strategy] = {
                **score_translations(hypotheses, references), "eos_count": completed, "config": asdict(config),
                "generation_seconds": seconds, "prediction_flags": flag_counts,
            }
        if args.training_metrics is not None:
            export_curves(args.training_metrics, staging / "training_curves.csv", checkpoint)
            report["training_metrics_sha256"] = file_digest(args.training_metrics)
        (staging / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(args.output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/experiment_data.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-metrics", type=Path, help="Matching run metrics JSONL to export curves through the checkpoint epoch")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stratified", action="store_true", help="Spread a limited sample across source lengths")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    try:
        print(json.dumps(evaluate(parser.parse_args()), indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
