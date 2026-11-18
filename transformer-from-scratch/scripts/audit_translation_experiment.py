"""Verify a local IITB experiment and write private JSON and Markdown reports."""

import argparse
import html
import json
import math
import platform
import sys
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from itertools import chain
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_translations import prediction_flags, score_translations
from scripts.prepare_iitb_experiment import remove_leakage, verify_source
from src.data.cleaning import read_parallel_csv
from src.data.config import ROOT, DataConfig
from src.data.prepared import file_digest
from src.data.tokenizer import BilingualBPETokenizer
from src.inference import TransformerTranslator
from src.training.checkpoint import experiment_fingerprints, load_checkpoint, validate_checkpoint
from src.training.utils import load_prepared_data


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_generation(
    directory: Path, checkpoint_path: Path, checkpoint: dict, rows: dict,
) -> tuple[dict, list[dict], list[dict]]:
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if (report["checkpoint_sha256"] != file_digest(checkpoint_path)
            or report["experiment_fingerprints"] != experiment_fingerprints(checkpoint)):
        raise ValueError("Generated metrics do not match the checkpoint")
    if report["split"] not in ("val", "test"):
        raise ValueError("Prediction report must use an official held-out split")
    allowed = {(row.source, row.target) for row in rows[report["split"]]}
    greedy, beam = (jsonl(directory / f"{strategy}.jsonl") for strategy in ("greedy", "beam"))
    if len(greedy) != report["examples"] or len(beam) != len(greedy):
        raise ValueError("Prediction count differs from metric report")
    for first, second in zip(greedy, beam, strict=True):
        key = first["source"], first["reference"]
        if key not in allowed or key != (second["source"], second["reference"]):
            raise ValueError("Prediction rows are misaligned with official held-out references")
        allowed.remove(key)
        for row in (first, second):
            if row["flags"] != prediction_flags(row["prediction"]):
                raise ValueError("A prediction's recorded quality flags do not match its text")
    for strategy, predictions in (("greedy", greedy), ("beam", beam)):
        expected = score_translations([row["prediction"] for row in predictions],
                                      [row["reference"] for row in predictions])
        recorded = report["strategies"][strategy]
        for metric, info in expected.items():
            if info["signature"] != recorded[metric]["signature"] or not math.isclose(
                info["score"], recorded[metric]["score"], rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise ValueError(f"{strategy} {metric} differs from recomputation")
        if sum(row["reached_eos"] for row in predictions) != recorded["eos_count"]:
            raise ValueError("EOS count differs from predictions")
    return report, greedy, beam


def markdown(report: dict, greedy: list[dict], beam: list[dict]) -> str:
    corpus = report["source"]
    gen = report["generation"]
    last = report["checkpoint_epoch_metrics"]
    lines = [
        "# IIT Bombay English–Hindi experiment", "",
        f"Revision `{corpus['revision']}`. Publisher: {corpus['publisher']}.", "",
        f"License declared by publisher: {corpus['license']}. {corpus['license_warning']}", "",
        f"Train, validation and test pairs retained: {report['split_sizes']}.",
        f"Vocabulary: {report['vocab_size']}. Trainable parameters: {report['trainable_parameters']:,}.",
        f"Checkpoint epoch {report['checkpoint_epoch']}, optimizer step {report['global_step']}.",
        f"Training loss {last['train']['loss']:.6f}. Validation loss {last['validation']['loss']:.6f}.",
        f"Validation teacher-forced accuracy {last['validation']['accuracy']:.2%}.",
        f"Generation scored {gen['examples']} of {gen['official_split_examples']} official {gen['split']} rows "
        f"({gen['official_coverage']:.2%}).", "",
        "| Strategy | chrF2 | BLEU | EOS | Generation seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy, results in gen["strategies"].items():
        lines.append(f"| {strategy} | {results['chrf']['score']:.4f} | {results['bleu']['score']:.4f} | "
                     f"{results['eos_count']}/{gen['examples']} | {results['generation_seconds']:.3f} |")
    lines += ["", "## Length-stratified examples", "",
              "| English | Reference Hindi | Greedy | Beam | Greedy EOS / tokens | Beam EOS / tokens |",
              "| --- | --- | --- | --- | --- | --- |"]

    def cell(value: str) -> str:
        value = json.dumps(value, ensure_ascii=False)[1:-1]
        return html.escape(value).replace("|", "&#124;") or "*(empty)*"

    order = sorted(range(len(greedy)), key=lambda index: (len(greedy[index]["source"]), index))
    selected = sorted({round(i * (len(order) - 1) / 19) for i in range(20)})
    for index in (order[i] for i in selected):
        first, second = greedy[index], beam[index]
        text = [cell(first["source"]), cell(first["reference"]), cell(first["prediction"]), cell(second["prediction"])]
        text.extend(f"{row['reached_eos']} / {row['generated_tokens']}" for row in (first, second))
        lines.append("| " + " | ".join(text) + " |")
    lines += ["", "Examples and model artifacts are for private local research. Automated alignment and near-duplicate checks do not establish semantic equivalence.", ""]
    return "\n".join(lines)


def audit(args: argparse.Namespace) -> dict:
    source = verify_source(args.source_dir)
    config = DataConfig.load(args.data_config)
    processed, tokenizer_dir = args.experiment / "processed", args.experiment / "tokenizer"
    data = load_prepared_data(processed, tokenizer_dir, config)
    corpus = data.data_identity["corpus"]
    expected = {key: value for key, value in source.items() if key not in ("retrieved_at_utc", "requested_revision")}
    if corpus["source"] != expected:
        raise ValueError("Prepared provenance does not match the acquired source")
    rows = {name: read_parallel_csv(processed / f"{name}.csv")[0] for name in ("train", "val", "test")}
    sizes = {name: len(pairs) for name, pairs in rows.items()}
    if any(sizes[name] != stats["kept"] for name, stats in corpus["splits"].items()):
        raise ValueError("Prepared split counts do not match provenance")
    _, leakage = remove_leakage(rows, corpus["policy"]["near_threshold"])
    if leakage["cross_split_groups"]:
        raise ValueError("Duplicate groups cross prepared splits")
    if corpus["bpe_training_split"] != "train" or corpus["bpe_training_csv_sha256"] != file_digest(processed / "train.csv"):
        raise ValueError("Tokenizer fit provenance does not match training rows")
    rebuilt = BilingualBPETokenizer.train(
        chain.from_iterable((row.source, row.target) for row in rows["train"]), config.vocab_size,
    )
    with tempfile.TemporaryDirectory() as temp:
        rebuilt.save(Path(temp))
        if file_digest(Path(temp) / "tokenizer.json") != data.tokenizer_metadata["tokenizer_sha256"]:
            raise ValueError("Tokenizer does not reproduce from the prepared train split alone")
    checkpoint = load_checkpoint(args.checkpoint)
    translator = TransformerTranslator.from_checkpoint(args.checkpoint, tokenizer_dir, "cpu")
    validate_checkpoint(checkpoint, asdict(translator.model.config), data.tokenizer_metadata, data.data_identity)
    if (checkpoint["global_step"] < 1
            or checkpoint["scheduler_state_dict"]["last_epoch"] != checkpoint["global_step"]
            or not checkpoint["optimizer_state_dict"]["state"]):
        raise ValueError("Checkpoint does not contain compatible optimizer and scheduler updates")
    history = jsonl(args.run_dir / "metrics.jsonl")
    selected = next((row for row in history if row["epoch"] == checkpoint["epoch"]), None)
    if (selected is None or not selected["validation"] or selected["global_step"] != checkpoint["global_step"]
            or selected["fingerprints"] != experiment_fingerprints(checkpoint)):
        raise ValueError("Validation record does not match the checkpoint")
    gen, greedy, beam = verify_generation(args.generation, args.checkpoint, checkpoint, rows)
    report = {
        "source": source, "preparation": corpus, "split_sizes": sizes,
        "remaining_cross_split_groups": leakage["cross_split_groups"],
        "tokenizer_reproduced_from_train": True, "vocab_size": data.tokenizer.vocab_size,
        "tokenizer_metadata": data.tokenizer_metadata,
        "checkpoint_sha256": file_digest(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"], "checkpoint_epoch_metrics": selected,
        "trainable_parameters": sum(p.numel() for p in translator.model.parameters() if p.requires_grad),
        "model_config": checkpoint["model_config"], "training_config": checkpoint["training_config"],
        "fingerprints": experiment_fingerprints(checkpoint), "checkpoint_runtime": checkpoint.get("runtime"),
        "audit_environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{name: version(name) for name in ("pyarrow", "huggingface-hub", "sacrebleu", "tokenizers", "torch")}},
        "generation": gen,
    }
    (args.run_dir / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.run_dir / "experiment_report.md").write_text(markdown(report, greedy, beam), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/experiment_data.yaml")
    args = parser.parse_args()
    try:
        audit(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    print(f"Audit passed. Reports saved in {args.run_dir}.")


if __name__ == "__main__":
    main()
