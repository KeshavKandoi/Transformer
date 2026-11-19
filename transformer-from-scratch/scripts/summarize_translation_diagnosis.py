"""Compare aligned generation reports without changing their predictions."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_translations import score_translations
from src.data.tokenizer import BilingualBPETokenizer


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compare(root: Path, details: Path, output: Path, tokenizer: BilingualBPETokenizer) -> dict:
    all_metrics, predictions, summary = {}, {}, []
    reference_pairs = None
    for limit in (64, 120):
        directory = root / f"length{limit}"
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        all_metrics[limit] = metrics
        for strategy in ("greedy", "beam"):
            rows = records(directory / f"{strategy}.jsonl")
            pairs = [(row["source"], row["reference"]) for row in rows]
            if len(pairs) != metrics["examples"] or not pairs:
                raise ValueError("Prediction file and metric row counts differ")
            if reference_pairs is None:
                reference_pairs = pairs
            elif pairs != reference_pairs:
                raise ValueError("Length comparisons require identical ordered source/reference pairs")
            predictions[limit, strategy] = rows
            stats = metrics["strategies"][strategy]
            checked = score_translations([row["prediction"] for row in rows], [row["reference"] for row in rows])
            for metric, value in checked.items():
                if (stats[metric]["signature"] != value["signature"]
                        or not math.isclose(stats[metric]["score"], value["score"], rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError(f"{strategy} {limit} {metric} does not match the saved predictions")
            if stats["eos_count"] != sum(row["reached_eos"] for row in rows):
                raise ValueError(f"{strategy} {limit} EOS count does not match the saved predictions")
            summary.append({
                "max_new_tokens": limit, "strategy": strategy, "examples": len(rows),
                "chrf2": stats["chrf"]["score"], "bleu": stats["bleu"]["score"],
                "eos": sum(row["reached_eos"] for row in rows),
                "mean_generated_tokens": mean(row["generated_tokens"] for row in rows),
                "mean_reference_tokens_with_eos": mean(len(tokenizer.encode(row["reference"])) + 1 for row in rows),
                "capped_without_eos": sum(not row["reached_eos"] and row["generated_tokens"] == limit for row in rows),
                "long_over_64_without_eos": sum(not row["reached_eos"] and row["generated_tokens"] > 64 for row in rows),
                **{key: stats["prediction_flags"].get(key, 0)
                   for key in ("repetition", "empty", "replacement_character", "control")},
                "seconds": stats["generation_seconds"],
            })
    for strategy in ("greedy", "beam"):
        settings = [{key: value for key, value in all_metrics[limit]["strategies"][strategy]["config"].items()
                     if key != "max_new_tokens"} for limit in (64, 120)]
        if settings[0] != settings[1]:
            raise ValueError("Decoding settings other than the length cap must match")
    if all_metrics[64]["checkpoint_sha256"] != all_metrics[120]["checkpoint_sha256"]:
        raise ValueError("Length comparisons require the same checkpoint")
    with (output / "length_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    with (output / "validation_examples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "reference", "greedy64", "beam64", "greedy120", "beam120",
                         "greedy64_eos", "beam64_eos", "greedy120_eos", "beam120_eos"])
        for index, (source, reference) in enumerate(reference_pairs):
            rows = [predictions[key][index] for key in ((64, "greedy"), (64, "beam"), (120, "greedy"), (120, "beam"))]
            writer.writerow([source, reference, *(row["prediction"] for row in rows), *(row["reached_eos"] for row in rows)])

    detail = {}
    traces = records(details / "generation_trace.jsonl")
    for strategy in ("greedy", "beam"):
        trace = [item for item in traces if item["strategy"] == strategy]
        common = Counter(item["selected_id"] for item in trace).most_common(10)
        detail[strategy] = {
            "most_common_ids_in_five_traces": [
                {"id": token, "piece": tokenizer.id_to_token(token), "count": count} for token, count in common
            ],
            "mean_entropy_nats": mean(row["entropy_nats"] for row in trace),
            "eos_probabilities": [
                {"source": name, "first": path[0]["eos_probability"], "last": path[-1]["eos_probability"],
                 "max": max(row["eos_probability"] for row in path)}
                for name in dict.fromkeys(row["source"] for row in trace)
                if (path := [row for row in trace if row["source"] == name])
            ],
        }
    result = {"comparison": summary, "raw_metrics": all_metrics, "trace_summary": detail}
    (output / "quality_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "details", "output", "tokenizer-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    compare(args.comparison, args.details, args.output, BilingualBPETokenizer.load(args.tokenizer_dir))


if __name__ == "__main__":
    main()
