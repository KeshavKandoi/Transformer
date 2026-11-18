# English–Hindi Transformer from scratch

This repository implements a custom PyTorch encoder–decoder Transformer, from bilingual BPE and teacher forcing through training, checkpoint resume and autoregressive inference. The implementation and phase-by-phase documentation are in [transformer-from-scratch](transformer-from-scratch/README.md).

## Reproduce an experiment

Use Python 3.11 or newer. The CPU GitHub Actions workflow installs the CPU build of PyTorch and runs the full offline pytest suite.

```bash
cd transformer-from-scratch
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q -rs
```

Choose a **local, licensed** UTF-8 CSV with one aligned English source and Hindi target per row under the headers `source,target`. Consult the license for training, derivative checkpoints and redistribution of predictions before using a corpus. Record the publisher, release and acquisition URL or local source. The importer records the supplied terms and provenance, the raw SHA-256, the cleaning policy and the artifact fingerprints. It makes no dataset network requests. Keep third-party datasets, checkpoints and evaluation examples private unless redistribution is permitted.

```bash
run_name=en-hi-real-v1
corpus="experiments/$run_name"

.venv/bin/python scripts/import_corpus.py \
  --input /path/to/parallel.csv --output "$corpus" \
  --name 'Corpus name and release' --license 'License identifier or URL' \
  --provenance 'Publisher, release and acquisition URL or local record'

.venv/bin/python scripts/train.py --config configs/experiment_training.yaml \
  --data-config configs/experiment_data.yaml \
  --processed-dir "$corpus/processed" --tokenizer-dir "$corpus/tokenizer" \
  --run-name "$run_name" --device cpu --dry-run
```

Review split sizes, rejected rows, largest duplicate cluster and tokenized lengths in `$corpus/processed/manifest.json`. Inspect paired samples for translation alignment and language quality. Then select a device and epoch budget before running training. CUDA and MPS are available when PyTorch detects them.

```bash
.venv/bin/python scripts/train.py --config configs/experiment_training.yaml \
  --data-config configs/experiment_data.yaml \
  --processed-dir "$corpus/processed" --tokenizer-dir "$corpus/tokenizer" \
  --run-name "$run_name" --device cpu --epochs 20

.venv/bin/python scripts/evaluate_translations.py \
  --checkpoint "checkpoints/$run_name/best.pt" \
  --processed-dir "$corpus/processed" --tokenizer-dir "$corpus/tokenizer" \
  --split test --output "runs/$run_name/generated-test" \
  --training-metrics "runs/$run_name/metrics.jsonl" --device cpu

.venv/bin/python -m pip freeze > "runs/$run_name/environment.txt"
```

Choose the checkpoint and decoding settings on validation data before final test evaluation. For a quick pipeline check, use `scripts/import_corpus.py --demo --output experiments/en-hi-corpus-smoke-v1` with a fresh run name and a small CPU budget. The bundled pairs are smoke data. Translation quality requires a separate licensed corpus and an explicitly chosen training budget.

## IIT Bombay fixed-split experiment

For local noncommercial research, follow the [pinned IIT Bombay workflow](transformer-from-scratch/README.md#iit-bombay-englishhindi-fixed-split-experiment). It downloads published Parquet files explicitly, preserves official split membership, and keeps downloaded text, checkpoints, and private reports under ignored paths. CFILT declares CC BY-NC 4.0 and states that compiled sources have their own licenses. [CFILT publisher page](https://www.cfilt.iitb.ac.in/iitb_parallel/) · [Pinned Hugging Face revision](https://huggingface.co/datasets/cfilt/iitb-english-hindi/tree/e104eb47979d570d3df7af4912da85f83efde6fc)
