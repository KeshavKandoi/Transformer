# Run TransForge locally

Use Python 3.11 or later, Bash, and a PyTorch-compatible CPU. The browser app uses plain HTML, CSS, and JavaScript. The archive holds the original checkpoint, tokenizer, synthetic inventory, evaluation records, and training log. See the [artifact manifest](../assets/release/README.md).

## Start the demo

From the cloned repository root:

```bash
cd transformer-from-scratch
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
shasum -a 256 -c assets/release/SHA256SUMS
release=experiments/conversation-v2-release
.venv/bin/python -m zipfile -e assets/release/conversation-v2.zip "$release"
bash scripts/run_interview_demo.sh --experiment "$release" --port 8765
```

Open [localhost:8765](http://127.0.0.1:8765). Try **Can you help me?**, choose **Compare both**, and inspect cross-attention. Use another `--port` if 8765 is occupied. From a second terminal, `curl -fsS http://127.0.0.1:8765/api/health` confirms the loaded checkpoint hash.

## Reproduce evaluation

Run from `transformer-from-scratch` after extraction:

```bash
release=experiments/conversation-v2-release
.venv/bin/python scripts/evaluate_interview_model.py --experiment "$release" --output-dir runs/release-recheck --validation-only
.venv/bin/python scripts/evaluate_interview_model.py --experiment "$release" --output-dir runs/release-recheck --domain-only
.venv/bin/python -m pytest -q -rs
```

## Prepare and train

Use a fresh directory. The generator creates the split inventory, fits the shared BPE tokenizer on training rows, and writes the preparation manifest.

```bash
repro=experiments/conversation-v2-repro
.venv/bin/python scripts/prepare_conversation_v2.py --output "$repro"
TRANSFORMER_EXPERIMENT="$repro" bash scripts/train_conversation_v2.sh --dry-run
```

Remove `--dry-run` for the full configured run. The launcher reads the checked-in [training settings](../configs/interview_training.yaml). The original artifact is selected by validation loss and lives separately from fresh runs.
