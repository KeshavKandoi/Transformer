# TransForge README redesign

## Presentation

The [root README](../../README.md) now leads with the model and engineering work, then presents system data flow, attention, training, frozen results, and the local application. Setup, training, and evaluation commands live in [Getting started](GETTING_STARTED.md). The [technical report](PROJECT_REPORT.md) covers the implementation and failure analysis. [Experiment results](EXPERIMENT_RESULTS.md) separates the released v2 checkpoint from the historical v1 experiment. Architecture and interview material remain linked.

The README uses four original SVG diagrams and four live Chrome captures under [assets/readme](../assets/readme). The translation capture shows the beam-width-four output **क्या तुम मेरी मदद कर सकते हो?** for “Can you help me?” The attention capture shows the corresponding cross-attention. The architecture and metrics captures were taken from the same running checkpoint. The superseded social draft was replaced by [Project launch](../social/PROJECT_LAUNCH.md).

## Verified facts and sources

| Verified fact | Source |
|---|---|
| 1,126,656 trainable parameters, epoch 11, optimizer step 1,980 | Instantiated model and checkpoint payload from the verified archive. Weights SHA-256 `628ee5b85ef534e3efc8a7a32d2a887f63df0d8eb44970989382c614a84b7aa6`. |
| Two encoder and two decoder blocks, width 128, four 32-channel heads, FFN width 384, 64 positions | Checkpoint model configuration and independently loaded model. |
| 874 BPE tokens, 8,597 training rows | Checkpoint-compatible tokenizer and frozen `pairs.jsonl`. Recreated by the preparation dry run. |
| AdamW, clipping at 1.0, inverse-square-root schedule, validation selection, original run of 16 epochs | [Training config](../configs/interview_training.yaml), checkpoint and archived `training/scratch/metrics.jsonl`. |
| Validation 1,079/1,088, composition 1,621/1,642, lexical 2,399/2,434, withheld structures 330/433 | Archived `validation_evaluation.json` and `test_evaluation.json`. The four splits were independently regenerated in this review. |
| Exact match, chrF2, BLEU and EOS counts | Per-split aggregate fields and raw row records in the same evaluation files. |
| Greedy and beam width four, generation cap 32, layer/head attention | [Local inference server](../scripts/interview_server.py) and live Chrome smoke test. |

The [archive manifest](../assets/release/README.md) records the artifact identities. Exact match is a single-reference synthetic measure. The checkpoint has known possessive and agreement errors, and unseen names, longer sequences, and grammar outside the domain remain limitations. No representative latency benchmark is asserted.

## Verification

| Check | Result |
|---|---|
| `shasum -a 256 -c assets/release/SHA256SUMS` and Python zip extraction | Passed |
| `bash scripts/run_interview_demo.sh --experiment <extracted-dir> --port 0` | Fresh local server returned the expected health hash and live beam translation. |
| `.venv/bin/python -m pip check` | Passed in the existing local environment |
| `scripts/prepare_conversation_v2.py --output <fresh-dir>` and `TRANSFORMER_EXPERIMENT=<fresh-dir> bash scripts/train_conversation_v2.sh --dry-run` | Passed preparation and one forward/backward pass |
| `scripts/evaluate_interview_model.py --experiment <extracted-dir> --output-dir <fresh-results-dir>` with `--validation-only` and `--domain-only` | Both passed and reproduced the frozen split scores |
| `.venv/bin/python -m pytest -q -rs` | 301 passed, 9 skipped because accelerator devices were unavailable |
| `DEMO_SCREENSHOTS=assets/readme node tests/interview_browser.cjs` using local Playwright and Chrome | Passed live inference, attention, gallery, clipboard, and four viewport widths |
| GFM preview rendered with Marked and Chrome | All eight assets loaded at 390 and 1,440 pixels in light and dark styles. All four SVGs rendered with text inside their view boxes. |
| Local Markdown targets, SVG XML, and `git diff --check` | Passed |

The Markdown preview used a local GFM renderer. The README was not uploaded to GitHub. Dependency installation from a fresh online environment and a full retraining run were outside this verification. The author’s GitHub profile was verified. The supplied portfolio and LinkedIn URLs were retained, and the browser lookup could not confirm them.

## Branding and launch

**Recommended GitHub description**  
Scratch-trained PyTorch encoder–decoder for English-to-Hindi translation, with reproducible synthetic evaluation and interactive attention inspection.

**Recommended topics**  
`machine-translation`, `transformer`, `pytorch`, `encoder-decoder`, `attention-mechanism`, `english-hindi`, `nlp`, `beam-search`, `reproducible-research`

**X launch draft**

> I built TransForge, a PyTorch encoder–decoder for English → Hindi, and trained it from random weights. The local demo compares greedy and beam output and shows real attention maps. The repo includes reproducible synthetic tests and failure cases.
>
> https://github.com/KeshavKandoi/Transformer

An existing [TransForge website translation tool](https://github.com/LaserLloyd/TransForge) creates a discoverability conflict. The requested name remains in the draft. **HindiSeqLab** is a more distinctive candidate before public launch. The browser retains its existing short name, “Transformer.” The GitHub remote and package paths are unchanged. No push, release, or social publication was made.
