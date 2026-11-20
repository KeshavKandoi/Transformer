# Final release report

**READY WITH DOCUMENTED LIMITATIONS**

The local demo runs the original scratch-trained conversation v2 checkpoint from a verified synthetic-only archive. The repository remains unpushed and no deployment or social publication was performed. The repository has no license file, so this release makes no open-source licensing claim.

## Website and API

The browser interface now has five navigable sections for translation, architecture, attention, metrics, and project context. The translation workspace uses English input and Hindi output labels, with keyboard submission, clear and copy actions, live generation time, and visible loading and error states. Beam width four is the default. The gallery remains an explicitly labeled greedy validation gallery with 11 exact matches and nine mismatches. Selecting an example performs fresh inference.

The architecture section reads its dimensions from the actual checkpoint. The attention viewer plots diagnostic tensors from the selected model prefix, with encoder, decoder, cross-attention, padding and causal masks, plus layer and head selection. The metrics section uses saved evaluations and training logs for exact match, chrF2, BLEU, EOS, loss, learning rate, validation token accuracy, and surface error counts. Missing optional fields have unavailable states.

The HTTP server validates checkpoint, tokenizer, inventory and evaluation identity, returns structured errors, and exposes `/api/health`. It still binds only to loopback. A missing checkpoint produced a startup message and exit status 1.

## Verified translation behavior

| Input | Mode | Live result |
|---|---|---|
| `can  you help me` with extra surrounding whitespace | Beam 4 | क्या तुम मेरी मदद कर सकते हो? |
| `Can you help me?` | Greedy | क्या तुम मुझे फी मदद कर सकते हो? |
| `MY NAME IS RAVI!` | Beam 4 | मेरा नाम रवि है। |
| `I have two books.` | Beam 4 | मेरे पास दो किताबें हैं। |
| `What is your name?` | Greedy and beam | उनका नाम क्या है? |
| `Why is the sky blue?` | Input gate | HTTP 422, unsupported grammar |
| `kaise ho` | Input gate | HTTP 422, Romanized Hindi |
| `नमस्ते` | Input gate | HTTP 422, Hindi input |

The erroneous name question changes “your” to “their.” No reference is substituted for either mistake.

## Model and independent evaluation

The running checkpoint is `experiments/conversation-v2-release/checkpoints/scratch/best.pt`. It is the original best checkpoint from `experiments/interview-conversation-v2`, at epoch 11 and optimizer step 1,980. It has 1,126,656 trainable parameters, two encoder blocks, two decoder blocks, four 32-wide heads, width 128, FFN width 384, and a 64-position window. Its exact 874-token BPE and the artifact hashes are listed in the [release archive manifest](../assets/release/README.md).

Independent greedy generation reproduced every saved row and aggregate field across four synthetic splits:

| Split | Exact | chrF2 | BLEU | EOS |
|---|---:|---:|---:|---:|
| In-domain validation | 1,079/1,088 (99.17%) | 99.81 | 99.61 | 1,088/1,088 |
| Held-out composition | 1,621/1,642 (98.72%) | 99.68 | 99.31 | 1,639/1,642 |
| Lexical combinations | 2,399/2,434 (98.56%) | 99.92 | 99.24 | 2,434/2,434 |
| Withheld structures | 330/433 (76.21%) | 87.44 | 83.24 | 428/433 |

The model components and prior experiments were preserved. The release archive contains synthetic pairs and evaluations only. Prior private IITB and FLORES records and checkpoints are outside the archive.

## Documentation and media

- The [repository README](../../README.md) covers setup, training, architecture, evaluation, examples and data provenance.
- The [technical report](PROJECT_REPORT.md) cover the model, training, IITB audit, evaluation, serving and limitations.
- The [interview guide](INTERVIEW_GUIDE.md) includes three timed walkthroughs and 20 questions. The [implementation history](IMPLEMENTATION_HISTORY.md) and [corpus workflow](CORPUS_WORKFLOW.md) retain historical instructions and attribution.
- The [launch copy](../social/PROJECT_LAUNCH.md) contains two X posts, a six-post thread, GitHub and portfolio descriptions, and a LinkedIn draft.

## Verification

Run from `transformer-from-scratch`. The initial working tree was clean. The baseline `.venv/bin/python -m pytest -q` and final `.venv/bin/python -m pytest -q -rs` both returned **301 passed, 9 skipped**. All nine skips were accelerator-only tests. The updated checkpoint test extracts the actual release archive into a fresh directory, validates compatibility and tests attention dimensions, probability sums and blocked future keys.

| Check | Command or procedure | Result |
|---|---|---|
| Python dependencies | `.venv/bin/python -m pip install --no-index -r requirements.txt` and `.venv/bin/python -m pip check` | Exit 0 using the existing environment. Fresh online dependency installation was outside this local check. |
| JS syntax | `node --check interview_app/app.js` and `node --check tests/interview_browser.cjs` | Exit 0 |
| Archive and launch | `shasum -a 256 -c assets/release/SHA256SUMS`, extract and run the launcher below | Checksum passed and health returned HTTP 200 with the expected checkpoint hash |
| API | Browser smoke and HTTP health/prediction probes | Valid 200, 400, 404 and 422 paths |
| Browser | `node tests/interview_browser.cjs` with Playwright and local Chrome | Live predictions, gallery replay, clipboard, shortcuts, all navigation, cross-attention and causal mask, loading and HTTP errors, missing metadata, and four viewport widths verified |
| Evaluation | Both commands in the [repository README](../../README.md#evaluation) | Exit 0, all four synthetic splits reproduced |
| Training entry point | Prepare a fresh directory and run the documented `--dry-run` | Exit 0, forward/backward completed, loss 6.871485, 833 target tokens |
| Whitespace | `git diff --check` | Exit 0 |

There is no frontend build step. The browser slow/error and unusually long Hindi cases used controlled response fixtures. Translation and gallery predictions used the actual local server. No new full training run was started.

## Launch

From the Git repository root, with Python 3.11 or later:

```bash
cd transformer-from-scratch
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
shasum -a 256 -c assets/release/SHA256SUMS
.venv/bin/python -m zipfile -e assets/release/conversation-v2.zip experiments/conversation-v2-release
bash scripts/run_interview_demo.sh --experiment experiments/conversation-v2-release --port 8765
```

Open [localhost on port 8765](http://127.0.0.1:8765). For the optional browser test, install Playwright separately and run Chromium, or set `PLAYWRIGHT_PACKAGE` to a local Playwright installation and `CHROME_PATH` to the Chrome executable.

## Limitations

The finite English grammar, synthetic labels, fixed Hindi address and gender conventions, and observed possessive and agreement errors limit public claims. The site is intended for a local interview demonstration. Browser verification used Chromium. The checked dependency lock records the local macOS arm64 environment, and the general requirements remain available for other platforms.

## Files changed

The release changes the repository README, project ignore rules, the three files in `interview_app/`, the demo server, the conversational training launcher, and the evaluation CLI. It adds a platform-specific dependency snapshot, a browser smoke test, the archive and manifest under `assets/release/`, and the launch drafts under `social/`. It extends the checkpoint integration test and interview guide and adds the technical report, corpus workflow, implementation history and this release report.
