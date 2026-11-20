# Transformer From Scratch

A PyTorch encoder–decoder trained from random initialization for English-to-Hindi translation. The local demo runs the frozen checkpoint, compares greedy and beam decoding, and displays encoder, decoder, and cross-attention weights by layer and head.

The model was trained on synthetic conversational patterns. Its evaluation measures performance within that domain and on deliberately withheld patterns.

## Quick start

Requires Python 3.11 or later, Bash, and a CPU supported by PyTorch. The web interface uses plain HTML, CSS, and JavaScript.

From the repository root:

```bash
cd transformer-from-scratch
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
shasum -a 256 -c assets/release/SHA256SUMS
.venv/bin/python -m zipfile -e assets/release/conversation-v2.zip experiments/conversation-v2-release
bash scripts/run_interview_demo.sh --experiment experiments/conversation-v2-release --port 8765
```

Open [localhost:8765](http://127.0.0.1:8765). Enter **Can you help me?** and select **Compare both** to see how decoding changes the output. Open **Attention Visualizer** and select cross-attention to inspect how Hindi queries attend to English keys. Use another `--port` if 8765 is occupied.

## Model and data

| Component | Released checkpoint |
|---|---|
| Trainable parameters | 1,126,656 |
| Encoder and decoder | Two blocks each with post-layer normalization |
| Model width / feed-forward width | 128 / 384 |
| Attention | Four heads, width 32 per head |
| Embeddings and positions | Separate source and target embeddings with fixed sinusoidal positions |
| Vocabulary / context | 874 shared byte-level BPE tokens / 64 positions |
| Checkpoint | Epoch 11, optimizer step 1,980 |

The encoder reads the complete English sentence. The decoder reads the Hindi prefix through causal self-attention and the encoded English through cross-attention. Training uses shifted Hindi references, cross-entropy over non-padding labels, and AdamW with warmup and inverse-square-root decay. Generation feeds back predicted tokens until EOS or the length limit.

The corpus generator produced 8,597 training pairs. Normalized source strings, target strings, and semantic groups are disjoint across splits. The composition and lexical splits reuse training construction families, while the structure split reserves entire families. The [technical report](transformer-from-scratch/docs/PROJECT_REPORT.md) describes the data and training decisions.

## Evaluation

These greedy results were reproduced from the frozen checkpoint. Exact match compares each entire output with one synthetic reference. Validation selected the checkpoint.

| Split | Exact match | chrF2 | BLEU | EOS |
|---|---:|---:|---:|---:|
| In-domain validation | 1,079/1,088 (99.17%) | 99.81 | 99.61 | 1,088/1,088 |
| Held-out composition | 1,621/1,642 (98.72%) | 99.68 | 99.31 | 1,639/1,642 |
| Held-out lexical combinations | 2,399/2,434 (98.56%) | 99.92 | 99.24 | 2,434/2,434 |
| Withheld structures | 330/433 (76.21%) | 87.44 | 83.24 | 428/433 |

| Input | Decoder | Model output |
|---|---|---|
| My name is Ravi. | Greedy | मेरा नाम रवि है। |
| Can you help me? | Beam 4 | क्या तुम मेरी मदद कर सकते हो? |
| Can you help me? | Greedy | क्या तुम मुझे फी मदद कर सकते हो? |
| What is your name? | Greedy | उनका नाम क्या है? |

Beam search produces a grammatical help request where greedy decoding fails. Both methods change “your” to “their” in the name question. The dataset uses fixed Hindi gender and address conventions. The demo rejects English outside its recorded grammar. Small historical probes on IIT Bombay and FLORES yielded zero exact matches across 24 examples from each dataset.

## Reproduce

Run from `transformer-from-scratch` after completing Quick start. Regenerate the synthetic scores without writing into the release directory:

```bash
release=experiments/conversation-v2-release
.venv/bin/python scripts/evaluate_interview_model.py --experiment "$release" --output-dir runs/release-recheck --validation-only
.venv/bin/python scripts/evaluate_interview_model.py --experiment "$release" --output-dir runs/release-recheck --domain-only
.venv/bin/python -m pytest -q -rs
```

Prepare a fresh training directory and verify a forward and backward pass:

```bash
repro=experiments/conversation-v2-repro
.venv/bin/python scripts/prepare_conversation_v2.py --output "$repro"
TRANSFORMER_EXPERIMENT="$repro" bash scripts/train_conversation_v2.sh --dry-run
```

Remove `--dry-run` to train in that directory. Use a fresh path for each new preparation run. The original training run stopped after 16 epochs and 2,880 updates.

## Repository guide

| Path | Contents |
|---|---|
| `src/model/` | Attention, embeddings, encoder, decoder, and vocabulary projection |
| `src/data/` | Corpus preparation, bilingual BPE, batching, and masks |
| `src/training/` | Loss, optimizer, scheduler, and resumable checkpoints |
| `src/inference/` | Checkpoint loading, greedy decoding, and beam search |
| `interview_app/` | Local translation interface and attention visualizer |
| `scripts/` and `tests/` | Reproduction commands and regression tests |

[Architecture](transformer-from-scratch/docs/ARCHITECTURE.md) · [Technical report](transformer-from-scratch/docs/PROJECT_REPORT.md) · [Release verification](transformer-from-scratch/docs/FINAL_RELEASE_REPORT.md) · [Interview guide](transformer-from-scratch/docs/INTERVIEW_GUIDE.md) · [Implementation history](transformer-from-scratch/docs/IMPLEMENTATION_HISTORY.md)

## Provenance and license

The release archive includes the checkpoint, tokenizer, generated conversational pairs, and synthetic evaluations. It excludes IIT Bombay and FLORES data. Archive fingerprints and provenance are recorded in the [release manifest](transformer-from-scratch/assets/release/README.md). CFILT lists CC BY-NC 4.0 for its IIT Bombay corpus and notes distinct terms for compiled sources. [CFILT corpus page](https://www.cfilt.iitb.ac.in/iitb_parallel/) · [Historical corpus workflow](transformer-from-scratch/docs/CORPUS_WORKFLOW.md)

The repository has no license file. AI assistance was used in development and release preparation. [Keshav Kandoi](https://github.com/KeshavKandoi)

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [PyTorch](https://docs.pytorch.org/docs/stable/index.html) · [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index) · [SacreBLEU](https://github.com/mjpost/sacrebleu)
