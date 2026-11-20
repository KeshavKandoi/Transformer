# TransForge

**Neural Machine Translation Engine**

A custom encoder–decoder Transformer for English-to-Hindi neural machine translation, implemented in PyTorch and trained from random initialization. The repository connects the model implementation to a reproducible training run, frozen evaluations, and a browser interface for inspecting live predictions and attention tensors.

![English-to-Hindi inference through the encoder and decoder](transformer-from-scratch/assets/readme/hero.svg)

## Project overview

Translation models the probability of a Hindi sequence conditioned on an English sentence. TransForge encodes the whole source, then generates one target token at a time using the encoded source and the Hindi prefix. Implementing the architecture directly made it possible to test where source information enters the decoder, verify masking and generation, and separate model errors from data and evaluation problems.

## Key engineering highlights

- Custom scaled dot-product and multi-head attention with causal and padding masks, plus tests for gradients and blocked keys.
- Encoder and decoder stacks with separate embeddings, sinusoidal positions, cross-attention, and a vocabulary projection.
- Teacher-forced training with resumable optimizer-boundary checkpoints and tokenizer and dataset fingerprint checks.
- Greedy and beam decoding through the local CPU server, with layer-by-layer attention, evaluation examples, and training curves.

## System architecture

![English encoding, Hindi decoding, checkpoint evaluation, and local serving](transformer-from-scratch/assets/readme/architecture.svg)

The corpus generator freezes data splits and fits BPE on the training split. Training selects a checkpoint by validation loss. Offline evaluation and local serving use that checkpoint and its tokenizer. The browser calls the Python inference server and renders its predictions and diagnostic tensors.

## Inside the Transformer

![Encoder self-attention, causal decoder self-attention, and cross-attention](transformer-from-scratch/assets/readme/attention-overview.svg)

For each head, **Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V**. The implementation masks blocked scores before softmax and gives fully masked rows zero attention. Each of the four heads projects into 32 channels. Head outputs are concatenated and projected back to width 128.

| Attention | Queries | Keys and values | Visible keys |
|---|---|---|---|
| Encoder self-attention | English | English | Non-padding source positions |
| Decoder self-attention | Hindi prefix | Hindi prefix | Non-padding positions up to the current query |
| Cross-attention | Hindi prefix | Encoder states | Non-padding source positions |

Cross-attention weights have shape `[batch, heads, target_length, source_length]`. Fixed sine and cosine features give token embeddings position information. Each block applies residual addition and post-layer normalization around its attention and position-wise ReLU feed-forward sublayers. At inference, the decoder starts with BOS and feeds its predictions back until EOS or the length cap. [Architecture and tensor flow](transformer-from-scratch/docs/ARCHITECTURE.md)

## Model specifications

| Component | Released checkpoint |
|---|---|
| Architecture | Encoder–decoder Transformer, post-norm |
| Trainable parameters | 1,126,656 |
| Encoder / decoder blocks | 2 / 2 |
| Attention | 4 heads per layer, 32 channels per head |
| Hidden / feed-forward width | 128 / 384 |
| Tokenizer | Shared byte-level BPE, 874 tokens |
| Maximum sequence length | 64 positions |
| Initialization | Random weights |
| Released checkpoint | Epoch 11, optimizer step 1,980 |
| Demo decoding | Greedy and beam search with width 4 |

## Training methodology

![Synthetic pairs, training-only tokenization, teacher forcing, and checkpoint selection](transformer-from-scratch/assets/readme/training-pipeline.svg)

The generator produced 8,597 training pairs from a controlled conversational grammar. The split policy keeps normalized source and target strings and semantic groups disjoint. Composition and lexical test sets recombine familiar constructions. The structure test set withholds entire question families.

Teacher forcing pairs the shifted Hindi prefix with next-token labels. Padding labels are excluded from cross-entropy. Backpropagation uses AdamW, gradient clipping at 1.0, and warmup followed by inverse-square-root decay. The run stopped after 16 epochs, with epoch 11 selected by validation loss. [Training configuration](transformer-from-scratch/configs/interview_training.yaml) · [Technical report](transformer-from-scratch/docs/PROJECT_REPORT.md)

## Evaluation and results

Greedy evaluation from the frozen checkpoint. Exact match compares the whole prediction against one synthetic reference. chrF2 measures character n-gram overlap, BLEU measures token n-gram overlap with a brevity penalty, and EOS counts completed generations.

| Synthetic split | Exact match | chrF2 | BLEU | EOS |
|---|---:|---:|---:|---:|
| In-domain validation | 1,079/1,088 (99.17%) | 99.81 | 99.61 | 1,088/1,088 |
| Held-out composition | 1,621/1,642 (98.72%) | 99.68 | 99.31 | 1,639/1,642 |
| Held-out lexical combinations | 2,399/2,434 (98.56%) | 99.92 | 99.24 | 2,434/2,434 |
| Withheld structures | 330/433 (76.21%) | 87.44 | 83.24 | 428/433 |

The composition score measures recombination within the generated domain. The structure split shows a drop when entire patterns are withheld. [Evaluation history and metric sources](transformer-from-scratch/docs/EXPERIMENT_RESULTS.md)

## Application showcase

The translation capture shows the model’s beam-width-4 output for “Can you help me?” The attention capture uses the same prediction.

| Translation | Cross-attention |
|---|---|
| ![English request and its generated Hindi translation](transformer-from-scratch/assets/readme/translation.png) | ![Actual cross-attention heatmap for the generated Hindi prefix](transformer-from-scratch/assets/readme/attention.png) |

| Architecture | Training and metrics |
|---|---|
| ![Architecture view with checkpoint dimensions](transformer-from-scratch/assets/readme/architecture.png) | ![Recorded evaluation scores and training curves](transformer-from-scratch/assets/readme/metrics.png) |

## Engineering challenges

- **Alignment and tokenization.** Earlier IIT Bombay audits found source–target mismatches. The controlled generator isolates specific grammar and agreement cases. Byte-level BPE preserves UTF-8 round trips, while Hindi still consumes more subword pieces than English in the recorded tokenizer sample.
- **Source conditioning.** Shuffling English increased teacher-forced loss in an earlier IIT Bombay experiment, yet generated text still lost names and numbers. The released benchmark distinguishes source-conditioned likelihood from free-running translation quality.
- **Masking and training stability.** Mask and gradient tests check future-token isolation and padding. Clipping, warmup, and validated optimizer-boundary resume cover the training path.
- **Generation and generalization.** Beam search fixes the malformed greedy help request, but both methods turn “your” into “their” in a name question. Separate composition, lexical, and structure splits reveal failures that a pooled score would obscure. [Error analysis](transformer-from-scratch/docs/PROJECT_REPORT.md)

## Interactive features

Translate supported English, compare decoders, and inspect per-request generation time. Select an attention type, layer, or head, then inspect the model’s token weights or masks. The interface presents checkpoint dimensions, loss and optimization curves, and a synthetic validation gallery with successes and errors. Each gallery card reruns the actual model.

## Reproducibility

The release archive contains the checkpoint, tokenizer, synthetic dataset, evaluation outputs, and training log. The project runs on CPU with Python 3.11 or later. Follow the setup guide for dependency installation, archive verification, local serving, tokenizer fitting, training, and evaluation. [Getting started](transformer-from-scratch/docs/GETTING_STARTED.md) · [Artifact manifest](transformer-from-scratch/assets/release/README.md)

## Repository structure

```text
transformer-from-scratch/
├── src/
│   ├── model/       # Attention, encoder, decoder, and projection
│   ├── data/        # Corpus preparation, tokenization, and masks
│   ├── training/    # Optimization, metrics, and checkpoint resume
│   └── inference/   # Checkpoint loading and decoding
├── scripts/         # Preparation, training, evaluation, and serving
├── interview_app/   # Local web interface
├── assets/         # Frozen release and README visuals
├── tests/          # Python regression and browser smoke tests
└── docs/           # Architecture, reports, and setup
```

## Technical documentation

- [Architecture and attention](transformer-from-scratch/docs/ARCHITECTURE.md)
- [Technical project report](transformer-from-scratch/docs/PROJECT_REPORT.md)
- [Experiment results](transformer-from-scratch/docs/EXPERIMENT_RESULTS.md)
- [Interview guide](transformer-from-scratch/docs/INTERVIEW_GUIDE.md)
- [Getting started](transformer-from-scratch/docs/GETTING_STARTED.md)

## Research references

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html) and [Hugging Face Tokenizers byte-level pre-tokenization](https://huggingface.co/docs/tokenizers/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel).
- [SacreBLEU](https://github.com/mjpost/sacrebleu) and [IIT Bombay English–Hindi corpus](https://www.cfilt.iitb.ac.in/iitb_parallel/), used for historical experiments.

## Limitations and future work

The demo accepts a finite set of conversational patterns. Unseen names and constructions can be rejected or mistranslated. Longer inputs face the 64-position context and the demo’s 32-token generation cap. Fixed Hindi gender and address conventions, agreement errors, and occasional repetition limit coverage. The next evaluation should use independently reviewed paraphrases, name and role contrasts, and explicit Hindi agreement and politeness labels.

## Author and links

**Keshav Kandoi** · [GitHub](https://github.com/KeshavKandoi) · [Portfolio](https://keshavkandoi.me) · [LinkedIn](https://www.linkedin.com/in/keshavkandoi/)

## License

The repository has no license file. The release archive contains locally generated conversational data. Earlier IIT Bombay sources are separate and excluded from the release. CFILT lists CC BY-NC 4.0 for its corpus and notes that compiled sources may have distinct terms. AI assistance was used in development and release preparation. [Corpus provenance](transformer-from-scratch/docs/CORPUS_WORKFLOW.md)
