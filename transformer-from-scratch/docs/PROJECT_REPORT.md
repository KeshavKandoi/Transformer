# Transformer From Scratch
## English-to-Hindi translation

Technical project report · Keshav Kandoi

### Executive summary

This project implements an encoder–decoder Transformer in PyTorch, trains its translation weights from random initialization, and serves a frozen checkpoint in a local browser. The release checkpoint has **1,126,656 trainable parameters**. Greedy decoding matched **1,079/1,088** synthetic validation references and **1,621/1,642** held-out combinations. Performance dropped to **330/433** on entirely withheld structures. The project demonstrates the implementation and the limits of a controlled experiment.

## 1. Problem and objectives

Translation must preserve predicates, entities, quantities, and relationships across languages. A low teacher-forced loss and fluent output each give incomplete evidence about those requirements. The project implements the full training and generation path, measures source conditioning, evaluates structured holdouts, and exposes the actual outputs and attention tensors so failures can be inspected.

## 2. Transformer architecture


Input strings pass through byte-level BPE. Separate source and target embedding matrices map each ID to width 128. Fixed sinusoidal positions add sequence order and support the configured 64-position context. The encoder has two blocks with bidirectional self-attention and position-wise feed-forward layers. Each of the decoder’s two blocks applies causal self-attention, cross-attention, and a feed-forward network. Residuals are followed by layer normalization. The feed-forward network expands from 128 to 384, applies ReLU, and projects back. A separate linear layer without bias projects decoder states onto the 874-token vocabulary.

For each head, **Attention(Q,K,V) = softmax(QKᵀ / √dₖ + M)V**. The implementation learns Q, K, and V projections, reshapes to four heads of width 32, scales the scores, applies a Boolean allow mask, and concatenates and projects the head outputs. For encoder self-attention, all valid source keys are visible. Decoder self-attention combines key padding with a lower-triangular causal mask. Cross-attention uses decoder queries against encoder keys and values. Its mask allows valid source keys at every decoder position.

During training, the decoder receives BOS and the gold target prefix and predicts the shifted target ending in EOS. At inference, each decoder step conditions on generated tokens. Greedy chooses the next token directly. Beam width four retains several candidate prefixes and ranks complete sequences. Both strategies stop at EOS or the generation cap.

## 3. Data provenance and preparation

The release corpus is generated locally from bilingual grammar rules. It includes greetings, affect, possession, quantities, motion and time, named introductions, and help requests. Hindi references use controlled agreement, familiar second-person address, and a masculine first-person default where agreement requires one. They are programmatic labels without an independent bilingual review.

| Partition | Pairs | Purpose |
|---|---:|---|
| Train | 8,597 | Fit BPE and model weights |
| Validation | 1,088 | Early stopping and checkpoint selection |
| Composition | 1,642 | Reserved combinations of known constructions |
| Lexical | 2,434 | Reserved lexical combinations |
| Structure | 433 | Entire withheld question families |

Normalized English and Hindi strings are disjoint, and semantic groups do not span splits. Composition intentionally reuses familiar construction families. The shared train-only byte-level BPE vocabulary saturated at 874 tokens against a target of 2,000. The longest encoded sequence used 27 positions including EOS. The release archive contains only synthetic records and their evaluations.

## 4. Training and checkpointing

| Setting | Value |
|---|---|
| Seed, device, CPU threads | 91, CPU, 3 |
| Batch and accumulation | 48 pairs, one batch per update |
| Optimizer | AdamW, configured learning rate 0.0015 |
| Schedule | 4% planned-update warmup, then inverse square root |
| Regularization | 0.1 dropout, 0.05 attention dropout, gradient norm cap 1 |
| Selection | Validation token NLL, early-stopping patience 5 |
| Actual run | 16 epochs, 2,880 updates, 206.46 seconds summed epoch time |
| Selected checkpoint | Epoch 11, update 1,980 |
| Selected train / validation NLL | 0.005033 / 0.005822 |

Padding labels are excluded from cross-entropy. Backpropagation updates the scratch-trained weights through AdamW. The trainer supports periodic optimizer-boundary commits, deterministic CPU resume, and separate latest and best validation checkpoints. The demo matches the learned tokenizer mapping and source inventory checksum to the checkpoint. The release used the existing weights.


## 5. Earlier IIT Bombay experiments

The repository has earlier IIT Bombay pilot, 50K, and wider 125K experiment paths. Those used separately prepared third-party data. The wide checkpoint showed a 0.5756 increase in teacher-forced NLL when source sentences were shuffled on a 100-pair probe. Generation still lost meaning. A matched 30-example short validation diagnostic measured greedy chrF2 17.40 and BLEU 2.26. A manually audited 200-row training sample had 156 aligned, 25 partial, nine mismatched, and ten uncertain pairs.

The evidence establishes source sensitivity and confirmed alignment failures. The audit does not quantify how much noise, scale, tokenizer fragmentation, or domain mix contributed to translation errors. CFILT publishes the IITB corpus under CC BY-NC 4.0 and notes separate terms for compiled materials. The release does not redistribute those materials or their predictions.

## 6. Frozen evaluation

Evaluation decoded each frozen source with greedy generation and a 32-token cap, then compared its Hindi output with the single stored reference. Exact is a whole-sequence match. chrF2 compares character n-grams with recall weight β = 2. SacreBLEU uses international tokenization. EOS reports completion before the generation cap. Surface slot checks test known markers and are narrower than bilingual adequacy evaluation.

| Split | Exact | chrF2 | BLEU | EOS |
|---|---:|---:|---:|---:|
| Synthetic validation | 1,079/1,088 | 99.81 | 99.61 | 1,088/1,088 |
| Held-out composition | 1,621/1,642 | 99.68 | 99.31 | 1,639/1,642 |
| Reserved lexical combinations | 2,399/2,434 | 99.92 | 99.24 | 2,434/2,434 |
| Withheld structures | 330/433 | 87.44 | 83.24 | 428/433 |

The release rerun reproduced all 5,597 saved synthetic records and aggregate fields. The composition split had 25/30 successful name checks. All 97 withheld structure examples containing unseen English word forms missed exact match. Separately, historical 24-row diagnostics for IITB and FLORES each yielded zero exact matches, illustrating domain shift on small samples.

### Observed errors

“Can you help me?” generated **क्या तुम मुझे फी मदद कर सकते हो?** in greedy decoding and **क्या तुम मेरी मदद कर सकते हो?** with beam width four. “What is your name?” generated **उनका नाम क्या है?** with both strategies, changing the possessive. “My name is Ravi.” generated **मेरा नाम रवि है।** These are recorded outputs. The gallery includes nine validation mismatches.

## 7. Web application, API and attention

The local Python HTTP server serves a dependency-free browser interface. GET `/api/health` identifies the loaded checkpoint by hash. GET `/api/info` returns checkpoint configuration, evaluated scores, curves, scope and gallery. POST `/api/translate` takes JSON text plus a strategy and returns Hindi outputs, EOS, CPU generation latency, output flags and attention. Unsupported input returns HTTP 422. The input gate normalizes supported case, punctuation, whitespace, selected contractions and documented elliptical aliases.

The visualizer runs a diagnostic pass on the chosen generated prefix with dropout off. It displays source BPE labels and emitted target labels. Encoder attention has source-by-source axes. Causal decoder attention uses decoder-input axes. Cross-attention uses Hindi prefix queries and English source keys. Selecting a layer or head changes the rendered tensor. Attention weights characterize a specific mixture and do not establish causal importance or explain all model behavior.


The reported latency is elapsed CPU generation time for each strategy. Browser rendering, HTTP, queueing and the diagnostic pass are outside that measurement. Per-request latency and captured values depend on hardware, sentence length and warm-up.

## 8. Testing, limitations and next steps

The tests cover token mapping, masks, attention dimensions, autoregressive EOS, deterministic training and resume, checkpoint compatibility and evaluation. Browser tests exercise live predictions, malformed metadata, unsupported inputs, a delayed error response, keyboard shortcuts, clipboard, navigation and responsive layouts. The release report records exact test counts and commands.

The grammar gate accepts known constructions. Accepted inputs can still fail. The fixed synthetic references do not capture the diversity of valid Hindi output, and the available checkpoints do not support arbitrary English. Next experiments should freeze a bilingual-reviewed paraphrase and role-contrast suite, test politeness and gender variants separately, and measure human adequacy alongside surface metrics. Compare input-dependent errors and decoding changes under matched evaluation before scaling training.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html).
- [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers/index).
- [SacreBLEU](https://github.com/mjpost/sacrebleu).
- [CFILT IIT Bombay corpus and terms](https://www.cfilt.iitb.ac.in/iitb_parallel/).
- [Implementation history and experiment instructions](IMPLEMENTATION_HISTORY.md).
