# Transformer from scratch

## Phase 1 · Data preparation

This project prepares English–Hindi parallel text for an encoder–decoder Transformer. It cleans and splits a CSV, trains a shared byte-level BPE vocabulary from scratch, and builds padded PyTorch batches for teacher forcing. Phase 1 ends at the batch interface. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Setup and run

From the project root, using Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/inspect_batch.py
python -m pytest -q
```

Preparation writes `train.csv`, `val.csv` and `test.csv` to `data/processed/`, plus `tokenizer.json` and `metadata.json` to `artifacts/tokenizer/`. Inspection shows the first batch in split order and defaults to the training split. Pass `--split val` or `--split test` to inspect held-out data.

## Use a larger corpus

Supply a UTF-8 CSV with `source` and `target` headers. Quote fields containing commas or newlines using standard CSV quoting.

```bash
python scripts/prepare_data.py --input /path/to/parallel.csv \
  --processed-dir data/processed/real --tokenizer-dir artifacts/tokenizer/real
python scripts/inspect_batch.py \
  --processed-dir data/processed/real --tokenizer-dir artifacts/tokenizer/real
```

Edit `configs/data_config.yaml` or pass `--config /path/to/config.yaml` to both scripts. The defaults use seed 42, batch size 8, sequence limits of 128 IDs including boundary tokens, and a vocabulary target of 8,000.

Preparation trims surrounding whitespace, normalizes to Unicode NFC, drops invalid rows and removes duplicate pairs. The seeded split groups by the exact normalized English source. You need at least three distinct sources. Large groups can shift the row counts away from the requested ratios. Near-duplicates and repeated targets across different sources can still appear across splits.

BPE trains on both languages from the training split alone. Its 256-byte alphabet and four special tokens require a vocabulary target of at least 260. The learned vocabulary can be smaller than the requested target. Encoding preserves whitespace in the supplied text and decoding reconstructs its NFC form. For a sequence truncated at a byte-token boundary, the preview can end in a replacement character if the cutoff falls inside a Unicode character. [Hugging Face ByteLevel pre-tokenizer](https://huggingface.co/docs/tokenizers/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel)

## Sequences and masks

The encoder receives source IDs followed by `<eos>`. With teacher forcing, the decoder receives `<bos>` followed by the gold Hindi IDs. Its labels are those same Hindi IDs followed by `<eos>`. The shift lets query position zero predict the first Hindi token from `<bos>`. Later positions predict the next ID using the gold prefix. The sequence limits reserve room for the boundary tokens. The collator pads on the right using the tokenizer's actual `<pad>` ID.

`B` is the batch size, `S` is the longest encoder sequence in the batch and `T` is the longest decoder sequence. Count non-padding IDs along the last axis to recover individual sequence lengths.

| Batch key | Shape | Type and meaning |
| --- | --- | --- |
| `encoder_input_ids` | `[B, S]` | `torch.long` source IDs |
| `decoder_input_ids` | `[B, T]` | `torch.long` shifted decoder inputs |
| `decoder_target_ids` | `[B, T]` | `torch.long` next-token labels |
| `encoder_attention_mask` | `[B, 1, 1, S]` | `torch.bool` source key mask broadcast across heads and queries |
| `decoder_attention_mask` | `[B, 1, T, T]` | `torch.bool` causal mask combined with decoder key padding |
| `cross_attention_mask` | `[B, 1, T, S]` | `torch.bool` source key mask for every decoder query |

For attention scores shaped `[B, heads, queries, keys]`, `True` allows attention and `False` blocks it. The causal mask for four target positions is:

```text
T F F F
T T F F
T T T F
T T T T
```

Encoder `<eos>` and decoder `<bos>` ensure each valid query has an allowed key. Padded queries may still produce intermediate outputs because the masks operate on keys. Use `ignore_index=tokenizer.pad_id` on decoder labels for the loss. These booleans follow PyTorch scaled dot-product attention semantics. The decoder mask already includes causality, so pass it as `attn_mask` with `is_causal=False`. [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

## Phase 1 outputs for embeddings

Use the saved tokenizer's actual vocabulary size and padding ID to configure embeddings. The encoder and decoder ID tensors index those embeddings and determine the positional encoding lengths. The three masks are ready for attention layers, and the shifted labels are ready for a later training loop.

The 40-pair sample is for testing the pipeline. Its vocabulary and held-out splits are too small for meaningful translation training or evaluation.

## Phase 2 · Token embeddings and positions

Run this after Phase 1 preparation, from the project root:

```bash
python scripts/inspect_embeddings.py
python -m pytest -q
```

The inspection script uses the saved tokenizer's actual vocabulary size and padding ID. It accepts the same `--config`, `--processed-dir` and `--tokenizer-dir` options as the batch inspector, as well as `--split` and `--model-config`. In `configs/model_config.yaml`, set `max_seq_len` to at least the larger of the data config's source and target limits. The default embedding width is 256, the maximum length is 128 and dropout is 0.1. This phase uses a positive, even embedding width.

### From token IDs to vectors

An embedding matrix is a trainable table with one row per tokenizer ID and `d_model` values per row. For example, given IDs `[17, 42]`, PyTorch's `nn.Embedding` retrieves rows 17 and 42. The integer IDs are lookup indices. Attention needs floating-point vectors it can project into queries, keys and values.

Each table starts with randomly initialized parameters. The training objective will shape these parameters into representations useful for translation. The padding row starts at zero and receives no lookup gradient. The encoder and decoder have separate tables so they can learn independently, even though they index the same bilingual vocabulary. Sharing the tables or tying decoder embeddings to the output projection are later design choices. [PyTorch Embedding documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

The original Transformer multiplies token vectors by `sqrt(d_model)` before adding position information. This makes the token contribution larger relative to the bounded sinusoidal values. [Attention Is All You Need, §3.4–3.5](https://arxiv.org/abs/1706.03762)

### From vectors to position-aware representations

Positions start at zero because the first column in each batch sequence is indexed by zero. Each pair of features uses sine and cosine at the same frequency. Later pairs change more slowly across positions.

```text
PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))
```

For width 6, position zero is `[0, 1, 0, 1, 0, 1]`. Position one begins `[0.8415, 0.5403, 0.0464, 0.9989, 0.0022, 1.0000]`. Position and token vectors both have width `d_model`, so they can be added element by element and attention receives the same feature width. The precomputed position table is a persistent, non-trainable buffer that moves with the module.

Training-mode dropout randomly zeros features and rescales the survivors. In evaluation mode it passes the combined vector through as-is. After dropout, the module zeros every vector whose input ID equals the tokenizer's padding ID. Valid tokens, including `<bos>` and `<eos>`, keep their positional contribution.

| Input | Output |
| --- | --- |
| `encoder_input_ids [B, S]` | `encoder_x [B, S, d_model]` |
| `decoder_input_ids [B, T]` | `decoder_x [B, T, d_model]` |

Phase 3 self-attention will receive these position-aware vectors and the corresponding Phase 1 boolean masks. Use the encoder mask with `encoder_x` and the causal decoder mask with `decoder_x`. The embedding modules leave the batch's masks and shifted decoder labels untouched.

## Phase 3 · Scaled dot-product attention

After preparing the Phase 1 data, run:

```bash
python scripts/inspect_attention.py
python -m pytest -q
```

The inspector shows scaled scores, allowed keys, probabilities and gradient norms on a real batch. It uses evaluation mode so dropout is disabled. Use `--split val` or `--split test` to inspect another split.

### How one head builds context

A query describes what a token is looking for. A key describes what a candidate token can match against, and a value carries the information that candidate contributes. Three independent, bias-free linear projections produce Q, K and V. They start with random weights, which a future training objective can adjust.

The dot product between each query and key measures how strongly their projected features align. Divide each score by `sqrt(d_k)` to keep typical logits from growing with the feature dimension and pushing softmax toward saturation. Mask blocked keys before softmax. Softmax converts the allowed scores into a probability distribution across keys, separately for each query. Multiplying those probabilities by V builds a weighted sum of the candidate values. The result is contextual because it depends on the other tokens the query can attend to. [Attention Is All You Need, §3.2.1](https://arxiv.org/abs/1706.03762)

During training, attention dropout randomly drops and rescales some probabilities before multiplying by V. This regularizes how the head combines information across keys. The dropped and rescaled weights need not sum to one. When requested with `return_weights=True`, the module returns the original pre-dropout probabilities, which sum to one for each row with an allowed key.

### Inputs, shapes and masks

| Quantity | Shape |
| --- | --- |
| Projected Q | `[B, H, Tq, d_k]` |
| Projected K | `[B, H, Tk, d_k]` |
| Projected V | `[B, H, Tk, d_v]` |
| Scores and probabilities | `[B, H, Tq, Tk]` |
| Scaled-attention context | `[B, H, Tq, d_v]` |
| Single-head context | `[B, Tq, head_dim]` |

`Tq` and `Tk` count query and key positions, respectively. `SingleAttentionHead` takes input embeddings of width `d_model`, sets `H=1` and uses `head_dim` for both projected feature sizes. It removes the singleton head axis from the context and retains it in the optional probabilities. The default `head_dim` is `256 / 4 = 64`, derived from the divisible model width and configured head count. `attention_dropout` in the model config controls probability dropout separately from embedding dropout.

- **Encoder self-attention** uses encoder embeddings for Q, K and V. The mask `[B, 1, 1, S]` lets source queries see all valid source keys.
- **Decoder self-attention** uses decoder embeddings for Q, K and V. The mask `[B, 1, T, T]` also blocks future keys. For four non-padding positions the rows are `TFFF`, `TTFF`, `TTTF` and `TTTT`.
- **Cross-attention** uses decoder embeddings for Q and encoder embeddings for K and V. The mask `[B, 1, T, S]` allows valid source keys, and source and target lengths can differ.

All three modes reuse the Phase 1 masks. `True` allows attention and `False` blocks it. Masking padding keys keeps padding from consuming any probability mass. A fully masked query has zero probabilities and zero context. A padded query can still have nonzero context if the mask allows keys for that row.

### Worked example

Take Q and K both equal to the two-by-two identity and use value rows `[1, 2]` and `[3, 4]`. Then `QKᵀ` is the identity. Dividing by `sqrt(2)` makes each diagonal logit approximately 0.7071. With no mask, row-wise softmax yields `[p, 1-p]` and `[1-p, p]`, where `p = exp(1/sqrt(2)) / (1 + exp(1/sqrt(2))) ≈ 0.6698`. The first context is `p × [1, 2] + (1-p) × [3, 4] ≈ [1.6605, 2.6605]`. The second is approximately `[2.3395, 3.3395]`. Tests check both rows and compare the general implementation with PyTorch’s optimized reference.

## Phase 4 · Multi-head attention

Inspect a real batch in evaluation mode after preparing the Phase 1 data.

```bash
python scripts/inspect_multi_head_attention.py
python scripts/inspect_multi_head_attention.py --split val
python scripts/inspect_multi_head_attention.py --split test
```

Each head sees the full query and key/value sequences at width `d_model`. Independent Q, K and V projections let heads learn different ways to compare tokens and combine information. The default model uses four heads, so each projects 256 input features to `head_dim = 256 / 4 = 64`. Each head runs the same Phase 3 attention calculation.

The four contexts have shape `[B, Tq, 64]`. Concatenating along the feature axis preserves all four contexts in their head order and gives `[B, Tq, 256]`. The learned, bias-free `W_o` mixes features across heads and returns `[B, Tq, 256]`. Matching the input width will let Phase 5 add the attention result to the sublayer input. With the default settings, Q/K/V contribute `3 × 4 × 256 × 64` parameters and `W_o` adds `256 × 256`, for **262,144 trainable parameters per module**.

| Attention mode | Queries | Keys and values | Phase 1 mask | Output | Optional probabilities |
| --- | --- | --- | --- | --- | --- |
| Encoder self-attention | Encoder input | Encoder input | `[B, 1, 1, S]` | `[B, S, 256]` | `[B, 4, S, S]` |
| Decoder causal self-attention | Decoder input | Decoder input | `[B, 1, T, T]` | `[B, T, 256]` | `[B, 4, T, T]` |
| Decoder-to-encoder cross-attention | Decoder input | Encoder input | `[B, 1, T, S]` | `[B, T, 256]` | `[B, 4, T, S]` |

The singleton head axis of each Phase 1 mask broadcasts the same rules to every head. Encoder self-attention allows valid source keys. Decoder self-attention also blocks future target positions. Cross-attention lets each decoder query see valid source keys even when `S` and `T` differ. A mask with an explicit head dimension can assign different allowed keys per head. `True` allows attention. Fully masked rows across all heads stay zero through `W_o` because its bias is absent.

Use separate instances for the three roles so each learns its own projections. Pass only `query_input` for self-attention and additionally pass `key_value_input` for cross-attention. By default the call returns the projected context. Pass `return_attention_weights=True` to receive `(context, probabilities)`. The probabilities are from before dropout. During training, `attention_dropout` applies once to the probabilities used to compute each head's context.

Phase 5 can call the module with the position-aware input vectors and the matching boolean mask, then use its `[B, Tq, d_model]` context in the residual path and normalization.

## Phase 5 · Layer normalization, residual connections and feed-forward network

After preparing the data, run the component inspection and tests from the project root.

```bash
python scripts/inspect_transformer_components.py
python scripts/inspect_transformer_components.py --split val
python scripts/inspect_transformer_components.py --split test
python -m pytest tests/test_layer_norm.py tests/test_residual.py tests/test_feed_forward.py tests/test_phase5_integration.py -q
python -m pytest -q -rs
```

The inspector defaults to CPU and accepts `--device cuda` or `--device mps` when available. It uses a temporary regression objective on valid token positions to verify gradients and one optimizer step.

### Layer normalization

For each token independently, compute the mean and population variance across its feature values.

```text
mean = mean(x)
variance = mean((x - mean)²)
output = gamma × (x - mean) / sqrt(variance + epsilon) + beta
```

With the initial gamma of one and beta of zero, `[2, 4, 6]` becomes approximately `[-1.2247, 0, 1.2247]`. Epsilon keeps the denominator positive for constant vectors. Training can adjust gamma to scale each feature and beta to shift it. LayerNorm works across a token's features and uses the same computation in training and evaluation. BatchNorm typically aggregates per-feature statistics across examples during training and uses running statistics during evaluation. Attention softmax instead converts scores across keys into nonnegative probabilities that sum to one.

### Post-norm and padding

The residual path adds the original input to the dropped sublayer output. The addition provides a direct path for gradients through the sum. In the original post-norm design, normalization follows that addition. Each sublayer owns a separate gamma and beta. [Attention Is All You Need, §3.1 and §5.4](https://arxiv.org/abs/1706.03762)

```text
A = MultiHeadAttention(X, attention_mask)
Y = LayerNorm(X + Dropout(A))
F = FFN(Y)
Z = LayerNorm(Y + Dropout(F))
```

A padded query can attend to valid keys, and even a zero vector can become nonzero after normalization when beta learns a shift. Pass `token_valid_mask=token_ids.ne(pad_id)` to each residual wrapper to zero padded query vectors after normalization. This boolean `[B, T]` mask describes query positions. The Phase 1 attention mask controls allowed keys.

### Position-wise feed-forward network

The FFN applies the same parameters to each token independently. Its first linear layer expands 256 features to 1024 so the network can form more intermediate features. ReLU clips negative activations to zero and introduces nonlinearity. The second linear layer returns to width 256 so the result can be added to the residual input.

| Stage | Output shape with default configuration |
| --- | --- |
| Embedding, attention, first post-norm | `[B, T, 256]` |
| FFN first linear and ReLU | `[B, T, 1024]` |
| FFN second linear and final post-norm | `[B, T, 256]` |

The FFN applies hidden dropout after ReLU and before its second linear layer. Each residual wrapper separately drops features of the sublayer output before addition. Both dropout locations are disabled in evaluation mode. The defaults use `dropout: 0.1`, `layer_norm_eps: 0.00001`, `normalization_style: post_norm` and `ffn_activation: relu` in `configs/model_config.yaml`.

### Phase 6 interface

An encoder block can call `MultiHeadAttention` with the source representations and encoder key mask, use `ResidualAddNorm(x, attention_output, token_valid_mask)`, pass that result through `PositionwiseFeedForward`, and apply a second independent `ResidualAddNorm` using the same token-validity mask. The resulting `[B, S, d_model]` tensor can feed the next encoder block.
