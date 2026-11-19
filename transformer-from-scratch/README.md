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

## Phase 6 · Transformer encoder

The encoder turns source IDs into contextual vectors, one per source position. It applies the existing token embeddings, sinusoidal positions and embedding dropout, then passes the result through four independent encoder blocks by default. Each block refines the preceding block's representation with self-attention and a position-wise feed-forward network.

Inside an encoder block, the same source sequence supplies the inputs to the Q, K and V projections. Self-attention is bidirectional because all source tokens are available when encoding. An earlier source query may attend to a later valid source key, so the encoder uses no causal mask by default. Four independently learned heads can capture different relations in parallel. Their 64-feature contexts concatenate to width 256, then the learned output projection mixes features across heads and preserves the width required by the residual addition.

Each block applies the post-norm ordering from Phase 5. One custom LayerNorm follows the residual addition around attention. An independent LayerNorm follows the residual addition around the FFN. The FFN adds a nonlinear transformation for each token, expanding its features to 1024 with ReLU and projecting back to 256. The next block receives the resulting vectors.

| Stage | Observed training batch |
| --- | --- |
| Source IDs | `[8, 6]` |
| Embeddings | `[8, 6, 256]` |
| Blocks 1–4, each output | `[8, 6, 256]` |
| Final encoder memory | `[8, 6, 256]` |
| Optional attention maps | `[8, 4, 4, 6, 6]` |

```python
encoder = TransformerEncoder(tokenizer.vocab_size, tokenizer.pad_id, model_config)
memory, maps = encoder(
    source_ids=batch["encoder_input_ids"],
    attention_mask=batch["encoder_attention_mask"],
    return_attention_weights=True,
)
```

Import `TransformerEncoder` from `src.model.encoder` and load `model_config` with `ModelConfig.load`. Omit the return flag for memory alone. The returned maps contain pre-dropout probabilities with axes `[batch, layer, head, query, key]`. Omit the explicit attention mask to derive the padding mask from source IDs. A supplied boolean mask may restrict more keys and is intersected with the ID-derived key mask, so padding keys remain blocked. A separate token-validity mask zeros padded query vectors after each residual normalization. Empty source examples and rows containing only padding raise a validation error.

For Phase 7, retain the source padding mask and pass the memory as the key/value input to decoder cross-attention. Decoder representations supply queries, and the Phase 1 cross-attention mask broadcasts across heads with source and target lengths that can differ. Encoder memory retains its computation graph for end-to-end gradients.

```bash
python scripts/inspect_encoder.py
python scripts/inspect_encoder.py --split val
python scripts/inspect_encoder.py --split test
python -m pytest tests/test_encoder_*.py -q
python -m pytest -q -rs
```

The inspector defaults to CPU and accepts `--device cuda` or `--device mps` when available. It prints a small attention map from each layer, verifies padding and gradients, and checks one temporary AdamW update and a state-dictionary round-trip.

## Phase 7 · Transformer decoder

The decoder builds one contextual vector for each target input position. During teacher forcing, Phase 1 supplies gold target tokens shifted right. For example, the input `<bos> हम दोस्त हैं` aligns with labels `हम दोस्त हैं <eos>`. Each decoder position can see the gold prefix needed to predict the next token. Pass `decoder_input_ids` to the decoder and save `decoder_target_ids` for the later loss. Feeding the labels directly would expose the token the model is supposed to predict.

Four independent post-norm blocks refine the target states in sequence. Each block has three distinct Add & Norm operations because each of its three sublayers adds its result to the sublayer input before applying its own normalization.

1. **Masked self-attention** takes Q, K and V from decoder states. A target query can attend to its own position and earlier valid positions. Future keys are blocked, so changing future IDs leaves shared-prefix representations unchanged. Encoder self-attention is bidirectional because the complete source is available.
2. **Cross-attention** takes Q from the post-normalized decoder state because each target position is looking for relevant source information. K and V come from encoder memory and describe the source positions being searched and combined. This sublayer learns separate projections from masked self-attention.
3. **Position-wise FFN** expands each target vector from 256 to 1024 features, applies ReLU and hidden dropout, then projects back to 256 for the residual addition.

```python
from src.model.decoder import TransformerDecoder

decoder = TransformerDecoder(tokenizer.vocab_size, tokenizer.pad_id, model_config)
memory = encoder(batch["encoder_input_ids"], batch["encoder_attention_mask"])
states, self_maps, cross_maps = decoder(
    decoder_input_ids=batch["decoder_input_ids"],
    encoder_memory=memory,
    decoder_attention_mask=batch["decoder_attention_mask"],
    cross_attention_mask=batch["cross_attention_mask"],
    return_attention_weights=True,
)
```

The decoder derives the causal target key-padding mask from its input IDs. An explicit self-attention mask can further restrict allowed keys. Supply the cross-attention mask from the source IDs, which are where source padding is known. If source and target lengths differ, shape validation catches a swapped pair of standard masks. With equal lengths, shape alone cannot identify which mask is which. The decoder still enforces causality on its self-attention input.

Both masks use `True` for allowed keys. Padded target queries are zeroed after each residual normalization. Fully masked attention rows have zero probabilities and zero attention context. The residual path can still carry the input at a valid query. The encoder rejects entirely padded source examples, and the decoder rejects entirely padded target examples.

| Quantity | General shape | Training batch |
| --- | --- | --- |
| Shifted decoder IDs | `[B, T]` | `[8, 16]` |
| Encoder memory | `[B, S, d_model]` | `[8, 6, 256]` |
| Decoder embeddings and each block output | `[B, T, d_model]` | `[8, 16, 256]` |
| Self-attention maps | `[B, layers, heads, T, T]` | `[8, 4, 4, 16, 16]` |
| Cross-attention maps | `[B, layers, heads, T, S]` | `[8, 4, 4, 16, 6]` |

Maps show pre-dropout probabilities in `[batch, layer, head, query, key]` order. Omit the return flag to receive just the final decoder states. Phase 8 will project those states to vocabulary logits and align those logits with the existing shifted labels.

```bash
python scripts/inspect_decoder.py
python scripts/inspect_decoder.py --split val
python scripts/inspect_decoder.py --split test
python -m pytest tests/test_decoder_*.py tests/test_cross_attention_integration.py -q
python -m pytest -q -rs
```

The inspector defaults to CPU and accepts `--device cuda` or `--device mps` when available. It prints a small map from each layer and runs one temporary optimizer step and a state-dictionary round-trip.

## Phase 8 · Full Transformer and vocabulary logits

`EncoderDecoderTransformer` connects the Phase 6 encoder, Phase 7 decoder and a learned vocabulary projection. The encoder reads the source IDs and passes its memory to the decoder without detaching it. Decoder queries can attend to valid source keys through cross-attention. Causal self-attention limits each target position to itself and earlier valid positions. Because the memory stays in the graph, a loss on the logits can update the vocabulary head and both stacks.

```python
from src.model.transformer import EncoderDecoderTransformer

model = EncoderDecoderTransformer(tokenizer.vocab_size, tokenizer.pad_id, model_config)
logits = model(
    encoder_input_ids=batch["encoder_input_ids"],
    decoder_input_ids=batch["decoder_input_ids"],
    encoder_attention_mask=batch["encoder_attention_mask"],
    decoder_attention_mask=batch["decoder_attention_mask"],
    cross_attention_mask=batch["cross_attention_mask"],
)
```

The model derives padding and causality from the IDs when masks are omitted. Explicit boolean masks can further restrict allowed positions. Set `return_diagnostics=True` to get a `TransformerDiagnostics` result with logits, encoder memory, decoder hidden states, and three sets of pre-dropout attention maps. Use `torch.no_grad()` for inspection without a gradient graph.

| Stage | Observed training batch |
| --- | --- |
| Source IDs | `[8, 6]` |
| Encoder memory | `[8, 6, 256]` |
| Shifted decoder IDs and target labels | `[8, 16]` each |
| Decoder hidden states | `[8, 16, 256]` |
| Vocabulary logits | `[8, 16, 626]` |

For each batch item and target position, the vocabulary projection assigns one raw score, or logit, to every token. That gives the `[batch, target length, vocabulary size]` shape. The saved tokenizer supplies the vocabulary size and pad ID. The output projection has its own weights, separate from both embedding tables. Set `output_projection_bias: true` to enable a bias. With bias enabled, zero padded hidden states can still yield nonzero logits.

Apply `logits.softmax(dim=-1)` when you need probabilities for inspection. Give PyTorch cross-entropy raw logits because it computes log-softmax internally. Teacher forcing supplies `decoder_input_ids`, which start with BOS and contain the gold prefix. Score the corresponding `decoder_target_ids`, which end with EOS. The IDs are already shifted by the data pipeline.

```python
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, tokenizer.vocab_size),
    batch["decoder_target_ids"].reshape(-1),
    ignore_index=tokenizer.pad_id,
)
```

Labels are `torch.long`, and at least one label must be a valid non-padding target for the mean loss. The inspection helper rejects all-padding labels. Padding labels contribute neither to the mean nor to the gradient. Top token scores before training reflect random initialization and can decode to fragments, including partial UTF-8 tokens.

```bash
python scripts/inspect_full_transformer.py
python scripts/inspect_full_transformer.py --split val
python scripts/inspect_full_transformer.py --split test
python -m pytest tests/test_output_projection.py tests/test_full_transformer.py tests/test_transformer_*.py -q
python -m pytest -q -rs
```

The inspector runs one temporary optimizer step and verifies state-dictionary reload. Phase 9 will build the training and validation loops around this logits interface, including repeated optimization and reporting on held-out data. Autoregressive translation will require a later decoding loop.

## Phase 9 · Training and evaluation

Run preparation once before training. Preparation writes a manifest pairing the split files, tokenizer hash and split settings. Rerun preparation for artifacts from before Phase 9 or when changing preparation settings.

```bash
python scripts/prepare_data.py
python scripts/train.py --config configs/training_config.yaml --dry-run
python scripts/overfit_single_batch.py --curve runs/overfit.jsonl
python scripts/train.py --config configs/training_config.yaml --device cpu --run-name en-hi
python scripts/train.py --resume checkpoints/en-hi/latest.pt --epochs 25
python scripts/evaluate.py --checkpoint checkpoints/en-hi/best.pt --device cpu
python scripts/evaluate.py --checkpoint checkpoints/en-hi/best.pt --split test
```

The dry run executes a real forward, loss and backward pass. The overfit diagnostic repeatedly trains a small model with dropout disabled on one real batch. It exits with an error if it misses the configured loss target or 95% token accuracy. Use `--max-steps`, `--target-loss`, `--learning-rate` and `--batch-size` to control the experiment. The optional JSONL curve records the initial metrics and every update. If the loss stalls, check BOS/EOS alignment, padding masks, gradient norms and the effective learning rate before running a longer experiment.

### What happens on each update

The model uses shifted decoder inputs and returns raw vocabulary logits. Masked cross-entropy scores the correct next-token IDs and sums the loss over non-padding targets. Backpropagation computes gradients through the vocabulary head, decoder and encoder. AdamW uses running gradient statistics and decoupled weight decay to update the parameters.

For gradient accumulation, each microbatch backpropagates its **summed** loss. Before stepping, the trainer divides accumulated gradients by the total number of valid target tokens in that window. This produces a token-weighted mean even when batches have different amounts of padding. The final partial window also gets an update. Gradient clipping then limits the global norm before AdamW steps. The logged gradient norm is measured before clipping. A non-finite loss or gradient aborts the update with an error.

The default scheduler linearly raises the learning rate to its peak through the warmup boundary, then decays by the inverse square root of the update number. Warmup lets the optimizer start with smaller updates. This is the peak-LR-normalized variant. Set the scheduler type to `constant` for a fixed learning rate. One global step means one optimizer update. The first default update uses `3e-6` and update 100 uses the peak of `3e-4`. [Attention Is All You Need, §5.3](https://arxiv.org/abs/1706.03762)

### Metrics and checkpoints

Console logs show validation summaries and periodic update rates and gradient norms. `runs/<run_name>/metrics.jsonl` has one row per completed epoch, with train and validation metrics, valid-token counts and elapsed time. Loss is averaged over valid tokens. Token accuracy is **teacher-forced next-token accuracy**. Perplexity exponentiates unsmoothed validation loss. Overflow emits a warning and serializes perplexity as `null` with an `overflow` status. Optional label smoothing affects the training objective only and still ignores padded labels.

Validation runs at the first epoch and at the configured validation frequency. The test split is scored only by an explicit `evaluate.py --split test` command. `latest.pt` is replaced atomically after every 50 completed optimizer updates by default and after every completed epoch. `best.pt` tracks the lowest validation loss, and `checkpoint_frequency` controls numbered epoch snapshots. Early stopping uses consecutive validation checks without a gain exceeding `min_delta`.

Use `--save-every-updates N` or the YAML setting `save_every_updates` to set the periodic interval. An update means one completed optimizer and scheduler step followed by gradient reset, even when gradient accumulation uses multiple microbatches. An interval of zero disables periodic saves. A graceful Ctrl+C still saves the last safe optimizer boundary when available. If an update is already running, the signal is deferred until the update commits. If an accumulation window has only partially run, its gradients, metrics and random draws are discarded. A forced kill resumes from the most recent successfully published checkpoint.

Resume using `--resume checkpoints/<run>/latest.pt`. Set `--epochs` to the desired total, including completed epochs. Version 2 checkpoints record completed epochs, the active epoch's next zero-based batch offset, optimizer and scheduler state, committed token metrics, RNG states and configuration fingerprints. The loader rebuilds the epoch's seeded batch order and skips indices without fetching previously optimized examples. A checkpoint after the last update resumes at validation. Version 1 completed-epoch checkpoints still resume from the next epoch.

Bitwise continuation was verified on CPU with deterministic algorithms, the same runtime, and the project's zero-worker loader. Programmatic callers must provide a sized, cursor-aware `resume_train_loader(epoch, start_batch)` for mid-epoch saving and restore. Switching devices or numerical libraries can change results. Keep the matching tokenizer and prepared splits with a checkpoint. Load checkpoints from trusted sources.

`--device auto` prefers CUDA, then Apple MPS, then CPU. Select `--device cuda` or `--device mps` explicitly when available. Phase 9 runs in float32. Output roots can be changed with `--checkpoint-dir` and `--runs-dir`.

### Larger corpora and the Phase 10 interface

The included 40 examples are suitable for verifying the pipeline and too few for useful translation. Prepare a licensed, provenance-checked English–Hindi parallel CSV with `source` and `target` columns. Use separate artifact directories for each corpus. Preparation splits by English source and trains BPE on the training split only.

```bash
python scripts/prepare_data.py --input /path/to/parallel.csv \
  --processed-dir data/processed/real --tokenizer-dir artifacts/tokenizer/real
python scripts/train.py --processed-dir data/processed/real \
  --tokenizer-dir artifacts/tokenizer/real --run-name en-hi-real
```

Use matching directories and `--data-config` for preparation, training and evaluation. The preparation script calls that argument `--config`. Use a held-out corpus and review alignment, duplicates and domain coverage before interpreting metrics. Tune sequence lengths, batch size and warmup on validation data for larger runs.

Phase 10 can load the selected checkpoint's `model_state_dict` with its saved architecture and matching tokenizer, encode source IDs, and use the model's raw next-token logits and tokenizer BOS/EOS/PAD IDs to build autoregressive decoding.

```bash
python -m pytest tests/test_training_*.py tests/test_phase9_integration.py -q
python -m pytest -q -rs
```

## Phase 10 · Autoregressive translation inference

Training uses the shifted gold Hindi prefix through teacher forcing. Inference starts with just the tokenizer's BOS ID and feeds each generated token back as part of the next prefix. The source is tokenized with the saved BPE mapping, terminated with EOS, padded if batched, and encoded once per batch. Every decoder step reuses that encoder memory, applies a causal mask to the growing prefix, and masks padded source keys during cross-attention. The vocabulary head reads the last prefix position. A row stops when it generates EOS or hits `max_new_tokens`. The current decoder recomputes the prefix each step.

- **Greedy** chooses the highest-scoring eligible token. Batched rows stop independently.
- **Beam search** expands the active candidates for a source and prunes by cumulative log probability. It retains completed candidates without extending them. Completed hypotheses are ranked by log probability divided by `((5 + generated_length) / 6) ** length_penalty`, with EOS counted when present. If no beam completes, the highest-ranked active candidate is returned. Width one follows greedy selection.
- **Sampling** divides logits by temperature, keeps the top `k` candidates, then keeps the smallest probability-sorted prefix reaching `top_p`. The remaining distribution is normalized and sampled. Set `top_k: 0` or `top_p: 1.0` to disable the corresponding filter. A seed makes repeated calls with the same batch and configuration reproducible.

PAD and BOS are excluded from next-token selection. The decoder starts with BOS, and returned diagnostic token IDs exclude that initial BOS but include EOS when generated. Decoding removes BOS, EOS and PAD and preserves a generated `<unk>` marker. Byte-level BPE can produce a replacement character when the model emits an invalid UTF-8 byte sequence. The terminal CLI escapes control characters for display. Immediate EOS returns an empty translation. Overlong input is rejected, and the generation budget must fit the checkpoint's positional limit.

Use a Phase 9 checkpoint and its matching tokenizer. Inference verifies the actual vocabulary and special IDs, the tokenizer file hash and a strict state-dictionary load against the saved architecture.

```bash
python scripts/translate.py --checkpoint checkpoints/en-hi/best.pt --text "We are friends" --diagnostics
python scripts/translate.py --checkpoint checkpoints/en-hi/best.pt --text "We are friends" --strategy beam --beam-width 4
python scripts/translate.py --checkpoint checkpoints/en-hi/best.pt --text "We are friends" --strategy sampling --temperature 0.8 --top-k 20 --top-p 0.9 --seed 42
python scripts/translate.py --checkpoint checkpoints/en-hi/best.pt --interactive
python scripts/inspect_generation.py --checkpoint checkpoints/en-hi/best.pt --text "We are friends" --max-new-tokens 16 --top-candidates
python scripts/inspect_generation.py --checkpoint checkpoints/en-hi/best.pt --examples-per-split 2 --max-new-tokens 16
```

Set `--tokenizer-dir` for a custom corpus and `--device cpu`, `cuda`, or `mps` to override automatic device selection. Interactive mode loads the model once. The inspection script displays source IDs, encoder memory shape, selected IDs, optional top candidate scores, EOS status and elapsed time. The checkpoint from the 40-pair smoke run can generate mixed-language fragments and fail to emit EOS. Use a larger, well-aligned licensed corpus and held-out translation evaluation to assess quality.

```bash
python -m pytest tests/test_greedy_decoding.py tests/test_beam_search.py tests/test_sampling.py tests/test_generation_masks.py tests/test_generation_checkpoint.py tests/test_phase10_integration.py -q
python -m pytest -q -rs
```

For Phase 11, the vocabulary-logit step takes the current decoder prefix, source padding information and encoder memory. A future cached step can implement that contract while keeping the decoding strategies independent of attention internals.

## Reproducible local-corpus experiment

The [repository README](../README.md) gives the import, dry-run, training and evaluation commands. The experiment uses the existing Transformer architecture and separate data, tokenizer and checkpoint directories. Set the BPE vocabulary target, split fractions and sequence limits in `configs/experiment_data.yaml`. Set the device, batch size, seed and optimizer options in `configs/experiment_training.yaml`.

### Import, quality and leakage

The importer reads an explicitly selected UTF-8 CSV, checks row structure, collapses whitespace and uses NFC text. It rejects replacement characters, unexpected formatting controls, empty or script-inconsistent rows, excessive character and word lengths, and extreme source/target word-count ratios. A second offline LangID check uses bundled language profiles to reject high-confidence language mismatches on longer inputs. Short text and low-confidence cases still depend on the script filter. Spot-check for misaligned translations, code switching and language errors. These filters are quality heuristics. [LangID documentation](https://github.com/saffsd/langid.py)

The split comparison key normalizes case, punctuation and compatibility characters. Exact equal sources or targets and sentence pairs that reach the configured padded character-trigram Jaccard threshold on either side are grouped transitively. Connected groups stay together in a seeded three-way split. Large groups can skew the target ratios. Semantic paraphrases and documents that share context can still cross splits. If you're working with a corpus that has document or conversation IDs, keep those units separate when preparing your input. The importer validates post-BPE sequence lengths with room for boundary IDs and fails if an example would be truncated. Adjust the import filters or the data and model limits before retrying.

BPE is fitted using train rows only. A new import requires a fresh output directory. The manifests and checkpoints record SHA-256 fingerprints for raw input, splits, tokenizer mapping and resolved settings. Checkpoints also record runtime metadata. Back up the matching artifacts, settings, dependency versions and checkpoint together. Exact continuation is verified at optimizer-update boundaries on the same deterministic CPU runtime. Strict deterministic algorithms are enabled by the experiment config. CPU, CUDA and MPS can have different floating-point results, and strict mode can reject an operation on a backend lacking a deterministic implementation. Use CPU for strict reproducibility when necessary.

### Warmup, resume and generated metrics

The default experimental inverse-square-root schedule warms up for 10% of planned optimizer updates, rounded up and capped to leave a decay update when possible. The horizon accounts for batch size and accumulation. Set `--planned-epochs` when intentionally stopping the first invocation earlier than the original budget. Resume using `scripts/train.py --resume checkpoints/<run>/latest.pt --epochs <total>` with the same data-config, processed and tokenizer arguments. The original resolved warmup survives resume and epoch extensions.

`scripts/evaluate_translations.py` scores **generated** translations on validation or explicitly selected test rows using corpus chrF2 and international-tokenized BLEU. It saves metric signatures, raw predictions and references, EOS flags, and checkpoint fingerprints. With `--training-metrics runs/<run>/metrics.jsonl` it also exports `training_curves.csv` through the evaluated checkpoint epoch. Run-level JSONL retains all completed epochs. Use the same text and decode settings when comparing checkpoints. `--limit` produces a clearly marked subset report. [SacreBLEU metric documentation](https://github.com/mjpost/sacrebleu)

Run `python -m pytest tests/test_corpus_experiment.py -q` for quality, leakage, metrics, EOS and the local-corpus train/evaluate integration checks.

## IIT Bombay English–Hindi fixed-split experiment

The acquisition script defaults to CFILT's IIT Bombay English–Hindi Parallel Corpus, publisher version 3.1, and the exact Hugging Face revision `e104eb47979d570d3df7af4912da85f83efde6fc`. The published Parquet files at that revision contain 1,659,083 train, 520 validation, and 2,507 test rows with `translation.en` and `translation.hi` strings. The publisher's version table lists one fewer training segment. The acquisition manifest records the Parquet counts and file hashes. [CFILT publisher page](https://www.cfilt.iitb.ac.in/iitb_parallel/) · [Pinned Parquet files](https://huggingface.co/datasets/cfilt/iitb-english-hindi/tree/e104eb47979d570d3df7af4912da85f83efde6fc/data)

CFILT lists **CC BY-NC 4.0** and says texts compiled from other sources retain their respective licenses. The Hugging Face card does not independently list a license. Use the artifacts for local noncommercial research. Review each applicable license before redistributing text, predictions, or model checkpoints. Cite Anoop Kunchukuttan, Pratik Mehta and Pushpak Bhattacharyya, *The IIT Bombay English-Hindi Parallel Corpus*, LREC 2018. [Publisher license](https://www.cfilt.iitb.ac.in/iitb_parallel/#license) · [Corpus paper](https://aclanthology.org/L18-1548/)

### Acquire and prepare

```bash
pilot=experiments/iitb-pilot-v1
revision=e104eb47979d570d3df7af4912da85f83efde6fc
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/download_iitb_dataset.py \
  --dataset cfilt/iitb-english-hindi --revision "$revision" \
  --train-samples 10000 --seed 42 --output "$pilot/source"
.venv/bin/python scripts/prepare_iitb_experiment.py \
  --source-dir "$pilot/source" --output "$pilot" --data-config configs/experiment_data.yaml
```

The downloader stores only the three published Parquet files in a local Hub cache and reads them in batches. It scans the entire train split using a seed-based reservoir and writes selected rows in original order. Replay with `--offline` and the same SHA and seed, using a new experiment source directory under the same cache parent. The output is private and ignored by Git. `--force` explicitly replaces a complete acquisition output. No remote Python dataset loader runs.

Preparation reuses the existing Unicode, language and ratio filters. Exact and near-duplicate connected groups use **test, then validation, then train** priority. Members removed from a lower-priority split are discarded. Rows never move between official splits. Repeated keys and near-duplicate groups are boilerplate indicators, and word-count ratios flag suspicious alignment. Review the private length-stratified `processed/alignment_audit.jsonl` to inspect actual translations. Post-BPE overlength rows are excluded and counted per split, with raw source CSVs available for authorized audit. BPE is refitted if training exclusions change its fitting rows, so the final tokenizer can be reproduced from final `train.csv` alone. Evaluation references are never clipped.

The seed-42 pilot retained 8,633 train, 494 validation, and 2,163 test rows. Of 10,000 selected train rows, 109 were invalid or exact duplicates in the CSV, 701 failed quality filters, and 557 exceeded token limits after BPE. The duplicate index found 194 multi-pair groups, with six distinct pairs in the largest and zero cross-split groups at a 0.8 trigram Jaccard threshold. Semantic paraphrases can evade this index. Official test preparation retained 86.28% of records. Consider jointly increasing the data and model sequence limits if evaluating a higher share of the official test set.

### CPU dry-run, pilot training and resume

```bash
.venv/bin/python scripts/train.py --config configs/iitb_pilot_training.yaml \
  --model-config configs/iitb_pilot_model.yaml --data-config configs/experiment_data.yaml \
  --processed-dir "$pilot/processed" --tokenizer-dir "$pilot/tokenizer" \
  --run-name iitb-pilot-v1 --device cpu --dry-run

# Run only after selecting a compute and epoch budget.
.venv/bin/python scripts/train.py --config configs/iitb_pilot_training.yaml \
  --model-config configs/iitb_pilot_model.yaml --data-config configs/experiment_data.yaml \
  --processed-dir "$pilot/processed" --tokenizer-dir "$pilot/tokenizer" \
  --run-name iitb-pilot-v1 --device cpu --epochs 1 --planned-epochs 5 --save-every-updates 50

.venv/bin/python scripts/train.py --resume checkpoints/iitb-pilot-v1/latest.pt \
  --model-config configs/iitb_pilot_model.yaml --data-config configs/experiment_data.yaml \
  --processed-dir "$pilot/processed" --tokenizer-dir "$pilot/tokenizer" --device cpu --epochs 5 --save-every-updates 50
```

The pilot is width 128 with two encoder layers, two decoder layers, four heads, a 512-wide FFN and a 128-token positional limit. The independent `iitb_full_*.yaml` configs use width 256, four layers per stack and a 1024-wide FFN. For a 50,000-row selection, acquire a new source with `--train-samples 50000`, prepare a separate experiment and review its dry-run before selecting a compute budget. The pilot's 8,000-token BPE gives 3,994,624 trainable parameters and logits shaped `[batch, target_length, 8000]`. The measured CPU dry-run used `[8, 117, 8000]`.

The checkpoint stores state dictionaries, optimizer and scheduler counters, seed and RNG state, resolved settings and tokenizer/config/corpus fingerprints. Reuse the same processed data and tokenizer after resume. Exact continuation is tested on CPU at mid-epoch optimizer boundaries. An interruption before the old epoch-only code published its first checkpoint cannot be recovered. On the Apple M1 Pro, a one-batch MPS check passed with `deterministic_algorithms: false`. The strict setting failed because this PyTorch build lacks deterministic `embedding_dense_backward_mps`. Use CPU for strict reproduction or record the MPS override in a separate training config. Keep the environment versions, raw manifest and prepared artifacts together.

### Held-out generation and audit

```bash
.venv/bin/python scripts/evaluate_translations.py --checkpoint checkpoints/iitb-pilot-v1/best.pt \
  --processed-dir "$pilot/processed" --tokenizer-dir "$pilot/tokenizer" \
  --split val --output runs/iitb-pilot-v1/generated-validation \
  --training-metrics runs/iitb-pilot-v1/metrics.jsonl

.venv/bin/python scripts/audit_translation_experiment.py --source-dir "$pilot/source" \
  --experiment "$pilot" --checkpoint checkpoints/iitb-pilot-v1/best.pt \
  --run-dir runs/iitb-pilot-v1 --generation runs/iitb-pilot-v1/generated-validation
```

For final test evaluation after freezing the checkpoint and decoding settings, use the same evaluation command with `--split test --output runs/iitb-pilot-v1/generated-test`. Corpus chrF2 compares character n-grams and weighs recall twice as much as precision. Corpus BLEU measures token n-gram precision with a brevity penalty. The JSON report includes metric signatures, original and evaluated row counts, exclusion reasons, EOS counts, repetition and control-character flags, and generation timing.

A separate CPU smoke run used a seeded 256-row pre-length training sample and retained 238 train and 490 validation examples. It reached 30 optimizer updates after two epochs, with three warmup updates and a 3,992-token vocabulary. The run's 24 length-stratified validation generations repeated a Devanagari vowel mark, with zero EOS completions and zero BLEU for both strategies. The private audit report includes 20 varied examples. The full 10,000-selected pilot was prepared and passed its dry-run. Its training and official test evaluation are pending a selected budget and frozen model.


To prepare a separate larger run after choosing a compute budget, use the same pinned revision and retain its own source, tokenizer and checkpoints.

```bash
larger=experiments/iitb-50k-v1
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/download_iitb_dataset.py --revision "$revision" \
  --train-samples 50000 --seed 42 --output "$larger/source"
.venv/bin/python scripts/prepare_iitb_experiment.py --source-dir "$larger/source" \
  --output "$larger" --data-config configs/experiment_data.yaml
.venv/bin/python scripts/train.py --config configs/iitb_full_training.yaml \
  --model-config configs/iitb_full_model.yaml --data-config configs/experiment_data.yaml \
  --processed-dir "$larger/processed" --tokenizer-dir "$larger/tokenizer" \
  --run-name iitb-50k-v1 --device cpu --dry-run
```

### Diagnosing a pilot checkpoint

Use `scripts/diagnose_translation.py` with the checkpoint, its tokenizer and processed split directories to record ByteLevel pieces, Unicode and a raw next-token probability trace. Use `scripts/evaluate_translations.py --split val --limit 100 --stratified` with matching settings and separate 64- and 120-token output directories. `scripts/summarize_translation_diagnosis.py` checks ordered references, checkpoint identity and decoded metric consistency before comparing EOS, repetition and corpus scores. The traces and references should remain private. Successful NFC round-tripping validates known input strings and arbitrary generated IDs can still represent incomplete UTF-8 or poorly formed Hindi.

## Interview demo · constrained English to Hindi

The runnable interview project reuses the custom architecture above and trains a smaller checkpoint from random initialization on a separate, programmatically constructed conversational grammar. The unrestricted IITB runs remain useful failure analyses. The new checkpoint has **99.56% exact accuracy on 1,371 held-out grammar combinations**, but **12.28% on wholly withheld question structures**. It is a constrained technical demo.

From this directory, once the local scratch artifact is present:

```bash
bash scripts/run_interview_demo.sh
```

Open <http://127.0.0.1:8765>. The app loads the exact checkpoint and tokenizer, runs greedy or beam inference, displays validation examples, and plots genuine attention tensors, masks and losses. An out-of-scope warning accompanies inputs outside the recorded trainable grammar.

See [architecture](docs/ARCHITECTURE.md), [interview guide](docs/INTERVIEW_GUIDE.md), and [results](docs/EXPERIMENT_RESULTS.md). For a clean local rebuild, create `.venv` with Python 3.11+, install `requirements.txt`, run `scripts/prepare_interview_data.py`, then `bash scripts/train_interview_model.sh`. Run `scripts/evaluate_interview_model.py --validation-only` before launching the demo. Existing experiment directories are preserved; preparation requires a fresh destination. If startup reports a missing artifact, confirm `experiments/interview-transformer-v1` has its tokenizer, validation JSON and `checkpoints/scratch/best.pt`. If port 8765 is in use, pass `--port 8766`. The local lock snapshot and detailed commands are recorded in the experiment directory and run report.
