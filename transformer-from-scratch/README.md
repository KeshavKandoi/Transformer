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
