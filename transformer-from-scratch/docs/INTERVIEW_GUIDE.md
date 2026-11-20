# Interview guide

## 30 seconds

I implemented an encoder–decoder Transformer in PyTorch and used it to study English-to-Hindi translation. My early general-domain checkpoints learned to use the English input but often lost meaning. For this demo I generated a controlled conversational corpus, separated unseen compositions from unseen structures, and trained the same architecture from scratch. The interface runs the checkpoint locally and can display attention, masks and validation results from the actual model.

## 90 seconds

The encoder sees English subword IDs and uses bidirectional self-attention with a padding mask. The decoder attends to its own prefix through a causal mask and to the English representations through cross-attention. During training, I shift the reference by one token, then compute cross-entropy against the next reference token at every non-padding position. At inference, the model starts at BOS and generates one token at a time until EOS or a length limit. A useful distinction is that teacher-forced loss and free-running adequacy measure different things. That difference became obvious on IITB and FLORES, where the English affects likelihood without producing reliable Hindi. The controlled demo makes source–meaning tests easier to interpret. Pronouns, quantities, verbs and whole constructions have dedicated tests, and the demo gallery is drawn from validation.

## Three-minute walkthrough

1. Trace one input through NFC normalization and joint byte-level BPE. Explain subword units, shared vocabulary and the BOS/EOS/PAD roles. Show the backend vocabulary spellings in the inspector.
2. Describe the embedding lookup and the fixed sine/cosine positional vectors. The attention operator itself has no sequential position signal.
3. Walk through Q, K and V. An input vector makes a query, keys advertise which content can match, and values carry the information mixed into the output. Compute QKᵀ divided by √dₖ, apply the mask, softmax over the key axis, then multiply by V. Show the split from model width into four per-head representations and the final concatenation and projection.
4. Show the encoder’s residual, layer norm and position-wise feed-forward blocks. Then show that decoder self-attention excludes future columns and that cross-attention queries the English memory. The masks are boolean and true means allowed.
5. Demonstrate an output, switch between greedy and beam, and inspect the actual tensor for one head. Decoder visualization recomputes the selected generated prefix with dropout disabled. Distinct heads need not provide human-readable explanations.
6. Close with the experiment design. Explain the checksum of the selected checkpoint, disjoint normalized English/Hindi strings, grouped variants, test splits, and the difference between complete agreement and surface slot checks. Call out the numeric and unseen-structure failures that still remain.

## Ten questions to be ready for

1. **Why build it from scratch?** To test the mathematical components independently, tie behavior to tensors I can inspect, and understand whether failure comes from architecture, data or evaluation.
2. **Why have two stacks?** The encoder can use the whole source; the decoder needs a growing causal target prefix plus the source memory.
3. **Why divide by √dₖ?** Dot-product variance typically grows with the key dimension, and large logits can saturate the softmax.
4. **Why multiple heads?** Each head gets different projected subspaces. Multiple weighted mixtures can be computed in parallel before concatenation and projection.
5. **How do padding and causal masks differ?** Padding identifies invalid key positions; causality excludes keys after each query. Cross-attention usually keeps valid source keys visible at every target step.
6. **What is teacher forcing?** The decoder receives the previous gold target token during training. Next-token targets are shifted, with BOS in inputs and EOS in labels.
7. **How are weights updated?** Cross-entropy compares projected logits to the next target IDs. Padding labels are excluded. Backpropagation computes gradients, gradient clipping controls their norm, and AdamW updates the trainable parameters.
8. **How do decoding strategies differ?** Greedy selects the highest-scoring next token, while beam keeps several prefixes and ranks by sequence score. A higher probability can still produce an incorrect translation.
9. **Why did unrestricted translation fail?** Source-swap diagnostics show the model uses English, but previous checkpoints frequently lost numbers, names and meaning during generation. Corpus noise and limited scale are compatible explanations, and the experiments did not isolate them.
10. **What is the next improvement?** Add independently reviewed, natural sentence variants and a larger human-checked restricted-domain test. Measure whether the model understands paraphrases and distinct roles before expanding languages or size.

## Five debugging questions

1. **How would you catch target leakage?** Change future decoder input tokens and assert earlier logits stay unchanged in evaluation mode.
2. **How do you detect a bad head reshape?** Check per-head shapes and compare values with a scalar or single-head reference. Recombine only after attention.
3. **Why can validation loss fall while generation fails?** Gold prefix conditioning prevents errors from compounding. Free decoding measures how the model behaves under its own generated context.
4. **How do you debug a blank result?** Check generated token IDs for immediate EOS, confirm tokenizer special IDs and checkpoint hashes, then compare next-token logits and exact reference shift.
5. **What proves resume is correct?** Compare with a deterministic uninterrupted run across the same optimizer boundary, including batch position, parameters, optimizer and scheduler state, RNG and metrics.

## Limitations to say out loud

The data is programmatically constructed and has no independent bilingual review. It uses one controlled wording per pair and fixed Hindi gender/address conventions. A high exact rate within those templates is evidence about the measured combinations, not general translation. One input can have multiple valid Hindi translations outside this benchmark. The lexical split tests slot recombination, with unseen English word forms separately counted. The withheld question families and general-domain probes expose failures beyond the training grammar. Attention matrices are real model diagnostics and can suggest patterns, but they do not reconstruct reasoning.

## Five experiment and release questions

1. **Why do the stacks have positional encodings?** Attention alone does not represent order. Fixed sine and cosine channels provide positions alongside the learned token embeddings.
2. **When would you choose an encoder-only or decoder-only model?** An encoder-only model supports bidirectional representations for tasks such as classification. A decoder-only model conditions generation on a causal prefix. The encoder–decoder separates a fully visible English source from the causal Hindi target.
3. **What did composition testing measure?** Greedy decoding matched 1,621 of 1,642 withheld combinations of known grammar. The separately withheld structure split matched 330 of 433. These are synthetic references with one accepted surface form.
4. **Can an attention head explain a translation error?** It can identify where one query placed mass in one layer, but it does not measure the full effect of a token through values, residual paths, and later layers.
5. **How do you reproduce the result?** Check archive hashes, load the checkpoint and paired tokenizer, rerun greedy evaluation on the frozen groups, and compare each output and its EOS flag. Keep the test results separate from checkpoint selection.
