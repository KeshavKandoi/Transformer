# Transformer launch copy

## GitHub description

Scratch-trained PyTorch encoder–decoder for English-to-Hindi translation, with reproducible synthetic evaluation and interactive attention inspection.

## GitHub topics

`machine-translation`, `transformer`, `pytorch`, `encoder-decoder`, `attention-mechanism`, `english-hindi`, `nlp`, `beam-search`, `reproducible-research`

## Portfolio

I built a Transformer to trace English-to-Hindi translation from byte-level BPE through custom attention, scratch training, and autoregressive decoding. The local interface compares greedy and beam predictions and exposes real attention tensors. Frozen tests separate familiar grammar combinations from withheld structures.

## X post

I built an English-to-Hindi Transformer in PyTorch with AI assistance. I trained the weights from scratch on synthetic dialogue. The local demo compares greedy and beam decoding and lets you inspect real attention maps.

https://github.com/KeshavKandoi/Transformer

## X thread

1. With AI assistance, I built a Transformer to follow English-to-Hindi translation from source tokens to generated output. The repo includes the exact checkpoint, tokenizer, and local demo. https://github.com/KeshavKandoi/Transformer
2. I implemented the encoder, decoder, scaled dot-product attention, masks, and vocabulary projection in PyTorch. The 1.13M-parameter checkpoint started from random weights.
3. Early corpus experiments exposed a gap between source conditioning and translation quality. Changing the source affected loss even when generation dropped names or changed relationships.
4. On synthetic tests, greedy generation matched 1,621/1,642 held-out combinations and 330/433 withheld structures. The gap helps show what the model learned within its grammar.
5. In the demo, beam search fixes a malformed greedy help request. Both decoders still confuse “your” and “their” in one name question. You can inspect live attention for either strategy.

## LinkedIn announcement

With AI assistance, I built a custom PyTorch encoder–decoder for controlled English-to-Hindi translation. I implemented attention, causal and padding masks, teacher-forced training, and autoregressive decoding, then built a local interface to compare greedy and beam generation and inspect real attention tensors.

The most informative result was the difference between evaluation splits. The same checkpoint matched 1,621 of 1,642 synthetic composition examples and 330 of 433 examples from withheld structures. The repository includes the checkpoint, split recipe, and failure analysis so the results can be inspected and reproduced.

https://github.com/KeshavKandoi/Transformer
