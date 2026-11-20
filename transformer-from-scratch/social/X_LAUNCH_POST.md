# Launch copy

## Main X post

I built an English → Hindi demo around a PyTorch Transformer I trained from random initialization. The app shows real decoder outputs, attention heads, and failure cases. The evaluation is on controlled conversational grammar.

https://github.com/KeshavKandoi/Transformer

## Short alternative

I built and scratch-trained a PyTorch encoder–decoder for controlled English → Hindi translation. The local demo lets you compare greedy and beam outputs and inspect real attention maps.

https://github.com/KeshavKandoi/Transformer

## Technical thread

1. I built a small English → Hindi Transformer to understand the full path from bilingual tokenization to autoregressive decoding. The local demo and reproduction steps are here. https://github.com/KeshavKandoi/Transformer
2. I implemented multi-head attention, sinusoidal positions, padding and causal masks, encoder–decoder cross-attention, and the output projection in PyTorch. The 1.13M-parameter model started from random weights.
3. The first broad-data runs taught me to separate source conditioning from translation adequacy. Changing the English changed likelihood, but names, numbers and parts of a sentence’s meaning still disappeared in generation.
4. I moved to synthetic conversational grammar to test specific behaviors. Greedy decoding matched 1,079/1,088 synthetic validation references and 1,621/1,642 held-out combinations. On entire withheld structures it matched 330/433.
5. The demo shows both successes and failures. For “Can you help me?” greedy generated malformed Hindi and beam search produced the intended request. Both strategies still confuse “your” with “their” in one name question.
6. You can inspect real encoder, decoder and cross-attention weights by layer and head. The repository includes a frozen checkpoint, corpus recipe, evaluation scripts and a technical report.

## GitHub description

Scratch-trained PyTorch encoder–decoder for controlled English-to-Hindi translation, with live local inference, attention inspection and reproducible synthetic evaluation.

## LinkedIn announcement

I built a PyTorch encoder–decoder Transformer for English-to-Hindi translation and trained it from random initialization. The project includes multi-head attention, causal and padding masks, bilingual BPE, resumable training, and a local browser demo that shows actual predictions and attention tensors.

The broad-data work taught me that a decoder can use the English input and still omit important details. I built a controlled conversational benchmark to test those failures more directly. The final checkpoint matched 1,621 of 1,642 held-out combinations, with a separate result of 330 of 433 on withheld structures. Both are synthetic evaluations.

The repository includes reproducible commands, real failure examples, and a technical report. https://github.com/KeshavKandoi/Transformer

## Portfolio description

I built and scratch-trained a 1.13M-parameter PyTorch Transformer for controlled English-to-Hindi translation. I designed reproducible synthetic evaluations and a local app for greedy and beam decoding, real attention matrices, and training diagnostics. https://github.com/KeshavKandoi/Transformer
