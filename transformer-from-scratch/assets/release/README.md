# Conversation v2 release archive

The archive contains the original epoch-11, step-1,980 checkpoint and its exact tokenizer mapping, plus the synthetic English–Hindi grammar inventory, data config, evaluation outputs and training log. The archive excludes IIT Bombay and FLORES data, model outputs on those datasets, and other historical experiments.

| Artifact | SHA-256 |
|---|---|
| Checkpoint | `628ee5b85ef534e3efc8a7a32d2a887f63df0d8eb44970989382c614a84b7aa6` |
| Tokenizer JSON | `407c3d17b5eac31061f1d378f5439e94b04f70284c1e7a329100095b49fa2a50` |
| Synthetic inventory | `18fc5e59391e198a942635eab7921aa33cf0c16a1a553a9a0fa63c8f699926fb` |

Verify the outer archive with `shasum -a 256 -c assets/release/SHA256SUMS` from the project directory. The server additionally checks tokenizer compatibility, inventory identity and checkpoint fingerprints in the two evaluation files. The original training checkpoint is retained byte for byte. Reproduction uses the [conversation generator](../../scripts/prepare_conversation_v2.py), seed 91 for training, and [training configuration](../../configs/interview_training.yaml). The separate optimizer resume workflow requires prepared split CSVs and their manifest.
