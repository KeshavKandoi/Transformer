# Transformer experiment results

## Released checkpoint · conversation v2

The released model reproduced 1,079/1,088 exact validation rows, 1,621/1,642 composition rows, 2,399/2,434 lexical rows, and 330/433 withheld-structure rows with greedy decoding. The per-row counts, metric signatures and EOS fields are in `validation_evaluation.json` and `test_evaluation.json` inside the [release archive](../assets/release/README.md). Use the [evaluation commands](GETTING_STARTED.md#reproduce-evaluation) to regenerate them.

## Historical checkpoint · interview-transformer-v1

The scratch demo reached **1,365/1,371 exact translations (99.56%)** on frozen compositional combinations and **2,425/2,434 (99.63%)** on reserved lexical combinations. The lexical score concerns new combinations of known English word forms. Entire withheld question families fell to **42/342 (12.28%)** exact. The project therefore demonstrates controlled compositional translation and exposes a sizeable generalization boundary.

The original unrestricted IITB and FLORES experiments remain available locally. The v1 domain model was evaluated on 24 fixed FLORES and 24 fixed IITB probes plus 20 prior auxiliary conversations. It produced zero exact matches on the FLORES and IITB slices, and one on the auxiliary set. It should be presented as an architecture and controlled translation demonstration.

For the full counts, corpus recipe, checkpoints, examples, scores, and limitations, see the local run report at `runs/interview-transformer-v1/REPORT.md`.
