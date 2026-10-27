# MAIA

Run `hiantitoxinpred maia --fasta examples/AT6247.fasta --device cuda:0 --output results/AT6247_maia` for the main-text case P9WJ87 / AT6247. By default it uses the published seed42 L model; `--seed 43` etc. uses another registered model after downloading `main-models`. Identifiers are metadata, not output paths. No region annotations enter attribution inference.

The protocol uses complete width20 windows at stride1. Inputs shorter than20 have one full-input window. For each window, U19 replaces every standard amino acid by a uniformly sampled alternative among the other19. Two fixed RNG groups1001/2001 each contribute five repeats. Counter-based SHA256 RNG keys bind operator, input sequence SHA, window start, group and repeat, independently of task order. Optional `--operator SHUFFLE_COMPOSITION` retains window composition.

For a window, first average `original_logit - perturbed_logit` across the ten repeats. Divide this window mean by its actual width, then average over windows covering each residue. `signed_score` retains both signs. `absolute_sensitivity` averages the absolute **window mean** divided by width; it is not the mean of the absolute individual perturbation effects.

The original sequence and all perturbations use the same frozen encoder and classifier padding width (default 200). GPU encoding retains the original mixed precision; output features and classifier are FP32. Do not slice full-sequence embeddings to simulate a truncated/replaced sequence: each actual altered input is encoded anew. The optional cache binds the sequence, encoder identity/precision and original encoder batch. Cache hits do not regroup the remaining inputs.

Actual single-model CSV examples for this protein are available in [examples/expected](../examples/expected/README.md). They use the default padding width 200.

Output for each input:

- `residue_scores.csv`: signed score, absolute sensitivity and window coverage.
- `window_scores.csv`: coordinates, actual width and mean logit change.
- `raw_scores.json`: original score, per-perturbation logits and deterministic mutation ledger.
- `metadata.json`: sequence, model/encoder identity and numerical protocol.

Inputs for U19 must contain the 20 standard amino acids. Prediction also accepts X/U/Z/O/B (UZOB map to X), but U19 for ambiguous residues has no registered 19-alternative definition and is rejected. Negative attribution values are not clipped or converted into positive importance.

## Paper case settings

For the two paper cases, the original classifier padding widths were 136 for AT6247 and 149 for AT121. Use these settings with classifier batch size 2:

```bash
hiantitoxinpred maia --fasta examples/AT6247.fasta --padding-width 136 --batch-size 2 --device cuda:0 --output results/AT6247
hiantitoxinpred maia --fasta examples/AT121.fasta --padding-width 149 --batch-size 2 --device cuda:0 --output results/AT121
```

New inputs use width 200 by default. Original and perturbed inputs always share the chosen width.
