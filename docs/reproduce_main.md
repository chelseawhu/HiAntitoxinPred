# Reproduce Main

Main is a fixed split, not a new random split for each seed:

| Split | Positive | Negative | Total |
|---|---:|---:|---:|
| Train | 215 | 2150 | 2365 |
| Validation | 54 | 538 | 592 |
| Test | 67 | 672 | 739 |

`data/main/DATA_IDENTITY.json` binds the CSV and split manifest and retains the original feature hashes for historical-cache identification. The public workflow generates frozen ProtT5 residue embeddings locally. The reader checks sequence SHA, lengths, labels, split membership, feature offsets, file hashes and exact cross-split sequence equality. This is not a replacement for the separate homology-based dataset construction audit.

The fixed C1 recipe is weighted BCEWithLogits, positive weight `sqrt(N_negative_train/N_positive_train)` (sqrt10 on Main), natural shuffled sampling, no label smoothing, AdamW lr1e-6/wd0.05/betas(0.9,0.95)/eps1e-7, batch16, dropout0.1, gradient norm clipping0.5 and FP32 classifier. There is no scheduler. Training drops the last incomplete batch; evaluation keeps it. Historical training clips features to [-10,10], while validation/test use raw stored features. This asymmetry is retained explicitly.

Early stopping monitors the sample-weighted validation objective loss with patience30, min_delta1e-5, max300. **Any strict decrease saves L**, even if too small to reset the early-stopping counter. Equal loss retains the earlier epoch. The stopping reason distinguishes patience from max_epochs. The selection lock is written before held-out evaluation; strict reload checks compare validation logits and classifications. Test metrics never determine the saved checkpoint.

## Commands

```bash
hiantitoxinpred download --assets encoder
hiantitoxinpred extract-features --device cuda:0
# Train and evaluate a new model using the generated features:
hiantitoxinpred train --config configs/main.json --seed 42 --device cuda:0 --output runs/seed42
hiantitoxinpred evaluate --checkpoint runs/seed42/L.pt --split test --device cuda:0 --output results/retrained_seed42
# Or evaluate a supplied model on the generated features:
hiantitoxinpred download --assets main-models
hiantitoxinpred evaluate --models assets/models --seed 42 --split test --device cuda:0 --output results/supplied_seed42
```

Run feature extraction once and reuse its output for all seeds. It uses the pinned encoder files and cap200 sequences, maps U/Z/O/B to X only for encoding, removes special/padding tokens, and stores FP32 arrays. Encoding is frozen and does not fit on any split. The order is train, validation, then test, preserving each manifest list. The default encoder batch size is two. CPU extraction is supported with `--device cpu`; it uses FP32 encoder parameters and is slower than GPU encoding.

The default output `assets/main_features/` contains `per_residue.npy`, `index.json`, `FEATURES.json`, `ENCODER_IDENTITY.json` and `EXTRACTION.json`. The last file marks successful completion and records the input hashes, encoder identity and batch size. Training and evaluation validate these records automatically. Output directories are never overwritten; interrupted output is retained and rejected as incomplete. Use a new `--output` directory for another extraction, and pass that directory to training/evaluation with `--features-dir`. Do not replace a training run's features before evaluating its checkpoint.

This locally generated cache occupies approximately 1.58GB. It is not distributed as a Zenodo attachment. The original reference hashes are not imposed on regenerated arrays: numerical differences from hardware, encoder precision, software or batch geometry can change predictions and training trajectories. Evaluating a supplied checkpoint with fresh features is a new evaluation, not a claim of exact replay of the archived paper scores. The original registered cache remains readable when available.

Repeat with seeds 43–46 for five runs. Read `progress.json` during training; `COMPLETE.json` means training and validation reload passed, not that a test evaluation ran. The training directory records the effective config, input identity, environment, epoch loss, raw validation predictions, L and its lock. There is no implicit resume or automatic retry after failures.

Classification metrics use logit>=0. AUROC/AP use raw logits; AP means average precision. Compute each model's metrics separately, then mean and sample SD across the five seeds. Averaging probabilities first would instead evaluate an ensemble and does not reproduce the table.

`results/reference/` contains historical metrics. Pure-tensor converted weights preserve outputs within the tested numerical tolerance when the same features and input geometry are used. Retraining the smaller code implementation does not guarantee the old initialization stream, and CUDA/cuDNN nondeterminism also prevents a bitwise training guarantee. Record new results separately.
