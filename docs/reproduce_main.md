# Reproduce Main

Main is a fixed split, not a new random split for each seed:

| Split | Positive | Negative | Total |
|---|---:|---:|---:|
| Train | 215 | 2150 | 2365 |
| Validation | 54 | 538 | 592 |
| Test | 67 | 672 | 739 |

`data/main/DATA_IDENTITY.json` binds the CSV, split manifest and residue-feature files. The stored arrays contain frozen ProtT5 residue embeddings, not mean-pooled vectors. The reader checks sequence SHA, lengths, labels, split membership, feature offsets, file hashes and exact cross-split sequence equality. This is not a replacement for the separate homology-based dataset construction audit.

The fixed C1 recipe is weighted BCEWithLogits, positive weight `sqrt(N_negative_train/N_positive_train)` (sqrt10 on Main), natural shuffled sampling, no label smoothing, AdamW lr1e-6/wd0.05/betas(0.9,0.95)/eps1e-7, batch16, dropout0.1, gradient norm clipping0.5 and FP32 classifier. There is no scheduler. Training drops the last incomplete batch; evaluation keeps it. Historical training clips features to [-10,10], while validation/test use raw stored features. This asymmetry is retained explicitly.

Early stopping monitors the sample-weighted validation objective loss with patience30, min_delta1e-5, max300. **Any strict decrease saves L**, even if too small to reset the early-stopping counter. Equal loss retains the earlier epoch. The stopping reason distinguishes patience from max_epochs. The selection lock is written before held-out evaluation; strict reload checks compare validation logits and classifications. Test metrics never determine the saved checkpoint.

Use the commands in the README for all seeds42–46. Read `progress.json` during training; `COMPLETE.json` means training and validation reload passed, not that a test evaluation ran. The training directory records the effective config, input identity, environment, epoch loss, raw validation predictions, L and its lock. There is no implicit resume or automatic retry after failures.

Classification metrics use logit>=0. AUROC/AP use raw logits; AP means average precision. Compute each model's metrics separately, then mean and sample SD across the five seeds. Averaging probabilities first would instead evaluate an ensemble and does not reproduce the table.

`results/reference/` contains historical metrics. Pure-tensor converted weights preserve outputs within the tested numerical tolerance. Retraining the smaller code implementation does not guarantee the old initialization stream, and CUDA/cuDNN nondeterminism also prevents a bitwise training guarantee. Record new results separately.
