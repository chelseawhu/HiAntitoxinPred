# Quick Start

## Reproduce The Main Experiment (from the released checkpoint)

The release ships the locked main-benchmark Full Model (seed 42) at
`pretrained/best_model.pt`. Reproduce the main **test** metrics from it with a
single command:

```bash
python scripts/evaluate/evaluate_main_checkpoint.py --device cpu
```

This loads the checkpoint, scores the main validation/test split, selects the
MaxF1 threshold on validation, applies it (fixed) to the test set, and prints
the metrics next to the locked paper numbers.

Expected output (matches the locked seed_42 row):

| metric | value |
| --- | --- |
| F1 | 0.9365 |
| MCC | 0.9319 |
| precision | 0.9833 |
| recall | 0.8939 |
| confusion (TP/FP/TN/FN) | 59/1/671/7 |

It runs on CPU in a few minutes; pass `--device cuda:0` to use a GPU. It needs
the ESM2 per-residue feature file under `features/`; if it is missing, generate
it once with the embedding extraction workflow (below) or download the released
feature archive (see README).

## Predict From FASTA

```bash
python scripts/predict/predict_fasta.py \
  --fasta data/examples/example_mixed.fasta \
  --checkpoint pretrained/best_model.pt \
  --output results/example_predictions.csv \
  --device cuda:0
```

## Full Retraining (optional, reproduces the 5-seed result)

```bash
python scripts/train/run_full_model_benchmarks.py \
  --config configs/model_runs/main_full_model_5seed.json \
  --stages main \
  --device cuda:0
```

This requires the ESM2 feature files under `features/`. If they are missing,
generate them first:

```bash
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all --device cuda:0
```
