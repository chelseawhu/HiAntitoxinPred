# Quick Start

## Install

```bash
conda env create -f environment.yml
conda activate hiantitoxinpred
```

## Reproduce The Main Experiment (from the released checkpoint)

The release ships the locked main-benchmark Full Model (seed 42) at
`pretrained/best_model.pt`. First place the Zenodo artifact contents in the
repository root. Archive DOI: https://doi.org/10.5281/zenodo.21025508

```bash
wget -O HiAntitoxinPred_artifacts_v1.tar.gz \
  "https://zenodo.org/records/21025508/files/HiAntitoxinPred_artifacts_v1.tar.gz?download=1"
wget -O HiAntitoxinPred_artifacts_v1.tar.gz.sha256 \
  "https://zenodo.org/records/21025508/files/HiAntitoxinPred_artifacts_v1.tar.gz.sha256?download=1"
sha256sum -c HiAntitoxinPred_artifacts_v1.tar.gz.sha256

tar -xzf HiAntitoxinPred_artifacts_v1.tar.gz
cp -r HiAntitoxinPred_artifacts_v1/features .
cp -r HiAntitoxinPred_artifacts_v1/checkpoints .
cp -r HiAntitoxinPred_artifacts_v1/results/locked_metrics results/
```

Then reproduce the main **test** metrics with a single command:

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
the ESM2 per-residue feature file under `features/`; if the artifact package is
not available, generate the feature file with the embedding extraction workflow
below.

Generate the required feature file if needed:

```bash
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all \
  --device cuda:0 \
  --skip-completed
```

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
generate them first with the same command:

```bash
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all --device cuda:0 --skip-completed
```
