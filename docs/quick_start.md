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

## Reproduce The Headline 5-seed Result (from the released checkpoints)

The headline metric is the mean over five training seeds (42–46). With the
Zenodo `checkpoints/` copied into the repository (above), reproduce it from the
released weights without retraining:

```bash
python scripts/evaluate/evaluate_5seed_checkpoints.py --device cpu
```

For each seed it selects the MaxF1 threshold on validation, fixes it on the test
set, then aggregates mean ± sample std (ddof=1) across seeds.

Expected output:

| n seeds | F1 | MCC | precision | recall | accuracy |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.9206 ± 0.0161 | 0.9145 ± 0.0175 | 0.9667 ± 0.0166 | 0.8788 ± 0.0186 | 0.9864 ± 0.0027 |

## Predict From FASTA

```bash
python scripts/predict/predict_fasta.py \
  --fasta data/examples/example_mixed.fasta \
  --checkpoint pretrained/best_model.pt \
  --output results/example_predictions.csv \
  --device cuda:0
```

## Full Retraining (optional, trains the 5 seeds from scratch)

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
