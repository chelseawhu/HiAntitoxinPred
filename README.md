# HiAntitoxinPred

HiAntitoxinPred is a protein language model based framework for antitoxin
prediction. This repository provides the manuscript-facing Full Model workflow,
processed benchmark datasets, figure source data, and lightweight examples.

## Current Release Scope

- Predict antitoxin candidates from protein sequences.
- Extract ESM2-1280 per-residue embeddings from benchmark sequences.
- Train and evaluate the Full Model on the final manuscript benchmarks.
- Reproduce the manuscript-facing Main, strict cross-source, CD-HIT, and
  imbalance benchmark summaries.

## Included In GitHub

- Final model and training code snapshots.
- Full Model benchmark configs.
- Processed benchmark datasets, split manifests, and QC files.
- Locked manuscript-facing metric summaries.
- Figure/table source data and generation scripts for manuscript figures.

## Not Included In GitHub

- Precomputed ESM2 feature pickle files.
- Full 5-seed model checkpoint collections.
- Runtime logs and exploratory diagnostic experiments.
- Large baseline intermediate files.
- Optional large data-rebuild fallback pools.

Users can regenerate ESM2 embeddings with the provided extraction script and
retrain the Full Model from the processed benchmark datasets.

## Quick Start

Install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate hiantitoxinpred
```

Extract ESM2 embeddings for the included benchmark datasets:

```bash
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all \
  --device cuda:0
```

Run the Full Model main benchmark:

```bash
python scripts/train/run_full_model_benchmarks.py \
  --config configs/model_runs/main_full_model_5seed.json \
  --stages main \
  --device cuda:0
```

Run FASTA prediction with a trained checkpoint:

```bash
python scripts/predict/predict_fasta.py \
  --fasta data/examples/example_mixed.fasta \
  --checkpoint pretrained/best_model.pt \
  --output predictions.csv \
  --device cuda:0
```

Regenerate manuscript figures from the included source data:

```bash
python scripts/figures/plot_figure3.py
python scripts/figures/plot_figure4.py --method-set main_table
python scripts/figures/plot_figure5.py
```

Run lightweight release checks:

```bash
python -m unittest discover -s tests
```

See `docs/` for installation, embedding extraction, prediction, and benchmark
reproduction notes.
