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
- Processed benchmark datasets and split manifests.
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

## Reproduce The Main Experiment

The release ships the locked main-benchmark Full Model (seed 42) at
`pretrained/best_model.pt`. The fastest route is to download the Zenodo artifact
package, place its feature files in the repository, and run the evaluation
script.

Install the environment:

```bash
conda env create -f environment.yml
conda activate hiantitoxinpred
```

Download and extract `HiAntitoxinPred_artifacts_v1.tar.gz` in the repository
root, then place the archived paths into the repository:

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

Then reproduce the main **test** metrics from the released weights:

```bash
python scripts/evaluate/evaluate_main_checkpoint.py --device cpu
```

The script selects the MaxF1 threshold on validation, fixes it on the test set,
and prints your numbers next to the locked paper values. Expected (single
seed_42 checkpoint):

| F1 | MCC | precision | recall | TP/FP/TN/FN |
| --- | --- | --- | --- | --- |
| 0.9365 | 0.9319 | 0.9833 | 0.8939 | 59/1/671/7 |

Runtime is a few minutes on CPU (no GPU required); add `--device cuda:0` to use
a GPU.

If the feature file is missing, generate it once from the included benchmark
sequences:

```bash
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all --device cuda:0 --skip-completed
```

Full 5-seed retraining (seeds 42–46) reproduces the headline result
**F1 0.9206 ± 0.0161, MCC 0.9145 ± 0.0175** (see the retraining command in Quick
Start and `docs/reproduce_full_model_benchmarks.md`).

### Large files (ESM2 features and full checkpoints)

The ESM2 per-residue feature pickle and the full 5-seed checkpoint collection
are too large for GitHub. They are packaged as:

```text
HiAntitoxinPred_artifacts_v1
```

After downloading and extracting the artifact package, copy its contents into
the repository root (the archived paths already match this repository layout).
The package includes the main-benchmark ESM2 feature file, full 5-seed
main-benchmark checkpoints, locked metrics, and checkpoint provenance.

Download from Zenodo:

```bash
wget -O HiAntitoxinPred_artifacts_v1.tar.gz \
  "https://zenodo.org/records/21025508/files/HiAntitoxinPred_artifacts_v1.tar.gz?download=1"
wget -O HiAntitoxinPred_artifacts_v1.tar.gz.sha256 \
  "https://zenodo.org/records/21025508/files/HiAntitoxinPred_artifacts_v1.tar.gz.sha256?download=1"
sha256sum -c HiAntitoxinPred_artifacts_v1.tar.gz.sha256
```

If the artifact package is not available, regenerate the ESM2 features from the
included benchmark sequences:

```bash
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all --device cuda:0
```

Zenodo DOI: https://doi.org/10.5281/zenodo.21025508

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
