# HiAntitoxinPred

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21079335.svg)](https://doi.org/10.5281/zenodo.21079335)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

HiAntitoxinPred is a protein language model framework for predicting bacterial
antitoxins directly from amino-acid sequence. It encodes each protein with
ESM2-1280 (`facebook/esm2_t33_650M_UR50D`) per-residue embeddings and feeds them
to a multi-branch classifier that combines a Transformer encoder, multi-scale
convolutional motif branches, gated feature fusion, and top-k motif pooling.

This repository accompanies the HiAntitoxinPred manuscript (under review) and
provides the model and training code, the processed benchmark datasets, and
one-command reproduction of the main results from released checkpoints.

## Main results

The primary benchmark is a 1:10 class-imbalanced, length-stratified split of
antitoxin (positive) and non-antitoxin (negative) proteins capped at 200 residues
(train / validation / test = 2365 / 593 / 738; 215 / 55 / 66 positives). The
decision threshold is selected by maximising F1 on the validation split and then
fixed on the test split.

| Setting | F1 | MCC | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Released checkpoint (seed 42) | 0.9365 | 0.9319 | 0.9833 | 0.8939 | 0.9892 |
| Five seeds (42–46), mean ± sd | 0.9206 ± 0.0161 | 0.9145 ± 0.0175 | 0.9667 ± 0.0166 | 0.8788 ± 0.0186 | 0.9864 ± 0.0027 |

Both rows are reproduced from the released weights in a few minutes on CPU; see
[Reproducing the main results](#reproducing-the-main-results).

## Repository structure

```
HiAntitoxinPred/
├── src/hiantitoxinpred/      Model, training loop, dataset, and metrics
├── scripts/
│   ├── evaluate/             Checkpoint evaluation (single-seed and 5-seed)
│   ├── predict/              Antitoxin prediction from a FASTA file
│   ├── feature_extraction/   ESM2-1280 per-residue embedding extraction
│   └── train/                Full Model benchmark training
├── configs/                  Feature-extraction and model-run configurations
├── data/
│   ├── benchmark_datasets/   Processed sequences (main, CD-HIT, imbalance, cross-source)
│   └── manifests/            Train / validation / test split manifests
├── results/locked_metrics/   Locked manuscript metric summaries
├── pretrained/               Released-checkpoint provenance (weights via Zenodo)
├── docs/                     Installation, reproduction, and dataset documentation
└── tests/                    Unit tests (imports, metrics, release paths)
```

## Installation

```bash
conda env create -f environment.yml
conda activate hiantitoxinpred
```

The pinned, tested stack (Python 3.11, PyTorch 2.12, Transformers 5.12) is
documented in `environment.yml`. A GPU is optional: evaluation runs on CPU.

## Data and model availability

The source code, processed benchmark datasets, split manifests, configuration
files, and locked metric summaries are version-controlled in this repository.

The trained model weights and the ESM2-1280 feature files are too large for
GitHub and are archived on Zenodo under
[doi:10.5281/zenodo.21079335](https://doi.org/10.5281/zenodo.21079335). Download
and unpack them into the repository before running the evaluation scripts:

```bash
wget -O HiAntitoxinPred_artifacts_v2.tar.gz \
  "https://zenodo.org/records/21079335/files/HiAntitoxinPred_artifacts_v2.tar.gz?download=1"
wget -O HiAntitoxinPred_artifacts_v2.tar.gz.sha256 \
  "https://zenodo.org/records/21079335/files/HiAntitoxinPred_artifacts_v2.tar.gz.sha256?download=1"
sha256sum -c HiAntitoxinPred_artifacts_v2.tar.gz.sha256

tar -xzf HiAntitoxinPred_artifacts_v2.tar.gz
mkdir -p features
cp -r HiAntitoxinPred_artifacts_v2/checkpoints .
cp    HiAntitoxinPred_artifacts_v2/features/main_esm2_1280.pkl features/
cp    HiAntitoxinPred_artifacts_v2/best_model.pt               pretrained/
```

The archive contains the released seed-42 checkpoint (`best_model.pt`), the full
five-seed checkpoint set (`checkpoints/main/seed_42…46/`), and the main-benchmark
ESM2-1280 features (`features/main_esm2_1280.pkl`). See `docs/data_availability.md`
for details and `docs/dataset_construction.md` for how the datasets were built.

## Reproducing the main results

With the artifacts in place (above) and the environment active:

**Released checkpoint (seed 42).** Selects the validation-MaxF1 threshold, fixes
it on the test set, and prints the metrics next to the locked manuscript values.

```bash
python scripts/evaluate/evaluate_main_checkpoint.py --device cpu
```

Expected test metrics: F1 0.9365, MCC 0.9319, precision 0.9833, recall 0.8939,
confusion (TP/FP/TN/FN) 59/1/671/7.

**Five-seed headline.** Evaluates all five released checkpoints under the same
protocol and aggregates the mean ± sample standard deviation (ddof = 1).

```bash
python scripts/evaluate/evaluate_5seed_checkpoints.py --device cpu
```

Expected: F1 0.9206 ± 0.0161, MCC 0.9145 ± 0.0175.

Add `--device cuda:0` to use a GPU. To retrain all five seeds from scratch
instead of evaluating the released weights, see
`docs/reproduce_full_model_benchmarks.md`.

## Predicting antitoxins from a FASTA file

```bash
python scripts/predict/predict_fasta.py \
  --fasta data/examples/example_mixed.fasta \
  --checkpoint pretrained/best_model.pt \
  --output predictions.csv \
  --device cpu
```

When `--threshold` is omitted, the checkpoint's calibrated decision threshold
(0.09 for the released model; see `pretrained/best_model_provenance.json`) is
applied automatically and recorded in the output.

## Tests

```bash
python -m unittest discover -s tests
```

These checks (package imports, metric definitions, release-path hygiene) do not
require the large artifacts.

## Citation

If you use HiAntitoxinPred, please cite the manuscript (under review) and the
archived artifacts:

> Dang, Y. *HiAntitoxinPred artifacts* (v2). Zenodo. https://doi.org/10.5281/zenodo.21079335

## License

Released under the MIT License; see [LICENSE](LICENSE).
