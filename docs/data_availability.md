# Data and Model Availability

## In this repository

This repository version-controls all source code together with the processed
benchmark datasets, the train/validation/test split manifests, the
feature-extraction and model-run configurations, and the locked manuscript metric
summaries. The benchmark suites live under `data/benchmark_datasets/` (main,
CD-HIT redundancy, class-imbalance, and cross-source) with their splits in
`data/manifests/`. How the datasets were assembled is described in
`docs/dataset_construction.md`.

## Archived on Zenodo

The trained model weights and the ESM2-1280 feature files are too large for
GitHub and are archived on Zenodo:

**DOI: [10.5281/zenodo.21079335](https://doi.org/10.5281/zenodo.21079335)**

The archive `HiAntitoxinPred_artifacts_v2.tar.gz` contains:

- `best_model.pt` — the released main-benchmark checkpoint (seed 42);
- `checkpoints/main/seed_42…46/` — the full five-seed checkpoint collection;
- `features/main_esm2_1280.pkl` — the main-benchmark ESM2-1280 per-residue features.

Download, verify, and unpack it into the repository:

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

The ESM2-1280 features can alternatively be regenerated from the included
benchmark sequences with `scripts/feature_extraction/extract_esm2_embeddings.py`
(see `docs/embedding_extraction.md`).
