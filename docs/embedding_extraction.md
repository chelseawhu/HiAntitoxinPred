# ESM2 Embedding Extraction

HiAntitoxinPred uses ESM2-1280 per-residue embeddings
(`facebook/esm2_t33_650M_UR50D`) as model input. These feature files are not
version-controlled because of their size. The simplest way to obtain the
main-benchmark features is to download the Zenodo archive
(`features/main_esm2_1280.pkl`); see `docs/data_availability.md`. This page
documents how to regenerate them from the included sequences instead.

## Regenerate from sequences

```bash
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all --device cuda:0 --skip-completed
```

The extraction writes the main-benchmark features to:

```text
features/esm2_1280/final_hybrid_suite_1to10/global_cap200_unique/esm2_per_residue_features.pkl
```

The evaluation scripts read `features/main_esm2_1280.pkl` by default, so link the
generated file to that path (or pass `--features <path>` explicitly):

```bash
ln -sf esm2_1280/final_hybrid_suite_1to10/global_cap200_unique/esm2_per_residue_features.pkl \
  features/main_esm2_1280.pkl
```

## Offline model cache

If the ESM2 model is already cached locally, run in offline mode:

```bash
HF_HOME="$(pwd)/.cache/huggingface" HF_HUB_OFFLINE=1 \
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all --device cuda:0 --skip-completed
```

## Feature record format

Each record contains:

- `id` — sample identifier
- `sequence` — normalized protein sequence
- `features` — per-residue matrix of shape `L × 1280`
- `length` — sequence length
- `feature_dim` — 1280
