# ESM2 Embedding Extraction

HiAntitoxinPred uses ESM2-1280 per-residue embeddings as model input. The
repository does not include large precomputed embedding pickle files.

## When this step is needed

Run this step if the following file is missing:

```text
features/esm2_1280/final_hybrid_suite_1to10/global_cap200_unique/esm2_per_residue_features.pkl
```

This feature file is required for both:

- evaluating the released checkpoint with `scripts/evaluate/evaluate_main_checkpoint.py`;
- retraining the Full Model on the main benchmark.

If you do not want to regenerate the ESM2 features, download the archived
artifact package from Zenodo instead:
https://doi.org/10.5281/zenodo.21025508

```bash
wget -O HiAntitoxinPred_artifacts_v1.tar.gz \
  "https://zenodo.org/records/21025508/files/HiAntitoxinPred_artifacts_v1.tar.gz?download=1"
wget -O HiAntitoxinPred_artifacts_v1.tar.gz.sha256 \
  "https://zenodo.org/records/21025508/files/HiAntitoxinPred_artifacts_v1.tar.gz.sha256?download=1"
sha256sum -c HiAntitoxinPred_artifacts_v1.tar.gz.sha256
tar -xzf HiAntitoxinPred_artifacts_v1.tar.gz
cp -r HiAntitoxinPred_artifacts_v1/features .
```

## Generate ESM2 features

Generate embeddings from the included processed benchmark datasets:

```bash
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all \
  --device cuda:0 \
  --skip-completed
```

The default model reference is:

```text
facebook/esm2_t33_650M_UR50D
```

The key output feature file is:

```text
features/esm2_1280/final_hybrid_suite_1to10/global_cap200_unique/esm2_per_residue_features.pkl
```

After it is generated, reproduce the released main checkpoint:

```bash
python scripts/evaluate/evaluate_main_checkpoint.py --device cpu
```

For full retraining, use the same feature file and run:

```bash
python scripts/train/run_full_model_benchmarks.py \
  --config configs/model_runs/main_full_model_5seed.json \
  --stages main \
  --device cuda:0
```

## Offline model cache

If the ESM2 model is already cached locally, run in offline mode, for example:

```bash
HF_HOME="$(pwd)/.cache/huggingface" HF_HUB_OFFLINE=1 \
python scripts/feature_extraction/extract_esm2_embeddings.py \
  --config configs/feature_extraction/esm2_1280_final_benchmarks.json \
  --stages all \
  --device cuda:0 \
  --skip-completed
```

Each feature record contains:

- `id`: sample identifier.
- `sequence`: normalized protein sequence.
- `features`: per-residue matrix with shape `L x 1280`.
- `length`: sequence length.
- `feature_dim`: 1280.
