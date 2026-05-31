# ESM2 Embedding Extraction

HiAntitoxinPred uses ESM2-1280 per-residue embeddings as model input. The
repository does not include large precomputed embedding pickle files.

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

The output feature file is expected at:

```text
features/esm2_1280/final_hybrid_suite_1to10/global_cap200_unique/esm2_per_residue_features.pkl
```

Each feature record contains:

- `id`: sample identifier.
- `sequence`: normalized protein sequence.
- `features`: per-residue matrix with shape `L x 1280`.
- `length`: sequence length.
- `feature_dim`: 1280.

