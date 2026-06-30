# Data Availability

This release includes processed benchmark datasets, split manifests,
figure/table source data, and locked metric summaries used for the
manuscript-facing Full Model analyses.

Large generated artifacts are intentionally excluded from GitHub:

- ESM2 per-residue embedding pickle files.
- Full checkpoint collections.
- Runtime logs.
- Optional large intermediate pools used during dataset construction.

The model inputs can be regenerated from the included processed benchmark
datasets with the provided ESM2 extraction script.

## Large artifact package

Large generated artifacts are archived as:

```text
HiAntitoxinPred_artifacts_v1
```

It contains:

- the comprehensive main-benchmark ESM2 feature pickle;
- the full 5-seed main-benchmark checkpoint collection;
- locked metric summaries;
- released checkpoint provenance.

Archive DOI: https://doi.org/10.5281/zenodo.21025508

After downloading the archive, extract it and copy the archived paths into the
repository root before running the released checkpoint evaluation:

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
python scripts/evaluate/evaluate_main_checkpoint.py --device cpu
```
