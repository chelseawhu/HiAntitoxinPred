# Data Availability

This release includes processed benchmark datasets, split manifests, QC files,
and locked metric summaries used for the manuscript-facing Full Model analyses.

Large generated artifacts are intentionally excluded from GitHub:

- ESM2 per-residue embedding pickle files.
- Full checkpoint collections.
- Runtime logs.
- Optional large intermediate pools for complete dataset reconstruction, such
  as `hybrid_negative_pool_before_cdhit.csv`.

These artifacts can be regenerated with the included scripts and configs.
