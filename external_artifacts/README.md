# External Artifacts

Large generated artifacts are intentionally excluded from GitHub.

Use this directory for local copies of generated files such as:

- ESM2 per-residue embedding pickle files.
- Full benchmark checkpoint collections.
- Large intermediate baseline outputs.
- Optional full data-rebuild intermediates, for example
  `hybrid_negative_pool_before_cdhit.csv`. This fallback pool is not required
  to use the processed benchmark datasets already included in the repository.

These files can be regenerated from the scripts in this repository.
