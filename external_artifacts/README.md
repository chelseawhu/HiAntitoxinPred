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

The archived release artifact is available on Zenodo:

```text
https://doi.org/10.5281/zenodo.21025508
```

After downloading `HiAntitoxinPred_artifacts_v1.tar.gz`, extract it in the
repository root and copy the archived paths into place as described in
`README.md` and `docs/quick_start.md`.
