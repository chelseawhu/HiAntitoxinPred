# External Artifacts

This directory is a placeholder for local copies of large generated files that
are not version-controlled (ESM2 feature pickles, checkpoint collections, and
similar outputs).

The released model weights and the main-benchmark ESM2-1280 features are archived
on Zenodo (DOI
[10.5281/zenodo.21079335](https://doi.org/10.5281/zenodo.21079335)); see
`docs/data_availability.md` for download and verification instructions. The ESM2
features can alternatively be regenerated from the included benchmark sequences
with `scripts/feature_extraction/extract_esm2_embeddings.py` (see
`docs/embedding_extraction.md`).
