# Pretrained Checkpoint

The released HiAntitoxinPred checkpoint (`best_model.pt`) is the main-benchmark
Full Model trained with seed 42. It is **not** stored in this repository because
of its size; download it from the Zenodo archive (DOI
[10.5281/zenodo.21079335](https://doi.org/10.5281/zenodo.21079335)) and place it
here as `pretrained/best_model.pt` (see `docs/data_availability.md`).

The evaluation and prediction scripts expect the checkpoint at this path by
default. Its decision threshold (0.09) and provenance are recorded in
`best_model_provenance.json`, which is kept under version control.
