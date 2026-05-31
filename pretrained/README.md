# Pretrained Checkpoints

This directory contains the optional pretrained HiAntitoxinPred checkpoint used
by the default FASTA prediction example.

The included checkpoint is the manuscript main-benchmark Full Model seed 42
checkpoint converted to a PyTorch state_dict with floating tensors stored in
float16. The model is instantiated in float32 and `load_state_dict` converts the
stored tensors as needed.

Expected filename for the default prediction example:

```text
best_model.pt
```

See `best_model_provenance.json` for the checkpoint provenance.

The repository also supports full retraining from the processed benchmark
datasets and regenerated ESM2 embeddings.
