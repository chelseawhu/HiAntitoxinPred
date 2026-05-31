# Quick Start

## Predict From FASTA

Place a trained checkpoint at:

```text
pretrained/main_seed42_best_model.pt
```

Then run:

```bash
python scripts/predict/predict_fasta.py \
  --fasta data/examples/example_mixed.fasta \
  --checkpoint pretrained/main_seed42_best_model.pt \
  --output results/example_predictions.csv \
  --device cuda:0
```

If no checkpoint is provided with the release, train the Full Model first or
use your own checkpoint produced by `scripts/train/run_full_model_benchmarks.py`.

## Reproduce The Main Benchmark

```bash
python scripts/train/run_full_model_benchmarks.py \
  --config configs/model_runs/main_full_model_5seed.json \
  --stages preflight \
  --device cuda:0
```

The preflight stage requires ESM2 feature files under `features/`. If they are
missing, generate them first with the embedding extraction workflow.

