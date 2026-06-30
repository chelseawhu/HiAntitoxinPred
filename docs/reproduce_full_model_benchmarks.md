# Reproduce Full Model Benchmarks

This repository focuses on reproducing the HiAntitoxinPred Full Model results.

The commands below **retrain** each benchmark from scratch (GPU, longer runs). To
reproduce the main-benchmark headline directly from the released checkpoints
without retraining, use `scripts/evaluate/evaluate_5seed_checkpoints.py` instead
(see `docs/quick_start.md`).

## Main Benchmark

```bash
python scripts/train/run_full_model_benchmarks.py \
  --config configs/model_runs/main_full_model_5seed.json \
  --stages main \
  --device cuda:0
```

## Strict Cross-Source Benchmarks

```bash
python scripts/train/run_full_model_benchmarks.py \
  --config configs/model_runs/strict_cross_full_model_5seed.json \
  --stages cross_phylum \
  --device cuda:0
```

## CD-HIT Benchmarks

```bash
python scripts/train/run_full_model_benchmarks.py \
  --config configs/model_runs/cdhit_full_model_5seed.json \
  --stages cdhit \
  --device cuda:0
```

## Imbalance Benchmarks

```bash
python scripts/train/run_full_model_benchmarks.py \
  --config configs/model_runs/imbalance_full_model_5seed.json \
  --stages imbalance \
  --device cuda:0
```

All benchmark configs use seeds `42-46`, single-stage training, validation
MaxF1 checkpoint selection, and validation MaxF1 threshold selection.

