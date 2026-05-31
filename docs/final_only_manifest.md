# HiAntitoxinPred release Final-Only Manifest

Generated: 2026-05-18

This document separates the final manuscript-facing content from exploratory, diagnostic, or superseded runs inside ``. Use this as the whitelist when writing the manuscript, drawing figures, exporting tables, and preparing the GitHub release.

## Source Of Truth

Use this locked result file as the authoritative result record:

- `notes/final_experiment_results_locked.md`

Use this writing inventory only as a convenience summary:

- `paper_workspace/manuscript_inputs/current_results_inventory_20260518.md`

If another table or result disagrees with `final_experiment_results_locked.md`, treat the locked file as authoritative and investigate before using the number.

## Final Dataset Line

Final protocol:

- Negative design: hybrid old-negative short negatives plus Swiss-Prot reviewed 101-200 aa functional negatives.
- Sequence policy: cap200.
- Main ratio: 1:10.
- Main split: length-stratified.
- Cross benchmark: strict cross-source labels 1-4.
- CD-HIT benchmark: anchor80 threshold-compliant 80/70/60/50/40.
- Imbalance benchmark: nested negatives anchored to locked main 1:10.

Final dataset and QC paths:

| Purpose | Use this path |
|---|---|
| Positive cap200 source | `data/processed/positives_cap200.csv` |
| Hybrid negative pool | `data/processed/hybrid_oldneg_swissprot101_200_main_ablation/hybrid_negative_pool_final.csv` |
| Final main dataset | `data/benchmark_datasets/final_hybrid_suite_1to10/main_hybrid_1to10_cap200_length_stratified/` |
| Final main manifest | `data/manifests/final_hybrid_suite_1to10/main_hybrid_1to10_cap200_length_stratified/split_manifest.json` |
| Strict cross datasets | `data/benchmark_datasets/strict_cross_source_hybrid_1to10_cap200/strict_cross_source_hybrid_1to10_cap200/cross_label*_test/` |
| Strict cross manifests | `data/manifests/strict_cross_source_hybrid_1to10_cap200/strict_cross_source_hybrid_1to10_cap200/cross_label*_test/split_manifest.json` |
| Final CD-HIT datasets | `data/benchmark_datasets/cdhit_anchor80_threshold_compliant_1to10/threshold_*/` |
| Final CD-HIT manifests | `data/manifests/cdhit_anchor80_threshold_compliant_1to10/threshold_*/split_manifest.json` |
| Final imbalance datasets | `data/benchmark_datasets/final_hybrid_suite_imbalance_anchor_main_1to10/ratio_*/` |
| Final imbalance manifests | `data/manifests/final_hybrid_suite_imbalance_anchor_main_1to10/ratio_*/split_manifest.json` |
| Final main suite QC | `data/qc/final_hybrid_suite_1to10/` |
| Strict cross QC | `data/qc/strict_cross_source_hybrid_1to10_cap200/` |
| Final CD-HIT QC | `data/qc/cdhit_anchor80_threshold_compliant_1to10/` |
| Final imbalance QC | `data/qc/final_hybrid_suite_imbalance_anchor_main_1to10/` |

## Final Feature Line

| Purpose | Use this path |
|---|---|
| Final main/global ESM2-1280 feature root | `features/esm2_1280/final_hybrid_suite_1to10/` |
| Final main/global ESM2 feature pkl | `features/esm2_1280/final_hybrid_suite_1to10/global_cap200_unique/esm2_per_residue_features.pkl` |
| Strict cross ESM2 feature root | `features/esm2_1280/strict_cross_source_hybrid_1to10_cap200/` |
| CD-HIT ESM2 feature root | `features/esm2_1280/cdhit_anchor80_threshold_compliant_1to10/` |
| Imbalance ESM2 feature root | `features/esm2_1280/final_hybrid_suite_imbalance_anchor_main_1to10/` |
| Baseline feature root | `features/baseline_features/main_cross_full_rerun_v1/` |

## Final Model Result Line

| Experiment | Use this result path |
|---|---|
| Main 5-seed Full_Model | `results/model_runs/final_hybrid_suite_full_5seed/main/` |
| Strict cross 5-seed Full_Model | `results/model_runs/strict_cross_source_hybrid_full_5seed/cross_phylum/strict_cross_label*_test/Full_Model/seed_*/metrics.json` |
| Seed42 ablation, Full/No_CNN/No_Both | `results/model_runs/length_stratified_split_ablation_single_stage_maxf1/ablation/per_run_metrics.csv` |
| Seed42 ablation, strict No_Transformer_CNNOnly | `results/model_runs/parallel_trackb_hybrid_1to10_notransformer_cnnone_seed42/ablation/per_run_metrics.csv` |
| CD-HIT 5-seed Full_Model | `results/model_runs/cdhit_anchor80_threshold_compliant_full_5seed/cdhit/` |
| Imbalance 5-seed Full_Model | `results/model_runs/final_hybrid_imbalance_anchor_main_1to10_5seed/imbalance/` |

Important cross note:

- Do not use `results/model_runs/strict_cross_source_hybrid_full_5seed/cross_phylum/summary_metrics.csv` as the final cross summary because it currently only contains label1.
- Use the per-seed `metrics.json` files listed above, or the locked summary in `final_experiment_results_locked.md`.

## Final Baseline Result Line

Use these completed baseline files:

| Baseline family | Use this result path |
|---|---|
| PLM baselines | `results/baselines/main_cross_full_rerun_v1/plm/per_run_metrics.csv` |
| ESM2 per-residue + BiLSTM | `results/baselines/main_cross_full_rerun_v1/bilstm/per_run_metrics.csv` |
| Retrieval baselines | `results/baselines/main_cross_full_rerun_v1/retrieval/per_run_metrics.csv` |
| SProtFP | `results/baselines/main_cross_full_rerun_v1/sprotfp/per_run_metrics.csv` |
| AMP Scanner v2 | `results/baselines/main_cross_full_rerun_v1/amp_scanner/per_run_metrics.csv` |
| Foldseek / TM-align | `results/baselines/main_cross_full_rerun_v1/structure/global_structure_baseline_summary.csv` |

Pending baseline:

- `sAMPpred-GAT` is not final yet. It should only be added after feature QC and training complete.
- Ignore `results/baselines/main_cross_full_rerun_v1/sprotfp_bad_classpath_20260517_1518/`.

## Final Configs For Reproducibility

| Purpose | Use this config |
|---|---|
| Final main/cross/CD-HIT/imbalance dataset suite | `configs/data_build/final_hybrid_suite_1to10.json` |
| Strict cross-source dataset | `configs/data_build/strict_cross_source_hybrid_1to10.json` |
| Anchor80 threshold-compliant CD-HIT dataset | `configs/data_build/cdhit_anchor80_threshold_compliant_1to10.json` |
| Anchor-main imbalance dataset | `configs/data_build/final_hybrid_imbalance_anchor_main_1to10.json` |
| Final main model | `configs/model_runs/final_hybrid_suite_full_5seed.json` |
| Strict cross model | `configs/model_runs/strict_cross_source_hybrid_full_5seed.json` |
| Final CD-HIT model | `configs/model_runs/cdhit_anchor80_threshold_compliant_full_5seed.json` |
| Final imbalance model | `configs/model_runs/final_hybrid_imbalance_anchor_main_1to10_5seed.json` |
| Main/cross baselines | `configs/baselines/main_cross_full_baseline_rerun_v1.json` |

## Do Not Use For Final Manuscript Numbers

These are useful for audit/history but should not feed final figures or manuscript numbers unless explicitly re-approved:

| Pattern/path | Reason |
|---|---|
| `results/model_runs/final_model_suite/` | Early all-in-one run, superseded by locked final result lines. |
| `data/benchmark_datasets/main_1to10_cdhit80_cap200_length_matched/` | Early Swiss-Prot-only dataset line, superseded by hybrid final suite. |
| `data/benchmark_datasets/main_1to10_hybrid_oldneg_swissprot101_200_cap200_length_matched/` | Early main/ablation validation dataset, superseded by final hybrid suite. |
| `data/benchmark_datasets/cdhit_1to10_cap200_length_matched/` | Early CD-HIT line, superseded by anchor80 threshold-compliant CD-HIT. |
| `results/model_runs/length_stratified_split_ablation*/` | Split/diagnostic experiments, except the locked ablation rows explicitly listed above. |
| `results/model_runs/hybrid_oldneg_swissprot101_200_seed42_ablation/` | Early hybrid ablation probe. |
| `results/model_runs/serial_cls_hybrid_1to10_seed42_ablation/` | Serial architecture probe, not final. |
| `results/model_runs/parallel_trackb_hybrid_1to10_clean_ablation_seed42/` | Architecture probe, superseded by locked ablation composition. |
| `results/model_runs/parallel_trackb_hybrid_1to10_noboth_pooling_seed42/` | Pooling probe. |
| `results/model_runs/esm2_480_hybrid_1to10_clean_ablation_seed42/` | 480-dim PLM probe. |
| `results/model_runs/*lr1p5e5*` | Learning-rate probe. |
| `results/model_runs/*posweight2p5*` | Positive-weight probe. |
| `results/model_runs/*labelsmooth0p05*` | Label-smoothing probe. |
| `results/baselines/main_cross_plm_seed42_rerun_gpu1_v1/` | Seed42 PLM check, not the main 5-seed baseline table. |
| `results/baselines/imbalance_plm_rerun_v1/` | Auxiliary PLM-on-imbalance check, not currently locked for final manuscript. |

## Naming Caveats

- `b_gate012_model.py` and `b_gate012_training.py` are final model/training snapshots with historical names. The `b_gate012` name does not mean the final data are the old 1:16/cap100 setting.
- `old_negative_pool` is part of the final hybrid negative design for short negatives. It does not mean the final benchmark is the old dataset.
- Mentions of `<=100 aa bins` refer to the final hybrid negative-source split, not a cap100 truncation policy.

## Practical Rule

When writing the paper:

1. Start from `final_experiment_results_locked.md`.
2. Use only paths listed in this manifest.
3. If a number is not in the locked file or reproducible from a whitelisted path, do not put it into the manuscript.
