# Current Results Inventory For Manuscript Drafting

Generated: 2026-05-18

This file organizes the currently available `HiAntitoxinPred release` results for manuscript writing. It is a writing inventory, not a replacement for the locked raw result files.

## Current Status

| Module | Status | Notes |
|---|---|---|
| Main benchmark | Ready | Locked 5-seed result is available. |
| Seed42 ablation | Ready | Use strict `No_Transformer_CNNOnly` replacement table. |
| CD-HIT threshold benchmark | Ready | Anchor80 threshold-compliant 5-seed result is locked. |
| Imbalance benchmark | Ready | 1:1 and 1:20-1:50 completed; 1:10 should be interpreted as locked main benchmark. |
| Strict cross-phylum | Mostly ready | All 4 label run directories and `metrics.json` files exist, but `summary_metrics.csv` currently only contains label1 after rerun. Rebuild paper summary from per-seed `metrics.json`. |
| PLM baselines | Ready | ESM2/ProtBERT/ProtT5 with MLP/LogReg/RF, 5 seeds. |
| BiLSTM baseline | Ready | ESM2 per-residue + BiLSTM, 5 seeds. |
| Retrieval baselines | Ready | BLASTP, PSI-BLAST, HMMER/phmmer direct-hit transfer. |
| SProtFP baseline | Ready | Direct prediction, single run. Ignore `sprotfp_bad_classpath_20260517_1518`. |
| AMP Scanner baseline | Ready | Retrained AMP Scanner v2, 5 seeds. |
| Structure baselines | Ready | Foldseek and TM-align completed after ColabFold structure coverage reached 11818/11818. |
| sAMPpred-GAT baseline | Pending | Feature generation still running; do not include final result yet. |

## Locked Main Result

Source:
`results/model_runs/final_hybrid_suite_full_5seed/main/summary_metrics.csv`

Dataset/protocol:
hybrid old-negative + Swiss-Prot 101-200 aa negative dataset, cap200, length-stratified split, 1:10 positive:negative ratio, ESM2-1280, single-stage training, validation MaxF1 threshold selection, seeds 42-46.

| n seeds | F1 | MCC | Precision | Recall | Specificity | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.9206 ± 0.0161 | 0.9145 ± 0.0175 | 0.9667 ± 0.0166 | 0.8788 ± 0.0186 | 0.9970 ± 0.0015 | 0.9501 ± 0.0365 | 0.9237 ± 0.0253 |

Seed42 main result:
F1 0.9365, MCC 0.9319, Precision 0.9833, Recall 0.8939, TP/FP/TN/FN = 59/1/671/7.

## Locked Ablation Result

Sources:

- `results/model_runs/length_stratified_split_ablation_single_stage_maxf1/ablation/per_run_metrics.csv`
- `results/model_runs/parallel_trackb_hybrid_1to10_notransformer_cnnone_seed42/ablation/per_run_metrics.csv`

Use this table for the current manuscript unless a later 5-seed ablation is explicitly locked.

| Model | Structure | F1 | MCC | Precision | Recall | TP/FP/TN/FN | Threshold |
|---|---|---:|---:|---:|---:|---|---:|
| Full_Model | Transformer + CNN + fusion | 0.9268 | 0.9232 | 1.0000 | 0.8636 | 57/0/672/9 | 0.70 |
| No_CNN | Transformer only | 0.9048 | 0.8971 | 0.9500 | 0.8636 | 57/3/669/9 | 0.36 |
| No_Transformer_CNNOnly | CNN only | 0.8819 | 0.8716 | 0.9180 | 0.8485 | 56/5/667/10 | 0.16 |
| No_Both | pooled ESM baseline | 0.8615 | 0.8483 | 0.8750 | 0.8485 | 56/8/664/10 | 0.10 |

Interpretation:
`Full_Model > No_CNN > No_Transformer_CNNOnly > No_Both`.

## Strict Cross-Phylum Result

Source metrics:
`results/model_runs/strict_cross_source_hybrid_full_5seed/cross_phylum/strict_cross_label*_test/Full_Model/seed_*/metrics.json`

Important note:
`results/model_runs/strict_cross_source_hybrid_full_5seed/cross_phylum/summary_metrics.csv` currently only contains label1. For manuscript tables, rebuild from all per-seed `metrics.json` files.

| Test label | n seeds | F1 | MCC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Label1 | 5 | 0.7919 ± 0.0460 | 0.7741 ± 0.0471 | 0.8252 | 0.7655 |
| Label2 | 5 | 0.8381 ± 0.0136 | 0.8326 ± 0.0116 | 0.9618 | 0.7433 |
| Label3 | 5 | 0.9260 ± 0.0068 | 0.9192 ± 0.0079 | 0.9382 | 0.9155 |
| Label4 | 5 | 0.9198 ± 0.0143 | 0.9124 ± 0.0161 | 0.9438 | 0.8973 |

Writing note:
Label1 is the weakest leave-one-source group and should be discussed as the strictest generalization stress case, not hidden.

## CD-HIT Threshold Result

Source:
`results/model_runs/cdhit_anchor80_threshold_compliant_full_5seed/cdhit/summary_metrics.csv`

| CD-HIT threshold | n seeds | F1 | MCC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 80% | 5 | 0.9274 ± 0.0159 | 0.9202 ± 0.0178 | 0.9204 | 0.9353 |
| 70% | 5 | 0.9137 ± 0.0227 | 0.9056 ± 0.0242 | 0.9201 | 0.9088 |
| 60% | 5 | 0.8982 ± 0.0162 | 0.8883 ± 0.0178 | 0.8789 | 0.9194 |
| 50% | 5 | 0.8791 ± 0.0087 | 0.8684 ± 0.0099 | 0.9154 | 0.8459 |
| 40% | 5 | 0.8555 ± 0.0292 | 0.8416 ± 0.0324 | 0.8599 | 0.8526 |

Writing note:
This is the clean CD-HIT story: performance decreases with stricter sequence-identity control but remains usable.

## Imbalance Result

Source:
`results/model_runs/final_hybrid_imbalance_anchor_main_1to10_5seed/imbalance/summary_metrics.csv`

Use locked main result for the 1:10 row.

| Ratio | n seeds | F1 | MCC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 1:1 | 5 | 0.9390 ± 0.0164 | 0.8832 ± 0.0334 | 0.9715 | 0.9091 |
| 1:10 | 5 | 0.9206 ± 0.0161 | 0.9145 ± 0.0175 | 0.9667 | 0.8788 |
| 1:20 | 5 | 0.8643 ± 0.0191 | 0.8608 ± 0.0185 | 0.9265 | 0.8121 |
| 1:30 | 5 | 0.8543 ± 0.0116 | 0.8529 ± 0.0113 | 0.9291 | 0.7909 |
| 1:40 | 5 | 0.8427 ± 0.0272 | 0.8415 ± 0.0269 | 0.8976 | 0.7970 |
| 1:50 | 5 | 0.8451 ± 0.0143 | 0.8451 ± 0.0149 | 0.9164 | 0.7848 |

Writing note:
F1 and recall decline as the negative class becomes more dominant, while MCC remains relatively stable after 1:20-1:50.

## Completed Baselines

Main + strict cross datasets are available for the following baseline families.

| Baseline family | Methods | Run count | Suggested placement |
|---|---|---:|---|
| PLM mean-pool classical/MLP | ESM2, ProtBERT, ProtT5 × MLP/LogReg/RF | 225 rows | Main selected + supplementary full table |
| Per-residue neural baseline | ESM2 per-residue + BiLSTM | 25 rows | Main or supplementary |
| Retrieval | BLASTP, PSI-BLAST, HMMER/phmmer | 15 rows | Main selected + supplementary |
| Direct external model | SProtFP direct prediction | 5 rows | Supplementary or baseline table |
| AMP model | AMP Scanner v2 retrained | 25 rows | Supplementary |
| Structure retrieval | Foldseek, TM-align | 10 rows | Supplementary; can mention structure-only baselines are weaker |
| sAMPpred-GAT | sAMPpred-GAT retrained | pending | Add later if feature/training completes |

Selected baseline source files:

- `results/baselines/main_cross_full_rerun_v1/plm/per_run_metrics.csv`
- `results/baselines/main_cross_full_rerun_v1/bilstm/per_run_metrics.csv`
- `results/baselines/main_cross_full_rerun_v1/retrieval/per_run_metrics.csv`
- `results/baselines/main_cross_full_rerun_v1/sprotfp/per_run_metrics.csv`
- `results/baselines/main_cross_full_rerun_v1/amp_scanner/per_run_metrics.csv`
- `results/baselines/main_cross_full_rerun_v1/structure/global_structure_baseline_summary.csv`

## Baseline Highlights

Use these as candidate comparisons in the main paper.

| Method | Main F1 | Label1 F1 | Label2 F1 | Label3 F1 | Label4 F1 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Ours Full_Model | 0.9206 ± 0.0161 | 0.7919 ± 0.0460 | 0.8381 ± 0.0136 | 0.9260 ± 0.0068 | 0.9198 ± 0.0143 | Main model, 5 seeds |
| ESM2 per-residue + BiLSTM | 0.8871 ± 0.0077 | 0.8164 ± 0.0249 | 0.8223 ± 0.0116 | 0.9350 ± 0.0055 | 0.8996 ± 0.0165 | Strong per-residue neural baseline |
| ESM2 + MLP | 0.8611 ± 0.0317 | 0.8078 ± 0.0359 | 0.7750 ± 0.0175 | 0.9279 ± 0.0103 | 0.8858 ± 0.0217 | Mean-pool PLM baseline |
| ProtT5 + MLP | 0.8738 ± 0.0295 | 0.8371 ± 0.0320 | 0.7664 ± 0.0157 | 0.9352 ± 0.0135 | 0.9265 ± 0.0068 | Strong PLM baseline; can be close on some cross labels |
| BLASTP | 0.8527 | 0.7101 | 0.7000 | 0.6957 | 0.7576 | Direct-hit retrieval |
| PSI-BLAST | 0.8504 | 0.6959 | 0.6838 | 0.7482 | 0.8060 | Direct-hit retrieval |
| HMMER/phmmer | 0.8814 | 0.7055 | 0.6593 | 0.6964 | 0.7419 | Direct-hit retrieval |
| Foldseek | 0.7000 | 0.5047 | 0.5370 | 0.5909 | 0.6575 | Structure retrieval |
| TM-align | 0.3817 | 0.2722 | 0.3471 | 0.2857 | 0.4058 | Structure retrieval |
| SProtFP | 0.5503 | 0.4110 | 0.2574 | 0.5140 | 0.3944 | Direct prediction |
| AMP Scanner v2 | 0.4394 ± 0.0480 | 0.2100 ± 0.0484 | 0.1848 ± 0.1236 | 0.2868 ± 0.1621 | 0.2462 ± 0.0467 | Retrained AMP baseline |

Writing note:
Do not overclaim that the model beats every PLM baseline on every single cross label. The stronger and more defensible claim is that the proposed model gives the best overall main benchmark performance and strong robustness, while simple PLM baselines can be competitive in selected leave-one-source settings.

## sAMPpred-GAT Pending State

Current process:
`prepare_samppred_features` is still running through `generate_features.py` / `hhblits`.

Current feature counts at check time:

| Feature directory | Count |
|---|---:|
| `npz` | 1251 |
| `pssm/output` | 1251 |
| `hhm/output` | 1251 |
| nested `hhm/output/output` | 2719 |
| `a3m` | 0 |

Do not train or report sAMPpred-GAT until feature QC passes.

## Suggested Manuscript Result Layout

| Manuscript item | Content |
|---|---|
| Figure 1 | Dataset construction + model architecture schematic. |
| Figure 2 | Main benchmark performance and seed stability. |
| Figure 3 | Strict cross-phylum/source benchmark. |
| Figure 4 | CD-HIT threshold robustness. |
| Figure 5 | Imbalance robustness. |
| Figure 6 or panel | Ablation study showing branch contribution. |
| Main Table 1 | Dataset statistics and benchmark definitions. |
| Main Table 2 | Selected baseline comparison. |
| Supplementary Table S1 | Full per-seed main/cross/CD-HIT/imbalance results. |
| Supplementary Table S2 | Full baseline table across all methods. |
| Supplementary Table S3 | Data QC, length distribution, CD-HIT and split checks. |

## Immediate Writing-Prep TODO

1. Rebuild a clean strict-cross paper summary from all 20 `metrics.json` files and save it under `paper_workspace/tables/`.
2. Decide which baselines go into the main text table versus supplementary full table.
3. Wait for sAMPpred-GAT or explicitly mark it as pending/excluded if it cannot finish in time.
4. Prepare final figure/table scripts from the locked source files only.
5. Start drafting Methods around the final frozen protocol: hybrid negative construction, cap200, length-stratified split, validation MaxF1, 5 seeds.
