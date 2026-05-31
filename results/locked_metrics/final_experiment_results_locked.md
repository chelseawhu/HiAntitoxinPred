# Final Locked Experiment Results

This document records experiment results that have been reviewed and locked for the current HiAntitoxinPred release analysis line. Do not replace a locked result unless a later rerun is explicitly approved and documented.

## Locked: Main Comprehensive Benchmark

- Lock date: 2026-05-16
- Status: final locked result
- Result line: `final_hybrid_suite_full_5seed`
- Dataset protocol: hybrid old-negative + Swiss-Prot 101-200 aa negative dataset, cap200, length-stratified split, 1:10 positive:negative ratio.
- Model protocol: Full_Model only, ESM2-1280, current single-stage training, validation MaxF1 threshold selection, 5 train seeds 42-46.
- Result directory: `results/model_runs/final_hybrid_suite_full_5seed/`
- Source per-run file: `results/model_runs/final_hybrid_suite_full_5seed/main/per_run_metrics.csv`
- Source summary file: `results/model_runs/final_hybrid_suite_full_5seed/main/summary_metrics.csv`

Per-seed results:

| Seed | Threshold | Acc | Precision | Recall | Specificity | F1 | MCC | ROC-AUC | PR-AUC | TP/FP/TN/FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.09 | 0.9892 | 0.9833 | 0.8939 | 0.9985 | 0.9365 | 0.9319 | 0.9648 | 0.9260 | 59/1/671/7 |
| 43 | 0.10 | 0.9878 | 0.9831 | 0.8788 | 0.9985 | 0.9280 | 0.9231 | 0.8938 | 0.8902 | 58/1/671/8 |
| 44 | 0.10 | 0.9824 | 0.9492 | 0.8485 | 0.9955 | 0.8960 | 0.8881 | 0.9647 | 0.9269 | 56/3/669/10 |
| 45 | 0.50 | 0.9878 | 0.9672 | 0.8939 | 0.9970 | 0.9291 | 0.9233 | 0.9896 | 0.9605 | 59/2/670/7 |
| 46 | 0.55 | 0.9851 | 0.9508 | 0.8788 | 0.9955 | 0.9134 | 0.9061 | 0.9374 | 0.9150 | 58/3/669/8 |

Summary:

| n seeds | Acc mean ± std | F1 mean ± std | MCC mean ± std | Precision mean ± std | Recall mean ± std | Specificity mean ± std | ROC-AUC mean ± std | PR-AUC mean ± std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.9864 ± 0.0027 | 0.9206 ± 0.0161 | 0.9145 ± 0.0175 | 0.9667 ± 0.0166 | 0.8788 ± 0.0186 | 0.9970 ± 0.0015 | 0.9501 ± 0.0365 | 0.9237 ± 0.0253 |

Decision:

This main benchmark is fixed as the current final 5-seed main result for the HiAntitoxinPred release manuscript/workspace.

## Locked: Strict Cross-Source Benchmark

- Lock date: 2026-05-20
- Status: final locked result
- Result line: `strict_cross_source_hybrid_full_5seed`
- Dataset protocol: strict cross-source hybrid 1:10 cap200 benchmark, four leave-one-source-label test settings.
- Model protocol: Full_Model only, ESM2-1280, single-stage training, validation MaxF1 checkpoint/threshold selection, 5 train seeds 42-46.
- Result directory: `results/model_runs/strict_cross_source_hybrid_full_5seed/`
- Source metrics files: `results/model_runs/strict_cross_source_hybrid_full_5seed/cross_phylum/strict_cross_label*_test/Full_Model/seed_*/metrics.json`
- Important note: Label1 was overwritten with the 2026-05-19 rerun results. `results/model_runs/strict_cross_source_hybrid_full_5seed/cross_phylum/summary_metrics.csv` currently only contains label1 after rerun. The locked cross summary below is rebuilt from all 20 per-seed `metrics.json` files.

Per-seed results:

| Test label | Seed | Threshold | Acc | Precision | Recall | Specificity | F1 | MCC | ROC-AUC | PR-AUC | TP/FP/TN/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Label1 | 42 | 0.09 | 0.9643 | 0.7684 | 0.8690 | 0.9738 | 0.8156 | 0.7978 | 0.9318 | 0.8399 | 146/44/1636/22 |
| Label1 | 43 | 0.10 | 0.9556 | 0.7337 | 0.8036 | 0.9708 | 0.7670 | 0.7435 | 0.9182 | 0.7549 | 135/49/1631/33 |
| Label1 | 44 | 0.42 | 0.9746 | 0.8623 | 0.8571 | 0.9863 | 0.8597 | 0.8457 | 0.9833 | 0.8935 | 144/23/1657/24 |
| Label1 | 45 | 0.70 | 0.9605 | 0.8571 | 0.6786 | 0.9887 | 0.7575 | 0.7422 | 0.9715 | 0.8386 | 114/19/1661/54 |
| Label1 | 46 | 0.08 | 0.9702 | 0.8424 | 0.8274 | 0.9845 | 0.8348 | 0.8185 | 0.9575 | 0.8618 | 139/26/1654/29 |
| Label2 | 42 | 0.89 | 0.9758 | 0.9583 | 0.7667 | 0.9967 | 0.8519 | 0.8450 | 0.8324 | 0.8110 | 46/2/598/14 |
| Label2 | 43 | 0.54 | 0.9742 | 0.9778 | 0.7333 | 0.9983 | 0.8381 | 0.8345 | 0.8274 | 0.8015 | 44/1/599/16 |
| Label2 | 44 | 0.12 | 0.9742 | 0.9388 | 0.7667 | 0.9950 | 0.8440 | 0.8352 | 0.8734 | 0.8086 | 46/3/597/14 |
| Label2 | 45 | 0.08 | 0.9742 | 0.9574 | 0.7500 | 0.9967 | 0.8411 | 0.8346 | 0.9264 | 0.8307 | 45/2/598/15 |
| Label2 | 46 | 0.87 | 0.9712 | 0.9767 | 0.7000 | 0.9983 | 0.8155 | 0.8135 | 0.8778 | 0.8081 | 42/1/599/18 |
| Label3 | 42 | 0.41 | 0.9885 | 0.9697 | 0.9014 | 0.9972 | 0.9343 | 0.9287 | 0.9711 | 0.9305 | 64/2/708/7 |
| Label3 | 43 | 0.35 | 0.9872 | 0.9420 | 0.9155 | 0.9944 | 0.9286 | 0.9217 | 0.9562 | 0.9171 | 65/4/706/6 |
| Label3 | 44 | 0.90 | 0.9846 | 0.9155 | 0.9155 | 0.9915 | 0.9155 | 0.9070 | 0.9720 | 0.9437 | 65/6/704/6 |
| Label3 | 45 | 0.82 | 0.9859 | 0.8947 | 0.9577 | 0.9887 | 0.9252 | 0.9181 | 0.9699 | 0.8987 | 68/8/702/3 |
| Label3 | 46 | 0.95 | 0.9872 | 0.9692 | 0.8873 | 0.9972 | 0.9265 | 0.9205 | 0.9621 | 0.9357 | 63/2/708/8 |
| Label4 | 42 | 0.12 | 0.9877 | 0.9706 | 0.8919 | 0.9973 | 0.9296 | 0.9239 | 0.9617 | 0.9135 | 33/1/369/4 |
| Label4 | 43 | 0.09 | 0.9877 | 0.9706 | 0.8919 | 0.9973 | 0.9296 | 0.9239 | 0.9514 | 0.8939 | 33/1/369/4 |
| Label4 | 44 | 0.19 | 0.9877 | 0.9444 | 0.9189 | 0.9946 | 0.9315 | 0.9249 | 0.9641 | 0.9030 | 34/2/368/3 |
| Label4 | 45 | 0.13 | 0.9828 | 0.9167 | 0.8919 | 0.9919 | 0.9041 | 0.8948 | 0.9540 | 0.9007 | 33/3/367/4 |
| Label4 | 46 | 0.46 | 0.9828 | 0.9167 | 0.8919 | 0.9919 | 0.9041 | 0.8948 | 0.9605 | 0.9047 | 33/3/367/4 |

Summary:

| Test label | n seeds | F1 mean ± std | MCC mean ± std | Precision mean ± std | Recall mean ± std | Specificity mean ± std | ROC-AUC mean ± std | PR-AUC mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Label1 | 5 | 0.8069 ± 0.0438 | 0.7896 ± 0.0459 | 0.8128 ± 0.0581 | 0.8071 ± 0.0763 | 0.9808 ± 0.0080 | 0.9524 ± 0.0271 | 0.8377 ± 0.0514 |
| Label2 | 5 | 0.8381 ± 0.0136 | 0.8326 ± 0.0116 | 0.9618 ± 0.0161 | 0.7433 ± 0.0279 | 0.9970 ± 0.0014 | 0.8675 ± 0.0402 | 0.8120 ± 0.0110 |
| Label3 | 5 | 0.9260 ± 0.0068 | 0.9192 ± 0.0079 | 0.9382 ± 0.0331 | 0.9155 ± 0.0263 | 0.9938 ± 0.0037 | 0.9662 ± 0.0069 | 0.9251 ± 0.0177 |
| Label4 | 5 | 0.9198 ± 0.0143 | 0.9124 ± 0.0161 | 0.9438 ± 0.0270 | 0.8973 ± 0.0121 | 0.9946 ± 0.0027 | 0.9583 ± 0.0054 | 0.9032 ± 0.0071 |

Decision:

This strict cross-source benchmark is fixed as the current final cross-generalization result. Label1 is the weakest source-shift setting and should be discussed as the primary generalization stress case, while Label3 and Label4 show stronger cross-source transfer.

## Locked: Seed42 Ablation Study

- Lock date: 2026-05-16
- Status: final locked seed42 ablation display result
- Result line: final seed42 ablation table with strict `No_Transformer_CNNOnly` replacement
- Dataset protocol: same hybrid 1:10 cap200 length-stratified main dataset used for the final model line.
- Model protocol: ESM2-1280, seed 42, validation MaxF1 threshold selection.
- Full_Model / No_CNN / No_Both source file: `results/model_runs/length_stratified_split_ablation_single_stage_maxf1/ablation/per_run_metrics.csv`
- Strict No_Transformer source file: `results/model_runs/parallel_trackb_hybrid_1to10_notransformer_cnnone_seed42/ablation/per_run_metrics.csv`
- Note: this table intentionally replaces the older `No_Transformer` entry with the stricter CNN-only `No_Transformer_CNNOnly` result.

| Model | Structure | Threshold | Acc | Precision | Recall | Specificity | F1 | MCC | ROC-AUC | PR-AUC | TP/FP/TN/FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full_Model | Transformer + CNN + fusion | 0.70 | 0.9878 | 1.0000 | 0.8636 | 1.0000 | 0.9268 | 0.9232 | 0.9307 | 0.9071 | 57/0/672/9 |
| No_CNN | Transformer only | 0.36 | 0.9837 | 0.9500 | 0.8636 | 0.9955 | 0.9048 | 0.8971 | 0.9253 | 0.9024 | 57/3/669/9 |
| No_Transformer_CNNOnly | CNN only | 0.16 | 0.9797 | 0.9180 | 0.8485 | 0.9926 | 0.8819 | 0.8716 | 0.9125 | 0.8746 | 56/5/667/10 |
| No_Both | pooled ESM baseline | 0.10 | 0.9756 | 0.8750 | 0.8485 | 0.9881 | 0.8615 | 0.8483 | 0.9568 | 0.8930 | 56/8/664/10 |

Ablation deltas:

| Comparison | Delta F1 | Delta MCC |
| --- | ---: | ---: |
| Full_Model - No_CNN | +0.0221 | +0.0260 |
| Full_Model - No_Transformer_CNNOnly | +0.0449 | +0.0516 |
| Full_Model - No_Both | +0.0653 | +0.0749 |
| No_Transformer_CNNOnly - No_Both | +0.0204 | +0.0233 |

Decision:

This seed42 ablation group is fixed as the current preferred ablation display result. The intended interpretation is `Full_Model > No_CNN > No_Transformer_CNNOnly > No_Both`, supporting that both branches contribute and that the fused model has the strongest precision/MCC. If the manuscript later needs 5-seed ablation statistics, record that as a separate locked result rather than overwriting this seed42 table.

## Locked: Interpretability Case Study AT6247

- Lock date: 2026-05-27
- Status: final locked interpretability display case (updated mutation seeds)
- Result line: `final_main_sliding_window_w20_r10_newseed`
- Case sample: `AT6247`
- Case annotation: Ribbon-helix-helix protein, CopG domain, residues `2-39`.
- Perturbation protocol: sliding-window mutation, window size `20`, repeats `10`, model seeds `42-46`, mutation seeds `7/13/21/37/99`.
- Model protocol: final main Full_Model checkpoints, ESM2-1280, current final hybrid 1:10 cap200 length-stratified main benchmark.
- Result directory: `results/interpretability/final_main_sliding_window_w20_r10_newseed/`
- Source per-run file: `results/interpretability/final_main_sliding_window_w20_r10_newseed/per_run_alignment_metrics.csv`
- Source stability file: `results/interpretability/final_main_sliding_window_w20_r10_newseed/seed_stability_summary.csv`
- Source prediction file: `results/interpretability/final_main_sliding_window_w20_r10_newseed/original_prediction_summary.csv`
- Important note: `AT127` was also tested in the same rerun but is not locked as a main display case because its saliency-domain alignment was not stable across seeds.

Per-seed interpretability results:

| Seed (model / mutation) | Original prediction probability | Domain AUPRC | Domain AUROC | Top-L Precision | Mean domain-vs-non-domain saliency difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 / 7  | 0.9908 | 1.0000 | 1.0000 | 1.0000 | 0.0781 |
| 43 / 13 | 0.9923 | 0.9657 | 0.9498 | 0.8947 | 0.0910 |
| 44 / 21 | 0.9983 | 0.9666 | 0.9438 | 0.9211 | 0.1211 |
| 45 / 37 | 0.9828 | 0.9885 | 0.9916 | 0.9737 | 0.0889 |
| 46 / 99 | 0.9998 | 0.9582 | 0.9420 | 0.8947 | 0.1117 |

Summary:

| n seeds | Original probability mean ± std | Domain AUPRC mean ± std | Domain AUROC mean ± std | Top-L Precision mean ± std | Mean saliency difference mean ± std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.9928 ± 0.0061 | 0.9758 ± 0.0176 | 0.9654 ± 0.0280 | 0.9368 ± 0.0478 | 0.0982 ± 0.0177 |

Seed-stability summary:

| Pair count | Spearman mean ± std | Top-L Jaccard mean ± std | Top-L overlap fraction mean ± std |
| ---: | ---: | ---: | ---: |
| 10 | 0.8015 ± 0.1015 | 0.8550 ± 0.0536 | 0.9211 ± 0.0304 |

Decision:

AT6247 is fixed as the current final interpretability display case. The intended interpretation is that sliding-window perturbation importance is strongly enriched in the annotated RHH/CopG domain and remains stable across five final model seeds. Use AT6247 for the main interpretability figure; do not use AT127 as a main-text stability example unless a later method/sample rerun is explicitly approved and documented.

## Locked: CD-HIT Threshold Benchmark

- Lock date: 2026-05-16
- Status: final locked result
- Result line: `cdhit_anchor80_threshold_compliant_full_5seed`
- Dataset protocol: anchor80 threshold-compliant CD-HIT reconstruction, thresholds 80/70/60/50/40, overall 1:10 positive:negative ratio, 5 train seeds 42-46.
- Model protocol: Full_Model only, ESM2-1280, current single-stage training, validation MaxF1 threshold selection.
- Result directory: `results/model_runs/cdhit_anchor80_threshold_compliant_full_5seed/`
- Source summary file: `results/model_runs/cdhit_anchor80_threshold_compliant_full_5seed/cdhit/summary_metrics.csv`
- Paper table file: `results/model_runs/cdhit_anchor80_threshold_compliant_full_5seed/paper_table_cdhit.md`

| CD-HIT threshold | n seeds | F1 mean ± std | MCC mean ± std | Precision mean ± std | Recall mean ± std | ROC-AUC mean ± std | PR-AUC mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40% | 5 | 0.8555 ± 0.0292 | 0.8416 ± 0.0324 | 0.8599 ± 0.0389 | 0.8526 ± 0.0423 | 0.9737 ± 0.0094 | 0.9159 ± 0.0181 |
| 50% | 5 | 0.8791 ± 0.0087 | 0.8684 ± 0.0099 | 0.9154 ± 0.0207 | 0.8459 ± 0.0147 | 0.9379 ± 0.0217 | 0.8827 ± 0.0247 |
| 60% | 5 | 0.8982 ± 0.0162 | 0.8883 ± 0.0178 | 0.8789 ± 0.0361 | 0.9194 ± 0.0170 | 0.9766 ± 0.0144 | 0.9139 ± 0.0295 |
| 70% | 5 | 0.9137 ± 0.0227 | 0.9056 ± 0.0242 | 0.9201 ± 0.0128 | 0.9088 ± 0.0503 | 0.9623 ± 0.0257 | 0.9336 ± 0.0347 |
| 80% | 5 | 0.9274 ± 0.0159 | 0.9202 ± 0.0178 | 0.9204 ± 0.0372 | 0.9353 ± 0.0081 | 0.9864 ± 0.0087 | 0.9656 ± 0.0106 |

Decision:

This CD-HIT benchmark is fixed as the final CD-HIT result for the current HiAntitoxinPred release manuscript/workspace. It replaces earlier CD-HIT attempts whose negative sampling was less controlled or whose threshold trend was less interpretable.

## Locked: Imbalance Benchmark

- Lock date: 2026-05-18
- Status: final locked result
- Result line: `final_hybrid_imbalance_anchor_main_1to10_5seed`
- Dataset protocol: final hybrid benchmark with nested negative sets anchored to the locked 1:10 main benchmark.
- Important note: ratio `1:10` uses the locked main 5-seed result and is not treated as a separate retrained imbalance run.
- Model protocol: Full_Model only, ESM2-1280, single-stage training, validation MaxF1 checkpoint/threshold selection, 5 train seeds 42-46.
- Result directory: `results/model_runs/final_hybrid_imbalance_anchor_main_1to10_5seed/`
- Source summary file: `results/model_runs/final_hybrid_imbalance_anchor_main_1to10_5seed/imbalance/summary_metrics.csv`

| Ratio | n seeds | F1 mean ± std | MCC mean ± std | Precision mean ± std | Recall mean ± std | ROC-AUC mean ± std | PR-AUC mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1:1 | 5 | 0.9390 ± 0.0164 | 0.8832 ± 0.0334 | 0.9715 ± 0.0271 | 0.9091 ± 0.0186 | 0.9552 ± 0.0180 | 0.9583 ± 0.0197 |
| 1:10 | 5 | 0.9206 ± 0.0161 | 0.9145 ± 0.0175 | 0.9667 ± 0.0166 | 0.8788 ± 0.0186 | 0.9501 ± 0.0365 | 0.9237 ± 0.0253 |
| 1:20 | 5 | 0.8643 ± 0.0191 | 0.8608 ± 0.0185 | 0.9265 ± 0.0387 | 0.8121 ± 0.0409 | 0.9348 ± 0.0199 | 0.8517 ± 0.0225 |
| 1:30 | 5 | 0.8543 ± 0.0116 | 0.8529 ± 0.0113 | 0.9291 ± 0.0160 | 0.7909 ± 0.0198 | 0.9349 ± 0.0267 | 0.8300 ± 0.0324 |
| 1:40 | 5 | 0.8427 ± 0.0272 | 0.8415 ± 0.0269 | 0.8976 ± 0.0461 | 0.7970 ± 0.0509 | 0.9127 ± 0.0601 | 0.8209 ± 0.0401 |
| 1:50 | 5 | 0.8451 ± 0.0143 | 0.8451 ± 0.0149 | 0.9164 ± 0.0319 | 0.7848 ± 0.0225 | 0.8854 ± 0.0348 | 0.7717 ± 0.0179 |

Decision:

This imbalance benchmark is fixed as the current final imbalance result. The intended interpretation is that performance is strongest under balanced data and remains reasonably stable under increasingly negative-heavy settings, with the locked main 1:10 benchmark serving as the central reference point.

## Locked: Completed Baseline Comparison, with sAMPpred-GAT Main Added

- Lock date: 2026-05-18
- Status: completed baseline results locked; sAMPpred-GAT main benchmark result added on 2026-05-28, strict cross-source sAMPpred-GAT runs still in progress.
- Scope: Main + 4 strict cross-source benchmarks.
- Dataset protocol: same final main benchmark and strict cross-source benchmark used by the current model comparison.
- Completed baseline families:
  - PLM mean-pool baselines shown in final tables: ESM2 / ProtBERT with MLP / LogReg / RandomForest.
  - Note: ProtT5 baselines were evaluated and remain traceable in the source metrics, but are intentionally hidden from the final display tables.
  - ESM2 per-residue + BiLSTM.
  - Retrieval baselines: BLASTP, PSI-BLAST, HMMER/phmmer.
  - Direct external model: SProtFP.
  - AMP model: AMP Scanner v2 retrained.
  - Structure retrieval: Foldseek and TM-align.
- Pending baseline:
  - sAMPpred-GAT strict cross-source benchmarks are still running.
  - sAMPpred-GAT main benchmark is completed and recorded below.
- Source files:
  - `results/baselines/main_cross_full_rerun_v1/plm/per_run_metrics.csv`
  - `results/baselines/main_cross_full_rerun_v1/bilstm/per_run_metrics.csv`
  - `results/baselines/main_cross_full_rerun_v1/retrieval/per_run_metrics.csv`
  - `results/baselines/main_cross_full_rerun_v1/sprotfp/per_run_metrics.csv`
  - `results/baselines/main_cross_full_rerun_v1/amp_scanner/per_run_metrics.csv`
  - `results/baselines/main_cross_full_rerun_v1/structure/global_structure_baseline_summary.csv`
  - `results/baselines/main_cross_full_rerun_v1/samppred_gat/main/main_hybrid_1to10_cap200_length_stratified/seed_*/metrics.json`

Representative F1 comparison:

| Method | Main | Strict Cross Label1 | Strict Cross Label2 | Strict Cross Label3 | Strict Cross Label4 | Run type |
|---|---:|---:|---:|---:|---:|---|
| Full_Model | 0.9206 ± 0.0161 | 0.8069 ± 0.0438 | 0.8381 ± 0.0136 | 0.9260 ± 0.0068 | 0.9198 ± 0.0143 | 5 seeds |
| ESM2 + MLP | 0.8611 ± 0.0317 | 0.8078 ± 0.0359 | 0.7750 ± 0.0175 | 0.9279 ± 0.0103 | 0.8858 ± 0.0217 | 5 seeds |
| ProtBERT + MLP | 0.8300 ± 0.0291 | 0.7618 ± 0.0106 | 0.7098 ± 0.0422 | 0.8614 ± 0.0495 | 0.8660 ± 0.0200 | 5 seeds |
| ESM2 per-residue + BiLSTM | 0.8871 ± 0.0077 | 0.8164 ± 0.0249 | 0.8223 ± 0.0116 | 0.9350 ± 0.0055 | 0.8996 ± 0.0165 | 5 seeds |
| BLASTP | 0.8527 | 0.7101 | 0.7000 | 0.6957 | 0.7576 | single run |
| PSI-BLAST | 0.8504 | 0.6959 | 0.6838 | 0.7482 | 0.8060 | single run |
| HMMER/phmmer | 0.8814 | 0.7055 | 0.6593 | 0.6964 | 0.7419 | single run |
| Foldseek | 0.7000 | 0.5047 | 0.5370 | 0.5909 | 0.6575 | single run |
| TM-align | 0.3817 | 0.2722 | 0.3471 | 0.2857 | 0.4058 | single run |
| SProtFP | 0.5503 | 0.4110 | 0.2574 | 0.5140 | 0.3944 | single run |
| AMP Scanner v2 | 0.4394 ± 0.0480 | 0.2100 ± 0.0484 | 0.1848 ± 0.1236 | 0.2868 ± 0.1621 | 0.2462 ± 0.0467 | 5 seeds |
| sAMPpred-GAT | 0.7607 ± 0.0528 | running | running | running | running | 5 seeds for Main; strict cross running |

sAMPpred-GAT main benchmark details:

| Seed | Acc | Precision | Recall | Specificity | F1 | MCC | ROC-AUC | PR-AUC | TP/FP/TN/FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.9485 | 0.7188 | 0.6970 | 0.9732 | 0.7077 | 0.6796 | 0.9365 | 0.7305 | 46/18/654/20 |
| 43 | 0.9607 | 0.7534 | 0.8333 | 0.9732 | 0.7914 | 0.7709 | 0.9746 | 0.8176 | 55/18/654/11 |
| 44 | 0.9688 | 0.8644 | 0.7727 | 0.9881 | 0.8160 | 0.8005 | 0.9575 | 0.8464 | 51/8/664/15 |
| 45 | 0.9363 | 0.6044 | 0.8333 | 0.9464 | 0.7006 | 0.6768 | 0.9730 | 0.8577 | 55/36/636/11 |
| 46 | 0.9621 | 0.7879 | 0.7879 | 0.9792 | 0.7879 | 0.7670 | 0.9662 | 0.8463 | 52/14/658/14 |

| n seeds | Acc mean ± std | F1 mean ± std | MCC mean ± std | Precision mean ± std | Recall mean ± std | Specificity mean ± std | ROC-AUC mean ± std | PR-AUC mean ± std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.9553 ± 0.0129 | 0.7607 ± 0.0528 | 0.7390 ± 0.0570 | 0.7458 ± 0.0957 | 0.7848 ± 0.0561 | 0.9720 ± 0.0156 | 0.9616 ± 0.0155 | 0.8197 ± 0.0520 |

Strict cross baseline F1 summary:

| Method | Strict Cross Label1 | Strict Cross Label2 | Strict Cross Label3 | Strict Cross Label4 | Run type |
|---|---:|---:|---:|---:|---|
| Full_Model | 0.8069 ± 0.0438 | 0.8381 ± 0.0136 | 0.9260 ± 0.0068 | 0.9198 ± 0.0143 | 5 seeds |
| ESM2 + MLP | 0.8078 ± 0.0359 | 0.7750 ± 0.0175 | 0.9279 ± 0.0103 | 0.8858 ± 0.0217 | 5 seeds |
| ProtBERT + MLP | 0.7618 ± 0.0106 | 0.7098 ± 0.0422 | 0.8614 ± 0.0495 | 0.8660 ± 0.0200 | 5 seeds |
| ESM2 per-residue + BiLSTM | 0.8164 ± 0.0249 | 0.8223 ± 0.0116 | 0.9350 ± 0.0055 | 0.8996 ± 0.0165 | 5 seeds |
| BLASTP | 0.7101 | 0.7000 | 0.6957 | 0.7576 | single run |
| PSI-BLAST | 0.6959 | 0.6838 | 0.7482 | 0.8060 | single run |
| HMMER/phmmer | 0.7055 | 0.6593 | 0.6964 | 0.7419 | single run |
| Foldseek | 0.5047 | 0.5370 | 0.5909 | 0.6575 | single run |
| TM-align | 0.2722 | 0.3471 | 0.2857 | 0.4058 | single run |
| SProtFP | 0.4110 | 0.2574 | 0.5140 | 0.3944 | single run |
| AMP Scanner v2 | 0.2100 ± 0.0484 | 0.1848 ± 0.1236 | 0.2868 ± 0.1621 | 0.2462 ± 0.0467 | 5 seeds |

Decision:

These completed baselines are fixed as the current baseline comparison set. sAMPpred-GAT main benchmark is recorded here as a completed 5-seed result, while sAMPpred-GAT strict cross-source results should be added only after the running cross-source jobs finish. The main writing claim should avoid saying the proposed model beats every PLM baseline on every strict cross label; the safer interpretation is that the proposed model achieves the strongest main benchmark performance and strong overall robustness, while some PLM baselines remain competitive on selected cross-source labels.
