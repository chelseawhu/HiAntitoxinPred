# Supplementary Table S3. Seed42 ablation study

Locked seed42 ablation display result. Ablations are shown using standard `w/o` module notation. The `w/o TGCE` entry uses the strict CNN-only replacement.

| Ablation variant | Module configuration | Structure | Threshold | Accuracy | Precision | Recall | Specificity | F1 | MCC | ROC-AUC | PR-AUC | TP/FP/TN/FN |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| HiAntitoxinPred | PERE + DMCME + TGCE + GIAC | Transformer + CNN + gated fusion | 0.70 | 0.9878 | 1.0000 | 0.8636 | 1.0000 | 0.9268 | 0.9232 | 0.9307 | 0.9071 | 57/0/672/9 |
| w/o DMCME | PERE + TGCE + classifier | Transformer-based global context branch only | 0.36 | 0.9837 | 0.9500 | 0.8636 | 0.9955 | 0.9048 | 0.8971 | 0.9253 | 0.9024 | 57/3/669/9 |
| w/o TGCE | PERE + DMCME + classifier | Dynamic multi-scale CNN motif branch only | 0.16 | 0.9797 | 0.9180 | 0.8485 | 0.9926 | 0.8819 | 0.8716 | 0.9125 | 0.8746 | 56/5/667/10 |
| No_Both | PERE + pooled classifier | PLM-derived pooled representation baseline | 0.10 | 0.9756 | 0.8750 | 0.8485 | 0.9881 | 0.8615 | 0.8483 | 0.9568 | 0.8930 | 56/8/664/10 |

Abbreviations: PERE, PLM-derived Evolutionary Residue Encoder; DMCME, Dynamic Multi-scale CNN Motif Encoder; TGCE, Transformer-based Global Context Encoder; GIAC, Gated Imbalance-Adaptive Classifier.
