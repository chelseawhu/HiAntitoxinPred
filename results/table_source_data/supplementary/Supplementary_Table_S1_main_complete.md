# Supplementary Table S1. Complete main benchmark results

Values are reported as mean ± standard deviation for 5-seed methods and as single-run values for retrieval/direct-transfer methods. ProtT5 baselines were evaluated but are intentionally hidden from final display tables; they remain traceable in the source PLM metrics.

| Method                         | Baseline family                  | Run type   | F1              | MCC             | Precision       | Recall          | ROC-AUC         | PR-AUC          |
|:-------------------------------|:---------------------------------|:-----------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| HiAntitoxinPred                | Proposed model                   | 5 seeds    | 0.9206 ± 0.0161 | 0.9145 ± 0.0175 | 0.9667 ± 0.0166 | 0.8788 ± 0.0186 | 0.9501 ± 0.0365 | 0.9237 ± 0.0253 |
| ESM2 + MLP                     | PLM mean-pool classifier         | 5 seeds    | 0.8611 ± 0.0317 | 0.8483 ± 0.0351 | 0.8762 ± 0.0616 | 0.8485 ± 0.0186 | 0.9621 ± 0.0141 | 0.9138 ± 0.0104 |
| ESM2 + Logistic Regression     | PLM mean-pool classifier         | 5 seeds    | 0.8682 ± 0.0000 | 0.8559 ± 0.0000 | 0.8889 ± 0.0000 | 0.8485 ± 0.0000 | 0.9771 ± 0.0000 | 0.9407 ± 0.0000 |
| ESM2 + Random Forest           | PLM mean-pool classifier         | 5 seeds    | 0.8762 ± 0.0109 | 0.8660 ± 0.0116 | 0.9205 ± 0.0187 | 0.8364 ± 0.0198 | 0.9830 ± 0.0010 | 0.9264 ± 0.0032 |
| ProtBERT + MLP                 | PLM mean-pool classifier         | 5 seeds    | 0.8300 ± 0.0291 | 0.8187 ± 0.0268 | 0.8979 ± 0.0307 | 0.7758 ± 0.0681 | 0.9614 ± 0.0074 | 0.8950 ± 0.0084 |
| ProtBERT + Logistic Regression | PLM mean-pool classifier         | 5 seeds    | 0.8167 ± 0.0000 | 0.8054 ± 0.0000 | 0.9074 ± 0.0000 | 0.7424 ± 0.0000 | 0.9615 ± 0.0000 | 0.9051 ± 0.0000 |
| ProtBERT + Random Forest       | PLM mean-pool classifier         | 5 seeds    | 0.8230 ± 0.0202 | 0.8060 ± 0.0228 | 0.8262 ± 0.0471 | 0.8212 ± 0.0127 | 0.9760 ± 0.0016 | 0.9021 ± 0.0052 |
| ESM2 per-residue + BiLSTM      | Per-residue PLM sequence model   | 5 seeds    | 0.8871 ± 0.0077 | 0.8784 ± 0.0085 | 0.9346 ± 0.0322 | 0.8455 ± 0.0249 | 0.9800 ± 0.0016 | 0.9417 ± 0.0043 |
| BLASTP                         | Sequence retrieval               | single run | 0.8527          | 0.8389          | 0.8730          | 0.8333          | 0.9812          | 0.9103          |
| PSI-BLAST                      | Sequence retrieval               | single run | 0.8504          | 0.8371          | 0.8852          | 0.8182          | 0.9578          | 0.8834          |
| HMMER/phmmer                   | Sequence retrieval               | single run | 0.8814          | 0.8785          | 1.0000          | 0.7879          | 0.9799          | 0.8741          |
| SProtFP                        | External model direct prediction | single run | 0.5503          | 0.5046          | 0.4940          | 0.6212          | 0.8218          | 0.5907          |
| AMP Scanner v2                 | AMP model retrained              | 5 seeds    | 0.4394 ± 0.0480 | 0.4353 ± 0.0352 | 0.6676 ± 0.1378 | 0.3424 ± 0.0762 | 0.8685 ± 0.0156 | 0.5439 ± 0.0386 |
| Foldseek                       | Structure retrieval              | single run | 0.7000          | 0.6778          | 0.7778          | 0.6364          | 0.9389          | 0.7606          |
| TM-align                       | Structure retrieval              | single run | 0.3817          | 0.3215          | 0.3846          | 0.3788          | 0.6723          | 0.3395          |
