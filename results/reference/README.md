# Main reference results

`main_per_seed.csv` is recomputed from the original registered L prediction
files for seeds42–46. `main_mean_sd.csv` summarizes each split independently
using sample SD across those five models. `PROVENANCE.json` identifies the
original prediction and checkpoint hashes without local filesystem paths.

All ranking metrics use raw logits; binary predictions use logit>=0. These
reference results are from the archived GPU runs, not new training or migrated
CPU inference. Converted-weight replay is checked separately. Validation loss
controls L selection; F1 and test scores do not select the exported checkpoint.
