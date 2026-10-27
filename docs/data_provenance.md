# Data, models and rights

The repository includes the fixed Main members and split assignments used by the current ProtT5 Dynamic experiments. Original input strings and sequence hashes are preserved. CSV provenance columns retain source categories, functional support and original lengths. Their historical `old_split` fields do not override the current `split` column or `split_manifest.json`.

Positive identifiers and annotations derive from TADB; negative source categories include UniProtKB/Swiss-Prot, function-supported UniProtKB/TrEMBL and the additional sources recorded in the CSV. These are existing biological records, not newly measured sequences. Consult the retained identifiers, source categories and original providers for record-level provenance. Main identity checks and split manifests should accompany derived experiments rather than silently resampling proteins.

Official providers:

- TADB: https://bioinfo-mml.sjtu.edu.cn/TADB3/
- UniProt: https://www.uniprot.org/
- BAGEL4: http://bagel4.molgenrug.nl/
- T1TAdb: https://d-lab.arna.cnrs.fr/t1tadb

The encoder is Rostlab's `prot_t5_xl_half_uniref50-enc`, pinned at Hugging Face commit `94a6abc029ae13029317b140b7424e012bf8dfbf`. File SHA256 values are independently bound to the experimental encoder. Download from https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc rather than assuming any ProtT5 variant is interchangeable. ProtTrans paper: Elnaggar et al., *ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning*, IEEE TPAMI, DOI https://doi.org/10.1109/TPAMI.2021.3095381 . Consult the original model card and ProtTrans license at https://github.com/agemagician/ProtTrans .

The MIT license in this repository covers the authors' code. It does not override upstream sequence/annotation database terms or assign new rights to ProtT5 weights. The proposed Zenodo deposit contains author-produced classifier weights and derived features, with source identity metadata. The final deposit metadata and any dataset-specific rights statement must be reviewed before publication; no public deposit has yet been created by this release preparation.

The asset archives exclude credentials, personal local paths, unrelated experiments, alternative baselines and old network implementations. Sanitized provenance preserves source hashes while replacing local filesystem locations with source names. Historical source hashes are not hashes of the transformed metadata or converted weights; both identities are explicitly distinguished.
