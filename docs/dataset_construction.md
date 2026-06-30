# Dataset Construction

This document records how the HiAntitoxinPred benchmark datasets were assembled.
The processed sequences and splits are version-controlled under
`data/benchmark_datasets/` and `data/manifests/`; this page documents their
provenance and the construction choices behind them.

> **Note for completion:** items marked _[to be provided]_ below should be filled
> in before submission. They cannot be recovered from the processed files alone.

## Task and labels

Binary classification of bacterial proteins as **antitoxin** (positive, label 1)
or **non-antitoxin** (negative, label 0).

## Positive set (antitoxins)

The positive set comprises 336 antitoxin proteins (identifiers `AT1`, `AT2`, …).

_[to be provided: the database(s) and/or literature the antitoxin set was curated
from, and what the `AT…` identifier scheme refers to.]_

Each positive also carries a `source_label` (1–4) recording its origin category;
this is used by the cross-source generalization benchmark.

_[to be provided: what categories 1, 2, 3, and 4 correspond to.]_

## Negative set (non-antitoxins)

Negatives are drawn from two pools, recorded in the `negative_source` and
`source_pool` columns of `processed_sequences.csv`:

- **`old_negative_pool`** (2210 sequences) — _[to be provided: source of this pool
  and the operational definition of "non-antitoxin".]_
- **`swissprot_functional_101_200`** (1150 sequences) — functionally annotated
  UniProt/Swiss-Prot proteins of length 101–200 residues, identified by UniProt
  accession. _[to be provided: exact retrieval query — reviewed status, taxonomic
  scope, functional/annotation filters, and UniProt release/date.]_

## Redundancy reduction

Sequence redundancy is controlled with CD-HIT. The main benchmark uses the
anchor-80 threshold-compliant protocol; dedicated CD-HIT benchmarks additionally
sweep identity thresholds of 40 / 50 / 60 / 70 / 80 %
(`data/benchmark_datasets/cdhit/`).

## Length handling and class ratio

Sequences are capped at 200 residues and stratified into six length bins
(`bin1_42_77` … `bin6_133_200`). For the main benchmark, negatives are sampled to
match positives within each length bin at an overall **1:10** positive:negative
ratio.

## Splits

The main benchmark is split into train / validation / test =
**2365 / 593 / 738** sequences (215 / 55 / 66 positives), recorded in
`data/manifests/main/split_manifest.json`. Additional benchmarks vary the class
ratio (`data/benchmark_datasets/imbalance/`, 1:1 through 1:50) and hold out entire
source categories for cross-source evaluation
(`data/benchmark_datasets/strict_cross/`).

## Processed sequence columns

Key columns in `processed_sequences.csv`:

| Column | Meaning |
| --- | --- |
| `id`, `source_id` | Sequence identifier (antitoxin `AT…` id, or UniProt accession for negatives) |
| `sequence`, `label` | Amino-acid sequence and class label (1 = antitoxin, 0 = negative) |
| `source_label` | Origin category of positives (1–4; used by cross-source benchmark) |
| `record_type` | `positive` or `negative` |
| `negative_source`, `source_pool` | Pool a negative was drawn from |
| `original_length`, `cap200_length` | Length before and after the 200-residue cap |
| `length_bin` | Length-stratification bin |

## License and citation

_[to be provided: license for the released datasets, and how to cite the datasets
and their original data sources.]_
