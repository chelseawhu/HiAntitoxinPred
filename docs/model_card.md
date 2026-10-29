# Current model and checkpoint identity

This package implements the adopted SimplifiedDynamic,
cap200, C1/L HiAntitoxinPred model. Model assets are available in [Zenodo v3.0](https://doi.org/10.5281/zenodo.23755956).

## Inputs and outputs

The classifier consumes 1024-dimensional frozen ProtT5 residue features for up
to 200 residues. It produces one positive-class raw logit per protein. A logit
of zero is the decision boundary (sigmoid 0.5). The sigmoid score is not a claim
of probability calibration.

The effective architecture retains six local convolution scales, mean pooling,
a shared sample-dependent branch scorer, a two-layer global Transformer,
attention pooling, the global projection MLP/residual, concatenation fusion and
the classifier. The extracted network contains 25,484,547 registered trainable
parameters, excluding the frozen encoder. The historical wrapper registered
26,371,589: the difference of 887,042 is an inactive auxiliary head and a
single-modality gate that had no effect on C1 inference. No learned effective
weight is omitted by conversion. The shared scorer bias remains for exact key
mapping, although its common shift cancels in the branch softmax.

## Released models

The default deployment model is `best_model.pt`, derived from the registered Main
seed42 L checkpoint at epoch159. It has the lowest validation objective loss
among the five registered models. The other seeds are 43,44,45,46 with L epochs
134,83,173,175. The model index records the supplied checkpoint hashes.

The default model is not the average of the five models. Tables under
`results/reference` contain separately computed historical per-seed validation
and test metrics and sample-SD summaries. They use the original archived GPU
predictions; migration replays are distinct verification records. Removing
unused modules changes initialization/random-number consumption for new
training, so this package does not promise the same historical trajectory from
only the same integer seed.

## Input geometry and release checks

Public feature prediction pads each input to 200 positions; original Main
reference evaluation uses the registered split order and batch16 longest-in-batch
padding. The difference and validation checks are documented in
[numerical behavior](numerical_behavior.md). The original checkpoints and paper
results remain unchanged.

The package also provides standalone Main feature extraction, training, evaluation, FASTA prediction and MAIA interfaces. Numerical evidence and limitations are listed in [release checks](release_checks.md). Main features are generated locally and are not included in the model download.
