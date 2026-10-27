# Numerical behavior of the current extraction

This package extracts the effective SimplifiedDynamic network and
converts its five registered minimum-validation-loss checkpoints. It does not
change the convolution or padding mathematics.

## Prediction replay

The registered Main evaluation uses its fixed split order, batches of 16 and
zero padding to the longest protein in each batch. The independent replay
compares original and standalone implementations using identical inputs,
weights, FP32 dtype and CPU environment. The acceptance criterion is a maximum
absolute raw-logit difference of 1e-4 and identical zero-threshold predictions.
Comparisons to previously saved GPU scores are reported separately.

## Padding limitation

The shared biased projection can make padded positions nonzero. The local
convolutions see these projected values before their outputs are masked. The
kernel-size-2 branch also convolves without padding, then pads its last output
position. Therefore changing a batch's padded width can change scores even for
the same protein. The low-level model retains that behavior for auditable weight
migration.

Public FASTA prediction pads to 200 by default.
This removes the dependence of padding width on other proteins in the batch.
Across five models and all 592 validation proteins, this policy changed logits
by at most 0.000844955 relative to the original batch16 geometry, with zero
classification changes. Reversing the validation order with fixed-width batches
gave identical CPU scores. Seed42 single-versus-batch checks on 32 validation
proteins differed by at most 9.54e-7. These observations are not a guarantee of
bitwise equivalence across hardware or encoder environments.

The tensor API rejects an all-empty input, a non-prefix mask, non-finite
features and nonzero input padding. Tensor width must be 2 through 200. A
one-residue protein would require an explicitly padded width of at least 2;
the public FASTA interface pads it to200. Actual GPU tests cover lengths1,2,19,20,199,200 and truncation of201.

Main reference replay retains the original order and longest-in-batch padding;
the public fixed-width policy is not retroactively applied to paper results.
No convolution or masking formula was changed. Test performance did not select
the padding policy. Fresh-encoding MAIA controls cover a10-residue sequence, the82-residue AT6247 case and a200-residue negative. Across2453 original/perturbed scores, converted and original classifiers had zero logit difference under the same GPU inputs.

## Initialization and training

Removing unused modules can change random-number consumption during model
construction and during training dropout calls. Matching converted checkpoint
outputs does not demonstrate that a fresh training run with the same integer
seed will reproduce the historical trajectory. The independent trainer retains the fixed C1 recipe and records this limitation. Original source weights, predictions,
selection locks and environments remain the reference evidence.

The post-pooling global MLP and residual projection are retained because they
affect predictions. Only the inactive single-input modality gate and unused
auxiliary head are removed. The shared branch-scorer bias is retained; its
common shift cancels in the six-way softmax.

## Encoder boundary

The encoder pins all five files to the experimental ProtT5 identity. GPU loading uses171 FP16 and24 FP32 `wo` parameters. CPU uses FP32 throughout. The latter is explicitly a different numerical environment: a10-residue control had logit-2.0295174 on CPU versus-2.0297277 with GPU encoding. The class was unchanged, but exact equality is not promised.

Two original Main inputs had exactly matching fresh-versus-cached features and logits in the GPU encoder control. CPU raw-score aggregation replay passed for all739 archived MAIA proteins. These finite controls do not claim that every possible new encoding is bitwise identical across devices.

The optional feature cache binds the original input (before UZOB mapping), encoder file hashes, precision/software identity and encoder batch membership. Partial cache hits preserve the original batch geometry. Public prediction and MAIA default to width 200; MAIA also accepts the original per-protein reference width. Main evaluation remains longest-in-batch. See the MAIA guide for the two case commands.
