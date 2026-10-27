# Validation summary

These checks were performed while preparing the release; users do not need to repeat them before prediction.

- The standalone network and five converted checkpoints were compared with the registered implementation on all6655 validation/test predictions. Same-environment GPU logits were identical; the CPU maximum difference was7.63e-6. Classifications agreed.
- The original739 MAIA raw-score records reproduced their saved signed and absolute profiles. Both five-model cases reproduced Region AP, Top-D overlap and seed stability.
- Real GPU MAIA controls covered a short input, a positive case and a200-residue negative:2453 original/perturbed scores agreed between implementations.
- Public prediction was tested on lengths1,2,19,20,199,200 and201, plus ambiguous residues. CPU FP32 and GPU mixed-precision encoding are documented separately.
- The default model was chosen by validation loss. Training accepts only train/validation datasets and evaluates held-out data through a separate command.
- The public tests cover the above contracts, selection behavior and the input/download interfaces. Private migration tools and experiment runners are not shipped.

A historical-GPU score comparison showed a maximum difference of approximately1.10e-4 with no classification changes. This is a recorded numerical difference, not a change to model weights, threshold or scientific results. Exact floating-point equality across machines is not required for use. See [numerical behavior](numerical_behavior.md).
