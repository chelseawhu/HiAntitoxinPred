# Example output: P9WJ87 / AT6247

These small output examples use the 82-residue main-text case in `../AT6247.fasta`, the default seed42/epoch159 model and public padding width 200. They were exported from recorded fresh GPU inference. `PROVENANCE.json` records the model and numerical settings.

| File | Contents |
|---|---|
| `AT6247_prediction.csv` | One protein-level score and predicted label |
| `maia_AT6247/residue_scores.csv` | 82 residue rows with signed scores and absolute sensitivity |
| `maia_AT6247/window_scores.csv` | 63 windows with mean changes over 10 perturbations each |

The CLI writes the same CSV columns. A new MAIA run also writes `raw_scores.json`, `metadata.json`, `progress.json` and `COMPLETE.json`; these larger/intermediate records are generated when running the command rather than included in this example.

A positive `signed_score` means that perturbing a covering window reduced the antitoxin logit. `coverage` counts the windows covering that residue. Negative values are retained. These files illustrate the default single-model interface. The paper's two-case settings, including their original padding widths, are listed in `../../docs/maia.md`.

Small numerical differences are expected across CPU/GPU environments. The prediction shown here is antitoxin (`prediction=1`).
