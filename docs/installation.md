# Installation

The tested Python stack is Python 3.9, PyTorch 2.5.1, NumPy 2.0.2, scikit-learn 1.6.1, SciPy 1.13.1, transformers 4.37.2 and sentencepiece 0.2.1.

## GPU installation

The README gives the complete clone, environment and installation commands for Linux with NVIDIA CUDA 12.4. Other supported builds are listed in the [official PyTorch 2.5.1 instructions](https://pytorch.org/get-started/previous-versions/#v251). Install the appropriate PyTorch build before `python -m pip install .`.

## CPU installation

```bash
git clone https://github.com/chelseawhu/HiAntitoxinPred.git
cd HiAntitoxinPred
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
hiantitoxinpred download --assets best
hiantitoxinpred download --assets encoder
hiantitoxinpred predict --fasta examples/AT6247.fasta --device cpu --output results/AT6247_predictions.csv
```

CPU encoding uses FP32 and is slower than GPU encoding.

## Environment and assets

`environment.yml` lists the tested Python dependencies. The release checks use RTX3090 GPUs with 24GB. Keep encoder batches at the default two sequences and classifier batches at 16 to match the recorded controls. CPU uses FP32 encoding; GPU uses the original FP16 load with 24 FP32 feed-forward `wo` parameters. The package rejects a blanket `.half()` layout and checks all five encoder file identities.

Asset groups: `best` (default model and metadata), `main-models` (all five), and `encoder`. [Zenodo v3.0](https://doi.org/10.5281/zenodo.23755956) contains three files: `best_model.pt`, `model_metadata.zip`, and `main_other_four_models.zip`. The manifest provides their download URLs, filenames and hashes. The encoder is downloaded separately from its pinned upstream revision.

Main features are generated locally with `hiantitoxinpred extract-features --device cuda:0` after installing the encoder. No `main_features.zip` download is needed. Use `--device cpu` for CPU extraction; see [Main reproduction](reproduce_main.md) for the complete workflow. FASTA prediction and MAIA generate their input features directly and do not require the Main feature store.

`download --offline-dir DIR` reads the same filenames from a local directory. Files are verified by both size and SHA256 before installation. Existing files of a different identity cause an error; archives reject traversal paths and symbolic links. Downloads are retained in `assets/downloads/` for verification and reuse. They may be removed manually after confirming the installed assets, if disk space is limited.

The encoder is approximately 2.42GB, the default classifier 117MB, all five classifiers 587MB and Main residue features 1.58GB uncompressed. Downloads plus installed assets temporarily use more space. Training stores one L checkpoint and small epoch validation archives rather than hundreds of full checkpoints. Full MAIA cost scales with protein length: 10 perturbations per complete 20-residue window, plus the original sequence.
