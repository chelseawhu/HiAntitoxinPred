# Installation

Create the recommended conda environment:

```bash
conda env create -f environment.yml
conda activate hiantitoxinpred
```

For editable development installs:

```bash
pip install -e .
```

GPU execution requires a PyTorch build compatible with the local CUDA driver.
If the default conda solve installs a CPU-only build, install the appropriate
PyTorch/CUDA package from the official PyTorch instructions for your machine.

