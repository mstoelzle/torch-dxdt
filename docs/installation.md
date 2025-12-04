# Installation

## From PyPI

The easiest way to install **torch-dxdt** is via pip:

```bash
pip install torch-dxdt
```

## From Source

To install the latest development version:

```bash
git clone https://github.com/mstoelzle/torch-dxdt.git
cd torch-dxdt
pip install -e .
```

## Development Installation

For development, install with all optional dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- Testing dependencies (pytest, derivative)
- Documentation dependencies (sphinx, sphinx-rtd-theme)
- Code quality tools (ruff, mypy)

## Requirements

- Python >= 3.9
- PyTorch >= 1.10.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## Conda Environment

You can also create a dedicated conda environment:

```bash
conda create -n torch-dxdt python=3.13 -y
conda activate torch-dxdt
pip install torch-dxdt
```

## Verifying Installation

To verify the installation:

```python
import torch_dxdt
print(torch_dxdt.__version__)
print(torch_dxdt.available_methods())
```
