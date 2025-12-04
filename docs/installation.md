# Installation

## From PyPI

The easiest way to install **ptdxdt** is via pip:

```bash
pip install ptdxdt
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
conda create -n ptdxdt python=3.13 -y
conda activate ptdxdt
pip install ptdxdt
```

## Verifying Installation

To verify the installation:

```python
import ptdxdt
print(ptdxdt.__version__)
print(ptdxdt.available_methods())
```
