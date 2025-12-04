# torch-dxdt Documentation

**torch-dxdt** - PyTorch Differentiable Numerical Differentiation

A PyTorch implementation of numerical differentiation methods for noisy time series data.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
methods
api
examples
```

## Features

- 🔥 **Fully Differentiable**: All methods support PyTorch autograd for backpropagation
- 🚀 **GPU Accelerated**: Leverage PyTorch's GPU support for fast computation
- 📊 **Multiple Methods**: Seven differentiation algorithms for different use cases
- 📈 **Higher-Order Derivatives**: Support for 2nd-order and multi-order derivative computation
- 🔧 **Easy API**: Simple functional and object-oriented interfaces
- 🧪 **Well Tested**: Validated against the reference `derivative` package

## Quick Example

```python
import torch
import torch_dxdt

t = torch.linspace(0, 2 * torch.pi, 100)
x = torch.sin(t) + 0.1 * torch.randn(100)

# Compute derivative
dx = torch_dxdt.dxdt(x, t, kind="savitzky_golay", window_length=11, polyorder=3)
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
