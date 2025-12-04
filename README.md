# torch-dxdt - Numerical Derivatives of Timeseries Data in PyTorch

[![PyPI version](https://badge.fury.io/py/torch_dxdt.svg)](https://badge.fury.io/py/torch-dxdt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**torch-dxdt** is a PyTorch implementation of numerical differentiation methods for (noisy) time series data. It provides differentiable versions of common differentiation algorithms, allowing them to be used as part of neural network training pipelines, physics-informed neural networks (PINNs), and other gradient-based optimization tasks.

This package is inspired by and aims to be compatible with the [derivative](https://github.com/andgoldschmidt/derivative) package by Andy Goldschmidt.

## Features

- 🔥 **Fully Differentiable**: All methods support PyTorch autograd for backpropagation
- 🚀 **GPU Accelerated**: Leverage PyTorch's GPU support for fast computation
- 📊 **Multiple Methods**: Seven differentiation algorithms for different use cases
- 📈 **Higher-Order Derivatives**: Support for 2nd-order and multi-order derivative computation
- 🔧 **Easy API**: Simple functional and object-oriented interfaces
- 🧪 **Well Tested**: Validated against the reference `derivative` package

## Installation

```bash
pip install torch-dxdt
```

Or install from source:

```bash
git clone https://github.com/mstoelzle/torch-dxdt.git
cd torch-dxdt
pip install -e .
```

## Quick Start

```python
import torch
import torch_dxdt

# Create sample data
t = torch.linspace(0, 2 * torch.pi, 100)
x = torch.sin(t) + 0.1 * torch.randn(100)

# Compute derivative using functional interface
dx = torch_dxdt.dxdt(x, t, kind="savitzky_golay", window_length=11, polyorder=3)

# Or use object-oriented interface
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3)
dx = sg.d(x, t)
```

## Available Methods

| Method | Class | Differentiable? | Best For |
|--------|-------|-----------------|----------|
| Finite Differences | `FiniteDifference` | ✅ Yes | Fast, simple differentiation |
| Savitzky-Golay | `SavitzkyGolay` | ✅ Yes | Noisy data with polynomial smoothing |
| Spectral | `Spectral` | ✅ Yes | Smooth, periodic signals |
| Spline | `Spline` | ✅ Yes | Smoothing with controllable regularization |
| Kernel (GP) | `Kernel` | ✅ Yes | Probabilistic smoothing |
| Kalman | `Kalman` | ✅ Yes | State estimation with noise model |
| Whittaker-Eilers | `Whittaker` | ✅ Yes | Global smoothing with penalized least squares |

## Method Details

### Finite Difference
```python
# Symmetric finite differences using Taylor series coefficients
# k=1 gives 3-point central difference, k=2 gives 5-point, etc.
dx = torch_dxdt.dxdt(x, t, kind="finite_difference", k=1)
```

### Savitzky-Golay Filter
```python
# Polynomial smoothing filter - great for noisy data
dx = torch_dxdt.dxdt(x, t, kind="savitzky_golay", 
                 window_length=11,  # Must be odd
                 polyorder=3)       # Polynomial order
```

### Spectral Differentiation
```python
# FFT-based differentiation - very accurate for periodic signals
dx = torch_dxdt.dxdt(x, t, kind="spectral")

# With frequency filtering
dx = torch_dxdt.dxdt(x, t, kind="spectral", 
                 filter_func=lambda k: (k < 10).float())
```

### Spline Smoothing
```python
# Whittaker/spline smoothing with regularization
dx = torch_dxdt.dxdt(x, t, kind="spline", s=0.01)  # s controls smoothing
```

### Kernel (Gaussian Process)
```python
# GP-based differentiation with RBF kernel
dx = torch_dxdt.dxdt(x, t, kind="kernel", 
                 sigma=1.0,   # Kernel length scale
                 lmbd=0.1)    # Noise variance
```

### Kalman Smoother
```python
# Kalman smoother assuming Brownian motion derivative
dx = torch_dxdt.dxdt(x, t, kind="kalman", alpha=1.0)
```

### Whittaker-Eilers Smoother
```python
# Global smoother using penalized least squares with Cholesky decomposition
# lmbda controls smoothness (larger = smoother)
dx = torch_dxdt.dxdt(x, t, kind="whittaker", lmbda=100.0)

# With different difference penalty order
dx = torch_dxdt.dxdt(x, t, kind="whittaker", lmbda=1000.0, d_order=2)
```

## Higher-Order Derivatives

Several methods support computing higher-order derivatives:

```python
# Second-order derivative with Savitzky-Golay (order is a constructor parameter)
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=2)
d2x = sg.d(x, t)  # Second derivative

# Or via functional interface (order passed as kwarg to constructor)
d2x = torch_dxdt.dxdt(x, t, kind="savitzky_golay", 
                  window_length=11, polyorder=4, order=2)

# Whittaker also supports second-order derivatives via d_orders()
wh = torch_dxdt.Whittaker(lmbda=100.0)
derivs = wh.d_orders(x, t, orders=[1, 2])
dx, d2x = derivs[1], derivs[2]
```

## Multi-Order Derivatives

For efficiency, you can compute multiple derivative orders simultaneously:

```python
# Compute smoothed signal, first and second derivatives in one call
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4)
derivs = sg.d_orders(x, t, orders=[0, 1, 2])

x_smooth = derivs[0]  # Smoothed signal (order 0)
dx = derivs[1]        # First derivative
d2x = derivs[2]       # Second derivative

# Also available via functional interface
derivs = torch_dxdt.dxdt_orders(x, t, kind="savitzky_golay",
                            window_length=11, polyorder=4,
                            orders=[0, 1, 2])
```

This is more efficient than calling `d()` multiple times as it avoids redundant computation.

## Using with Neural Networks

The key feature of **torch-dxdt** is that all operations are differentiable, so you can use them in training loops:

```python
import torch
import torch.nn as nn
import torch_dxdt

class PhysicsInformedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.diff = torch_dxdt.SavitzkyGolay(window_length=5, polyorder=2)
    
    def forward(self, t):
        x = self.net(t.unsqueeze(-1)).squeeze(-1)
        dx = self.diff.d(x, t)
        return x, dx

# Training loop
model = PhysicsInformedModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

t = torch.linspace(0, 2*torch.pi, 100)
for epoch in range(100):
    optimizer.zero_grad()
    x, dx = model(t)
    
    # Physics loss: dx/dt should equal some target
    physics_loss = (dx - torch.cos(t)).pow(2).mean()
    physics_loss.backward()
    optimizer.step()
```

## Smoothing

Some methods also support smoothing without differentiation:

```python
# Get smoothed signal
x_smooth = torch_dxdt.smooth_x(x, t, kind="spline", s=0.1)
x_smooth = torch_dxdt.smooth_x(x, t, kind="kernel", sigma=1.0, lmbd=0.1)
x_smooth = torch_dxdt.smooth_x(x, t, kind="kalman", alpha=1.0)
x_smooth = torch_dxdt.smooth_x(x, t, kind="whittaker", lmbda=100.0)
```

## Batched Processing

All methods support batched inputs:

```python
# Process multiple signals at once
x_batch = torch.stack([torch.sin(t), torch.cos(t)], dim=0)  # Shape: (2, 100)
dx_batch = torch_dxdt.dxdt(x_batch, t, kind="savitzky_golay", 
                        window_length=11, polyorder=3)
# dx_batch has shape (2, 100)
```

## Comparison with `derivative` Package

**torch-dxdt** is designed to be API-compatible with the [derivative](https://github.com/andgoldschmidt/derivative) package where possible, but with PyTorch tensors instead of NumPy arrays:

```python
# derivative (NumPy)
from derivative import dxdt as np_dxdt
dx_np = np_dxdt(x_np, t_np, kind="finite_difference", k=1)

# torch-dxdt (PyTorch)
import torch_dxdt
dx_torch = torch_dxdt.dxdt(x_torch, t_torch, kind="finite_difference", k=1)
```

## API Reference

### Functional Interface

```python
torch_dxdt.dxdt(x, t, kind=None, dim=-1, **kwargs)
```
Compute the derivative of `x` with respect to `t`.

```python
torch_dxdt.smooth_x(x, t, kind=None, dim=-1, **kwargs)
```
Compute the smoothed version of `x` (only for methods that support it).

### Classes

All derivative classes inherit from `torch_dxdt.Derivative` and implement:
- `d(x, t, dim=-1)`: Compute derivative
- `d_orders(x, t, orders=[0, 1, 2], dim=-1)`: Compute multiple derivative orders efficiently
- `smooth(x, t, dim=-1)`: Compute smoothed signal (if supported)

For higher-order derivatives, pass `order=N` to the constructor (e.g., `SavitzkyGolay(order=2)`) or use `d_orders()`.

```python
torch_dxdt.dxdt_orders(x, t, kind=None, orders=(1, 2), dim=-1, **kwargs)
```
Compute multiple derivative orders simultaneously.

## Examples

For a comprehensive comparison of all methods with varying noise levels and computational benchmarks, see the Jupyter notebook:

📓 **[examples/comparing_methods.ipynb](examples/comparing_methods.ipynb)**

The notebook includes:
- Visual comparison of all 7 differentiation methods
- RMSE accuracy analysis across noise levels (no noise, low noise, high noise)
- Computational efficiency benchmarks (forward and backward pass timing)
- Parameter tuning examples for noisy data
- Smoothing method comparisons

## Requirements

- Python >= 3.9
- PyTorch >= 1.10.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## Development

```bash
# Clone the repository
git clone https://github.com/mstoelzle/torch-dxdt.git
cd torch-dxdt

# Create conda environment
conda create -n torch-dxdt python=3.13 -y
conda activate torch-dxdt

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with the reference derivative package
pip install derivative
pytest tests/test_correctness.py -v
```

## Citation

If you use this package in your research, please consider citing:

```bibtex
@software{torch-dxdt,
  author = {Stölzle, Maximilian},
  title = {torch-dxdt: Differentiable Numerical Differentiation in PyTorch},
  url = {https://github.com/mstoelzle/torch-dxdt},
  year = {2025}
}
```

This package builds upon the work in the `derivative` package:

```bibtex
@article{kaptanoglu2022pysindy,
  doi = {10.21105/joss.03994},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {69},
  pages = {3994},
  author = {Alan A. Kaptanoglu and others},
  title = {PySINDy: A comprehensive Python package for robust sparse system identification},
  journal = {Journal of Open Source Software}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- **Repository**: [https://github.com/mstoelzle/torch-dxdt](https://github.com/mstoelzle/torch-dxdt)
- **Documentation**: [https://mstoelzle.github.io/torch-dxdt](https://mstoelzle.github.io/torch-dxdt)
- **Original derivative package**: [https://github.com/andgoldschmidt/derivative](https://github.com/andgoldschmidt/derivative)
