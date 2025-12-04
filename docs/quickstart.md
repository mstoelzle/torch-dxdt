# Quick Start

This guide will help you get started with **ptdxdt** quickly.

## Basic Usage

### Functional Interface

The simplest way to use ptdxdt is through the `dxdt` function:

```python
import torch
import ptdxdt

# Create sample data
t = torch.linspace(0, 2 * torch.pi, 100)
x = torch.sin(t) + 0.1 * torch.randn(100)

# Compute derivative using Savitzky-Golay filter
dx = ptdxdt.dxdt(x, t, kind="savitzky_golay", window_length=11, polyorder=3)
```

### Object-Oriented Interface

For more control, you can use the derivative classes directly:

```python
import torch
import ptdxdt

t = torch.linspace(0, 2 * torch.pi, 100)
x = torch.sin(t)

# Create a Savitzky-Golay filter
sg = ptdxdt.SavitzkyGolay(window_length=11, polyorder=3)

# Compute derivative
dx = sg.d(x, t)
```

## Available Methods

To see all available differentiation methods:

```python
print(ptdxdt.available_methods())
# ['finite_difference', 'savitzky_golay', 'spectral', 'spline', 'kernel', 'kalman']
```

## Working with Gradients

All methods are fully differentiable:

```python
import torch
import ptdxdt

t = torch.linspace(0, 2 * torch.pi, 100)
x = torch.sin(t).requires_grad_(True)

# Compute derivative
dx = ptdxdt.dxdt(x, t, kind="finite_difference", k=1)

# Backpropagate through the operation
loss = dx.sum()
loss.backward()

# x.grad now contains gradients
print(x.grad is not None)  # True
```

## Batched Processing

All methods support batched inputs along the first dimension:

```python
import torch
import ptdxdt

t = torch.linspace(0, 2 * torch.pi, 100)

# Batch of 3 signals
x_batch = torch.stack([
    torch.sin(t),
    torch.cos(t),
    torch.sin(2*t)
], dim=0)  # Shape: (3, 100)

# Compute derivatives for all signals at once
dx_batch = ptdxdt.dxdt(x_batch, t, kind="finite_difference", k=1)
# dx_batch has shape (3, 100)
```

## Smoothing

Some methods also support smoothing without differentiation:

```python
import torch
import ptdxdt

t = torch.linspace(0, 2 * torch.pi, 100)
x = torch.sin(t) + 0.2 * torch.randn(100)  # Noisy signal

# Get smoothed signal
x_smooth = ptdxdt.smooth_x(x, t, kind="spline", s=0.1)
```

## Next Steps

- See {doc}`methods` for detailed documentation of each method
- Check out {doc}`examples` for more advanced usage patterns
- Refer to {doc}`api` for complete API reference
