# Examples

## Jupyter Notebook

For an interactive comparison of all methods with visualizations and benchmarks, see the Jupyter notebook:

📓 **[examples/comparing_methods.ipynb](https://github.com/mstoelzle/torch-dxdt/blob/main/examples/comparing_methods.ipynb)**

The notebook includes:
- Visual comparison of all 7 differentiation methods
- RMSE accuracy analysis across noise levels (no noise, low noise, high noise)
- Computational efficiency benchmarks (forward and backward pass timing)
- Parameter tuning examples for noisy data
- Smoothing method comparisons

## Basic Usage

### Computing Derivatives of Smooth Functions

```python
import torch
import ptdxdt
import matplotlib.pyplot as plt

# Create smooth data
t = torch.linspace(0, 4 * torch.pi, 200)
x = torch.sin(t)

# True derivative
dx_true = torch.cos(t)

# Compute with different methods
dx_fd = ptdxdt.dxdt(x, t, kind="finite_difference", k=1)
dx_sg = ptdxdt.dxdt(x, t, kind="savitzky_golay", window_length=11, polyorder=3)
dx_spec = ptdxdt.dxdt(x, t, kind="spectral")

# Compare results
plt.figure(figsize=(10, 6))
plt.plot(t, dx_true, 'k-', label='True', linewidth=2)
plt.plot(t, dx_fd, '--', label='Finite Difference')
plt.plot(t, dx_sg, '--', label='Savitzky-Golay')
plt.plot(t, dx_spec, '--', label='Spectral')
plt.legend()
plt.xlabel('t')
plt.ylabel('dx/dt')
plt.title('Derivative Estimation Methods')
plt.show()
```

### Handling Noisy Data

```python
import torch
import ptdxdt

# Create noisy data
t = torch.linspace(0, 2 * torch.pi, 100)
x_clean = torch.sin(t)
noise = 0.1 * torch.randn(100)
x_noisy = x_clean + noise

# Use smoothing methods for noisy data
dx_sg = ptdxdt.dxdt(x_noisy, t, kind="savitzky_golay", 
                    window_length=15, polyorder=3)
dx_spline = ptdxdt.dxdt(x_noisy, t, kind="spline", s=0.1)
dx_kernel = ptdxdt.dxdt(x_noisy, t, kind="kernel", sigma=0.5, lmbd=0.1)
dx_kalman = ptdxdt.dxdt(x_noisy, t, kind="kalman", alpha=0.1)
dx_whittaker = ptdxdt.dxdt(x_noisy, t, kind="whittaker", lmbda=100.0)
```

## Autodiff Integration

### Training a Neural Network with Derivative Loss

```python
import torch
import torch.nn as nn
import ptdxdt

# Simple neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, t):
        return self.fc(t)

# Create training data
t = torch.linspace(0, 2 * torch.pi, 100).unsqueeze(1)
x_true = torch.sin(t.squeeze())
dx_true = torch.cos(t.squeeze())

# Train with derivative loss
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    
    # Forward pass
    x_pred = model(t).squeeze()
    x_pred.requires_grad_(True)
    
    # Compute derivative of prediction
    dx_pred = ptdxdt.dxdt(x_pred, t.squeeze(), kind="finite_difference")
    
    # Loss includes both value and derivative matching
    loss_x = ((x_pred - x_true) ** 2).mean()
    loss_dx = ((dx_pred - dx_true) ** 2).mean()
    loss = loss_x + 0.1 * loss_dx
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

### Computing Second Derivatives

```python
import torch
import ptdxdt

t = torch.linspace(0, 2 * torch.pi, 100)
x = torch.sin(t)

# First derivative (should be cos(t))
dx = ptdxdt.dxdt(x, t, kind="savitzky_golay", 
                 window_length=11, polyorder=5, order=1)

# Second derivative (should be -sin(t))
d2x = ptdxdt.dxdt(x, t, kind="savitzky_golay", 
                  window_length=11, polyorder=5, order=2)

# Or compute spectral derivatives of any order
d2x_spectral = ptdxdt.dxdt(x, t, kind="spectral", order=2)
```

### Computing Multiple Derivative Orders Efficiently

Use `d_orders()` or `dxdt_orders()` to compute multiple derivatives in one pass:

```python
import torch
import ptdxdt

t = torch.linspace(0, 2 * torch.pi, 100)
x = torch.sin(t) + 0.1 * torch.randn(100)

# Compute smoothed signal, first and second derivatives simultaneously
# This is more efficient than calling dxdt() multiple times
derivs = ptdxdt.dxdt_orders(x, t, kind="savitzky_golay",
                            window_length=11, polyorder=4,
                            orders=[0, 1, 2])

x_smooth = derivs[0]  # Smoothed signal
dx = derivs[1]        # First derivative  
d2x = derivs[2]       # Second derivative

# Or using the class interface
sg = ptdxdt.SavitzkyGolay(window_length=11, polyorder=4)
derivs = sg.d_orders(x, t, orders=[0, 1, 2])

# Whittaker also supports efficient multi-order computation
wh = ptdxdt.Whittaker(lmbda=100.0)
derivs = wh.d_orders(x, t, orders=[0, 1, 2])
```

## GPU Acceleration

### Moving Computations to GPU

```python
import torch
import ptdxdt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data on GPU
t = torch.linspace(0, 2 * torch.pi, 10000, device=device)
x = torch.sin(t) + 0.01 * torch.randn(10000, device=device)

# All computations happen on GPU
dx = ptdxdt.dxdt(x, t, kind="savitzky_golay", window_length=21, polyorder=3)

print(f"Result device: {dx.device}")  # cuda:0
```

## Batch Processing

### Processing Multiple Signals

```python
import torch
import ptdxdt

# Batch of signals: (batch_size, time_steps)
batch_size = 32
time_steps = 100

t = torch.linspace(0, 2 * torch.pi, time_steps)
freqs = torch.linspace(1, 5, batch_size).unsqueeze(1)
x = torch.sin(freqs * t)  # Shape: (32, 100)

# Compute derivatives for all signals at once
dx = ptdxdt.dxdt(x, t, kind="finite_difference", axis=-1)
print(f"Output shape: {dx.shape}")  # (32, 100)
```

## Smoothing Data

### Using smooth_x for Denoising

```python
import torch
import ptdxdt

t = torch.linspace(0, 2 * torch.pi, 100)
x_clean = torch.sin(t)
x_noisy = x_clean + 0.2 * torch.randn(100)

# Smooth the data
x_smooth_sg = ptdxdt.smooth_x(x_noisy, t, kind="savitzky_golay",
                               window_length=15, polyorder=3)
x_smooth_spline = ptdxdt.smooth_x(x_noisy, t, kind="spline", s=0.5)
x_smooth_kalman = ptdxdt.smooth_x(x_noisy, t, kind="kalman", alpha=0.5)
x_smooth_whittaker = ptdxdt.smooth_x(x_noisy, t, kind="whittaker", lmbda=100.0)
```

## Class-Based Interface

### Using Method Classes Directly

```python
import torch
from ptdxdt import SavitzkyGolay, Spline, Kalman

t = torch.linspace(0, 2 * torch.pi, 100)
x = torch.sin(t) + 0.1 * torch.randn(100)

# Create reusable method instances
sg = SavitzkyGolay(window_length=11, polyorder=3, order=1)
spline = Spline(s=0.1, order=1)
kalman = Kalman(alpha=0.1)

# Compute derivatives
dx_sg = sg.d(x, t)
dx_spline = spline.d(x, t)
dx_kalman = kalman.d(x, t)

# Get smoothed data
x_smooth = sg.smooth(x, t)
```
