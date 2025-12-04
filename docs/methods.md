# Differentiation Methods

This page describes each differentiation method in detail.

## Finite Difference

The `FiniteDifference` class computes derivatives using symmetric finite difference schemes.

### Theory

For a function $f(x)$, the central difference approximation is:

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

Higher-order approximations use more points. The window size parameter `k` controls how many points are used: a scheme with parameter `k` uses $2k+1$ points.

### Usage

```python
import torch_dxdt

# 3-point central difference (k=1)
fd = torch_dxdt.FiniteDifference(k=1)

# 5-point difference (k=2)
fd = torch_dxdt.FiniteDifference(k=2)

# Periodic boundary conditions
fd = torch_dxdt.FiniteDifference(k=1, periodic=True)
```

### Parameters

- `k` (int): Window size. Uses 2k+1 points. Default: 1
- `periodic` (bool): If True, uses circular boundary conditions. Default: False

### When to Use

- Fast computation needed
- Data is relatively clean (low noise)
- Simple differentiation without smoothing

---

## Savitzky-Golay Filter

The `SavitzkyGolay` class fits a polynomial to local windows and computes derivatives from the polynomial coefficients.

### Theory

The Savitzky-Golay filter fits a polynomial of order $m$ to a window of $2k+1$ points using least squares, then evaluates the derivative of the polynomial at the center point.

### Usage

```python
import torch_dxdt

# Window of 11 points, cubic polynomial
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3)

# First derivative (default)
dx = sg.d(x, t)

# Higher derivatives
sg2 = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=2)
d2x = sg2.d(x, t)

# Different padding modes for boundary handling
sg_reflect = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3, pad_mode='reflect')
sg_circular = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3, pad_mode='circular')
```

### Parameters

- `window_length` (int): Length of the filter window. Must be odd.
- `polyorder` (int): Order of the polynomial. Must be less than window_length.
- `order` (int): Derivative order. Default: 1
- `pad_mode` (str): Padding mode for boundary handling. Options:
  - `'replicate'` (default): Repeat edge values. Good for monotonic signals.
  - `'reflect'`: Mirror the signal at boundaries. Good for symmetric signals.
  - `'circular'`: Wrap around. **Only for periodic signals.**
- `periodic` (bool): Deprecated. Use `pad_mode='circular'` instead.

### Padding Mode Details

The choice of padding mode affects derivative accuracy at signal boundaries:

```python
# For monotonic signals (e.g., exponential growth)
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3, pad_mode='replicate')

# For signals with local symmetry at boundaries
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3, pad_mode='reflect')

# For truly periodic signals (e.g., full sine wave cycles)
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3, pad_mode='circular')
```

**Warning:** Using `'circular'` padding on non-periodic signals can cause severe artifacts at boundaries!

### When to Use

- Data is noisy
- You want smoothing with differentiation
- The signal can be locally approximated by a polynomial

---

## Spectral Differentiation

The `Spectral` class computes derivatives using Fourier transforms.

### Theory

In Fourier space, differentiation becomes multiplication:

$$\mathcal{F}[f'(x)] = i\omega \mathcal{F}[f(x)]$$

This method:
1. Transforms to Fourier space using FFT
2. Multiplies by $(i\omega)^n$ for the $n$-th derivative
3. Transforms back

### Usage

```python
import torch_dxdt
import torch

# Basic usage
spec = torch_dxdt.Spectral()

# Higher-order derivatives
spec2 = torch_dxdt.Spectral(order=2)

# With frequency filtering
spec_filtered = torch_dxdt.Spectral(
    filter_func=lambda k: (k < 10).float()
)
```

### Parameters

- `order` (int): Order of the derivative. Default: 1
- `filter_func` (callable): Optional function to filter frequencies

### When to Use

- Data is smooth and periodic
- Very high accuracy is needed
- The signal is band-limited

### Note

This method assumes the data is periodic. For non-periodic data, consider using other methods or applying windowing.

---

## Spline Smoothing

The `Spline` class uses Whittaker smoothing to compute derivatives.

### Theory

Whittaker smoothing solves:

$$\min_z \|x - z\|^2 + \lambda \|Dz\|^2$$

where $D$ is a difference operator and $\lambda$ controls smoothing. The derivative is computed from the smoothed signal.

### Usage

```python
import torch_dxdt

# Light smoothing
spl = torch_dxdt.Spline(s=0.01)

# Heavy smoothing
spl = torch_dxdt.Spline(s=10.0)

# Get smoothed signal without derivative
x_smooth = spl.smooth(x, t)
```

### Parameters

- `s` (float): Smoothing parameter. Larger values give more smoothing.
- `order` (int): Order of the difference operator. Default: 3

### When to Use

- Data is noisy and needs smoothing
- You want controllable regularization
- You need both smoothing and differentiation

---

## Kernel (Gaussian Process)

The `Kernel` class uses Gaussian process regression for differentiation.

### Theory

Given observations $y$ at times $t$, the GP posterior mean is:

$$\hat{f}(t^*) = k(t^*, t)(K + \lambda I)^{-1}y$$

The derivative is computed using the derivative of the kernel:

$$\hat{f}'(t^*) = k'(t^*, t)(K + \lambda I)^{-1}y$$

### Usage

```python
import torch_dxdt

# Gaussian (RBF) kernel
ker = torch_dxdt.Kernel(sigma=1.0, lmbd=0.1)

# Compute derivative
dx = ker.d(x, t)

# Get smoothed signal
x_smooth = ker.smooth(x, t)
```

### Parameters

- `sigma` (float): Kernel length scale. Controls smoothness.
- `lmbd` (float): Noise variance / regularization.
- `kernel` (str): Kernel type, "gaussian" or "rbf". Default: "gaussian"

### When to Use

- You want probabilistic smoothing
- The signal is reasonably smooth
- You need both smoothed signal and derivative

### Note

This method has O(n³) complexity and can be slow for large datasets.

---

## Kalman Smoother

The `Kalman` class uses Kalman smoothing for differentiation.

### Theory

The Kalman smoother assumes the derivative follows Brownian motion:

$$dx = \sigma dW$$

It finds the maximum likelihood estimate for both the signal and its derivative given noisy observations.

### Usage

```python
import torch_dxdt

# Default smoothing
kal = torch_dxdt.Kalman(alpha=1.0)

# More smoothing
kal = torch_dxdt.Kalman(alpha=10.0)

# Compute derivative
dx = kal.d(x, t)

# Get smoothed signal
x_smooth = kal.smooth(x, t)
```

### Parameters

- `alpha` (float): Regularization parameter. Larger values give smoother results.

### When to Use

- You have a physical model for the derivative (random walk)
- You want statistically principled smoothing
- You need both smoothed signal and derivative

---

## Whittaker-Eilers Smoother

The `Whittaker` class uses penalized least squares with Cholesky decomposition for global smoothing.

### Theory

Whittaker-Eilers smoothing solves the same optimization as Spline but uses efficient Cholesky factorization:

$$\min_z \|x - z\|^2 + \lambda \|D^d z\|^2$$

where $D^d$ is the $d$-th order difference matrix and $\lambda$ controls smoothing. The system $(I + \lambda D^T D)z = x$ is solved using Cholesky decomposition, making it efficient and fully differentiable.

### Usage

```python
import torch_dxdt

# Basic usage
wh = torch_dxdt.Whittaker(lmbda=100.0)

# More smoothing
wh = torch_dxdt.Whittaker(lmbda=1000.0)

# Different difference order
wh = torch_dxdt.Whittaker(lmbda=100.0, d_order=3)

# Get smoothed signal
x_smooth = wh.smooth(x, t)

# Compute derivative
dx = wh.d(x, t)

# Compute multiple derivative orders efficiently
derivs = wh.d_orders(x, t, orders=[0, 1, 2])
x_smooth, dx, d2x = derivs[0], derivs[1], derivs[2]
```

### Parameters

- `lmbda` (float): Smoothing parameter. Larger values give smoother results. Typical range: 1 to 1e6.
- `d_order` (int): Order of the difference penalty. Default: 2
  - 1: Penalizes first differences (piecewise constant)
  - 2: Penalizes second differences (piecewise linear)
  - 3: Penalizes third differences (smoother curves)

### When to Use

- Data is noisy and needs global smoothing
- You want explicit control over smoothness vs. fidelity
- You need efficient computation with full autodiff support
- You want to compute multiple derivative orders efficiently

---

## Method Comparison

| Method | Speed | Noise Robustness | Accuracy (Clean Data) | Boundary Behavior | Differentiable |
|--------|-------|------------------|----------------------|-------------------|----------------|
| Finite Difference | ⚡⚡⚡ | ⚠️ Low | ✅ Good | ⚠️ Moderate | ✅ Yes |
| Savitzky-Golay | ⚡⚡⚡ | ✅ High | ✅ Good | ⚠️ Poor | ✅ Yes |
| Spectral | ⚡⚡⚡ | ⚠️ Medium | ⭐ Excellent | ⭐ Excellent (periodic) | ✅ Yes |
| Spline | ⚡⚡ | ✅ High | ✅ Good | ✅ Good | ✅ Yes |
| Kernel | ⚡ | ✅ High | ✅ Good | ✅ Good | ✅ Yes |
| Kalman | ⚡ | ✅ High | ✅ Good | ✅ Good | ✅ Yes |
| Whittaker | ⚡⚡ | ✅ High | ✅ Good | ⚠️ Moderate | ✅ Yes |

---

## Boundary Behavior

Different methods handle signal boundaries (edges) differently. This is important when derivative accuracy at the start or end of your signal matters.

| Method | Boundary Behavior | Reason |
|--------|-------------------|--------|
| **Spectral** | ⭐ Excellent (periodic) | Assumes periodicity, no boundaries exist |
| **Spline** | ✅ Good | Global fit with natural boundary conditions |
| **Kalman** | ✅ Good | Global smoother, handles edges gracefully |
| **Kernel (GP)** | ✅ Good | Global regression, graceful degradation at edges |
| **Finite Difference** | ⚠️ Moderate | Uses replicate padding, simple forward/backward differences at edges |
| **Whittaker** | ⚠️ Moderate | Global smooth but uses finite differences for derivatives at boundaries |
| **Savitzky-Golay** | ⚠️ Poor | Window-based local fit, edge values are extrapolated with padding |

### Why Boundaries Matter

Local/window-based methods (Savitzky-Golay, Finite Difference) must "pad" the signal at boundaries since they don't have enough neighboring points. Common padding strategies include:

- **Replicate**: Repeat edge values (default) - assumes signal is flat beyond boundaries
- **Reflect**: Mirror the signal - assumes symmetry at boundaries
- **Circular/Periodic**: Wrap around - only valid for truly periodic signals

Global methods (Spline, Kalman, Kernel) fit a model to the entire signal at once, providing more natural handling of boundaries without artificial padding.

### Configuring Boundary Padding

For `SavitzkyGolay`, you can control padding behavior with the `pad_mode` parameter:

```python
import torch_dxdt

# Default: replicate edge values (good for monotonic signals)
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3, pad_mode='replicate')

# Mirror at boundaries (good for symmetric signals)
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3, pad_mode='reflect')

# Wrap around (ONLY for periodic signals)
sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3, pad_mode='circular')
```

**Choosing the right padding mode:**

| Signal Type | Recommended Mode | Example |
|-------------|------------------|---------|
| Monotonic (growing/decaying) | `'replicate'` | Exponential growth, ramp signals |
| Symmetric at edges | `'reflect'` | Signals that level off at boundaries |
| Truly periodic | `'circular'` | Complete sine wave cycles |

### Recommendations for Boundary-Sensitive Applications

If you need accurate derivatives at the **beginning or end** of your signal:

1. **For periodic signals**: Use `Spectral` - it's designed for this case
2. **For non-periodic signals**: Use `Spline`, `Kalman`, or `Kernel` - these are global methods with better boundary handling
3. **If speed is critical**: Use `Savitzky-Golay` with `periodic=True` if your data allows, or trim edge values from results
