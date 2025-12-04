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
import ptdxdt

# 3-point central difference (k=1)
fd = ptdxdt.FiniteDifference(k=1)

# 5-point difference (k=2)
fd = ptdxdt.FiniteDifference(k=2)

# Periodic boundary conditions
fd = ptdxdt.FiniteDifference(k=1, periodic=True)
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
import ptdxdt

# Window of 11 points, cubic polynomial
sg = ptdxdt.SavitzkyGolay(window_length=11, polyorder=3)

# First derivative (default)
dx = sg.d(x, t)

# Higher derivatives
sg2 = ptdxdt.SavitzkyGolay(window_length=11, polyorder=4, deriv=2)
d2x = sg2.d(x, t)
```

### Parameters

- `window_length` (int): Length of the filter window. Must be odd.
- `polyorder` (int): Order of the polynomial. Must be less than window_length.
- `deriv` (int): Derivative order. Default: 1
- `periodic` (bool): If True, uses circular padding. Default: False

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
import ptdxdt
import torch

# Basic usage
spec = ptdxdt.Spectral()

# Higher-order derivatives
spec2 = ptdxdt.Spectral(order=2)

# With frequency filtering
spec_filtered = ptdxdt.Spectral(
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
import ptdxdt

# Light smoothing
spl = ptdxdt.Spline(s=0.01)

# Heavy smoothing
spl = ptdxdt.Spline(s=10.0)

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
import ptdxdt

# Gaussian (RBF) kernel
ker = ptdxdt.Kernel(sigma=1.0, lmbd=0.1)

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
import ptdxdt

# Default smoothing
kal = ptdxdt.Kalman(alpha=1.0)

# More smoothing
kal = ptdxdt.Kalman(alpha=10.0)

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

## Method Comparison

| Method | Speed | Noise Robustness | Accuracy (Clean Data) | Differentiable |
|--------|-------|------------------|----------------------|----------------|
| Finite Difference | ⚡⚡⚡ | ⚠️ Low | ✅ Good | ✅ Yes |
| Savitzky-Golay | ⚡⚡⚡ | ✅ High | ✅ Good | ✅ Yes |
| Spectral | ⚡⚡⚡ | ⚠️ Medium | ⭐ Excellent | ✅ Yes |
| Spline | ⚡⚡ | ✅ High | ✅ Good | ✅ Yes |
| Kernel | ⚡ | ✅ High | ✅ Good | ✅ Yes |
| Kalman | ⚡ | ✅ High | ✅ Good | ✅ Yes |
