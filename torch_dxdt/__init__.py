"""
torch_dxdt - Differentiable Numerical Differentiation in PyTorch

A PyTorch implementation of numerical differentiation methods for noisy time
series data. This package provides differentiable versions of common
differentiation algorithms, allowing them to be used as part of neural network
training pipelines.

Available Methods:
    - FiniteDifference: Symmetric finite differences using conv1d
    - SavitzkyGolay: Savitzky-Golay polynomial filtering
    - Spectral: FFT-based spectral differentiation
    - Spline: Smoothing spline differentiation
    - Kernel: Gaussian process kernel methods
    - Kalman: Kalman smoother for probabilistic differentiation
    - Whittaker: Whittaker-Eilers global smoother with penalized least squares

Example:
    >>> import torch
    >>> import torch_dxdt
    >>>
    >>> t = torch.linspace(0, 2 * torch.pi, 100)
    >>> x = torch.sin(t) + 0.1 * torch.randn(100)
    >>>
    >>> # Functional interface
    >>> dx = torch_dxdt.dxdt(x, t, kind="savitzky_golay", window_length=11, polyorder=3)
    >>>
    >>> # Object-oriented interface
    >>> sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3)
    >>> dx = sg.d(x, t)
"""

__version__ = "0.1.0"

# Import base class
from .base import Derivative

# Import all methods
from .finite_difference import FiniteDifference
from .kalman import Kalman
from .kernel import Kernel
from .savitzky_golay import SavitzkyGolay
from .spectral import Spectral
from .spline_method import Spline
from .whittaker import Whittaker

# Method registry
_methods = {
    "finite_difference": FiniteDifference,
    "savitzky_golay": SavitzkyGolay,
    "spectral": Spectral,
    "spline": Spline,
    "kernel": Kernel,
    "kalman": Kalman,
    "whittaker": Whittaker,
}

# Default method
_default_method = "finite_difference"
_default_kwargs = {"k": 1}


def dxdt(x, t, kind=None, dim=-1, **kwargs):
    """
    Compute the derivative of x with respect to t using the specified method.

    This is the functional interface to the differentiation methods. It creates
    an instance of the specified method and calls its `d` method.

    Args:
        x: torch.Tensor of shape (..., T) containing the signal values.
        t: torch.Tensor of shape (T,) containing the time points.
        kind: Method name. One of:
            - "finite_difference": Symmetric finite differences
            - "savitzky_golay": Savitzky-Golay polynomial filtering
            - "spectral": FFT-based spectral differentiation
            - "spline": Smoothing spline differentiation
            - "kernel": Gaussian process kernel methods
            - "kalman": Kalman smoother
            If None, uses finite_difference with k=1.
        dim: Dimension along which to differentiate. Default -1.
        **kwargs: Keyword arguments passed to the method constructor.

    Returns:
        torch.Tensor of same shape as x containing dx/dt.

    Available kwargs by method:
        - finite_difference: k (window size), periodic (bool)
        - savitzky_golay: window_length, polyorder, order, periodic
        - spectral: order, filter_func
        - spline: s (smoothing parameter)
        - kernel: sigma, lmbd, kernel
        - kalman: alpha
        - whittaker: lmbda (smoothing parameter), d_order (difference order)

    Example:
        >>> t = torch.linspace(0, 2*torch.pi, 100)
        >>> x = torch.sin(t)
        >>> dx = dxdt(x, t, kind="finite_difference", k=1)
    """
    if kind is None:
        kind = _default_method
        kwargs = {**_default_kwargs, **kwargs}

    if kind not in _methods:
        raise ValueError(
            f"Unknown method '{kind}'. Available methods: {list(_methods.keys())}"
        )

    method_class = _methods[kind]
    method = method_class(**kwargs)
    return method.d(x, t, dim=dim)


def smooth_x(x, t, kind=None, dim=-1, **kwargs):
    """
    Compute the smoothed version of x using the specified method.

    Not all methods support smoothing. Methods that do:
    - spline
    - kernel
    - kalman
    - whittaker

    Args:
        x: torch.Tensor of shape (..., T) containing the signal values.
        t: torch.Tensor of shape (T,) containing the time points.
        kind: Method name (see dxdt for available methods).
        dim: Dimension along which to smooth. Default -1.
        **kwargs: Keyword arguments passed to the method constructor.

    Returns:
        torch.Tensor of same shape as x containing the smoothed signal.

    Raises:
        NotImplementedError: If the specified method does not support smoothing.
    """
    if kind is None:
        kind = "spline"
        kwargs = {"s": 0.01, **kwargs}

    if kind not in _methods:
        raise ValueError(
            f"Unknown method '{kind}'. Available methods: {list(_methods.keys())}"
        )

    method_class = _methods[kind]
    method = method_class(**kwargs)
    return method.smooth(x, t, dim=dim)


def dxdt_orders(x, t, orders=(1, 2), kind=None, dim=-1, **kwargs):
    """
    Compute multiple derivative orders simultaneously.

    This function efficiently computes multiple derivative orders in a single
    pass, avoiding redundant computation. For methods that support it
    (e.g., SavitzkyGolay), shared computation is reused across orders.

    Args:
        x: torch.Tensor of shape (..., T) containing the signal values.
        t: torch.Tensor of shape (T,) containing the time points.
        orders: Sequence of derivative orders to compute. Default (1, 2).
            Order 0 returns the smoothed signal (if supported).
        kind: Method name. Currently best supported by:
            - "savitzky_golay": Most efficient, computes all orders in one pass
            Other methods fall back to computing each order separately.
        dim: Dimension along which to differentiate. Default -1.
        **kwargs: Keyword arguments passed to the method constructor.

    Returns:
        dict mapping order -> torch.Tensor of same shape as x.

    Example:
        >>> t = torch.linspace(0, 2*torch.pi, 100)
        >>> x = torch.sin(t)
        >>> derivs = dxdt_orders(x, t, orders=[0, 1, 2],
        ...                      kind="savitzky_golay",
        ...                      window_length=11, polyorder=4)
        >>> x_smooth = derivs[0]  # Smoothed signal
        >>> dx = derivs[1]        # First derivative
        >>> d2x = derivs[2]       # Second derivative
    """
    if kind is None:
        kind = "savitzky_golay"
        kwargs = {"window_length": 11, "polyorder": 4, **kwargs}

    if kind not in _methods:
        raise ValueError(
            f"Unknown method '{kind}'. Available methods: {list(_methods.keys())}"
        )

    method_class = _methods[kind]
    method = method_class(**kwargs)
    return method.d_orders(x, t, orders=orders, dim=dim)


# List available methods
def available_methods():
    """Return a list of available differentiation methods."""
    return list(_methods.keys())


__all__ = [
    # Version
    "__version__",
    # Base class
    "Derivative",
    # Methods
    "FiniteDifference",
    "SavitzkyGolay",
    "Spectral",
    "Spline",
    "Kernel",
    "Kalman",
    "Whittaker",
    # Functional interfaces
    "dxdt",
    "dxdt_orders",
    "smooth_x",
    "available_methods",
]
