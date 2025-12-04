"""
Savitzky-Golay filter numerical differentiation using PyTorch.

Uses scipy to compute the filter coefficients and nn.Conv1d for efficient
differentiable application.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from scipy.signal import savgol_coeffs

from .base import Derivative


class SavitzkyGolay(Derivative):
    """
    Compute numerical derivatives using Savitzky-Golay filtering.

    Fits a polynomial of given order to a window of points and takes the
    derivative of the polynomial. This provides smoothing while computing
    derivatives, making it robust to noise.

    The implementation uses scipy to compute the filter coefficients once,
    then applies them efficiently using PyTorch convolution.

    This method is particularly efficient for computing multiple derivative
    orders at once via `d_orders()`, as the same polynomial fit can yield
    all derivative orders in a single convolution pass.

    Args:
        window_length: Length of the filter window (must be odd and > polyorder).
        polyorder: Order of the polynomial used to fit the samples.
        order: Order of the derivative to compute. Default is 1.
            Use order=1 for first derivative, order=2 for second derivative, etc.
        deriv: Alias for order (deprecated, use order instead).
        pad_mode: Padding mode for handling boundaries. Options are:
            - 'replicate': Repeat the edge value (default, good for monotonic signals)
            - 'reflect': Mirror the signal at the boundary (good for symmetric signals)
            - 'circular': Wrap around (only for periodic signals)
        periodic: Deprecated, use pad_mode='circular' instead. If True, sets
            pad_mode='circular'. Default False.

    Example:
        >>> sg = SavitzkyGolay(window_length=11, polyorder=4, order=1)
        >>> t = torch.linspace(0, 2*torch.pi, 100)
        >>> x = torch.sin(t) + 0.1 * torch.randn(100)
        >>> dx = sg.d(x, t)  # First derivative
        >>> # Compute multiple orders efficiently:
        >>> derivs = sg.d_orders(x, t, orders=[0, 1, 2])
        >>> x_smooth, dx, d2x = derivs[0], derivs[1], derivs[2]
        >>> # Use reflect padding for better boundary behavior:
        >>> sg_reflect = SavitzkyGolay(window_length=11, polyorder=4, pad_mode='reflect')
    """

    VALID_PAD_MODES = ('replicate', 'reflect', 'circular')

    def __init__(
        self,
        window_length: int,
        polyorder: int,
        order: int = 1,
        deriv: int | None = None,
        pad_mode: str = 'replicate',
        periodic: bool = False,
    ):
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd.")
        if window_length <= polyorder:
            raise ValueError("window_length must be greater than polyorder.")

        # Handle deriv as deprecated alias for order
        if deriv is not None:
            order = deriv

        if order > polyorder:
            raise ValueError(
                f"order ({order}) must be <= polyorder ({polyorder}) "
                "for meaningful derivative estimation."
            )

        # Handle pad_mode and periodic (periodic is deprecated)
        if periodic:
            pad_mode = 'circular'
        
        if pad_mode not in self.VALID_PAD_MODES:
            raise ValueError(
                f"pad_mode must be one of {self.VALID_PAD_MODES}, "
                f"got '{pad_mode}'"
            )

        self.window_length = window_length
        self.polyorder = polyorder
        self.order = order
        self.deriv = order  # Keep for backward compatibility
        self.pad_mode = pad_mode
        self.periodic = pad_mode == 'circular'  # Keep for backward compatibility
        self.pad_size = window_length // 2

        # Coefficients will be computed lazily based on delta (dt)
        self._coeffs_cache = {}

    def _get_kernel(
        self,
        delta: float,
        device: torch.device,
        dtype: torch.dtype,
        order: int | None = None,
    ) -> torch.Tensor:
        """Get the Savitzky-Golay filter kernel for the given delta and order."""
        if order is None:
            order = self.order
        cache_key = (delta, device, dtype, order)
        if cache_key in self._coeffs_cache:
            return self._coeffs_cache[cache_key]

        # Compute coefficients using scipy
        coeffs = savgol_coeffs(
            self.window_length,
            self.polyorder,
            deriv=order,
            delta=delta,
        )

        # Flip for convolution (scipy returns filter, conv1d expects kernel)
        coeffs = coeffs[::-1].copy()

        # Convert to tensor
        kernel = torch.tensor(coeffs, dtype=dtype, device=device).view(1, 1, -1)

        self._coeffs_cache[cache_key] = kernel
        return kernel

    def d(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the derivative of x with respect to t using Savitzky-Golay.

        Args:
            x: Input tensor of shape (..., T) or (T,)
            t: Time points tensor of shape (T,)
            dim: Dimension along which to differentiate. Default -1.

        Returns:
            Derivative tensor of same shape as x.
        """
        if x.numel() == 0:
            return x.clone()

        # Move differentiation dim to last position
        x, original_dim = self._move_dim_to_last(x, dim)
        original_shape = x.shape

        # Get dt
        dt = (t[1] - t[0]).item()

        # Handle 1D input
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Flatten batch dimensions
        batch_shape = x.shape[:-1]
        T = x.shape[-1]
        x_flat = x.reshape(-1, T)

        # Get kernel
        kernel = self._get_kernel(dt, x.device, x.dtype)

        # Add channel dimension: (B, 1, T)
        x_conv = x_flat.unsqueeze(1)

        # Apply padding
        pad = (self.pad_size, self.pad_size)
        x_padded = F.pad(x_conv, pad, mode=self.pad_mode)

        # Convolve
        dx = F.conv1d(x_padded, kernel)

        # Remove channel dimension and reshape
        dx = dx.squeeze(1).reshape(*batch_shape, T)

        # Restore original shape
        if len(original_shape) == 1:
            dx = dx.squeeze(0)

        return self._restore_dim(dx, original_dim)

    def _compute_order(
        self, x: torch.Tensor, t: torch.Tensor, order: int, dim: int
    ) -> torch.Tensor:
        """Compute a specific derivative order."""
        if order > self.polyorder:
            raise ValueError(f"order ({order}) must be <= polyorder ({self.polyorder})")

        # Save and temporarily override self.order
        original_order = self.order
        self.order = order
        try:
            result = self.d(x, t, dim=dim)
        finally:
            self.order = original_order
        return result

    def d_orders(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        orders: Sequence[int] = (1, 2),
        dim: int = -1,
    ) -> dict[int, torch.Tensor]:
        """
        Compute multiple derivative orders simultaneously and efficiently.

        This method computes all requested derivative orders with shared
        preprocessing (dim permutation, padding), avoiding redundant
        computation. This is more efficient than calling `d()` multiple times.

        Args:
            x: Input tensor of shape (..., T) or (T,)
            t: Time points tensor of shape (T,)
            orders: Sequence of derivative orders to compute. Default is (1, 2).
                Order 0 returns the smoothed signal. Maximum order is polyorder.
            dim: Dimension along which to differentiate. Default -1.

        Returns:
            Dictionary mapping order -> derivative tensor.
            Each tensor has the same shape as x.

        Example:
            >>> sg = SavitzkyGolay(window_length=11, polyorder=4)
            >>> derivs = sg.d_orders(x, t, orders=[0, 1, 2])
            >>> x_smooth = derivs[0]  # Smoothed signal
            >>> dx = derivs[1]        # First derivative
            >>> d2x = derivs[2]       # Second derivative
        """
        orders = list(orders)
        max_order = max(orders)

        if max_order > self.polyorder:
            raise ValueError(
                f"Maximum order ({max_order}) must be <= polyorder ({self.polyorder})"
            )

        if x.numel() == 0:
            return {order: x.clone() for order in orders}

        # Move differentiation dim to last position
        x_moved, original_dim = self._move_dim_to_last(x, dim)
        was_1d = x_moved.ndim == 1

        # Get dt
        dt = (t[1] - t[0]).item()

        # Handle 1D input
        if was_1d:
            x_moved = x_moved.unsqueeze(0)

        # Flatten batch dimensions
        batch_shape = x_moved.shape[:-1]
        T = x_moved.shape[-1]
        x_flat = x_moved.reshape(-1, T)

        # Add channel dimension: (B, 1, T)
        x_conv = x_flat.unsqueeze(1)

        # Apply padding once (shared across all orders)
        pad = (self.pad_size, self.pad_size)
        x_padded = F.pad(x_conv, pad, mode=self.pad_mode)

        # Compute each derivative order
        results = {}
        for order in orders:
            # Get kernel for this order
            kernel = self._get_kernel(dt, x_moved.device, x_moved.dtype, order)

            # Convolve
            dx = F.conv1d(x_padded, kernel)

            # Remove channel dimension and reshape
            dx = dx.squeeze(1).reshape(*batch_shape, T)

            # Restore original shape
            if was_1d:
                dx = dx.squeeze(0)

            results[order] = self._restore_dim(dx, original_dim)

        return results

    def smooth(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the smoothed version of x using Savitzky-Golay filtering.

        This is equivalent to computing the 0th order derivative, which
        returns the polynomial fit (smoothed signal).

        Args:
            x: Input tensor of shape (..., T) or (T,)
            t: Time points tensor of shape (T,)
            dim: Dimension along which to smooth. Default -1.

        Returns:
            Smoothed tensor of same shape as x.
        """
        return self._compute_order(x, t, order=0, dim=dim)


# Alias for backward compatibility
SavGolFilter = SavitzkyGolay
