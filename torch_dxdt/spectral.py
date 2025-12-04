"""
Spectral differentiation using PyTorch FFT.

Computes derivatives by multiplying by frequency in Fourier space.
"""

import math

import torch

from .base import Derivative


class Spectral(Derivative):
    """
    Compute numerical derivatives using spectral (Fourier) methods.

    Transforms to Fourier space, multiplies by (i * omega)^order, and transforms back.
    This method is very accurate for smooth, periodic data.

    Args:
        order: Order of the derivative. Default is 1.
        filter_func: Optional function to filter frequencies before differentiation.
                    Takes wavenumbers as input and returns weights.
                    Example: lambda k: (torch.abs(k) < 10).float()

    Note:
        - Assumes the data is periodic over the sample interval.
        - Works best for smooth, band-limited signals.
        - For non-periodic data, consider windowing or other methods.

    Example:
        >>> spec = Spectral(order=1)
        >>> t = torch.linspace(0, 2*torch.pi, 100, endpoint=False)
        >>> x = torch.sin(t)
        >>> dx = spec.d(x, t)  # Should approximate cos(t)
    """

    def __init__(self, order: int = 1, filter_func=None):
        self.order = order
        self.filter_func = filter_func

    def d(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the derivative of x with respect to t using spectral methods.

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

        # Handle 1D input
        was_1d = x.ndim == 1
        if was_1d:
            x = x.unsqueeze(0)

        T = x.shape[-1]

        # Compute the period
        t[-1] - t[0] + (t[1] - t[0])  # Full period including endpoint

        # Compute wavenumbers
        # For rfft, we only get positive frequencies up to Nyquist
        freqs = torch.fft.rfftfreq(T, d=(t[1] - t[0]).item(), device=x.device)
        omega = 2 * math.pi * freqs

        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)

        # Multiply by (i * omega)^order
        # For first derivative: multiply by i * omega
        # For second derivative: multiply by -omega^2, etc.
        multiplier = (1j * omega) ** self.order

        # Apply filter if provided
        if self.filter_func is not None:
            # Convert frequencies to integer wavenumbers for the filter
            k = torch.arange(len(omega), device=x.device, dtype=x.dtype)
            weights = self.filter_func(k)
            multiplier = multiplier * weights

        # Apply differentiation in frequency space
        dx_fft = x_fft * multiplier

        # Inverse FFT
        dx = torch.fft.irfft(dx_fft, n=T, dim=-1)

        # Restore original shape
        if was_1d:
            dx = dx.squeeze(0)

        return self._restore_dim(dx, original_dim)
