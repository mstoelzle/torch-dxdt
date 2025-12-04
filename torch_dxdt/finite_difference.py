"""
Finite Difference numerical differentiation using PyTorch.

This implements symmetric finite differences using nn.Conv1d for efficiency
and differentiability.
"""

import torch
import torch.nn.functional as F

from .base import Derivative


class FiniteDifference(Derivative):
    """
    Compute the symmetric numerical derivative using finite differences.

    Uses the Taylor series expansion to compute coefficients for symmetric
    finite difference schemes of arbitrary window size. The implementation
    uses nn.Conv1d with fixed weights for efficiency and differentiability.

    Args:
        k: Window size parameter. Uses 2k+1 points centered at each location.
           k=1 gives the standard 3-point central difference.
        periodic: If True, treats the data as periodic. Default False.

    Example:
        >>> fd = FiniteDifference(k=1)
        >>> t = torch.linspace(0, 2*torch.pi, 100)
        >>> x = torch.sin(t)
        >>> dx = fd.d(x, t)  # Should approximate cos(t)
    """

    def __init__(self, k: int = 1, periodic: bool = False):
        self.k = k
        self.periodic = periodic
        self._kernel = None
        self._kernel_device = None
        self._kernel_dtype = None

    def _compute_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Compute the finite difference kernel coefficients.

        The coefficients are derived from the Taylor series expansion
        for symmetric finite differences of order 2k+1.
        """
        if (
            self._kernel is not None
            and self._kernel_device == device
            and self._kernel_dtype == dtype
        ):
            return self._kernel

        # Compute coefficients from Taylor series
        # For symmetric FD: alpha_j = 2 * (-1)^(j+1) * C(k,j) / (2j)
        # where C(k,j) = product_{i=1}^{j-1} (k-i+1)/(k+i)
        coeffs = []
        coefficient = 1.0
        for j in range(1, self.k + 1):
            coefficient *= (self.k - j + 1) / (self.k + j)
            alpha_j = 2 * ((-1) ** (j + 1)) * coefficient / (2 * j)
            coeffs.append(alpha_j)

        # Build the kernel: [-alpha_k, ..., -alpha_1, 0, alpha_1, ..., alpha_k]
        kernel = torch.zeros(2 * self.k + 1, device=device, dtype=dtype)
        for j in range(1, self.k + 1):
            kernel[self.k + j] = coeffs[j - 1]
            kernel[self.k - j] = -coeffs[j - 1]

        # Shape for conv1d: (out_channels, in_channels, kernel_size)
        self._kernel = kernel.view(1, 1, -1)
        self._kernel_device = device
        self._kernel_dtype = dtype

        return self._kernel

    def d(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the derivative of x with respect to t using finite differences.

        Args:
            x: Input tensor of shape (..., T) or (T,)
            t: Time points tensor of shape (T,)
            dim: Dimension along which to differentiate. Default -1.

        Returns:
            Derivative tensor of same shape as x.
        """
        # Handle empty input
        if x.numel() == 0:
            return x.clone()

        # Move differentiation dim to last position
        x, original_dim = self._move_dim_to_last(x, dim)
        original_shape = x.shape

        # Get dt (assuming uniform spacing)
        dt = t[1] - t[0]

        # Handle 1D input
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Flatten batch dimensions
        batch_shape = x.shape[:-1]
        T = x.shape[-1]
        x_flat = x.reshape(-1, T)

        # Get kernel
        kernel = self._compute_kernel(x.device, x.dtype)

        # Add channel dimension: (B, 1, T)
        x_conv = x_flat.unsqueeze(1)

        # Apply padding
        if self.periodic:
            # Circular padding for periodic data
            x_padded = F.pad(x_conv, (self.k, self.k), mode="circular")
        else:
            # Replicate padding for non-periodic data
            x_padded = F.pad(x_conv, (self.k, self.k), mode="replicate")

        # Convolve
        dx = F.conv1d(x_padded, kernel)

        # Divide by dt
        dx = dx.squeeze(1) / dt

        # Reshape back
        dx = dx.reshape(*batch_shape, T)

        # Handle boundary conditions for non-periodic case
        if not self.periodic:
            # Use first-order differences at boundaries
            # Left boundary: forward difference
            dx[..., 0] = (x_flat[..., 1] - x_flat[..., 0]).reshape(*batch_shape) / dt
            # Right boundary: backward difference
            dx[..., -1] = (x_flat[..., -1] - x_flat[..., -2]).reshape(*batch_shape) / dt

        # Restore original shape
        if len(original_shape) == 1:
            dx = dx.squeeze(0)

        return self._restore_dim(dx, original_dim)
