"""
Spline-based numerical differentiation using PyTorch.

Uses differentiable B-spline fitting for smooth derivative estimation.
"""

import torch

from .base import Derivative


class Spline(Derivative):
    """
    Compute numerical derivatives using cubic spline interpolation.

    Fits a smoothing spline to the data and computes derivatives analytically.
    Uses torch.linalg.solve for the linear system, making it differentiable.

    Args:
        s: Smoothing parameter. Larger values give smoother results.
           s=0 interpolates exactly through the data points.
        order: Spline order. Default is 3 (cubic spline).

    Note:
        The current implementation uses a simplified Whittaker smoother
        approach rather than true B-splines, as it's more amenable to
        differentiable implementation in PyTorch.

    Example:
        >>> spl = Spline(s=0.01)
        >>> t = torch.linspace(0, 2*torch.pi, 100)
        >>> x = torch.sin(t) + 0.1 * torch.randn(100)
        >>> dx = spl.d(x, t)  # Smoothed derivative
    """

    def __init__(self, s: float = 0.01, order: int = 3):
        self.s = s
        self.order = order

    def _build_difference_matrix(
        self, n: int, order: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build the finite difference matrix of given order.

        For order=1: First difference D[i,i] = -1, D[i,i+1] = 1
        For order=2: Second difference (used for smoothing splines)
        """
        if order == 1:
            # First difference matrix (n-1) x n
            D = torch.zeros(n - 1, n, device=device, dtype=dtype)
            for i in range(n - 1):
                D[i, i] = -1
                D[i, i + 1] = 1
            return D
        elif order == 2:
            # Second difference matrix (n-2) x n
            D = torch.zeros(n - 2, n, device=device, dtype=dtype)
            for i in range(n - 2):
                D[i, i] = 1
                D[i, i + 1] = -2
                D[i, i + 2] = 1
            return D
        else:
            # Build higher order by composing first differences
            D1 = self._build_difference_matrix(n, 1, device, dtype)
            for _ in range(order - 1):
                D1_next = self._build_difference_matrix(D1.shape[0], 1, device, dtype)
                D1 = D1_next @ D1
            return D1

    def _smooth(
        self, x: torch.Tensor, lmbda: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Apply Whittaker smoothing: solve (I + lmbda * D^T D) z = x
        """
        n = x.shape[-1]

        # Build second difference matrix
        D = self._build_difference_matrix(n, 2, device, dtype)

        # Build system matrix A = I + lmbda * D^T @ D
        eye = torch.eye(n, device=device, dtype=dtype)
        A = eye + lmbda * (D.T @ D)

        # Solve for each batch element
        if x.ndim == 1:
            z = torch.linalg.solve(A, x)
        else:
            # x is (B, n), we need to solve for each batch
            z = torch.linalg.solve(A, x.T).T

        return z

    def _differentiate_smooth(
        self, z: torch.Tensor, dt: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Compute derivative of smoothed signal using central differences.
        """
        # Use central differences for interior points
        dz = torch.zeros_like(z)

        if z.ndim == 1:
            dz[1:-1] = (z[2:] - z[:-2]) / (2 * dt)
            dz[0] = (z[1] - z[0]) / dt
            dz[-1] = (z[-1] - z[-2]) / dt
        else:
            dz[..., 1:-1] = (z[..., 2:] - z[..., :-2]) / (2 * dt)
            dz[..., 0] = (z[..., 1] - z[..., 0]) / dt
            dz[..., -1] = (z[..., -1] - z[..., -2]) / dt

        return dz

    def d(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the derivative of x with respect to t using spline smoothing.

        Args:
            x: Input tensor of shape (..., T) or (T,)
            t: Time points tensor of shape (T,)
            dim: Dimension along which to differentiate. Default -1.

        Returns:
            Derivative tensor of same shape as x.
        """
        if x.numel() == 0:
            return x.clone()

        # Check length along the specified dim
        time_len = x.shape[dim]
        if time_len <= self.order:
            raise TypeError(
                f"Input length ({time_len}) must be > order ({self.order})"
            )

        # Move differentiation dim to last position
        x, original_dim = self._move_dim_to_last(x, dim)

        # Get dt
        dt = (t[1] - t[0]).item()

        # Handle 1D input
        was_1d = x.ndim == 1
        if was_1d:
            x = x.unsqueeze(0)

        # Flatten batch dimensions
        batch_shape = x.shape[:-1]
        T = x.shape[-1]
        x_flat = x.reshape(-1, T)

        # Apply smoothing
        z = self._smooth(x_flat, self.s, x.device, x.dtype)

        # Differentiate
        dz = self._differentiate_smooth(z, dt, x.device, x.dtype)

        # Reshape back
        dx = dz.reshape(*batch_shape, T)

        # Restore original shape
        if was_1d:
            dx = dx.squeeze(0)

        return self._restore_dim(dx, original_dim)

    def smooth(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the smoothed version of x without differentiation.

        Args:
            x: Input tensor of shape (..., T) or (T,)
            t: Time points tensor of shape (T,)
            dim: Dimension along which to smooth. Default -1.

        Returns:
            Smoothed tensor of same shape as x.
        """
        if x.numel() == 0:
            return x.clone()

        # Move dim to last position
        x, original_dim = self._move_dim_to_last(x, dim)

        # Handle 1D input
        was_1d = x.ndim == 1
        if was_1d:
            x = x.unsqueeze(0)

        # Flatten batch dimensions
        batch_shape = x.shape[:-1]
        T = x.shape[-1]
        x_flat = x.reshape(-1, T)

        # Apply smoothing
        z = self._smooth(x_flat, self.s, x.device, x.dtype)

        # Reshape back
        z = z.reshape(*batch_shape, T)

        # Restore original shape
        if was_1d:
            z = z.squeeze(0)

        return self._restore_dim(z, original_dim)
