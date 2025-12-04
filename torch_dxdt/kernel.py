"""
Kernel-based numerical differentiation using Gaussian Processes.

Uses kernel (covariance) functions to smooth data and compute derivatives.
"""

import torch

from .base import Derivative


class Kernel(Derivative):
    """
    Compute numerical derivatives using kernel (Gaussian Process) methods.

    Fits a Gaussian process to the data using a specified kernel function,
    then computes derivatives from the posterior mean.

    Args:
        sigma: Kernel length scale parameter. Controls smoothness.
        lmbd: Noise variance (regularization) parameter.
        kernel: Kernel type. Currently supports "gaussian" or "rbf".

    Note:
        This method is differentiable but can be slow for large datasets
        due to the O(n^3) complexity of solving the linear system.

    Example:
        >>> ker = Kernel(sigma=1.0, lmbd=0.1)
        >>> t = torch.linspace(0, 2*torch.pi, 100)
        >>> x = torch.sin(t) + 0.1 * torch.randn(100)
        >>> dx = ker.d(x, t)  # Kernel-smoothed derivative
    """

    def __init__(self, sigma: float = 1.0, lmbd: float = 0.1, kernel: str = "gaussian"):
        if kernel not in ("gaussian", "rbf"):
            raise ValueError("kernel must be 'gaussian' or 'rbf'")
        self.sigma = sigma
        self.lmbd = lmbd
        self.kernel = kernel

    def _gaussian_kernel(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Compute the Gaussian (RBF) kernel matrix."""
        # t1: (n,), t2: (m,) -> output: (n, m)
        diff = t1.unsqueeze(1) - t2.unsqueeze(0)
        return torch.exp(-(diff**2) / (2 * self.sigma**2))

    def _gaussian_kernel_dt(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the kernel with respect to the first argument.

        d/dt1 k(t1, t2) = -(t1 - t2) / sigma^2 * k(t1, t2)

        This gives the derivative of the prediction at t1.
        """
        diff = t1.unsqueeze(1) - t2.unsqueeze(0)
        k = torch.exp(-(diff**2) / (2 * self.sigma**2))
        return -(diff / self.sigma**2) * k

    def _solve_kernel_regression(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the kernel regression problem and return smoothed values and derivatives.

        Returns:
            Tuple of (x_hat, x_dot_hat) where x_hat is the smoothed signal
            and x_dot_hat is the derivative estimate.
        """
        n = t.shape[0]

        # Compute kernel matrix K(t, t)
        K = self._gaussian_kernel(t, t)

        # Compute derivative kernel K'(t, t) with respect to second argument
        K_dt = self._gaussian_kernel_dt(t, t)

        # Add noise term: K_noisy = K + lmbd * I
        K_noisy = K + self.lmbd * torch.eye(n, device=t.device, dtype=t.dtype)

        # Solve for alpha: K_noisy @ alpha = x
        # Handle batch dimension
        if x.ndim == 1:
            alpha = torch.linalg.solve(K_noisy, x)
            x_hat = K @ alpha
            x_dot_hat = K_dt @ alpha
        else:
            # x is (B, n), solve for each batch element
            alpha = torch.linalg.solve(K_noisy, x.T).T  # (B, n)
            x_hat = (K @ alpha.T).T  # (B, n)
            x_dot_hat = (K_dt @ alpha.T).T  # (B, n)

        return x_hat, x_dot_hat

    def d(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the derivative of x with respect to t using kernel methods.

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

        # Flatten batch dimensions
        batch_shape = x.shape[:-1]
        T = x.shape[-1]
        x_flat = x.reshape(-1, T)

        # Solve kernel regression
        _, x_dot_hat = self._solve_kernel_regression(x_flat, t)

        # Reshape back
        dx = x_dot_hat.reshape(*batch_shape, T)

        # Restore original shape
        if was_1d:
            dx = dx.squeeze(0)

        return self._restore_dim(dx, original_dim)

    def smooth(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the smoothed version of x using kernel regression.

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

        # Solve kernel regression
        x_hat, _ = self._solve_kernel_regression(x_flat, t)

        # Reshape back
        z = x_hat.reshape(*batch_shape, T)

        # Restore original shape
        if was_1d:
            z = z.squeeze(0)

        return self._restore_dim(z, original_dim)
