"""
Kalman filter/smoother for numerical differentiation.

Uses the Kalman smoother to estimate derivatives under a Brownian motion model.
"""

import torch

from .base import Derivative


class Kalman(Derivative):
    """
    Compute numerical derivatives using Kalman smoothing.

    The Kalman smoother finds the maximum likelihood estimator for a process
    whose derivative follows Brownian motion. This provides smooth,
    probabilistically-principled derivative estimates.

    The method minimizes:
        ||H*x - z||^2 + alpha * ||G*(x, dx)||_Q^2

    where z is the noisy observation, x is the smoothed signal,
    dx is its derivative, and Q encodes the Brownian motion covariance.

    Args:
        alpha: Regularization parameter. Larger values give smoother results.
               If None, uses a default value of 1.

    Note:
        This implementation uses a batched linear solver and is fully
        differentiable, but may be slow for very long time series.

    Example:
        >>> kal = Kalman(alpha=1.0)
        >>> t = torch.linspace(0, 2*torch.pi, 100)
        >>> x = torch.sin(t) + 0.1 * torch.randn(100)
        >>> dx = kal.d(x, t)  # Kalman-smoothed derivative
    """

    def __init__(self, alpha: float = 1.0):
        if alpha is None:
            alpha = 1.0
        self.alpha = alpha

    def _solve_kalman(
        self, z: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the Kalman smoothing problem.

        Following the formulation from the derivative package:
        - State is ordered as [dx_0, x_0, dx_1, x_1, ...]
        - G encodes the dynamics constraint
        - H extracts the position states

        Returns:
            Tuple of (x_hat, x_dot_hat) where x_hat is the smoothed signal
            and x_dot_hat is the derivative estimate.
        """
        n = t.shape[0]
        device = t.device
        dtype = t.dtype

        # Compute time differences
        delta_times = t[1:] - t[:-1]

        # Build Q matrices for each time step
        # Q = [[dt, dt^2/2], [dt^2/2, dt^3/3]]
        Qs = []
        for dt in delta_times:
            Q = torch.tensor(
                [[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]], device=device, dtype=dtype
            )
            Qs.append(Q)

        # Build block diagonal Q inverse
        Q_inv_blocks = [torch.linalg.inv(Q) for Q in Qs]
        Q_inv = torch.block_diag(*Q_inv_blocks) * self.alpha

        # Symmetrize to avoid numerical issues
        Q_inv = (Q_inv + Q_inv.T) / 2

        # Build G matrix (dynamics constraint)
        # State ordering: [dx_0, x_0, dx_1, x_1, ...]
        # G_left: -[[1, 0], [dt, 1]] blocks for [dx_i, x_i]
        # G_right: +I blocks for [dx_{i+1}, x_{i+1}]
        # G is 2*(n-1) x 2*n
        G = torch.zeros(2 * (n - 1), 2 * n, device=device, dtype=dtype)

        for i, dt in enumerate(delta_times):
            # -[[1, 0], [dt, 1]] block for current state [dx_i, x_i]
            G[2 * i, 2 * i] = -1  # -dx_i coefficient
            G[2 * i, 2 * i + 1] = 0  # -x_i coefficient for dx equation
            G[2 * i + 1, 2 * i] = -dt  # -dt*dx_i for x equation
            G[2 * i + 1, 2 * i + 1] = -1  # -x_i coefficient for x equation

            # +I block for next state [dx_{i+1}, x_{i+1}]
            G[2 * i, 2 * (i + 1)] = 1
            G[2 * i + 1, 2 * (i + 1) + 1] = 1

        # Build H matrix (observation model)
        # H extracts position x from state [dx, x] - position is at odd indices
        # H is n x 2n
        H = torch.zeros(n, 2 * n, device=device, dtype=dtype)
        for i in range(n):
            H[i, 2 * i + 1] = 1  # Extract x_i from position 2*i+1

        # Solve the system: (H^T H + G^T Q^{-1} G) sol = H^T z
        lhs = H.T @ H + G.T @ Q_inv @ G

        # Handle batch dimension
        if z.ndim == 1:
            rhs = H.T @ z
            sol = torch.linalg.solve(lhs, rhs)
            # Extract x_hat and x_dot_hat
            # Position x is at odd indices (1, 3, 5, ...)
            # Velocity dx is at even indices (0, 2, 4, ...)
            x_hat = sol[1::2]  # Odd indices - position
            x_dot_hat = sol[0::2]  # Even indices - velocity
        else:
            # z is (B, n)
            rhs = H.T @ z.T  # (2n, B)
            sol = torch.linalg.solve(lhs, rhs)  # (2n, B)
            x_hat = sol[1::2, :].T  # (B, n)
            x_dot_hat = sol[0::2, :].T  # (B, n)

        return x_hat, x_dot_hat

        return x_hat, x_dot_hat

    def d(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the derivative of x with respect to t using Kalman smoothing.

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

        # Solve Kalman smoothing
        _, x_dot_hat = self._solve_kalman(x_flat, t)

        # Reshape back
        dx = x_dot_hat.reshape(*batch_shape, T)

        # Restore original shape
        if was_1d:
            dx = dx.squeeze(0)

        return self._restore_dim(dx, original_dim)

    def smooth(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the smoothed version of x using Kalman smoothing.

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

        # Solve Kalman smoothing
        x_hat, _ = self._solve_kalman(x_flat, t)

        # Reshape back
        z = x_hat.reshape(*batch_shape, T)

        # Restore original shape
        if was_1d:
            z = z.squeeze(0)

        return self._restore_dim(z, original_dim)
