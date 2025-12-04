"""
Whittaker-Eilers Global Smoother/Derivative using PyTorch.

Implements the Whittaker-Eilers smoother which uses penalized least squares
to smooth data. Derivatives can be computed by applying difference matrices
to the smoothed signal.
"""

from collections.abc import Sequence

import torch

from .base import Derivative


class WhittakerFunction(torch.autograd.Function):
    """
    Custom autograd function for Whittaker-Eilers smoothing.

    Solves the linear system (I + lmbda * D^T D) z = x where D is a
    second-order difference matrix. Uses Cholesky factorization for
    efficient solving and reuses the factorization in the backward pass.
    """

    @staticmethod
    def forward(ctx, x, lmbda, d_order):
        """
        Solve (I + lmbda * D^T D) z = x using Cholesky factorization.

        Args:
            ctx: Autograd context for saving tensors.
            x: Input tensor of shape (B, T) where B is batch size.
            lmbda: Smoothing parameter (larger = smoother).
            d_order: Order of the difference matrix (2 for standard smoothing).

        Returns:
            Smoothed tensor z of shape (B, T).
        """
        B, T = x.shape
        device = x.device
        dtype = x.dtype

        # 1. Construct D (difference matrix of order d_order)
        D = WhittakerFunction._build_difference_matrix(T, d_order, device, dtype)

        # 2. Construct A = I + lmbda * D^T D
        A = torch.eye(T, device=device, dtype=dtype) + lmbda * (D.T @ D)

        # 3. Cholesky Decomposition (A = L L^T)
        L, info = torch.linalg.cholesky_ex(A)
        if info.any():
            raise RuntimeError(
                "Cholesky decomposition failed. Matrix not positive definite."
            )

        # 4. Solve for z: A z^T = x^T  ->  L L^T z^T = x^T
        # x is (B, T), x.T is (T, B)
        # cholesky_solve solves A @ X = B, where A is (n, n) and B is (n, k)
        # So we solve for z^T: L @ L^T @ z^T = x^T
        # Result z^T is (T, B), then z = (z^T)^T = (B, T)
        z = torch.cholesky_solve(x.T, L).T

        # 5. Save L for the backward pass
        ctx.save_for_backward(L)

        return z

    @staticmethod
    def _build_difference_matrix(
        n: int, order: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build a difference matrix of specified order.

        Args:
            n: Size of the signal.
            order: Order of differences (1=first diff, 2=second diff, etc.)
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            Difference matrix D of shape (n - order, n).
        """
        if order == 1:
            # First order difference: D[i, :] = [0..0, -1, 1, 0..0]
            D = torch.zeros(n - 1, n, device=device, dtype=dtype)
            for i in range(n - 1):
                D[i, i] = -1
                D[i, i + 1] = 1
        elif order == 2:
            # Second order difference: D[i, :] = [0..0, 1, -2, 1, 0..0]
            D = torch.zeros(n - 2, n, device=device, dtype=dtype)
            for i in range(n - 2):
                D[i, i] = 1
                D[i, i + 1] = -2
                D[i, i + 2] = 1
        elif order == 3:
            # Third order difference
            D = torch.zeros(n - 3, n, device=device, dtype=dtype)
            for i in range(n - 3):
                D[i, i] = -1
                D[i, i + 1] = 3
                D[i, i + 2] = -3
                D[i, i + 3] = 1
        else:
            raise ValueError(f"Difference order {order} not supported (use 1, 2, or 3)")

        return D

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the linear system solve.

        Since A is symmetric (A = A^T), the backward pass solves:
        A * grad_x = grad_output using the cached Cholesky factor L.
        """
        (L,) = ctx.saved_tensors

        # grad_output is (B, T), solve A * grad_x = grad_output
        # Same as forward: use cholesky_solve with transposed input
        grad_x = torch.cholesky_solve(grad_output.T, L).T

        # Gradient w.r.t lmbda and d_order are not computed
        return grad_x, None, None


class Whittaker(Derivative):
    """
    Compute numerical derivatives using Whittaker-Eilers smoothing.

    The Whittaker-Eilers smoother uses penalized least squares with a
    difference penalty to smooth noisy data. Derivatives are computed
    by applying finite differences to the smoothed signal.

    This method is particularly effective for:
    - Strongly noisy data
    - Signals where global smoothness is desired
    - Cases where you want explicit control over smoothness vs. fidelity

    The smoothness is controlled by the parameter lmbda:
    - Small lmbda (~1): Less smoothing, follows data closely
    - Large lmbda (~1e6): Heavy smoothing, very smooth result

    Args:
        lmbda: Smoothing parameter. Larger values give smoother results.
            Typical values range from 1 to 1e6 depending on noise level.
        d_order: Order of the difference penalty (default 2).
            - 1: Penalizes first differences (piecewise constant)
            - 2: Penalizes second differences (piecewise linear, most common)
            - 3: Penalizes third differences (smoother curves)

    Example:
        >>> wh = Whittaker(lmbda=100.0)
        >>> t = torch.linspace(0, 2*torch.pi, 100)
        >>> x = torch.sin(t) + 0.1 * torch.randn(100)
        >>> dx = wh.d(x, t)  # Smoothed derivative
        >>> x_smooth = wh.smooth(x, t)  # Just smoothing
    """

    def __init__(self, lmbda: float = 100.0, d_order: int = 2):
        if lmbda <= 0:
            raise ValueError("lmbda must be positive")
        if d_order not in (1, 2, 3):
            raise ValueError("d_order must be 1, 2, or 3")
        self.lmbda = lmbda
        self.d_order = d_order

    def _smooth_internal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Whittaker-Eilers smoothing to the input.

        Args:
            x: Input tensor of shape (B, T).

        Returns:
            Smoothed tensor of shape (B, T).
        """
        return WhittakerFunction.apply(x, self.lmbda, self.d_order)

    def _compute_derivative(
        self, z: torch.Tensor, dt: float, order: int = 1
    ) -> torch.Tensor:
        """
        Compute derivative of smoothed signal using finite differences.

        Args:
            z: Smoothed signal of shape (B, T).
            dt: Time step.
            order: Derivative order (1 or 2).

        Returns:
            Derivative tensor of shape (B, T).
        """
        B, T = z.shape

        if order == 1:
            # Central differences for interior, forward/backward at edges
            dz = torch.zeros_like(z)
            # Central differences for interior points
            dz[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2 * dt)
            # Forward difference at start
            dz[:, 0] = (z[:, 1] - z[:, 0]) / dt
            # Backward difference at end
            dz[:, -1] = (z[:, -1] - z[:, -2]) / dt
        elif order == 2:
            # Second derivative using central differences
            dz = torch.zeros_like(z)
            # Central second difference for interior
            dz[:, 1:-1] = (z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]) / (dt**2)
            # Edge handling: use one-sided differences
            dz[:, 0] = (z[:, 2] - 2 * z[:, 1] + z[:, 0]) / (dt**2)
            dz[:, -1] = (z[:, -1] - 2 * z[:, -2] + z[:, -3]) / (dt**2)
        else:
            raise ValueError(f"Derivative order {order} not supported (use 1 or 2)")

        return dz

    def d(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the derivative of x with respect to t using Whittaker smoothing.

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

        # Get dt
        dt = (t[1] - t[0]).item()

        # Smooth the signal
        z = self._smooth_internal(x_flat)

        # Compute derivative
        dz = self._compute_derivative(z, dt, order=1)

        # Reshape back
        dx = dz.reshape(*batch_shape, T)

        # Restore original shape
        if was_1d:
            dx = dx.squeeze(0)

        return self._restore_dim(dx, original_dim)

    def smooth(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the smoothed version of x using Whittaker-Eilers filtering.

        Args:
            x: Input tensor of shape (..., T) or (T,)
            t: Time points tensor of shape (T,) (not used but kept for API consistency)
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

        # Smooth
        z = self._smooth_internal(x_flat)

        # Reshape back
        z = z.reshape(*batch_shape, T)

        # Restore original shape
        if was_1d:
            z = z.squeeze(0)

        return self._restore_dim(z, original_dim)

    def _compute_order(
        self, x: torch.Tensor, t: torch.Tensor, order: int, dim: int
    ) -> torch.Tensor:
        """Compute a specific derivative order."""
        if order > 2:
            raise ValueError("Whittaker only supports derivative orders 1 and 2")

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

        # Get dt
        dt = (t[1] - t[0]).item()

        # Smooth the signal (done once, shared across orders in d_orders)
        z = self._smooth_internal(x_flat)

        # Compute derivative of requested order
        dz = self._compute_derivative(z, dt, order=order)

        # Reshape back
        dx = dz.reshape(*batch_shape, T)

        # Restore original shape
        if was_1d:
            dx = dx.squeeze(0)

        return self._restore_dim(dx, original_dim)

    def d_orders(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        orders: Sequence[int] = (1, 2),
        dim: int = -1,
    ) -> dict[int, torch.Tensor]:
        """
        Compute multiple derivative orders simultaneously and efficiently.

        The Whittaker smoother computes the smoothed signal once and then
        derives multiple derivative orders from it, avoiding redundant
        smoothing computation.

        Args:
            x: Input tensor of shape (..., T) or (T,)
            t: Time points tensor of shape (T,)
            orders: Sequence of derivative orders to compute. Default is (1, 2).
                Order 0 returns the smoothed signal.
            dim: Dimension along which to differentiate. Default -1.

        Returns:
            Dictionary mapping order -> derivative tensor.
            Each tensor has the same shape as x.

        Example:
            >>> wh = Whittaker(lmbda=100.0)
            >>> derivs = wh.d_orders(x, t, orders=[0, 1, 2])
            >>> x_smooth = derivs[0]  # Smoothed signal
            >>> dx = derivs[1]        # First derivative
            >>> d2x = derivs[2]       # Second derivative
        """
        orders = list(orders)
        max_order = max(orders)

        if max_order > 2:
            raise ValueError("Whittaker only supports derivative orders up to 2")

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

        # Smooth the signal ONCE (shared across all orders)
        z = self._smooth_internal(x_flat)

        # Compute each derivative order
        results = {}
        for order in orders:
            if order == 0:
                dx = z.clone()
            else:
                dx = self._compute_derivative(z, dt, order=order)

            # Reshape back
            dx = dx.reshape(*batch_shape, T)
            if was_1d:
                dx = dx.squeeze(0)
            results[order] = self._restore_dim(dx, original_dim)

        return results
