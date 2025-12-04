"""
Base class for differentiable numerical differentiation methods.
"""

import abc
from collections.abc import Sequence

import torch


class Derivative(abc.ABC):
    """
    Abstract base class for numerical differentiation methods.

    All differentiation methods should inherit from this class and implement
    the `d` method for computing derivatives.
    """

    @abc.abstractmethod
    def d(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the derivative of x with respect to t.

        Args:
            x: Tensor of shape (..., T) containing the signal values.
               Multiple signals can be batched along leading dimensions.
            t: Tensor of shape (T,) containing the time points.
               Must be evenly spaced for most methods.
            dim: The dimension along which to differentiate. Default is -1
                (last dimension).

        Returns:
            Tensor of same shape as x containing the derivative dx/dt.
        """
        pass

    def d_orders(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        orders: Sequence[int] = (1, 2),
        dim: int = -1,
    ) -> dict[int, torch.Tensor]:
        """
        Compute multiple derivative orders simultaneously.

        This method computes multiple derivative orders in an efficient manner,
        avoiding redundant computation where possible. For methods that support
        it (e.g., SavitzkyGolay), shared computation like polynomial fitting
        is reused across orders.

        Args:
            x: Tensor of shape (..., T) containing the signal values.
            t: Tensor of shape (T,) containing the time points.
            orders: Sequence of derivative orders to compute. Default is (1, 2).
                Order 0 returns the smoothed signal (if supported).
            dim: The dimension along which to differentiate. Default is -1
                (last dimension).

        Returns:
            Dictionary mapping order -> derivative tensor.
            Each tensor has the same shape as x.

        Example:
            >>> sg = SavitzkyGolay(window_length=11, polyorder=4)
            >>> derivs = sg.d_orders(x, t, orders=[1, 2])
            >>> dx = derivs[1]   # First derivative
            >>> d2x = derivs[2]  # Second derivative
        """
        # Default implementation: call d() for each order
        # Subclasses can override this for more efficient implementations
        results = {}
        for order in orders:
            if order == 0:
                try:
                    results[0] = self.smooth(x, t, dim=dim)
                except NotImplementedError:
                    # If smoothing not supported, return original
                    results[0] = x.clone()
            else:
                # Create a copy of self with the requested order
                results[order] = self._compute_order(x, t, order, dim)
        return results

    def _compute_order(
        self, x: torch.Tensor, t: torch.Tensor, order: int, dim: int
    ) -> torch.Tensor:
        """
        Compute a specific derivative order. Override in subclasses.

        Default implementation raises NotImplementedError if order != 1.
        """
        if order == 1:
            return self.d(x, t, dim=dim)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support d_orders with order={order}. "
            "Override _compute_order or d_orders for multi-order support."
        )

    def smooth(self, x: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute the smoothed version of x (if supported by the method).

        Args:
            x: Tensor of shape (..., T) containing the signal values.
            t: Tensor of shape (T,) containing the time points.
            dim: The dimension along which to smooth. Default is -1
                (last dimension).

        Returns:
            Tensor of same shape as x containing the smoothed signal.

        Raises:
            NotImplementedError: If the method does not support smoothing.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support smoothing. "
            "Only certain global methods (like Kalman, Kernel, Spline) support this."
        )

    def _move_dim_to_last(
        self, x: torch.Tensor, dim: int
    ) -> tuple[torch.Tensor, int]:
        """
        Move the specified dimension to the last position.

        Returns:
            Tuple of (moved tensor, original dim position)
        """
        if dim == -1 or dim == x.ndim - 1:
            return x, dim

        # Normalize negative dim
        dim = dim if dim >= 0 else x.ndim + dim

        # Move dim to last position
        perm = list(range(x.ndim))
        perm.remove(dim)
        perm.append(dim)

        return x.permute(*perm), dim

    def _restore_dim(self, x: torch.Tensor, original_dim: int) -> torch.Tensor:
        """
        Restore the dimension to its original position.
        """
        if original_dim == -1 or original_dim == x.ndim - 1:
            return x

        # Normalize negative dim
        original_dim = original_dim if original_dim >= 0 else x.ndim + original_dim

        # Move last dim back to original position
        perm = list(range(x.ndim - 1))
        perm.insert(original_dim, x.ndim - 1)

        return x.permute(*perm)
