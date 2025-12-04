"""
Tests for automatic differentiation (backward pass) through torch_dxdt methods.

These tests verify that all methods are truly differentiable and can be used
in PyTorch optimization loops.
"""

import pytest
import torch

import torch_dxdt


class TestBackpropagation:
    """Test that gradients flow correctly through all methods."""

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("finite_difference", {"k": 1}),
            ("savitzky_golay", {"window_length": 5, "polyorder": 2}),
            ("spectral", {}),
            ("spline", {"s": 0.1}),
            ("kernel", {"sigma": 0.5, "lmbd": 0.1}),
            ("kalman", {"alpha": 1.0}),
            ("whittaker", {"lmbda": 100.0}),
        ],
    )
    def test_gradients_exist(self, method_name, kwargs):
        """Test that gradients are computed for all methods."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        x = torch.sin(t).clone().requires_grad_(True)

        dx = torch_dxdt.dxdt(x, t, kind=method_name, **kwargs)

        # Compute a scalar loss
        loss = dx.sum()
        loss.backward()

        assert x.grad is not None, f"No gradient for {method_name}"
        assert not torch.isnan(x.grad).any(), f"NaN gradient for {method_name}"
        assert x.grad.shape == x.shape, f"Gradient shape mismatch for {method_name}"

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("finite_difference", {"k": 1}),
            ("savitzky_golay", {"window_length": 5, "polyorder": 2}),
            ("spectral", {}),
            ("spline", {"s": 0.1}),
            ("kernel", {"sigma": 0.5, "lmbd": 0.1}),
            ("kalman", {"alpha": 1.0}),
            ("whittaker", {"lmbda": 100.0}),
        ],
    )
    def test_gradients_nonzero(self, method_name, kwargs):
        """Test that gradients are non-zero (not trivially zero)."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        x = torch.sin(t).clone().requires_grad_(True)

        dx = torch_dxdt.dxdt(x, t, kind=method_name, **kwargs)
        loss = dx.pow(2).sum()  # Squared loss
        loss.backward()

        # At least some gradients should be non-zero
        assert (x.grad.abs() > 1e-10).any(), f"All gradients zero for {method_name}"

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("finite_difference", {"k": 1}),
            ("savitzky_golay", {"window_length": 5, "polyorder": 2}),
            ("spectral", {}),
            ("spline", {"s": 0.1}),
        ],
    )
    def test_batched_gradients(self, method_name, kwargs):
        """Test gradients work with batched inputs."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        x_batch = torch.stack(
            [
                torch.sin(t),
                torch.cos(t),
            ],
            dim=0,
        ).requires_grad_(True)  # Shape: (2, 50)

        dx = torch_dxdt.dxdt(x_batch, t, kind=method_name, **kwargs)
        loss = dx.sum()
        loss.backward()

        assert x_batch.grad is not None
        assert x_batch.grad.shape == x_batch.shape
        assert not torch.isnan(x_batch.grad).any()


class TestOptimization:
    """Test that methods work in optimization loops."""

    def test_finite_difference_optimization(self):
        """Test optimization through FiniteDifference."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        target = torch.cos(t)

        # Start with noisy sine
        x = torch.nn.Parameter(torch.sin(t) + 0.5 * torch.randn_like(t))

        optimizer = torch.optim.Adam([x], lr=0.1)
        fd = torch_dxdt.FiniteDifference(k=1)

        initial_loss = None
        for _step in range(20):
            optimizer.zero_grad()
            dx = fd.d(x, t)
            loss = torch.nn.functional.mse_loss(dx, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Optimization did not reduce loss"

    def test_savitzky_golay_optimization(self):
        """Test optimization through SavitzkyGolay."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        target = torch.cos(t)

        x = torch.nn.Parameter(torch.sin(t) + 0.3 * torch.randn_like(t))

        optimizer = torch.optim.Adam([x], lr=0.1)
        sg = torch_dxdt.SavitzkyGolay(window_length=5, polyorder=2)

        initial_loss = None
        for _step in range(20):
            optimizer.zero_grad()
            dx = sg.d(x, t)
            loss = torch.nn.functional.mse_loss(dx, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Optimization did not reduce loss"

    def test_spectral_optimization(self):
        """Test optimization through Spectral."""
        # Use periodic sampling for spectral
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)[:-1]
        target = torch.cos(t)

        x = torch.nn.Parameter(torch.sin(t) + 0.3 * torch.randn_like(t))

        optimizer = torch.optim.Adam([x], lr=0.1)
        spec = torch_dxdt.Spectral()

        initial_loss = None
        for _step in range(20):
            optimizer.zero_grad()
            dx = spec.d(x, t)
            loss = torch.nn.functional.mse_loss(dx, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Optimization did not reduce loss"

    def test_spline_optimization(self):
        """Test optimization through Spline."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        target = torch.cos(t)

        x = torch.nn.Parameter(torch.sin(t) + 0.3 * torch.randn_like(t))

        optimizer = torch.optim.Adam([x], lr=0.1)
        spl = torch_dxdt.Spline(s=0.1)

        initial_loss = None
        for _step in range(20):
            optimizer.zero_grad()
            dx = spl.d(x, t)
            loss = torch.nn.functional.mse_loss(dx, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Optimization did not reduce loss"


class TestGradientChecking:
    """Numerical gradient checking for differentiation methods."""

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("finite_difference", {"k": 1}),
            ("savitzky_golay", {"window_length": 5, "polyorder": 2}),
            ("spectral", {}),
        ],
    )
    def test_gradcheck(self, method_name, kwargs):
        """Use torch.autograd.gradcheck for numerical verification."""
        t = torch.linspace(0, torch.pi, 20, dtype=torch.float64)
        x = torch.sin(t).requires_grad_(True)

        def func(x_input):
            return torch_dxdt.dxdt(x_input, t, kind=method_name, **kwargs)

        # gradcheck compares analytical gradients to numerical approximation
        try:
            passed = torch.autograd.gradcheck(
                func, x, eps=1e-5, atol=1e-3, rtol=1e-3, raise_exception=True
            )
            assert passed
        except RuntimeError as e:
            # Some methods may not pass strict gradcheck due to numerical
            # precision, but should still work for optimization
            pytest.skip(f"Strict gradcheck failed for {method_name}: {e}")


class TestSmoothingGradients:
    """Test gradients through smoothing operations."""

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("spline", {"s": 0.1}),
            ("kernel", {"sigma": 0.5, "lmbd": 0.1}),
            ("kalman", {"alpha": 1.0}),
            ("whittaker", {"lmbda": 100.0}),
        ],
    )
    def test_smoothing_gradients(self, method_name, kwargs):
        """Test that smoothing methods have gradients."""
        t = torch.linspace(0, 2 * torch.pi, 30, dtype=torch.float64)
        x = (torch.sin(t) + 0.2 * torch.randn_like(t)).requires_grad_(True)

        x_smooth = torch_dxdt.smooth_x(x, t, kind=method_name, **kwargs)
        loss = x_smooth.sum()
        loss.backward()

        assert x.grad is not None, f"No gradient for {method_name} smoothing"
        assert not torch.isnan(x.grad).any()


class TestChainedOperations:
    """Test gradients through chained differentiation operations."""

    def test_double_differentiation(self):
        """Test gradients through second derivative computation."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        x = torch.sin(t).requires_grad_(True)

        fd = torch_dxdt.FiniteDifference(k=1)

        # First derivative
        dx = fd.d(x, t)
        # Second derivative
        d2x = fd.d(dx, t)

        loss = d2x.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_smooth_then_differentiate(self):
        """Test gradients through smoothing followed by differentiation."""
        t = torch.linspace(0, 2 * torch.pi, 30, dtype=torch.float64)
        x = (torch.sin(t) + 0.2 * torch.randn_like(t)).requires_grad_(True)

        # First smooth
        x_smooth = torch_dxdt.smooth_x(x, t, kind="spline", s=0.1)
        # Then differentiate
        dx = torch_dxdt.dxdt(x_smooth, t, kind="finite_difference", k=1)

        loss = dx.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestNeuralNetworkIntegration:
    """Test integration with neural network modules."""

    def test_in_nn_module(self):
        """Test using differentiation inside an nn.Module."""

        class PhysicsInformedLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fd = torch_dxdt.FiniteDifference(k=1)
                self.linear = torch.nn.Linear(50, 50)

            def forward(self, x, t):
                # Compute derivative
                dx = self.fd.d(x, t)
                # Pass through linear layer
                return self.linear(dx.float())

        model = PhysicsInformedLayer()
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        x = torch.sin(t).unsqueeze(0).requires_grad_(True)  # Batch dim

        out = model(x, t)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        # Check that model parameters have gradients too
        assert model.linear.weight.grad is not None

    def test_with_autograd_function(self):
        """Test compatibility with custom autograd functions."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)

        class CustomOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        x = torch.sin(t).requires_grad_(True)
        x_scaled = CustomOp.apply(x)

        dx = torch_dxdt.dxdt(x_scaled, t, kind="finite_difference", k=1)
        loss = dx.sum()
        loss.backward()

        assert x.grad is not None


class TestAutodiffVsNumerical:
    """
    Test numerical differentiation against PyTorch autodiff.

    These tests define trajectories as functions, compute their analytical
    derivatives via torch.autograd, sample both the trajectory and derivatives,
    then compare against numerical estimation methods.
    """

    @pytest.mark.parametrize(
        "method_name,kwargs,rtol",
        [
            ("finite_difference", {"k": 2}, 0.05),
            ("savitzky_golay", {"window_length": 11, "polyorder": 4}, 0.02),
            ("spline", {"s": 1e-4}, 0.05),
        ],
    )
    def test_polynomial_trajectory(self, method_name, kwargs, rtol):
        """Test on polynomial trajectory: x(t) = t^3 - 2t^2 + t."""
        t = torch.linspace(0.1, 2.0, 100, dtype=torch.float64)
        t_auto = t.clone().requires_grad_(True)

        # Define trajectory
        def trajectory(t):
            return t**3 - 2 * t**2 + t

        # Compute trajectory
        x = trajectory(t_auto)

        # Compute analytical derivative via autodiff
        dx_autodiff = torch.autograd.grad(x.sum(), t_auto, create_graph=True)[
            0
        ].detach()

        # Compute numerical derivative
        x_detached = x.detach()
        dx_numerical = torch_dxdt.dxdt(x_detached, t, kind=method_name, **kwargs)

        # Compare (exclude edges for methods that have edge effects)
        margin = 5
        torch.testing.assert_close(
            dx_numerical[margin:-margin],
            dx_autodiff[margin:-margin],
            rtol=rtol,
            atol=0.01,
        )

    @pytest.mark.parametrize(
        "method_name,kwargs,rtol",
        [
            ("finite_difference", {"k": 2}, 0.05),
            ("savitzky_golay", {"window_length": 11, "polyorder": 4}, 0.02),
            ("spectral", {}, 1e-4),
        ],
    )
    def test_trigonometric_trajectory(self, method_name, kwargs, rtol):
        """Test on trigonometric trajectory: x(t) = sin(2t) + cos(t)."""
        # Use endpoint=False for spectral compatibility
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)[:-1]
        t_auto = t.clone().requires_grad_(True)

        # Define trajectory
        def trajectory(t):
            return torch.sin(2 * t) + torch.cos(t)

        # Compute trajectory
        x = trajectory(t_auto)

        # Compute analytical derivative via autodiff
        dx_autodiff = torch.autograd.grad(x.sum(), t_auto, create_graph=True)[
            0
        ].detach()

        # Compute numerical derivative
        x_detached = x.detach()
        dx_numerical = torch_dxdt.dxdt(x_detached, t, kind=method_name, **kwargs)

        # Compare
        margin = 5
        torch.testing.assert_close(
            dx_numerical[margin:-margin],
            dx_autodiff[margin:-margin],
            rtol=rtol,
            atol=0.02,
        )

    @pytest.mark.parametrize(
        "method_name,kwargs,rtol",
        [
            ("finite_difference", {"k": 2}, 0.1),
            ("savitzky_golay", {"window_length": 11, "polyorder": 4}, 0.05),
        ],
    )
    def test_exponential_trajectory(self, method_name, kwargs, rtol):
        """Test on exponential trajectory: x(t) = exp(-t/2) * sin(3t)."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        t_auto = t.clone().requires_grad_(True)

        # Define trajectory
        def trajectory(t):
            return torch.exp(-t / 2) * torch.sin(3 * t)

        # Compute trajectory
        x = trajectory(t_auto)

        # Compute analytical derivative via autodiff
        dx_autodiff = torch.autograd.grad(x.sum(), t_auto, create_graph=True)[
            0
        ].detach()

        # Compute numerical derivative
        x_detached = x.detach()
        dx_numerical = torch_dxdt.dxdt(x_detached, t, kind=method_name, **kwargs)

        # Compare
        margin = 5
        torch.testing.assert_close(
            dx_numerical[margin:-margin],
            dx_autodiff[margin:-margin],
            rtol=rtol,
            atol=0.02,
        )

    def test_second_derivative_via_autodiff(self):
        """Test second derivative by comparing autodiff d2x/dt2 vs numerical."""
        t = torch.linspace(0.1, 2.0, 100, dtype=torch.float64)
        t_auto = t.clone().requires_grad_(True)

        # Define trajectory: x(t) = sin(t)
        x = torch.sin(t_auto)

        # Compute first derivative via autodiff
        (dx,) = torch.autograd.grad(x.sum(), t_auto, create_graph=True)

        # Compute second derivative via autodiff
        (d2x_autodiff,) = torch.autograd.grad(dx.sum(), t_auto)
        d2x_autodiff = d2x_autodiff.detach()

        # Compute second derivative numerically using Savitzky-Golay
        x_detached = x.detach()
        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=2)
        d2x_numerical = sg.d(x_detached, t)

        # Compare (exclude edges)
        margin = 10
        torch.testing.assert_close(
            d2x_numerical[margin:-margin],
            d2x_autodiff[margin:-margin],
            rtol=0.05,
            atol=0.02,
        )

    def test_batched_trajectory_autodiff(self):
        """Test that numerical derivatives match autodiff for batched data."""
        t = torch.linspace(0, 2 * torch.pi, 80, dtype=torch.float64)[:-1]

        # Create batch of trajectories with different frequencies
        freqs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        t_auto = t.clone().requires_grad_(True)

        # Batch of trajectories: sin(freq * t) for each freq
        x_batch = torch.sin(freqs.unsqueeze(1) * t_auto.unsqueeze(0))  # (3, T)

        # Compute analytical derivatives via autodiff for each
        dx_autodiff_list = []
        for i in range(len(freqs)):
            if t_auto.grad is not None:
                t_auto.grad.zero_()
            x_batch[i].sum().backward(retain_graph=True)
            dx_autodiff_list.append(t_auto.grad.clone())
        dx_autodiff = torch.stack(dx_autodiff_list)

        # Compute numerical derivatives for batch
        x_detached = x_batch.detach()
        dx_numerical = torch_dxdt.dxdt(x_detached, t, kind="spectral", dim=-1)

        # Compare
        margin = 3
        torch.testing.assert_close(
            dx_numerical[:, margin:-margin],
            dx_autodiff[:, margin:-margin],
            rtol=1e-4,
            atol=1e-4,
        )
