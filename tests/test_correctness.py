"""
Tests comparing torch_dxdt implementations against the reference derivative package.

These tests verify that our PyTorch implementations produce mathematically
correct results by comparing them to the established derivative package.
"""

import numpy as np
import pytest
import torch
from derivative import dxdt as np_dxdt

import torch_dxdt

# Test parameters
N_POINTS = 100
TOLERANCE = 1e-4  # Tolerance for comparing results


class TestFiniteDifferenceParity:
    """Test FiniteDifference against the derivative package."""

    def test_basic_sine(self):
        """Test finite difference on a sine wave."""
        t = np.linspace(0, 2 * np.pi, N_POINTS)
        x = np.sin(t)

        # Reference implementation
        ref = np_dxdt(x, t, kind="finite_difference", k=1)

        # PyTorch implementation
        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)
        fd = torch_dxdt.FiniteDifference(k=1)
        result = fd.d(x_torch, t_torch).numpy()

        # Compare (excluding boundaries where methods may differ)
        np.testing.assert_allclose(
            result[2:-2],
            ref[2:-2],
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg="FiniteDifference results differ from reference",
        )

    def test_different_k_values(self):
        """Test finite difference with different window sizes."""
        t = np.linspace(0, 2 * np.pi, N_POINTS)
        x = np.sin(t)

        for k in [1, 2, 3]:
            ref = np_dxdt(x, t, kind="finite_difference", k=k)

            t_torch = torch.tensor(t, dtype=torch.float64)
            x_torch = torch.tensor(x, dtype=torch.float64)
            fd = torch_dxdt.FiniteDifference(k=k)
            result = fd.d(x_torch, t_torch).numpy()

            # Compare interior points
            margin = k + 2
            np.testing.assert_allclose(
                result[margin:-margin],
                ref[margin:-margin],
                rtol=TOLERANCE,
                atol=TOLERANCE,
                err_msg=f"FiniteDifference(k={k}) results differ",
            )

    def test_functional_interface(self):
        """Test the dxdt functional interface."""
        t = np.linspace(0, 2 * np.pi, N_POINTS)
        x = np.sin(t)

        ref = np_dxdt(x, t, kind="finite_difference", k=1)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)
        result = torch_dxdt.dxdt(x_torch, t_torch, kind="finite_difference", k=1).numpy()

        np.testing.assert_allclose(
            result[2:-2], ref[2:-2], rtol=TOLERANCE, atol=TOLERANCE
        )


class TestSavitzkyGolayParity:
    """Test SavitzkyGolay against the derivative package."""

    def test_basic_sine(self):
        """Test Savitzky-Golay on a sine wave."""
        t = np.linspace(0, 2 * np.pi, N_POINTS)
        x = np.sin(t)

        # Use a larger window (11 points) with polyorder 3 to avoid
        # ill-conditioning warnings. This gives 11 points for 4 coefficients.
        window_length = 11
        left = 5
        right = 5
        polyorder = 3

        ref = np_dxdt(
            x,
            t,
            kind="savitzky_golay",
            order=polyorder,
            left=left,
            right=right,
            iwindow=True,
        )

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)
        sg = torch_dxdt.SavitzkyGolay(window_length=window_length, polyorder=polyorder)
        result = sg.d(x_torch, t_torch).numpy()

        # Compare interior points (edge handling may differ)
        margin = window_length
        np.testing.assert_allclose(
            result[margin:-margin],
            ref[margin:-margin],
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg="SavitzkyGolay results differ from reference",
        )

    def test_noisy_signal(self):
        """Test Savitzky-Golay on a noisy signal."""
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, N_POINTS)
        x = np.sin(t) + 0.1 * np.random.randn(N_POINTS)

        window_length = 11
        polyorder = 3

        ref = np_dxdt(
            x, t, kind="savitzky_golay", order=polyorder, left=5, right=5, iwindow=True
        )

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)
        sg = torch_dxdt.SavitzkyGolay(window_length=window_length, polyorder=polyorder)
        result = sg.d(x_torch, t_torch).numpy()

        margin = window_length
        np.testing.assert_allclose(
            result[margin:-margin], ref[margin:-margin], rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_pad_modes(self):
        """Test Savitzky-Golay with different padding modes."""
        t = np.linspace(0, 2 * np.pi, N_POINTS)
        x = np.sin(t)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)

        window_length = 11
        polyorder = 3

        # Test all valid padding modes
        for pad_mode in ['replicate', 'reflect', 'circular']:
            sg = torch_dxdt.SavitzkyGolay(
                window_length=window_length,
                polyorder=polyorder,
                pad_mode=pad_mode
            )
            result = sg.d(x_torch, t_torch)
            
            # Check output shape matches input
            assert result.shape == x_torch.shape
            # Check no NaNs
            assert not torch.isnan(result).any()

        # Test that interior results are the same regardless of padding mode
        results = {}
        for pad_mode in ['replicate', 'reflect', 'circular']:
            sg = torch_dxdt.SavitzkyGolay(
                window_length=window_length,
                polyorder=polyorder,
                pad_mode=pad_mode
            )
            results[pad_mode] = sg.d(x_torch, t_torch)

        # Interior points should be very similar
        margin = window_length
        np.testing.assert_allclose(
            results['replicate'][margin:-margin].numpy(),
            results['reflect'][margin:-margin].numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="Interior points should match between pad modes"
        )

    def test_pad_mode_backward_compatibility(self):
        """Test that periodic=True still works (backward compatibility)."""
        t = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
        x = np.sin(t)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)

        # Using deprecated periodic=True
        sg_periodic = torch_dxdt.SavitzkyGolay(
            window_length=11, polyorder=3, periodic=True
        )
        # Using new pad_mode='circular'
        sg_circular = torch_dxdt.SavitzkyGolay(
            window_length=11, polyorder=3, pad_mode='circular'
        )

        result_periodic = sg_periodic.d(x_torch, t_torch)
        result_circular = sg_circular.d(x_torch, t_torch)

        # Results should be identical
        np.testing.assert_allclose(
            result_periodic.numpy(),
            result_circular.numpy(),
            rtol=1e-14,
            atol=1e-14,
            err_msg="periodic=True should be equivalent to pad_mode='circular'"
        )

        # Check that periodic attribute is set correctly
        assert sg_periodic.periodic is True
        assert sg_periodic.pad_mode == 'circular'
        assert sg_circular.periodic is True
        assert sg_circular.pad_mode == 'circular'

    def test_invalid_pad_mode(self):
        """Test that invalid pad_mode raises ValueError."""
        with pytest.raises(ValueError, match="pad_mode must be one of"):
            torch_dxdt.SavitzkyGolay(
                window_length=11, polyorder=3, pad_mode='invalid'
            )


class TestSpectralParity:
    """Test Spectral differentiation against the derivative package."""

    def test_basic_sine(self):
        """Test spectral differentiation on a sine wave."""
        # For spectral methods, we need periodic data
        t = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
        x = np.sin(t)

        ref = np_dxdt(x, t, kind="spectral")

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)
        spec = torch_dxdt.Spectral()
        result = spec.d(x_torch, t_torch).numpy()

        # Spectral methods should be very accurate for smooth periodic signals
        np.testing.assert_allclose(
            result,
            ref,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Spectral results differ from reference",
        )

    def test_against_analytical(self):
        """Test spectral differentiation against analytical derivative."""
        t = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
        x = np.sin(t)
        dx_analytical = np.cos(t)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)
        spec = torch_dxdt.Spectral()
        result = spec.d(x_torch, t_torch).numpy()

        np.testing.assert_allclose(
            result,
            dx_analytical,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Spectral derivative differs from analytical",
        )


class TestSplineParity:
    """Test Spline differentiation against the derivative package."""

    def test_basic_sine(self):
        """Test spline differentiation on a sine wave."""
        t = np.linspace(0, 2 * np.pi, N_POINTS)
        x = np.sin(t)

        # Note: Our implementation uses Whittaker smoothing, which is similar
        # but not identical to scipy splines. We test for reasonable accuracy.
        ref = np_dxdt(x, t, kind="spline", s=0.01)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)
        spl = torch_dxdt.Spline(s=0.01)
        result = spl.d(x_torch, t_torch).numpy()

        # Allow larger tolerance due to different smoothing approaches
        np.testing.assert_allclose(
            result[5:-5],
            ref[5:-5],
            rtol=0.1,
            atol=0.1,
            err_msg="Spline results differ significantly from reference",
        )

    def test_smoothing_behavior(self):
        """Test that larger s values produce smoother results."""
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, N_POINTS)
        x = np.sin(t) + 0.3 * np.random.randn(N_POINTS)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)

        spl_small_s = torch_dxdt.Spline(s=0.01)
        spl_large_s = torch_dxdt.Spline(s=10.0)

        result_small = spl_small_s.d(x_torch, t_torch).numpy()
        result_large = spl_large_s.d(x_torch, t_torch).numpy()

        # Larger s should produce smoother (less varying) result
        var_small = np.var(np.diff(result_small))
        var_large = np.var(np.diff(result_large))

        assert var_large < var_small, "Larger s should produce smoother results"


class TestKernelParity:
    """Test Kernel differentiation against the derivative package."""

    def test_basic_sine(self):
        """Test kernel differentiation on a sine wave."""
        # Use fewer points for kernel methods (O(n^3) complexity)
        n_points = 50
        t = np.linspace(0, 2 * np.pi, n_points)
        x = np.sin(t)

        sigma = 0.5
        lmbd = 0.01

        ref = np_dxdt(x, t, kind="kernel", sigma=sigma, lmbd=lmbd, kernel="rbf")

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)
        ker = torch_dxdt.Kernel(sigma=sigma, lmbd=lmbd, kernel="gaussian")
        result = ker.d(x_torch, t_torch).numpy()

        np.testing.assert_allclose(
            result,
            ref,
            rtol=0.1,
            atol=0.1,
            err_msg="Kernel results differ from reference",
        )

    def test_smoothing(self):
        """Test kernel smoothing functionality."""
        np.random.seed(42)
        n_points = 50
        t = np.linspace(0, 2 * np.pi, n_points)
        x = np.sin(t) + 0.3 * np.random.randn(n_points)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)

        ker = torch_dxdt.Kernel(sigma=0.5, lmbd=0.1)
        smoothed = ker.smooth(x_torch, t_torch).numpy()

        # Smoothed signal should be closer to true sine than noisy input
        true_signal = np.sin(t)
        error_noisy = np.mean((x - true_signal) ** 2)
        error_smoothed = np.mean((smoothed - true_signal) ** 2)

        assert error_smoothed < error_noisy, "Smoothing should reduce error"


class TestKalmanParity:
    """Test Kalman differentiation against the derivative package."""

    def test_basic_sine(self):
        """Test Kalman differentiation on a sine wave."""
        n_points = 50  # Use fewer points for speed
        t = np.linspace(0, 2 * np.pi, n_points)
        x = np.sin(t)

        alpha = 1.0

        ref = np_dxdt(x, t, kind="kalman", alpha=alpha)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)
        kal = torch_dxdt.Kalman(alpha=alpha)
        result = kal.d(x_torch, t_torch).numpy()

        np.testing.assert_allclose(
            result,
            ref,
            rtol=0.1,
            atol=0.1,
            err_msg="Kalman results differ from reference",
        )

    def test_smoothing(self):
        """Test Kalman smoothing functionality."""
        np.random.seed(42)
        n_points = 50
        t = np.linspace(0, 2 * np.pi, n_points)
        x = np.sin(t) + 0.2 * np.random.randn(n_points)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)

        kal = torch_dxdt.Kalman(alpha=1.0)
        smoothed = kal.smooth(x_torch, t_torch).numpy()

        # Smoothed signal should be closer to true sine
        true_signal = np.sin(t)
        error_noisy = np.mean((x - true_signal) ** 2)
        error_smoothed = np.mean((smoothed - true_signal) ** 2)

        assert error_smoothed < error_noisy, "Kalman smoothing should reduce error"

    def test_different_alpha_values(self):
        """Test that alpha affects smoothing level."""
        np.random.seed(42)
        n_points = 50
        t = np.linspace(0, 2 * np.pi, n_points)
        x = np.sin(t) + 0.2 * np.random.randn(n_points)

        t_torch = torch.tensor(t, dtype=torch.float64)
        x_torch = torch.tensor(x, dtype=torch.float64)

        kal_small_alpha = torch_dxdt.Kalman(alpha=0.1)
        kal_large_alpha = torch_dxdt.Kalman(alpha=10.0)

        result_small = kal_small_alpha.d(x_torch, t_torch).numpy()
        result_large = kal_large_alpha.d(x_torch, t_torch).numpy()

        # Larger alpha should produce smoother derivatives
        var_small = np.var(np.diff(result_small))
        var_large = np.var(np.diff(result_large))

        assert var_large < var_small, "Larger alpha should produce smoother derivatives"


class TestBatchProcessing:
    """Test that all methods handle batched inputs correctly."""

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("finite_difference", {"k": 1}),
            ("savitzky_golay", {"window_length": 5, "polyorder": 2}),
            ("spectral", {}),
            ("spline", {"s": 0.1}),
            ("kernel", {"sigma": 0.5, "lmbd": 0.1}),
            ("kalman", {"alpha": 1.0}),
        ],
    )
    def test_batched_input(self, method_name, kwargs):
        """Test processing of batched inputs."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        x_batch = torch.stack(
            [torch.sin(t), torch.cos(t), torch.sin(2 * t)], dim=0
        )  # Shape: (3, 50)

        result = torch_dxdt.dxdt(x_batch, t, kind=method_name, **kwargs)

        assert result.shape == x_batch.shape, f"Batch shape mismatch for {method_name}"
        assert not torch.isnan(result).any(), f"NaN in result for {method_name}"


class TestArbitraryDimensions:
    """Test that methods correctly handle tensors with arbitrary dimensions."""

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("spline", {"s": 0.01}),
            ("finite_difference", {"k": 1}),
            ("savitzky_golay", {"window_length": 5, "polyorder": 2}),
            ("kernel", {"sigma": 0.5, "lmbd": 0.1}),
            ("kalman", {"alpha": 1.0}),
            ("whittaker", {"lmbda": 100.0}),
        ],
    )
    def test_dim_parameter_2d(self, method_name, kwargs):
        """Test differentiation along specified dim for 2D tensors."""
        T = 50
        t = torch.linspace(0, 2 * torch.pi, T, dtype=torch.float64)
        
        # Shape: (3, T) - differentiate along dim=1
        x = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], dim=0)
        
        method_class = getattr(torch_dxdt, method_name.title().replace("_", ""))
        if method_name == "savitzky_golay":
            method_class = torch_dxdt.SavitzkyGolay
        elif method_name == "finite_difference":
            method_class = torch_dxdt.FiniteDifference
        
        method = method_class(**kwargs)
        
        # Test dim=1 (last dim)
        result_dim1 = method.d(x, t, dim=1)
        assert result_dim1.shape == x.shape
        
        # Test dim=-1 (equivalent to dim=1)
        result_dim_neg1 = method.d(x, t, dim=-1)
        torch.testing.assert_close(result_dim1, result_dim_neg1)
        
        # Transpose and test dim=0
        x_T = x.T  # Shape: (T, 3)
        result_dim0 = method.d(x_T, t, dim=0)
        assert result_dim0.shape == x_T.shape
        
        # Results should match when transposed back
        torch.testing.assert_close(result_dim0.T, result_dim1, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("spline", {"s": 0.01}),
            ("finite_difference", {"k": 1}),
            ("savitzky_golay", {"window_length": 5, "polyorder": 2}),
            ("kernel", {"sigma": 0.5, "lmbd": 0.1}),
            ("kalman", {"alpha": 1.0}),
            ("whittaker", {"lmbda": 100.0}),
        ],
    )
    def test_dim_parameter_3d(self, method_name, kwargs):
        """Test differentiation along specified dim for 3D tensors."""
        T = 40
        t = torch.linspace(0, 2 * torch.pi, T, dtype=torch.float64)
        
        # Create 3D tensor with shape (2, 3, T)
        x_base = torch.sin(t)
        x = torch.stack([
            torch.stack([x_base, x_base * 2, x_base * 3], dim=0),
            torch.stack([x_base * 0.5, x_base * 1.5, x_base * 2.5], dim=0),
        ], dim=0)
        assert x.shape == (2, 3, T)
        
        method_class = getattr(torch_dxdt, method_name.title().replace("_", ""))
        if method_name == "savitzky_golay":
            method_class = torch_dxdt.SavitzkyGolay
        elif method_name == "finite_difference":
            method_class = torch_dxdt.FiniteDifference
        
        method = method_class(**kwargs)
        
        # Test dim=2 (last dim, time axis)
        result = method.d(x, t, dim=2)
        assert result.shape == x.shape
        assert not torch.isnan(result).any()
        
        # Test dim=-1 (equivalent)
        result_neg = method.d(x, t, dim=-1)
        torch.testing.assert_close(result, result_neg)

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("spline", {"s": 0.01}),
            ("finite_difference", {"k": 1}),
            ("savitzky_golay", {"window_length": 5, "polyorder": 2}),
            ("kernel", {"sigma": 0.5, "lmbd": 0.1}),
            ("kalman", {"alpha": 1.0}),
            ("whittaker", {"lmbda": 100.0}),
        ],
    )
    def test_dim_parameter_4d(self, method_name, kwargs):
        """Test differentiation along specified dim for 4D tensors."""
        T = 30
        t = torch.linspace(0, 2 * torch.pi, T, dtype=torch.float64)
        
        # Create 4D tensor with shape (2, 2, 3, T)
        x_base = torch.sin(t)
        x = x_base.expand(2, 2, 3, T).clone()
        # Add variation
        x[0, 0, :, :] *= 1.0
        x[0, 1, :, :] *= 1.5
        x[1, 0, :, :] *= 2.0
        x[1, 1, :, :] *= 2.5
        
        method_class = getattr(torch_dxdt, method_name.title().replace("_", ""))
        if method_name == "savitzky_golay":
            method_class = torch_dxdt.SavitzkyGolay
        elif method_name == "finite_difference":
            method_class = torch_dxdt.FiniteDifference
        
        method = method_class(**kwargs)
        
        # Test dim=3 (last dim)
        result = method.d(x, t, dim=3)
        assert result.shape == x.shape
        assert not torch.isnan(result).any()
        
        # Test dim=-1 (equivalent)
        result_neg = method.d(x, t, dim=-1)
        torch.testing.assert_close(result, result_neg)

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("spline", {"s": 0.01}),
            ("finite_difference", {"k": 1}),
            ("savitzky_golay", {"window_length": 5, "polyorder": 2}),
            ("kernel", {"sigma": 0.5, "lmbd": 0.1}),
            ("kalman", {"alpha": 1.0}),
            ("whittaker", {"lmbda": 100.0}),
        ],
    )
    def test_dim_middle_axis(self, method_name, kwargs):
        """Test differentiation along a middle dimension (not first or last)."""
        T = 40
        t = torch.linspace(0, 2 * torch.pi, T, dtype=torch.float64)
        
        # Create 3D tensor with shape (2, T, 3) - time in the middle
        x_base = torch.sin(t)
        x = torch.stack([
            torch.stack([x_base, x_base * 2, x_base * 3], dim=1),
            torch.stack([x_base * 0.5, x_base * 1.5, x_base * 2.5], dim=1),
        ], dim=0)
        assert x.shape == (2, T, 3)
        
        method_class = getattr(torch_dxdt, method_name.title().replace("_", ""))
        if method_name == "savitzky_golay":
            method_class = torch_dxdt.SavitzkyGolay
        elif method_name == "finite_difference":
            method_class = torch_dxdt.FiniteDifference
        
        method = method_class(**kwargs)
        
        # Test dim=1 (middle dimension)
        result = method.d(x, t, dim=1)
        assert result.shape == x.shape
        assert not torch.isnan(result).any()
        
        # Verify result is consistent: permute, compute, permute back
        x_permuted = x.permute(0, 2, 1)  # (2, 3, T)
        result_via_permute = method.d(x_permuted, t, dim=-1).permute(0, 2, 1)
        torch.testing.assert_close(result, result_via_permute, rtol=1e-5, atol=1e-5)

    def test_spline_consistency_across_dims(self):
        """Specifically test Spline method handles dim parameter correctly."""
        T = 50
        t = torch.linspace(0, 2 * torch.pi, T, dtype=torch.float64)
        
        # Create a signal
        x_1d = torch.sin(t)
        
        spline = torch_dxdt.Spline(s=0.01)
        
        # 1D result
        dx_1d = spline.d(x_1d, t)
        
        # 2D: (1, T) with dim=-1
        x_2d = x_1d.unsqueeze(0)
        dx_2d = spline.d(x_2d, t, dim=-1)
        torch.testing.assert_close(dx_2d.squeeze(0), dx_1d)
        
        # 2D: (T, 1) with dim=0
        x_2d_t = x_1d.unsqueeze(1)
        dx_2d_t = spline.d(x_2d_t, t, dim=0)
        torch.testing.assert_close(dx_2d_t.squeeze(1), dx_1d)
        
        # 3D: (1, T, 1) with dim=1
        x_3d = x_1d.unsqueeze(0).unsqueeze(2)
        dx_3d = spline.d(x_3d, t, dim=1)
        torch.testing.assert_close(dx_3d.squeeze(), dx_1d)
        
        # 4D: (1, 1, T, 1) with dim=2
        x_4d = x_1d.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        dx_4d = spline.d(x_4d, t, dim=2)
        torch.testing.assert_close(dx_4d.squeeze(), dx_1d)

    def test_spline_smooth_with_dim(self):
        """Test that Spline.smooth() also respects the dim parameter."""
        T = 50
        t = torch.linspace(0, 2 * torch.pi, T, dtype=torch.float64)
        
        # Noisy signal
        torch.manual_seed(42)
        x_1d = torch.sin(t) + 0.1 * torch.randn(T, dtype=torch.float64)
        
        spline = torch_dxdt.Spline(s=0.1)
        
        # 1D result
        smooth_1d = spline.smooth(x_1d, t)
        
        # 3D: (2, T, 3) with dim=1
        x_3d = x_1d.unsqueeze(0).unsqueeze(2).expand(2, T, 3).clone()
        smooth_3d = spline.smooth(x_3d, t, dim=1)
        
        assert smooth_3d.shape == x_3d.shape
        # Each slice should match the 1D result
        torch.testing.assert_close(smooth_3d[0, :, 0], smooth_1d)
        torch.testing.assert_close(smooth_3d[1, :, 2], smooth_1d)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Test handling of empty inputs."""
        t = torch.tensor([], dtype=torch.float64)
        x = torch.tensor([], dtype=torch.float64)

        fd = torch_dxdt.FiniteDifference(k=1)
        result = fd.d(x, t)

        assert result.numel() == 0

    def test_short_input_spline(self):
        """Test that spline raises error for too-short input."""
        t = torch.linspace(0, 1, 3, dtype=torch.float64)
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

        spl = torch_dxdt.Spline(s=0.01, order=3)

        with pytest.raises(TypeError):
            spl.d(x, t)

    def test_1d_input(self):
        """Test that 1D inputs are handled correctly."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        x = torch.sin(t)  # 1D tensor

        fd = torch_dxdt.FiniteDifference(k=1)
        result = fd.d(x, t)

        assert result.shape == x.shape
        assert result.ndim == 1


class TestAnalyticalCorrectness:
    """Test derivatives against known analytical solutions."""

    @pytest.mark.parametrize(
        "method_name,kwargs,tolerance",
        [
            ("finite_difference", {"k": 1}, 0.05),
            ("savitzky_golay", {"window_length": 7, "polyorder": 3}, 0.05),
            ("spectral", {}, 1e-5),  # Spectral should be very accurate
            ("spline", {"s": 0.001}, 0.1),
            ("kernel", {"sigma": 0.3, "lmbd": 0.001}, 0.1),
            ("kalman", {"alpha": 0.5}, 0.1),
        ],
    )
    def test_sine_derivative(self, method_name, kwargs, tolerance):
        """Test derivative of sin(x) = cos(x)."""
        n_points = 100 if method_name not in ["kernel", "kalman"] else 50

        # For spectral, use periodic sampling
        if method_name == "spectral":
            t = torch.linspace(0, 2 * torch.pi, n_points, dtype=torch.float64)[:-1]
        else:
            t = torch.linspace(0, 2 * torch.pi, n_points, dtype=torch.float64)

        x = torch.sin(t)
        dx_true = torch.cos(t)

        dx = torch_dxdt.dxdt(x, t, kind=method_name, **kwargs)

        # Check interior points (edges may have boundary effects)
        margin = 5
        error = torch.abs(dx[margin:-margin] - dx_true[margin:-margin]).mean()

        assert error < tolerance, (
            f"{method_name}: Mean error {error:.6f} exceeds tolerance {tolerance}"
        )

    @pytest.mark.parametrize(
        "method_name,kwargs,tolerance",
        [
            ("finite_difference", {"k": 2}, 0.05),
            ("savitzky_golay", {"window_length": 7, "polyorder": 3}, 0.05),
            ("spectral", {}, 1e-5),
        ],
    )
    def test_polynomial_derivative(self, method_name, kwargs, tolerance):
        """Test derivative of x^2 = 2x."""
        n_points = 100

        if method_name == "spectral":
            # Polynomial not periodic, skip
            pytest.skip("Spectral not suitable for non-periodic polynomials")

        t = torch.linspace(-2, 2, n_points, dtype=torch.float64)
        x = t**2
        dx_true = 2 * t

        dx = torch_dxdt.dxdt(x, t, kind=method_name, **kwargs)

        margin = 10
        error = torch.abs(dx[margin:-margin] - dx_true[margin:-margin]).mean()

        assert error < tolerance, (
            f"{method_name}: Mean error {error:.6f} exceeds tolerance {tolerance}"
        )


class TestSecondOrderDerivative:
    """Test second-order derivative estimation via Savitzky-Golay."""

    def test_sine_second_derivative(self):
        """Test d²sin(t)/dt² = -sin(t)."""
        t = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
        x = torch.sin(t)
        d2x_true = -torch.sin(t)

        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=2)
        d2x = sg.d(x, t)

        # Compare interior points
        margin = 15
        error = torch.abs(d2x[margin:-margin] - d2x_true[margin:-margin]).mean()

        assert error < 0.02, f"Mean error {error:.6f} exceeds tolerance 0.02"

    def test_polynomial_second_derivative(self):
        """Test d²(t³)/dt² = 6t."""
        t = torch.linspace(-2, 2, 200, dtype=torch.float64)
        x = t**3
        d2x_true = 6 * t

        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=2)
        d2x = sg.d(x, t)

        # Compare interior points
        margin = 15
        error = torch.abs(d2x[margin:-margin] - d2x_true[margin:-margin]).mean()

        assert error < 0.05, f"Mean error {error:.6f} exceeds tolerance 0.05"

    def test_exponential_second_derivative(self):
        """Test d²exp(t)/dt² = exp(t)."""
        t = torch.linspace(0, 2, 200, dtype=torch.float64)
        x = torch.exp(t)
        d2x_true = torch.exp(t)

        sg = torch_dxdt.SavitzkyGolay(window_length=15, polyorder=5, order=2)
        d2x = sg.d(x, t)

        # Compare interior points
        margin = 20
        # Relative error for exponential since values vary widely
        rel_error = (
            torch.abs(d2x[margin:-margin] - d2x_true[margin:-margin])
            / d2x_true[margin:-margin]
        ).mean()

        assert rel_error < 0.05, f"Mean rel error {rel_error:.6f} exceeds 0.05"

    def test_quadratic_second_derivative(self):
        """Test d²(at² + bt + c)/dt² = 2a."""
        a, b, c = 3.0, -2.0, 5.0
        t = torch.linspace(-1, 1, 200, dtype=torch.float64)
        x = a * t**2 + b * t + c
        d2x_true = torch.full_like(t, 2 * a)

        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=2)
        d2x = sg.d(x, t)

        # Compare interior points
        margin = 15
        error = torch.abs(d2x[margin:-margin] - d2x_true[margin:-margin]).mean()

        assert error < 0.01, f"Mean error {error:.6f} exceeds tolerance 0.01"

    def test_order_validation(self):
        """Test that order > polyorder raises an error."""
        with pytest.raises(ValueError, match="order.*must be <= polyorder"):
            torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3, order=4)

    def test_first_vs_second_order(self):
        """Test that first and second order give different results."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        x = torch.sin(t)

        sg1 = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=1)
        sg2 = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=2)

        dx = sg1.d(x, t)
        d2x = sg2.d(x, t)

        # They should not be the same
        assert not torch.allclose(dx, d2x)

        # First derivative of sin should be close to cos
        margin = 10
        dx_true = torch.cos(t)
        assert torch.abs(dx[margin:-margin] - dx_true[margin:-margin]).mean() < 0.02

        # Second derivative of sin should be close to -sin
        d2x_true = -torch.sin(t)
        assert torch.abs(d2x[margin:-margin] - d2x_true[margin:-margin]).mean() < 0.02


class TestMultiOrderDerivative:
    """Test multi-order derivative computation via d_orders()."""

    def test_d_orders_basic(self):
        """Test basic d_orders functionality on sine wave."""
        t = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
        x = torch.sin(t)

        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4)
        derivs = sg.d_orders(x, t, orders=[1, 2])

        assert 1 in derivs
        assert 2 in derivs
        assert derivs[1].shape == x.shape
        assert derivs[2].shape == x.shape

        # Check first derivative
        margin = 10
        dx_true = torch.cos(t)
        error1 = torch.abs(derivs[1][margin:-margin] - dx_true[margin:-margin]).mean()
        assert error1 < 0.02, f"First derivative error {error1:.6f} exceeds tolerance"

        # Check second derivative
        d2x_true = -torch.sin(t)
        error2 = torch.abs(derivs[2][margin:-margin] - d2x_true[margin:-margin]).mean()
        assert error2 < 0.02, f"Second derivative error {error2:.6f} exceeds tolerance"

    def test_d_orders_matches_individual_calls(self):
        """Test that d_orders gives same results as individual d() calls."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        x = torch.sin(t) + 0.5 * torch.cos(2 * t)

        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4)
        derivs = sg.d_orders(x, t, orders=[0, 1, 2])

        # Compare with individual calls
        sg0 = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=1)
        sg0.order = 0  # Use order 0 for smoothing
        x_smooth_individual = sg0._compute_order(x, t, order=0, dim=-1)

        sg1 = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=1)
        dx_individual = sg1.d(x, t)

        sg2 = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4, order=2)
        d2x_individual = sg2.d(x, t)

        torch.testing.assert_close(derivs[0], x_smooth_individual)
        torch.testing.assert_close(derivs[1], dx_individual)
        torch.testing.assert_close(derivs[2], d2x_individual)

    def test_d_orders_with_order_zero(self):
        """Test that order 0 returns smoothed signal."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        torch.manual_seed(42)
        x = torch.sin(t) + 0.1 * torch.randn(100, dtype=torch.float64)

        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4)
        derivs = sg.d_orders(x, t, orders=[0, 1])

        # Smoothed signal should be closer to true sine than noisy input
        x_true = torch.sin(t)
        error_original = torch.abs(x - x_true).mean()
        error_smoothed = torch.abs(derivs[0] - x_true).mean()

        assert error_smoothed < error_original, (
            f"Smoothed error {error_smoothed:.6f} >= original {error_original:.6f}"
        )

    def test_d_orders_batched(self):
        """Test d_orders with batched input."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        # Create batch of signals: sin, cos, sin(2t)
        x = torch.stack(
            [
                torch.sin(t),
                torch.cos(t),
                torch.sin(2 * t),
            ]
        )  # Shape: (3, 100)

        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4)
        derivs = sg.d_orders(x, t, orders=[1, 2])

        assert derivs[1].shape == (3, 100)
        assert derivs[2].shape == (3, 100)

        # Check derivatives for each signal
        margin = 10

        # sin -> cos -> -sin
        dx_sin_true = torch.cos(t)
        d2x_sin_true = -torch.sin(t)
        assert (
            torch.abs(derivs[1][0, margin:-margin] - dx_sin_true[margin:-margin]).mean()
            < 0.02
        )
        assert (
            torch.abs(
                derivs[2][0, margin:-margin] - d2x_sin_true[margin:-margin]
            ).mean()
            < 0.02
        )

        # cos -> -sin -> -cos
        dx_cos_true = -torch.sin(t)
        d2x_cos_true = -torch.cos(t)
        assert (
            torch.abs(derivs[1][1, margin:-margin] - dx_cos_true[margin:-margin]).mean()
            < 0.02
        )
        assert (
            torch.abs(
                derivs[2][1, margin:-margin] - d2x_cos_true[margin:-margin]
            ).mean()
            < 0.02
        )

    def test_d_orders_functional_interface(self):
        """Test the dxdt_orders functional interface."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        x = torch.sin(t)

        derivs = torch_dxdt.dxdt_orders(
            x, t, orders=[1, 2], kind="savitzky_golay", window_length=11, polyorder=4
        )

        assert 1 in derivs
        assert 2 in derivs
        assert derivs[1].shape == x.shape
        assert derivs[2].shape == x.shape

    def test_d_orders_polynomial(self):
        """Test d_orders on polynomial: x = t³ + 2t² + 3t + 4."""
        t = torch.linspace(-1, 1, 200, dtype=torch.float64)
        x = t**3 + 2 * t**2 + 3 * t + 4

        # True derivatives:
        # dx/dt = 3t² + 4t + 3
        # d²x/dt² = 6t + 4
        # d³x/dt³ = 6
        dx_true = 3 * t**2 + 4 * t + 3
        d2x_true = 6 * t + 4
        d3x_true = torch.full_like(t, 6.0)

        sg = torch_dxdt.SavitzkyGolay(window_length=15, polyorder=5)
        derivs = sg.d_orders(x, t, orders=[1, 2, 3])

        margin = 20

        error1 = torch.abs(derivs[1][margin:-margin] - dx_true[margin:-margin]).mean()
        assert error1 < 0.01, f"First derivative error {error1:.6f}"

        error2 = torch.abs(derivs[2][margin:-margin] - d2x_true[margin:-margin]).mean()
        assert error2 < 0.05, f"Second derivative error {error2:.6f}"

        error3 = torch.abs(derivs[3][margin:-margin] - d3x_true[margin:-margin]).mean()
        assert error3 < 0.5, f"Third derivative error {error3:.6f}"

    def test_d_orders_order_validation(self):
        """Test that requesting order > polyorder raises error."""
        t = torch.linspace(0, 1, 50, dtype=torch.float64)
        x = torch.sin(t)

        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=3)

        with pytest.raises(ValueError, match="Maximum order.*must be <="):
            sg.d_orders(x, t, orders=[1, 2, 4])  # order 4 > polyorder 3

    def test_d_orders_smooth_method(self):
        """Test that smooth() method works via d_orders infrastructure."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        x = torch.sin(t)

        sg = torch_dxdt.SavitzkyGolay(window_length=11, polyorder=4)

        # smooth() should give same result as d_orders with order=0
        x_smooth_via_smooth = sg.smooth(x, t)
        derivs = sg.d_orders(x, t, orders=[0])

        torch.testing.assert_close(x_smooth_via_smooth, derivs[0])


class TestWhittakerSmootherDerivative:
    """Test Whittaker-Eilers smoother/derivative implementation."""

    def test_basic_smoothing(self):
        """Test that Whittaker smoothing reduces noise."""
        t = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
        torch.manual_seed(42)
        x_true = torch.sin(t)
        x_noisy = x_true + 0.2 * torch.randn_like(t)

        wh = torch_dxdt.Whittaker(lmbda=100.0)
        x_smooth = wh.smooth(x_noisy, t)

        # Smoothed signal should be closer to true signal than noisy
        error_noisy = torch.abs(x_noisy - x_true).mean()
        error_smooth = torch.abs(x_smooth - x_true).mean()

        assert error_smooth < error_noisy, (
            f"Smoothed error {error_smooth:.4f} >= noisy error {error_noisy:.4f}"
        )

    def test_sine_derivative(self):
        """Test derivative of sin(t) should be cos(t)."""
        t = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
        x = torch.sin(t)
        dx_true = torch.cos(t)

        wh = torch_dxdt.Whittaker(lmbda=10.0)  # Lower lambda for cleaner signal
        dx = wh.d(x, t)

        # Compare interior points
        margin = 10
        error = torch.abs(dx[margin:-margin] - dx_true[margin:-margin]).mean()

        assert error < 0.05, f"Mean derivative error {error:.6f} exceeds tolerance"

    def test_noisy_sine_derivative(self):
        """Test derivative estimation on noisy sine wave."""
        t = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
        torch.manual_seed(42)
        x = torch.sin(t) + 0.1 * torch.randn_like(t)
        dx_true = torch.cos(t)

        wh = torch_dxdt.Whittaker(lmbda=1000.0)  # Higher lambda for noisy data
        dx = wh.d(x, t)

        # Compare interior points
        margin = 15
        error = torch.abs(dx[margin:-margin] - dx_true[margin:-margin]).mean()

        assert error < 0.1, f"Mean derivative error {error:.6f} exceeds tolerance"

    def test_polynomial_derivative(self):
        """Test derivative of polynomial: x = t² + 2t + 1, dx/dt = 2t + 2."""
        t = torch.linspace(-1, 1, 200, dtype=torch.float64)
        x = t**2 + 2 * t + 1
        dx_true = 2 * t + 2

        wh = torch_dxdt.Whittaker(lmbda=1.0)  # Low lambda for clean signal
        dx = wh.d(x, t)

        margin = 10
        error = torch.abs(dx[margin:-margin] - dx_true[margin:-margin]).mean()

        assert error < 0.01, f"Mean derivative error {error:.6f} exceeds tolerance"

    def test_smoothness_parameter(self):
        """Test that higher lambda gives smoother results."""
        t = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
        torch.manual_seed(42)
        x = torch.sin(t) + 0.3 * torch.randn_like(t)

        wh_low = torch_dxdt.Whittaker(lmbda=10.0)
        wh_high = torch_dxdt.Whittaker(lmbda=10000.0)

        x_smooth_low = wh_low.smooth(x, t)
        x_smooth_high = wh_high.smooth(x, t)

        # Higher lambda should give smaller second derivative (smoother)
        d2_low = torch.diff(torch.diff(x_smooth_low))
        d2_high = torch.diff(torch.diff(x_smooth_high))

        roughness_low = torch.abs(d2_low).mean()
        roughness_high = torch.abs(d2_high).mean()

        assert roughness_high < roughness_low, (
            f"High-lambda roughness {roughness_high:.6f} >= "
            f"low-lambda roughness {roughness_low:.6f}"
        )

    def test_d_order_parameter(self):
        """Test different difference orders in the penalty."""
        t = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
        x = torch.sin(t)

        wh1 = torch_dxdt.Whittaker(lmbda=100.0, d_order=1)
        wh2 = torch_dxdt.Whittaker(lmbda=100.0, d_order=2)
        wh3 = torch_dxdt.Whittaker(lmbda=100.0, d_order=3)

        x_smooth1 = wh1.smooth(x, t)
        x_smooth2 = wh2.smooth(x, t)
        x_smooth3 = wh3.smooth(x, t)

        # All should produce valid results
        assert x_smooth1.shape == x.shape
        assert x_smooth2.shape == x.shape
        assert x_smooth3.shape == x.shape

        # They should be different
        assert not torch.allclose(x_smooth1, x_smooth2)
        assert not torch.allclose(x_smooth2, x_smooth3)

    def test_batched_input(self):
        """Test Whittaker with batched input."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        x = torch.stack(
            [
                torch.sin(t),
                torch.cos(t),
                torch.sin(2 * t),
            ]
        )  # Shape: (3, 100)

        wh = torch_dxdt.Whittaker(lmbda=100.0)
        dx = wh.d(x, t)
        x_smooth = wh.smooth(x, t)

        assert dx.shape == (3, 100)
        assert x_smooth.shape == (3, 100)

    def test_functional_interface(self):
        """Test the dxdt functional interface with Whittaker."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        x = torch.sin(t)

        dx = torch_dxdt.dxdt(x, t, kind="whittaker", lmbda=100.0)

        assert dx.shape == x.shape
        # Check derivative is approximately cos(t)
        margin = 10
        error = torch.abs(dx[margin:-margin] - torch.cos(t)[margin:-margin]).mean()
        assert error < 0.1

    def test_smooth_interface(self):
        """Test smooth_x functional interface with Whittaker."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        torch.manual_seed(42)
        x = torch.sin(t) + 0.2 * torch.randn_like(t)

        x_smooth = torch_dxdt.smooth_x(x, t, kind="whittaker", lmbda=1000.0)

        assert x_smooth.shape == x.shape

    def test_second_order_derivative(self):
        """Test second order derivative via d_orders."""
        t = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
        x = torch.sin(t)
        d2x_true = -torch.sin(t)

        wh = torch_dxdt.Whittaker(lmbda=10.0)
        derivs = wh.d_orders(x, t, orders=[1, 2])

        margin = 15
        error = torch.abs(derivs[2][margin:-margin] - d2x_true[margin:-margin]).mean()

        assert error < 0.15, f"Second derivative error {error:.6f} exceeds tolerance"

    def test_d_orders_efficiency(self):
        """Test that d_orders returns multiple orders."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        x = torch.sin(t)

        wh = torch_dxdt.Whittaker(lmbda=100.0)
        derivs = wh.d_orders(x, t, orders=[0, 1, 2])

        assert 0 in derivs
        assert 1 in derivs
        assert 2 in derivs
        assert derivs[0].shape == x.shape
        assert derivs[1].shape == x.shape
        assert derivs[2].shape == x.shape

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="lmbda must be positive"):
            torch_dxdt.Whittaker(lmbda=-1.0)

        with pytest.raises(ValueError, match="d_order must be"):
            torch_dxdt.Whittaker(lmbda=100.0, d_order=4)

    def test_gradient_flow(self):
        """Test that gradients flow through Whittaker smoother."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        x = torch.sin(t)
        x.requires_grad_(True)

        wh = torch_dxdt.Whittaker(lmbda=100.0)
        dx = wh.d(x, t)

        loss = dx.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_nonzero(self):
        """Test that gradients are non-zero and reasonable."""
        t = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        x = torch.sin(t).clone()
        x.requires_grad_(True)

        wh = torch_dxdt.Whittaker(lmbda=10.0)
        x_smooth = wh.smooth(x, t)

        # Loss encourages smoothed output to match a target
        target = torch.cos(t)
        loss = ((x_smooth - target) ** 2).mean()
        loss.backward()

        assert x.grad is not None
        grad_norm = x.grad.norm()
        assert grad_norm > 0, "Gradient norm should be positive"

    def test_1d_input(self):
        """Test that 1D input works correctly."""
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=torch.float64)
        x = torch.sin(t)  # 1D tensor

        wh = torch_dxdt.Whittaker(lmbda=100.0)
        dx = wh.d(x, t)
        x_smooth = wh.smooth(x, t)

        assert dx.ndim == 1
        assert x_smooth.ndim == 1
        assert dx.shape == x.shape
        assert x_smooth.shape == x.shape

    def test_empty_input(self):
        """Test handling of empty input."""
        t = torch.tensor([], dtype=torch.float64)
        x = torch.tensor([], dtype=torch.float64)

        wh = torch_dxdt.Whittaker(lmbda=100.0)
        dx = wh.d(x, t)
        x_smooth = wh.smooth(x, t)

        assert dx.shape == x.shape
        assert x_smooth.shape == x.shape
