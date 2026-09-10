"""Tests for the regularizator classes."""

import numpy as np
import pytest
import quickpaver
from inv_toolbox.regularization import (
    DiscreteRegularizator,
    TikhonovFVMRegularizator,
    TikhonovMatRegularizator,
    TikhonovRegularizator,
    TVFVMRegularizator,
    TVMatRegularizator,
    TVRegularizator,
)
from inv_toolbox.utils import NDArrayFloat
from inv_toolbox.utils.preconditioner import LogTransform


def get_param_values() -> NDArrayFloat:
    """Generate a parameter field with some noise."""
    nx: int = 15
    ny: int = 26
    param: NDArrayFloat = np.zeros((nx, ny), dtype=np.float64)
    param[0:10, 5:15] = 5.0
    param[6:14, 7:14] = 10.0
    param[8:9, 2:25] = 20.0

    # Add some noise with a seed
    rng = np.random.default_rng(26659)
    param += rng.random((nx, ny)) * 5.0

    return param.ravel("F")


def test_discrete_exceptions() -> None:
    for modes in [[], [1.0]]:
        with pytest.raises(ValueError, match="At least two modes must be provided!"):
            DiscreteRegularizator(modes=modes)

    with pytest.raises(
        ValueError, match=r'penalty should be among \["least-squares", "gaussian"\]'
    ):
        DiscreteRegularizator(modes=[1, 20], penalty="Anything")  # ty: ignore[invalid-argument-type]

    with pytest.raises(
        ValueError, match=r'penalty should be among \["least-squares", "gaussian"\]'
    ):
        instance = DiscreteRegularizator(modes=[1, 20], penalty="gaussian")
        instance.penalty = "not valid"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "regularizator",
    [
        TikhonovRegularizator(quickpaver.RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        TikhonovMatRegularizator(
            quickpaver.RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)
        ),
        TikhonovFVMRegularizator(
            quickpaver.RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)
        ),
        TVRegularizator(quickpaver.RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        TVMatRegularizator(quickpaver.RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        TVFVMRegularizator(quickpaver.RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        DiscreteRegularizator(modes=[7.0, 15.0], penalty="gaussian"),
        DiscreteRegularizator(modes=[7.0, 8.5, 2.3, 15.0], penalty="gaussian"),
        DiscreteRegularizator(
            modes=[7.0, 8.5, 2.3, 15.0],
            penalty="gaussian",
            preconditioner=LogTransform(),
        ),
        DiscreteRegularizator(modes=[2.3, 15.0], penalty="least-squares"),
        DiscreteRegularizator(modes=[7.0, 8.5, 2.3, 15.0], penalty="least-squares"),
        DiscreteRegularizator(
            modes=[7.0, 8.5, 2.3, 15.0],
            penalty="least-squares",
            preconditioner=LogTransform(),
        ),
    ],
)
def test_regularizator_gradients_by_fd(regularizator) -> None:
    """Test the correctness of the gradients by finite differences."""
    param_values = get_param_values()

    grad_reg_fd = regularizator.eval_loss_gradient(
        param_values, is_finite_differences=True
    )
    grad_reg_analytic = regularizator.eval_loss_gradient(param_values)
    np.testing.assert_allclose(grad_reg_fd, grad_reg_analytic, atol=1e-5)
