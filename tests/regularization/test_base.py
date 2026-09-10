import numpy as np
import pytest
import quickpaver
from inv_toolbox.regularization.base import (
    ConstantRegWeight,
    Regularizator,
    RegWeightUpdateStrategy,
    make_spatial_gradient_matrices,
    make_spatial_permutation_matrices,
)
from inv_toolbox.utils import NDArrayFloat


class _DummyRegularizator(Regularizator):
    """Minimal concrete Regularizator used to exercise the base-class logic."""

    def _eval_loss(self, values: NDArrayFloat) -> float:
        return float(np.sum(values**2))

    def _eval_loss_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
        return 2.0 * values


@pytest.mark.parametrize("sub_selection", (None, np.arange(1000), np.arange(1000)[::2]))
def test_make_spatial_permutation_matrices(sub_selection) -> None:
    out_x, out_y = make_spatial_permutation_matrices(
        quickpaver.RectilinearGrid(nx=10, ny=100, dx=1.0, dy=1.0),
        sub_selection=sub_selection,
    )


@pytest.mark.parametrize("which", ("forward", "backward", "both"))
@pytest.mark.parametrize("sub_selection", (None, np.arange(1000), np.arange(1000)[::5]))
def test_make_spatial_gradient_matrices(which, sub_selection) -> None:
    make_spatial_gradient_matrices(
        quickpaver.RectilinearGrid(nx=10, ny=100, dx=1.0, dy=1.0),
        sub_selection=sub_selection,
        which=which,
    )


def test_make_spatial_gradient_matrices_nx_below_2() -> None:
    """When nx < 2, the x contribution must be skipped (all zeros)."""
    grid = quickpaver.RectilinearGrid(nx=1, ny=10, dx=1.0, dy=1.0)
    mat_grad_x, mat_grad_y = make_spatial_gradient_matrices(grid)
    assert mat_grad_x.nnz == 0
    assert mat_grad_y.nnz > 0


def test_make_spatial_gradient_matrices_ny_below_2() -> None:
    """When ny < 2, the y contribution must be skipped (all zeros)."""
    grid = quickpaver.RectilinearGrid(nx=10, ny=1, dx=1.0, dy=1.0)
    mat_grad_x, mat_grad_y = make_spatial_gradient_matrices(grid)
    assert mat_grad_x.nnz > 0
    assert mat_grad_y.nnz == 0


def test_make_spatial_permutation_matrices_nx_below_2() -> None:
    """When nx < 2, the x permutation matrix must stay empty."""
    grid = quickpaver.RectilinearGrid(nx=1, ny=10, dx=1.0, dy=1.0)
    mat_perm_x, mat_perm_y = make_spatial_permutation_matrices(grid)
    assert mat_perm_x.nnz == 0
    assert mat_perm_y.nnz > 0


def test_make_spatial_permutation_matrices_ny_below_2() -> None:
    """When ny < 2, the y permutation matrix must stay empty."""
    grid = quickpaver.RectilinearGrid(nx=10, ny=1, dx=1.0, dy=1.0)
    mat_perm_x, mat_perm_y = make_spatial_permutation_matrices(grid)
    assert mat_perm_x.nnz > 0
    assert mat_perm_y.nnz == 0


class TestRegWeightUpdateStrategy:
    def test_default_reg_weight(self) -> None:
        strategy = RegWeightUpdateStrategy()
        assert strategy.reg_weight == 1.0

    def test_getter_setter(self) -> None:
        strategy = RegWeightUpdateStrategy(2.0)
        assert strategy.reg_weight == 2.0
        strategy.reg_weight = 5.0
        assert strategy.reg_weight == 5.0

    def test_is_adaptive_is_false(self) -> None:
        assert RegWeightUpdateStrategy.is_adaptive() is False
        assert RegWeightUpdateStrategy(1.0).is_adaptive() is False

    def test_update_reg_weight_default_no_op(self) -> None:
        strategy = RegWeightUpdateStrategy(1.0)
        has_changed = strategy.update_reg_weight(
            loss_ls_history=[1.0, 2.0],
            loss_reg_history=[0.5, 0.4],
            reg_weight_history=[1.0, 1.0],
            loss_ls_grad=np.array([1.0]),
            loss_reg_grad=np.array([1.0]),
            n_obs=10,
        )
        assert has_changed is False
        assert strategy.reg_weight == 1.0

    def test_update_reg_weight_with_logger(self) -> None:
        import logging

        strategy = RegWeightUpdateStrategy(1.0)
        logger = logging.getLogger("test-base")
        has_changed = strategy.update_reg_weight(
            [1.0], [1.0], [1.0], np.array([1.0]), np.array([1.0]), 5, logger=logger
        )
        assert has_changed is False


class TestConstantRegWeight:
    def test_default(self) -> None:
        c = ConstantRegWeight()
        assert c.reg_weight == 1.0
        assert c.is_adaptive() is False

    def test_custom_value(self) -> None:
        c = ConstantRegWeight(reg_weight=42.0)
        assert c.reg_weight == 42.0
        assert (
            c.update_reg_weight(
                [1.0], [1.0], [1.0], np.array([1.0]), np.array([1.0]), 1
            )
            is False
        )
        # a constant strategy never changes the weight
        assert c.reg_weight == 42.0


class TestRegularizator:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            Regularizator()  # type: ignore[abstract]

    def test_eval_loss_1d(self) -> None:
        reg = _DummyRegularizator()
        assert reg.eval_loss(np.array([1.0, 2.0, 3.0])) == pytest.approx(14.0)

    def test_eval_loss_raises_on_non_1d(self) -> None:
        reg = _DummyRegularizator()
        with pytest.raises(ValueError, match="expects a 1D vector"):
            reg.eval_loss(np.array([[1.0, 2.0]]))

    def test_eval_loss_gradient_analytical_1d(self) -> None:
        reg = _DummyRegularizator()
        np.testing.assert_allclose(
            reg.eval_loss_gradient_analytical(np.array([1.0, 2.0])), [2.0, 4.0]
        )

    def test_eval_loss_gradient_analytical_raises_on_non_1d(self) -> None:
        reg = _DummyRegularizator()
        with pytest.raises(ValueError, match="expects a 1D vector"):
            reg.eval_loss_gradient_analytical(np.array([[1.0, 2.0]]))

    def test_eval_loss_gradient_raises_on_non_1d(self) -> None:
        reg = _DummyRegularizator()
        with pytest.raises(ValueError, match="expects a 1D vector"):
            reg.eval_loss_gradient(np.array([[1.0, 2.0]]))

    def test_eval_loss_gradient_analytical_path(self) -> None:
        reg = _DummyRegularizator()
        np.testing.assert_allclose(
            reg.eval_loss_gradient(np.array([1.0, 2.0, 3.0])), [2.0, 4.0, 6.0]
        )

    def test_eval_loss_gradient_finite_differences_path(self) -> None:
        reg = _DummyRegularizator()
        np.testing.assert_allclose(
            reg.eval_loss_gradient(
                np.array([1.0, 2.0, 3.0]), is_finite_differences=True
            ),
            [2.0, 4.0, 6.0],
            atol=1e-4,
        )
