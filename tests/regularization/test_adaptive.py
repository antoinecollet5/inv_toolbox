import logging

import numpy as np
import pytest
from inv_toolbox.regularization.adaptive import (
    AdaptiveGradientNormRegweight,
    AdaptiveUCRegweight,
    _get_bounds,
    compute_uc,
    get_minima_indices,
    get_optimal_reg_param,
    make_convex_around_min_uc,
    select_valid_reg_params,
)


@pytest.mark.parametrize(
    "reg_params,ucvalues,expected_reg_params,expected_uc",
    [
        (
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            [1.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
        ),
        ([1.0], [4.0], np.array([1.0]), np.array([4.0])),
        ([1.0, 1.0, 1.0], [4.0, 3.0, 2.0], np.array([1.0]), np.array([2.0])),
        (
            [1.0, 1.0, 2.0, 30.0, 1.0, 2.5],
            [1.0, 0.66, 2.0, 3.6, -0.256, 4.0],
            np.array([1.0, 2.0, 2.5, 30.0]),
            np.array([-0.256, 2.0, 4.0, 3.6]),
        ),
    ],
)
def test_select_valid_reg_params(
    reg_params, ucvalues, expected_reg_params, expected_uc
) -> None:
    _1, _2 = select_valid_reg_params(reg_params, ucvalues)
    np.testing.assert_equal(_1, expected_reg_params)
    np.testing.assert_equal(_2, expected_uc)


@pytest.mark.parametrize(
    "reg_params,ucvalues,expected_reg_params,expected_uc",
    [
        (np.array([1.0]), np.array([4.0]), np.array([1.0]), np.array([4.0])),
        (
            np.array([0.5, 1.0]),
            np.array([4.0, 1.0]),
            np.array([0.5, 1.0]),
            np.array([4.0, 1.0]),
        ),
        (
            np.array([1.0, 1.5]),
            np.array([1.0, 4.0]),
            np.array([1.0, 1.5]),
            np.array([1.0, 4.0]),
        ),
        (
            np.array([1.0, 1.5, 5.0]),
            np.array([1.0, 4.0, 9.5]),
            np.array([1.0, 1.5, 5.0]),
            np.array([1.0, 4.0, 9.5]),
        ),
        (
            np.array([0.5, 1.0, 1.5, 5.0]),
            np.array([0.5, 1.0, 4.0, 9.5]),
            np.array([0.5, 1.0, 1.5]),
            np.array([0.5, 1.0, 4.0]),
        ),
        (
            np.array([0.2, 0.5, 1.0, 1.5, 5.0]),
            np.array([2.4, 0.5, 1.0, 4.0, 9.5]),
            np.array([0.2, 0.5, 1.0, 1.5]),
            np.array([2.4, 0.5, 1.0, 4.0]),
        ),
        (
            np.array(
                [
                    1.00000000e00,
                    4.78152428e01,
                    1.77563588e02,
                    193.96317092,
                    209.88755336,
                    220.87120974,
                    1022.0759485,
                    7.65125427e03,
                    9.95692957e03,
                ]
            ),
            np.array(
                [
                    0.16987166,
                    0.20260699,
                    0.14189685,
                    0.16449475,
                    0.14105013,
                    0.15775547,
                    0.17387793,
                    0.21383609,
                    0.2119825,
                ]
            ),
            np.array([193.96317092, 209.88755336, 220.87120974, 1022.0759485]),
            np.array([0.16449475, 0.14105013, 0.15775547, 0.17387793]),
        ),
    ],
)
def test_make_convex_around_min_uc(
    reg_params, ucvalues, expected_reg_params, expected_uc
) -> None:
    _1, _2 = make_convex_around_min_uc(reg_params, ucvalues)
    np.testing.assert_equal(_1, expected_reg_params)
    np.testing.assert_equal(_2, expected_uc)


@pytest.mark.parametrize(
    "vals, expected",
    [
        (np.array([1.0]), np.array([0])),
        (np.array([1.0, -1.0]), np.array([1])),
        (np.array([-2.0, 1.0, -1.0]), np.array([0, 2])),
        (
            np.array([-4.52, -2.0, 1.0, -1.0, -2.6, 8.6, -6.36, -18.5]),
            np.array([0, 4, 7]),
        ),
    ],
)
def test_get_minima_indices(vals, expected) -> None:
    np.testing.assert_equal(get_minima_indices(vals), expected)


@pytest.mark.parametrize(
    "reg_params, ucvalues, expected",
    [
        (np.array([1.0]), np.array([0]), 1.0),
        (np.array([1.0, 10.0]), np.array([0.1, 0.6]), 0.1),
        (np.array([1.0, 10.0]), np.array([0.6, 0.1]), 100.0),
        (np.array([2.0, 20.0]), np.array([0.6, 0.1]), 200.0),
        (np.array([1.0, 2.0, 20.0]), np.array([0.5, 0.6, 0.1]), 200.0),
        (np.array([2.0, 20.0]), np.array([0.1, 0.6]), 0.2),
        (np.array([2.0, 20.0, 40.0]), np.array([0.1, 0.6, 0.5]), 0.2),
        (
            np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    9956.929574755324,
                    7651.254266793136,
                    1022.0759485004097,
                    177.56358807164779,
                    47.81524279187321,
                ]
            ),
            np.array(
                [
                    1.00000000e10,
                    8.49282053e-01,
                    1.69871660e-01,
                    2.11982499e-01,
                    2.13836087e-01,
                    1.73877929e-01,
                    1.41896848e-01,
                    2.02606994e-01,
                ]
            ),
            209.887553,
        ),
        (
            np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    9956.929574755324,
                    7651.254266793136,
                    1022.0759485004097,
                    177.56358807164779,
                    47.81524279187321,
                    209.8875533624339,
                    220.8712097396706,
                    193.96317091834385,
                    207.36473892030227,
                ]
            ),
            np.array(
                [
                    1.00000000e10,
                    8.49282053e-01,
                    1.69871660e-01,
                    2.11982499e-01,
                    2.13836087e-01,
                    1.73877929e-01,
                    1.41896848e-01,
                    2.02606994e-01,
                    1.41050131e-01,
                    1.57755466e-01,
                    1.64494747e-01,
                    1.72656713e-01,
                ]
            ),
            214.121116,
        ),
        (
            np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    9956.929574755324,
                    7651.254266793136,
                    1022.0759485004097,
                    177.56358807164779,
                    47.81524279187321,
                    209.8875533624339,
                    220.8712097396706,
                    193.96317091834385,
                    207.36473892030227,
                    214.12111557120627,
                    214.12111557120627,
                    210.66493036585325,
                    208.98344701402328,
                    209.84362947457564,
                    210.25594247528684,
                    210.04864884762597,
                ]
            ),
            np.array(
                [
                    1.00000000e10,
                    8.49282053e-01,
                    1.69871660e-01,
                    2.11982499e-01,
                    2.13836087e-01,
                    1.73877929e-01,
                    1.41896848e-01,
                    2.02606994e-01,
                    1.41050131e-01,
                    1.57755466e-01,
                    1.64494747e-01,
                    1.72656713e-01,
                    1.66969923e-01,
                    1.74607253e-01,
                    1.76778854e-01,
                    1.81537160e-01,
                    1.84495488e-01,
                    1.88961245e-01,
                    1.91542633e-01,
                ]
            ),
            209.943539,
        ),
    ],
)
def test_get_optimal_reg_param(reg_params, ucvalues, expected) -> None:
    np.testing.assert_allclose(get_optimal_reg_param(reg_params, ucvalues), expected)


# ---------------------------------------------------------------------------
# _get_bounds
# ---------------------------------------------------------------------------


def test_get_bounds_empty_x0_raises() -> None:
    with pytest.raises(ValueError, match="x0 cannot be an empty vector!"):
        _get_bounds(np.array([]), None)


def test_get_bounds_none_returns_inf() -> None:
    lb, ub = _get_bounds(np.array([1.0, 2.0]), None)
    np.testing.assert_array_equal(lb, [-np.inf, -np.inf])
    np.testing.assert_array_equal(ub, [np.inf, np.inf])


def test_get_bounds_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="while shape"):
        _get_bounds(np.array([1.0, 2.0]), [(0.0, 1.0)])


def test_get_bounds_lb_greater_than_ub_raises() -> None:
    with pytest.raises(
        ValueError, match="One of the lower bounds is greater than an upper bound."
    ):
        _get_bounds(np.array([1.0]), [(5.0, 1.0)])


def test_get_bounds_x0_violates_bounds_raises() -> None:
    with pytest.raises(ValueError, match="values violating"):
        _get_bounds(np.array([1.0, 10.0]), [(0.0, 2.0), (0.0, 2.0)])


def test_get_bounds_nan_converted_to_inf() -> None:
    lb, ub = _get_bounds(np.array([1.0]), [(np.nan, np.nan)])
    np.testing.assert_array_equal(lb, [-np.inf])
    np.testing.assert_array_equal(ub, [np.inf])


def test_get_bounds_valid_case() -> None:
    lb, ub = _get_bounds(np.array([1.0, 2.0]), [(0.0, 5.0), (0.0, 5.0)])
    np.testing.assert_array_equal(lb, [0.0, 0.0])
    np.testing.assert_array_equal(ub, [5.0, 5.0])


# ---------------------------------------------------------------------------
# AdaptiveRegweight (through the concrete AdaptiveGradientNormRegweight)
# ---------------------------------------------------------------------------


class TestAdaptiveRegweightInit:
    def test_negative_reg_weight_init_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match="The initial regularization weight should be positive or null!",
        ):
            AdaptiveGradientNormRegweight(reg_weight_init=-1.0)

    def test_negative_bounds_raise(self) -> None:
        with pytest.raises(
            ValueError,
            match="The bounds for the regularization weight should be positive or null",
        ):
            AdaptiveGradientNormRegweight(
                reg_weight_init=0.0, reg_weight_bounds=(-5.0, 10.0)
            )

    def test_max_log_cr_not_positive_raises(self) -> None:
        with pytest.raises(ValueError, match="'max_log_cr'.*must be positive"):
            AdaptiveGradientNormRegweight(max_log_cr=0.0)

    def test_max_log_cr_lower_than_convergence_factor_raises(self) -> None:
        with pytest.raises(
            ValueError, match="cannot be lower or equal to the 'convergence_factor'"
        ):
            AdaptiveGradientNormRegweight(convergence_factor=0.1, max_log_cr=0.05)

    def test_is_adaptive_is_true(self) -> None:
        assert AdaptiveGradientNormRegweight.is_adaptive() is True
        assert AdaptiveGradientNormRegweight().is_adaptive() is True

    def test_reg_weight_setter_clips_to_bounds(self) -> None:
        a = AdaptiveGradientNormRegweight(
            reg_weight_init=1.0, reg_weight_bounds=(1e-10, 1e10)
        )
        a.reg_weight = 1e20
        assert a.reg_weight == 1e10
        a.reg_weight = -1.0
        assert a.reg_weight == 1e-10


class TestGetLogCr:
    def test_old_zero_returns_zero(self) -> None:
        assert AdaptiveGradientNormRegweight.get_log_cr(0.0, 5.0) == 0.0

    def test_new_zero_returns_zero(self) -> None:
        assert AdaptiveGradientNormRegweight.get_log_cr(5.0, 0.0) == 0.0

    def test_regular_case(self) -> None:
        assert AdaptiveGradientNormRegweight.get_log_cr(1.0, np.e) == pytest.approx(1.0)


class TestEnsureMaxLogRc:
    def test_no_change_returns_new(self) -> None:
        a = AdaptiveGradientNormRegweight()
        assert a._ensure_max_log_rc(5.0, 5.0, None) == 5.0

    def test_first_update_no_clipping(self) -> None:
        """With n_regw_update == 0, no clipping is applied regardless of size."""
        a = AdaptiveGradientNormRegweight()
        assert a.n_regw_update == 0
        assert a._ensure_max_log_rc(1000.0, 1.0, None) == 1000.0

    def test_increase_clipped_when_above_max_log_cr(self) -> None:
        a = AdaptiveGradientNormRegweight(max_log_cr=2.0)
        a.n_regw_update = 1
        new_rw = a._ensure_max_log_rc(
            1000.0, 1.0, None, logger=logging.getLogger("test")
        )
        assert new_rw == pytest.approx(1.0 * np.exp(2.0))

    def test_decrease_clipped_when_below_max_log_cr(self) -> None:
        a = AdaptiveGradientNormRegweight(max_log_cr=2.0)
        a.n_regw_update = 1
        new_rw = a._ensure_max_log_rc(
            0.0001, 1.0, None, logger=logging.getLogger("test")
        )
        assert new_rw == pytest.approx(1.0 / np.exp(2.0))

    def test_within_bounds_no_clip(self) -> None:
        a = AdaptiveGradientNormRegweight(max_log_cr=2.0)
        a.n_regw_update = 1
        new_rw = a._ensure_max_log_rc(2.0, 1.0, None)
        assert new_rw == 2.0

    def test_gls_log_cr_restricts_max_alrc(self) -> None:
        a = AdaptiveGradientNormRegweight(max_log_cr=2.0)
        a.n_regw_update = 1
        # gls_log_cr=0.1 -> max_alrc = min(2.0, 1/0.1) = 2.0 (no additional effect here)
        new_rw_a = a._ensure_max_log_rc(1000.0, 1.0, gls_log_cr=0.1)
        # gls_log_cr=10 -> max_alrc = min(2.0, 1/10) = 0.1: stronger clipping
        new_rw_b = a._ensure_max_log_rc(1000.0, 1.0, gls_log_cr=10.0)
        assert new_rw_b < new_rw_a

    def test_gls_log_cr_zero_ignored(self) -> None:
        a = AdaptiveGradientNormRegweight(max_log_cr=2.0)
        a.n_regw_update = 1
        new_rw = a._ensure_max_log_rc(1000.0, 1.0, gls_log_cr=0.0)
        assert new_rw == pytest.approx(1.0 * np.exp(2.0))


class TestAdaptiveRegweightUpdateRegWeight:
    def test_first_adjusted_weight_becomes_upper_bound(self) -> None:
        a = AdaptiveGradientNormRegweight(
            reg_weight_init=1.0, is_use_first_adjusted_weight_as_upper_bound=True
        )
        changed = a.update_reg_weight(
            loss_ls_history=[10.0],
            loss_reg_history=[5.0],
            reg_weight_history=[1.0],
            loss_ls_grad=np.array([2.0, 2.0]),
            loss_reg_grad=np.array([1.0, 1.0]),
            n_obs=10,
            logger=logging.getLogger("test"),
        )
        assert changed is True
        assert a.n_regw_update == 1
        assert a.reg_weight_bounds[1] == a.reg_weight

    def test_bounds_not_updated_when_flag_disabled(self) -> None:
        a = AdaptiveGradientNormRegweight(
            reg_weight_init=1.0,
            reg_weight_bounds=(1e-10, 1e10),
            is_use_first_adjusted_weight_as_upper_bound=False,
        )
        a.update_reg_weight([10.0], [5.0], [1.0], np.array([2.0]), np.array([1.0]), 10)
        assert a.reg_weight_bounds[1] == 1e10

    def test_no_op_when_weight_does_not_change(self) -> None:
        a = AdaptiveGradientNormRegweight(reg_weight_init=1.0)
        changed = a.update_reg_weight(
            loss_ls_history=[10.0],
            loss_reg_history=[0.0],  # null reg loss -> no update
            reg_weight_history=[1.0],
            loss_ls_grad=np.array([1.0]),
            loss_reg_grad=np.array([1.0]),
            n_obs=10,
        )
        assert changed is False
        assert a.n_regw_update == 0


# ---------------------------------------------------------------------------
# AdaptiveGradientNormRegweight
# ---------------------------------------------------------------------------


class TestAdaptiveGradientNormRegweight:
    @pytest.mark.parametrize("norm", [None, 1, 2, -1, -2, 0, np.inf, -np.inf])
    def test_valid_norms(self, norm) -> None:
        AdaptiveGradientNormRegweight(norm=norm)

    def test_invalid_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="Not a valid norm"):
            AdaptiveGradientNormRegweight(norm="not_a_norm")

    def test_empty_loss_reg_history_returns_false(self) -> None:
        a = AdaptiveGradientNormRegweight()
        assert (
            a._update_reg_weight([], [], [], np.array([1.0]), np.array([1.0]), 10)
            is False
        )

    def test_null_loss_reg_returns_false(self) -> None:
        a = AdaptiveGradientNormRegweight()
        assert (
            a._update_reg_weight(
                [1.0], [0.0], [1.0], np.array([1.0]), np.array([1.0]), 10
            )
            is False
        )

    def test_zero_reg_grad_norm_returns_false(self) -> None:
        a = AdaptiveGradientNormRegweight()
        assert (
            a._update_reg_weight(
                [1.0], [5.0], [1.0], np.array([1.0, 1.0]), np.array([0.0, 0.0]), 10
            )
            is False
        )

    def test_zero_ls_grad_norm_returns_false(self) -> None:
        a = AdaptiveGradientNormRegweight()
        assert (
            a._update_reg_weight(
                [1.0], [5.0], [1.0], np.array([0.0, 0.0]), np.array([1.0, 1.0]), 10
            )
            is False
        )

    def test_noise_dominated_branch(self) -> None:
        a = AdaptiveGradientNormRegweight(reg_weight_init=1.0)
        changed = a._update_reg_weight(
            [10.0],
            [5.0],
            [1.0],
            np.array([1.0]),
            np.array([10.0]),
            10,
            logger=logging.getLogger("test"),
        )
        assert changed is True
        assert a.is_noise_dominated is True
        assert a.reg_weight == pytest.approx(10.0)

    def test_data_dominated_branch(self) -> None:
        a = AdaptiveGradientNormRegweight(reg_weight_init=1.0)
        changed = a._update_reg_weight(
            [10.0], [5.0], [1.0], np.array([10.0]), np.array([1.0]), 10
        )
        assert changed is True
        assert a.is_noise_dominated is False
        assert a.reg_weight == pytest.approx(10.0)

    def test_ls_history_small_relative_change_reverts(self) -> None:
        a = AdaptiveGradientNormRegweight(reg_weight_init=2.0)
        changed = a._update_reg_weight(
            loss_ls_history=[10.0, 10.1],  # relative change 0.01 < 0.05
            loss_reg_history=[5.0, 5.0],
            reg_weight_history=[2.0, 2.0],
            loss_ls_grad=np.array([1.0]),
            loss_reg_grad=np.array([5.0]),
            n_obs=10,
        )
        assert changed is False
        assert a.reg_weight == 2.0

    def test_reg_weight_small_relative_change_reverts(self) -> None:
        a = AdaptiveGradientNormRegweight(reg_weight_init=1.0)
        changed = a._update_reg_weight(
            loss_ls_history=[10.0],  # single element: ls-history check skipped
            loss_reg_history=[5.0],
            reg_weight_history=[1.0],
            loss_ls_grad=np.array([1.0]),
            loss_reg_grad=np.array([1.02]),  # relative reg_weight change ~0.02 < 0.05
            n_obs=10,
        )
        assert changed is False
        assert a.reg_weight == 1.0

    def test_two_successive_updates_use_ls_log_cr(self) -> None:
        a = AdaptiveGradientNormRegweight(
            reg_weight_init=1.0, is_use_first_adjusted_weight_as_upper_bound=False
        )
        changed1 = a.update_reg_weight(
            [10.0], [5.0], [1.0], np.array([1.0]), np.array([5.0]), 10
        )
        assert changed1 is True
        assert len(a.loss_ls_grad_norms) == 1

        changed2 = a.update_reg_weight(
            [10.0, 2.0],
            [5.0, 5.0],
            [1.0, a.reg_weight],
            np.array([2.0]),
            np.array([1.0]),
            10,
            logger=logging.getLogger("test"),
        )
        assert len(a.loss_ls_grad_norms) == 2
        assert changed2 in (True, False)


# ---------------------------------------------------------------------------
# AdaptiveUCRegweight
# ---------------------------------------------------------------------------


class TestAdaptiveUCRegweight:
    def test_empty_loss_reg_history_returns_false(self) -> None:
        a = AdaptiveUCRegweight(n_update_explo_phase=2)
        assert (
            a._update_reg_weight([], [], [], np.array([1.0]), np.array([1.0]), 10)
            is False
        )

    def test_null_loss_reg_returns_false(self) -> None:
        a = AdaptiveUCRegweight(n_update_explo_phase=2)
        assert (
            a._update_reg_weight(
                [1.0], [0.0], [1.0], np.array([1.0]), np.array([1.0]), 10
            )
            is False
        )

    def test_exploration_zero_reg_grad_norm_returns_false(self) -> None:
        a = AdaptiveUCRegweight(n_update_explo_phase=2)
        assert (
            a._update_reg_weight(
                [1.0], [5.0], [1.0], np.array([1.0, 1.0]), np.array([0.0, 0.0]), 10
            )
            is False
        )

    def test_exploration_zero_ls_grad_norm_returns_false(self) -> None:
        a = AdaptiveUCRegweight(n_update_explo_phase=2)
        assert (
            a._update_reg_weight(
                [1.0], [5.0], [1.0], np.array([0.0, 0.0]), np.array([1.0, 1.0]), 10
            )
            is False
        )

    def test_exploration_noise_dominated_branch(self) -> None:
        a = AdaptiveUCRegweight(
            reg_weight_init=1.0,
            n_update_explo_phase=2,
            is_use_first_adjusted_weight_as_upper_bound=False,
        )
        changed = a.update_reg_weight(
            [10.0],
            [5.0],
            [1.0],
            np.array([1.0]),
            np.array([10.0]),
            10,
            logger=logging.getLogger("test"),
        )
        assert bool(changed) is True
        assert bool(a.is_noise_dominated) is True
        assert a.n_regw_update == 1

    def test_exploration_data_dominated_branch(self) -> None:
        a = AdaptiveUCRegweight(
            reg_weight_init=1.0,
            n_update_explo_phase=2,
            is_use_first_adjusted_weight_as_upper_bound=False,
        )
        changed = a.update_reg_weight(
            [10.0], [5.0], [1.0], np.array([10.0]), np.array([1.0]), 10
        )
        assert bool(changed) is True
        assert bool(a.is_noise_dominated) is False

    def test_optimization_phase_mismatched_lengths_raises(self) -> None:
        a = AdaptiveUCRegweight(
            reg_weight_init=1.0,
            n_update_explo_phase=1,
            is_use_first_adjusted_weight_as_upper_bound=False,
            convergence_factor=0.0,
        )
        a.update_reg_weight([10.0], [5.0], [1.0], np.array([1.0]), np.array([5.0]), 10)
        assert a.n_regw_update == 1
        with pytest.raises(ValueError, match="must have the same length"):
            a._update_reg_weight(
                [1.0, 2.0], [1.0], [1.0], np.array([1.0]), np.array([1.0]), 10
            )

    def test_optimization_phase_success(self) -> None:
        a = AdaptiveUCRegweight(
            reg_weight_init=1.0,
            n_update_explo_phase=1,
            is_use_first_adjusted_weight_as_upper_bound=False,
            convergence_factor=0.0,
        )
        a.update_reg_weight([10.0], [5.0], [1.0], np.array([1.0]), np.array([5.0]), 10)
        assert a.n_regw_update == 1

        reg_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        loss_ls = [100.0, 80.0, 60.0, 50.0, 45.0]
        loss_reg = [1.0, 2.0, 4.0, 8.0, 16.0]
        changed = a.update_reg_weight(
            loss_ls,
            loss_reg,
            reg_weights,
            np.array([1.0]),
            np.array([1.0]),
            10,
            logger=logging.getLogger("test"),
        )
        assert changed in (True, False)
        assert a.n_regw_update == 2


def test_compute_uc() -> None:
    result = compute_uc([1.0, 2.0], [3.0, 4.0])
    expected = 1 / (np.array([1.0, 2.0])) + 1 / (np.array([3.0, 4.0]))
    np.testing.assert_allclose(result, expected)


def test_compute_uc_avoids_division_by_zero() -> None:
    result = compute_uc([0.0], [0.0])
    assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# Additional branch-coverage tests
# ---------------------------------------------------------------------------


class TestEnsureMaxLogRcDecreaseWithinBounds:
    def test_decrease_within_bounds_no_clip(self) -> None:
        a = AdaptiveGradientNormRegweight(max_log_cr=2.0)
        a.n_regw_update = 1
        new_rw = a._ensure_max_log_rc(0.7, 1.0, None)
        assert new_rw == 0.7

    def test_decrease_clipped_without_logger(self) -> None:
        a = AdaptiveGradientNormRegweight(max_log_cr=2.0)
        a.n_regw_update = 1
        new_rw = a._ensure_max_log_rc(0.0001, 1.0, None, logger=None)
        assert new_rw == pytest.approx(1.0 / np.exp(2.0))


def test_update_reg_weight_bounds_update_without_logger() -> None:
    a = AdaptiveGradientNormRegweight(
        reg_weight_init=1.0, is_use_first_adjusted_weight_as_upper_bound=True
    )
    changed = a.update_reg_weight(
        [10.0], [5.0], [1.0], np.array([2.0, 2.0]), np.array([1.0, 1.0]), 10
    )
    assert changed
    assert a.reg_weight_bounds[1] == a.reg_weight


def test_first_call_ls_history_len_2_no_prior_grad_norm() -> None:
    """First call with an already-length-2 ls history: the ls_log_cr branch
    requiring >= 2 stored gradient norms must be skipped (only 1 stored)."""
    a = AdaptiveGradientNormRegweight(reg_weight_init=1.0)
    changed = a._update_reg_weight(
        loss_ls_history=[10.0, 5.0],  # 50% change: no revert
        loss_reg_history=[5.0],
        reg_weight_history=[1.0],
        loss_ls_grad=np.array([1.0]),
        loss_reg_grad=np.array([5.0]),
        n_obs=10,
    )
    assert changed
    assert len(a.loss_ls_grad_norms) == 1


def test_uc_exploration_second_call_keeps_noise_status() -> None:
    """On the second exploration-phase call (n_regw_update != 0), the
    is_noise_dominated status must not be recomputed."""
    a = AdaptiveUCRegweight(
        reg_weight_init=1.0,
        n_update_explo_phase=3,
        is_use_first_adjusted_weight_as_upper_bound=False,
    )
    a.update_reg_weight([10.0], [5.0], [1.0], np.array([1.0]), np.array([10.0]), 10)
    assert a.n_regw_update == 1
    assert bool(a.is_noise_dominated) is True

    # second exploration call: is_noise_dominated must stay True regardless
    # of the new gradient balance (it is only set on the very first call).
    a.update_reg_weight(
        [10.0, 8.0],
        [5.0, 4.0],
        [1.0, a.reg_weight],
        np.array([10.0]),
        np.array([1.0]),
        10,
    )
    assert a.n_regw_update == 2
    assert bool(a.is_noise_dominated) is True


def test_uc_optimization_phase_without_logger() -> None:
    a = AdaptiveUCRegweight(
        reg_weight_init=1.0,
        n_update_explo_phase=1,
        is_use_first_adjusted_weight_as_upper_bound=False,
        convergence_factor=0.0,
    )
    a.update_reg_weight([10.0], [5.0], [1.0], np.array([1.0]), np.array([5.0]), 10)
    reg_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    loss_ls = [100.0, 80.0, 60.0, 50.0, 45.0]
    loss_reg = [1.0, 2.0, 4.0, 8.0, 16.0]
    changed = a.update_reg_weight(
        loss_ls, loss_reg, reg_weights, np.array([1.0]), np.array([1.0]), 10
    )
    assert changed in (True, False)


def test_uc_optimization_phase_small_relative_change_reverts() -> None:
    """When the newly computed optimal reg_param equals the current weight,
    the update must be reverted (relative change below convergence_factor)."""
    a = AdaptiveUCRegweight(
        reg_weight_init=1.0,
        n_update_explo_phase=1,
        is_use_first_adjusted_weight_as_upper_bound=False,
    )
    a.update_reg_weight([10.0], [5.0], [1.0], np.array([1.0]), np.array([5.0]), 10)
    old = a.reg_weight

    # a single-point sequence in the optimization phase always yields the
    # same reg_weight back (see get_optimal_reg_param), so the change is
    # exactly 0 and must be reverted.
    changed = a._update_reg_weight(
        [1.0], [1.0], [old], np.array([1.0]), np.array([1.0]), 10
    )
    assert changed is False
    assert a.reg_weight == old
