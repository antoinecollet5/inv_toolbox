import sys
from contextlib import nullcontext as does_not_raise
from typing import Optional, Tuple

import numdifftools as nd
import numpy as np
import pytest
from inv_toolbox.utils import (
    MeanType,
    NDArrayFloat,
    amean_gradient,
    arithmetic_mean,
    dxi_arithmetic_mean,
    dxi_harmonic_mean,
    get_mean_values_for_last_axis,
    get_mean_values_gradient_for_last_axis,
    gmean_gradient,
    harmonic_mean,
    hmean_gradient,
)
from scipy.stats import gmean, hmean


@pytest.mark.parametrize(
    "xi,xj,expected", [(0.0, 0.0, 0.0), (0.0, 5.0, 2.5), (6.0, 3.0, 4.5)]
)
def test_arithmetic_mean(xi, xj, expected) -> None:
    assert arithmetic_mean(xi, xj) == expected


def test_dxi_arithmetic_mean() -> None:
    xi = np.linspace(0.0, 1e3, num=30)
    xj = np.linspace(1e2, 1e5, num=30)

    np.testing.assert_allclose(
        dxi_arithmetic_mean(xi, xj),
        np.asarray(
            nd.Derivative(arithmetic_mean, n=1, step=sys.float_info.epsilon * 1e10)(
                xi, xj
            )
        ),
        rtol=0.05,
    )


@pytest.mark.parametrize("xi,xj,expected", [(1e-5, 1e-7, 1.98e-7), (1e6, 1e0, 2.0)])
def test_harmonic_mean(xi, xj, expected) -> None:
    np.testing.assert_allclose(harmonic_mean(xi, xj), expected, rtol=1e-1)


@pytest.mark.parametrize(
    "xi,xj,expected",
    [
        (0.0, 0.0, 0.0),  # both zero
        (0.0, 5.0, 0.0),  # one zero -> limit is 0
        (5.0, 0.0, 0.0),
        (5.0, -5.0, np.nan),  # undefined: xi + xj == 0 with non-zero values
    ],
)
def test_harmonic_mean_zero_division(xi, xj, expected) -> None:
    with np.errstate(all="raise"):  # no numpy warning must be emitted
        res = harmonic_mean(xi, xj)
    np.testing.assert_array_equal(res, expected)


def test_harmonic_mean_zero_division_vectorized() -> None:
    xi = np.array([0.0, 0.0, 2.0, 4.0, 3.0])
    xj = np.array([0.0, 5.0, 0.0, 4.0, -3.0])
    with np.errstate(all="raise"):
        res = harmonic_mean(xi, xj)
    np.testing.assert_array_equal(res, np.array([0.0, 0.0, 0.0, 4.0, np.nan]))


def test_dxi_harmonic_mean() -> None:
    xi = np.power(10, np.linspace(-12, -3, num=20))
    xj = np.power(10, np.linspace(-9, -1, num=20))

    np.testing.assert_allclose(
        dxi_harmonic_mean(xi, xj),
        np.asarray(
            nd.Derivative(harmonic_mean, n=1, step=sys.float_info.epsilon)(xi, xj)
        ),
        rtol=0.05,
    )


@pytest.mark.parametrize(
    "xi,xj,expected",
    [
        (0.0, 0.0, 0.5),  # direction-dependent: symmetric convention
        (0.0, 5.0, 2.0),  # regular formula still valid with xi == 0
        (5.0, 0.0, 0.0),
        (5.0, -5.0, np.nan),  # undefined: xi + xj == 0 with non-zero values
    ],
)
def test_dxi_harmonic_mean_zero_division(xi, xj, expected) -> None:
    with np.errstate(all="raise"):
        res = dxi_harmonic_mean(xi, xj)
    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize(
    "arr, mean_type, weights, expected",
    (
        (np.ones((3, 3, 4)), MeanType.ARITHMETIC, None, np.ones((4))),
        (np.ones((3, 3, 4)), MeanType.GEOMETRIC, None, np.ones((4))),
        (np.ones((3, 3, 4)), MeanType.HARMONIC, None, np.ones((4))),
        (np.ones((1, 3, 5)), MeanType.ARITHMETIC, np.ones((3)), np.ones((5))),
        (np.ones((1, 2, 4)), MeanType.GEOMETRIC, np.ones((2)) * 0.5, np.ones((4))),
        (np.ones((3, 1, 2)), MeanType.HARMONIC, np.ones((3)), np.ones((2))),
        (
            np.repeat(np.array([[1, 2], [3, 4]])[:, :, np.newaxis], 5, axis=-1),
            MeanType.ARITHMETIC,
            np.ones((4)),
            np.ones((5)) * 2.5,
        ),
        (
            np.repeat(np.array([[1, 2], [3, 4]])[:, :, np.newaxis], 4, axis=-1),
            MeanType.GEOMETRIC,
            np.ones((4)),
            np.ones((4)) * 2.213364,
        ),
        (
            np.repeat(np.array([[1, 2], [3, 4]])[:, :, np.newaxis], 3, axis=-1),
            MeanType.HARMONIC,
            np.ones((4)),
            np.ones((3)) * 1.92,
        ),
    ),
)
def test_get_mean_values_for_last_axis(
    arr: NDArrayFloat,
    mean_type: MeanType,
    weights: Optional[NDArrayFloat],
    expected: NDArrayFloat,
) -> None:
    np.testing.assert_allclose(
        get_mean_values_for_last_axis(arr, mean_type, weights), expected
    )


@pytest.mark.parametrize(
    "mean_type, expected",
    (
        (MeanType.ARITHMETIC, np.array([1.5, 2.5, 3.5])),
        (MeanType.GEOMETRIC, np.array([0.0, 2.0, np.sqrt(10.0)])),
        (MeanType.HARMONIC, np.array([0.0, 1.6, 20.0 / 7.0])),
    ),
)
def test_get_mean_values_for_last_axis_with_zero(
    mean_type: MeanType, expected: NDArrayFloat
) -> None:
    # first column contains a zero: geometric and harmonic means are 0
    arr = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    with np.errstate(all="raise"):
        res = get_mean_values_for_last_axis(arr, mean_type)
    np.testing.assert_allclose(res, expected)


def test_get_mean_values_for_last_axis_error() -> None:
    with pytest.raises(
        ValueError, match="The number of weights must match the number of grid cells."
    ):
        get_mean_values_for_last_axis(np.ones((4, 12)), MeanType.ARITHMETIC, np.ones(5))


@pytest.mark.parametrize(
    "mean, mean_gradient",
    ((np.average, amean_gradient), (gmean, gmean_gradient), (hmean, hmean_gradient)),
)
def test_means_gradient(mean, mean_gradient) -> None:
    rng = np.random.default_rng(19680801)
    exponents = rng.integers(1, 10, size=20)
    test_values = np.power(10, -exponents.astype(np.float64))
    weights = rng.random(20)  # between 0 and 1

    fd = nd.Gradient(
        mean, step=np.min(test_values) * 1e-4
    )  # impose the step to avoid being below zero.
    # not optimal to test on a logsclae... so we use atol instead or rtol
    np.testing.assert_allclose(
        np.asarray(fd(test_values)), mean_gradient(test_values), atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(fd(test_values, weights=weights)),
        mean_gradient(test_values, weights),
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "values, weights, expected",
    (
        # exactly one zero, unweighted: gradient is n at the zero entry
        (np.array([0.0, 2.0, 4.0]), None, np.array([3.0, 0.0, 0.0])),
        # exactly one zero, weighted: gradient is sum(w) / w_k at the zero entry
        (
            np.array([0.0, 2.0, 4.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([6.0, 0.0, 0.0]),
        ),
        (
            np.array([2.0, 0.0, 4.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 3.0, 0.0]),
        ),
        # several zeros: gradient is set to 0
        (np.array([0.0, 0.0, 4.0]), None, np.zeros(3)),
        (np.array([0.0, 0.0, 4.0]), np.array([1.0, 2.0, 3.0]), np.zeros(3)),
    ),
)
def test_hmean_gradient_with_zero(values, weights, expected) -> None:
    with np.errstate(all="raise"):
        res = hmean_gradient(values, weights)
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize(
    "values, weights, expected",
    (
        # unweighted, one zero: infinite slope at the zero, null elsewhere
        (np.array([0.0, 2.0, 4.0]), None, np.array([np.inf, 0.0, 0.0])),
        # weighted, one zero carrying only part of the weight
        (
            np.array([2.0, 4.0, 0.0]),
            np.array([1.0, 1.0, 2.0]),
            np.array([0.0, 0.0, np.inf]),
        ),
        # the zero entry carries the whole weight: gradient is 1
        (
            np.array([0.0, 2.0, 4.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ),
        # single value
        (np.array([0.0]), None, np.array([1.0])),
    ),
)
def test_gmean_gradient_with_zero(values, weights, expected) -> None:
    with np.errstate(all="raise"):
        res = gmean_gradient(values, weights)
    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize(
    "arr_shape, mean_type, weights_shape, expected_exception",
    (
        ((3, 3, 4), MeanType.ARITHMETIC, None, does_not_raise()),
        ((3, 3, 4), MeanType.GEOMETRIC, None, does_not_raise()),
        ((3, 3, 4), MeanType.HARMONIC, None, does_not_raise()),
        ((1, 3, 5), MeanType.ARITHMETIC, 3, does_not_raise()),
        ((1, 2, 4), MeanType.GEOMETRIC, 2, does_not_raise()),
        ((3, 1, 2), MeanType.HARMONIC, 3, does_not_raise()),
        ((3, 2), MeanType.HARMONIC, 3, does_not_raise()),
        ((3, 1, 1), MeanType.HARMONIC, 3, does_not_raise()),
        ((3, 1), MeanType.HARMONIC, 3, does_not_raise()),
        ((3), MeanType.HARMONIC, 3, does_not_raise()),
        (
            (3, 1),
            MeanType.HARMONIC,
            2,
            pytest.raises(
                ValueError,
                match="The number of weights must match the number of grid cells.",
            ),
        ),
    ),
)
def test_get_mean_values_gradient_for_last_axis(
    arr_shape: Tuple[int],
    mean_type: MeanType,
    weights_shape: Optional[Tuple[int]],
    expected_exception,
) -> None:
    arr = np.random.default_rng(2023).random(size=arr_shape)
    if weights_shape is not None:
        weights = np.random.default_rng(2023).random(size=weights_shape)
    else:
        weights = None

    with expected_exception:
        res = get_mean_values_gradient_for_last_axis(arr, mean_type, weights)

        # Wrapper to evaluate the mean by timestep
        def wrapper(x: NDArrayFloat) -> NDArrayFloat:
            return get_mean_values_for_last_axis(
                x.reshape(arr.shape), mean_type, weights
            )

        def wrapper2(x: NDArrayFloat) -> NDArrayFloat:
            return get_mean_values_for_last_axis(
                x.reshape(arr.shape), mean_type, weights
            )[0]

        if len(arr.shape) == 3:
            deriv = np.zeros((arr.shape[0] * arr.shape[1], arr.shape[-1]))
        elif len(arr.shape) == 2:
            deriv = np.zeros((arr.shape[0], arr.shape[-1]))
        else:
            deriv = np.zeros((arr.shape[0], 1))

        if arr.shape[-1] != 1 and len(arr.shape) != 1:
            jac = np.asarray(nd.Jacobian(wrapper, step=1e-6)(arr.ravel()))
            for i in range(arr.shape[-1]):
                deriv[:, i] = jac[i][i :: arr.shape[-1]]
        else:
            jac = np.asarray(nd.Gradient(wrapper2, step=1e-6)(arr.ravel())).reshape(
                -1, 1
            )
            deriv = jac

        np.testing.assert_allclose(res, deriv.reshape(arr.shape))
