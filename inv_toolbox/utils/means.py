# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide some classic means."""

# pylint: disable=C0103 # doesn't conform to snake_case naming style
from typing import Optional

import numpy as np
from scipy.stats import gmean, hmean

from inv_toolbox.utils.enum import StrEnum
from inv_toolbox.utils.types import NDArrayFloat


def arithmetic_mean(xi: NDArrayFloat, xj: NDArrayFloat) -> NDArrayFloat:
    """
    Return the arithmetic mean of xi and xj.

    Parameters
    ----------
    xi : NDArrayFloat
        First array of values.
    xj : NDArrayFloat
        Second array of values.

    Returns
    -------
    NDArrayFloat
        The arithmetic mean of `xi` and `xj`.
    """
    return (xi + xj) / 2.0


def dxi_arithmetic_mean(xi: NDArrayFloat, xj: NDArrayFloat) -> NDArrayFloat:
    """
    Return the first derivative of xi and xj arithmetic mean with respect to xi.

    Parameters
    ----------
    xi : NDArrayFloat
        First array of values (used only to broadcast the output to the
        correct shape; the derivative itself does not depend on `xi`).
    xj : NDArrayFloat
        Second array of values (unused; kept for a signature consistent
        with :func:`arithmetic_mean`).

    Returns
    -------
    NDArrayFloat
        The derivative, i.e. 0.5 everywhere, broadcast to the shape of `xi`.
    """
    # pylint: disable=W0613 # unused argument
    return 0.5 + xi * 0.0  # required to work with vectors


def harmonic_mean(xi: NDArrayFloat, xj: NDArrayFloat) -> NDArrayFloat:
    """
    Return the harmonic mean of xi and xj.

    Computed as ``2 * xi * xj / (xi + xj)`` so that zeros are handled
    without any division by zero: if `xi` or `xj` is zero, the result is 0
    (the limit of the harmonic mean for non-negative values).

    Parameters
    ----------
    xi : NDArrayFloat
        First array of values.
    xj : NDArrayFloat
        Second array of values.

    Returns
    -------
    NDArrayFloat
        The harmonic mean of `xi` and `xj`. Entries where ``xi + xj == 0``
        are 0 if both values are zero, NaN otherwise (opposite-sign values,
        for which the mean is undefined).
    """
    xi, xj = np.broadcast_arrays(
        np.asarray(xi, dtype=float), np.asarray(xj, dtype=float)
    )
    den = xi + xj
    out = np.full(den.shape, np.nan)
    np.divide(2.0 * xi * xj, den, out=out, where=den != 0.0)
    return np.where((xi == 0.0) & (xj == 0.0), 0.0, out)


def dxi_harmonic_mean(xi: NDArrayFloat, xj: NDArrayFloat) -> NDArrayFloat:
    """
    Return the first derivative of xi and xj harmonic mean with respect to xi.

    Parameters
    ----------
    xi : NDArrayFloat
        First array of values.
    xj : NDArrayFloat
        Second array of values.

    Returns
    -------
    NDArrayFloat
        The derivative of :func:`harmonic_mean` w.r.t. `xi`. Where both
        values are zero the derivative is direction-dependent; the symmetric
        value 0.5 is returned by convention. Where ``xi + xj == 0`` with
        non-zero values, the derivative is undefined and NaN is returned.
    """
    xi, xj = np.broadcast_arrays(
        np.asarray(xi, dtype=float), np.asarray(xj, dtype=float)
    )
    den = (xi + xj) ** 2.0
    out = np.full(den.shape, np.nan)
    np.divide(2.0 * xj**2.0, den, out=out, where=den != 0.0)
    return np.where((xi == 0.0) & (xj == 0.0), 0.5, out)


class MeanType(StrEnum):
    HARMONIC = "harmonic"
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


def _reduce_1d(
    values_1d: NDArrayFloat, mean_type: MeanType, weights: Optional[NDArrayFloat]
) -> float:
    """
    Reduce a 1D array of values to a scalar mean, dispatching on `mean_type`.

    Small helper isolating the dispatch used by :func:`get_mean_values_for_last_axis`
    so that :func:`numpy.apply_along_axis` is given a single, uniformly-typed
    callable (``np.average``, `gmean` and `hmean` do not share an identical
    signature, which a dict-based dispatch would otherwise expose to
    `apply_along_axis` as an unmatchable union of callables).

    Harmonic and geometric means of non-negative values containing a zero are
    0 (the limit), returned explicitly to avoid division by zero / log(0)
    warnings.

    Parameters
    ----------
    values_1d : NDArrayFloat
        1D array of values to average.
    mean_type : MeanType
        Type of mean to compute.
    weights : Optional[NDArrayFloat]
        Weights to apply.

    Returns
    -------
    float
        The mean of `values_1d`.
    """
    if mean_type == MeanType.ARITHMETIC:
        return float(np.average(values_1d, weights=weights))
    if np.any(values_1d == 0.0):
        return 0.0
    if mean_type == MeanType.GEOMETRIC:
        return float(gmean(values_1d, weights=weights))
    return float(hmean(values_1d, weights=weights))


def get_mean_values_for_last_axis(
    arr: NDArrayFloat, mean_type: MeanType, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """
    Get the mean values for the last axis of the input array.

    Parameters
    ----------
    arr : NDArrayFloat
        Array of values with shape (nx, ny, nt) or (npts, nt).
    mean_type: MeanType
        Type of mean chosen to average the simulated value when the observed one
        is defined over several grid cells of the domain.
    weights: Optional[NDArrayFloat]
        Weights to apply
    Returns
    -------
    NDArrayFloat
        Averaged values for the last axis.

    """
    _arr = np.asarray(arr)
    # ensure a second axis
    if len(_arr.shape) == 1:
        _arr = _arr[:, np.newaxis]
    # or make 2D
    else:
        _arr = _arr.reshape(-1, _arr.shape[-1])
    if weights is not None and _arr.shape[0] != weights.size:
        raise ValueError("The number of weights must match the number of grid cells.")

    return np.apply_along_axis(
        _reduce_1d,
        axis=0,
        arr=_arr,
        mean_type=mean_type,
        weights=weights,
    )


def amean_gradient(
    values: NDArrayFloat, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """
    Return the gradient of the (optionally weighted) arithmetic mean.

    Parameters
    ----------
    values : NDArrayFloat
        Values the arithmetic mean is computed from.
    weights : Optional[NDArrayFloat]
        Weights associated with `values`. If None, all values are equally
        weighted. The default is None.

    Returns
    -------
    NDArrayFloat
        Gradient of the arithmetic mean with respect to each entry of
        `values`, with the same shape as `values`.
    """
    if weights is not None:
        return weights / np.sum(weights)
    return np.ones(values.shape) / values.size


def hmean_gradient(
    values: NDArrayFloat, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """
    Return the gradient of the (optionally weighted) harmonic mean.

    Zeros are handled through the limit of the (non-negative) harmonic mean:

    - no zero: the usual closed-form gradient;
    - exactly one zero at index k: the gradient is ``n`` (``sum(w) / w_k``
      if weighted) at k and 0 elsewhere;
    - several zeros: the mean is identically 0 in a neighbourhood along the
      non-zero entries and the gradient is set to 0.

    Parameters
    ----------
    values : NDArrayFloat
        Values the harmonic mean is computed from.
    weights : Optional[NDArrayFloat]
        Weights associated with `values`. If None, all values are equally
        weighted. The default is None.

    Returns
    -------
    NDArrayFloat
        Gradient of the harmonic mean with respect to each entry of
        `values`, with the same shape as `values`.
    """
    zeros = values == 0.0
    n_zeros = int(np.count_nonzero(zeros))
    if n_zeros > 0:
        grad = np.zeros(values.shape, dtype=float)
        if n_zeros == 1:
            if weights is None:
                grad[zeros] = values.size
            else:
                grad[zeros] = np.sum(weights) / weights[zeros]
        return grad

    if weights is None:
        return values.size / (np.square(values * np.sum(1.0 / values)))

    return weights * np.sum(weights) / np.square(values * np.sum(weights / values))


def gmean_gradient(
    values: NDArrayFloat, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """
    Return the gradient of the (optionally weighted) geometric mean.

    If `values` contains zeros, the geometric mean is 0 and its gradient is
    0 for the non-zero entries. For the zero entries the gradient is infinite
    (the mean behaves like a power of exponent < 1), except when that entry
    carries the whole weight (e.g. a single value), where it is 1.

    Parameters
    ----------
    values : NDArrayFloat
        Values the geometric mean is computed from. Must be non-negative.
    weights : Optional[NDArrayFloat]
        Weights associated with `values`. If None, all values are equally
        weighted. The default is None.

    Returns
    -------
    NDArrayFloat
        Gradient of the geometric mean with respect to each entry of
        `values`, with the same shape as `values`.
    """
    k: int = values.size
    zeros = values == 0.0
    if np.any(zeros):
        share = (
            np.full(values.shape, 1.0 / k)
            if weights is None
            else weights / np.sum(weights)
        )
        grad = np.zeros(values.shape, dtype=float)
        grad[zeros] = np.where(share[zeros] >= 1.0, 1.0, np.inf)
        return grad

    if weights is None:
        return 1 / k * np.power(np.prod(values), (1 / k)) / values
    return weights / (values * np.sum(weights)) * gmean(values, weights=weights)


def _reduce_gradient_1d(
    values_1d: NDArrayFloat, mean_type: MeanType, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """
    Reduce a 1D array of values to its mean gradient, dispatching on `mean_type`.

    Small helper isolating the dispatch used by
    :func:`get_mean_values_gradient_for_last_axis`, for the same reason as
    :func:`_reduce_1d`.

    Parameters
    ----------
    values_1d : NDArrayFloat
        1D array of values the mean gradient is computed from.
    mean_type : MeanType
        Type of mean to differentiate.
    weights : Optional[NDArrayFloat]
        Weights associated with `values_1d`.

    Returns
    -------
    NDArrayFloat
        The gradient of the mean of `values_1d`, with the same shape as
        `values_1d`.
    """
    if mean_type == MeanType.ARITHMETIC:
        return amean_gradient(values_1d, weights=weights)
    if mean_type == MeanType.GEOMETRIC:
        return gmean_gradient(values_1d, weights=weights)
    return hmean_gradient(values_1d, weights=weights)


def get_mean_values_gradient_for_last_axis(
    arr: NDArrayFloat, mean_type: MeanType, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """
    Get the mean values for the last axis of the input array.

    Parameters
    ----------
    arr : NDArrayFloat
        Array of values with shape (nx, ny, nt) or (npts, nt).
    mean_type: MeanType
        Type of mean chosen to average the simulated value when the observed one
        is defined over several grid cells of the domain.
    weights: Optional[NDArrayFloat]
        Weights to apply
    Returns
    -------
    NDArrayFloat
        Averaged values for the last axis.

    """
    # ensure a second axis
    if len(arr.shape) == 1:
        _arr = arr[:, np.newaxis]
    # or make 2D
    else:
        _arr = arr.reshape(-1, arr.shape[-1])
    if weights is not None and _arr.shape[0] != weights.size:
        raise ValueError("The number of weights must match the number of grid cells.")

    return np.apply_along_axis(
        _reduce_gradient_1d,
        axis=0,
        arr=_arr,
        mean_type=mean_type,
        weights=weights,
    ).reshape(arr.shape)
