# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats._stats_py import _validate_distribution

from inv_toolbox.regularization.base import Regularizator
from inv_toolbox.utils import NDArrayFloat, NDArrayInt
from inv_toolbox.utils.preconditioner import NoTransform, Preconditioner


def ffill(arr: NDArray) -> NDArray:
    """
    Forward fill the NaN values in the given array.

    Each NaN entry is replaced by the closest preceding non-NaN value. If the
    array starts with one or several NaN values, they are left unchanged
    (there is nothing to forward-fill them from).

    A new array is returned; `arr` is not modified in place.

    Parameters
    ----------
    arr : NDArray
        1D array, possibly containing NaN values.

    Returns
    -------
    NDArray
        A new array of the same shape as `arr`, with NaN values forward
        filled.
    """
    prev = np.arange(np.size(arr))
    prev[np.isnan(arr)] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


def make_dist_values_unique(
    dist: ArrayLike, weights: Optional[ArrayLike] = None
) -> Tuple[ArrayLike, Union[NDArrayFloat, NDArrayInt], NDArrayInt, NDArrayInt]:
    """
    Make the values in the given distribution unique.

    Weights are summed.

    Parameters
    ----------
    dist : ArrayLike
        Values of the (empirical) distribution.
    weights : Optional[ArrayLike], optional
        Weight associated with each value in `dist`. Must have the same
        length as `dist` if provided. If None, each value is implicitly
        given a weight of 1 (so the aggregated weight of a unique value is
        simply its number of occurrences). By default None.

    Returns
    -------
    Tuple[ArrayLike, Union[NDArrayFloat, NDArrayInt], NDArrayInt, NDArrayInt]
        Arrays containing:
        - The unique values by increasing order.
        - The associated (aggregated) weights. If no weights were provided, it is
        simply the count for each unique value.
        - The indices of first occurrence in the original array.
        - The sorter (indices) by increasing order in the original array (`dist`).
    """
    sorter = np.argsort(dist)
    vals, _indices, _counts = np.unique(
        np.asarray(dist)[sorter], return_counts=True, return_index=True
    )

    if weights is None:
        _weights = _counts
    else:
        idx = np.zeros_like(dist)
        idx[:] = np.nan
        idx[_indices] = np.arange(np.size(vals))
        idx = np.asarray(ffill(idx)[np.argsort(sorter)], dtype=np.int64)
        _weights = np.bincount(idx, weights)  # ty: ignore[no-matching-overload]
    return vals, _weights, _indices, sorter


def get_cdfs(
    u_values: ArrayLike,
    v_values: ArrayLike,
    u_weights: Optional[ArrayLike] = None,
    v_weights: Optional[ArrayLike] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """
    Return the merged cumulative density functions and associated weights.

    The values of `u_values` and `v_values` are merged and sorted, and the
    (weighted) empirical CDFs of `u` and `v` are evaluated at each of the
    resulting breakpoints (excluding the last one).

    Parameters
    ----------
    u_values, v_values : ArrayLike
        Values observed in the (empirical) distributions `u` and `v`.
    u_weights, v_weights : Optional[ArrayLike], optional
        Weight for each value of `u_values` (resp. `v_values`). If
        unspecified, each value is assigned the same weight. By default None.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]
        A tuple containing:
        - `u_cdf`: the empirical CDF of `u` evaluated at the merged
          breakpoints.
        - `v_cdf`: the empirical CDF of `v` evaluated at the merged
          breakpoints.
        - `deltas`: the spacing between consecutive merged/sorted values.
        - `all_sorter`: the indices that sort the concatenation of
          `u_values` and `v_values`.
    """
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_sorter = np.argsort(all_values, kind="mergesort")
    all_values = all_values[all_sorter]

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], "right")
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], "right")

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0], np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0], np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    return u_cdf, v_cdf, deltas, all_sorter


def cdf_distance(
    p: float,
    u_values: ArrayLike,
    v_values: ArrayLike,
    u_weights: Optional[ArrayLike] = None,
    v_weights: Optional[ArrayLike] = None,
) -> float:
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    p : float
        Positive parameter of the :math:`l_p` distance; p = 1 gives the
        Wasserstein distance, p = 2 gives the energy distance.
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
    # First make values in the distribution unique to reduce the complexity
    # of the algorithm
    new_u, new_u_weights = make_dist_values_unique(u_values, u_weights)[:2]
    new_v, new_v_weights = make_dist_values_unique(v_values, v_weights)[:2]

    u_cdf, v_cdf, deltas, _ = get_cdfs(new_u, new_v, new_u_weights, new_v_weights)

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    return np.power(
        np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p), deltas)), 1 / p
    )


def cdf_distance_gradient(
    p: float,
    u_values: ArrayLike,
    v_values: ArrayLike,
    u_weights: Optional[ArrayLike] = None,
    v_weights: Optional[ArrayLike] = None,
    dtype: np.typing.DTypeLike = np.float64,
) -> NDArrayFloat:
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    gradient with respect to :math:`u` of the statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    p : float
        Positive parameter of the :math:`l_p` distance; p = 1 gives the
        Wasserstein distance, p = 2 gives the energy distance.
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance gradient : NDArrayFloat
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """

    # First make values in the distribution unique to reduce the complexity
    # And allow the derivation with respect to repeated values (in u)
    new_u, new_u_weights, old_u_indices, u_sorter = make_dist_values_unique(
        u_values, u_weights
    )
    new_v_values, new_v_weights, _, _ = make_dist_values_unique(v_values, v_weights)

    u_cdf, v_cdf, deltas, all_sorter = get_cdfs(
        new_u, new_v_values, new_u_weights, new_v_weights
    )

    # Note about the derivation => the derivative of u_cdf with respect to u_values
    # is always null (the order is preserved)
    # So the only term that matters in the derivation is the one with resepct to deltas
    if p == 1:
        _temp: NDArrayFloat = np.abs(u_cdf - v_cdf)
    elif p == 2:
        _temp = (
            0.5
            / np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
            * np.square(u_cdf - v_cdf)
        )
    else:
        _temp = (
            1
            / p
            * np.power(
                np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p), deltas)),
                1 / p - 1,
            )
            * np.power(np.abs(u_cdf - v_cdf), p)
        )

    # First approach = more expensive because one needs to build J,
    # the derivatives of deltas with respect to the sorted u_values
    # it is a sparse matrix filled by 1 and -1
    # n = u_cdf_indices.size
    # J = sp.sparse.diags_array([-np.ones(n + 1), np.ones(n)], offsets=[0, 1]).tocsc()[
    #     :-1, np.argsort(all_sorter)
    # ][:, :np.size(u_values)]
    # conversion to csc for faster multiplication
    # return J.tocsc().T @ _temp

    # Second approach strictly equivalent but faster
    grad = (
        -np.concatenate([_temp, [0.0]], dtype=dtype)[np.argsort(all_sorter)][
            : np.size(new_u)
        ]
        + np.concatenate([[0.0], _temp], dtype=dtype)[np.argsort(all_sorter)][
            : np.size(new_u)
        ]
    )

    # This is to handle duplictaed values
    out = np.zeros_like(u_values)
    out[:] = np.nan
    out[old_u_indices] = grad / new_u_weights
    out = ffill(out)[np.argsort(u_sorter)]
    if u_weights is not None:
        return out * np.array(u_weights, dtype=dtype)
    else:
        return out


def _get_subsel(sub_selection: Optional[NDArrayInt]) -> Union[NDArrayInt, slice]:
    """
    Return `sub_selection` if provided, otherwise a full slice.

    Small helper so that indexing code can uniformly write
    ``values[_get_subsel(sub_selection)]`` regardless of whether a sub
    selection was provided.

    Parameters
    ----------
    sub_selection : Optional[NDArrayInt]
        Optional array of indices, or None.

    Returns
    -------
    Union[NDArrayInt, slice]
        `sub_selection` if not None, otherwise `slice(None)` (selects
        everything).
    """
    if sub_selection is not None:
        return sub_selection
    return slice(None)


@dataclass
class ProbDistFitting(Regularizator):
    r"""
    Apply an (empirical) probability distribution fitting regularization.

    The fit is based on the Wasserstein distance.

    Attributes
    ----------
    target_values: ArrayLike
        Values observed in the (empirical) distribution.
    target_weights: Optional[ArrayLike]
        Weight for each target value. If unspecified, each value is assigned the same
        weight.
        `target_weights` must have the same length as `target_values`.
        If the weight sum differs from 1, it must still be positive and finite so that
        the weights can be normalized to sum to 1. The default is None.
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    order: float
        Positive parameter; p = 1 gives the Wasserstein distance, p = 2
        gives the energy distance. The default is one.

    """

    target_values: ArrayLike
    target_weights: Optional[ArrayLike] = None
    sub_selection: Optional[NDArrayInt] = None
    preconditioner: Preconditioner = NoTransform()
    order: float = 1.0

    def _eval_loss(self, values: NDArrayFloat) -> float:
        r"""
        Compute the probability distribution fitting regularization loss.

        The loss is the :math:`l_p` (CDF-based) statistical distance between
        the empirical distribution of `values` and the target distribution:

        .. math::

        \mathcal{R}_{W}(u) = l_p(u, v) = \left( \int_{-\infty}^{+\infty}
        |U(x) - V(x)|^{p} \, dx \right)^{1/p}

        where :math:`U` and :math:`V` are respectively the empirical CDFs of
        `values` (restricted to `sub_selection` if provided) and of the
        target distribution.

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        float
            The regularization loss value.
        """

        return cdf_distance(
            p=self.order,
            u_values=values[_get_subsel(self.sub_selection)].ravel("F"),
            v_values=self.target_values,
            u_weights=None,
            v_weights=self.target_weights,
        )

    def _eval_loss_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
        r"""
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        out = np.zeros_like(values)
        out[_get_subsel(self.sub_selection)] = cdf_distance_gradient(
            p=self.order,
            u_values=values[_get_subsel(self.sub_selection)].ravel("F"),
            v_values=self.target_values,
            u_weights=None,
            v_weights=self.target_weights,
        )
        return out.reshape(values.shape, order="F")
