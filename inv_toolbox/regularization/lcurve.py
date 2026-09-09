# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Provide functions to interpolate an L-curve (data-fit loss vs. regularization
loss, parametrized by the regularization weight) and evaluate its curvature,
so that the regularization weight of maximum curvature ("corner" of the
L-curve) can be identified.

@author: acollet
"""

from typing import Tuple

import numpy as np
import scipy as sp
from inv_toolbox.utils import NDArrayFloat
from numpy.typing import ArrayLike


def _get_curvature(
    interp_loss_ls: NDArrayFloat,
    interp_loss_reg: NDArrayFloat,
    is_logspace: bool = False,
) -> NDArrayFloat:
    """
    Evaluate the discrete curvature of the (interpolated) L-curve.

    The curvature is computed from first- and second-order finite
    differences of `interp_loss_ls` and `interp_loss_reg` (optionally taken
    in log-space). When `is_logspace` is True, the two extreme points on
    each side are excluded from the second-order finite difference (edge
    effects) and instead padded with the closest valid curvature value.

    Parameters
    ----------
    interp_loss_ls : NDArrayFloat
        Interpolated least-squares (data-fit) loss values.
    interp_loss_reg : NDArrayFloat
        Interpolated regularization loss values.
    is_logspace : bool, optional
        Whether the curvature should be computed in log-log space (useful
        when the losses span several orders of magnitude), by default False.

    Returns
    -------
    NDArrayFloat
        Curvature evaluated at each interpolation point, same shape as
        `interp_loss_ls`.
    """
    if is_logspace:
        dx_dt = np.gradient(np.log(interp_loss_ls)[2:-2])
        dy_dt = np.gradient(np.log(interp_loss_reg)[2:-2])
    else:
        dx_dt = np.gradient(interp_loss_ls)
        dy_dt = np.gradient(interp_loss_reg)

    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.zeros_like(interp_loss_ls)
    tmp = (
        np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
        / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
    if is_logspace:
        curvature[2:-2] = tmp
        curvature[:2] = tmp[0]
        curvature[-2:] = tmp[-1]
    else:
        curvature = tmp
    return curvature


def _interpolate_reg_weights(
    reg_weights: ArrayLike, loss_ls_list: ArrayLike, interp_loss_ls: NDArrayFloat
) -> NDArrayFloat:
    """
    Interpolate the regularization weights at the given interpolated LS losses.

    Parameters
    ----------
    reg_weights : ArrayLike
        Regularization weights, in increasing order, matching `loss_ls_list`.
    loss_ls_list : ArrayLike
        Least-squares (data-fit) loss values associated with `reg_weights`.
    interp_loss_ls : NDArrayFloat
        Interpolated least-squares loss values at which the corresponding
        regularization weight is sought.

    Returns
    -------
    NDArrayFloat
        Regularization weights interpolated at `interp_loss_ls`.
    """
    # reg_weights is an increasing sequence
    # but it is not necessarily the case for loss_ls_list
    # so we eliminate non increasing entries
    valid_rw = [reg_weights[0]]
    valid_ls = [loss_ls_list[0]]

    for i in np.arange(len(reg_weights) - 1) + 1:
        if loss_ls_list[i] > valid_ls[-1]:
            valid_rw.append(reg_weights[i])
            valid_ls.append(loss_ls_list[i])

    # since reg_weights and loss_ls_list are both increasing sequence, using linear
    # interpolation give a monotonic interpolation
    return np.interp(interp_loss_ls, np.asarray(valid_ls), np.asarray(valid_rw))


def _interpolate_lcurve(
    reg_weights: ArrayLike,
    loss_ls_list: ArrayLike,
    loss_reg_list: ArrayLike,
    is_logspace: bool = False,
    target_n: int = 500,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """
    Interpolate the L-curve trace to the desired number of points.

    The regularization loss is fit as a smooth power-law function of the
    least-squares loss (in log-log space), then both are resampled on a
    regular (linear or log-spaced) grid of `target_n + 1` points, along with
    the corresponding interpolated regularization weights.

    Parameters
    ----------
    reg_weights : ArrayLike
        List of regularization weights, in increasing order.
    loss_ls_list : ArrayLike
        List of least-squares (data-fit) loss values.
    loss_reg_list : ArrayLike
        List of regularization loss values.
    is_logspace : bool, optional
        Whether to interpolate `interp_loss_ls` on a logarithmic scale, by
        default False.
    target_n : int, optional
        Number of interpolation intervals (the returned arrays have
        `target_n + 1` points), by default 500.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
        A tuple containing the interpolated regularization weights, the
        interpolated least-squares loss, and the interpolated (fitted)
        regularization loss.
    """
    if is_logspace:
        interp_loss_ls: NDArrayFloat = np.logspace(
            np.log10(np.min(loss_ls_list)),
            np.log10(np.max(loss_ls_list)),
            target_n + 1,
            base=10,
        )
    else:
        interp_loss_ls: NDArrayFloat = np.linspace(
            np.min(loss_ls_list),
            np.max(loss_ls_list),
            target_n + 1,
        )

    # sort by increasing loss_ls
    x_sorted, y_sorted, z_sorted = np.array(
        sorted(zip(loss_ls_list, loss_reg_list, reg_weights))
    ).T

    # Transform the response subspace
    def lcurve(x, a, b) -> NDArrayFloat:
        return np.max(np.log(y_sorted)) + a * (x - np.min(np.log(x_sorted))) ** b

    # fit parameters
    popt, err = sp.optimize.curve_fit(
        lcurve,
        np.log(x_sorted),
        np.log(y_sorted),
        p0=(
            -np.abs(np.min(np.log(y_sorted))),
            0.5,
        ),
        bounds=np.array(
            [
                (-np.inf, 0),
                (1e-10, np.inf),
            ]
        ).T,
    )

    # interpolate loss reg
    interp_loss_reg: NDArrayFloat = np.exp(lcurve(np.log(interp_loss_ls), *popt))

    interp_reg_weights: NDArrayFloat = _interpolate_reg_weights(
        reg_weights, loss_ls_list, interp_loss_ls
    )

    return interp_reg_weights, interp_loss_ls, interp_loss_reg


def get_l_curvature(
    reg_weights: ArrayLike,
    loss_ls_list: ArrayLike,
    loss_reg_list: ArrayLike,
    is_logspace: bool = False,
    nb_interp_points: int = 500,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, int]:
    """
    Interpolate and evaluate the L-curve curvature.

    Parameters
    ----------
    reg_weights : ArrayLike
        List of regularization weights in increasing order.
    loss_ls_list : ArrayLike
        List of least square objective function (or equivalent data fit measure).
    loss_reg_list : ArrayLike
        List of regularization objective function.
    is_logspace : bool, optional
        Whether to use logspace for the fit, it depends on data scaling.
        by default False.
    nb_interp_points : int, optional
        Number of interpolation points, by default 500

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, int]
        A tuple containing:
        - the interpolated regularization weights,
        - the interpolated least-squares (data-fit) loss,
        - the interpolated regularization loss,
        - the curvature evaluated at each interpolation point,
        - the index (within the interpolated arrays) of maximum absolute
          curvature, i.e., the corner of the L-curve.
    """

    # step 1) Interpolate the functions and the regularization weights to get
    # smoother curves
    interp_reg_weights, interp_loss_ls, interp_loss_reg = _interpolate_lcurve(
        reg_weights, loss_ls_list, loss_reg_list, is_logspace, nb_interp_points
    )

    # evaluate the curvature from the smooth interpolations
    curvature = _get_curvature(interp_loss_ls, interp_loss_reg, is_logspace)

    return (
        interp_reg_weights,
        interp_loss_ls,
        interp_loss_reg,
        curvature,
        int(np.argmax(np.abs(curvature))),
    )
