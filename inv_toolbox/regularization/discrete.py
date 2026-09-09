# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Implement a discrete regularizator.

@author: acollet
"""

from typing import List, Literal

import numpy as np
from inv_toolbox.regularization.base import Regularizator
from inv_toolbox.utils import NDArrayFloat
from inv_toolbox.utils.preconditioner import NoTransform, Preconditioner


def get_closest_mode(x: NDArrayFloat, modes: NDArrayFloat) -> NDArrayFloat:
    """
    Return, for each entry of `x`, the closest value among `modes`.

    Parameters
    ----------
    x : NDArrayFloat
        Values for which the closest mode is sought.
    modes : NDArrayFloat
        Sorted (increasing order) array of discrete modes.

    Returns
    -------
    NDArrayFloat
        Array with the same shape as `x`, containing the closest mode value
        for each entry.
    """
    return modes[np.digitize(x, modes[:-1] + np.diff(modes) / 2.0)]


def min_squared_distance(x: NDArrayFloat, modes: List[float]) -> NDArrayFloat:
    """
    Return the squared distance between `x` and its closest mode.

    Parameters
    ----------
    x : NDArrayFloat
        Values for which the distance is computed.
    modes : List[float]
        Sorted (increasing order) list of discrete modes.

    Returns
    -------
    NDArrayFloat
        Squared distance to the closest mode, same shape as `x`.
    """
    return (x - get_closest_mode(x, np.asarray(modes))) ** 2


def dmin_squared_distance(x: NDArrayFloat, modes: List[float]) -> NDArrayFloat:
    """
    Return the derivative w.r.t. `x` of :func:`min_squared_distance`.

    Parameters
    ----------
    x : NDArrayFloat
        Values for which the derivative is computed.
    modes : List[float]
        Sorted (increasing order) list of discrete modes.

    Returns
    -------
    NDArrayFloat
        Derivative of the squared distance to the closest mode, same shape
        as `x`.
    """
    return 2 * (x - get_closest_mode(x, np.asarray(modes)))


def scaled_gaussian_pdf(x: NDArrayFloat, std: float, mean: float) -> NDArrayFloat:
    """
    Return an unnormalized Gaussian bump centered on `mean` (peak value 1).

    Parameters
    ----------
    x : NDArrayFloat
        Values at which the Gaussian is evaluated.
    std : float
        Standard deviation of the Gaussian. Must be strictly positive.
    mean : float
        Mean (center) of the Gaussian.

    Returns
    -------
    NDArrayFloat
        Values of the (unnormalized) Gaussian, same shape as `x`.
    """
    return np.exp(-((x - mean) ** 2) / (2 * std**2))


def dscaled_gaussian_pdf(x: NDArrayFloat, std: float, mean: float) -> NDArrayFloat:
    """
    Return the derivative w.r.t. `x` of :func:`scaled_gaussian_pdf`.

    Parameters
    ----------
    x : NDArrayFloat
        Values at which the derivative is evaluated.
    std : float
        Standard deviation of the Gaussian. Must be strictly positive.
    mean : float
        Mean (center) of the Gaussian.

    Returns
    -------
    NDArrayFloat
        Derivative values, same shape as `x`.
    """
    return -(x - mean) / (std**2) * scaled_gaussian_pdf(x, std, mean)


def gaussian_distance_from_modes(x: NDArrayFloat, modes: List[float]) -> NDArrayFloat:
    """
    Return a smooth (Gaussian-based) distance of `x` to the set of modes.

    The distance is 1.0 minus the sum, over all modes, of a Gaussian bump
    centered on that mode. It is close to 0 when `x` is close to one of the
    modes, and close to 1 otherwise. The standard deviation of each Gaussian
    is derived from the smallest spacing between consecutive (sorted) modes.

    Parameters
    ----------
    x : NDArrayFloat
        Values for which the distance is computed.
    modes : List[float]
        List of discrete modes (at least two distinct values).

    Returns
    -------
    NDArrayFloat
        Distance values, same shape as `x`.

    Raises
    ------
    ZeroDivisionError
        Implicitly, if `modes` contains duplicated values (the minimum
        spacing would be zero).
    """
    std = np.min(np.diff(sorted(modes))) / 6.0
    return 1.0 - np.sum([scaled_gaussian_pdf(x, std, mode) for mode in modes], axis=0)


def dgaussian_distance_from_modes(x: NDArrayFloat, modes: List[float]) -> NDArrayFloat:
    """
    Return the derivative w.r.t. `x` of :func:`gaussian_distance_from_modes`.

    Parameters
    ----------
    x : NDArrayFloat
        Values for which the derivative is computed.
    modes : List[float]
        List of discrete modes (at least two distinct values).

    Returns
    -------
    NDArrayFloat
        Derivative values, same shape as `x`.
    """
    std = np.min(np.diff(sorted(modes))) / 6.0
    return -np.sum([dscaled_gaussian_pdf(x, std, mode) for mode in modes], axis=0)


class DiscreteRegularizator(Regularizator):
    r"""
    Apply a discrete (values takes specific discrete values) regularization.

    Attributes
    ----------
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.

    """

    def __init__(
        self,
        modes: List[float],
        penalty: Literal["least-squares", "gaussian"] = "least-squares",
        preconditioner: Preconditioner = NoTransform(),
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        modes : List[float]
            List of modes (discrete values that the field should take).
        penalty : Literal["least-squares", "gaussian"], optional
            Penalty used. If \"least-squares\", then the sum of squared distances to
            closest modes is considered. If \"gaussian\", then the sum of distances
            computed as a gaussian is used. By default "least-squares".
        preconditioner: Preconditioner
            Parameter pre-transformation operator (variable change for the solver).
            The default is the identity function: f(x) = x, which means no change
            is made.

        Raises
        ------
        ValueError
            If less than two modes are provided.
        """
        super().__init__(preconditioner)
        if len(modes) < 2:
            raise ValueError("At least two modes must be provided!")

        self.modes: List[float] = sorted(self.preconditioner(np.asarray(modes)))
        if len(set(self.modes)) != len(self.modes):
            raise ValueError(
                "The provided modes must be distinct! Duplicated modes would "
                "result in a null spacing and, for the 'gaussian' penalty, in "
                "a division by zero."
            )
        self.penalty = penalty

    @property
    def penalty(self) -> str:
        return self._penalty

    @penalty.setter
    def penalty(self, value: Literal["least-squares", "gaussian"]) -> None:
        if value not in ["least-squares", "gaussian"]:
            raise ValueError('penalty should be among ["least-squares", "gaussian"]')
        self._penalty = value

    def _eval_loss(self, param: NDArrayFloat) -> float:
        r"""
        Compute the discrete regularization loss function.

        For the "least-squares" penalty:

        .. math::

        \mathcal{R}_{D}(u) = \sum_{i=1}^{N} \min_{k} \left(u_{i} - c_{k}\right)^{2}

        For the "gaussian" penalty:

        .. math::

        \mathcal{R}_{D}(u) = \sum_{i=1}^{N} \left(1 - \sum_{k} \exp\left(
        -\dfrac{(u_{i} - c_{k})^{2}}{2 \sigma^{2}} \right) \right)

        where :math:`c_{k}` denotes the k-th mode and :math:`\sigma` is derived
        from the smallest spacing between consecutive (sorted) modes.

        Parameters
        ----------
        param : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        float
            The regularization loss value.
        """
        if self.penalty == "least-squares":
            return float(np.sum(min_squared_distance(param, self.modes)))
        return float(np.sum(gaussian_distance_from_modes(param, self.modes)))

    def _eval_loss_gradient_analytical(self, param: NDArrayFloat) -> NDArrayFloat:
        """
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        param : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        if self.penalty == "least-squares":
            return dmin_squared_distance(param, self.modes)
        # Gaussian
        return dgaussian_distance_from_modes(param, self.modes)  # Gaussian
