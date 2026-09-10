# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Provide classes and functions for geostatistical regularization.

TODO: add the formulas.
"""

import covmats
import numpy as np

from inv_toolbox.regularization.base import Regularizator
from inv_toolbox.utils import NDArrayFloat
from inv_toolbox.utils.preconditioner import NoTransform, Preconditioner


def identify_function(x: NDArrayFloat) -> NDArrayFloat:
    """Return x untransformed (f(x) = x)."""
    return x


def one(x: NDArrayFloat) -> NDArrayFloat:
    """Return 1.0, whatever the input."""
    return np.ones(x.shape)


class GeostatisticalRegularizator(Regularizator):
    """
    Implement a regularization based on the parameter covariance matrix.

    Attributes
    ----------
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    __slots__ = ["cov_m", "prior"]

    def __init__(
        self,
        cov_m: covmats.CovarianceMatrix,
        prior: covmats.PriorTerm = covmats.NullPriorTerm(),
        preconditioner: Preconditioner = NoTransform(),
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        cov_m : CovarianceMatrix
            The parameter covariance matrix used to weight the residuals
            (its `solve` method is used to apply :math:`Q^{-1}`).
        prior : Optional[covmats.PriorTerm], optional
            A prior term for `(x - prior_term)`, by default NullPriorTerm.
        preconditioner: Preconditioner
            Parameter pre-transformation operator (variable change for the solver).
            The default is the identity function: f(x) = x, which means no change
            is made.
        """
        super().__init__(preconditioner)
        self.cov_m: covmats.CovarianceMatrix = cov_m
        self.prior: covmats.PriorTerm = prior

    def _eval_loss(self, values: NDArrayFloat) -> float:
        r"""
        Compute the geostatistical regularization loss function.

        .. math::

        \mathcal{R}_{Q}(u) = \frac{1}{2} \left(s-Xb\right)^TQ^{-1}\left(s-Xb\right)

        Parameters
        ----------
        values : NDArrayFloat
            Values of the parameter for which the regularization is computed.
            Always a 1D vector (enforced by :meth:`Regularizator.eval_loss`).

        Returns
        -------
        float
            The regularization loss value.
        """
        _values = values
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        return float(
            0.5
            * np.dot(
                residuals.T,
                self.cov_m.solve(residuals),
            ).item()
        )

    def _eval_loss_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
        """
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        values : NDArrayFloat
            Values of the parameter for which the regularization is computed.
            Always a 1D vector (enforced by :meth:`Regularizator.eval_loss_gradient`).

        Returns
        -------
        NDArrayFloat
            The regularization gradient (1D vector).
        """
        _values = values
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        # right part $Q^{-1} * (m - m_{prior})$
        _right_part = self.cov_m.solve(residuals).ravel()
        # left part gradient -> special method to get more efficient
        # $ [I - dm_{prior}/dm]^{T} Q^{-1} (m - m_{prior})$
        return _right_part - self.prior.get_gradient_dot_product(_right_part)


# def compute_best_beta(
#     values: NDArrayFloat, cov_m: CovarianceMatrix, drift_matrix: DriftMatrix
# ) -> NDArrayFloat:
#     """
#     Compute the optimal beta (minimal objective function).

#     TODO: Add the maths here.

#     Parameters
#     ----------
#     values : NDArrayFloat
#         Values of the parameter for which the regularization is computed.
#         Should be 2D array / 1d vector.
#     cov_m : CovarianceMatrix
#         The covariance matrix.
#     drift_matrix : DriftMatrix
#         The drift matrix instance for which to compute beta.

#     Returns
#     -------
#     NDArrayFloat
#         The best beta.
#     """
#     # This is valid for the linear one only.
#     invQs = cov_m.solve(values)
#     invQX = cov_m.solve(drift_matrix.mat)

#     XTinvQs = np.dot(drift_matrix.mat.T, invQs)
#     XTinvQX = np.dot(drift_matrix.mat.T, invQX)

#     # inexpensive solve p by p where p <= 3, usually p = 1 (scalar division)
#     return np.linalg.solve(np.atleast_2d(XTinvQX), XTinvQs).ravel()
