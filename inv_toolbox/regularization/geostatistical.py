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
from inv_toolbox.utils.finite_differences import finite_gradient
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


class EnsembleRegularizator(GeostatisticalRegularizator):
    """
    Implement a regularization based on an ensemble.

    Compared to a classic regularization, an ensemble is passed to the functions.

    TODO: here add the objective function of the regularization terM.

    Attributes
    ----------
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    def eval_loss(self, values: NDArrayFloat) -> float:
        r"""
        Compute the ensemble-averaged geostatistical regularization loss.

        .. math::

        \mathcal{R}_{Q}(u) = \frac{1}{2 N_{e}} \sum_{e=1}^{N_{e}}
        \left(s_{e}-Xb\right)^TQ^{-1}\left(s_{e}-Xb\right)

        where :math:`N_{e}` is the number of ensemble members and
        :math:`s_{e}` the values of the e-th member.

        Parameters
        ----------
        values : NDArrayFloat
            Ensemble of shape (N_s, Ne). N_s being the number of optimized
            values, Ne, the number of members in the ensemble.

        Returns
        -------
        float
            The regularization loss value, averaged over the ensemble.
        """
        if not values.ndim == 2:
            raise ValueError(
                "The 'EnsembleRegularizator.eval_loss' method expects a 2D vector!"
            )

        _values = values
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        # residuals = - self.prior.get_values(_values)

        # And this is strictly equivalent (element wise multiplication)
        return (
            0.5
            * float(np.sum(residuals * self.cov_m.solve(residuals)))
            / values.shape[1]
        )

    def eval_loss_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
        """
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        values : NDArrayFloat
            Ensemble of shape (N_s, Ne). N_s being the number of optimized values,
            Ne, the number of members in the ensemble.

        Returns
        -------
        NDArrayFloat
            The regularization gradient (2d).
        """
        if not values.ndim == 2:
            raise ValueError(
                "The 'EnsembleRegularizator.eval_loss_gradient_analytical' "
                "method expects a 2D vector!"
            )
        _values = values
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        # residuals = _values * 0.0 - self.prior.get_values(_values)

        # right part $Q^{-1} * (m - m_{prior})$
        _right_part = self.cov_m.solve(residuals)

        # We should have the same shape
        if _right_part.shape != values.shape:
            raise ValueError(
                "The covariance solve did not return an array with the same "
                f"shape as the input ensemble: got {_right_part.shape}, "
                f"expected {values.shape}."
            )

        # TODO: here we considered than the derivative of the covariance matrix w.r.t.
        # the parameters is null, but that is not necessary the case all the time.

        # left part gradient -> special method to get more efficient
        # $ [I - dm_{prior}/dm]^{T} Q^{-1} (m - m_{prior})$
        # Note: dot product must be distributed
        # The mean operator comes from the fact that a member j is involved
        # in all derivations of the mean operator.
        return (
            _right_part
            - np.mean(
                self.prior.get_gradient_dot_product(_right_part),
                axis=1,
                keepdims=True,
            )
        ) / values.shape[1]

    def eval_loss_gradient(
        self,
        values: NDArrayFloat,
        is_finite_differences: bool = False,
        max_workers: int = 1,
    ) -> NDArrayFloat:
        """
        Compute the gradient of the regularization loss function.

        Parameters
        ----------
        values : NDArrayFloat
            Ensemble of shape (N_s, Ne). N_s being the number of optimized
            values, Ne, the number of members in the ensemble.
        is_finite_differences: bool
            If true, a numerical approximation by 2nd order finite difference is
            returned. Cost twice the `values` dimensions in terms of loss function
            calls. The default is False.
        max_workers: int
            Number of workers used  if the gradient is approximated by finite
            differences. If different from one, the calculation relies on
            multi-processing to decrease the computation time. The default is 1.

        Returns
        -------
        NDArrayFloat
            The regularization gradient (not preconditioned).
        """
        if not values.ndim == 2:
            raise ValueError(
                "The 'EnsembleRegularizator.eval_loss_gradient_analytical' "
                "method expects a 2D vector!"
            )

        if is_finite_differences:
            return finite_gradient(values, self.eval_loss, max_workers=max_workers)
        else:
            return self.preconditioner.dtransform_vec(
                values,
                self.eval_loss_gradient_analytical(self.preconditioner(values)),
            )


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
