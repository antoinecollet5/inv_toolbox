from typing import Tuple, no_type_check

import covmats
import numpy as np
import pytest
import scipy as sp
from inv_toolbox.regularization import (
    GeostatisticalRegularizator,
)
from inv_toolbox.regularization.geostatistical import identify_function, one
from inv_toolbox.utils import NDArrayFloat
from inv_toolbox.utils.preconditioner import LinearTransform


def test_identify_function() -> None:
    x = np.array([1.0, -2.5, 3.0])
    np.testing.assert_array_equal(identify_function(x), x)
    # must be the untransformed input, not a copy with altered values
    assert identify_function(x) is x


def test_one() -> None:
    x = np.array([1.0, -2.5, 3.0])
    np.testing.assert_array_equal(one(x), np.ones(3))

    x2d = np.zeros((2, 4))
    np.testing.assert_array_equal(one(x2d), np.ones((2, 4)))


@no_type_check
def _get_L_D_P(A: sp.sparse.sparray):
    """
    Return L, D and P from the factorization L @ D @ L' = P @ A @ P' using sksparse.

    Note that sksparse uses SuiteSparse which is LGPL licence.
    """
    import sksparse.cholmod as cholmod

    # Need to take the API change into account
    try:
        # sksparse 4.x
        L, D, P = cholmod.ldl(A, order="amd")
    except AttributeError:
        # sksparse 5.x
        f = cholmod.cholesky(A)
        (L, D), P = f.L_D(), f.P()
    return L, D, P


# For now we use the exact parameters, we will complexify a bit later
prior_std = 1.0
len_scale: NDArrayFloat = np.array([250.0, 250.0])


def _get_scf(A: sp.sparse.sparray) -> covmats.SparseCholeskyFactor:
    """Return a cholesky factorization of the precision matrix."""
    return covmats.SparseCholeskyFactor(*_get_L_D_P(A))


def exponential_kernel(r: float) -> NDArrayFloat:
    """Test covariance kernel."""
    return (prior_std**2) * np.exp(-r)


def get_domain_shape() -> Tuple[int, int]:
    nx: int = 7
    ny: int = 11
    return (nx, ny)


def get_mesh_dim() -> Tuple[float, float]:
    dx: float = 3.5
    dy: float = 4.2
    return (dx, dy)


def get_pts() -> NDArrayFloat:
    dx, dy = get_mesh_dim()
    nx, ny = get_domain_shape()
    x = np.linspace(0.0 + dx / 2.0, nx * dx - dx / 2.0, nx)
    y = np.linspace(0.0 + dy / 2.0, ny * dy - dy / 2.0, ny)

    XX, YY = np.meshgrid(x, y)
    return np.hstack((XX.ravel("F")[:, np.newaxis], YY.ravel("F")[:, np.newaxis]))


def get_param_values() -> NDArrayFloat:
    """Generate a parameter field with some noise."""
    param: NDArrayFloat = np.zeros(get_domain_shape(), dtype=np.float64)
    param[0:5, 2:6] = 5.0
    param[3:6, 6:7] = 10.0
    param[2:5, 1:8] = 20.0

    # Add some noise with a seed
    rng = np.random.default_rng(26659)
    param += rng.random(get_domain_shape()) * 5.0

    return param.ravel("F")


@pytest.mark.parametrize(
    "cov_mat,atol",
    [
        # (
        #     covmats.ge
        #     generate_dense_matrix(
        #         pts=get_pts(),
        #         kernel=exponential_kernel,
        #         len_scale=len_scale,
        #     ),
        #     1e-4,
        # ),
        # (
        #     covmats.eigen_factorize_cov_mat(
        #         covmats.CovViaCholesky(sp.linalg.cholesky(covmats.
        # CovKernelAsLinopViaFFT(
        #             kernel=exponential_kernel, len_scale=len_scale, pts=get_pts()
        #         ).todense(),
        #         n_pc=32,
        #     ),
        #     1e-5,
        # ),
        (
            covmats.CovKernelAsLinopViaFFT(
                kernel=exponential_kernel,
                mesh_dim=get_mesh_dim(),
                domain_shape=get_domain_shape(),
                len_scale=len_scale,
                k=30,
            ),
            1e-2,
        ),
        (
            covmats.eigen_factorize_cov_mat(
                covmats.CovKernelAsLinopViaFFT(
                    kernel=exponential_kernel,
                    mesh_dim=get_mesh_dim(),
                    domain_shape=get_domain_shape(),
                    len_scale=len_scale,
                    k=30,
                ),
                n_pc=32,
            ),
            1e-4,
        ),
        (
            covmats.CovViaEnsemble(
                np.random.default_rng(2023).random(
                    size=(200, np.prod(get_domain_shape()))
                )
            ),
            1e-4,
        ),
    ],
)
def test_regularizator_gradients_by_fd(cov_mat, atol) -> None:
    """Test the correctness of the gradients by finite differences."""
    param_values = get_param_values()

    regularizator = GeostatisticalRegularizator(cov_mat)

    print(f"loss_reg_dense = {regularizator.eval_loss(param_values)}")

    grad_reg_fd = regularizator.eval_loss_gradient(
        param_values, is_finite_differences=True
    )
    grad_reg_analytic = regularizator.eval_loss_gradient(param_values)
    np.testing.assert_allclose(grad_reg_fd, grad_reg_analytic, atol=atol)


@pytest.mark.parametrize(
    "prior",
    [
        covmats.NullPriorTerm(),
        covmats.ConstantPriorTerm(
            np.full(get_param_values().size, np.mean(get_param_values()))
        ),
        covmats.MeanPriorTerm(),
        #        DriftMatrix(),
        #        LinearDriftMatrix,
    ],
)
def test_regularizator_gradients_with_priors_by_fd(prior) -> None:
    """Test the correctness of the gradients by finite differences."""
    param_values = get_param_values()

    cov_mat = covmats.eigen_factorize_cov_mat(
        covmats.CovKernelAsLinopViaFFT(
            kernel=exponential_kernel,
            mesh_dim=get_mesh_dim(),
            domain_shape=get_domain_shape(),
            len_scale=len_scale,
            k=30,
        ),
        n_pc=32,
    )

    # transform to test the change of variable
    regularizator = GeostatisticalRegularizator(
        cov_mat, prior, preconditioner=LinearTransform(1.0, 4.0)
    )

    print(f"loss_reg_dense = {regularizator.eval_loss(param_values)}")

    grad_reg_fd = regularizator.eval_loss_gradient(
        param_values, is_finite_differences=True
    )
    grad_reg_analytic = regularizator.eval_loss_gradient(param_values)
    np.testing.assert_allclose(grad_reg_fd, grad_reg_analytic, atol=1e-4)
