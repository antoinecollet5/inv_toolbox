# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
inv_toolbox submodule providing tools and utilities for other submodules.

.. currentmodule:: inv_toolbox.utils.enum

Working with string enums
^^^^^^^^^^^^^^^^^^^^^^^^^

Provide a str enum class.

.. autosummary::
   :toctree: _autosummary

    StrEnum


.. currentmodule:: inv_toolbox.utils.finite_differences

Numerical approximation by finite differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide functions to compute the gradient of a function by
finite difference numerical approximation.

.. autosummary::
   :toctree: _autosummary

    finite_jacobian
    finite_gradient
    is_all_close
    is_jacobian_correct
    is_gradient_correct


.. currentmodule:: inv_toolbox.utils.operators

Spatial differential operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide functions for spatial differentiation.

.. autosummary::
   :toctree: _autosummary

    gradient_ffd
    gradient_bfd
    hessian_cfd
    get_angle_btw_vectors_deg
    get_angle_btw_vectors_rad

.. currentmodule:: inv_toolbox.utils.means

Mean operators
^^^^^^^^^^^^^^

Provide functions to perform mean and their first derivative.

.. autosummary::
   :toctree: _autosummary

    arithmetic_mean
    dxi_arithmetic_mean
    harmonic_mean
    dxi_harmonic_mean
    MeanType
    get_mean_values_for_last_axis
    amean_gradient
    gmean_gradient
    hmean_gradient
    get_mean_values_gradient_for_last_axis

.. currentmodule:: inv_toolbox.utils.spatial_filters

Filters
^^^^^^^

Provide some spatial filters.

.. autosummary::
   :toctree: _autosummary

    Filter
    GaussianFilter

.. currentmodule:: inv_toolbox.utils

Others
^^^^^^

Other functions

.. autosummary::
   :toctree: _autosummary

    Callback
    object_or_object_sequence_to_list
    get_super_ilu_preconditioner
    check_random_state

Types
^^^^^

Type aliases.

.. autosummary::
   :toctree: _autosummary

    NDArrayFloat
    NDArrayInt
    NDArrayBool
    Int
    ArrayLike

Preconditioners
^^^^^^^^^^^^^^^

Sub module providing preconditioners and parametrization tools.

.. autosummary::
   :toctree: _autosummary

    preconditioner

.. currentmodule:: inv_toolbox.utils

"""

from scipy._lib._util import check_random_state  # To handle random_state

from inv_toolbox.utils.callbacks import Callback
from inv_toolbox.utils.enum import StrEnum
from inv_toolbox.utils.finite_differences import (
    finite_gradient,
    finite_jacobian,
    is_all_close,
    is_gradient_correct,
    is_jacobian_correct,
)
from inv_toolbox.utils.means import (
    MeanType,
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
from inv_toolbox.utils.operators import (
    get_angle_btw_vectors_deg,
    get_angle_btw_vectors_rad,
    get_super_ilu_preconditioner,
    gradient_bfd,
    gradient_ffd,
    hessian_cfd,
)
from inv_toolbox.utils.preconditioner import (
    GDPCS,
    GDPNCS,
    BoundsClipper,
    BoundsRescaler,
    ChainedTransforms,
    GradientScalerConfig,
    InvAbsTransform,
    LinearTransform,
    LogTransform,
    Normalizer,
    NoTransform,
    Preconditioner,
    RangeRescaler,
    SigmoidRescaler,
    SigmoidRescalerBounded,
    Slicer,
    SqrtTransform,
    StdRescaler,
    SubSelector,
    Uniform2Gaussian,
)
from inv_toolbox.utils.spatial_filters import Filter, GaussianFilter
from inv_toolbox.utils.types import (
    ArrayLike,
    Int,
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    object_or_object_sequence_to_list,
)

__all__ = [
    "ArrayLike",
    "BoundsClipper",
    "BoundsRescaler",
    "Callback",
    "ChainedTransforms",
    "Filter",
    "GDPCS",
    "GDPNCS",
    "GaussianFilter",
    "GradientScalerConfig",
    "Int",
    "InvAbsTransform",
    "LinearTransform",
    "LogTransform",
    "MeanType",
    "NDArrayBool",
    "NDArrayFloat",
    "NDArrayInt",
    "NoTransform",
    "Normalizer",
    "Preconditioner",
    "RangeRescaler",
    "SigmoidRescaler",
    "SigmoidRescalerBounded",
    "Slicer",
    "SqrtTransform",
    "StdRescaler",
    "StrEnum",
    "SubSelector",
    "Uniform2Gaussian",
    "amean_gradient",
    "arithmetic_mean",
    "check_random_state",
    "dxi_arithmetic_mean",
    "dxi_harmonic_mean",
    "finite_gradient",
    "finite_jacobian",
    "get_angle_btw_vectors_deg",
    "get_angle_btw_vectors_rad",
    "get_mean_values_for_last_axis",
    "get_mean_values_gradient_for_last_axis",
    "get_super_ilu_preconditioner",
    "gmean_gradient",
    "gradient_bfd",
    "gradient_ffd",
    "harmonic_mean",
    "hessian_cfd",
    "hmean_gradient",
    "is_all_close",
    "is_gradient_correct",
    "is_jacobian_correct",
    "object_or_object_sequence_to_list",
]
