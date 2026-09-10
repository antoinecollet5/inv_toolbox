# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
inv_toolbox submodule providing tools and utilities for other submodules.

.. currentmodule:: inv_toolbox.utils.dataclass

Working with dataclasses
^^^^^^^^^^^^^^^^^^^^^^^^

Utilities for python dataclasses.

.. autosummary::
   :toctree: _autosummary

    default_field


.. currentmodule:: inv_toolbox.utils.grid

Regular grids
^^^^^^^^^^^^^

Provide utilities to work with regular grids.

.. autosummary::
   :toctree: _autosummary

    indices_to_node_number
    node_number_to_indices
    span_to_node_numbers_2d
    span_to_node_numbers_3d
    get_array_borders_selection_2d
    get_array_borders_selection_3d
    get_a_not_in_b_1d
    get_pts_coords_regular_grid
    create_selections_array_2d
    get_polygon_selection_with_dilation_2d
    get_extended_grid_shape


.. currentmodule:: inv_toolbox.utils.wellfield

WellField
^^^^^^^^^

Utilities to create wellfields.

.. autosummary::
   :toctree: _autosummary

    gen_wells_coordinates

.. currentmodule:: inv_toolbox.utils.enum

Working string enums
^^^^^^^^^^^^^^^^^^^^

Provide a str enum class.u

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
Provide some spatial filters

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

    object_or_object_sequence_to_list
    get_super_ilu_preconditioner
    check_random_state

Types
^^^^^
Other functions

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
    "Callback",
    "Filter",
    "GaussianFilter",
    "Int",
    "MeanType",
    "NDArrayBool",
    "NDArrayFloat",
    "NDArrayInt",
    "ArrayLike",
    "StrEnum",
    "amean_gradient",
    "arithmetic_mean",
    "check_random_state",
    "create_selections_array_2d",
    "default_field",
    "dxi_arithmetic_mean",
    "dxi_harmonic_mean",
    "finite_gradient",
    "finite_jacobian",
    "gen_random_ensemble",
    "get_a_not_in_b_1d",
    "get_angle_btw_vectors_deg",
    "get_angle_btw_vectors_rad",
    "get_array_borders_selection_2d",
    "get_array_borders_selection_3d",
    "get_extended_grid_shape",
    "get_mean_values_for_last_axis",
    "get_mean_values_gradient_for_last_axis",
    "get_normalized_mean_from_lognormal_params",
    "get_normalized_std_from_lognormal_params",
    "get_polygon_selection_with_dilation_2d",
    "get_pts_coords_regular_grid",
    "get_super_ilu_preconditioner",
    "gmean_gradient",
    "gradient_bfd",
    "gradient_ffd",
    "harmonic_mean",
    "hessian_cfd",
    "hmean_gradient",
    "indices_to_node_number",
    "is_all_close",
    "is_gradient_correct",
    "is_jacobian_correct",
    "node_number_to_indices",
    "object_or_object_sequence_to_list",
    "span_to_node_numbers_2d",
    "span_to_node_numbers_3d",
]
