# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
inv_toolbox sub module providing regularization tools.

The following functionalities are directly provided on module-level.

.. currentmodule:: inv_toolbox.regularization

Abstract classes
================

Base class from which to derive regularizator implementations.

.. autosummary::
   :toctree: _autosummary

   Regularizator

Local
=====

Tikhonov (for smooth spatial distribution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    TikhonovRegularizator
    TikhonovMatRegularizator
    TikhonovFVMRegularizator

Total Variation (for blocky spatial distribution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    TVRegularizator
    TVMatRegularizator
    TVFVMRegularizator

Discrete to impose specific discrete values to the field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    DiscreteRegularizator

Global
======

Fitting empirical distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    ProbDistFitting

Geostatistic regularizator
^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide classes to implement regularization based on a parameter covariance matrix.
The first one work with a single vector while the second class works with
an ensemble of realizations.

.. autosummary::
   :toctree: _autosummary

    GeostatisticalRegularizator
    EnsembleRegularizator

Regularization weights selection
================================

Strategies for the weight evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Class to indicate what strategy to use to weight the objective
function regularization term.

.. autosummary::
   :toctree: _autosummary

    RegWeightUpdateStrategy
    AdaptiveRegweight
    AdaptiveUCRegweight
    AdaptiveGradientNormRegweight
    ConstantRegWeight

Curvature in the context of L-curve plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate curvature of a L-curve

.. autosummary::
   :toctree: _autosummary

    get_l_curvature

"""

from inv_toolbox.regularization.adaptive import (
    AdaptiveGradientNormRegweight,
    AdaptiveRegweight,
    AdaptiveUCRegweight,
)
from inv_toolbox.regularization.base import (
    ConstantRegWeight,
    Regularizator,
    RegWeightUpdateStrategy,
)
from inv_toolbox.regularization.discrete import DiscreteRegularizator
from inv_toolbox.regularization.distribution import ProbDistFitting
from inv_toolbox.regularization.geostatistical import (
    EnsembleRegularizator,
    GeostatisticalRegularizator,
)
from inv_toolbox.regularization.lcurve import get_l_curvature
from inv_toolbox.regularization.tikhonov import (
    TikhonovFVMRegularizator,
    TikhonovMatRegularizator,
    TikhonovRegularizator,
)
from inv_toolbox.regularization.tv import (
    TVFVMRegularizator,
    TVMatRegularizator,
    TVRegularizator,
)

__all__ = [
    "RegWeightUpdateStrategy",
    "Regularizator",
    "TikhonovRegularizator",
    "TikhonovMatRegularizator",
    "TVRegularizator",
    "TVMatRegularizator",
    "GeostatisticalRegularizator",
    "EnsembleRegularizator",
    "AdaptiveUCRegweight",
    "AdaptiveRegweight",
    "ConstantRegWeight",
    "DiscreteRegularizator",
    "get_l_curvature",
    "AdaptiveGradientNormRegweight",
    "TikhonovFVMRegularizator",
    "TVFVMRegularizator",
    "ProbDistFitting",
]
