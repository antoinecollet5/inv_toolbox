# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Purpose
=======

**inv_toolbox** is an open-source, pure python, and object-oriented library that provides
user friendly tools for inverse problem resolution (geoscience oriented).

Submodules
==========

.. autosummary::
    utils
    plot
    regularization

"""

from inv_toolbox import plot, regularization, utils
from inv_toolbox.__about__ import __author__, __email__, __version__


__all__ = [
    "__version__",
    "__email__",
    "__author__",
    "utils",
    "plot",
    "regularization",
]
