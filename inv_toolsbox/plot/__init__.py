# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
inv_toolbox submodule providing a set of handy plot tools.

.. currentmodule:: inv_toolbox.plot

Plot functions
^^^^^^^^^^^^^^
Functions to plot inversion results.

.. autosummary::
   :toctree: _autosummary

   plot_observed_vs_simulated

"""

from inv_toolbox.plot.plt_gradient import plot_2d_grad_res_adj_vs_fd
from inv_toolbox.plot.plt_obs_vs_simu import plot_observed_vs_simulated
from inv_toolbox.plot.plt_percentiles import plot_percentiles

__all__ = [
    "plot_observed_vs_simulated",
    "plot_2d_grad_res_adj_vs_fd",
    "plot_percentiles",
]
