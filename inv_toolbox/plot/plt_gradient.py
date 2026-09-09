# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide plot utilities for gradient comparison"""

from pathlib import Path
from typing import List, Optional, Union

import covmats
import matplotlib.pyplot as plt
import nested_grid_plotter as ngp
import numpy as np
from inv_toolbox.utils import NDArrayFloat


def plot_2d_grad_res_adj_vs_fd(
    adj_grad: NDArrayFloat,
    fd_grad: NDArrayFloat,
    geom: covmats.RectilinearGrid,
    fname: str,
    fig_save_path: Path,
    grid_scaling: float = 1.0,
    res_scaling: Optional[float] = None,
    prod_locations: Optional[Union[NDArrayFloat, List[NDArrayFloat]]] = None,
    inj_locations: Optional[Union[NDArrayFloat, List[NDArrayFloat]]] = None,
) -> None:
    """
    Plot a side-by-side comparison of the adjoint-state and finite-difference
    2D gradients, along with their residuals, and save the figure to disk.

    Parameters
    ----------
    adj_grad : NDArrayFloat
        The 2D gradient computed with the adjoint-state method.
    fd_grad : NDArrayFloat
        The 2D gradient computed with finite differences, used as a
        reference to validate ``adj_grad``.
    geom : covmats.RectilinearGrid
        The rectilinear grid geometry (providing ``nx``, ``ny``, ``dx`` and
        ``dy``) used to set the plot extent and well marker positions.
    fname : str
        The base file name (without extension) used to save the figure.
    fig_save_path : Path
        The directory in which the figure is saved. It is created if it
        does not already exist.
    grid_scaling : float, optional
        A scaling factor applied to well indices before converting them to
        physical coordinates. The default is 1.0.
    res_scaling : Optional[float], optional
        An explicit multiplicative factor applied to the residuals for
        display purposes. If None (default), a factor is automatically
        chosen so that the scaled residuals' amplitude stays just below
        that of ``adj_grad``.
    prod_locations : Optional[Union[NDArrayFloat, List[NDArrayFloat]]], optional
        Grid-index locations (i, j) of the production wells to display as
        markers on every subplot. The default is None (no markers).
    inj_locations : Optional[Union[NDArrayFloat, List[NDArrayFloat]]], optional
        Grid-index locations (i, j) of the injection wells to display as
        markers on every subplot. The default is None (no markers).

    Returns
    -------
    None
        The figure is saved as both a PNG and a PDF file in
        ``fig_save_path`` under ``fname``; nothing is returned.
    """
    plotter = ngp.NestedGridPlotter(
        plt.figure(constrained_layout=True, figsize=(10, 8)),
        ngp.SubplotsMosaicBuilder(
            mosaic=[["ax1-1"], ["ax1-2"], ["ax1-3"]],
            sharey=True,
            sharex=True,
        ),
    )

    # We multiply the residuals so that the high residulas is just below the max values
    residuals = adj_grad - fd_grad

    if res_scaling is None:
        res_factor = 1.0
        max_adj = np.max(np.abs(adj_grad))
        max_res = np.max(np.abs(residuals))
        # Guard against an infinite loop: if the residuals are exactly zero
        # (adj_grad == fd_grad everywhere) or adj_grad is itself all zero,
        # the original "while max_adj > max_res * res_factor" condition
        # would never become False (0 * res_factor stays 0 forever) and
        # res_factor would grow without bound.
        if max_res > 0.0 and max_adj > 0.0:
            while max_adj > max_res * res_factor:
                res_factor *= 2.0
            # Make sure it is below
            res_factor /= 2.0
    else:
        res_factor = res_scaling

    ngp.multi_imshow(
        axes=plotter.axes,
        fig=plotter.fig,
        data={
            "Finite differences": fd_grad,
            "Adjoint state": adj_grad,
            f"Residuals (x {res_factor:.0e})": residuals * res_factor,
        },
        imshow_kwargs={
            "extent": [0.0, geom.nx * geom.dx, 0.0, geom.ny * geom.dy],
            "aspect": "equal",
        },
        xlabel="X [m]",
        ylabel="Z [m]",
        is_symmetric_cbar=True,
    )

    for ax in plotter.ax_dict.values():
        # Add some vertical lines to indicate the well
        if prod_locations is not None:
            for i, well_pos in enumerate(prod_locations):
                ax.plot(
                    well_pos[0] * grid_scaling * geom.dx + geom.dx / 2,
                    well_pos[1] * grid_scaling * geom.dy + geom.dy / 2,
                    # Only label the first marker so the legend does not
                    # end up with one duplicate "prod wells" entry per well.
                    label="prod wells" if i == 0 else "_nolegend_",
                    marker="^",
                    markersize=10,
                    c="black",
                    linestyle="none",
                )

        if inj_locations is not None:
            for i, well_pos in enumerate(inj_locations):
                ax.plot(
                    well_pos[0] * grid_scaling * geom.dx + geom.dx / 2,
                    well_pos[1] * grid_scaling * geom.dy + geom.dy / 2,
                    # Same fix as above for injection wells.
                    label="inj wells" if i == 0 else "_nolegend_",
                    marker="^",
                    markersize=10,
                    c="red",
                    linestyle="none",
                )

    # Make sure the output directory exists before saving, otherwise
    # fig.savefig raises a FileNotFoundError.
    fig_save_path.mkdir(parents=True, exist_ok=True)

    for fmt in ["png", "pdf"]:
        plotter.fig.savefig(str(fig_save_path.joinpath(f"{fname}.{fmt}")), format=fmt)
