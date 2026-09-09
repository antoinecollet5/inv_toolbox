# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

from typing import Optional

import numpy as np
from inv_toolbox.utils import NDArrayFloat
from matplotlib.axes import Axes


def plot_observed_vs_simulated(
    ax: Axes,
    obs_vector: NDArrayFloat,
    pred_vector: Optional[NDArrayFloat] = None,
    pred_vector_initial: Optional[NDArrayFloat] = None,
    units: Optional[str] = None,
) -> None:
    """
    Plot observed vs simulated data.

    Parameters
    ----------
    ax: Axes
        The ax on which to plot the data.
    obs_vector : NDArrayFloat
        The vector of observed data.
    pred_vector : Optional[NDArrayFloat], optional
        The vector of predicted data. The default is None.
    pred_vector_initial : Optional[NDArrayFloat], optional
        An optional additional vector of 'initial' predicted data.
        It allows to see the difference between before and after an inversion.
        The default is None.
    units : Optional[str], optional
        The unit of the data (for display). The default is None.

    Returns
    -------
    None
        The plot is drawn directly on ``ax``; nothing is returned.

    Raises
    ------
    ValueError
        If both ``pred_vector`` and ``pred_vector_initial`` are None, since
        there would then be nothing to plot against ``obs_vector``.
    """
    ax.set_title("obs. vs simul.", fontweight="bold")
    if pred_vector_initial is None and pred_vector is None:
        raise ValueError(
            'At least one for "pred_vector_initial" or "pred_vector" should be given !'
        )
    VERY_LARGE_NB = 1.0e40
    minobs = VERY_LARGE_NB
    maxobs = -VERY_LARGE_NB
    if pred_vector_initial is not None:
        ax.plot(
            obs_vector, pred_vector_initial, ".", c="r", zorder=1, label="initial guess"
        )
        minobs: float = min(
            minobs, float(np.min(np.vstack((obs_vector, pred_vector_initial.ravel()))))
        )
        maxobs: float = max(
            maxobs, float(np.max(np.vstack((obs_vector, pred_vector_initial.ravel()))))
        )
    if pred_vector is not None:
        ax.plot(obs_vector, pred_vector, ".", c="b", zorder=2, label="post-inversion")
        minobs: float = min(
            minobs, float(np.min(np.vstack((obs_vector, pred_vector.ravel()))))
        )
        maxobs: float = max(
            maxobs, float(np.max(np.vstack((obs_vector, pred_vector.ravel()))))
        )

    ax.set_aspect("equal", adjustable="box")
    # ax.axis('square')
    _suffix = ""
    if units is not None:
        _suffix = f" [{units}]"
    ax.set_xlabel("observed" + _suffix, fontweight="bold")
    ax.set_ylabel("simulated" + _suffix, fontweight="bold")
    margin: float = 0.05 * np.abs(
        maxobs - minobs
    )  # 5% on each side for a nicer display
    if margin == 0.0:
        # All values are identical (maxobs == minobs): a zero-width margin
        # would give matplotlib an xlim/ylim with identical low and high
        # bounds, which it cannot render (raises/warns "Attempting to set
        # identical low and high xlims"). Fall back to a small absolute
        # margin instead.
        margin = 1.0 if maxobs == 0.0 else 0.05 * np.abs(maxobs)
    ax.set_xlim((minobs - margin, maxobs + margin))
    ax.set_ylim(*ax.get_xlim())

    ax.plot(
        np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 20),
        np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 20),
        "k-",
    )
