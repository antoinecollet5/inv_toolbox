# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from inv_toolbox.utils import NDArrayFloat


def plot_percentiles(
    ax: Axes,
    data: NDArrayFloat,
    x: Optional[NDArrayFloat] = None,
    cmap: Colormap = plt.get_cmap("Reds"),
    method: str = "median_unbiased",
) -> Axes:
    """
    Plot the percentiles of a timeseries using color gradient.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot.
    data : NDArrayFloat
        Array of timeseries with shape (nx, n_sample), n_sample being the
        number of timeseries in the sample.
    x : Optional[NDArrayFloat], optional
        The x-axis coordinates (length nx) at which to plot the
        percentiles. If None (default), it defaults to
        ``np.arange(data.shape[0])``.
    cmap : Colormap, optional
        Colormap to use for the percentile bands. The default is the
        "Reds" colormap.
    method : str, optional
        The interpolation method passed to ``numpy.percentile`` (e.g.
        "median_unbiased", "linear", ...). The default is
        "median_unbiased".

    Returns
    -------
    Axes
        Updated input axis.
    """

    n = 19  # 9 bins + the P50 that we won't use
    percentiles = np.linspace(start=5.0, stop=95.0, num=n)
    s_dist = np.percentile(data, q=percentiles, method=method, axis=1)  # ty:ignore[no-matching-overload]

    if x is None:
        _x: NDArrayFloat = np.arange(data.shape[0])
    else:
        _x = x

    # plot the color ranges
    for i in range(int(n / 2)):
        ax.fill_between(
            _x,
            s_dist[i],
            s_dist[-(i + 1)],
            color=cmap(0.1 + i / n * 2 / 1.6),
            # Label each band with its actual lower/upper percentile bounds.
            # The previous label (percentiles[i * 2 + 1]) did not correspond
            # to the band being shaded - e.g. the innermost band (P45-P55,
            # i=8) was mislabeled using percentiles[17] (P90).
            label=f"P{percentiles[i]:.0f}-P{percentiles[-(i + 1)]:.0f}",
        )

    # plot the median
    ax.plot(_x, s_dist[int(n / 2)], linestyle="-", c="k", label="Median")

    return ax
