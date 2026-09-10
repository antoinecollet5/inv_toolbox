# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Tests for :func:`inv_toolbox.plot.plt_gradient.plot_2d_grad_res_adj_vs_fd`.


"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, required for CI / no display

import numpy as np
import pytest
import quickpaver
from inv_toolbox.plot.plt_gradient import plot_2d_grad_res_adj_vs_fd


@pytest.fixture
def geom() -> quickpaver.RectilinearGrid:
    """A small rectilinear grid geometry shared by the tests."""
    return quickpaver.RectilinearGrid(nx=3, ny=2, dx=10.0, dy=20.0)


def _assert_saved_figure(save_dir: Path, fname: str) -> None:
    """Assert that both the PNG and PDF outputs were written and non-empty."""
    png_path = save_dir / f"{fname}.png"
    pdf_path = save_dir / f"{fname}.pdf"
    assert png_path.is_file()
    assert pdf_path.is_file()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_full_featured_plot_triggers_autoscaling_and_both_well_types(
    geom: quickpaver.RectilinearGrid, tmp_path: Path
) -> None:
    """
    res_scaling=None with a large adj_grad vs small residuals forces the
    auto-scaling `while` loop to iterate more than once before exiting,
    and multiple wells of each type exercise both the first-well
    ("prod wells" / "inj wells") and subsequent-well ("_nolegend_")
    label branches.
    """
    adj_grad = np.array([[10.0, -10.0, 3.0], [5.0, 5.0, -2.0]])
    fd_grad = np.array([[9.0, -9.5, 3.1], [5.2, 4.8, -2.0]])

    save_dir = tmp_path / "nested" / "output"  # does not exist yet
    assert not save_dir.exists()

    plot_2d_grad_res_adj_vs_fd(
        adj_grad=adj_grad,
        fd_grad=fd_grad,
        geom=geom,
        fname="full_featured",
        fig_save_path=save_dir,
        grid_scaling=1.0,
        res_scaling=None,
        prod_locations=[(0, 0), (2, 1)],
        inj_locations=[(1, 0), (0, 1)],
    )

    # mkdir(parents=True, exist_ok=True) must have created the nested dir
    assert save_dir.is_dir()
    _assert_saved_figure(save_dir, "full_featured")


def test_explicit_res_scaling_and_no_wells(
    geom: quickpaver.RectilinearGrid, tmp_path: Path
) -> None:
    """
    res_scaling given explicitly skips the auto-scaling block entirely,
    and prod_locations/inj_locations both None skips both well-plotting
    blocks.
    """
    adj_grad = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, -1.0]])
    fd_grad = np.array([[1.1, 1.9, 0.4], [3.2, 3.8, -0.9]])

    save_dir = tmp_path / "output"

    plot_2d_grad_res_adj_vs_fd(
        adj_grad=adj_grad,
        fd_grad=fd_grad,
        geom=geom,
        fname="explicit_scaling",
        fig_save_path=save_dir,
        res_scaling=2.5,
        prod_locations=None,
        inj_locations=None,
    )

    _assert_saved_figure(save_dir, "explicit_scaling")


def test_zero_residuals_does_not_hang(
    geom: quickpaver.RectilinearGrid, tmp_path: Path
) -> None:
    """
    Regression test for the infinite-loop bug: when adj_grad == fd_grad
    (residuals are exactly zero) the auto-scaling loop's guard must
    prevent the `while max_adj > max_res * res_factor` condition from
    being evaluated at all (0 * res_factor never exceeds max_adj > 0,
    so without the guard this would loop forever).
    """
    same_grad = np.array([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
    save_dir = tmp_path / "output"

    # If the guard were missing/broken, this call would never return.
    plot_2d_grad_res_adj_vs_fd(
        adj_grad=same_grad,
        fd_grad=same_grad.copy(),
        geom=geom,
        fname="zero_residuals",
        fig_save_path=save_dir,
        res_scaling=None,
        prod_locations=None,
        inj_locations=None,
    )

    _assert_saved_figure(save_dir, "zero_residuals")
