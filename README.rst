============
inv_toolbox
============

|License| |Stars| |Python| |PyPI| |Downloads| |Build Status| |Documentation Status| |Coverage| |Precommit: enabled| |Ruff| |ty|

🐍 A toolbox for inverse problem resolution.

**The complete and up to date documentation can be found here**: https://inv-toolbox.readthedocs.io.

===============
🎯 Motivations
===============

Solving an inverse problem (calibrating a geoscience model against observed data, for
instance) usually means minimizing a cost function made of a data-misfit term and one
or several regularization terms, then checking that the gradients fed to the optimizer
are actually correct. Each of these steps relies on a handful of recurring numerical
building blocks: spatial regularizators, parameter pre-transformations
(preconditioners), finite-difference approximations to validate analytical gradients,
and a few plotting helpers to inspect the result.

`inv_toolbox` gathers these building blocks behind a small, consistent, object-oriented
API so that they don't have to be reimplemented for every new inversion project. It
builds on `quickpaver <https://pypi.org/project/quickpaver>`_ for the regular grids
used by the local regularizators, and on
`covmats <https://github.com/antoinecollet5/covmats>`_ for the covariance matrix
representations used by the geostatistical regularizator.

===============
🚀 Quick start
===============

To install `inv_toolbox`, the easiest way is through `pip`:

.. code-block::

    pip install inv_toolbox

Or alternatively using `conda`

.. code-block::

    conda install inv_toolbox

You might also clone the repository and install from source

.. code-block::

    pip install -e .

Once the installation is done, `inv_toolbox` is straightforward to use and proposes
three submodules: `regularization`, `utils` and `plot`.

Let's start by importing `numpy`, `quickpaver` and `inv_toolbox` for the tests:

.. code-block:: python

    import numpy as np
    import quickpaver
    import inv_toolbox

Regularization
~~~~~~~~~~~~~~~

Local regularizators
^^^^^^^^^^^^^^^^^^^^^

`TikhonovRegularizator` penalizes spatial roughness of a field defined on a
`quickpaver.RectilinearGrid`, favoring smooth solutions. `TVRegularizator` uses a
total variation penalty instead, favoring blocky (piecewise constant) solutions.
Both are also available in a sparse-matrix form (`TikhonovMatRegularizator`,
`TVMatRegularizator`) and in a finite-volume form (`TikhonovFVMRegularizator`,
`TVFVMRegularizator`):

.. code-block:: python

    grid = quickpaver.RectilinearGrid(nx=15, ny=26, dx=3.6, dy=7.5)
    reg = inv_toolbox.regularization.TikhonovRegularizator(grid)

    rng = np.random.default_rng(2026)
    values = rng.random(grid.nx * grid.ny)
    reg.eval_loss(values)

.. code-block:: python

    3.2932279607534594

Every `Regularizator` exposes `eval_loss` as well as
`eval_loss_gradient_analytical`, so the gradient can be checked against a
finite-difference approximation with `inv_toolbox.utils.is_gradient_correct`:

.. code-block:: python

    inv_toolbox.utils.is_gradient_correct(
        values, reg.eval_loss, reg.eval_loss_gradient_analytical
    )

.. code-block:: python

    True

`DiscreteRegularizator` pulls a field toward a fixed set of discrete modes (for
facies or lithology parametrizations), and `ProbDistFitting` regularizes toward a
target empirical distribution.

Geostatistical regularizator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`GeostatisticalRegularizator` penalizes deviation from a prior using a parameter
covariance matrix supplied as a `covmats.CovarianceMatrix` (any of the
representations from `covmats`, dense or low-rank, can be used):

.. code-block:: python

    import covmats

    cov_m = covmats.CovViaDiagonal([1.0, 2.0, 3.0])
    reg = inv_toolbox.regularization.GeostatisticalRegularizator(cov_m)
    reg.eval_loss(np.array([4.0, -2.0, 5.0]))

.. code-block:: python

    13.166666666666666

An optional `covmats.PriorTerm` (e.g. a `ConstantPriorTerm` or a
`LinearDriftMatrix`-based term) can be passed to regularize toward something other
than zero.

Regularization weight strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The relative weight between the data-misfit and the regularization terms can be kept
`ConstantRegWeight` or updated on the fly with an `AdaptiveRegweight` strategy
(`AdaptiveUCRegweight`, `AdaptiveGradientNormRegweight`), and `get_l_curvature` helps
pick a weight from an L-curve analysis.

Utils
~~~~~~

`inv_toolbox.utils` provides the supporting numerical tools used above and
throughout an inversion workflow:

- Finite differences: `finite_gradient` and `finite_jacobian` compute
  finite-difference approximations of a gradient/Jacobian, and
  `is_gradient_correct`/`is_jacobian_correct` compare them against an analytical
  implementation, as shown earlier.
- Preconditioners: `Preconditioner` subclasses (`LogTransform`, `SqrtTransform`,
  `BoundsRescaler`, `StdRescaler`, `Normalizer`, `ChainedTransforms`, ...) implement
  variable changes applied to the parameters before the cost function is evaluated,
  to ease the optimizer's job or enforce constraints:

.. code-block:: python

    from inv_toolbox.utils.preconditioner import LogTransform

    LogTransform()(np.array([1.0, 10.0, 100.0]))

.. code-block:: python

    array([0.        , 2.30258509, 4.60517019])

- Spatial differential operators (`gradient_ffd`, `gradient_bfd`, `hessian_cfd`) and
  mean operators (`arithmetic_mean`, `harmonic_mean`, with their derivatives) used to
  build the regularizators above.
- Regular-grid helpers (`get_pts_coords_regular_grid`, `indices_to_node_number`,
  `get_array_borders_selection_2d`, ...) and spatial filters (`GaussianFilter`) to
  manipulate gridded parameter fields.

Plot
~~~~~

`inv_toolbox.plot` provides a few Matplotlib-based helpers to inspect inversion
results: `plot_observed_vs_simulated` for a calibration scatter plot,
`plot_percentiles` for ensemble percentile plots, and `plot_2d_grad_res_adj_vs_fd` to
visually compare an adjoint (or reverse-mode) gradient against its finite-difference
approximation.

===========
🔑 License
===========

This project is released under the **BSD 3-Clause License**.

Copyright (c) 2026, Antoine COLLET. All rights reserved.

For more details, see the `LICENSE <https://github.com/antoinecollet5/inv_toolbox/blob/master/LICENSE>`_ file included in this repository.

==============
⚠️ Disclaimer
==============

This software is provided "as is", without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a particular purpose,
or non-infringement. In no event shall the authors or copyright holders be liable for
any claim, damages, or other liability, whether in an action of contract, tort,
or otherwise, arising from, out of, or in connection with the software or the use
or other dealings in the software.

By using this software, you agree to accept full responsibility for any consequences,
and you waive any claims against the authors or contributors.

==========
📧 Contact
==========

For questions, suggestions, or contributions, you can reach out via:

- Email: antoine.collet5@gmail.com
- GitHub: https://github.com/antoinecollet5/inv_toolbox

We welcome contributions!

=============
📚 References
=============

TODO

* Free software: SPDX-License-Identifier: BSD-3-Clause

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/antoinecollet5/inv_toolbox/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/antoinecollet5/inv_toolbox.svg?style=social&label=Star&maxAge=2592000
    :target: https://github.com/antoinecollet5/inv_toolbox/stargazers
    :alt: Stars

.. |Python| image:: https://img.shields.io/pypi/pyversions/inv_toolbox.svg
    :target: https://pypi.org/pypi/inv_toolbox
    :alt: Python

.. |PyPI| image:: https://img.shields.io/pypi/v/inv_toolbox.svg
    :target: https://pypi.org/pypi/inv_toolbox
    :alt: PyPI

.. |Downloads| image:: https://static.pepy.tech/badge/inv_toolbox
    :target: https://pepy.tech/project/inv_toolbox
    :alt: Downloads

.. |Build Status| image:: https://github.com/antoinecollet5/inv_toolbox/actions/workflows/main.yml/badge.svg
    :target: https://github.com/antoinecollet5/inv_toolbox/actions/workflows/main.yml
    :alt: Build Status

.. |Documentation Status| image:: https://readthedocs.org/projects/inv_toolbox/badge/?version=latest
    :target: https://inv-toolbox.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Coverage| image:: https://codecov.io/gh/antoinecollet5/inv_toolbox/graph/badge.svg?token=oY3XZ1QTz3
    :target: https://codecov.io/gh/antoinecollet5/inv_toolbox
    :alt: Coverage

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/122673cd1d104aa28ada0c44b1f4e7d6
    :target: https://app.codacy.com/gh/antoinecollet5/inv_toolbox/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
    :alt: codacy

.. |Precommit: enabled| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |ty| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json
    :target: https://github.com/astral-sh/ty
    :alt: Checked with ty

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.18900358.svg
   :target: https://doi.org/10.5281/zenodo.18900358
