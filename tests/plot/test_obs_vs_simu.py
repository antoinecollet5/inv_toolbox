import matplotlib.pyplot as plt
import numpy as np
import pytest
from inv_toolbox.plot import plot_observed_vs_simulated


@pytest.mark.parametrize("n", (1, 200))
def test_plot_observed_s_simulated(n: int) -> None:
    obs = np.random.normal(loc=1, scale=1, size=n)
    pred_after_inversion = obs + np.random.normal(loc=0, scale=0.1, size=n)
    pred_before_inversion = obs + np.random.normal(loc=0, scale=0.5, size=n)

    # Test with minimum arguments
    fig, ax = plt.subplots()
    plot_observed_vs_simulated(ax, obs, pred_after_inversion)

    # Test with all arguments
    fig, ax = plt.subplots()
    plot_observed_vs_simulated(
        ax, obs, pred_after_inversion, pred_before_inversion, units="my unit"
    )
    ax.legend()

    with pytest.raises(
        ValueError,
        match=(
            'At least one for "pred_vector_initial" or "pred_vector" should be given !'
        ),
    ):
        plot_observed_vs_simulated(ax, obs, units="my unit")
