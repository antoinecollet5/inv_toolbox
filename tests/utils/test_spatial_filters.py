import numpy as np
import pytest
from inv_toolbox.utils.spatial_filters import GaussianFilter


# get the function outside the class -> easier to test
@pytest.mark.parametrize(
    "sigmas, iteration",
    [
        (1, 1),
        (1.0, 1),
        ([1, 1], 1),
        ([1.0, 1.0], 1),
        ([[1.0, 1.0], 2, 3, 4], 1),
        ((1, 2, 3, 4), 5),
    ],
)
def test_gaussian_filter(sigmas, iteration) -> None:
    _filter = GaussianFilter(sigmas=sigmas)
    _filter.filter(np.random.random((200, 200)), iteration=iteration)
    assert _filter is not None


def test_gaussian_filter_error() -> None:
    with pytest.raises(
        ValueError,
        match="Sigmas should have the same dimension as the given parameter !",
    ):
        _filter = GaussianFilter(sigmas=[[2.0, 2.0, 2.0]])
        _filter.filter(np.random.random((200, 200)), iteration=1)


@pytest.mark.parametrize("iteration", [0, -1])
def test_gaussian_filter_iteration_below_one_error(iteration) -> None:
    _filter = GaussianFilter(sigmas=1.0)
    with pytest.raises(
        ValueError,
        match="iteration should be >= 1",
    ):
        _filter.filter(np.random.random((10, 10)), iteration=iteration)
