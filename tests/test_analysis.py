import pytest


def test_quantiles_interpolate_and_validate_observations():
    from weightscope.analysis import quantile

    assert quantile([8, 2, 4, 6], 0.25) == 3.5
    assert quantile([2, 4], 0) == 2 and quantile([2, 4], 1) == 4
    assert quantile([3], 0.9) == 3
    for values, probability in (([], 0.5), ([True], 0.5), ([1], 1.1), ([float("nan")], 0.5)):
        with pytest.raises(ValueError):
            quantile(values, probability)
