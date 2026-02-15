import pytest


def test_quantiles_interpolate_and_validate_observations():
    from weightscope.analysis import quantile

    assert quantile([8, 2, 4, 6], 0.25) == 3.5
    assert quantile([2, 4], 0) == 2 and quantile([2, 4], 1) == 4
    assert quantile([3], 0.9) == 3
    for values, probability in (([], 0.5), ([True], 0.5), ([1], 1.1), ([float("nan")], 0.5)):
        with pytest.raises(ValueError):
            quantile(values, probability)


def test_distribution_summary_distinguishes_outliers_from_medians():
    from weightscope.analysis import distribution_summary

    result = distribution_summary([1, 2, 3, 100])
    assert result["median"] == 2.5 and result["mean"] == 26.5
    assert result["q1"] == 1.75 and result["q3"] == 27.25
    assert distribution_summary([5])["std"] == 0


def test_mad_is_translation_invariant_and_resists_one_extreme_value():
    from weightscope.analysis import median_absolute_deviation

    assert median_absolute_deviation([1, 2, 3, 4, 1000]) == 1
    assert median_absolute_deviation([11, 12, 13, 14, 1010]) == 1
    assert median_absolute_deviation([7, 7]) == 0


def test_trimmed_mean_removes_equal_tail_counts():
    from weightscope.analysis import trimmed_mean

    report = trimmed_mean([-100, 1, 2, 3, 100], 0.2)
    assert report["mean"] == 2 and report["removed"] == 2
    assert trimmed_mean([4], 0.49)["retained"] == 1
    with pytest.raises(ValueError):
        trimmed_mean([1, 2], 0.5)


def test_weighted_mean_preserves_weight_scale_invariance():
    from weightscope.analysis import weighted_mean

    assert weighted_mean([1, 5], [3, 1]) == 2
    assert weighted_mean([1, 5], [300, 100]) == 2
    assert weighted_mean([1, 5], [0, 1]) == 5
    with pytest.raises(ValueError):
        weighted_mean([1, 2], [0, 0])
    with pytest.raises(ValueError):
        weighted_mean([1, 2], [1, -1])
