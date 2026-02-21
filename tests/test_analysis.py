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


def test_standard_error_uses_sample_dispersion_and_sample_count():
    from weightscope.analysis import mean_standard_error

    report = mean_standard_error([1, 3])
    assert report["sample_std"] == pytest.approx(2**0.5)
    assert report["standard_error"] == pytest.approx(1)
    with pytest.raises(ValueError):
        mean_standard_error([1])


def test_group_summaries_keep_conditions_and_id_types_separate():
    from weightscope.analysis import group_summaries

    rows = [{"seed": "a", "loss": 1}, {"seed": "b", "loss": 8}, {"seed": "a", "loss": 3}]
    report = group_summaries(rows, "seed", "loss")
    assert report[0]["group"] == "a" and report[0]["mean"] == 2
    assert report[1]["count"] == 1
    with pytest.raises(ValueError):
        group_summaries([{"seed": None, "loss": 2}], "seed", "loss")


def test_metric_selection_does_not_silently_drop_missing_observations():
    from weightscope.analysis import metric_values

    assert metric_values([{"loss": 1, "note": "a"}, {"loss": 2}], "loss") == [1.0, 2.0]
    with pytest.raises(ValueError, match="selected metric"):
        metric_values([{"loss": 1}, {"accuracy": 0.5}], "loss")
    with pytest.raises(ValueError):
        metric_values([{"loss": True}], "loss")


def test_pair_alignment_handles_reordering_and_rejects_duplicate_seeds():
    from weightscope.analysis import align_pairs

    before = [{"seed": 1, "loss": 3}, {"seed": 2, "loss": 4}]
    after = [{"seed": 2, "loss": 2}, {"seed": 1, "loss": 1}]
    report = align_pairs(before, after, "seed", "loss")
    assert report["ids"] == [1, 2] and report["after"] == [1, 2]
    with pytest.raises(ValueError, match="Duplicate"):
        align_pairs(before + before[:1], after, "seed", "loss")
    with pytest.raises(ValueError, match="unmatched"):
        align_pairs(before, after[:1], "seed", "loss")
    assert align_pairs(before, after[:1], "seed", "loss", strict=False)["only_before"] == [1]


def test_paired_summary_preserves_pairwise_changes():
    from weightscope.analysis import paired_summary

    report = paired_summary([3, 1, 2], [1, 4, 2])
    assert report["delta"]["mean"] == pytest.approx(1 / 3)
    assert (report["positive"], report["negative"], report["unchanged"]) == (1, 1, 1)
    with pytest.raises(ValueError):
        paired_summary([1], [1, 2])


def test_paired_bootstrap_preserves_shared_variation():
    from weightscope.analysis import paired_bootstrap

    report = paired_bootstrap([1, 100, 1000], [3, 102, 1002], samples=30, seed=4)
    assert report["lower"] == report["upper"] == report["mean"] == 2
    assert report == paired_bootstrap([1, 100, 1000], [3, 102, 1002], samples=30, seed=4)


def test_paired_permutation_is_seeded_symmetric_and_never_returns_zero():
    from weightscope.analysis import paired_permutation

    result = paired_permutation([0] * 8, [1] * 8, samples=100, seed=3)
    reverse = paired_permutation([1] * 8, [0] * 8, samples=100, seed=3)
    assert result["pvalue"] == reverse["pvalue"]
    assert 1 / 101 <= result["pvalue"] <= 1
    assert result == paired_permutation([0] * 8, [1] * 8, samples=100, seed=3)
    assert paired_permutation([1, 2], [1, 2], samples=10)["pvalue"] == 1


def test_pvalue_adjustment_preserves_order_and_handles_dependence_correction():
    from weightscope.analysis import adjust_pvalues

    assert adjust_pvalues([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert adjust_pvalues([0.01, 0.04, 0.03], "by") == pytest.approx(
        [0.055, 0.073333333333, 0.073333333333]
    )
    assert adjust_pvalues([0, 1]) == [0, 1]
    with pytest.raises(ValueError):
        adjust_pvalues([1.1])


def test_pearson_handles_rescaling_constants_and_reversed_association():
    from weightscope.analysis import pearson_correlation

    assert pearson_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1)
    assert pearson_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert pearson_correlation([1, 1], [2, 3]) is None
    assert pearson_correlation([1e300, 2e300, 3e300], [1, 2, 3]) == pytest.approx(1)


def test_rank_association_is_invariant_to_monotone_rescaling():
    from weightscope.analysis import rank_correlation

    assert rank_correlation([1, 1, 3], [2, 2, 99]) == pytest.approx(1)
    assert rank_correlation([1, 2, 3], [100, 10, 1]) == pytest.approx(-1)
    assert rank_correlation([1, 1], [2, 3]) is None


def test_trend_respects_nonuniform_step_coordinates():
    from weightscope.analysis import linear_trend

    report = linear_trend([2, 4, 6], [1, 3, 5])
    assert report["slope"] == pytest.approx(1) and report["intercept"] == pytest.approx(1)
    assert report["r_squared"] == pytest.approx(1)
    assert linear_trend([3, 3])["r_squared"] is None
    with pytest.raises(ValueError):
        linear_trend([1, 2], [1, 1])
