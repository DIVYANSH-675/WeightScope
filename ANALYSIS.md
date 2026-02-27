# Experiment analysis

These tools use Python's standard library and work without loading a model. Example data below are deliberately small, authored fixtures; they are not research benchmark measurements. Reinstall the package after updating to register the `weightscope-analysis` entry point, or use the module commands shown here.

## Runnable workflows

Run from the repository root and use fresh output filenames:

```bash
python -m weightscope.analysis_cli distribution --input examples/analysis-before.json --key score --output runs/analysis-distribution.json
python -m weightscope.analysis_cli paired --before examples/analysis-before.json --after examples/analysis-after.json --id-key seed --key score --samples 1000 --seed 7 --output runs/analysis-paired.json
python -m weightscope.analysis_cli grouped --input examples/analysis-before.json --group-key condition --key score --output runs/analysis-groups.json
python -m weightscope.analysis_cli curve --input examples/analysis-curve.json --step-key step --key loss --window 2 --threshold 0.8 --mode min --patience 2 --output runs/analysis-curve.json
```

JSON arrays, single-record objects, JSONL, and reports containing `rows` or `episodes` arrays are supported. Records must contain the selected numeric fields. Boolean flags are not numerical observations. Nonfinite values and missing selected fields cause errors rather than silent row removal. Outputs are created exclusively, so existing reports are not overwritten.

Distribution reports include a mean, population standard deviation, median, quartiles, interquartile range, unscaled median absolute deviation, and a symmetric trimmed mean. Quantiles interpolate at `probability * (n - 1)`; each tail of a trimmed mean removes `floor(n * fraction)` observations. `--trim-fraction` defaults to 0.1.

Paired reports align by unique IDs before calculating after-minus-before differences. Reordered files are supported. Duplicate IDs are rejected; aggregate multiple snapshots from a run first. ID types are significant: integer `1`, numeric `1.0`, and string `"1"` are distinct. Unmatched IDs are errors by default; `--allow-unmatched` explicitly permits intersection-only analysis and records the excluded IDs. Reported intervals use paired percentile bootstrapping with `--confidence` (default 0.95). `--samples` and `--seed` control both randomized procedures.

Grouped reports preserve each experimental condition rather than pooling its observations. Curve reports sort an explicitly supplied step key, reject duplicate steps, and include raw values, trailing-window averages, exponential smoothing, linear trend, an observed threshold crossing, and a retrospective early-stopping decision. Without `--step-key`, input row order defines steps. Window and smoothing parameters count observations, not elapsed time. Early-stopping diagnostics do not alter any training run.

## Python API

Import functions from `weightscope.analysis`:

| Purpose | Functions |
|---|---|
| Distribution shape | `quantile`, `distribution_summary`, `median_absolute_deviation`, `trimmed_mean` |
| Aggregation and precision | `weighted_mean`, `mean_standard_error`, `metric_values`, `group_summaries` |
| Paired experiments | `align_pairs`, `paired_summary`, `paired_bootstrap`, `paired_permutation` |
| Multiple testing and association | `adjust_pvalues`, `pearson_correlation`, `rank_correlation` |
| Training curves | `linear_trend`, `rolling_mean`, `exponential_mean`, `first_threshold`, `early_stopping`, `curve_summary` |
| Multi-objective studies | `pareto_indices`, `compare_objectives` |
| Reference-set scaling | `fit_minmax`, `transform_minmax`, `fit_standardizer`, `transform_standard` |

```python
from weightscope.analysis import compare_objectives, pareto_indices

configurations = [{"quality": 0.8, "cost": 3}, {"quality": 0.9, "cost": 2}]
directions = {"quality": "max", "cost": "min"}
assert pareto_indices(configurations, directions) == [1]
comparison = compare_objectives(configurations[0], configurations[1], directions)
assert comparison["quality"]["outcome"] == "improved"
```

Pareto selection preserves tied nondominated rows and uses a quadratic comparison algorithm intended for modest configuration tables. Direction-aware comparisons use explicit `min`/`max` declarations and an optional absolute tolerance; they do not combine objectives into a hidden aggregate score.

Fit normalization state on an explicitly chosen reference or training set and reuse it for evaluation. Transforms do not refit or clip new observations. Constant min-max reference sets map to zero; constant standardizer reference sets use unit scale. Pearson and Spearman correlations are null when either variable is constant. Sample standard errors need at least two observations; they are not confidence intervals by themselves.

## Statistical assumptions and limitations

Use independent experimental repetitions for uncertainty estimates. Consecutive training steps, checkpoints from one run, and correlated pixels do not become independent merely because they occupy separate rows. Pair observations only when there is a meaningful matched design, such as the same evaluation seed. Grouping data does not establish independence or causality.

The paired permutation implementation flips the sign of each matched difference and tests the absolute mean difference. It assumes within-pair label exchangeability under the null. Randomized p-values use `(extreme + 1) / (samples + 1)`. This absolute-tail convention is explicit and may differ from other two-sided conventions; the [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) describes permutation designs and randomized-p-value adjustments.

`adjust_pvalues` implements Benjamini-Hochberg (`bh`) and Benjamini-Yekutieli (`by`) adjustments. BH's false-discovery interpretation requires suitable independence or positive-dependence conditions; BY is more conservative and supports broader dependence. Inputs must already be valid p-values. See the [SciPy false-discovery-control documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html) for these assumptions and the original references.

Small samples can produce unstable estimates or degenerate bootstrap intervals. A low p-value does not measure practical importance, prove model superiority, or correct a biased experiment. Do not select hypotheses, checkpoint criteria, or normalization data using the final test results. The original model limitations and benchmark observations in README.md and VALIDATION.md still apply.
