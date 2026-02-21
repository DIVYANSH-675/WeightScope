"""Standard-library diagnostics for independent experimental observations."""

from .artifacts import _finite_number


def _values(values, minimum=1):
    values = [_finite_number(value) for value in values]
    if len(values) < minimum:
        raise ValueError(f"At least {minimum} observations are required")
    return values


def _paired(before, after, minimum=1):
    before, after = _values(before, minimum), _values(after, minimum)
    if len(before) != len(after):
        raise ValueError("Paired observations must have equal lengths")
    return before, after


def quantile(values, probability):
    """Linear interpolation at probability*(n-1), including both endpoints."""
    values = sorted(_values(values))
    probability = _finite_number(probability)
    if not 0 <= probability <= 1:
        raise ValueError("Quantile probability must lie in [0,1]")
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return (1 - fraction) * values[lower] + fraction * values[upper]


def distribution_summary(values):
    import statistics

    values = _values(values)
    q1, median, q3 = (quantile(values, p) for p in (0.25, 0.5, 0.75))
    return dict(
        count=len(values),
        mean=statistics.fmean(values),
        std=statistics.pstdev(values),
        minimum=min(values),
        maximum=max(values),
        median=median,
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
    )


def median_absolute_deviation(values):
    """Unscaled MAD; this is not a normal-distribution standard deviation estimate."""
    values = _values(values)
    center = quantile(values, 0.5)
    return quantile([abs(value - center) for value in values], 0.5)


def trimmed_mean(values, fraction=0.1):
    import statistics

    values = sorted(_values(values))
    fraction = _finite_number(fraction)
    if not 0 <= fraction < 0.5:
        raise ValueError("Trim fraction must lie in [0,0.5)")
    cut = int(len(values) * fraction)
    kept = values[cut : len(values) - cut]
    return dict(mean=statistics.fmean(kept), retained=len(kept), removed=2 * cut, fraction=fraction)


def weighted_mean(values, weights):
    import math

    values, weights = _paired(values, weights)
    if any(weight < 0 for weight in weights) or not max(weights):
        raise ValueError("Weights must be nonnegative with a positive total")
    scaled = [weight / max(weights) for weight in weights]
    total = math.fsum(scaled)
    return math.fsum(value * (weight / total) for value, weight in zip(values, scaled))


def mean_standard_error(values):
    import math
    import statistics

    values = _values(values, minimum=2)
    deviation = statistics.stdev(values)
    return dict(
        count=len(values),
        mean=statistics.fmean(values),
        sample_std=deviation,
        standard_error=deviation / math.sqrt(len(values)),
    )


def _identity(value):
    if type(value) is str and value.strip():
        return ("str", value)
    if type(value) in (int, float):
        _finite_number(value)
        return (type(value).__name__, value)
    raise ValueError("Group and pair IDs must be nonempty strings or finite numbers")


def group_summaries(records, group_key, metric_key):
    groups = {}
    for row in records:
        if not isinstance(row, dict) or group_key not in row or metric_key not in row:
            raise ValueError("Every record must contain the group and metric keys")
        identity = _identity(row[group_key])
        groups.setdefault(identity, []).append(_finite_number(row[metric_key]))
    if not groups:
        raise ValueError("At least one grouped observation is required")
    return [
        dict(group=identity[1], **distribution_summary(values))
        for identity, values in groups.items()
    ]


def metric_values(records, key):
    if not isinstance(key, str) or not key:
        raise ValueError("A nonempty metric key is required")
    values = []
    for row in records:
        if not isinstance(row, dict) or key not in row:
            raise ValueError("Every record must contain the selected metric: " + key)
        values.append(_finite_number(row[key]))
    return _values(values)


def align_pairs(before, after, id_key, metric_key, strict=True):
    if type(strict) is not bool:
        raise ValueError("strict must be a boolean")

    def index(rows):
        indexed = {}
        for row in rows:
            if not isinstance(row, dict) or id_key not in row or metric_key not in row:
                raise ValueError("Paired records require ID and metric fields")
            identity = _identity(row[id_key])
            if identity in indexed:
                raise ValueError("Duplicate pair ID; aggregate repeated observations first")
            indexed[identity] = _finite_number(row[metric_key])
        return indexed

    left, right = index(before), index(after)
    only_left = [key for key in left if key not in right]
    only_right = [key for key in right if key not in left]
    if strict and (only_left or only_right):
        raise ValueError("Paired reports have unmatched observation IDs")
    common = [key for key in left if key in right]
    if not common:
        raise ValueError("Paired reports have no common observation IDs")
    return dict(
        ids=[key[1] for key in common],
        before=[left[key] for key in common],
        after=[right[key] for key in common],
        only_before=[key[1] for key in only_left],
        only_after=[key[1] for key in only_right],
    )


def paired_summary(before, after):
    before, after = _paired(before, after)
    differences = [new - old for old, new in zip(before, after)]
    return dict(
        delta=distribution_summary(differences),
        positive=sum(value > 0 for value in differences),
        negative=sum(value < 0 for value in differences),
        unchanged=sum(value == 0 for value in differences),
    )


def paired_bootstrap(before, after, samples=1000, confidence=0.95, seed=0):
    from .artifacts import bootstrap_mean

    before, after = _paired(before, after)
    return bootstrap_mean([new - old for old, new in zip(before, after)], samples, confidence, seed)


def paired_permutation(before, after, samples=1000, seed=0):
    """Absolute-mean two-sided sign-flip test; assumes within-pair label exchangeability."""
    import random
    import statistics

    before, after = _paired(before, after)
    differences = _values([new - old for old, new in zip(before, after)])
    if type(samples) is not int or not 1 <= samples <= 10000 or type(seed) is not int or seed < 0:
        raise ValueError("Use 1..10000 permutations and a nonnegative integer seed")
    signed = statistics.fmean(differences)
    observed = abs(signed)
    rng = random.Random(seed)
    extreme = 0
    tolerance = 1e-12 * max(1.0, observed)
    for _ in range(samples):
        statistic = abs(statistics.fmean(value * rng.choice((-1, 1)) for value in differences))
        extreme += statistic >= observed - tolerance
    return dict(
        count=len(differences),
        mean_delta=signed,
        statistic=observed,
        pvalue=(extreme + 1) / (samples + 1),
        samples=samples,
        seed=seed,
        method="Monte Carlo paired sign flips, absolute-mean tail",
    )


def adjust_pvalues(values, method="bh"):
    """Benjamini-Hochberg or Benjamini-Yekutieli adjusted p-values, original order."""
    import math

    values = _values(values)
    if any(not 0 <= value <= 1 for value in values) or method not in ("bh", "by"):
        raise ValueError("Provide p-values in [0,1] and method bh or by")
    n = len(values)
    factor = math.fsum(1 / index for index in range(1, n + 1)) if method == "by" else 1.0
    order = sorted(range(n), key=values.__getitem__)
    result = [0.0] * n
    ceiling = 1.0
    for position in range(n - 1, -1, -1):
        index = order[position]
        ceiling = min(ceiling, values[index] * n * factor / (position + 1))
        result[index] = ceiling
    return result


def pearson_correlation(left, right):
    import math
    import statistics

    left, right = _paired(left, right, minimum=2)

    def centered(values):
        scale = max(map(abs, values)) or 1.0
        normalized = [value / scale for value in values]
        mean = statistics.fmean(normalized)
        return [value - mean for value in normalized]

    left, right = centered(left), centered(right)
    norm = math.sqrt(
        math.fsum(value * value for value in left) * math.fsum(value * value for value in right)
    )
    if not norm:
        return None
    correlation = math.fsum(a * b for a, b in zip(left, right)) / norm
    return max(-1.0, min(1.0, correlation))


def rank_correlation(left, right):
    left, right = _paired(left, right, minimum=2)

    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0.0] * len(values)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[start]] == values[order[end]]:
                end += 1
            for index in order[start:end]:
                result[index] = (start + end - 1) / 2
            start = end
        return result

    return pearson_correlation(ranks(left), ranks(right))


def linear_trend(values, steps=None):
    import math
    import statistics

    values = _values(values, minimum=2)
    steps = list(range(len(values))) if steps is None else steps
    steps, values = _paired(steps, values, minimum=2)
    center_x, center_y = statistics.fmean(steps), statistics.fmean(values)
    dx, dy = [step - center_x for step in steps], [value - center_y for value in values]
    denominator = math.fsum(value * value for value in dx)
    if not denominator:
        raise ValueError("Trend estimation requires distinct step values")
    slope = _finite_number(math.fsum(x * y for x, y in zip(dx, dy)) / denominator)
    correlation = pearson_correlation(steps, values)
    return dict(
        count=len(values),
        slope=slope,
        intercept=_finite_number(center_y - slope * center_x),
        r_squared=correlation**2 if correlation is not None else None,
    )
