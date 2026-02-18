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
