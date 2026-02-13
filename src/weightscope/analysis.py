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
