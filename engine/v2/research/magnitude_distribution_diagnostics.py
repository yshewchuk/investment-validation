"""Source-free forecast diagnostics, not calibration or sample admission."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from math import fsum, isfinite

from engine.v2.data.errors import fail
from engine.v2.research.magnitude_distribution_math import cdf, components

__all__ = ["quantile", "empirical_quantile", "pit_bracket", "interval_diagnostics",
           "prepare_empirical_diagnostics"]

_LEVELS = ((0.5, 0.25, 0.75), (0.8, 0.1, 0.9),
           (0.9, 0.05, 0.95), (0.95, 0.025, 0.975))


def _number(value, *, probability=False):
    if type(value) not in (int, float):
        raise fail("CONTRACT_MISMATCH", "invalid magnitude diagnostic input")
    try:
        value = float(value)
    except (ValueError, OverflowError):
        value = float("nan")
    if not isfinite(value) or value < 0 or (probability and value > 1):
        raise fail("CONTRACT_MISMATCH", "invalid magnitude diagnostic input")
    return value


def _samples(samples):
    if type(samples) is not list or not samples:
        raise fail("CONTRACT_MISMATCH", "invalid magnitude diagnostic samples")
    return tuple(sorted(_number(value) for value in samples))


def _quantile(parts, probability):
    if probability == 0:
        return parts[0][0]
    if probability == 1:
        return parts[-1][1]
    total = correction = 0.0
    for lo, hi, weight in parts:
        before = fsum((total, correction))
        after = total + weight
        correction += (total - after) + weight if total >= weight else (weight - after) + total
        total = after
        if probability <= fsum((total, correction)):
            fraction = min(1.0, max(0.0, (probability - before) / weight))
            return lo + (hi - lo) * fraction
    return parts[-1][1]  # Total-mass floating-point roundoff only.


def quantile(document, probability) -> float:
    """Generalized inverse inf{x: F(x)>=p} for 0<p<1; support ends at p=0/1.

    Uniform finite bins and empirical tail atoms use the existing components;
    empty bins have no support. Unresolved tails refuse even at zero overflow.
    """
    probability = _number(probability, probability=True)
    return _quantile(components(document), probability)


def _empirical_quantile(values, probability):
    # Search the floating CDF ranks directly: ceil(p*n) can cross an atom
    # incorrectly when multiplication rounds an adjacent probability to an int.
    index = bisect_left(range(len(values)), probability, key=lambda i: (i + 1) / len(values))
    return values[index]


def empirical_quantile(samples, probability) -> float:
    """Equal-weight empirical inverse; retain zero/duplicate atoms and endpoints."""
    probability = _number(probability, probability=True)
    return _empirical_quantile(_samples(samples), probability)


def pit_bracket(document, observation) -> tuple[float, float]:
    """Return (P(A<y), P(A<=y)); no midpoint or random draw replaces atoms.

    A later caller must seal its reproducible PIT randomization policy.
    """
    observation = _number(observation)
    return cdf(document, observation, strict=True), cdf(document, observation)


def _intervals(quantile_function):
    return tuple((level, quantile_function(lower), quantile_function(upper))
                 for level, lower, upper in _LEVELS)


def _coverage(intervals, observation):
    return [{"level": level, "lower": lower, "upper": upper,
             "width": upper - lower, "covered": lower <= observation <= upper}
            for level, lower, upper in intervals]


def interval_diagnostics(document, observation) -> list[dict]:
    """50/80/90/95% equal-tail intervals, width in points, inclusive coverage.

    Reconstruct once per call; no empirical interpolation, fitting or tuning.
    """
    observation = _number(observation)
    parts = components(document)
    return _coverage(_intervals(lambda probability: _quantile(parts, probability)), observation)


def prepare_empirical_diagnostics(samples):
    """Freeze equal-weight samples for repeated interval/PIT diagnostics.

    Validate/detach/sort once in O(n log n) time/O(n) space, preserving zeros
    and duplicates. Calls use O(log n) searches and return detached dicts with
    `intervals`, `pit_lower` and `pit_upper`. No label/source reads, randomization,
    cache, pairwise matrix, smoothing or causal admission is provided.
    """
    values = _samples(samples)
    intervals = _intervals(lambda probability: _empirical_quantile(values, probability))

    def diagnose(observation):
        observation = _number(observation)
        return {"intervals": _coverage(intervals, observation),
                "pit_lower": bisect_left(values, observation) / len(values),
                "pit_upper": bisect_right(values, observation) / len(values)}

    return diagnose
