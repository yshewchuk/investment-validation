"""Source-free magnitude reconstruction and proper scores, not admission.

Arithmetic convention: divide validated masses by their fsum (the format admits
only 1e-12 absolute roundoff), never mutate the document or smooth zero mass.
CDF endpoints are bounded to [0, 1] for arithmetic roundoff. Endpoint atom modes
are finite-bin sensitivity analyses, not changes to the recorded bin labels.
"""
from __future__ import annotations

from math import fsum, isfinite, log

from engine.v2.data.errors import fail
from engine.v2.research.magnitude_distribution import bin_index, validate_distribution

__all__ = ["components", "cdf", "mean", "crps", "empirical_crps", "bin_scores", "brier"]


def _refuse(message="invalid magnitude distribution arithmetic"):
    raise fail("CONTRACT_MISMATCH", message)


def _scalar(value, *, nonnegative=False):
    if type(value) not in (int, float):
        _refuse()
    try:
        result = float(value)
    except (ValueError, OverflowError):
        _refuse()
    if not isfinite(result) or (nonnegative and result < 0):
        _refuse()
    return result


def _finite_sum(values):
    try:
        result = fsum(values)
    except (ValueError, OverflowError):
        _refuse()
    if not isfinite(result):
        _refuse()
    return result


def components(document, reconstruction="uniform") -> tuple:
    """Return ordered immutable (lo, hi, mass) uniforms/atoms; atoms have lo=hi.

    Modes 'lower'/'upper' put finite-bin mass at its respective endpoint. Positive
    overflow mass is split equally over atoms at 20 + excess; zero masses are omitted;
    duplicate tail atoms remain separate. Unresolved tails refuse even when
    overflow mass is zero. Validation proves no causal/sample provenance.
    """
    distribution = validate_distribution(document)
    if type(reconstruction) is not str or reconstruction not in ("uniform", "lower", "upper"):
        _refuse()
    if distribution["tail"]["policy"] == "unresolved":
        _refuse("unresolved magnitude distribution tail")
    masses = distribution["probabilities"]
    total = fsum(masses)
    edges = distribution["finite_bin_edges"]
    result = []
    for index, mass in enumerate(masses[:-1]):
        if not mass:
            continue
        lo, hi = edges[index:index + 2]
        if reconstruction == "lower":
            hi = lo
        elif reconstruction == "upper":
            lo = hi
        result.append((lo, hi, mass / total))
    if masses[-1]:
        samples = distribution["tail"]["excesses"]
        weight = (masses[-1] / total) / len(samples)
        if not weight:
            _refuse("unrepresentable magnitude distribution tail weight")
        for excess in samples:
            point = _scalar(edges[-1] + float(excess), nonnegative=True)
            result.append((point, point, weight))
    return tuple(sorted(result))


def cdf(document, threshold, *, strict=False, reconstruction="uniform") -> float:
    """P(A <= threshold), or P(A < threshold) when strict=True; finite threshold."""
    threshold = _scalar(threshold)
    if type(strict) is not bool:
        _refuse()
    weights = []
    for lo, hi, weight in components(document, reconstruction):
        if lo == hi:
            fraction = float(lo < threshold or (not strict and lo == threshold))
        else:
            fraction = min(1.0, max(0.0, (threshold - lo) / (hi - lo)))
        weights.append(weight * fraction)
    return min(1.0, max(0.0, fsum(weights)))


def mean(document, *, reconstruction="uniform") -> float:
    """Mean magnitude in percentage points; refuses nonfinite arithmetic."""
    return _finite_sum(weight * (lo + (hi - lo) / 2)
                       for lo, hi, weight in components(document, reconstruction))


def _area(lo, hi, left, right, observation):
    if hi == lo:
        return 0.0
    if lo < observation < hi:
        middle = left + (right - left) * ((observation - lo) / (hi - lo))
        return _area(lo, observation, left, middle, observation) + _area(
            observation, hi, middle, right, observation)
    indicator = float(lo >= observation)
    a, b = left - indicator, right - indicator
    return (hi - lo) * ((a * a + a * b + b * b) / 3)


def crps(document, observation, *, reconstruction="uniform") -> float:
    """CRPS in percentage points, analytically integrating the piecewise CDF.

    Sorted disjoint uniforms/tail atoms cost O(n log n) time and O(n) space;
    no pairwise tail matrix or Monte Carlo. Observations above 20 are uncapped.
    """
    observation = _scalar(observation, nonnegative=True)
    return _crps_components(components(document, reconstruction), observation)


def empirical_crps(samples, observation) -> float:
    """Equal-weight nonnegative empirical CRPS; preserve atoms, including zero.

    Plain nonempty sample lists only, without binning, smoothing or clipping.
    The source-free caller is responsible for causal residual provenance.
    """
    observation = _scalar(observation, nonnegative=True)
    if type(samples) is not list or not samples:
        _refuse()
    values = sorted(_scalar(value, nonnegative=True) for value in samples)
    weight = 1 / len(values)
    return _crps_components(((value, value, weight) for value in values), observation)


def _crps_components(parts, observation):
    terms = []
    previous = cumulative = correction = 0.0
    for lo, hi, weight in parts:
        left = min(1.0, max(0.0, cumulative))
        terms.append(_area(previous, lo, left, left, observation))
        # Compensated cumulative mass avoids error growing with tail count.
        increment = weight - correction
        after = cumulative + increment
        correction = (after - cumulative) - increment
        right = min(1.0, max(0.0, after))
        terms.append(_area(lo, hi, left, right, observation))
        previous, cumulative = hi, after
    if observation > previous:
        terms.append(observation - previous)
    return _finite_sum(terms)


def bin_scores(document, observation) -> dict:
    """Unscaled 80-threshold RPS and natural-log class loss; unresolved allowed.

    Bin thresholds mean P(A < edge); exactly 20 belongs to overflow. These
    scores use recorded bin probabilities, not endpoint reconstructions.
    Zero observed-class probability returns +infinity, without smoothing.
    """
    distribution = validate_distribution(document)
    observed = bin_index(observation)
    masses = distribution["probabilities"]
    total = fsum(masses)
    probabilities = [mass / total for mass in masses]
    cumulative, terms = 0.0, []
    for index, probability in enumerate(probabilities[:-1]):
        cumulative = fsum((cumulative, probability))
        terms.append((cumulative - float(observed <= index)) ** 2)
    probability = probabilities[observed]
    return {"rps": fsum(terms), "log_loss": -log(probability) if probability else float("inf")}


def brier(probability, outcome) -> float:
    """Binary Brier loss; probability is a plain number, outcome a plain bool."""
    probability = _scalar(probability, nonnegative=True)
    if probability > 1 or type(outcome) is not bool:
        _refuse()
    return (probability - float(outcome)) ** 2
