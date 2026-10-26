"""Source-free fixed-bridge diagnostics, not executable STR-THRU economics.

Inputs are supplied decision-quote proxies in per-share dollars (spot/cost),
percentage points (magnitudes) and spot fractions (bridge/residual values).
No coefficient fitting, sample admission, IV repricing or fees are performed.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from math import fsum, isfinite

from engine.v2.data.errors import fail
from engine.v2.research.magnitude_distribution_math import components

__all__ = ["bridge_summary", "empirical_bridge_summary", "mean_matched_control"]


def _refuse():
    raise fail("CONTRACT_MISMATCH", "invalid magnitude payoff inputs or arithmetic")


def _number(value):
    if type(value) not in (int, float):
        _refuse()
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _refuse()
    if not isfinite(result):
        _refuse()
    return result


def _samples(values):
    if type(values) is not list or not values:
        _refuse()
    return [_number(value) for value in values]


def _sum(values):
    try:
        return _number(fsum(values))
    except OverflowError:
        _refuse()


def _law(parts, spot, intercept, slope):
    uniforms, atoms = [], []
    for lo, hi, mass in parts:
        low, high = sorted((_number((intercept + slope * lo) * spot),
                            _number((intercept + slope * hi) * spot)))
        if low == high or high <= 0:
            atoms.append((max(0.0, high), mass))
        elif low >= 0:
            uniforms.append((low, high, mass))
        else:
            positive = mass * (high / _number(high - low))
            atoms.append((0.0, mass - positive))
            uniforms.append((0.0, high, positive))
    atoms.sort()
    points, prefix, total, correction = [], [0.0], 0.0, 0.0
    for point, mass in atoms:
        points.append(point)
        adjusted = mass - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
        prefix.append(total)
    mean = _sum([mass * (lo / 2 + hi / 2) for lo, hi, mass in uniforms]
                + [point * mass for point, mass in atoms])
    return uniforms, points, prefix, mean


def _cdf(law, threshold, *, shift, strict):
    uniforms, points, prefix, _ = law
    # Compare the actual shifted atom, not threshold-shift: subtraction can
    # round differently at exact break-even and invent a profitable atom.
    search = bisect_left if strict else bisect_right
    index = search(points, threshold, key=lambda point: _number(point + shift))
    boundary = _number(threshold - shift) if uniforms else 0.0
    value = _sum([prefix[index]] + [mass * max(0.0, min(1.0, (boundary - lo) / (hi - lo)))
                                  for lo, hi, mass in uniforms])
    return max(0.0, min(1.0, value))  # Floating-point endpoint roundoff only.


def _integrate(parts, *, spot, entry_cost, intercept, slope, payoff_residuals):
    spot, entry_cost = _number(spot), _number(entry_cost)
    intercept, slope = _number(intercept), _number(slope)
    if spot <= 0 or entry_cost < 0:
        _refuse()
    shifts = [_number(value * spot) for value in _samples(payoff_residuals)]
    law = _law(parts, spot, intercept, slope)
    endpoints = law[1] + [edge for lo, hi, _ in law[0] for edge in (lo, hi)]
    low, high = min(endpoints), max(endpoints)
    for shift in shifts:
        _number(low + shift)
        _number(high + shift)
    size = len(shifts)
    mean_exit = _sum([law[3], _sum(value / size for value in shifts)])
    profit = _sum((1 - _cdf(law, entry_cost, shift=value, strict=False)) / size
                  for value in shifts)
    negative = _sum(_cdf(law, 0.0, shift=value, strict=True) / size for value in shifts)
    return {"expected_profit": _number(mean_exit - entry_cost),
            "probability_profit": profit, "probability_negative_exit": negative,
            "mean_exit_value": mean_exit}


def bridge_summary(document, *, spot, entry_cost, intercept, slope,
                   payoff_residuals, reconstruction="uniform") -> dict:
    """Integrate max(0, (intercept + slope*A)*spot) + residual*spot.

    Residuals are supplied equal-weight draws, independent of A in this frozen
    bridge. Fixed spot/cost are decision-quote proxies, not future paid prices.
    Profit is exit value minus entry_cost per share; broker fees, multipliers,
    fills and capacity are not modelled. Profit means strictly greater than 0.
    Residuals are added AFTER clipping: negative exit values remain visible.

    Uses the validated reconstruction's arithmetic mass normalization; no tail
    fitting, causal admission or realized-profit claim. An unresolved tail or
    nonfinite/overflowing arithmetic refuses. O(n log n + m log n + 80*m) time
    and O(n+m) space for n components and m residuals; no pairwise tail product,
    I/O, cache, writes or retries. CDF endpoints clip floating-point roundoff.
    """
    return _integrate(components(document, reconstruction=reconstruction), spot=spot,
                      entry_cost=entry_cost, intercept=intercept, slope=slope,
                      payoff_residuals=payoff_residuals)


def empirical_bridge_summary(magnitudes, *, spot, entry_cost, intercept, slope,
                             payoff_residuals) -> dict:
    """Apply the identical bridge to equal-weight empirical nonnegative atoms.

    Preserves zero/duplicate atoms of mean_matched_control without histogramming.
    Samples are not admitted or fitted; same numerical/proxy refusals as above.
    """
    samples = _samples(magnitudes)
    if any(value < 0 for value in samples):
        _refuse()
    parts = [(value, value, 1 / len(samples)) for value in samples]
    return _integrate(parts, spot=spot, entry_cost=entry_cost, intercept=intercept,
                      slope=slope, payoff_residuals=payoff_residuals)


def mean_matched_control(target_mean, residuals) -> dict:
    """Return equal-weight nonnegative empirical draws anchored to a mean.

    Construct max(0, target_mean + residual), then multiply by target/raw mean.
    This is the explicitly support-repaired control, not incumbent parity.
    Zero draws are atoms, not histogram first-bin uniforms. The target must be
    positive; an all-zero repaired pool or nonfinite arithmetic refuses rather
    than inventing a distribution. Returned achieved mean exposes roundoff.
    Residual origin/earlier-OOF eligibility is NOT authenticated here.
    O(m) time/detached space; no input mutation, fitting, I/O or retry.
    """
    target = _number(target_mean)
    if target <= 0:
        _refuse()
    repaired = [max(0.0, _number(target + value)) for value in _samples(residuals)]
    size = len(repaired)
    raw_mean = _sum(value / size for value in repaired)
    if raw_mean <= 0:
        _refuse()
    scale = _number(target / raw_mean)
    if scale <= 0:
        _refuse()
    draws = [_number(value * scale) for value in repaired]
    return {"magnitudes": draws, "target_mean": target, "raw_mean": raw_mean,
            "scale": scale, "zero_count": sum(value == 0 for value in draws),
            "anchored_mean": _sum(value / size for value in draws)}
