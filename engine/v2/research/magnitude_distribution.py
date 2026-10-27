"""Source-free absolute-move histograms; numerical validity is not admission.

This additive research format does not change earnings_magnitude.v1. It neither
fits a tail nor authenticates caller-supplied samples, forecasts or provenance.
No scoring, reconstruction, source access or production caller is provided.
"""
from __future__ import annotations

from bisect import bisect_right
from math import fsum, isfinite

from engine.v2.data.errors import fail

__all__ = ["recipe", "histogram", "validate_distribution", "bin_index"]

_VERSION = "research_magnitude_distribution.v1"
_EDGES = tuple(index / 4 for index in range(81))
_MASS_TOLERANCE = 1e-12


def recipe() -> dict:
    """Return detached numerical semantics, not a fitted/registered recipe.

    Eighty left-closed, right-open finite bins are followed by [20, infinity).
    Zero is in the first bin; this version adds no separate zero atom. All
    magnitudes and empirical excesses are in percentage points, not ratios.
    """
    return {
        "schema_version": _VERSION,
        "target": "absolute_simple_return",
        "unit": "percentage_points",
        "finite_bin_edges": list(_EDGES),
        "finite_bin_policy": "uniform",
        "overflow_policy": "uncapped",
    }


def _refuse() -> None:
    # Fixed, public-safe text: never echo caller values or sample contents.
    raise fail("CONTRACT_MISMATCH", "invalid magnitude distribution")


def _finite(value) -> float:
    if type(value) not in (int, float):
        _refuse()
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _refuse()
    if not isfinite(result) or result < 0:
        _refuse()
    return result


def _fields(value, expected) -> None:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _refuse()
    if set(value) != set(expected):
        _refuse()


def _numbers(values, *, length=None) -> list:
    if type(values) is not list or (length is not None and len(values) != length):
        _refuse()
    for value in values:
        _finite(value)
    return values.copy()


def _tail(value) -> dict:
    _fields(value, ("policy", "sample_count", "excesses"))
    policy = value["policy"]
    if type(policy) is not str or policy not in ("unresolved", "empirical_excess"):
        _refuse()
    excesses = _numbers(value["excesses"])
    count = value["sample_count"]
    if type(count) is not int or count != len(excesses):
        _refuse()
    if (policy == "unresolved" and count != 0) or (policy == "empirical_excess" and count == 0):
        _refuse()
    return {"policy": policy, "sample_count": count, "excesses": excesses}


def validate_distribution(document) -> dict:
    """Validate a closed plain-JSON document and return a detached copy.

    Require 81 finite masses in [0, 1], with absolute sum error <= 1e-12;
    preserve accepted values without normalization, smoothing or clipping.
    An empirical tail is the equal-weight conditional law of 20 + each excess;
    duplicates and their order/count are retained. Zero excess is valid.
    Unresolved tails remain explicit, even for zero overflow probability.

    This validates representation only, not sample eligibility, causality,
    calibration or full-CDF/economic readiness. Malformed inputs refuse with
    CONTRACT_MISMATCH; there is no partial result, I/O, cache or retry.
    """
    expected = recipe()
    _fields(document, (*expected, "probabilities", "tail"))
    for name, value in expected.items():
        actual = document[name]
        if name == "finite_bin_edges":
            if _numbers(actual, length=81) != value:
                _refuse()
        elif type(actual) is not str or actual != value:
            _refuse()
    probabilities = _numbers(document["probabilities"], length=81)
    if any(value > 1 for value in probabilities):
        _refuse()
    if abs(fsum(probabilities) - 1) > _MASS_TOLERANCE:
        _refuse()
    tail = _tail(document["tail"])
    return {**expected, "probabilities": probabilities, "tail": tail}


def histogram(probabilities, *, tail_excesses=None) -> dict:
    """Build a validated document from plain lists; never infer missing classes.

    None means unresolved tail; an explicit empirical tail must be nonempty.
    Samples are nonnegative excesses relative to 20, NOT full magnitudes.
    The caller supplies the tail; this function does not learn/authenticate it.
    Learned source/fold/calibration identities belong to a later admission
    envelope, not this source-free numerical document. No mean or CDF follows
    from overflow probability alone.
    """
    if tail_excesses is None:
        tail = {"policy": "unresolved", "sample_count": 0, "excesses": []}
    else:
        if type(tail_excesses) is not list:
            _refuse()
        tail = {"policy": "empirical_excess", "sample_count": len(tail_excesses),
                "excesses": tail_excesses}
    return validate_distribution({**recipe(), "probabilities": probabilities, "tail": tail})


def bin_index(magnitude) -> int:
    """Map one finite nonnegative magnitude to its bin, retaining overflow.

    A boundary belongs to the bin on its right. Negative signed returns must
    be converted by the target owner; this helper never applies abs or clips.
    Uses comparisons rather than dividing huge magnitudes by the bin width.
    """
    value = _finite(magnitude)
    return bisect_right(_EDGES, value) - 1
