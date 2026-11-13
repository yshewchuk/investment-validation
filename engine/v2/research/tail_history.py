"""Pure tail-history statistics of a ticker's prior absolute earnings moves (SIZE feature group G1)."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

from engine.v2.data.errors import fail

TAIL_HISTORY_COLUMNS = (
    "tail_max_abs",
    "tail_std_abs",
    "tail_p75_abs",
    "tail_p90_abs",
    "tail_share_gt10",
    "tail_max4_abs",
    "tail_std_over_mean",
    "tail_max_over_mean",
)
_BIG_MOVE_PERCENT = 10.0


def _contract(message):
    return fail("CONTRACT_MISMATCH", message)


def _variant(message):
    return fail("EXPERIMENT_VARIANT_FAILED", message)


def _finite(value, message):
    if not math.isfinite(value):
        raise _variant(message)
    return value


def _percentile(ordered, q):
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _validated(abs_moves):
    if isinstance(abs_moves, (str, bytes, bytearray)) or not isinstance(abs_moves, Sequence):
        raise _contract("abs_moves must be a non-string sequence of finite nonnegative real numbers")
    values = []
    for value in abs_moves:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise _contract("abs_moves must contain finite nonnegative real numbers")
        try:
            finite = math.isfinite(value)
            converted = float(value)
        except (TypeError, OverflowError, ValueError):
            raise _contract("abs_moves must contain finite nonnegative real numbers") from None
        if not finite or not math.isfinite(converted) or converted < 0:
            raise _contract("abs_moves must contain finite nonnegative real numbers")
        values.append(converted)
    return values


def compute_tail_history(abs_moves) -> dict[str, float | None]:
    """Summarise a causal own-ticker sequence of prior absolute moves."""
    values = _validated(abs_moves)
    result = {key: None for key in TAIL_HISTORY_COLUMNS}
    n = len(values)
    if n == 0:
        return result
    ordered = sorted(values)
    maximum = ordered[-1]
    result["tail_max_abs"] = _finite(maximum, "tail history max is not finite")
    result["tail_p75_abs"] = _finite(_percentile(ordered, 0.75), "tail history p75 is not finite")
    result["tail_p90_abs"] = _finite(_percentile(ordered, 0.90), "tail history p90 is not finite")
    big = sum(1 for value in values if value > _BIG_MOVE_PERCENT)
    result["tail_share_gt10"] = float(big / n)
    result["tail_max4_abs"] = float(max(values[-4:]))
    total = _finite(sum(values), "tail history sum is not finite")
    mean = _finite(total / n, "tail history mean is not finite")
    if n >= 2:
        squared = sum((value - mean) * (value - mean) for value in values)
        variance = _finite(squared / n, "tail history variance is not finite")
        std = _finite(math.sqrt(variance), "tail history std is not finite")
        result["tail_std_abs"] = float(std)
        if mean > 0:
            result["tail_std_over_mean"] = float(
                _finite(std / mean, "tail history std/mean is not finite")
            )
    if mean > 0:
        result["tail_max_over_mean"] = float(
            _finite(maximum / mean, "tail history max/mean is not finite")
        )
    return result
