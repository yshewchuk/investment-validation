from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from datetime import date

from engine.v2.data.errors import fail

_KEYS = frozenset({"date", "implied_move", "iv10", "iv30", "exern_iv30", "rvol30"})
_FIELDS = ("implied_move", "iv10", "iv30", "exern_iv30", "rvol30")
_OUTPUT_KEYS = (
    "iv10_30", "iv_rv", "ex_ratio", "imp_chg5", "imp_chg10",
    "iv30_chg5", "iv30_chg20", "iv10_chg5",
)
_LAGS = (
    ("imp_chg5", "implied_move", 5),
    ("imp_chg10", "implied_move", 10),
    ("iv30_chg5", "iv30", 5),
    ("iv30_chg20", "iv30", 20),
    ("iv10_chg5", "iv10", 5),
)
_SEQUENCE = "surface_rows must be a sequence of mappings"
_DATE = "surface date is not a canonical ISO date"
_VALUE = "surface value must be a finite real number or None"
_RATIO = "iv dynamics ratio is not finite"


def _contract(message):
    """Raise a CONTRACT_MISMATCH failure with the given message."""
    return fail("CONTRACT_MISMATCH", message)


def _variant(message):
    """Raise an EXPERIMENT_VARIANT_FAILED failure with the given message."""
    return fail("EXPERIMENT_VARIANT_FAILED", message)


def _parse_date(value):
    """Parse a canonical ISO date string, failing on any other value."""
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise _contract(_DATE)
    except (TypeError, ValueError):
        raise _contract(_DATE) from None
    return parsed


def _parse_value(value):
    """Convert a real, non-bool surface value to a finite float, or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise _contract(_VALUE)
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise _contract(_VALUE) from None
    if not math.isfinite(number):
        raise _contract(_VALUE)
    return number


def _row(row):
    """Validate one surface row and return its parsed date and values."""
    if not isinstance(row, Mapping):
        raise _contract("surface_rows row is not a mapping")
    if set(row.keys()) != _KEYS:
        raise _contract("surface row keys do not match the contract")
    return _parse_date(row["date"]), {field: _parse_value(row[field]) for field in _FIELDS}


def _rows(surface_rows):
    """Validate surface_rows and return parsed rows in strictly ascending date order."""
    if isinstance(surface_rows, (str, bytes)) or not isinstance(surface_rows, Sequence):
        raise _contract(_SEQUENCE)
    parsed, previous = [], None
    for row in surface_rows:
        parsed_date, values = _row(row)
        if previous is not None and parsed_date <= previous:
            raise _contract("surface dates must be strictly ascending")
        parsed.append((parsed_date, values))
        previous = parsed_date
    return parsed


def _ratio(numerator, denominator):
    """Divide optional finite values, mapping zero to None and overflow to a variant failure."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    try:
        value = numerator / denominator
    except (OverflowError, ZeroDivisionError, TypeError, ValueError):
        raise _variant(_RATIO) from None
    if not math.isfinite(value):
        raise _variant(_RATIO)
    return value


def _lagged(rows, field, lag, current_value, current_date):
    """Return the current value over the lag-th prior value, or None when absent or stale."""
    if len(rows) < lag + 1:
        return None
    reference_date, reference = rows[-1 - lag]
    if (current_date - reference_date).days > 2 * lag + 4:
        return None
    return _ratio(current_value, reference[field])


def compute_iv_dynamics(surface_rows) -> dict[str, float | None]:
    """Compute same-row and lagged IV ratios from causal surface rows."""
    rows = _rows(surface_rows)
    if not rows:
        return {key: None for key in _OUTPUT_KEYS}
    current_date, current = rows[-1]
    result = {
        "iv10_30": _ratio(current["iv10"], current["iv30"]),
        "iv_rv": _ratio(current["iv30"], current["rvol30"]),
        "ex_ratio": _ratio(current["exern_iv30"], current["iv30"]),
    }
    for key, field, lag in _LAGS:
        result[key] = _lagged(rows, field, lag, current[field], current_date)
    return result
