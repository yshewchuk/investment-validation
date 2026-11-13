"""Causal own-volatility features for one adjusted-close series (design row G3)."""
from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from datetime import date

from engine.v2.data.errors import fail

OWN_VOL_COLUMNS = ("own_rvol5", "own_rvol10", "own_rvol60", "own_absret1", "own_rvol5_60")

_WINDOW = 61
_NONFINITE = "own volatility is not finite"
_SEQUENCE = "closes must be a sequence of (iso_date, close) pairs"
_PAIR = "each close row must be a two-item sequence"
_DATE = "close dates must be canonical ISO date strings"
_ORDER = "close dates must be strictly ascending"
_CAUSAL = "close date is after the decision ceiling"
_CLOSE = "close must be a finite real number"
_CEILING = "ceiling must be a canonical ISO date"


def _contract(message):
    """Raise a CONTRACT_MISMATCH failure with the given message."""
    return fail("CONTRACT_MISMATCH", message)


def _variant(message):
    """Raise an EXPERIMENT_VARIANT_FAILED failure with the given message."""
    return fail("EXPERIMENT_VARIANT_FAILED", message)


def _finite(value):
    """Return value if finite, otherwise raise a variant failure."""
    if not math.isfinite(value):
        raise _variant(_NONFINITE)
    return value


def _parse_date(value, message):
    """Parse a canonical ISO date string or refuse it as a contract failure."""
    if not isinstance(value, str):
        raise _contract(message)
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise _contract(message) from None
    if parsed.isoformat() != value:
        raise _contract(message)
    return parsed


def _parse_close(value):
    """Convert a finite real close to float, refusing bools and non-finite values."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise _contract(_CLOSE)
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise _contract(_CLOSE) from None
    if not math.isfinite(number):
        raise _contract(_CLOSE)
    return number


def _rows(closes, ceiling):
    """Validate close pairs and return parsed (date, close) rows."""
    if isinstance(closes, (str, bytes)) or not isinstance(closes, Sequence):
        raise _contract(_SEQUENCE)
    rows, previous = [], None
    for item in closes:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise _contract(_PAIR)
        parsed = _parse_date(item[0], _DATE)
        if previous is not None and parsed <= previous:
            raise _contract(_ORDER)
        if parsed > ceiling:
            raise _contract(_CAUSAL)
        rows.append((parsed, _parse_close(item[1])))
        previous = parsed
    return rows


def _log_returns(prices):
    """Log returns between consecutive closes, refusing non-finite math."""
    returns = []
    for previous, current in zip(prices, prices[1:]):
        try:
            returns.append(math.log(current / previous))
        except (OverflowError, ValueError, ZeroDivisionError):
            raise _variant(_NONFINITE) from None
    return returns


def _rvol(returns, window):
    """Annualized sample volatility over the last window returns, or None."""
    if len(returns) < window:
        return None
    sample = returns[-window:]
    try:
        mean = sum(sample) / window
        variance = sum((value - mean) ** 2 for value in sample) / (window - 1)
        value = math.sqrt(variance) * math.sqrt(252) * 100
    except (OverflowError, ValueError, ZeroDivisionError):
        raise _variant(_NONFINITE) from None
    return _finite(value)


def compute_own_vol(closes, ceiling) -> dict[str, float | None]:
    """Compute the G3 own-volatility columns for one causal close series."""
    ceiling_date = _parse_date(ceiling, _CEILING)
    rows = _rows(closes, ceiling_date)
    result = {name: None for name in OWN_VOL_COLUMNS}
    if not rows or (ceiling_date - rows[-1][0]).days > 5:
        return result
    window = rows[-_WINDOW:]
    if any(close <= 0 for _, close in window):
        return result
    returns = _log_returns([close for _, close in window])
    values = {size: _rvol(returns, size) for size in (5, 10, 60)}
    absret = _finite(abs(returns[-1]) * 100) if returns else None
    ratio = None
    if values[5] is not None and values[60] is not None and values[60] != 0:
        try:
            ratio = values[5] / values[60]
        except (OverflowError, ZeroDivisionError):
            raise _variant(_NONFINITE) from None
        _finite(ratio)
    return {
        "own_rvol5": values[5],
        "own_rvol10": values[10],
        "own_rvol60": values[60],
        "own_absret1": absret,
        "own_rvol5_60": ratio,
    }
