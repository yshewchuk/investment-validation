"""Strict-JSON normalization, in one place.

Two parts of the program write JSON that other software has to read back: the
dashboard bundle (a phone, over a tunnel) and the prediction ledger (append-only,
therefore uncorrectable). Both are fed from DataFrames, and a DataFrame has no
``None`` — every missing value is ``nan`` or ``NaT``, every integer in a column
with a gap is a float, and every scalar is a numpy type. ``json.dumps`` writes
``NaN`` for the first of those without complaint: valid JavaScript, invalid JSON,
and a file a strict parser is entitled to reject.

So the conversion happens once, here, rather than being re-derived per writer —
two subtly different sanitizers is how one of them ends up missing a case.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["json_safe"]


def json_safe(value: Any, *, round_to: int | None = None) -> Any:
    """Recursively make ``value`` strict-JSON writable.

    * missing (``None``, ``nan``, ``NaT``, ``pd.NA``, ``inf``) → ``None``;
    * numpy scalars → their Python equivalents;
    * ``Timestamp`` → an ISO date string;
    * everything else is returned as-is.

    ``round_to`` rounds floats to that many places. The dashboard rounds (a
    board shows six figures at most, and rounding keeps the bundle small); the
    ledger does not, because a frozen prediction is evidence and rounding it
    would discard precision nothing can recover later.
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        if not np.isfinite(out):
            return None
        return round(out, round_to) if round_to is not None else out
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, np.ndarray):
        return [json_safe(v, round_to=round_to) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe(v, round_to=round_to) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, round_to=round_to) for v in value]
    return value
