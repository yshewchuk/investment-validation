"""Shared normalization primitives: the unit and convention traps, fixed once.

Every trap in this module was found the hard way and is documented in
``engine/data/schemas.py::CONVENTIONS``. They live here rather than in each
consumer because the alternative — every script remembering the ORATS market-cap
era boundaries — is how a feature ends up with a step discontinuity in the
middle of the sample and nobody notices for a year.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

__all__ = [
    "MCAP_ERAS",
    "mcap_to_usd",
    "mcap_era",
    "FLT_MAX_THRESHOLD",
    "mask_sentinels",
    "read_gz_json",
    "PLAUSIBLE_RANGES",
    "clip_implausible",
]

# --------------------------------------------------------------------------
# ORATS market-cap units — THREE eras, not two
# --------------------------------------------------------------------------

#: ``(start_date_inclusive, multiplier_to_usd, label)``, oldest first.
#:
#: Verified on 300 randomly sampled tickers: on 2017-06-28 every ticker with
#: history spanning that date (148/300) jumps by a median factor of 990, and on
#: 2026-03-11 all 300 jump by a median factor of 997. Spot-checked against known
#: values — AAPL 2007-01-03 reads ``72`` at a $83.80 share price, which is $72B,
#: not $72M.
#:
#: The legacy master panel applied ×1e6 across everything before 2026-03-11, so
#: its ``or_mcap_log`` is understated by ``log(1000)`` for every event before
#: 2017-06-28. That is roughly half the panel, and it is a step discontinuity in
#: a feature the size model and the mcap-slice work both consume.
MCAP_ERAS: tuple[tuple[pd.Timestamp, float, str], ...] = (
    (pd.Timestamp("1900-01-01"), 1e9, "billions"),
    (pd.Timestamp("2017-06-28"), 1e6, "millions"),
    (pd.Timestamp("2026-03-11"), 1e3, "thousands"),
)

#: Below this, a pre-2017 integer market cap has rounded away most of its
#: precision (a $1.4B name reads as ``1``), so slice work on small caps in that
#: era is quantized rather than merely noisy.
MCAP_QUANTIZED_BEFORE = pd.Timestamp("2017-06-28")


def mcap_era(dates) -> np.ndarray:
    """Era label per date."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    out = np.full(len(d), MCAP_ERAS[0][2], dtype=object)
    for start, _, label in MCAP_ERAS:
        out[d >= np.datetime64(start)] = label
    return out


def mcap_to_usd(raw, dates) -> np.ndarray:
    """Convert ORATS ``mktCap`` to true USD using the three-era rule.

    Non-positive and non-finite values become NaN: ORATS emits zeros for names
    it has no size for, and a zero market cap would otherwise become
    ``log(0) = -inf`` downstream.
    """
    values = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    mult = np.full(len(values), MCAP_ERAS[0][1], dtype=float)
    for start, factor, _ in MCAP_ERAS:
        mult[d >= np.datetime64(start)] = factor
    usd = values * mult
    return np.where(np.isfinite(usd) & (values > 0), usd, np.nan)


# --------------------------------------------------------------------------
# ORATS missing-value sentinel
# --------------------------------------------------------------------------

#: ORATS encodes "missing" as FLT_MAX (~3.4e38) rather than null, in about
#: 0.097% of numeric cells. Left in place it poisons any mean, z-score, or
#: model fit that touches the column.
FLT_MAX_THRESHOLD = 1e30


def mask_sentinels(values) -> np.ndarray:
    """Replace FLT_MAX-style sentinels and infinities with NaN."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return np.where(np.abs(arr) >= FLT_MAX_THRESHOLD, np.nan, arr)


#: Ranges outside which a value is not a real quote. Applied after sentinel
#: masking, and recorded in the validation report rather than silently dropped.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "implied_move": (0.0, 100.0),
    "iv10": (0.0, 500.0),
    "iv30": (0.0, 500.0),
    "exern_iv10": (0.0, 500.0),
    "exern_iv30": (0.0, 500.0),
    "rvol30": (0.0, 500.0),
    "skew": (-50.0, 50.0),
    "contango": (-100.0, 100.0),
    "fwd90_30": (0.0, 500.0),
    "fexern90_30": (0.0, 500.0),
    "iee": (-50.0, 50.0),
    "spot": (0.0, 1e6),
}


def clip_implausible(df: pd.DataFrame, ranges: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """NaN out values outside their plausible range; report how many, per column."""
    ranges = ranges or PLAUSIBLE_RANGES
    out = df.copy()
    counts: dict[str, int] = {}
    for col, (lo, hi) in ranges.items():
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        bad = series.notna() & ((series < lo) | (series > hi))
        n = int(bad.sum())
        if n:
            counts[col] = n
            out.loc[bad, col] = np.nan
    return out, counts


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def read_gz_json(path: Path) -> Any:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)
