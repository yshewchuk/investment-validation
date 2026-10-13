"""Read-only normalization of the two legacy yfinance price sources into one
frame shape (module docstring of ``engine.v2.data.price_history`` explains
the 2026-09-14 design settlement -- a normalized bitemporal ``price_history``
Tier-2 catalog table -- and why the original whole-download ``price_downloads``
module it superseded was removed).

Target shape: ``(ticker, date, close_adj, close_raw, high_raw, retrieved_at,
source_hash)`` -- the coordinator's normalized-table columns. Every function
but one opens legacy files for reading only; :func:`write_legacy_px_csv` is
the sole writer, added 2026-09-14 for ``legacy_materialization.
materialize_price_series`` to call without spending its own fan-out budget
on ``pandas``.

Two sources, both read by ``engine.data.features.panel.add_runup_features``:

* **Grandfathered px csv** -- ``earnings_predictions/data/raw/yfinance/px_<T>.csv``,
  read at panel.py:533 as ``pd.read_csv(path, parse_dates=["date"]).sort_values("date")``.
  :func:`read_legacy_px_csv` calls that exact same pandas expression -- reused,
  not reimplemented, so there is no way for it to drift from what panel.py
  itself reads. Columns: ``date, close_adj, close_raw, high_raw``. This source
  has no embedded fetch timestamp; the ops-layer scan (see the layering note
  below) reports each file's mtime as the only available ``retrieved_at``
  proxy.
* **Tier-1 fallback** -- ``Fetcher(root).body_path("yfinance", key)`` with
  ``key = cache_key("yfinance", "history", {"ticker": T, "period": "max"})``,
  a CSV body from ``engine/data/sources/yf.py``'s
  ``yfinance.Ticker.history(auto_adjust=False, actions=False)``: columns
  ``Date, Open, High, Low, Close, Adj Close, Volume``. :func:`read_tier1_body`
  parses ``close_adj = Adj Close`` (dividend-adjusted), ``close_raw = Close``
  (split-adjusted only), ``high_raw = High`` -- **not** a reuse of legacy's
  own ``panel._yf_history_from_tier1`` (panel.py:444-474), which maps
  ``Close`` to ITS OWN ``close_adj`` for a different purpose (run-up features,
  where only split-adjustment matters, per that function's own docstring) and
  never reads ``high``/dividend-adjusted close at all. Defect found on the
  first real shadow capture (2026-09-15): an earlier version of this function
  copied that legacy shape verbatim, so every Tier-1 retrieval silently
  restated ``close_adj`` under the wrong quantity and left ``close_raw``/
  ``high_raw`` all-``NaN`` for every Tier-1-sourced ticker (2,814 of them on
  the real shadow capture). Fixed 2026-09-15; the price_history table
  contract was bumped to ``price_history.v2`` alongside the fix (see
  ``price_history_table.py``'s module docstring) so a fixed parser is never
  silently blocked from recapturing by an already-recorded ``source_hash``
  from the wrongly-parsed generation. Refuses ``CONTRACT_MISMATCH`` typed if
  ``Close``/``Adj Close``/``High`` is missing from the body's header, or if
  any of the three parses entirely ``NaN`` while the body has rows (a
  completeness guard against exactly this class of silent mis-mapping,
  applied here rather than in the ops layer since ``read_tier1_body`` is
  where the raw, pre-``dropna`` column values are still in hand). Tier-1
  DOES carry ``close_raw``/``high_raw`` now -- the "Tier-1 never carries
  them" claim this module's docstrings made before 2026-09-15 was itself part
  of the same defect and is retracted here; every normalized frame still
  shares :data:`NORMALIZED_COLUMNS`, now with real values in all three
  columns from either source.

**Layering note.** This module is ``engine.v2.data`` (Layer 1) and must never
import ``engine.v2.ops`` (``system_rearchitecture.md`` Sec 4.1). Enumerating
which files/cache entries exist under a source root needs
``engine.v2.ops.legacy_adapter.iter_raw_fetch_cache`` for the Tier-1 side, so
that enumeration lives in the ops layer instead:
``engine.v2.ops.price_history_store``'s private ``_scan_legacy_px_tree``/
``_tier1_retrievals``. Only pure parsing (bytes/paths already in hand) lives
here.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from . import errors

__all__ = [
    "NORMALIZED_COLUMNS",
    "normalize_px_csv",
    "normalize_tier1_body",
    "read_legacy_px_csv",
    "read_tier1_body",
    "write_legacy_px_csv",
]

NORMALIZED_COLUMNS = ("ticker", "date", "close_adj", "close_raw", "high_raw",
                     "retrieved_at", "source_hash")


# --------------------------------------------------------------------------
# parsing -- exact legacy read shapes
# --------------------------------------------------------------------------


def read_legacy_px_csv(path: Path) -> pd.DataFrame:
    """Exactly ``panel.add_runup_features``'s own read at panel.py:533,
    ``float_precision="round_trip"`` included: pandas' default C-parser is
    not a true round trip for every float64 (measured ~1 ULP off, common
    enough on real close_adj values that nearly every ticker in the real
    shadow snapshot hit it on at least one row) -- panel.py carries the
    same kwarg, so this stays a faithful mirror of the real reader rather
    than a looser stand-in for it."""
    return pd.read_csv(path, parse_dates=["date"],
                       float_precision="round_trip").sort_values("date")


def write_legacy_px_csv(path: Path, rows) -> pd.DataFrame:
    """Write ``rows`` (any iterable of objects carrying ``date``, ``close_adj``,
    ``close_raw``, ``high_raw`` attributes -- a ``PriceSeriesRow`` sequence in
    practice) as a legacy-shaped px csv at ``path``, and return the
    ``DataFrame`` actually written, for a caller's own read-back check
    against :func:`read_legacy_px_csv`. The one writing counterpart to this
    module's otherwise read-only parsing (task brief 2026-09-14):
    ``engine.v2.data.legacy_materialization.materialize_price_series`` is the
    only caller, kept here rather than there purely to stay within
    ``legacy_materialization.py``'s own §4.3 fan-out budget -- this module
    already depends on ``pandas``.
    """
    frame = pd.DataFrame([{"date": row.date, "close_adj": row.close_adj,
                           "close_raw": row.close_raw, "high_raw": row.high_raw}
                          for row in rows])
    frame.to_csv(path, index=False)
    return frame


_TIER1_REQUIRED_COLUMNS = ("Close", "Adj Close", "High")


def read_tier1_body(body_bytes: bytes) -> pd.DataFrame:
    """Decode one decompressed Tier-1 ``yfinance``/``history`` CSV body (the
    real shape: ``Date, Open, High, Low, Close, Adj Close, Volume`` -- see the
    module docstring's "Tier-1 fallback" section) into
    ``(date, close_adj, close_raw, high_raw)``: ``close_adj = Adj Close``
    (dividend-adjusted), ``close_raw = Close`` (split-adjusted only),
    ``high_raw = High``.

    Refuses ``CONTRACT_MISMATCH`` typed if any of ``Close``/``Adj Close``/
    ``High`` is missing from the header, or if any of the three parses
    entirely ``NaN`` (via ``pd.to_numeric(..., errors="coerce")``) while the
    body has rows -- a completeness guard against a repeat of the 2026-09-15
    defect (a wrong column mapped in, or a source format change), rather than
    scattered per-row missing values, which are dropped normally below.
    """
    frame = pd.read_csv(io.BytesIO(body_bytes))
    if frame.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                             "close_adj": pd.Series(dtype="float64"),
                             "close_raw": pd.Series(dtype="float64"),
                             "high_raw": pd.Series(dtype="float64")})
    missing = [c for c in _TIER1_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise errors.fail("CONTRACT_MISMATCH",
                          "Tier-1 yfinance body is missing required column(s)",
                          details={"missing_columns": missing})
    dates = pd.to_datetime(frame[frame.columns[0]], errors="coerce", utc=True)
    dates = dates.dt.tz_localize(None)
    close_adj = pd.to_numeric(frame["Adj Close"], errors="coerce")
    close_raw = pd.to_numeric(frame["Close"], errors="coerce")
    high_raw = pd.to_numeric(frame["High"], errors="coerce")
    all_nan = [name for name, series in
              (("close_adj", close_adj), ("close_raw", close_raw), ("high_raw", high_raw))
              if series.isna().all()]
    if all_nan:
        raise errors.fail("CONTRACT_MISMATCH",
                          "Tier-1 column(s) parsed entirely NaN with a non-empty source body",
                          details={"all_nan_columns": all_nan, "source_row_count": len(frame)})
    out = pd.DataFrame({"date": dates, "close_adj": close_adj, "close_raw": close_raw,
                        "high_raw": high_raw})
    return out.dropna(subset=["date", "close_adj"]).sort_values("date")


# --------------------------------------------------------------------------
# normalization -- one shared frame shape for either source kind
# --------------------------------------------------------------------------


def normalize_px_csv(ticker: str, frame: pd.DataFrame, *, retrieved_at: str,
                     source_hash: str) -> pd.DataFrame:
    """``read_legacy_px_csv``'s output, reshaped to :data:`NORMALIZED_COLUMNS`."""
    out = frame.copy()
    for column in ("close_adj", "close_raw", "high_raw"):
        if column not in out.columns:
            out[column] = float("nan")
    out["ticker"] = ticker
    out["retrieved_at"] = retrieved_at
    out["source_hash"] = source_hash
    return out[list(NORMALIZED_COLUMNS)].reset_index(drop=True)


def normalize_tier1_body(ticker: str, frame: pd.DataFrame, *, retrieved_at: str,
                         source_hash: str) -> pd.DataFrame:
    """``read_tier1_body``'s output, reshaped to :data:`NORMALIZED_COLUMNS`.
    ``close_raw``/``high_raw`` carry ``read_tier1_body``'s real parsed values
    (``Close``/``High``) -- fixed 2026-09-15; they are no longer forced NaN.
    """
    out = frame.copy()
    out["ticker"] = ticker
    out["retrieved_at"] = retrieved_at
    out["source_hash"] = source_hash
    return out[list(NORMALIZED_COLUMNS)].reset_index(drop=True)
