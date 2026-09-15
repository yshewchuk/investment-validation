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
  read by panel.py's private ``_yf_history_from_tier1`` (panel.py:444-474):
  first column as date (UTC-parsed then tz-dropped), ``Close`` as
  ``close_adj``, dropna, sorted. :func:`read_tier1_body` reimplements that
  parse (not a straight reuse) because ``_yf_history_from_tier1`` always
  builds ``Fetcher()`` off the process-global ``engine.paths.RAW_FETCH`` with
  no root override, which this module must never depend on (same discipline
  ``engine.v2.ops.capture_inputs`` documents for itself). Byte-for-byte parse
  equality against the real ``panel._yf_history_from_tier1`` is pinned by a
  test in ``tests/test_v2_data_price_downloads.py`` that points
  ``engine.paths.RAW_FETCH`` at a fixture via monkeypatch and compares outputs
  on the same underlying body. Tier-1 carries no ``close_raw``/``high_raw``
  (yfinance's own adjusted-close-only series); those columns are ``NaN``, not
  omitted, so every normalized frame shares the same schema regardless of
  source kind -- legacy's own precedence (px over Tier-1) is what
  ``engine.v2.ops.price_history_store.capture`` uses to prefer the fuller row
  when both exist for one ticker (px file takes the whole ticker; Tier-1 is
  only ever consulted when no px file exists).

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
    """Exactly ``panel.add_runup_features``'s own read at panel.py:533."""
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


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


def read_tier1_body(body_bytes: bytes) -> pd.DataFrame:
    """What ``panel._yf_history_from_tier1`` decodes from one decompressed
    Tier-1 body (panel.py:444-474): the module docstring records why this is
    a reimplementation rather than a direct call, and where the parse
    equality against the real function is pinned.
    """
    frame = pd.read_csv(io.BytesIO(body_bytes))
    if frame.empty or "Close" not in frame.columns:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                             "close_adj": pd.Series(dtype="float64")})
    dates = pd.to_datetime(frame[frame.columns[0]], errors="coerce", utc=True)
    dates = dates.dt.tz_localize(None)
    closes = pd.to_numeric(frame["Close"], errors="coerce")
    out = pd.DataFrame({"date": dates, "close_adj": closes})
    return out.dropna().sort_values("date")


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
    ``close_raw``/``high_raw`` are always ``NaN``: Tier-1 never carries them.
    """
    out = frame.copy()
    out["close_raw"] = float("nan")
    out["high_raw"] = float("nan")
    out["ticker"] = ticker
    out["retrieved_at"] = retrieved_at
    out["source_hash"] = source_hash
    return out[list(NORMALIZED_COLUMNS)].reset_index(drop=True)
