"""Normalize ORATS daily summaries + cores into Tier-2 ``daily_market``.

Source: ``earnings_predictions/data/raw/orats/{summaries,cores}/{TICKER}.json.gz``,
2,936 tickers back to 2007 — about 9.4M rows, so the build streams per ticker
and flushes per year rather than materializing the table.

Unit handling, all applied here and nowhere else:

* IV-like fields arrive as decimals (``0.407453``) and are stored as vol points
  (``40.7453``), matching the panel convention every existing model was fit on.
* ``mktCap`` goes through the three-era conversion in
  :func:`~engine.data.normalize.common.mcap_to_usd`.
* FLT_MAX sentinels are masked before anything else touches the numbers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from engine import paths
from engine.data.normalize.common import (
    clip_implausible,
    mask_sentinels,
    mcap_to_usd,
    read_gz_json,
)

__all__ = [
    "SUMMARY_FIELDS",
    "normalize_ticker",
    "iter_normalized",
    "list_tickers",
    "fetch_daily_index",
    "normalize_frames",
    "count_daily_dates",
    "history_body_path",
    "read_history_body",
    "SOURCE_COLUMNS",
]

#: ``raw ORATS key -> (tier-2 column, multiplier)``. The multiplier turns
#: decimals into vol points / percent, which is the panel's unit.
SUMMARY_FIELDS: dict[str, tuple[str, float]] = {
    "stockPrice": ("spot", 1.0),
    "iv10d": ("iv10", 100.0),
    "iv30d": ("iv30", 100.0),
    "exErnIv10d": ("exern_iv10", 100.0),
    "exErnIv30d": ("exern_iv30", 100.0),
    "impliedMove": ("implied_move", 100.0),
    "rVol30": ("rvol30", 100.0),
    "skewing": ("skew", 1.0),
    "contango": ("contango", 1.0),
    "fwd90_30": ("fwd90_30", 100.0),
    "fexErn90_30": ("fexern90_30", 100.0),
    "ieeEarnEffect": ("iee", 1.0),
}


#: The raw columns :func:`normalize_frames` reads, per kind.
SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "summaries": ("tradeDate", *SUMMARY_FIELDS),
    "cores": ("tradeDate", "mktCap"),
}


def list_tickers(root: Path | None = None, *, include_fetch: bool = True) -> list[str]:
    """Every ticker with daily data, from the legacy trees and the fetch store."""
    root = root or paths.RAW_ORATS_SUMMARIES
    tickers = set()
    if root.exists():
        tickers |= {p.name[: -len(".json.gz")] for p in root.glob("*.json.gz")}
    if include_fetch:
        tickers |= set(fetch_daily_index())
    return sorted(tickers)


class _CompactRows:
    """Rows stored by column, expanded into dicts only when a caller iterates.

    One value list per field replaces one dict per row; missing fields use a
    private sentinel so a sparse response stays sparse.
    """

    __slots__ = ("_columns", "_n")
    _MISSING = object()

    def __init__(self) -> None:
        """Start an empty column store."""
        self._columns: dict[str, list] = {}
        self._n = 0

    def append(self, row: dict) -> None:
        """Append one row, adding any columns it is the first to carry."""
        columns = self._columns
        for key, column in columns.items():
            column.append(row.get(key, self._MISSING))
        for key, value in row.items():
            if key not in columns:
                columns[key] = [self._MISSING] * self._n + [value]
        self._n += 1

    def freeze(self) -> "_CompactRows":
        """Store each column as a tuple once appends are finished."""
        self._columns = {key: tuple(values) for key, values in self._columns.items()}
        return self

    def __len__(self) -> int:
        """Number of stored rows."""
        return self._n

    def __iter__(self) -> Iterator[dict]:
        """Expand the columns back into rows, omitting missing fields."""
        columns = self._columns
        missing = self._MISSING
        for i in range(self._n):
            yield {
                key: values[i]
                for key, values in columns.items()
                if values[i] is not missing
            }


@lru_cache(maxsize=1)
def fetch_daily_index() -> dict[str, dict[str, _CompactRows]]:
    """Index the fetch store's MARKET-WIDE summaries/cores rows by ticker.

    On a fresh machine this is the *only* source of daily data: the legacy
    per-ticker trees do not exist, and the restore path re-pulls through the
    fetch wrapper. Without this bridge a restored machine rebuilds empty tables,
    which is the failure the recovery drill could not see (it checks imports and
    the no-data test subset, not a full rebuild).

    **Per-ticker responses are deliberately NOT indexed here.** ORATS serves a
    ticker's entire history — ~4,900 rows — from one ``?ticker=X`` call, and
    once the nightly began backfilling those, indexing every row of every
    response meant holding ~1.8M dicts in memory: the rebuild reached 5 GB RSS
    in 28 seconds and was killed. Those responses do not need an index at all,
    because they are addressable by cache key: :func:`_ticker_history_rows`
    loads exactly the one being asked for. The index is left holding only the
    market-wide (``tradeDate``) responses, which are a few thousand rows each.
    Each is stored compactly by column in :class:`_CompactRows` and expands to
    row dicts only for the requested ticker.

    Cached because a rebuild asks for it once per ticker.

    Snapshots are indexed oldest-fetched first. Among market-wide snapshots,
    later fetch time sets overlap precedence: a later-fetched response is
    appended after the ones it supersedes. That ordering says nothing about
    whether the later response is more complete.
    :func:`_rows_for` appends the legacy per-ticker rows before these, then the
    per-ticker history after them; normalization's ``keep="last"`` therefore
    lets the history win any overlap across all sources.
    """
    from engine.data.normalize.fetch_store import iter_orats_rows

    index: dict[str, dict[str, _CompactRows]] = {}
    for endpoint, kind in (("hist/summaries", "summaries"), ("hist/cores", "cores")):
        for source in iter_orats_rows(endpoint, order_by_fetched_at=True):
            # A response fetched for ONE ticker is fetched back on demand.
            if (source.params or {}).get("ticker"):
                continue
            for row in source.rows:
                ticker = row.get("ticker")
                if not ticker:
                    continue
                key = str(ticker)
                parts = index.get(key)
                if parts is None:
                    parts = {"summaries": _CompactRows(), "cores": _CompactRows()}
                    index[key] = parts
                parts[kind].append(row)
    for parts in index.values():
        for rows in parts.values():
            rows.freeze()
    return index


def _ticker_history_rows(ticker: str, kind: str) -> list[dict]:
    """The cached ``?ticker=X`` response for one ticker, or nothing.

    A direct key lookup rather than a scan: the Tier-1 key is a hash of
    (source, endpoint, params), so the entry for this exact request is found
    without touching any other. This is what keeps a rebuild's memory flat as
    per-ticker history accumulates.
    """
    path = history_body_path(ticker, kind)
    if not path.exists():
        return []
    rows = read_history_body(path)
    return [] if rows is None else rows


def history_body_path(ticker: str, kind: str) -> Path:
    """Where the cached ``?ticker=X`` response for one ticker would live."""
    from engine.data.fetch import Fetcher, cache_key

    endpoint = {"summaries": "hist/summaries", "cores": "hist/cores"}[kind]
    key = cache_key("orats", endpoint, {"ticker": str(ticker)})
    return Fetcher().body_path("orats", key)


def read_history_body(path: Path) -> list | None:
    """The rows of one per-ticker history body, or ``None`` when unreadable."""
    import gzip
    import json as _json

    try:
        with gzip.open(path, "rb") as fh:
            payload = _json.loads(fh.read())
    except (OSError, EOFError, ValueError):
        return None
    rows = payload.get("data") if isinstance(payload, dict) else payload
    return list(rows or [])


def _rows_for(ticker: str, kind: str, directory: Path) -> list[dict]:
    """Legacy per-ticker file plus anything the fetch store holds, unioned.

    Order defines precedence because the caller collapses duplicate trade dates
    with ``keep="last"``. This appends, in turn: the legacy per-ticker file, the
    market-wide snapshots (oldest fetched first, so the newest response is last
    and wins), then the per-ticker history pull. Both generations are kept: the
    legacy tree is the historical bulk and the fetch store is everything pulled
    since, so a daily refresh extends the series rather than replacing it.
    """
    rows: list[dict] = []
    path = directory / f"{ticker}.json.gz"
    if path.exists():
        rows.extend(read_gz_json(path) or [])
    rows.extend(fetch_daily_index().get(ticker, {}).get(kind, []))
    # The per-ticker history pull, loaded by key rather than through the index.
    rows.extend(_ticker_history_rows(ticker, kind))
    return rows


def normalize_ticker(
    ticker: str,
    *,
    summaries_dir: Path | None = None,
    cores_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return one ticker's ``daily_market`` rows plus a small quality report."""
    summaries_dir = summaries_dir or paths.RAW_ORATS_SUMMARIES
    cores_dir = cores_dir or paths.RAW_ORATS_CORES

    rows = _rows_for(ticker, "summaries", summaries_dir)
    if not rows:
        return pd.DataFrame(), {"ticker": ticker, "reason": "no summaries rows"}

    def cores_frame() -> pd.DataFrame | None:
        crows = _rows_for(ticker, "cores", cores_dir)
        return pd.DataFrame(crows) if crows else None

    return normalize_frames(ticker, pd.DataFrame(rows), cores_frame)


def normalize_frames(ticker: str, src: pd.DataFrame, cores_frame) -> tuple[pd.DataFrame, dict]:
    """:func:`normalize_ticker` from the raw summaries frame onward.

    ``src`` is the summaries rows as a frame (non-empty); ``cores_frame`` is a
    zero-argument callable returning the cores rows as a frame, or ``None``
    when there are none. Only the columns named in :data:`SUMMARY_FIELDS`,
    ``tradeDate`` and ``mktCap`` are read, which is what lets the incremental
    rebuild hand over a frame assembled from per-file parses.
    """
    if "tradeDate" not in src.columns:
        return pd.DataFrame(), {"ticker": ticker, "reason": "no tradeDate column"}

    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(src["tradeDate"], errors="coerce"),
        }
    )
    for raw_key, (col, mult) in SUMMARY_FIELDS.items():
        if raw_key in src.columns:
            out[col] = mask_sentinels(src[raw_key]) * mult
        else:
            out[col] = np.nan

    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    out["src_iv"] = "orats.summaries"

    # -- cores: market cap, three-era conversion --------------------------
    #
    # The two ORATS series do not share a date grid, and for some tickers they
    # barely overlap at all: ABX, RTX and META carry cores history back to 2007
    # while their summaries start in 2026 (reused or renamed symbols). Keying
    # the table on summaries alone would throw away nearly twenty years of
    # market cap for those names, so the row index is the UNION of both series
    # and each block keeps its own source column.
    mcap_rows = 0
    cores_only_rows = 0
    core = None
    candidate = cores_frame()
    if candidate is not None:
        if {"tradeDate", "mktCap"} <= set(candidate.columns):
            candidate = candidate.assign(
                date=pd.to_datetime(candidate["tradeDate"], errors="coerce")
            )
            candidate = candidate.dropna(subset=["date"]).drop_duplicates("date", keep="last")
            candidate["mcap_usd"] = mcap_to_usd(candidate["mktCap"], candidate["date"])
            core = candidate.dropna(subset=["mcap_usd"]).sort_values("date")

    if core is not None and len(core):
        extra_dates = core.loc[~core["date"].isin(set(out["date"])), ["date"]]
        cores_only_rows = int(len(extra_dates))
        if cores_only_rows:
            out = pd.concat([out, extra_dates.assign(ticker=ticker)], ignore_index=True)
        out = out.sort_values("date").reset_index(drop=True)
        # As-of (backward) rather than exact: every row carries the most recent
        # market cap known on that trade date. Strictly backward looking, so it
        # cannot leak.
        out = pd.merge_asof(
            out,
            core[["date", "mcap_usd"]].rename(columns={"date": "mcap_asof"}),
            left_on="date",
            right_on="mcap_asof",
            direction="backward",
        )
        # How old the carried figure is. An unbounded backward join will happily
        # carry a delisted name's last cap forward for years, and nothing
        # downstream could tell that from a fresh observation. Recording the
        # age keeps the staleness visible to the analog buckets that consume it
        # rather than silently baked into a size feature.
        out["mcap_age_days"] = (out["date"] - out["mcap_asof"]).dt.days
        out.loc[out["mcap_usd"].isna(), "mcap_age_days"] = np.nan
        mcap_rows = int(out["mcap_usd"].notna().sum())
    for column in ("mcap_usd", "mcap_asof", "mcap_age_days"):
        if column not in out.columns:
            out[column] = pd.NaT if column == "mcap_asof" else np.nan
    out = out.sort_values("date").reset_index(drop=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["mcap_log"] = np.where(out["mcap_usd"] > 0, np.log(out["mcap_usd"]), np.nan)

    # oquants implied moves before 2022 are reconstructions; the ORATS
    # impliedMove used here is a quoted figure for its whole history, so the
    # flag is False rather than absent — the column means "this number is a
    # reconstruction", and for this source it never is.
    out["implied_reconstructed"] = False
    out["year"] = out["date"].dt.year
    out["src_spot"] = out["src_iv"]
    out["src_mcap"] = np.where(out["mcap_usd"].notna(), "orats.cores", None)

    out, clipped = clip_implausible(out)

    # ORATS writes `impliedMove == 0` to mean "no quote", not "no expected
    # movement": zero outnumbered true null 119:1 across 9.0M rows (25.8%), in
    # every year since 2009, and it is a LIQUIDITY fact — 31.5% zero at $0.2B
    # against 1.6% at $82B (EXP-110, EXP-111). Left in, it tells a model the
    # market expects nothing on precisely the events that move most: mean
    # realized |move| 6.87% where it is zero against 5.75% where it is not.
    #
    # This is a sentinel, not an implausible value, so it is masked here rather
    # than by widening `clip_implausible`'s range — the range says what a real
    # quote can be, and zero is not a quote at all.
    #
    # EXP-111 landed `has_implied_quote` FIRST, on purpose: the absence is real
    # information and nulling the magic number without preserving it would have
    # deleted a signal the models were using.
    implied = pd.to_numeric(out.get("implied_move"), errors="coerce")
    sentinel = implied.notna() & (implied <= 0)
    n_sentinel = int(sentinel.sum())
    if n_sentinel:
        out.loc[sentinel, "implied_move"] = np.nan

    report = {
        "ticker": ticker,
        "rows": int(len(out)),
        "mcap_rows": mcap_rows,
        "cores_only_rows": cores_only_rows,
        "clipped": clipped,
        "implied_sentinel_nulled": n_sentinel,
        "first": str(out["date"].min().date()) if len(out) else None,
        "last": str(out["date"].max().date()) if len(out) else None,
    }
    return out, report


#: Raw summary fields that are the clipping and validation inputs for the
#: count. Sentinel masking and clipping can turn a value into a missing one,
#: which validation accepts; this set names what the gate reads, not what is
#: guaranteed to invalidate a row.
_COUNT_FIELDS = ("stockPrice", "iv30d", "impliedMove", "rVol30")


def count_daily_dates(
    ticker: str, src: pd.DataFrame, cores_frame
) -> tuple[set, dict]:
    """The published dates and validated row counts for one ticker.

    Count-only counterpart of :func:`normalize_frames`: same date parsing,
    duplicate collapse, cores-only union, sentinel masking, clipping and
    :func:`~engine.data.validate.validate_daily` gate, but materializes only the
    columns that gate reads. Returns ``(present, counts)``: ``present`` includes
    every normalized date, even one whose rows all fail validation or whose last
    duplicate is invalid; ``counts`` has an entry only for dates with at least
    one accepted row.
    """
    from engine.data.validate import validate_daily

    if "tradeDate" not in src.columns:
        return set(), {}

    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(src["tradeDate"], errors="coerce"),
        }
    )
    for raw_key in _COUNT_FIELDS:
        col, mult = SUMMARY_FIELDS[raw_key]
        if raw_key in src.columns:
            out[col] = mask_sentinels(src[raw_key]) * mult
        else:
            out[col] = np.nan
    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last")

    # Cores-only dates join the grid exactly as in normalize_frames: the union
    # keeps the later row and needs a usable market cap.
    candidate = cores_frame()
    if candidate is not None and {"tradeDate", "mktCap"} <= set(candidate.columns):
        candidate = candidate.assign(
            date=pd.to_datetime(candidate["tradeDate"], errors="coerce")
        )
        candidate = candidate.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        candidate = candidate.reset_index(drop=True)
        mcap = mcap_to_usd(candidate["mktCap"], candidate["date"])
        usable = pd.Series(mcap, index=candidate.index).notna()
        extra = candidate.loc[usable, ["date"]]
        extra = extra[~extra["date"].isin(set(out["date"]))]
        if len(extra):
            out = pd.concat([out, extra.assign(ticker=ticker)], ignore_index=True)
            out = out.sort_values("date").reset_index(drop=True)

    out, _ = clip_implausible(out)
    implied = pd.to_numeric(out.get("implied_move"), errors="coerce")
    sentinel = implied.notna() & (implied <= 0)
    if sentinel.any():
        out.loc[sentinel, "implied_move"] = np.nan

    present = set(out["date"])
    clean, _ = validate_daily(out)
    return present, clean.groupby("date").size().to_dict()


def iter_normalized(
    tickers: Iterable[str] | None = None, **kwargs
) -> Iterator[tuple[pd.DataFrame, dict]]:
    """Stream ``(frame, report)`` per ticker, in sorted order (deterministic)."""
    for ticker in sorted(tickers) if tickers is not None else list_tickers():
        yield normalize_ticker(ticker, **kwargs)
