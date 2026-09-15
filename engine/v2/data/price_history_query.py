"""``get_price_series``/``get_close`` -- a typed read over one ticker's
``price_history`` versions, built on ``Repository.scan(DataQuery)`` with a
``SnapshotRef``, exposed on ``Repository`` (``get_price_series``/``get_close``)
the same way ``get_chain`` is -- modeled directly on
``engine.v2.data.chains.get_chain`` (coordinator direction, 2026-09-14).

Both functions here do one bounded ``price_history`` scan for ``query.ticker``
(every date, every ``retrieved_at`` -- the whole bitemporal history the as-of
reconstruction needs), then hand the fetched rows to the pure
``price_history.as_of_view``/``_provenance_by_date`` logic unchanged. No
``QUERY_NOT_BOUNDED``/``RESULT_LIMIT_EXCEEDED`` escape hatch beyond what
``Repository.scan`` itself already enforces (``max_batch_rows``/
``max_result_rows``, sized for one ticker's full history).

One read rule, no policy choice (user decision 2026-09-14, replacing an
earlier two-policy design): the latest retrieval of any source at or before
``observation_ceiling``, else the ticker's earliest.
"""
from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from engine.v2.contracts import DataQuery, KeyPredicate, PriceQuery, PriceSeriesRow, SnapshotRef

from . import errors, price_history
from .price_history import as_of_view
from .price_history_table import PRICE_HISTORY_TABLE_NAME

__all__ = ["get_close", "get_price_series", "has_price_history", "tickers_with_price_history"]

_COLUMNS = ("date", "close_adj", "close_raw", "high_raw", "retrieved_at", "deleted", "source_kind",
           "source_hash")
_BATCH_CAP = 50_000
_RESULT_CAP = 200_000

#: task brief 2026-09-15 send-back: :func:`tickers_with_price_history`'s
#: in-process existence cache, keyed by ``dataset_version_id`` alone -- the
#: PINNED table version is immutable once committed, so the full
#: ``{ticker with any row}`` set computed for it is valid FOREVER and for
#: every caller/ticker-subset, not just a repeat call with the identical
#: arguments. The supervisor's own ``prepare_launch`` and, later,
#: ``materialize_effect`` (same request, same pinned version) collapse to
#: one ``fragment_records`` lookup plus a cache hit. Bounded so a
#: long-running supervisor process serving many DIFFERENT snapshot
#: generations over its lifetime does not grow this without bound.
_EXISTENCE_CACHE: "OrderedDict[str, frozenset[str]]" = OrderedDict()
_EXISTENCE_CACHE_MAXSIZE = 64


def get_price_series(repository, query: PriceQuery, snapshot_ref: SnapshotRef
                     ) -> tuple[PriceSeriesRow, ...]:
    """The as-of series for ``query.ticker`` under ``snapshot_ref``, in date
    order, each row carrying the ``retrieved_at``/``source_hash`` it came from.

    Refuses ``QUERY_NOT_BOUNDED`` if ``session_date`` is after
    ``observation_ceiling``. Refuses ``CONTRACT_MISMATCH`` if the snapshot has
    no ``price_history`` table, or the scan returns no rows for this ticker at
    all -- mirroring ``chains.get_chain``'s "snapshot has no option_chains
    table."
    """
    ceiling_date = query.observation_ceiling[:10]
    if query.session_date > ceiling_date:
        raise errors.fail("QUERY_NOT_BOUNDED", "session_date is after the observation ceiling")
    if PRICE_HISTORY_TABLE_NAME not in snapshot_ref.table_versions:
        raise errors.fail("CONTRACT_MISMATCH", "snapshot has no price_history table")
    stored_rows = _fetch_rows(repository, snapshot_ref, query.ticker)
    if stored_rows.empty:
        raise errors.fail("CONTRACT_MISMATCH", "no price_history for this ticker",
                          details={"ticker": query.ticker})
    view = as_of_view(stored_rows, query.observation_ceiling)
    view = view[view["date"].astype(str).str.slice(0, 10) <= query.session_date]
    view = view.sort_values("date")
    if query.lookback_sessions > 0:
        view = view.tail(query.lookback_sessions)
    provenance = _provenance_by_date(stored_rows, query.observation_ceiling)
    rows = []
    for row in view.itertuples(index=False):
        date_str = str(row.date)[:10]
        retrieved_at, source_hash = provenance.get(date_str, (None, None))
        rows.append(PriceSeriesRow(
            date=date_str,
            close_adj=None if pd.isna(row.close_adj) else float(row.close_adj),
            close_raw=None if pd.isna(row.close_raw) else float(row.close_raw),
            high_raw=None if pd.isna(row.high_raw) else float(row.high_raw),
            retrieved_at=retrieved_at, source_hash=source_hash))
    return tuple(rows)


def has_price_history(repository, snapshot_ref: SnapshotRef, ticker: str) -> bool:
    """True iff ``snapshot_ref`` carries at least one ``price_history`` row
    for ``ticker`` -- the one existence check callers should use to decide
    "this ticker has no price history at all" BEFORE calling
    :func:`get_price_series`, instead of catching its ``CONTRACT_MISMATCH``
    (which also covers the snapshot-wide "no price_history table" case, a
    different condition than one ticker having no rows).

    Still refuses ``CONTRACT_MISMATCH`` if the snapshot has no
    ``price_history`` table at all -- that is a snapshot-level invariant,
    not a per-ticker absence, so it is not something a caller should treat
    as "this ticker is absent" and silently skip.

    task brief 2026-09-15 send-back: a thin single-ticker convenience over
    :func:`tickers_with_price_history` -- no separate lookup of its own, and
    every caller checking MANY tickers should call that function directly
    rather than loop this one (that loop is exactly the "N per-ticker
    scans" defect this send-back removed from ``px_expected_paths`` and
    ``materialize_price_series``).
    """
    return ticker in tickers_with_price_history(repository, snapshot_ref, (ticker,))


def tickers_with_price_history(repository, snapshot_ref: SnapshotRef, tickers=None) -> frozenset[str]:
    """The subset of ``tickers`` (or every ticker, if ``tickers`` is
    ``None``) that has at least one ``price_history`` row under this
    PINNED snapshot's table version.

    Answered from ``repository.fragment_records`` -- SQLite catalog
    metadata, never a ``Repository.scan`` of actual row content.
    ``price_history`` partitions ``partition_columns=("ticker",)`` (one
    fragment per ticker, holding that ticker's ENTIRE bitemporal history,
    rewritten whole on any change -- see ``price_history_table.py``'s
    module docstring), so a ticker has a ``FragmentRecord`` at this pinned
    dataset version iff it has ever been captured; ``row_count > 0`` is
    still checked explicitly rather than trusted bare, so "this ticker has
    a fragment" and "this ticker has at least one row" are never conflated
    by assumption alone.

    Refuses ``CONTRACT_MISMATCH`` if the snapshot has no ``price_history``
    table at all, the same snapshot-wide case :func:`has_price_history`
    already covered.

    task brief 2026-09-15 send-back: replaces a per-ticker
    ``has_price_history`` loop (N ``Repository.scan`` calls, 8 columns,
    each ticker's WHOLE bitemporal history) that ran at both
    ``prepare_launch`` and ``materialize_effect`` in the coordinator, plus a
    THIRD time inside ``materialize_price_series``.

    First implementation of this primitive used ``Repository.scan`` with a
    ``KeyPredicate(operator="in", values=tickers)`` (columns=("ticker",)) --
    real measurement against the actual failing attempt's request (201
    tickers) hit ``RESULT_LIMIT_EXCEEDED`` against ``price_history``'s own
    ``maximum_result_rows`` cap (500,000): the real data's combined
    bitemporal row count for just 201 tickers already exceeds it, so even a
    request-scoped ``in``-filtered scan is not viable at this data's real
    scale, let alone an unfiltered one over the D14 corpus's ~36M rows.
    Fragment-record metadata has no such ceiling -- its cost is the number
    of FRAGMENTS (one per ticker, ~2,975 for the full D14 corpus), never
    the number of ROWS.

    Cached in-process by ``dataset_version_id`` alone (not by the requested
    ticker subset): the pinned table version is immutable, and every
    caller's answer is a filter over the SAME full per-version set, so one
    ``fragment_records`` call per pinned version, ever, serves every
    caller and every ticker subset for that version. See
    :data:`_EXISTENCE_CACHE`'s own comment for the bound.
    """
    if PRICE_HISTORY_TABLE_NAME not in snapshot_ref.table_versions:
        raise errors.fail("CONTRACT_MISMATCH", "snapshot has no price_history table")
    dvr = snapshot_ref.table_versions[PRICE_HISTORY_TABLE_NAME]
    cache_key = dvr.dataset_version_id
    cached = _EXISTENCE_CACHE.get(cache_key)
    if cached is None:
        records = repository.fragment_records(snapshot_ref, PRICE_HISTORY_TABLE_NAME)
        cached = frozenset(record.partition_key for record in records if record.row_count > 0)
        _EXISTENCE_CACHE[cache_key] = cached
        if len(_EXISTENCE_CACHE) > _EXISTENCE_CACHE_MAXSIZE:
            _EXISTENCE_CACHE.popitem(last=False)
    _EXISTENCE_CACHE.move_to_end(cache_key)
    if tickers is None:
        return cached
    return frozenset(ticker for ticker in tickers if ticker in cached)


def _fetch_rows(repository, snapshot_ref: SnapshotRef, ticker: str) -> pd.DataFrame:
    dvr = snapshot_ref.table_versions[PRICE_HISTORY_TABLE_NAME]
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id, table_contract_ref=dvr.table_contract_ref,
        columns=_COLUMNS, key_filter=(KeyPredicate(column="ticker", operator="eq", values=(ticker,)),),
        order_by=("ticker", "date", "retrieved_at"), max_batch_rows=_BATCH_CAP,
        max_result_rows=_RESULT_CAP)
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=PRICE_HISTORY_TABLE_NAME):
        rows.extend(batch.to_pylist())
    return pd.DataFrame(rows, columns=list(_COLUMNS))


def _provenance_by_date(stored_rows: pd.DataFrame, cutoff: str) -> dict[str, tuple[str, str]]:
    """Per date: the ``(retrieved_at, source_hash)`` of the version
    ``as_of_view`` actually selected -- ``price_history.resolve_pool``'s own
    pool, kept here only for the two extra columns ``as_of_view`` itself
    drops. The SAME resolution, never a second one.
    """
    pool = price_history.resolve_pool(stored_rows, cutoff)
    return {str(row.date)[:10]: (row.retrieved_at, row.source_hash)
           for row in pool.itertuples(index=False)}


def get_close(repository, ticker: str, date: str, observation_ceiling: str,
             snapshot_ref: SnapshotRef) -> PriceSeriesRow:
    """One closing date's value and its provenance -- a convenience wrapper
    over :func:`get_price_series` with ``lookback_sessions=1``.
    """
    query = PriceQuery(ticker=ticker, session_date=date, observation_ceiling=observation_ceiling,
                       lookback_sessions=1)
    series = get_price_series(repository, query, snapshot_ref)
    if not series or series[-1].date != date:
        raise errors.fail("CONTRACT_MISMATCH", "no price_history version covers this date",
                          details={"ticker": ticker, "date": date})
    return series[-1]
