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
"""
from __future__ import annotations

import pandas as pd

from engine.v2.contracts import DataQuery, KeyPredicate, PriceQuery, PriceSeriesRow, SnapshotRef

from . import errors
from .price_history import as_of_view
from .price_history_table import PRICE_HISTORY_TABLE_NAME

__all__ = ["get_close", "get_price_series"]

_COLUMNS = ("date", "close_adj", "close_raw", "high_raw", "retrieved_at", "deleted", "source_hash")
_BATCH_CAP = 50_000
_RESULT_CAP = 200_000


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
    ``as_of_view`` actually selected -- the same eligible/fallback pool
    ``as_of_view`` computes, kept here only for the two extra columns
    ``as_of_view`` itself drops.
    """
    eligible = stored_rows[stored_rows["retrieved_at"] <= cutoff]
    pool = eligible if len(eligible) else stored_rows[
        stored_rows["retrieved_at"] == stored_rows["retrieved_at"].min()]
    idx = pool.groupby("date")["retrieved_at"].idxmax()
    winners = pool.loc[idx]
    return {str(row.date)[:10]: (row.retrieved_at, row.source_hash)
            for row in winners.itertuples(index=False)}


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
