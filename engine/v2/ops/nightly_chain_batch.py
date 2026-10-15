"""One batched ``option_chains`` read per producer build (#754).

``scan_chain_slices`` replaces the per-event pair of exact ``(ticker, obs_date)``
scans: one ``DataQuery`` with ``ticker in <tickers>`` and ``obs_date eq
<decision session>`` over the union of the candidate and quote columns, grouped
by ticker in primary-key order (so each group equals what the per-key scan
returns). Failure semantics: a ticker with no rows maps to ``[]``; a missing
table is ``Repository.table_contract``'s own ``CONTRACT_MISMATCH``; every
repository error (including ``RESULT_LIMIT_EXCEEDED``) propagates and never
falls back to per-key scans.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from engine.v2.contracts import DataQuery, KeyPredicate, SnapshotRef
from engine.v2.data.repository import Repository

__all__ = ["CHAIN_COLUMNS", "scan_chain_slices"]

CHAIN_TABLE = "option_chains"
#: Union of the candidate-expiry columns and the quote-row columns.
CHAIN_COLUMNS = ("ticker", "obs_date", "expiry", "strike", "right", "bid", "ask")
_BATCH_CAP = 50_000
_RESULT_CAP = 2_000_000


def scan_chain_slices(repository: Repository, snapshot: SnapshotRef, tickers: Iterable[str], *,
                      decision_session: str) -> dict[str, list[dict[str, Any]]]:
    """Every ``tickers`` entry's ``option_chains`` rows for ``decision_session``
    (an ISO ``YYYY-MM-DD`` string), from a single ``Repository.scan``."""
    wanted = sorted(set(tickers))
    slices: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in wanted}
    if not wanted:
        return slices
    contract = repository.table_contract(snapshot, CHAIN_TABLE)
    version = snapshot.table_versions[CHAIN_TABLE]
    key_filter = (KeyPredicate(column="ticker", operator="in", values=tuple(wanted)),
                  KeyPredicate(column="obs_date", operator="eq", values=(decision_session,)))
    population_bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=CHAIN_TABLE,
        table_contract_ref=version.table_contract_ref,
        key_filter=key_filter, time_interval=None)
    max_result_rows = min(_RESULT_CAP, population_bound)
    max_batch_rows = (min(contract.maximum_batch_rows, _BATCH_CAP, max_result_rows)
                      if max_result_rows > 0
                      else min(contract.maximum_batch_rows, _BATCH_CAP))
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=version.table_contract_ref,
        columns=CHAIN_COLUMNS,
        key_filter=key_filter,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows)
    for batch in repository.scan(query, table_name=CHAIN_TABLE):
        for row in batch.to_pylist():
            slices[row["ticker"]].append(row)
    return slices