"""Cutover PR-6's raw-row producer, slice 6a: the events-scan + ``BoardRequest``
enumeration half only. Row-staging (``calendar_row``/``panel_row``/
``panel_anchor``/``tier4_row``/``quote_rows`` -> ``native_score_batch.
NightlyEventInputs``) is a later slice, not built here.

The scan deliberately applies no ``src_orats`` filter (unlike
``computed_moves_store._scan_once``, whose backward-looking selection has its
own reason to): ``engine/data/schemas.py``'s ``EARNINGS_EVENTS`` docstring
names ORATS/oquants history-only sources, while Nasdaq/yfinance carry the
forward dates the monitoring board scores.

Like ``native_board_universe``, this module never imports ``engine.score`` /
``engine.structures`` / ``engine.replay`` / ``engine.fills``.
"""
from __future__ import annotations

import pandas as pd

from engine.v2.contracts import DataQuery, KeyPredicate, SnapshotRef
from engine.v2.data.repository import Repository
from engine.v2.ops.native_board_universe import BoardRequest, board_requests

__all__ = ["scan_forward_board_requests"]

_EVENTS_TABLE = "earnings_events"
_EVENTS_COLUMNS = ("ticker", "event_date", "session")


def scan_forward_board_requests(
    repository: Repository,
    snapshot: SnapshotRef,
    *,
    as_of,
    horizon_days: int,
    tickers=None,
) -> tuple[BoardRequest, ...]:
    """Scan ``earnings_events`` off ``snapshot`` through ``repository`` -- the
    same read ``computed_moves_store._scan_once`` uses for the same table --
    and return ``native_board_universe.board_requests``'s ordered tuple for
    the forward window. As this module's own docstring says, the scan adds no
    ``src_orats`` filter of its own.
    """
    contract_ref = snapshot.table_versions[_EVENTS_TABLE].table_contract_ref
    contract = repository.table_contract(snapshot, _EVENTS_TABLE)
    years = tuple(sorted({int(record.partition_key)
                          for record in repository.fragment_records(snapshot, _EVENTS_TABLE)}))
    if not years:
        return board_requests(as_of, horizon_days, tickers,
                              pd.DataFrame(columns=_EVENTS_COLUMNS))
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=contract_ref,
        columns=_EVENTS_COLUMNS,
        key_filter=(KeyPredicate(column="year", operator="in", values=years),),
        order_by=tuple(contract.primary_key),
        max_batch_rows=min(contract.maximum_batch_rows, 50_000),
        max_result_rows=min(contract.maximum_result_rows, 50_000_000))
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=_EVENTS_TABLE):
        rows.extend(batch.to_pylist())
    events_table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_EVENTS_COLUMNS)
    return board_requests(as_of, horizon_days, tickers, events_table)
