"""Direct tests for ``nightly_raw_rows.scan_forward_board_requests`` (cutover
PR-6a, issue #199): a real catalog + ArtifactStore snapshot built from the
four REAL ``earnings_events`` rows the slice's spec captured from the pinned
``shadow`` snapshot (2026-09-29) -- proving the scan itself applies no
``src_orats`` filter, that the ``session``/window filtering is
``board_requests``'s own, and that the result is exactly what calling
``board_requests`` directly yields."""
from __future__ import annotations

import pandas as pd

from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops.native_board_universe import BoardRequest, board_requests
from engine.v2.ops.nightly_raw_rows import scan_forward_board_requests
from tests.data_scan_support import commit_tables, contract_for, contract_ref_for, publish_and_inspect
from tests.ops_support import catalog

_EVENTS = contract_for("earnings_events")
_EVENTS_REF = contract_ref_for(_EVENTS)
_DAILY = contract_for("daily_market")

_AS_OF = "2026-09-29"
_HORIZON_DAYS = 60


def _event_row(ticker: str, event_date, session, src_orats: bool) -> dict:
    d = pd.Timestamp(event_date).to_pydatetime()
    return dict(
        event_id=f"{ticker}_{d.date()}", ticker=ticker, event_date=d, year=d.year,
        session=session, session_src="orats", annc_tod=None, src_orats=src_orats,
        src_oquants=False, src_nasdaq=False, src_yfinance=False, date_agree=True,
        date_conflict=False, updated_at=None, event_cluster_id=None, claim_count=None,
        reconciliation=None)


def _head_row(conn, scope="shadow"):
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (scope,)).fetchone()


def _build_parent(conn, clock, store, *, events_rows=()):
    tables = {"daily_market": []}
    if events_rows:
        record = publish_and_inspect(store, _EVENTS, _EVENTS_REF, list(events_rows), "2026")
        tables["earnings_events"] = [record]
    else:
        tables["earnings_events"] = []
    contracts = {"earnings_events": _EVENTS, "daily_market": _DAILY}
    commit_tables(conn, clock, tables, contracts, store=store)
    return _head_row(conn)


def _scan(repository, snapshot) -> tuple[BoardRequest, ...]:
    return scan_forward_board_requests(repository, snapshot, as_of=_AS_OF,
                                       horizon_days=_HORIZON_DAYS)


def test_scan_forward_board_requests_includes_a_forward_row_with_src_orats_false(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store,
                         events_rows=[_event_row("ACI", "2026-10-13", "BMO", False)])
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])

    result = _scan(repository, snapshot)

    assert result
    assert all(request.ticker == "ACI" for request in result)


def test_scan_forward_board_requests_excludes_a_null_session_row(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store,
                         events_rows=[_event_row("ADXN", "2026-10-05", None, False)])
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])

    assert _scan(repository, snapshot) == ()


def test_scan_forward_board_requests_excludes_a_row_outside_the_window(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store,
                         events_rows=[_event_row("Z", "2026-05-06", "AMC", True)])
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])

    assert _scan(repository, snapshot) == ()


def test_scan_forward_board_requests_on_a_snapshot_with_no_earnings_events_fragments_returns_empty(
        tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store, events_rows=())
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])

    assert _scan(repository, snapshot) == ()


def test_scan_forward_board_requests_matches_board_requests_called_directly(tmp_path):
    rows = [_event_row("ACI", "2026-10-13", "BMO", False),
            _event_row("ACN", "2026-10-01", "BMO", False),
            _event_row("ADXN", "2026-10-05", None, False),
            _event_row("Z", "2026-05-06", "AMC", True)]
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store, events_rows=rows)
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])

    scanned = _scan(repository, snapshot)
    events_table = pd.DataFrame([{"ticker": row["ticker"], "event_date": row["event_date"],
                                  "session": row["session"]} for row in rows])
    direct = board_requests(_AS_OF, _HORIZON_DAYS, None, events_table)

    assert scanned == direct
