"""Direct tests for ``nightly_raw_rows.scan_forward_board_requests`` (cutover
PR-6a, issue #199): a real catalog + ArtifactStore snapshot built from the
four REAL ``earnings_events`` rows the slice's spec captured from the pinned
``shadow`` snapshot (2026-09-29) -- proving the scan itself applies no
``src_orats`` filter, that the ``session``/window filtering is
``board_requests``'s own, and that the result is exactly what calling
``board_requests`` directly yields."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from engine.v2.contracts import EventRef
from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_board_universe import (
    _COVERED_STRATEGIES, BoardRequest, board_requests)
from engine.v2.ops.nightly_raw_rows import scan_calendar_row, scan_forward_board_requests
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal, assemble_nightly_source_bundle)
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
    rows_by_year: dict[str, list[dict]] = {}
    for row in events_rows:
        rows_by_year.setdefault(str(row["year"]), []).append(row)
    tables["earnings_events"] = [
        publish_and_inspect(store, _EVENTS, _EVENTS_REF, rows, year)
        for year, rows in sorted(rows_by_year.items())
    ]
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

    expected = tuple(
        BoardRequest(ticker, strategy, pd.Timestamp(event_date), "BMO")
        for ticker, event_date in (("ACN", "2026-10-01"), ("ACI", "2026-10-13"))
        for strategy in (*_COVERED_STRATEGIES, "DYN-SV")
    )

    assert scanned == direct
    assert scanned == expected
    corrupted = expected[:-1] + (replace(expected[-1], session="AMC"),)
    # same length as `expected` -- proves the assertion checks VALUES, not just tuple length
    assert scanned != corrupted


def test_scan_forward_board_requests_spans_a_december_to_january_window(tmp_path):
    as_of = "2026-12-20"
    horizon_days = 45
    rows = [_event_row("ACI", "2026-12-28", "BMO", False),
            _event_row("ACN", "2027-01-15", "BMO", False)]
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store, events_rows=rows)
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])

    result = scan_forward_board_requests(repository, snapshot, as_of=as_of,
                                         horizon_days=horizon_days)

    tickers = {request.ticker for request in result}
    assert tickers == {"ACI", "ACN"}


_CONTEXT = dict(entry_date="2026-10-12", exit_date="2026-10-13",
                expiry="2026-10-16", spot=20.0, calendar_observed_through=_AS_OF)
_KEY = BoardRequest("ACI", "STR-THRU", pd.Timestamp("2026-10-13"), "BMO")


@pytest.fixture
def calendar_source(tmp_path):
    # Captured ACI row above, with an intentionally non-derived ID to detect synthesis.
    row = {**_event_row("ACI", "2026-10-13", "BMO", False), "event_id": "persisted-aci"}
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store, events_rows=[row])
    repository = Repository(conn, store)
    return conn, clock, store, repository, repository.resolve(head["snapshot_id"])


def test_calendar_row_exact_fields_revision_and_event_lookup(calendar_source, monkeypatch):
    _, _, _, repository, snapshot = calendar_source
    queries = []
    original = repository.scan

    def recorded(query, *, table_name):
        queries.append((query, table_name))
        return original(query, table_name=table_name)

    monkeypatch.setattr(repository, "scan", recorded)
    result = scan_calendar_row(repository, snapshot, _KEY, **_CONTEXT)
    assert result.calendar_row == {
        "event_id": "persisted-aci", "ticker": "ACI", "event_date": "2026-10-13",
        "session": "BMO", **_CONTEXT,
    }
    assert result.calendar_revision == snapshot.table_versions["earnings_events"].dataset_version_id
    assert result.calendar_revision != snapshot.calendar_version
    query, table = queries[0]
    assert len(queries) == 1 and table == "earnings_events"
    assert query.snapshot_id == snapshot.snapshot_id
    assert query.table_contract_ref == snapshot.table_versions[table].table_contract_ref
    assert query.columns == ("event_id", "ticker", "event_date", "session")
    assert [(p.column, p.operator, p.values) for p in query.key_filter] == [
        ("ticker", "eq", ("ACI",)), ("year", "eq", (2026,))]
    assert query.order_by == _EVENTS.primary_key
    assert 0 < query.max_batch_rows <= _EVENTS.maximum_batch_rows
    assert 0 < query.max_result_rows <= _EVENTS.maximum_result_rows
    repeat = scan_calendar_row(repository, snapshot, _KEY, **_CONTEXT)
    assert repeat == result and repeat.calendar_row is not result.calendar_row
    event = repository.get_event(EventRef(
        event_id=result.calendar_row["event_id"], calendar_revision=result.calendar_revision), snapshot)
    assert event.ticker_at_event == "ACI" and event.scheduled_event_date == "2026-10-13"


def test_calendar_row_keeps_pinned_identity_after_head_moves(calendar_source):
    conn, clock, store, repository, snapshot = calendar_source
    changed = {**_event_row("ACI", "2026-10-13", "BMO", False), "event_id": "new-aci"}
    fragment = publish_and_inspect(store, _EVENTS, _EVENTS_REF, [changed], "2026")
    newer = commit_tables(conn, clock, {"earnings_events": [fragment]},
                          {"earnings_events": _EVENTS}, store=store, scope="new-shadow",
                          receipt_id="r2", attempt_id="att-2")
    conn.execute("UPDATE data_snapshot_heads SET snapshot_id = ?, generation = generation + 1 "
                 "WHERE scope = ?", (newer.snapshot_id, "shadow"))
    assert repository.resolve_pinned("shadow").snapshot_id == newer.snapshot_id
    old_result = scan_calendar_row(repository, snapshot, _KEY, **_CONTEXT)
    new_result = scan_calendar_row(repository, newer, _KEY, **_CONTEXT)
    assert old_result.calendar_row["event_id"] == "persisted-aci"
    assert new_result.calendar_row["event_id"] == "new-aci"
    assert old_result.calendar_revision != new_result.calendar_revision


@pytest.mark.parametrize("changes", [
    {"ticker": "MISSING"}, {"event_date": pd.Timestamp("2026-10-14")},
    {"session": "AMC"}, {"event_date": pd.Timestamp("2027-10-13")},
])
def test_calendar_row_requires_exact_source_match(calendar_source, changes):
    repository, snapshot = calendar_source[-2:]
    with pytest.raises(DataError) as exc:
        scan_calendar_row(repository, snapshot, replace(_KEY, **changes), **_CONTEXT)
    assert exc.value.code == "EVENT_NOT_FOUND"


def test_calendar_row_refuses_ambiguous_distinct_ids(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    # Deliberate boundary mutation: two persisted IDs for one ticker/date/session.
    rows = [{**_event_row("ACI", "2026-10-13", "BMO", False), "event_id": name}
            for name in ("first", "second")]
    head = _build_parent(conn, clock, store, events_rows=rows)
    repository = Repository(conn, store)
    with pytest.raises(DataError) as exc:
        scan_calendar_row(repository, repository.resolve(head["snapshot_id"]), _KEY, **_CONTEXT)
    assert exc.value.code == "IDENTITY_CONFLICT"


@pytest.mark.parametrize("field", ["entry_date", "exit_date", "expiry", "calendar_observed_through"])
@pytest.mark.parametrize("bad", [None, True, 20261013, pd.NaT, "not-a-date",
                                  "2026-10-13T01:00:00", "2026-10-13T00:00:00Z"])
def test_calendar_row_rejects_invalid_staged_dates(calendar_source, field, bad):
    repository, snapshot = calendar_source[-2:]
    with pytest.raises(OpsError) as exc:
        scan_calendar_row(repository, snapshot, _KEY, **{**_CONTEXT, field: bad})
    assert exc.value.code == "INVALID_REQUEST"
    assert "not-a-date" not in str(exc.value)


@pytest.mark.parametrize("bad", [None, True, "20", 0, -1, float("nan"), float("inf"), 2j])
def test_calendar_row_rejects_invalid_spot(calendar_source, bad):
    repository, snapshot = calendar_source[-2:]
    with pytest.raises(OpsError) as exc:
        scan_calendar_row(repository, snapshot, _KEY, **{**_CONTEXT, "spot": bad})
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("key", [None, replace(_KEY, ticker=""), replace(_KEY, session=None),
                                  replace(_KEY, strategy=" "), replace(_KEY, event_date=True),
                                  replace(_KEY, event_date=pd.NaT)])
def test_calendar_row_rejects_invalid_key(calendar_source, key):
    repository, snapshot = calendar_source[-2:]
    with pytest.raises(OpsError) as exc:
        scan_calendar_row(repository, snapshot, key, **_CONTEXT)
    assert exc.value.code == "INVALID_REQUEST"


def test_calendar_row_refuses_blank_persisted_identity(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    row = {**_event_row("ACI", "2026-10-13", "BMO", False), "event_id": " "}
    head = _build_parent(conn, clock, store, events_rows=[row])
    repository = Repository(conn, store)
    with pytest.raises(OpsError) as exc:
        scan_calendar_row(repository, repository.resolve(head["snapshot_id"]), _KEY, **_CONTEXT)
    assert exc.value.code == "INVALID_REQUEST"


def test_calendar_row_missing_table_and_invalid_revision(calendar_source):
    repository, snapshot = calendar_source[-2:]
    with pytest.raises(DataError) as exc:
        scan_calendar_row(repository, replace(snapshot, table_versions={}), _KEY, **_CONTEXT)
    assert exc.value.code == "CONTRACT_MISMATCH"
    version = replace(snapshot.table_versions["earnings_events"], dataset_version_id="")
    broken = replace(snapshot, table_versions={**snapshot.table_versions, "earnings_events": version})
    with pytest.raises(OpsError) as exc:
        scan_calendar_row(repository, broken, _KEY, **_CONTEXT)
    assert exc.value.code == "INVALID_REQUEST"


def test_calendar_row_propagates_repository_failure(calendar_source, monkeypatch):
    repository, snapshot = calendar_source[-2:]
    problem = data_fail("MANIFEST_CORRUPT", "synthetic corrupt fragment")

    def corrupted(*args, **kwargs):
        raise problem

    monkeypatch.setattr(repository, "scan", corrupted)
    with pytest.raises(DataError) as exc:
        scan_calendar_row(repository, snapshot, _KEY, **_CONTEXT)
    assert exc.value is problem


def test_calendar_row_meets_source_bundle_contract_and_causal_checks(calendar_source):
    repository, snapshot = calendar_source[-2:]
    result = scan_calendar_row(repository, snapshot, _KEY, **_CONTEXT)
    kwargs = dict(source_ref="test:calendar", calendar_row=result.calendar_row,
                  panel_row={"date": "2026-10-13", "signal": 1.5}, panel_anchor=_AS_OF,
                  tier4_row={}, quote_rows=[], quote_status="empty", feature_names=("signal",))
    bundle = assemble_nightly_source_bundle(as_of=_AS_OF, **kwargs)
    assert bundle.context == {"ticker": "ACI", "event_date": "2026-10-13", **_CONTEXT}
    assert bundle.feature_vector == {"signal": 1.5}
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(as_of="2026-09-28", **kwargs)
    assert exc.value.code == "POST_AS_OF_ROW"
