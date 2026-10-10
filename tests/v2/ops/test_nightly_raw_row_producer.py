"""Regression coverage for cutover PR-6 slice 4b:
``engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events``.

Pinned by ``engine/v2/ops/ARCHITECTURE.md``: enumeration through
``nightly_raw_rows.scan_forward_board_requests`` in request order, intraday
admission refused before any context read, one build-scoped decision calendar
and shared panel rows/anchors per ``(ticker, event_date, session)``,
per-``BoardRequest`` calendar/quote composition, empty-window short circuit and
deterministic public-safe refusal documents. The producer is the unit under
test; its 4a readers are faked, over a real catalog/ArtifactStore snapshot
carrying only ``earnings_events``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.features import panel_row_inputs
from engine.v2.features.panel_row_inputs import PanelRowInputs
from engine.v2.foundation.market_calendar import CalendarSessions
from engine.v2.ops import nightly_calendar_inputs, nightly_quote_rows, nightly_raw_rows
from engine.v2.ops import nightly_raw_row_producer as nrp
from engine.v2.ops.native_score_batch import NativeScoreBatchRowRefusal
from engine.v2.ops.nightly_quote_rows import QuoteRowInputs
from engine.v2.ops.nightly_raw_rows import CalendarRowInputs
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

_EVENTS_NAME = "earnings_events"
_EVENTS = contract_for(_EVENTS_NAME)
_EVENTS_REF = contract_ref_for(_EVENTS)

_AS_OF = "2024-01-05"
_HORIZON_DAYS = 30
_MIDNIGHT = pd.Timestamp("2024-01-16")
_SECOND = pd.Timestamp("2024-01-17")
_INTRADAY = pd.Timestamp("2024-01-16 13:30:00")
_MIDNIGHT_WIRE = "2024-01-16"
_INTRADAY_WIRE = "2024-01-16T13:30:00"
_REFUSAL_SCHEMA = "native_score_batch_producer_refusals.v1.0"
_PANEL_HISTORY_SESSIONS = 253
_PANEL_HISTORY_DETAIL = "the pinned snapshot lacks required earlier panel sessions"
_INTRADAY_DETAIL = "the board request carries an intraday event timestamp"
_PROJECTED_DAYS = ("2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
                   "2024-01-12", "2024-01-16", "2024-01-17")


def _event_row(ticker: str, event_date, session: str, *, event_id: str) -> dict:
    stamp = pd.Timestamp(event_date)
    return dict(
        event_id=event_id,
        ticker=ticker,
        event_date=stamp.to_pydatetime(),
        year=stamp.year,
        session=session,
        session_src="orats",
        annc_tod=None,
        src_orats=False,
        src_oquants=False,
        src_nasdaq=False,
        src_yfinance=False,
        date_agree=True,
        date_conflict=False,
        updated_at=None,
        event_cluster_id=None,
        claim_count=None,
        reconciliation=None,
    )


def _snapshot(tmp_path, events: list[dict], *, partition: str = "2024"):
    conn, clock, store = catalog_and_store(tmp_path)
    rows = sorted(events, key=lambda row: tuple(row[f] for f in _EVENTS.primary_key))
    fragment = publish_and_inspect(store, _EVENTS, _EVENTS_REF, rows, partition)
    snapshot = commit_tables(conn, clock, {_EVENTS_NAME: [fragment]},
                             {_EVENTS_NAME: _EVENTS}, scope="test")
    return Repository(conn, store), snapshot


def _wire_event_date(value) -> str:
    stamp = pd.Timestamp(value)
    return stamp.date().isoformat() if stamp.normalize() == stamp else stamp.isoformat()


def _event_keylike(key) -> dict:
    return {"ticker": key.ticker, "strategy": key.strategy,
            "event_date": _wire_event_date(key.event_date), "session": key.session}


def _identity(key_document) -> tuple:
    return (key_document["ticker"], key_document["strategy"],
            key_document["event_date"], key_document["session"])


def _distinct_events(requests) -> list[tuple]:
    seen: set[tuple] = set()
    ordered: list[tuple] = []
    for key in requests:
        marker = (key.ticker, key.event_date, key.session)
        if marker not in seen:
            seen.add(marker)
            ordered.append(marker)
    return ordered


def _calendar(*, sessions: int = _PANEL_HISTORY_SESSIONS) -> CalendarSessions:
    """The build-scoped calendar: ``sessions`` observed weekday sessions at or
    before ``_AS_OF`` -- the producer counts them for its panel-history floor --
    plus the projected tail its callers' event dates sit on. The real calendar
    reader is faked, so only the as-of-side count is load-bearing here."""
    history = pd.bdate_range(end=_AS_OF, periods=sessions)
    return CalendarSessions(
        days=tuple(day.date().isoformat() for day in history) + _PROJECTED_DAYS,
        observed_through=_AS_OF)


def _calendar_inputs(key) -> CalendarRowInputs:
    day = key.event_date.date().isoformat()
    return CalendarRowInputs(calendar_revision="ev-rev-1", calendar_row={
        "event_id": f"{key.ticker}-{key.strategy}", "ticker": key.ticker,
        "event_date": day, "session": key.session, "entry_date": _AS_OF,
        "exit_date": day, "expiry": day, "spot": 10.0,
        "calendar_observed_through": _AS_OF})


def _panel_inputs(key) -> PanelRowInputs:
    return PanelRowInputs(panel_row={
        "date": key.event_date.date().isoformat(),
        "event_tag": f"{key.ticker}|{key.session}",
        "spy_ret252": 0.0},
        panel_anchor=pd.Timestamp(key.event_date).normalize())


def _quote_inputs(key, *, expiry: str) -> QuoteRowInputs:
    return QuoteRowInputs(quote_rows=({
        "ticker": key.ticker, "right": "C", "strike": 100.0, "expiry": expiry,
        "bid": 1.0, "ask": 1.2, "observed_at": _AS_OF},), quote_status="recorded")


def _patch_reader(monkeypatch, module, name: str, fake) -> None:
    original = getattr(module, name)
    monkeypatch.setattr(module, name, fake)
    for attribute, value in list(vars(nrp).items()):
        if value is original:
            monkeypatch.setattr(nrp, attribute, fake)


class _ReaderRig:
    """Real enumeration plus recording fakes for every 4a context reader."""

    def __init__(self, monkeypatch):
        self.enum_calls: list[dict] = []
        self.calendar_calls: list[tuple] = []
        self.calendar_row_calls: list = []
        self.panel_calls: list = []
        self.quote_calls: list = []
        self.calendar_row_hook = None
        self.calendar_hook = None
        self.real_enumerate = nightly_raw_rows.scan_forward_board_requests

        def enumerator(repository, snapshot, *, as_of, horizon_days, tickers=None):
            self.enum_calls.append({"repository": repository, "snapshot": snapshot,
                                    "as_of": as_of, "horizon_days": horizon_days,
                                    "tickers": tickers})
            return self.real_enumerate(repository, snapshot, as_of=as_of,
                                       horizon_days=horizon_days, tickers=tickers)

        def decision_calendar(repository, snapshot, *, decision_session, event_through):
            self.calendar_calls.append((decision_session, event_through))
            if self.calendar_hook is not None:
                return self.calendar_hook(decision_session, event_through)
            return _calendar()

        def calendar_row(repository, snapshot, key, *, decision_session, calendar):
            self.calendar_row_calls.append(key)
            if self.calendar_row_hook is not None:
                return self.calendar_row_hook(key)
            return _calendar_inputs(key)

        def panel_row(repository, snapshot, key, *, decision_session, history_start,
                      spy_market_cache=None):
            self.panel_calls.append(key)
            return _panel_inputs(key)

        def quote_rows(repository, snapshot, key, *, expiry, decision_session):
            self.quote_calls.append(key)
            return _quote_inputs(key, expiry=expiry)

        _patch_reader(monkeypatch, nightly_raw_rows, "scan_forward_board_requests",
                      enumerator)
        _patch_reader(monkeypatch, nightly_calendar_inputs, "scan_decision_calendar",
                      decision_calendar)
        _patch_reader(monkeypatch, nightly_calendar_inputs, "scan_calendar_row_inputs",
                      calendar_row)
        _patch_reader(monkeypatch, panel_row_inputs, "scan_panel_row", panel_row)
        _patch_reader(monkeypatch, nightly_quote_rows, "scan_quote_rows", quote_rows)

    def direct(self, repository, snapshot):
        return self.real_enumerate(repository, snapshot, as_of=_AS_OF,
                                   horizon_days=_HORIZON_DAYS, tickers=None)


def test_mixed_midnight_and_intraday_rows_keep_order_and_refuse_intraday(
        tmp_path, monkeypatch):
    repository, snapshot = _snapshot(tmp_path, [
        _event_row("AAA", _MIDNIGHT, "BMO", event_id="aaa-midnight"),
        _event_row("AAA", _INTRADAY, "BMO", event_id="aaa-intraday"),
    ])
    rig = _ReaderRig(monkeypatch)

    events_document, refusals_document = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=_HORIZON_DAYS)

    direct = rig.direct(repository, snapshot)
    midnight = [key for key in direct if key.event_date == _MIDNIGHT]
    intraday = [key for key in direct if key.event_date == _INTRADAY]
    assert midnight and intraday
    assert list(direct) == midnight + intraday
    assert set(key.event_date for key in direct) == {_MIDNIGHT, _INTRADAY}

    assert len(rig.enum_calls) == 1
    call = rig.enum_calls[0]
    assert call["repository"] is repository and call["snapshot"] is snapshot
    assert pd.Timestamp(call["as_of"]) == pd.Timestamp(_AS_OF)
    assert call["horizon_days"] == _HORIZON_DAYS
    assert call["tickers"] is None
    assert len(rig.calendar_calls) == 1

    assert rig.calendar_row_calls == midnight
    assert rig.quote_calls == midnight
    assert [(key.ticker, key.event_date, key.session) for key in rig.panel_calls] \
        == _distinct_events(midnight)

    refusals = refusals_document["refusals"]
    assert [document["key"] for document in events_document] \
        == [_event_keylike(key) for key in midnight]
    assert all(document["key"]["event_date"] == _MIDNIGHT_WIRE
               for document in events_document)
    assert [refusal["key"] for refusal in refusals] \
        == [_event_keylike(key) for key in intraday]
    assert all(refusal["code"] == "INTRADAY_EVENT_NOT_ADMITTED" for refusal in refusals)
    assert all(refusal["key"]["event_date"] == _INTRADAY_WIRE for refusal in refusals)
    assert all(pd.Timestamp(refusal["key"]["event_date"]) == _INTRADAY
               for refusal in refusals)

    successes = {_identity(document["key"]) for document in events_document}
    refused = {_identity(refusal["key"]) for refusal in refusals}
    assert successes and refused and not (successes & refused)
    assert all(document["tier4_row"] == {} for document in events_document)
    assert json.dumps(events_document)
    assert json.dumps(refusals_document)


def test_empty_enumeration_returns_empty_documents_before_context_readers(
        tmp_path, monkeypatch):
    repository, snapshot = _snapshot(tmp_path, [
        _event_row("AAA", pd.Timestamp("2025-06-17"), "BMO", event_id="aaa-2025"),
    ], partition="2025")

    def forbidden(*args, **kwargs):
        raise AssertionError("context reader invoked on an empty enumeration")

    _patch_reader(monkeypatch, nightly_calendar_inputs, "scan_decision_calendar", forbidden)
    _patch_reader(monkeypatch, nightly_calendar_inputs, "scan_calendar_row_inputs",
                  forbidden)
    _patch_reader(monkeypatch, panel_row_inputs, "scan_panel_row", forbidden)
    _patch_reader(monkeypatch, nightly_quote_rows, "scan_quote_rows", forbidden)

    events_document, refusals_document = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=_HORIZON_DAYS)

    assert events_document == []
    assert refusals_document == {"schema_version": _REFUSAL_SCHEMA, "refusals": []}


def test_shared_panel_reads_with_per_request_calendar_and_quotes(tmp_path, monkeypatch):
    repository, snapshot = _snapshot(tmp_path, [
        _event_row("AAA", _MIDNIGHT, "BMO", event_id="aaa-bmo"),
    ])
    rig = _ReaderRig(monkeypatch)

    events_document, refusals_document = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=_HORIZON_DAYS)

    direct = rig.direct(repository, snapshot)
    expected_panel = _distinct_events(direct)
    assert len(direct) > 1
    assert len(expected_panel) == 1
    assert [(key.ticker, key.event_date, key.session) for key in rig.panel_calls] \
        == expected_panel
    assert rig.calendar_row_calls == list(direct)
    assert rig.quote_calls == list(direct)
    assert refusals_document["refusals"] == []
    assert [document["key"] for document in events_document] \
        == [_event_keylike(key) for key in direct]

    for document, key in zip(events_document, direct, strict=True):
        assert document["tier4_row"] == {}
        assert document["panel_row"] == _panel_inputs(key).panel_row
        assert pd.Timestamp(document["panel_anchor"]) \
            == pd.Timestamp(key.event_date).normalize()
        assert document["calendar_row"] == _calendar_inputs(key).calendar_row
        assert list(document["quote_rows"]) == list(
            _quote_inputs(key, expiry=_calendar_inputs(key).calendar_row["expiry"]).quote_rows)
        assert document["quote_status"] == "recorded"
    assert json.dumps(events_document)


def test_panel_reader_called_once_per_ticker(
        tmp_path, monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    import hashlib

    import tests.test_v2_features_panel_row_inputs as fixture
    from engine.v2.ops.native_board_universe import BoardRequest

    history = pd.bdate_range(end=fixture._DECISION, periods=300)
    batches = fixture._default_batches()
    batches[("daily_market", "SPY")] = fixture._spy_rows(periods=300)
    for ticker in ("AAA", "BBB"):
        batches[(fixture.COMPUTED_MOVES_TABLE_NAME, ticker)] = \
            fixture._computed_rows([(history[20].date().isoformat(), 4.0, False)])
    price_history_table, price_history_rows = next(
        (table, rows) for (table, _ticker), rows in batches.items()
        if table == "price_history" and rows)
    batches[(price_history_table, "BBB")] = price_history_rows
    snapshot = fixture._snapshot()
    reads = []

    class CountingRepository(fixture._FakeRepository):
        def scan(self, query, *, table_name):
            ticker = (query.key_filter[0].values[0]
                      if query.key_filter else "*")
            reads.append((table_name, ticker))
            if query.time_interval is not None:
                rows = self._batches.get((table_name, ticker), [])
                selected = [row for row in rows
                            if fixture._bound_row_selected(
                                row, query.key_filter, query.time_interval)]
                yield fixture._Batch(selected)
            else:
                yield from super().scan(query, table_name=table_name)

        def fragment_records(self, snapshot_ref, table_name):
            return tuple(
                SimpleNamespace(partition_key=ticker, row_count=len(rows))
                for (candidate_table, ticker), rows in self._batches.items()
                if candidate_table == table_name)

    repository = CountingRepository(snapshot, fixture._contracts(), batches)
    keys = [BoardRequest(ticker=ticker, strategy="STR-THRU",
                         event_date=fixture._EVENT, session="BMO")
            for ticker in ("AAA", "BBB")]

    def enumerate_keys(repo, snap, *, as_of, horizon_days, tickers=None):
        return keys

    def decision_calendar(repo, snap, *, decision_session, event_through):
        days = tuple(day.date().isoformat() for day in history) + (
            fixture._EVENT.date().isoformat(),)
        return CalendarSessions(days=days, observed_through=decision_session)

    def calendar_row(repo, snap, key, *, decision_session, calendar):
        day = key.event_date.date().isoformat()
        return CalendarRowInputs(calendar_revision="events-rev", calendar_row={
            "event_id": f"{key.ticker}-{key.strategy}", "ticker": key.ticker,
            "event_date": day, "session": key.session,
            "entry_date": decision_session, "exit_date": day,
            "expiry": day, "spot": 10.0,
            "calendar_observed_through": decision_session})

    def quote_rows(repo, snap, key, *, expiry, decision_session):
        return QuoteRowInputs(quote_rows=({
            "ticker": key.ticker, "right": "C", "strike": 100.0,
            "expiry": expiry, "bid": 1.0, "ask": 1.2,
            "observed_at": decision_session},), quote_status="recorded")

    _patch_reader(monkeypatch, nightly_raw_rows,
                  "scan_forward_board_requests", enumerate_keys)
    _patch_reader(monkeypatch, nightly_calendar_inputs,
                  "scan_decision_calendar", decision_calendar)
    _patch_reader(monkeypatch, nightly_calendar_inputs,
                  "scan_calendar_row_inputs", calendar_row)
    _patch_reader(monkeypatch, nightly_quote_rows, "scan_quote_rows", quote_rows)

    events, refusals = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=fixture._DECISION.date().isoformat(),
        horizon_days=_HORIZON_DAYS)

    assert refusals["refusals"] == []
    assert {event["key"]["ticker"] for event in events} == {"AAA", "BBB"}
    counts = Counter(reads)
    first_build_pairs = (
        ("daily_market", "SPY"),
        (fixture.COMPUTED_MOVES_TABLE_NAME, "AAA"),
        (fixture.COMPUTED_MOVES_TABLE_NAME, "BBB"),
        (fixture.PRICE_HISTORY_TABLE_NAME, "AAA"),
        (fixture.PRICE_HISTORY_TABLE_NAME, "BBB"),
        ("daily_market", "AAA"),
        ("daily_market", "BBB"),
    )
    assert counts[("daily_market", "SPY")] == 1
    for pair in first_build_pairs[1:]:
        assert counts[pair] == 1, pair
    assert batches[("daily_market", "SPY")] == fixture._spy_rows(periods=300)
    event_bytes = json.dumps(
        events, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # Captured by the first fixture run, before the implementation cache edit.
    # Pin the pre-optimization event bytes so the shared read cannot change output.
    assert hashlib.sha256(event_bytes).hexdigest() == (
        "43e9c8530d6600ba49ead8aa61c24dda4175af38fcd47d8f00e6c73ceac69b74")

    again_events, again_refusals = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=fixture._DECISION.date().isoformat(),
        horizon_days=_HORIZON_DAYS)
    assert again_events == events
    assert again_refusals == refusals
    for pair in first_build_pairs:
        assert Counter(reads)[pair] == 2, pair


def test_panel_spy_cache_key_includes_snapshot_version_and_window():
    """The build-scoped SPY cache separates by snapshot identity, pinned daily_market
    dataset version and read window -- and never stubs the real panel path."""
    import dataclasses

    import tests.test_v2_features_panel_row_inputs as fixture

    class CountingRepository(fixture._FakeRepository):
        def __init__(self, snapshot, contracts, batches_by_key) -> None:
            super().__init__(snapshot, contracts, batches_by_key)
            self.spy_scans = 0

        def scan(self, query, *, table_name):
            if (table_name == "daily_market"
                    and query.key_filter
                    and query.key_filter[0].values == ("SPY",)):
                self.spy_scans += 1
            yield from super().scan(query, table_name=table_name)

    def _run(repository, snapshot, *, history_start):
        return panel_row_inputs.scan_panel_row(
            repository, snapshot, fixture._key(), decision_session=fixture._DECISION,
            history_start=history_start, spy_market_cache=shared)

    batches = fixture._default_batches()
    batches[("daily_market", "SPY")] = fixture._spy_rows(periods=300)
    original = fixture._snapshot()
    version_changed = dataclasses.replace(
        original,
        table_versions={
            **original.table_versions,
            "daily_market": dataclasses.replace(
                original.table_versions["daily_market"],
                dataset_version_id="dsv-dm-next"),
        })
    snapshot_changed = dataclasses.replace(original, snapshot_id="snap-panel-next")

    shared: dict = {}
    original_repo = CountingRepository(original, fixture._contracts(), batches)
    version_repo = CountingRepository(version_changed, fixture._contracts(), batches)
    snapshot_repo = CountingRepository(snapshot_changed, fixture._contracts(), batches)

    _run(original_repo, original, history_start=fixture._HISTORY_START)
    _run(original_repo, original,
         history_start=fixture._HISTORY_START + pd.Timedelta(days=1))
    _run(version_repo, version_changed, history_start=fixture._HISTORY_START)
    _run(snapshot_repo, snapshot_changed, history_start=fixture._HISTORY_START)

    assert original_repo.spy_scans == 2
    assert version_repo.spy_scans == 1
    assert snapshot_repo.spy_scans == 1
    assert len(shared) == 4


def test_earlier_success_does_not_hide_a_later_failure(tmp_path, monkeypatch):
    repository, snapshot = _snapshot(tmp_path, [
        _event_row("AAA", _MIDNIGHT, "BMO", event_id="aaa-bmo"),
        _event_row("BBB", _SECOND, "BMO", event_id="bbb-bmo"),
    ])
    rig = _ReaderRig(monkeypatch)
    problem = data_fail("RESULT_LIMIT_EXCEEDED", "synthetic repository failure")

    def fail_on_bbb(key):
        if key.ticker == "BBB":
            raise problem
        return _calendar_inputs(key)

    rig.calendar_row_hook = fail_on_bbb
    direct = rig.direct(repository, snapshot)
    aaa = [key for key in direct if key.ticker == "AAA"]
    assert aaa

    with pytest.raises(DataError) as excinfo:
        nrp.build_native_score_batch_events(repository, snapshot, as_of=_AS_OF,
                                            horizon_days=_HORIZON_DAYS)

    assert excinfo.value is problem
    assert [key for key in rig.calendar_row_calls if key.ticker == "AAA"] == aaa
    assert rig.calendar_row_calls[-1].ticker == "BBB"
    assert any(key.ticker == "AAA" for key in rig.panel_calls)
    assert any(key.ticker == "AAA" for key in rig.quote_calls)


def test_refusal_details_are_fixed_public_safe_text(tmp_path, monkeypatch):
    repository, snapshot = _snapshot(tmp_path, [
        _event_row("AAA", _MIDNIGHT, "BMO", event_id="aaa-bmo"),
        _event_row("BBB", _INTRADAY, "BMO", event_id="bbb-intraday"),
    ])
    rig = _ReaderRig(monkeypatch)

    def refuse_calendar(key):
        raise NativeScoreBatchRowRefusal(
            key, "NO_RESOLVABLE_EXPIRY", "SENSITIVE_EXCEPTION_TEXT_MUST_NOT_LEAK")

    rig.calendar_row_hook = refuse_calendar
    first = nrp.build_native_score_batch_events(repository, snapshot, as_of=_AS_OF,
                                                horizon_days=_HORIZON_DAYS)
    second = nrp.build_native_score_batch_events(repository, snapshot, as_of=_AS_OF,
                                                 horizon_days=_HORIZON_DAYS)
    assert first == second

    events_document, refusals_document = first
    assert events_document == []
    refusals = refusals_document["refusals"]
    assert {refusal["code"] for refusal in refusals} == {
        "NO_RESOLVABLE_EXPIRY", "INTRADAY_EVENT_NOT_ADMITTED"}
    _FIXED_DETAILS = {
        "NO_RESOLVABLE_EXPIRY":
            "no strategy-eligible listed expiry for the board request",
        "INTRADAY_EVENT_NOT_ADMITTED":
            "the board request carries an intraday event timestamp",
    }
    for refusal in refusals:
        detail = refusal["detail"]
        assert detail == _FIXED_DETAILS[refusal["code"]]
        assert "SENSITIVE_EXCEPTION_TEXT_MUST_NOT_LEAK" not in detail
        assert isinstance(detail, str) and detail.strip()
        assert "Traceback" not in detail
        assert "BoardRequest(" not in detail
        assert "NO_RESOLVABLE_EXPIRY:" not in detail
        assert refusal["key"]["ticker"] not in detail
        assert _INTRADAY_WIRE not in detail
    assert json.dumps(refusals_document)


def test_producer_recursively_converts_missing_and_numpy_panel_values(
        tmp_path, monkeypatch):
    import numpy as np

    repository, snapshot = _snapshot(tmp_path, [
        _event_row("AAA", _MIDNIGHT, "BMO", event_id="aaa-midnight"),
    ])
    _ReaderRig(monkeypatch)

    def special_panel(*args, **kwargs):
        return PanelRowInputs(
            panel_row={
                "date": _MIDNIGHT_WIRE,
                "missing_float": float("nan"),
                "missing_na": pd.NA,
                "missing_nat": pd.NaT,
                "count": np.int64(7),
                "asof": np.datetime64("2024-01-16"),
                "spy_ret252": 0.0,
            },
            panel_anchor=pd.Timestamp("2024-01-16"))

    _patch_reader(monkeypatch, panel_row_inputs, "scan_panel_row", special_panel)

    events_document, refusals_document = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=_HORIZON_DAYS)

    assert events_document and refusals_document["refusals"] == []
    for document in events_document:
        panel_row = document["panel_row"]
        assert panel_row["missing_float"] is None
        assert panel_row["missing_na"] is None
        assert panel_row["missing_nat"] is None
        assert panel_row["count"] == 7 and type(panel_row["count"]) is int
        assert panel_row["asof"] == "2024-01-16"
        assert panel_row["date"] == "2024-01-16"
        assert document["panel_anchor"] == "2024-01-16"
    json.dumps(events_document, allow_nan=False)


def test_typed_calendar_error_codes_become_per_key_refusals(tmp_path, monkeypatch):
    repository, snapshot = _snapshot(tmp_path, [
        _event_row("AAA", _MIDNIGHT, "BMO", event_id="aaa-bmo"),
    ])
    rig = _ReaderRig(monkeypatch)
    direct = rig.direct(repository, snapshot)
    strategies = [key.strategy for key in direct]
    assert len(strategies) == len(set(strategies)) > 1
    missing_strategy, conflict_strategy = strategies[0], strategies[1]

    def fail_calendar_rows(key):
        if key.strategy == missing_strategy:
            raise data_fail("EVENT_NOT_FOUND", "SENSITIVE_EVENT_ERROR")
        raise data_fail("IDENTITY_CONFLICT", "SENSITIVE_CONFLICT_ERROR")

    rig.calendar_row_hook = fail_calendar_rows

    events_document, refusals_document = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=_HORIZON_DAYS)

    assert events_document == []
    refusals = refusals_document["refusals"]
    assert {refusal["code"] for refusal in refusals} == {
        "EVENT_NOT_FOUND", "IDENTITY_CONFLICT"}
    assert refusals[strategies.index(conflict_strategy)]["code"] == "IDENTITY_CONFLICT"
    assert [refusal["code"] for refusal in refusals] == [
        "EVENT_NOT_FOUND" if key.strategy == missing_strategy else "IDENTITY_CONFLICT"
        for key in direct]
    refused_strategies = set(strategies)
    assert [refusal["key"] for refusal in refusals] == [
        _event_keylike(key) for key in direct if key.strategy in refused_strategies]

    fixed_details = {
        "EVENT_NOT_FOUND":
            "no exact calendar event for the board request in the snapshot",
        "IDENTITY_CONFLICT":
            "multiple exact calendar events for the board request",
    }
    for refusal in refusals:
        assert refusal["detail"] == fixed_details[refusal["code"]]
        assert "SENSITIVE_" not in refusal["detail"]
        assert isinstance(refusal["detail"], str) and refusal["detail"].strip()
    serialized = json.dumps(refusals_document)
    assert "SENSITIVE_EVENT_ERROR" not in serialized
    assert "SENSITIVE_CONFLICT_ERROR" not in serialized
    assert refusals_document["schema_version"] == _REFUSAL_SCHEMA


def test_insufficient_panel_history_refuses_every_request_before_context_reads(
        tmp_path, monkeypatch):
    repository, snapshot = _snapshot(tmp_path, [
        _event_row("AAA", _MIDNIGHT, "BMO", event_id="aaa-midnight"),
        _event_row("AAA", _INTRADAY, "BMO", event_id="aaa-intraday"),
    ])
    rig = _ReaderRig(monkeypatch)
    direct = rig.direct(repository, snapshot)
    midnight = [key for key in direct if key.event_date == _MIDNIGHT]
    intraday = [key for key in direct if key.event_date == _INTRADAY]
    assert midnight and intraday
    assert list(direct) == midnight + intraday

    short = _calendar(sessions=_PANEL_HISTORY_SESSIONS - 1)
    assert len([day for day in short.days if day <= _AS_OF]) == _PANEL_HISTORY_SESSIONS - 1

    def short_calendar(decision_session, event_through):
        return short

    rig.calendar_hook = short_calendar
    events_document, refusals_document = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=_HORIZON_DAYS)

    assert events_document == []
    assert refusals_document["schema_version"] == _REFUSAL_SCHEMA
    refusals = refusals_document["refusals"]
    assert len(rig.enum_calls) == 1
    assert rig.calendar_calls == [(_AS_OF, _MIDNIGHT_WIRE)]
    assert [refusal["key"] for refusal in refusals] == [_event_keylike(key) for key in direct]
    assert [refusal["code"] for refusal in refusals] == [
        "PANEL_HISTORY_NOT_AVAILABLE" if key.event_date == _MIDNIGHT
        else "INTRADAY_EVENT_NOT_ADMITTED" for key in direct]

    refused_midnight = [refusal for refusal in refusals
                        if refusal["code"] == "PANEL_HISTORY_NOT_AVAILABLE"]
    refused_intraday = [refusal for refusal in refusals
                        if refusal["code"] == "INTRADAY_EVENT_NOT_ADMITTED"]
    assert [refusal["key"] for refusal in refused_midnight] \
        == [_event_keylike(key) for key in midnight]
    assert [refusal["key"] for refusal in refused_intraday] \
        == [_event_keylike(key) for key in intraday]
    assert all(refusal["key"]["event_date"] == _INTRADAY_WIRE for refusal in refused_intraday)
    assert all(pd.Timestamp(refusal["key"]["event_date"]) == _INTRADAY
               for refusal in refused_intraday)

    fixed_details = {"PANEL_HISTORY_NOT_AVAILABLE": _PANEL_HISTORY_DETAIL,
                     "INTRADAY_EVENT_NOT_ADMITTED": _INTRADAY_DETAIL}
    for refusal in refusals:
        detail = refusal["detail"]
        assert detail == fixed_details[refusal["code"]]
        assert isinstance(detail, str) and detail.strip()
        assert "Traceback" not in detail
        assert "BoardRequest(" not in detail
        assert refusal["key"]["ticker"] not in detail
        assert _INTRADAY_WIRE not in detail
    assert json.dumps(refusals_document)

    assert rig.calendar_row_calls == []
    assert rig.panel_calls == []
    assert rig.quote_calls == []


def test_projected_decision_session_is_not_observed_history(tmp_path, monkeypatch):
    """``_AS_OF`` is in ``days`` only as a projected session because
    ``observed_through`` is the previous observed day: the as-of-side count clears
    the pinned floor while the decision date itself is unobserved history."""
    repository, snapshot = _snapshot(tmp_path, [
        _event_row("AAA", _MIDNIGHT, "BMO", event_id="aaa-midnight"),
    ])
    rig = _ReaderRig(monkeypatch)
    direct = rig.direct(repository, snapshot)
    midnight = [key for key in direct if key.event_date == _MIDNIGHT]
    assert midnight and len(midnight) == len(direct)

    history = pd.bdate_range(end=_AS_OF, periods=_PANEL_HISTORY_SESSIONS + 1)
    previous_observed = history[-2].date().isoformat()
    projected = CalendarSessions(
        days=tuple(day.date().isoformat() for day in history) + _PROJECTED_DAYS,
        observed_through=previous_observed)
    assert projected.observed_through == previous_observed < _AS_OF
    assert _AS_OF in projected.days
    assert len([day for day in projected.days if day <= _AS_OF]) \
        == _PANEL_HISTORY_SESSIONS + 1
    assert len([day for day in projected.days if day <= previous_observed]) == _PANEL_HISTORY_SESSIONS

    def projected_calendar(decision_session, event_through):
        return projected

    rig.calendar_hook = projected_calendar
    events_document, refusals_document = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=_HORIZON_DAYS)

    assert events_document == []
    assert refusals_document["schema_version"] == _REFUSAL_SCHEMA
    refusals = refusals_document["refusals"]
    assert len(rig.enum_calls) == 1
    assert rig.calendar_calls == [(_AS_OF, _MIDNIGHT_WIRE)]
    assert [refusal["key"] for refusal in refusals] == [
        _event_keylike(key) for key in midnight]
    assert [refusal["code"] for refusal in refusals] == [
        "PANEL_HISTORY_NOT_AVAILABLE"] * len(midnight)
    assert [refusal["detail"] for refusal in refusals] == [
        _PANEL_HISTORY_DETAIL] * len(midnight)
    assert json.dumps(refusals_document)

    assert rig.calendar_row_calls == []
    assert rig.panel_calls == []
    assert rig.quote_calls == []


def test_real_panel_reader_preserving_full_history(tmp_path, monkeypatch):
    """The real panel reader keeps available moves older than the regime window."""
    import tests.test_v2_features_panel_row_inputs as fixture
    from engine.v2.ops.nightly_raw_row_producer import _shared_panel_rows

    history = pd.bdate_range(end=fixture._DECISION, periods=300)
    old_day = history[20].date().isoformat()
    assert old_day < history[-253].date().isoformat()
    batches = fixture._default_batches()
    batches[("daily_market", "SPY")] = fixture._spy_rows(periods=300)
    batches[(fixture.COMPUTED_MOVES_TABLE_NAME, "AAA")] = \
        fixture._computed_rows([(old_day, 4.0, False)])
    snapshot = fixture._snapshot()
    
    # Create an interval-aware repository subclass
    class IntervalAwareRepository(fixture._FakeRepository):
        def scan(self, query, *, table_name):
            # Apply the interval/key predicate filtering if query has time_interval
            if query.time_interval is not None:
                ticker = query.key_filter[0].values[0]
                filtered_rows = []
                rows = self._batches.get((table_name, ticker), [])
                for row in rows:
                    if fixture._bound_row_selected(row, query.key_filter, query.time_interval):
                        filtered_rows.append(row)
                yield fixture._Batch(filtered_rows)
            else:
                # Fall back to original behavior
                yield from super().scan(query, table_name=table_name)

        def fragment_records(self, snapshot_ref, table_name):
            """Catalog-free membership metadata for the pinned table: one
            partition-per-ticker record per stored ``(candidate_table,
            ticker) -> rows`` batch matching ``table_name``, the shape
            ``tickers_with_price_history`` reads through the real
            ``Repository``."""
            return tuple(
                SimpleNamespace(partition_key=ticker, row_count=len(rows))
                for (candidate_table, ticker), rows in self._batches.items()
                if candidate_table == table_name)

    repository = IntervalAwareRepository(snapshot, fixture._contracts(), batches)

    def enumerate_one(repo, snap, *, as_of, horizon_days, tickers=None):
        from engine.v2.ops.native_board_universe import BoardRequest
        return [BoardRequest(ticker="AAA", strategy="BFLY-P",
                             event_date=fixture._EVENT, session="AMC")]

    def decision_calendar(repo, snap, *, decision_session, event_through):
        days = tuple(day.date().isoformat() for day in history) + _PROJECTED_DAYS
        return CalendarSessions(days=days,
                                observed_through=fixture._DECISION.date().isoformat())

    calendar_row_calls: list = []

    def calendar_row(repo, snap, key, *, decision_session, calendar):
        calendar_row_calls.append(key)
        day = key.event_date.date().isoformat()
        observed = fixture._DECISION.date().isoformat()
        return CalendarRowInputs(calendar_revision="ev-rev-1", calendar_row={
            "event_id": f"{key.ticker}-{key.strategy}", "ticker": key.ticker,
            "event_date": day, "session": key.session, "entry_date": observed,
            "exit_date": day, "expiry": day, "spot": 10.0,
            "calendar_observed_through": observed})

    def quote_rows(repo, snap, key, *, expiry, decision_session):
        observed = fixture._DECISION.date().isoformat()
        return QuoteRowInputs(quote_rows=({
            "ticker": key.ticker, "right": "C", "strike": 100.0,
            "expiry": expiry, "bid": 1.0, "ask": 1.2,
            "observed_at": observed},), quote_status="recorded")

    _patch_reader(monkeypatch, nightly_raw_rows,
                  "scan_forward_board_requests", enumerate_one)
    _patch_reader(monkeypatch, nightly_calendar_inputs,
                  "scan_decision_calendar", decision_calendar)
    _patch_reader(monkeypatch, nightly_calendar_inputs,
                  "scan_calendar_row_inputs", calendar_row)
    _patch_reader(monkeypatch, nightly_quote_rows, "scan_quote_rows", quote_rows)

    # First, run the unrestricted call and verify it passes its existing full-history assertions
    events, refusals = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=fixture._DECISION.date().isoformat(),
        horizon_days=_HORIZON_DAYS)
    assert len(events) == 1
    assert refusals["refusals"] == []
    assert events[0]["panel_row"]["n_prior"] == 1
    assert events[0]["panel_row"]["mean_prior_move"] == 4.0

    # Add a negative control that forces the same producer path to call _shared_panel_rows 
    # with history_start=history[-253] (the defective trailing cutoff)
    def mock_shared_panel_rows_with_truncated_history(repository, snapshot, keys, *, decision_session, history_start):
        # Force history_start to be the defective trailing cutoff
        truncated_history_start = history[-253].date().isoformat()
        return _shared_panel_rows(repository, snapshot, keys, decision_session=decision_session, history_start=truncated_history_start)

    # Use monkeypatch.context() to temporarily override _shared_panel_rows
    with monkeypatch.context() as m:
        m.setattr("engine.v2.ops.nightly_raw_row_producer._shared_panel_rows", mock_shared_panel_rows_with_truncated_history)
        
        # Invoke build_native_score_batch_events again under that temporary override
        events_negative, refusals_negative = nrp.build_native_score_batch_events(
            repository, snapshot, as_of=fixture._DECISION.date().isoformat(),
            horizon_days=_HORIZON_DAYS)
        
        # The negative control should show different results (truncated history)
        # The old row should be excluded due to the truncated history
        assert len(events_negative) == 1
        assert refusals_negative["refusals"] == []
        # The negative-control event should have the actual truncated result
        assert events_negative[0]["panel_row"]["n_prior"] == 0
        assert events_negative[0]["panel_row"]["mean_prior_move"] is None
        
        # Now assert that the full-history expectation fails for the negative control
        # by checking that the original assertion would fail
        # In the original test, we expect n_prior == 1 and mean_prior_move == 4.0
        # But with the truncated history, we get n_prior == 0 and mean_prior_move is None
        # So asserting the original expectation should fail
        with pytest.raises(AssertionError):
            assert events_negative[0]["panel_row"]["n_prior"] == 1
        with pytest.raises(AssertionError):
            assert events_negative[0]["panel_row"]["mean_prior_move"] == 4.0

    # Regression: sufficient decision-calendar history must not admit a row
    # whose real panel read cannot produce the longest regime value. Same
    # calendar history and price-history coverage, a thin SPY daily_market
    # batch, and the real scan_panel_row path -- never a mocked
    # _shared_panel_rows or panel reader.
    thin_batches = dict(batches)
    thin_batches[("daily_market", "SPY")] = fixture._spy_rows(periods=22)
    thin_repository = IntervalAwareRepository(snapshot, fixture._contracts(),
                                              thin_batches)
    calendar_calls_before_thin = len(calendar_row_calls)

    thin_events, thin_refusals_document = nrp.build_native_score_batch_events(
        thin_repository, snapshot, as_of=fixture._DECISION.date().isoformat(),
        horizon_days=_HORIZON_DAYS)

    assert thin_events == []
    thin_refusals = thin_refusals_document["refusals"]
    assert len(thin_refusals) == 1
    assert thin_refusals[0]["key"] == {
        "ticker": "AAA", "strategy": "BFLY-P",
        "event_date": fixture._EVENT.date().isoformat(), "session": "AMC"}
    assert thin_refusals[0]["code"] == "PANEL_HISTORY_NOT_AVAILABLE"
    assert thin_refusals[0]["detail"] == _PANEL_HISTORY_DETAIL
    assert len(calendar_row_calls) == calendar_calls_before_thin


def test_exact_mixed_refusal_order(tmp_path, monkeypatch):
    """Refusals preserve request order across calendar and intraday categories."""
    repository, snapshot = _snapshot(tmp_path, [
        _event_row("MIXED", _MIDNIGHT, "BMO", event_id="mixed-midnight"),
        _event_row("MIXED", _INTRADAY, "BMO", event_id="mixed-intraday"),
    ])
    rig = _ReaderRig(monkeypatch)

    def calendar_row(key):
        if key.event_date == _MIDNIGHT:
            raise data_fail("EVENT_NOT_FOUND",
                            "no exact calendar event for the board request in the snapshot")
        return _calendar_inputs(key)

    rig.calendar_row_hook = calendar_row
    events, refusal_document = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=_HORIZON_DAYS)
    direct = rig.direct(repository, snapshot)
    assert events == []
    refusals = refusal_document["refusals"]
    actual = [(_identity(item["key"]), item["code"]) for item in refusals]
    expected = [
        (_identity(_event_keylike(key)),
         "EVENT_NOT_FOUND" if key.event_date == _MIDNIGHT
         else "INTRADAY_EVENT_NOT_ADMITTED")
        for key in direct
    ]
    assert actual == expected
    assert len(refusals) == len(direct)
