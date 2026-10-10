"""Regression coverage for a missing ``price_history`` ticker in the native
producer, and for one whose series is present but unusable.

``engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events``
refuses ``PRICE_HISTORY_NOT_AVAILABLE`` for a ticker absent from the pinned
``price_history`` table (via ``tickers_with_price_history``), while every other
present ticker still composes into an event; the refusal decodes through
``_decode_producer_refusals`` and lands in the real parity worker's report
under ``native_refused``. A ticker whose series is present but lacks an exact-session row or has an
unusable ``close_raw`` receives the per-key ``PRICE_HISTORY_NOT_AVAILABLE``
refusal. Repository failures during the real spot read still fail the whole
producer build.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.price_history_table import (
    PRICE_HISTORY_CONTRACT,
    PRICE_HISTORY_TABLE_NAME,
)
from engine.v2.data.repository import Repository
from engine.v2.features.panel_row_inputs import PanelRowInputs
from engine.v2.foundation.market_calendar import CalendarSessions
from engine.v2.ops import nightly_calendar_inputs as nci
from engine.v2.ops import nightly_raw_row_producer as nrp
from engine.v2.ops.decision_validation import population_key
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_parity_report import run_native_parity_worker
from engine.v2.ops.native_score_batch import (
    _decode_producer_refusals,
    run_native_score_batch_worker,
)
from engine.v2.ops.nightly_quote_rows import QuoteRowInputs
from engine.v2.ops.nightly_raw_rows import CalendarRowInputs
from engine.v2.parity.dimensions import (
    ANALOG_FIELDS,
    FINANCIAL_FIELDS,
    FORECAST_FIELDS,
    GATE_FIELDS,
    SIMULATION_FIELDS,
)
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)
from tests.test_v2_ops_native_score_batch import _stage_release

_AS_OF = "2024-01-05"
_EVENT_DATE = "2024-01-16"
_HISTORY_TICKERS = ("AAA", "BBB")
_MISSING_TICKER = "CCC"
_STRATEGY = "STR-THRU"
_SESSION = "BMO"
_CODE = "PRICE_HISTORY_NOT_AVAILABLE"
_DETAIL = "the pinned snapshot has no price history for this ticker"
_PH_REF = contract_ref_for(PRICE_HISTORY_CONTRACT)
#: The one fixed value every legacy fixture row carries in every canonical
#: parity comparison field, so the parity report compares native output
#: against a stable independent fixture rather than a copy of itself.
_LEGACY_FIXTURE_VALUE = 0.0


def _history_row(ticker: str, *, close_raw: float | None = 100.0) -> dict:
    """One pinned ``price_history`` row for ``ticker`` at the fixture as-of."""
    return {
        "ticker": ticker,
        "date": _AS_OF,
        "close_adj": 100.0,
        "close_raw": close_raw,
        "high_raw": 101.0,
        "retrieved_at": "2024-01-01T00:00:00Z",
        "deleted": False,
        "source_kind": "yfinance",
        "source_hash": fake_hash(f"px-{ticker}"),
        "capture_id": f"capture-{ticker}",
    }


def _repository_with_history(tmp_path, rows: list[dict]):
    """Build a real repository and snapshot pinning ONLY the ``price_history``
    table built from ``rows`` (no ``option_chains`` table)."""
    conn, clock, store = catalog_and_store(tmp_path)
    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)
    records = [
        publish_and_inspect(
            store, PRICE_HISTORY_CONTRACT, _PH_REF, ticker_rows, ticker)
        for ticker, ticker_rows in sorted(by_ticker.items())
    ]
    snapshot = commit_tables(
        conn,
        clock,
        {PRICE_HISTORY_TABLE_NAME: records},
        {PRICE_HISTORY_TABLE_NAME: PRICE_HISTORY_CONTRACT},
        scope="test",
    )
    return Repository(conn, store), snapshot


def _canonical_key(ticker: str) -> str:
    """The 4-field native score-batch canonical key for ``ticker``."""
    return f"{ticker}|{_STRATEGY}|{_EVENT_DATE}|{_SESSION}"


def _population_key(ticker: str) -> str:
    """The 3-field legacy population key for ``ticker``."""
    return population_key(
        {"ticker": ticker, "strategy": _STRATEGY, "event_date": _EVENT_DATE})


def _requests(tickers: tuple[str, ...]) -> tuple[BoardRequest, ...]:
    """One forward-board request per ticker, sharing the fixture identity."""
    return tuple(
        BoardRequest(
            ticker=ticker,
            strategy=_STRATEGY,
            event_date=pd.Timestamp(_EVENT_DATE),
            session=_SESSION,
        )
        for ticker in tickers
    )


def _patch_context(
        monkeypatch, requests: tuple[BoardRequest, ...], *, real_spot=False):
    """Patch the producer's board/calendar/panel scans onto fixture data;
    with ``real_spot`` False the calendar/quote scans are faked too, with it
    True the real spot and quote paths run."""
    history_days = tuple(
        value.date().isoformat()
        for value in pd.bdate_range(end=_AS_OF, periods=253))
    calendar = CalendarSessions(
        days=history_days + (
            "2024-01-08", "2024-01-09", "2024-01-10",
            "2024-01-11", "2024-01-12", _EVENT_DATE),
        observed_through=_AS_OF)

    def board_requests(repository, snapshot, *, as_of, horizon_days,
                       tickers=None):
        """Return the requested fixture board requests verbatim."""
        return requests

    def decision_calendar(repository, snapshot, *, decision_session,
                          event_through):
        """Return the fixture decision calendar covering the event date."""
        return calendar

    def panels(repository, snapshot, keys, *, decision_session, history_start):
        """One shared fixture panel row per requested key."""
        return {
            nrp._panel_marker(key): PanelRowInputs(
                panel_row={
                    "date": key.event_date.date().isoformat(),
                    "event_tag": key.ticker,
                    "spy_ret252": 0.0,
                    "x": 1.0,
                },
                panel_anchor=pd.Timestamp(_AS_OF),
            )
            for key in keys
        }

    def forbidden_quotes(repository, snapshot, key, *, expiry,
                        decision_session):
        """Fail loudly if a quote scan is reached after a spot refusal."""
        raise AssertionError("quote scan must not be reached after a spot refusal")

    monkeypatch.setattr(nrp, "scan_forward_board_requests", board_requests)
    monkeypatch.setattr(nrp, "scan_decision_calendar", decision_calendar)
    monkeypatch.setattr(nrp, "_shared_panel_rows", panels)
    monkeypatch.setattr(nrp, "scan_quote_rows", forbidden_quotes)
    if real_spot:
        return

    def calendar_row(repository, snapshot, key, *, decision_session, calendar):
        """One fixture calendar row with a usable pinned spot for ``key``."""
        return CalendarRowInputs(
            calendar_revision="cal-v1",
            calendar_row={
                "event_id": f"evt-{key.ticker}",
                "ticker": key.ticker,
                "event_date": _EVENT_DATE,
                "session": key.session,
                "entry_date": _AS_OF,
                "exit_date": _EVENT_DATE,
                "expiry": _EVENT_DATE,
                "spot": 100.0,
                "calendar_observed_through": _AS_OF,
            })

    def quotes(repository, snapshot, key, *, expiry, decision_session):
        """One fixture call/put quote pair per key at the fixture expiry."""
        rows = tuple(
            {
                "ticker": key.ticker,
                "right": right,
                "strike": 100.0,
                "expiry": expiry,
                "bid": 1.0,
                "ask": 1.2,
                "observed_at": _AS_OF,
            }
            for right in ("C", "P")
        )
        return QuoteRowInputs(quote_rows=rows, quote_status="recorded")

    monkeypatch.setattr(nrp, "scan_calendar_row_inputs", calendar_row)
    monkeypatch.setattr(nrp, "scan_quote_rows", quotes)


def _legacy_score_rows(tickers: tuple[str, ...]) -> list[dict]:
    """Independent legacy fixture rows: identifiers come from the requested
    ticker constants only, never from native record values, and every
    canonical parity dimension field carries the fixed
    ``_LEGACY_FIXTURE_VALUE`` -- so the parity report compares native output
    against a stable independent fixture rather than a copy of itself."""
    rows = []
    for ticker in tickers:
        row = {
            "ticker": ticker,
            "strategy": _STRATEGY,
            "event_date": _EVENT_DATE,
        }
        for fields in (FORECAST_FIELDS, SIMULATION_FIELDS, FINANCIAL_FIELDS,
                       GATE_FIELDS, ANALOG_FIELDS):
            row.update(dict.fromkeys(fields, _LEGACY_FIXTURE_VALUE))
        rows.append(row)
    return rows


def test_missing_ticker_history_is_refused_while_other_tickers_compose(
        tmp_path, monkeypatch):
    """A ticker absent from the pinned ``price_history`` table is refused
    per-key while the present tickers still compose, score through the real
    score worker, and surface as ``native_refused`` entries in the real
    parity worker's report; the report's comparison mismatches all stem from
    the independent fixed legacy fixture."""
    repository, snapshot = _repository_with_history(
        tmp_path, [_history_row(ticker) for ticker in _HISTORY_TICKERS])
    _patch_context(
        monkeypatch, _requests((*_HISTORY_TICKERS, _MISSING_TICKER)))

    events, refusals_document = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=30)

    assert [event["key"]["ticker"] for event in events] == list(_HISTORY_TICKERS)
    assert refusals_document["refusals"] == [{
        "key": {
            "ticker": _MISSING_TICKER,
            "strategy": _STRATEGY,
            "event_date": _EVENT_DATE,
            "session": _SESSION,
        },
        "code": _CODE,
        "detail": _DETAIL,
    }]

    decoded = _decode_producer_refusals(refusals_document)
    assert [refusal.key.ticker for refusal in decoded] == [_MISSING_TICKER]
    assert [refusal.code for refusal in decoded] == [_CODE]
    assert [refusal.detail for refusal in decoded] == [_DETAIL]

    later_root = tmp_path / "later-snapshot"
    later_root.mkdir()
    later_repository, later_snapshot = _repository_with_history(
        later_root,
        [_history_row(ticker) for ticker in (*_HISTORY_TICKERS, _MISSING_TICKER)])
    later_events, later_refusals = nrp.build_native_score_batch_events(
        later_repository, later_snapshot, as_of=_AS_OF, horizon_days=30)
    assert len(later_events) == 3
    assert later_refusals["refusals"] == []

    job_root = tmp_path / "score-job"
    job_root.mkdir()
    (job_root / "events.json").write_text(json.dumps(events))
    (job_root / "producer_refusals.json").write_text(
        json.dumps(refusals_document))
    release_root = tmp_path / "release"
    release_root.mkdir()
    _stage_release(release_root)
    parameters = {
        "expected_ids": (f"{_AS_OF}|scope",),
        "release_root": str(release_root),
        "as_of": _AS_OF,
        "snapshot_id": snapshot.snapshot_id,
        "calendar_revision": "cal-v1",
        "feature_names": ("x",),
        "gate_policy": {"STR-THRU": {"threshold": 0.0}},
    }

    score_result = run_native_score_batch_worker(parameters, job_root)
    records_document = json.loads((job_root / "records.json").read_text())
    refusals_document = json.loads((job_root / "refusals.json").read_text())
    assert score_result["completed_ids"] == list(parameters["expected_ids"])
    assert score_result["no_work"] is False
    assert score_result["refused"] == [_CODE]
    worker_records = records_document["records"]
    worker_refusals = refusals_document["refusals"]
    assert records_document["schema_version"] == "native_score_batch_records.v2.0"
    assert records_document["authoritative"] is False
    assert refusals_document["schema_version"] == "native_score_batch_refusals.v2.0"
    assert refusals_document["unkeyable_refusals"] == []
    assert sorted(worker_records) == [
        _canonical_key(ticker) for ticker in _HISTORY_TICKERS]
    assert worker_refusals == {
        _canonical_key(_MISSING_TICKER): {"code": _CODE, "detail": _DETAIL}}

    score_rows = _legacy_score_rows((*_HISTORY_TICKERS, _MISSING_TICKER))
    (job_root / "score.json").write_text(
        json.dumps({"rows": score_rows}, sort_keys=True))
    parity_result = run_native_parity_worker(parameters, job_root)
    assert parity_result["completed_ids"] == list(parameters["expected_ids"])
    assert parity_result["no_work"] is False
    report = json.loads(
        (job_root / "native_parity_report.json").read_text())
    refused = report["native_refused"]
    assert report["compared"] == [
        _population_key(ticker) for ticker in _HISTORY_TICKERS]
    assert report["mismatches"]
    for mismatch in report["mismatches"]:
        assert mismatch["row_key"] in report["compared"]
        assert mismatch["values"]
        assert all(values["legacy"] == _LEGACY_FIXTURE_VALUE
                   for values in mismatch["values"].values())
    assert report["only_legacy"] == []
    assert report["only_native"] == []
    assert [entry["row_key"] for entry in refused] == [
        _population_key(_MISSING_TICKER)]
    assert [entry["refusal_code"] for entry in refused] == [_CODE]
    assert [entry["ticker"] for entry in refused] == [_MISSING_TICKER]
    assert [entry["reason"] for entry in refused] == [_CODE]
    assert report["native_refused_unmatched"] == []


def test_missing_option_chains_still_fails_when_every_ticker_is_refused(
        tmp_path, monkeypatch):
    """When price-history admission refuses every non-intraday request there
    is no event left to compose, so the producer stops being per-key
    forgiving: this fixture snapshot pins ``price_history`` only, and its
    absent ``option_chains`` table propagates as ``CONTRACT_MISMATCH``
    instead of the build returning a per-ticker refusal document."""
    repository, snapshot = _repository_with_history(
        tmp_path, [_history_row(ticker) for ticker in _HISTORY_TICKERS])
    _patch_context(monkeypatch, _requests((_MISSING_TICKER,)))

    with pytest.raises(DataError) as caught:
        nrp.build_native_score_batch_events(
            repository, snapshot, as_of=_AS_OF, horizon_days=30)

    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.code != _CODE


def test_present_but_unusable_history_is_refused_per_key(tmp_path, monkeypatch):
    """The real exact-session reader turns a null raw close into the existing
    per-key producer refusal, without masking any repository exception."""
    ticker = "BAD"
    repository, snapshot = _repository_with_history(
        tmp_path, [_history_row(ticker, close_raw=None)])
    _patch_context(monkeypatch, _requests((ticker,)), real_spot=True)

    events, refusals = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=30)

    assert events == []
    assert refusals["refusals"] == [{
        "key": {"ticker": ticker, "strategy": _STRATEGY,
                "event_date": _EVENT_DATE, "session": _SESSION},
        "code": _CODE,
        "detail": "the pinned snapshot has no usable exact-session close for this ticker",
    }]


def test_missing_exact_session_spot_is_refused_and_other_ticker_scores(
        tmp_path, monkeypatch):
    """A real pinned date gap refuses one key while its exact-session sibling
    reaches the real score worker."""
    stale = _history_row("GAP")
    stale["date"] = "2024-01-04"
    repository, snapshot = _repository_with_history(
        tmp_path, [stale, _history_row("GOOD")])
    _patch_context(monkeypatch, _requests(("GAP", "GOOD")), real_spot=True)

    monkeypatch.setattr(nci, "scan_candidate_expiries",
                        lambda repository, snapshot, key, *, decision_session: (_EVENT_DATE,))

    def calendar_row(repository, snapshot, key, *, entry_date, exit_date,
                     expiry, spot, calendar_observed_through):
        return CalendarRowInputs(
            calendar_revision="cal-v1",
            calendar_row={
                "event_id": f"evt-{key.ticker}", "ticker": key.ticker,
                "event_date": _EVENT_DATE, "session": key.session,
                "entry_date": entry_date, "exit_date": exit_date,
                "expiry": expiry, "spot": spot,
                "calendar_observed_through": calendar_observed_through,
            })

    monkeypatch.setattr(nci, "scan_calendar_row", calendar_row)
    monkeypatch.setattr(nci, "planned_exit_date", lambda key, calendar: _EVENT_DATE)

    def quotes(repository, snapshot, key, *, expiry, decision_session):
        rows = tuple({
            "ticker": key.ticker, "right": right, "strike": 100.0,
            "expiry": expiry, "bid": 1.0, "ask": 1.2,
            "observed_at": _AS_OF,
        } for right in ("C", "P"))
        return QuoteRowInputs(quote_rows=rows, quote_status="recorded")

    monkeypatch.setattr(nrp, "scan_quote_rows", quotes)
    events, producer_refusals = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=_AS_OF, horizon_days=30)

    assert [event["key"]["ticker"] for event in events] == ["GOOD"]
    assert producer_refusals["refusals"] == [{
        "key": {"ticker": "GAP", "strategy": _STRATEGY,
                "event_date": _EVENT_DATE, "session": _SESSION},
        "code": _CODE,
        "detail": "the pinned snapshot has no usable exact-session close for this ticker",
    }]

    job_root = tmp_path / "score-job"
    job_root.mkdir()
    (job_root / "events.json").write_text(json.dumps(events))
    (job_root / "producer_refusals.json").write_text(json.dumps(producer_refusals))
    release_root = tmp_path / "release"
    release_root.mkdir()
    _stage_release(release_root)
    parameters = {
        "expected_ids": (f"{_AS_OF}|scope",),
        "release_root": str(release_root), "as_of": _AS_OF,
        "snapshot_id": snapshot.snapshot_id, "calendar_revision": "cal-v1",
        "feature_names": ("x",),
        "gate_policy": {"STR-THRU": {"threshold": 0.0}},
    }
    result = run_native_score_batch_worker(parameters, job_root)
    records = json.loads((job_root / "records.json").read_text())["records"]
    score_refusals = json.loads((job_root / "refusals.json").read_text())["refusals"]
    assert result["completed_ids"] == list(parameters["expected_ids"])
    assert sorted(records) == [_canonical_key("GOOD")]
    assert score_refusals == {
        _canonical_key("GAP"): {
            "code": _CODE,
            "detail": "the pinned snapshot has no usable exact-session close for this ticker",
        }}
    (job_root / "score.json").write_text(json.dumps({
        "rows": _legacy_score_rows(("GAP", "GOOD"))}, sort_keys=True))
    parity_result = run_native_parity_worker(parameters, job_root)
    report = json.loads(
        (job_root / "native_parity_report.json").read_text())
    assert parity_result["completed_ids"] == list(parameters["expected_ids"])
    assert [entry["ticker"] for entry in report["native_refused"]] == ["GAP"]
    assert [entry["refusal_code"] for entry in report["native_refused"]] == [_CODE]
    assert report["native_refused_unmatched"] == []


def test_repository_failure_during_spot_read_still_fails_the_batch(
        tmp_path, monkeypatch):
    """A genuine typed repository scan failure is not translated to a row refusal."""
    ticker = "FAIL"
    repository, snapshot = _repository_with_history(
        tmp_path, [_history_row(ticker)])
    _patch_context(monkeypatch, _requests((ticker,)), real_spot=True)
    assert any(record.partition_key == ticker and record.row_count > 0
               for record in repository.fragment_records(
                   snapshot, PRICE_HISTORY_TABLE_NAME))
    scan_calls = []

    def failed_scan(*args, **kwargs):
        scan_calls.append(True)
        raise data_fail("CONTRACT_MISMATCH", "synthetic repository scan failure")

    monkeypatch.setattr(repository, "scan", failed_scan)
    with pytest.raises(DataError) as caught:
        nrp.build_native_score_batch_events(
            repository, snapshot, as_of=_AS_OF, horizon_days=30)
    assert scan_calls == [True]
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.problem.message == "synthetic repository scan failure"


def test_malformed_history_is_rejected_before_snapshot_admission(tmp_path):
    """The real fragment validator rejects malformed non-finite stored history
    before it can ever be pinned into a snapshot."""
    ticker = "NONFINITE"
    with pytest.raises(DataError) as caught:
        _repository_with_history(
            tmp_path, [_history_row(ticker, close_raw=float("inf"))])

    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.problem.message == (
        "close_raw: an infinite value is never valid")
