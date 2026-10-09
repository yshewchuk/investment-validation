"""Regression coverage for a missing ``price_history`` ticker in the native
producer, and for one whose series is present but unusable.

``engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events``
refuses ``PRICE_HISTORY_NOT_AVAILABLE`` for a ticker absent from the pinned
``price_history`` table (via ``tickers_with_price_history``), while every other
present ticker still composes into an event; the refusal decodes through
``_decode_producer_refusals`` and lands in the real parity worker's report
under ``native_refused``. A ticker whose series IS present but whose exact-
session ``close_raw`` is unusable is NOT an absence -- the real
``scan_calendar_row_inputs`` / ``get_price_series`` path raises
``CONTRACT_MISMATCH`` and the whole producer build propagates it.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.price_history_table import (
    PRICE_HISTORY_CONTRACT,
    PRICE_HISTORY_TABLE_NAME,
)
from engine.v2.data.repository import Repository
from engine.v2.features.panel_row_inputs import PanelRowInputs
from engine.v2.foundation.market_calendar import CalendarSessions
from engine.v2.ops import nightly_raw_row_producer as nrp
from engine.v2.ops.decision_validation import population_key
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_parity_report import (
    _native_comparison_row,
    run_native_parity_worker,
)
from engine.v2.ops.native_score_batch import (
    _decode_producer_refusals,
    run_native_score_batch_worker,
)
from engine.v2.ops.nightly_quote_rows import QuoteRowInputs
from engine.v2.ops.nightly_raw_rows import CalendarRowInputs
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


def _history_row(ticker: str, *, close_raw: float | None = 100.0) -> dict:
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
    return f"{ticker}|{_STRATEGY}|{_EVENT_DATE}|{_SESSION}"


def _population_key(ticker: str) -> str:
    return population_key(
        {"ticker": ticker, "strategy": _STRATEGY, "event_date": _EVENT_DATE})


def _requests(tickers: tuple[str, ...]) -> tuple[BoardRequest, ...]:
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
        return requests

    def decision_calendar(repository, snapshot, *, decision_session,
                          event_through):
        return calendar

    def panels(repository, snapshot, keys, *, decision_session, history_start):
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
        raise AssertionError("quote scan must not be reached after a spot refusal")

    monkeypatch.setattr(nrp, "scan_forward_board_requests", board_requests)
    monkeypatch.setattr(nrp, "scan_decision_calendar", decision_calendar)
    monkeypatch.setattr(nrp, "_shared_panel_rows", panels)
    monkeypatch.setattr(nrp, "scan_quote_rows", forbidden_quotes)
    if real_spot:
        return

    def calendar_row(repository, snapshot, key, *, decision_session, calendar):
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


def _legacy_score_rows(records: dict[str, object]) -> list[dict]:
    rows = []
    for canonical_key, record in records.items():
        ticker, strategy, event_date, _session = canonical_key.split("|")
        row = {
            "ticker": ticker,
            "strategy": strategy,
            "event_date": event_date,
        }
        row.update(_native_comparison_row(record))
        rows.append(row)
    rows.append({
        "ticker": _MISSING_TICKER,
        "strategy": _STRATEGY,
        "event_date": _EVENT_DATE,
    })
    return rows


def test_missing_ticker_history_is_refused_while_other_tickers_compose(
        tmp_path, monkeypatch):
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

    score_rows = _legacy_score_rows(worker_records)
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
    assert report["mismatches"] == []
    assert report["only_legacy"] == []
    assert report["only_native"] == []
    assert [entry["row_key"] for entry in refused] == [
        _population_key(_MISSING_TICKER)]
    assert [entry["refusal_code"] for entry in refused] == [_CODE]
    assert [entry["ticker"] for entry in refused] == [_MISSING_TICKER]
    assert [entry["reason"] for entry in refused] == [_CODE]
    assert report["native_refused_unmatched"] == []


def test_present_but_unusable_history_still_fails_the_producer(
        tmp_path, monkeypatch):
    ticker = "BAD"
    repository, snapshot = _repository_with_history(
        tmp_path, [_history_row(ticker, close_raw=None)])
    _patch_context(monkeypatch, _requests((ticker,)), real_spot=True)

    with pytest.raises(DataError) as caught:
        nrp.build_native_score_batch_events(
            repository, snapshot, as_of=_AS_OF, horizon_days=30)

    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.problem.message == (
        "exact-session pinned spot close is unusable")


def test_malformed_present_history_still_fails_the_producer(
        tmp_path, monkeypatch):
    """A malformed-but-schema-valid series is present, never absent: the real
    membership check admits the ticker, and the real ``get_price_series`` read
    plus ``scan_calendar_row_inputs`` spot conversion refuses, so the producer
    build propagates the error instead of refusing
    ``PRICE_HISTORY_NOT_AVAILABLE``. The pinned fragment writer builds a
    ``float64`` array for ``close_raw`` and rejects a nonnumeric string before
    any reader sees it, so the malformed value here is a schema-valid
    non-finite price."""
    ticker = "NONFINITE"
    repository, snapshot = _repository_with_history(
        tmp_path, [_history_row(ticker, close_raw=float("inf"))])
    _patch_context(monkeypatch, _requests((ticker,)), real_spot=True)

    with pytest.raises(DataError) as caught:
        nrp.build_native_score_batch_events(
            repository, snapshot, as_of=_AS_OF, horizon_days=30)

    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.code != _CODE
    assert caught.value.problem.message == (
        "exact-session pinned spot close is unusable")
    assert nrp.tickers_with_price_history(
        repository, snapshot, {ticker}) == frozenset({ticker})
