"""Direct tests for ``nightly_quote_rows.scan_quote_rows`` (cutover PR-6
slice 2): a real catalog + ArtifactStore snapshot over synthetic
``option_chains`` fragments -- exact ``(ticker, obs_date)`` matching (never a
lookback), the requested-expiry filter, ticker isolation, null quotes, and
every documented refusal.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd
import pytest

from engine.v2.contracts import KeyPredicate
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.nightly_quote_rows import (
    QuoteRowInputs,
    _BATCH_CAP,
    _RESULT_CAP,
    scan_quote_rows,
)
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

_CHAINS = contract_for("option_chains")
_CHAINS_REF = contract_ref_for(_CHAINS)
_DAILY = contract_for("daily_market")

_EXPIRY = "2024-02-16"
_SESSION = "2024-01-05"
_KEY = BoardRequest("AAA", "STR-THRU", pd.Timestamp("2024-01-05"), "BMO")


def _chain_row(ticker: str, obs_date: datetime, expiry: datetime, strike: float, *,
               right="C", bid=1.0, ask=1.2, mid=1.1) -> dict:
    return dict(ticker=ticker, obs_date=obs_date, year=obs_date.year, expiry=expiry,
                dte=(expiry - obs_date).days, strike=strike, right=right, bid=bid, ask=ask, mid=mid,
                iv=30.0, delta=0.5, spot=100.0, src="orats", src_file="f.parquet", chain_kind="entry",
                volume=None, open_interest=None, bid_size=None, ask_size=None,
                quote_repaired=False)


def _chain_snapshot(tmp_path, rows):
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(store, _CHAINS, _CHAINS_REF, rows, "2024")
    snap = commit_tables(conn, clock, {"option_chains": [record]}, {"option_chains": _CHAINS})
    return Repository(conn, store), snap


def _snapshot_without_chains(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snap = commit_tables(conn, clock, {"daily_market": []}, {"daily_market": _DAILY})
    return Repository(conn, store), snap


def test_scan_quote_rows_happy_path_returns_exact_rows(tmp_path):
    repository, snapshot = _chain_snapshot(tmp_path, [
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0, right="C",
                   bid=1.0, ask=1.2),
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 105.0, right="P",
                   bid=2.0, ask=2.4),
    ])

    result = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                             decision_session=_SESSION)

    assert result == QuoteRowInputs(
        quote_rows=(
            {"ticker": "AAA", "right": "C", "strike": 100.0, "expiry": _EXPIRY,
             "bid": 1.0, "ask": 1.2, "observed_at": _SESSION},
            {"ticker": "AAA", "right": "P", "strike": 105.0, "expiry": _EXPIRY,
             "bid": 2.0, "ask": 2.4, "observed_at": _SESSION},
        ),
        quote_status="recorded",
    )
    assert result.quote_status == "recorded"


def test_scan_quote_rows_with_only_a_different_expiry_is_empty(tmp_path):
    repository, snapshot = _chain_snapshot(tmp_path, [
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 3, 15), 100.0),
    ])

    result = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                             decision_session=_SESSION)

    assert result.quote_rows == ()
    assert result.quote_status == "empty"


def test_scan_quote_rows_with_no_rows_for_the_ticker_is_empty(tmp_path):
    repository, snapshot = _chain_snapshot(tmp_path, [
        _chain_row("BBB", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0),
    ])

    result = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                             decision_session=_SESSION)

    assert result.quote_rows == ()
    assert result.quote_status == "empty"


def test_scan_quote_rows_excludes_other_obs_dates(tmp_path):
    repository, snapshot = _chain_snapshot(tmp_path, [
        _chain_row("AAA", datetime(2024, 1, 4), datetime(2024, 2, 16), 90.0),
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0),
        _chain_row("AAA", datetime(2024, 1, 6), datetime(2024, 2, 16), 110.0),
    ])

    result = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                             decision_session=_SESSION)

    assert result.quote_rows == (
        {"ticker": "AAA", "right": "C", "strike": 100.0, "expiry": _EXPIRY,
         "bid": 1.0, "ask": 1.2, "observed_at": _SESSION},
    )
    assert result.quote_status == "recorded"


def test_scan_quote_rows_isolates_the_requested_ticker(tmp_path):
    repository, snapshot = _chain_snapshot(tmp_path, [
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0),
        _chain_row("BBB", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0),
    ])

    result = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                             decision_session=_SESSION)

    assert len(result.quote_rows) == 1
    assert all(row["ticker"] == "AAA" for row in result.quote_rows)


def test_scan_quote_rows_keeps_null_quotes_as_none(tmp_path):
    repository, snapshot = _chain_snapshot(tmp_path, [
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0, right="C",
                   bid=None, ask=1.2),
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 105.0, right="P",
                   bid=1.0, ask=None),
    ])

    result = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                             decision_session=_SESSION)

    assert result.quote_status == "recorded"
    assert result.quote_rows[0]["bid"] is None
    assert result.quote_rows[0]["ask"] == 1.2
    assert result.quote_rows[1]["bid"] == 1.0
    assert result.quote_rows[1]["ask"] is None


@pytest.mark.parametrize("key", [replace(_KEY, ticker=""), replace(_KEY, session="")])
def test_scan_quote_rows_rejects_a_malformed_key(tmp_path, key):
    repository, snapshot = _chain_snapshot(tmp_path, [
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0),
    ])

    with pytest.raises(OpsError) as exc:
        scan_quote_rows(repository, snapshot, key, expiry=_EXPIRY, decision_session=_SESSION)

    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("field, bad", [
    ("expiry", None),
    ("expiry", "not-a-date"),
    ("expiry", "2024-02-16T01:00:00"),
    ("decision_session", None),
    ("decision_session", pd.Timestamp("2024-01-05", tz="UTC")),
])
def test_scan_quote_rows_rejects_a_malformed_date(tmp_path, field, bad):
    repository, snapshot = _chain_snapshot(tmp_path, [
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0),
    ])

    staged = {"expiry": _EXPIRY, "decision_session": _SESSION, field: bad}
    with pytest.raises(OpsError) as exc:
        scan_quote_rows(repository, snapshot, _KEY, **staged)

    assert exc.value.code == "INVALID_REQUEST"


def test_scan_quote_rows_refuses_a_decision_session_after_expiry(tmp_path):
    repository, snapshot = _chain_snapshot(tmp_path, [
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0),
    ])

    with pytest.raises(DataError) as exc:
        scan_quote_rows(repository, snapshot, _KEY, expiry="2024-02-15",
                        decision_session="2024-02-16")

    assert exc.value.code == "QUERY_NOT_BOUNDED"


def test_scan_quote_rows_without_an_option_chains_table_is_contract_mismatch(tmp_path):
    repository, snapshot = _snapshot_without_chains(tmp_path)

    with pytest.raises(DataError) as exc:
        scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                        decision_session=_SESSION)

    assert exc.value.code == "CONTRACT_MISMATCH"


def _capture_scan(monkeypatch, repository):
    captured = []
    original_scan = repository.scan

    def _scan(query, **kwargs):
        captured.append(query)
        return original_scan(query, **kwargs)

    monkeypatch.setattr(repository, "scan", _scan)
    return captured


def _capture_population_scan(monkeypatch, repository, population_bound):
    monkeypatch.setattr(
        repository, "scan_population_bound", lambda *_a, **_k: population_bound)
    return _capture_scan(monkeypatch, repository)


def _quote_rows():
    return [
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0,
                   right="C", bid=1.0, ask=1.2),
        _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 105.0,
                   right="P", bid=2.0, ask=2.4),
    ]


def test_scan_quote_rows_keeps_limit_when_population_bound_is_at_or_above_current(
        tmp_path, monkeypatch):
    rows = _quote_rows()
    repository, snapshot = _chain_snapshot(tmp_path, rows)
    expected = len(rows)
    monkeypatch.setitem(scan_quote_rows.__globals__, "_RESULT_CAP", expected)
    baseline = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                               decision_session=_SESSION)
    captured = _capture_scan(monkeypatch, repository)

    result = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                             decision_session=_SESSION)

    assert result == baseline
    assert result.quote_status == "recorded"
    assert result.quote_rows == (
        {"ticker": "AAA", "right": "C", "strike": 100.0, "expiry": _EXPIRY,
         "bid": 1.0, "ask": 1.2, "observed_at": _SESSION},
        {"ticker": "AAA", "right": "P", "strike": 105.0, "expiry": _EXPIRY,
         "bid": 2.0, "ask": 2.4, "observed_at": _SESSION},
    )
    assert captured[0].max_result_rows == expected
    assert captured[0].max_batch_rows == min(
        _CHAINS.maximum_batch_rows, _BATCH_CAP, expected)


def test_scan_quote_rows_uses_smaller_selected_population_bound(tmp_path, monkeypatch):
    repository, snapshot = _chain_snapshot(tmp_path, _quote_rows())
    baseline = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                               decision_session=_SESSION)
    version = snapshot.table_versions["option_chains"]
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=(_KEY.ticker,)),
                  KeyPredicate(column="obs_date", operator="eq", values=(_SESSION,)))
    bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name="option_chains",
        table_contract_ref=version.table_contract_ref,
        key_filter=key_filter, time_interval=None)
    captured = _capture_scan(monkeypatch, repository)

    result = scan_quote_rows(repository, snapshot, _KEY, expiry=_EXPIRY,
                             decision_session=_SESSION)

    existing_limit = _RESULT_CAP
    assert result == baseline
    assert result.quote_status == "recorded"
    assert result.quote_rows == (
        {"ticker": "AAA", "right": "C", "strike": 100.0, "expiry": _EXPIRY,
         "bid": 1.0, "ask": 1.2, "observed_at": _SESSION},
        {"ticker": "AAA", "right": "P", "strike": 105.0, "expiry": _EXPIRY,
         "bid": 2.0, "ask": 2.4, "observed_at": _SESSION},
    )
    assert 0 < bound < existing_limit
    assert captured[0].max_result_rows == min(existing_limit, bound)
    assert captured[0].max_batch_rows == min(
        _CHAINS.maximum_batch_rows, _BATCH_CAP, captured[0].max_result_rows)
