"""daily_market partial-coverage behaviour: a missing expected ticker is a typed gap, never `complete` (see issue 142)"""
from __future__ import annotations

import json

import pytest

from engine.v2.data import incremental as data_incremental
from engine.v2.data.errors import DataError
from engine.v2.ops.errors import OpsError
from engine.v2.ops.providers.orats_daily_market import orats_daily_market_fetcher
from tests.test_v2_data_incremental_persistence import _planned_keys_acquisition
from tests.test_v2_data_manifests import _DAILY_MARKET_CONTRACT

SESSION = "2026-05-01"
UNIT = {"request_id": "u1", "table_name": "daily_market",
        "partition_key": SESSION, "expected_keys": ["AAA", "BBB", "CCC"]}


def _summary(ticker):
    return {"ticker": ticker, "tradeDate": SESSION, "stockPrice": 100.0,
            "iv10d": 0.30, "iv30d": 0.32, "exErnIv10d": 0.29, "exErnIv30d": 0.31,
            "impliedMove": 0.05, "rVol30": 0.28, "skewing": 1.1, "contango": 0.5,
            "fwd90_30": 0.33, "fexErn90_30": 0.34, "ieeEarnEffect": 0.2}


def _core(ticker):
    return {"ticker": ticker, "tradeDate": SESSION, "mktCap": 1.0}


def _http_get(summaries, cores, calls):
    def get(url, *, timeout):
        calls.append(url)
        rows = summaries if "hist/summaries" in url else cores
        return 200, {}, json.dumps({"data": rows}).encode()
    return get


def _fetcher(tickers, calls):
    return orats_daily_market_fetcher(
        http_get=_http_get([_summary(ticker) for ticker in tickers],
                           [_core(ticker) for ticker in tickers], calls), api_key="k")


def test_one_missing_ticker_returns_partial_with_the_other_rows():
    calls = []
    raw, kind, meta, rows = _fetcher(["AAA", "BBB"], calls)(UNIT)
    assert kind == "partial"
    assert sorted(row["ticker"] for row in rows) == ["AAA", "BBB"]
    assert "CCC" not in raw.decode()
    assert meta["attempts"] == 2


def test_missing_ticker_retry_is_bounded():
    calls = []
    _fetcher(["AAA", "BBB"], calls)(UNIT)
    assert len(calls) == 4
    second = []
    _fetcher(["AAA", "BBB"], second)(UNIT)
    assert len(second) == 4


def test_complete_when_every_expected_ticker_is_returned():
    calls = []
    kind = _fetcher(["AAA", "BBB", "CCC"], calls)(UNIT)[1]
    assert kind == "complete"
    assert len(calls) == 2


def test_gap_is_committed_as_missing_outcome_and_receipt_is_never_complete(tmp_path):
    conn, _clock, store, _document, _parameters = _planned_keys_acquisition(tmp_path)
    calls = []
    fetched = data_incremental._fetch_unit(
        conn, store, _DAILY_MARKET_CONTRACT, UNIT, _fetcher(["AAA", "BBB"], calls))
    present = {outcome.key.ticker: outcome
               for outcome in fetched.outcomes if outcome.status == "present"}
    missing = [outcome for outcome in fetched.outcomes if outcome.status == "missing"]
    assert set(present) == {"AAA", "BBB"}
    assert all(outcome.revision_id is not None for outcome in present.values())
    assert len(missing) == 1 and missing[0].key.ticker == "CCC"
    assert missing[0].revision_id is None
    assert missing[0].receipt_id == fetched.receipt_id
    kinds = [row[0] for row in conn.execute(
        "SELECT response_kind FROM data_raw_receipts")]
    assert kinds == ["partial"]
    coverage = data_incremental._fetched_coverage(_DAILY_MARKET_CONTRACT, (fetched,))
    assert coverage.state == "partial"
    assert {"AAA", "BBB"} <= set(coverage.covered_tickers)
    assert "CCC" not in coverage.covered_tickers


def test_response_missing_every_expected_ticker_still_refuses(tmp_path):
    conn, _clock, store, _document, _parameters = _planned_keys_acquisition(tmp_path)
    calls = []
    with pytest.raises(DataError) as err:
        data_incremental._fetch_unit(
            conn, store, _DAILY_MARKET_CONTRACT, UNIT, _fetcher(["ZZZ"], calls))
    assert err.value.problem.code == "TRANSIENT_SOURCE"
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 0


def test_empty_2xx_response_refuses_not_final():
    calls = []
    with pytest.raises(OpsError) as err:
        _fetcher([], calls)(UNIT)
    assert err.value.problem.code == "SOURCE_NOT_FINAL"


def test_one_missing_ticker_does_not_block_the_others_across_reruns(tmp_path):
    conn, _clock, store, _document, _parameters = _planned_keys_acquisition(tmp_path)
    for _ in range(2):
        calls = []
        fetched = data_incremental._fetch_unit(
            conn, store, _DAILY_MARKET_CONTRACT, UNIT, _fetcher(["AAA", "BBB"], calls))
        assert {outcome.key.ticker for outcome in fetched.outcomes
                if outcome.status == "present"} == {"AAA", "BBB"}
        assert {outcome.key.ticker for outcome in fetched.outcomes
                if outcome.status == "missing"} == {"CCC"}
        assert len(calls) == 4
