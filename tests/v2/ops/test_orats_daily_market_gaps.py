"""daily_market partial-coverage behaviour: a missing expected ticker is a typed gap, never `complete` (see issue 142)"""
from __future__ import annotations

import dataclasses
import json

import pytest

from engine.v2.contracts import CompletedCoverage
from engine.v2.data import incremental as data_incremental
from engine.v2.data.errors import DataError
from engine.v2.foundation import canonical_json, from_document
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


def _stage_refresh(root, document, *, generation, head):
    root.mkdir(parents=True, exist_ok=True)
    staged = dict(document, scope="shadow", expected_head_generation=generation,
                  expected_head_snapshot_id=head)
    (root / "incremental_refresh_input.json").write_text(canonical_json(staged))
    (root / "refresh_plan.json").write_text(canonical_json({"fetch_units": [UNIT]}))


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


def test_partial_response_commits_returned_rows_and_the_gap_through_the_refresh(tmp_path):
    conn, _clock, _store, document, parameters = _planned_keys_acquisition(tmp_path)
    _stage_refresh(tmp_path, document,
                   generation=parameters.expected_head_generation,
                   head=parameters.parent_snapshot_id)
    calls = []
    result = data_incremental.run_daily_market_refresh(
        parameters, tmp_path, fetcher=_fetcher(["AAA", "BBB"], calls))
    assert result["status"] == "complete"
    assert result["coverage_advanced"] is True
    assert len(calls) == 4
    revisions = conn.execute(
        "SELECT ticker FROM data_daily_market_revisions").fetchall()
    assert sorted(row[0] for row in revisions) == ["AAA", "BBB"]
    receipts = conn.execute(
        "SELECT raw_receipt_id, response_kind FROM data_raw_receipts").fetchall()
    assert len(receipts) == 1 and receipts[0][1] == "partial"
    # committed coverage is read back from the persisted data_snapshot_coverage row
    coverage_json = conn.execute(
        "SELECT coverage_json FROM data_snapshot_coverage WHERE snapshot_id = ? "
        "AND table_name = ?", (result["candidate_snapshot_id"], "daily_market")).fetchone()[0]
    coverage = from_document(CompletedCoverage, json.loads(coverage_json))
    assert coverage.state == "partial"
    missing = [outcome for outcome in coverage.outcomes if outcome.status == "missing"]
    assert len(missing) == 1 and missing[0].key.ticker == "CCC"
    assert missing[0].receipt_id == receipts[0][0]


def test_rerun_from_committed_partial_state_still_succeeds_for_returned_tickers(tmp_path):
    conn, _clock, _store, document, parameters = _planned_keys_acquisition(tmp_path)
    _stage_refresh(tmp_path, document,
                   generation=parameters.expected_head_generation,
                   head=parameters.parent_snapshot_id)
    first = data_incremental.run_daily_market_refresh(
        parameters, tmp_path, fetcher=_fetcher(["AAA", "BBB"], []))
    assert first["status"] == "complete"
    head = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        ("shadow",)).fetchone()
    rerun = dataclasses.replace(
        parameters, parent_snapshot_id=first["candidate_snapshot_id"],
        expected_head_snapshot_id=first["candidate_snapshot_id"],
        expected_head_generation=head[1], refresh_plan_hash="sha256:" + "7" * 64)
    second_root = tmp_path / "second"
    _stage_refresh(second_root, document, generation=head[1],
                   head=first["candidate_snapshot_id"])
    calls = []
    result = data_incremental.run_daily_market_refresh(
        rerun, second_root, fetcher=_fetcher(["AAA", "BBB"], calls))
    # identical rerun of already-committed rows can be a snapshot-level noop;
    # run_incremental_refresh still reports that commit as "complete".
    assert result["status"] in ("complete", "noop")
    assert len(calls) == 4
    head_after = conn.execute(
        "SELECT snapshot_id FROM data_snapshot_heads WHERE scope = ?",
        ("shadow",)).fetchone()[0]
    coverage_json = conn.execute(
        "SELECT coverage_json FROM data_snapshot_coverage WHERE snapshot_id = ? "
        "AND table_name = ?", (head_after, "daily_market")).fetchone()[0]
    coverage = from_document(CompletedCoverage, json.loads(coverage_json))
    present = {outcome.key.ticker for outcome in coverage.outcomes
               if outcome.status == "present"}
    assert {"AAA", "BBB"} <= present
    assert coverage.state == "partial"
    missing = {outcome.key.ticker for outcome in coverage.outcomes
               if outcome.status == "missing"}
    assert missing == {"CCC"}
    assert conn.execute(
        "SELECT COUNT(*) FROM data_daily_market_revisions WHERE ticker = 'CCC'"
    ).fetchone()[0] == 0
