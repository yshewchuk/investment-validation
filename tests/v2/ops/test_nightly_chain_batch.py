"""Batched ``option_chains`` read (#754): equivalence with the per-key path,
one scan per build, failure semantics and the read counters' amplification."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.nightly_calendar_inputs import scan_candidate_expiries
from engine.v2.ops.nightly_chain_batch import scan_chain_slices
from engine.v2.ops.nightly_quote_rows import scan_quote_rows
from engine.v2.ops.nightly_chain_batch import require_chain_rows
from tests.v2.ops.chain_support import (
    CHAINS_NAME as _CHAINS_NAME,
    SESSION as _SESSION,
    STRADDLE_KEY as _STRADDLE_KEY,
    chain_row as _chain_row,
    domain_chain_rows as _domain_chain_rows,
    price_row as _price_row,
    snapshot as _snapshot,
)

_AAA = _STRADDLE_KEY
_BBB = BoardRequest("BBB", "STR-THRU", pd.Timestamp("2024-01-16"), "BMO")
_CCC = BoardRequest("CCC", "STR-THRU", pd.Timestamp("2024-01-16"), "BMO")
_EXPIRY = "2024-01-19"


def _rows() -> list[dict]:
    filler = [_chain_row(f"Z{i:02d}", datetime(2024, 1, 4), datetime(2024, 1, 19), 100.0,
                         right="C") for i in range(40)]
    return _domain_chain_rows() + filler


def _snap(tmp_path):
    return _snapshot(tmp_path, price={"AAA": [_price_row("AAA", _SESSION)]}, chains=_rows())


def _count_scans(monkeypatch) -> list[str]:
    scanned: list[str] = []
    real = Repository.scan

    def counting(self, query, *, table_name):
        scanned.append(table_name)
        return real(self, query, table_name=table_name)

    monkeypatch.setattr(Repository, "scan", counting)
    return scanned


def test_scan_chain_slices_is_one_scan_grouped_by_ticker(tmp_path, monkeypatch):
    repository, snapshot = _snap(tmp_path)
    scans = _count_scans(monkeypatch)
    slices = scan_chain_slices(repository, snapshot, ["AAA", "BBB", "CCC", "AAA"],
                               decision_session=_SESSION)
    assert scans == ["option_chains"]
    assert sorted(slices) == ["AAA", "BBB", "CCC"]
    assert len(slices["AAA"]) == 5
    assert len(slices["BBB"]) == 2
    assert slices["CCC"] == []
    assert all(pd.Timestamp(row["obs_date"]) == pd.Timestamp(_SESSION)
               for rows in slices.values() for row in rows)


def test_batched_output_identical_to_per_key_path(tmp_path):
    repository, snapshot = _snap(tmp_path)
    slices = scan_chain_slices(repository, snapshot, ["AAA", "BBB", "CCC"],
                               decision_session=_SESSION)
    for key in (_AAA, _BBB, _CCC):
        per_key = scan_candidate_expiries(repository, snapshot, key, decision_session=_SESSION)
        batched = scan_candidate_expiries(repository, snapshot, key, decision_session=_SESSION,
                                          chain_rows=slices[key.ticker])
        assert per_key == batched
        per_key_quotes = scan_quote_rows(repository, snapshot, key, expiry=_EXPIRY,
                                         decision_session=_SESSION)
        batched_quotes = scan_quote_rows(repository, snapshot, key, expiry=_EXPIRY,
                                         decision_session=_SESSION,
                                         chain_rows=slices[key.ticker])
        assert per_key_quotes == batched_quotes
    assert scan_candidate_expiries(repository, snapshot, _AAA, decision_session=_SESSION)
    assert scan_quote_rows(repository, snapshot, _AAA, expiry=_EXPIRY,
                           decision_session=_SESSION).quote_rows
    assert scan_candidate_expiries(repository, snapshot, _CCC, decision_session=_SESSION) == ()
    assert scan_quote_rows(repository, snapshot, _CCC, expiry=_EXPIRY,
                           decision_session=_SESSION).quote_status == "empty"


def test_per_key_path_decodes_far_more_rows_than_it_returns_and_batch_reads_once(tmp_path):
    (tmp_path / "per_key").mkdir()
    (tmp_path / "batched").mkdir()
    repository, snapshot = _snap(tmp_path / "per_key")
    for key in (_AAA, _BBB):
        scan_candidate_expiries(repository, snapshot, key, decision_session=_SESSION)
        scan_quote_rows(repository, snapshot, key, expiry=_EXPIRY, decision_session=_SESSION)
    counters = repository.read_counters.snapshot()["option_chains"]
    assert counters["scans"] == 4
    assert counters["fragments_opened"] == 4
    assert counters["rows_returned"] > 0
    assert counters["rows_decoded"] >= 5 * counters["rows_returned"]
    assert counters["bytes_read"] > 0
    batched_repository, batched_snapshot = _snap(tmp_path / "batched")
    scan_chain_slices(batched_repository, batched_snapshot, ["AAA", "BBB"],
                      decision_session=_SESSION)
    batched = batched_repository.read_counters.snapshot()["option_chains"]
    assert batched["scans"] == 1
    assert batched["fragments_opened"] == 1


def test_missing_option_table_is_contract_mismatch(tmp_path):
    repository, snapshot = _snapshot(tmp_path, price={"AAA": [_price_row("AAA", _SESSION)]})
    with pytest.raises(DataError) as exc:
        scan_chain_slices(repository, snapshot, ["AAA"], decision_session=_SESSION)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"table_name": _CHAINS_NAME}


@pytest.mark.parametrize("code", ["RESULT_LIMIT_EXCEEDED", "QUERY_NOT_BOUNDED"])
def test_repository_errors_propagate_without_per_key_fallback(tmp_path, monkeypatch, code):
    repository, snapshot = _snap(tmp_path)
    problem = data_fail(code, "synthetic option scan failure")
    calls: list[str] = []

    def broken(query, *, table_name):
        calls.append(table_name)
        raise problem

    monkeypatch.setattr(repository, "scan", broken)
    with pytest.raises(DataError) as exc:
        scan_chain_slices(repository, snapshot, ["AAA", "BBB"], decision_session=_SESSION)
    assert exc.value is problem
    assert calls == ["option_chains"]


def test_empty_ticker_set_does_no_scan(tmp_path, monkeypatch):
    repository, snapshot = _snap(tmp_path)
    scans = _count_scans(monkeypatch)
    assert scan_chain_slices(repository, snapshot, [], decision_session=_SESSION) == {}
    assert scans == []


@pytest.mark.parametrize("session", ["2024-13-45", "not-a-date", None, 20240105])
def test_non_iso_decision_session_is_invalid_request(tmp_path, session):
    repository, snapshot = _snap(tmp_path)
    with pytest.raises(OpsError) as exc:
        scan_chain_slices(repository, snapshot, ["AAA"], decision_session=session)
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("tickers", [[None], [""], ["AAA", 7]])
def test_bad_tickers_are_invalid_request(tmp_path, tickers):
    repository, snapshot = _snap(tmp_path)
    with pytest.raises(OpsError) as exc:
        scan_chain_slices(repository, snapshot, tickers, decision_session=_SESSION)
    assert exc.value.code == "INVALID_REQUEST"


def test_chain_rows_for_another_ticker_or_session_are_refused(tmp_path):
    repository, snapshot = _snap(tmp_path)
    slices = scan_chain_slices(repository, snapshot, ["AAA", "BBB"], decision_session=_SESSION)
    with pytest.raises(OpsError) as exc:
        scan_quote_rows(repository, snapshot, _AAA, expiry=_EXPIRY,
                        decision_session=_SESSION, chain_rows=slices["BBB"])
    assert exc.value.code == "INVALID_REQUEST"
    with pytest.raises(OpsError):
        scan_candidate_expiries(repository, snapshot, _AAA, decision_session="2024-01-04",
                                chain_rows=slices["AAA"])
    assert require_chain_rows(iter(slices["AAA"]), "AAA", _SESSION) == slices["AAA"]