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
from engine.v2.ops.nightly_chain_batch import require_chain_rows, scan_chain_slices
from engine.v2.ops.nightly_quote_rows import scan_quote_rows
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
    assert counters["rows_returned"] == 14
    assert counters["rows_decoded"] == 4 * len(_rows())
    batched_repository, batched_snapshot = _snap(tmp_path / "batched")
    scan_chain_slices(batched_repository, batched_snapshot, ["AAA", "BBB"],
                      decision_session=_SESSION)
    batched = batched_repository.read_counters.snapshot()["option_chains"]
    assert batched["scans"] == 1
    assert batched["fragments_opened"] == 1
    assert counters["bytes_read"] == 4 * batched["bytes_read"] > 0


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


@pytest.mark.parametrize("session", ["2024-13-45", "not-a-date", None, 20240105,
                                     "2024-01-05 10:00"])
def test_non_iso_decision_session_is_invalid_request(tmp_path, session):
    repository, snapshot = _snap(tmp_path)
    with pytest.raises(OpsError) as exc:
        scan_chain_slices(repository, snapshot, ["AAA"], decision_session=session)
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("tickers", [None, 5, [None], [""], ["AAA", 7], [["AAA"]], "AAA"])
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
    with pytest.raises(OpsError) as session_exc:
        scan_candidate_expiries(repository, snapshot, _AAA, decision_session="2024-01-04",
                                chain_rows=slices["AAA"])
    assert session_exc.value.code == "INVALID_REQUEST"
    with pytest.raises(OpsError) as quote_session_exc:
        scan_quote_rows(repository, snapshot, _AAA, expiry=_EXPIRY,
                        decision_session="2024-01-04", chain_rows=slices["AAA"])
    assert quote_session_exc.value.code == "INVALID_REQUEST"
    assert require_chain_rows(iter(slices["AAA"]), "AAA", _SESSION) == slices["AAA"]


def test_readers_given_chain_rows_never_scan(tmp_path, monkeypatch):
    repository, snapshot = _snap(tmp_path)
    slices = scan_chain_slices(repository, snapshot, ["AAA"], decision_session=_SESSION)
    scans = _count_scans(monkeypatch)
    scan_candidate_expiries(repository, snapshot, _AAA, decision_session=_SESSION,
                            chain_rows=slices["AAA"])
    scan_quote_rows(repository, snapshot, _AAA, expiry=_EXPIRY, decision_session=_SESSION,
                    chain_rows=slices["AAA"])
    assert scans == []


def test_short_chain_rows_are_invalid_request(tmp_path):
    repository, snapshot = _snap(tmp_path)
    short = [{"ticker": "AAA", "obs_date": datetime(2024, 1, 5)}]
    with pytest.raises(OpsError) as exc:
        scan_quote_rows(repository, snapshot, _AAA, expiry=_EXPIRY,
                        decision_session=_SESSION, chain_rows=short)
    assert exc.value.code == "INVALID_REQUEST"
    with pytest.raises(OpsError) as non_mapping:
        scan_candidate_expiries(repository, snapshot, _AAA, decision_session=_SESSION,
                                chain_rows=[("AAA",)])
    assert non_mapping.value.code == "INVALID_REQUEST"


def test_result_ceiling_fails_the_build_without_fallback(tmp_path, monkeypatch):
    import engine.v2.ops.nightly_chain_batch as batch

    repository, snapshot = _snap(tmp_path)
    monkeypatch.setattr(batch, "_RESULT_CAP", 1)
    scans = _count_scans(monkeypatch)
    with pytest.raises(DataError) as exc:
        scan_chain_slices(repository, snapshot, ["AAA", "BBB"], decision_session=_SESSION)
    assert exc.value.code == "RESULT_LIMIT_EXCEEDED"
    assert scans == ["option_chains"]


def test_seconds_use_the_counter_clock(tmp_path, monkeypatch):
    import itertools

    from engine.v2.data import read_counters

    repository, snapshot = _snap(tmp_path)
    ticks = itertools.count()
    monkeypatch.setattr(read_counters, "clock", lambda: next(ticks) * 0.5)
    scan_chain_slices(repository, snapshot, ["AAA"], decision_session=_SESSION)
    assert repository.read_counters.snapshot()["option_chains"]["seconds"] == 1.0


@pytest.mark.parametrize("obs_date", ["2024-01-05 10:00", "2024-01-05 garbage"])
def test_obs_date_with_trailing_text_is_refused(tmp_path, obs_date):
    rows = [dict(_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0),
                 obs_date=obs_date)]
    repository, snapshot = _snap(tmp_path)
    with pytest.raises(OpsError) as exc:
        scan_quote_rows(repository, snapshot, _AAA, expiry=_EXPIRY,
                        decision_session=_SESSION, chain_rows=rows)
    assert exc.value.code == "INVALID_REQUEST"


def test_one_ticker_with_two_keys_shares_its_slice(tmp_path):
    repository, snapshot = _snap(tmp_path)
    second = BoardRequest("AAA", "BFLY-P", pd.Timestamp("2024-01-16"), "BMO")
    slices = scan_chain_slices(repository, snapshot, ["AAA"], decision_session=_SESSION)
    shared = slices["AAA"]
    size = len(shared)
    for key in (_AAA, second):
        assert scan_candidate_expiries(
            repository, snapshot, key, decision_session=_SESSION, chain_rows=shared
        ) == scan_candidate_expiries(repository, snapshot, key, decision_session=_SESSION)
        assert scan_quote_rows(
            repository, snapshot, key, expiry=_EXPIRY, decision_session=_SESSION,
            chain_rows=shared
        ) == scan_quote_rows(repository, snapshot, key, expiry=_EXPIRY,
                             decision_session=_SESSION)
    assert len(shared) == size == 5


@pytest.mark.parametrize("rows", [None, 5, "AAA"])
def test_non_collection_chain_rows_are_invalid_request(tmp_path, rows):
    repository, snapshot = _snap(tmp_path)
    with pytest.raises(OpsError) as exc:
        scan_quote_rows(repository, snapshot, _AAA, expiry=_EXPIRY,
                        decision_session=_SESSION, chain_rows=rows) if rows is not None else (
            require_chain_rows(rows, "AAA", _SESSION))
    assert exc.value.code == "INVALID_REQUEST"


def test_calendar_reader_forwards_chain_rows_and_never_scans(tmp_path, monkeypatch):
    from engine.v2.foundation.market_calendar import CalendarSessions
    from engine.v2.ops import nightly_calendar_inputs as nci
    from engine.v2.ops.nightly_raw_rows import CalendarRowInputs

    repository, snapshot = _snap(tmp_path)

    def fake_row(repo, snap, key, *, entry_date, exit_date, expiry, spot,
                 calendar_observed_through):
        return CalendarRowInputs(calendar_revision="rev",
                                 calendar_row={"expiry": expiry, "spot": spot})

    monkeypatch.setattr(nci, "scan_calendar_row", fake_row)
    calendar = CalendarSessions(
        days=("2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09",
              "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17",
              "2024-01-18", "2024-01-19"),
        observed_through=_SESSION)
    slices = scan_chain_slices(repository, snapshot, ["AAA"], decision_session=_SESSION)
    per_key = nci.scan_calendar_row_inputs(repository, snapshot, _AAA,
                                           decision_session=_SESSION, calendar=calendar)
    scans: list[str] = []
    real = Repository.scan

    def counting(self, query, *, table_name):
        scans.append(table_name)
        return real(self, query, table_name=table_name)

    monkeypatch.setattr(Repository, "scan", counting)
    batched = nci.scan_calendar_row_inputs(repository, snapshot, _AAA,
                                           decision_session=_SESSION, calendar=calendar,
                                           chain_rows=iter(slices["AAA"]))
    assert batched == per_key
    assert "option_chains" not in scans


def test_chain_rows_are_checked_before_other_refusals(tmp_path):
    from engine.v2.foundation.market_calendar import CalendarSessions
    from engine.v2.ops import nightly_calendar_inputs as nci

    repository, snapshot = _snap(tmp_path)
    slices = scan_chain_slices(repository, snapshot, ["AAA"], decision_session=_SESSION)
    with pytest.raises(OpsError) as after_expiry:
        scan_quote_rows(repository, snapshot, _AAA, expiry="2024-01-04",
                        decision_session=_SESSION, chain_rows=[{"ticker": "BBB"}])
    assert after_expiry.value.code == "INVALID_REQUEST"
    calendar = CalendarSessions(days=("2024-01-05", "2024-01-08"), observed_through="2024-01-08")
    with pytest.raises(OpsError) as no_spot:
        nci.scan_calendar_row_inputs(repository, snapshot, _AAA, decision_session="2024-01-08",
                                     calendar=calendar, chain_rows=slices["AAA"])
    assert no_spot.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("change", [
    {"strike": None}, {"strike": "abc"}, {"right": "X"}, {"right": None},
    {"expiry": None}, {"bid": "abc"}, {"ask": "abc"}])
def test_unusable_chain_row_fields_are_contract_mismatch(tmp_path, change):
    row = dict(_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0), **change)
    repository, snapshot = _snap(tmp_path)
    with pytest.raises(DataError) as exc:
        scan_quote_rows(repository, snapshot, _AAA, expiry=_EXPIRY,
                        decision_session=_SESSION, chain_rows=[row])
    assert exc.value.code == "CONTRACT_MISMATCH"
