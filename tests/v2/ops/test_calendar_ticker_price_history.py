"""SPY, the trading-calendar series the native nightly reads, is a declared
price_history dependency: price-refresh always fetches it daily, ``ops
price-history capture`` refuses ``SOURCE_NOT_FOUND`` before any write when a
required ticker is absent from both the sources and the prior dataset version,
and the supervisor's native-score-batch problem report keeps typed errors'
error_type/code/details instead of swallowing them. Synthetic px trees only --
no network, no real market data.
"""
from __future__ import annotations

import json
import os
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_calendar_inputs import scan_decision_calendar
from engine.v2.ops.price_history_store import capture
from engine.v2.ops.supervisor import Service
from tests.test_v2_ops_price_history import (
    _base_snapshot,
    _epoch,
    _px_dir,
    _scan_price_history,
    _write_px,
)

_SPY_DATES = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
              "2024-01-08", "2024-01-09")
_DECISION_SESSION = "2024-01-09"
_EVENT_THROUGH = "2024-01-17"


def _write_sources(source_root, *, with_spy: bool) -> None:
    """An AAPL-only px tree (SPY absent, the legacy price-refresh universe's
    own shape), optionally with SPY rows on consecutive weekday dates."""
    _write_px(source_root, "AAPL", {"2024-01-02": 185.0, "2024-01-03": 186.0})
    if with_spy:
        _write_px(source_root, "SPY", {d: 470.0 + i for i, d in enumerate(_SPY_DATES)})


def test_capture_without_required_ticker_refuses_before_any_write(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=("SPY",))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": ["SPY"]}
    with pytest.raises(OpsError) as dry:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                dry_run=True, required_tickers=("SPY",))
    assert dry.value.problem.code == "SOURCE_NOT_FOUND"
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                        ).fetchone()
    assert head["snapshot_id"] == base.snapshot_id  # nothing was committed


def test_capture_required_tickers_may_be_a_one_shot_iterator(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=iter(("SPY",)))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": ["SPY"]}
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                        ).fetchone()
    assert head["snapshot_id"] == base.snapshot_id  # nothing was committed


def test_capture_with_unparseable_required_ticker_source_refuses_and_commits_nothing(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    (_px_dir(source_root) / "px_SPY.csv").write_bytes(b"not,a,price\nfile\n")
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=("SPY",))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": ["SPY"]}
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                        ).fetchone()
    assert head["snapshot_id"] == base.snapshot_id  # nothing was committed


def test_capture_with_required_ticker_in_a_shadow_scope_without_spy_universe_stages_spy(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                     required_tickers=("SPY",))
    rows = _scan_price_history(conn, store, report["result_snapshot_id"], ticker="SPY")
    assert not rows.empty
    assert sorted(rows["date"]) == list(_SPY_DATES)


def test_native_calendar_scan_succeeds_on_the_captured_snapshot(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                     required_tickers=("SPY",))
    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])
    calendar = scan_decision_calendar(repository, snapshot,
                                      decision_session=_DECISION_SESSION,
                                      event_through=_EVENT_THROUGH)
    assert calendar.observed_through == _DECISION_SESSION


def test_native_calendar_scan_on_a_snapshot_without_spy_is_contract_mismatch(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])
    with pytest.raises(DataError) as exc:
        scan_decision_calendar(repository, snapshot, decision_session=_DECISION_SESSION,
                               event_through=_EVENT_THROUGH)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"ticker": "SPY"}


def test_required_ticker_in_prior_dataset_version_does_not_refuse(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
            required_tickers=("SPY",))
    second_root = tmp_path / "legacy_no_spy"
    _write_sources(second_root, with_spy=False)
    report = capture(conn, store, second_root, root=tmp_path, scope="shadow", clock=clock,
                     required_tickers=("SPY",))
    assert "result_snapshot_id" in report


def test_price_refresh_plan_always_plans_forced_ticker_daily():
    from engine.data.pulls import price_refresh as pr

    events = pd.DataFrame({"ticker": [], "event_date": []})
    plan = pr.plan_refresh("2026-09-14", events=events, price_universe=["ZZZ"],
                           always_daily=("SPY",))
    assert plan["daily"] == ["SPY"]
    assert plan["monthly"] == ["ZZZ"]
    fetched = pr.plan_refresh("2026-09-14", events=events, price_universe=["ZZZ"],
                              fetch_history={"SPY": [date(2026, 9, 14)]},
                              always_daily=("SPY",))
    assert "SPY" in fetched["skipped_already_fetched"]
    assert "SPY" not in fetched["daily"]
    without = pr.plan_refresh("2026-09-14", events=events, price_universe=["ZZZ"])
    assert without["daily"] == []
    assert without["monthly"] == ["ZZZ"]
    assert "SPY" not in (without["daily"] + without["monthly"]
                         + without["skipped_already_fetched"])


def test_supervisor_reports_typed_error_type_code_and_details(capsys):
    holder = SimpleNamespace(_last_native_score_batch_problem=None)
    boom = data_fail("CONTRACT_MISMATCH", "no price_history for this ticker",
                     details={"ticker": "SPY"})
    Service._report_native_score_batch_problem(holder, boom)
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "native_score_batch_reconcile_failed"
    assert event["error_type"] == "DataError"
    assert event["problem"]["code"] == "CONTRACT_MISMATCH"
    assert event["problem"]["details"] == {"ticker": "SPY"}
    Service._report_native_score_batch_problem(holder, boom)
    assert capsys.readouterr().out == ""  # deduped
    Service._report_native_score_batch_problem(holder, RuntimeError("x"))
    plain = json.loads(capsys.readouterr().out)
    assert plain["error_type"] == "RuntimeError"
    assert plain["problem"]["code"] == "VALIDATION_FAILED"


def test_supervisor_report_drops_unbounded_and_nonscalar_details(capsys):
    holder = SimpleNamespace(_last_native_score_batch_problem=None)
    boom = data_fail("CONTRACT_MISMATCH", "m",
                     details={"ticker": "SPY", "blob": {"a": 1}, "long": "x" * 500,
                              "tickers": ["SPY"]})
    Service._report_native_score_batch_problem(holder, boom)
    event = json.loads(capsys.readouterr().out)
    assert event["problem"]["details"] == {"ticker": "SPY", "tickers": ["SPY"]}


def test_supervisor_report_drops_long_keys_and_non_finite_floats(capsys):
    holder = SimpleNamespace(_last_native_score_batch_problem=None)
    boom = data_fail("CONTRACT_MISMATCH", "m",
                     details={"k" * 500: "v", "bad": float("nan"), "inf": float("inf"),
                              "ok": 1.5})
    Service._report_native_score_batch_problem(holder, boom)
    line = capsys.readouterr().out
    event = json.loads(line, parse_constant=lambda value: pytest.fail(
        f"non-finite constant {value!r} in {line!r}"))
    assert event["problem"]["details"] == {"ok": 1.5}
    assert "NaN" not in line
    assert "Infinity" not in line


def test_capture_with_all_dates_tombstoned_required_ticker_refuses(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    spy_path = _px_dir(source_root) / "px_SPY.csv"
    os.utime(spy_path, (_epoch("2024-01-01T00:00:00+00:00"),) * 2)
    capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
            required_tickers=("SPY",))

    # A header-only px file parses as a valid, empty retrieval; a later mtime
    # makes it a legitimate later retrieval whose diff tombstones every prior
    # live SPY row.
    second_root = tmp_path / "legacy_tombstoned"
    tomb_path = _px_dir(second_root) / "px_SPY.csv"
    tomb_path.write_bytes(b"date,close_adj,close_raw,high_raw\n")
    os.utime(tomb_path, (_epoch("2024-06-01T00:00:00+00:00"),) * 2)
    with pytest.raises(OpsError) as exc:
        capture(conn, store, second_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=("SPY",))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": ["SPY"]}


class _NonProblemError(Exception):
    def __init__(self):
        super().__init__("x")
        self.problem = SimpleNamespace(code="X", message="y")


def test_usable_tickers_skips_live_row_reconstruction_without_required_tickers(monkeypatch):
    from engine.v2.ops import price_history_store as phs

    def _boom(rows):
        raise AssertionError("should not run")

    monkeypatch.setattr(phs, "_has_live_rows", _boom)
    assert phs._usable_tickers(frozenset(), ["AAPL"], {}, None,
                               {"AAPL": pd.DataFrame()}) == set()


def test_supervisor_report_ignores_a_non_problem_problem_attribute(capsys):
    holder = SimpleNamespace(_last_native_score_batch_problem=None)
    Service._report_native_score_batch_problem(holder, _NonProblemError())
    event = json.loads(capsys.readouterr().out)
    assert event["problem"]["code"] == "VALIDATION_FAILED"
    assert event["problem"]["message"] == "native_score_batch shadow reconciliation failed"
