"""Same-cohort comparison over committed synthetic Parquet, no outcome stubs."""
import json
import os
import sqlite3
import subprocess
import sys

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.ops.errors import OpsError
from experiments import runup_exit_report as comparison
from experiments import v2_runup_exits as cli
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

DATES = pd.bdate_range("2024-04-12", "2024-05-03")
EVENT_ID = "TEST_2024-05-02"


def request(alpha=0.5):
    return {"event_ids": [EVENT_ID], "fill_alpha": alpha,
            "threshold_units": "quote_price_times_quantity",
            "arms": [{"arm_id": "early", "target_pnl": 2.0, "stop_pnl": -2.0},
                     {"arm_id": "fallback", "target_pnl": 100.0, "stop_pnl": -100.0}]}


def chain_rows():
    rows = []
    expiry = pd.Timestamp("2024-05-17")
    for index, day in enumerate(DATES[:15]):
        bid = 2.0 if index == 0 else 3.0 if index == 1 else 4.0
        for strike in (100.0, 105.0):
            for right in ("C", "P"):
                rows.append({"ticker": "TEST", "obs_date": day, "expiry": expiry,
                             "strike": strike, "right": right, "bid": bid, "ask": bid + 2,
                             "dte": (expiry - day).days, "spot": 100.0,
                             "quote_repaired": False, "year": 2024})
    return rows


def event_rows():
    return [{"event_id": EVENT_ID, "ticker": "TEST", "event_date": DATES[14],
             "year": 2024, "session": "AMC", "session_src": "test",
             "date_conflict": False, "event_cluster_id": None}]


def commit(source, rows=None, *, dates=DATES, events=None, scope="shadow"):
    conn, clock, store = source
    sources = {"earnings_events": event_rows() if events is None else events,
               "option_chains": chain_rows() if rows is None else rows,
               "daily_market": [{"ticker": "MKT", "date": day, "year": 2024} for day in dates]}
    tables, contracts = {}, {}
    for name, values in sources.items():
        contract = contracts[name] = contract_for(name)
        values = sorted(values, key=lambda row: tuple(row[k] for k in contract.primary_key))
        tables[name] = [publish_and_inspect(store, contract, contract_ref_for(contract), values, "2024")]
    return commit_tables(conn, clock, tables, contracts, scope=scope,
                         receipt_id=scope, store=store)


@pytest.fixture
def source(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    yield conn, clock, store
    conn.close()


def run(source, *, document=None, snapshot=None, **kwargs):
    snapshot = commit(source, **kwargs) if snapshot is None else snapshot
    return comparison.compare_runup_exits(Repository(source[0], source[2]), snapshot,
                                         as_of_month="2025-01", document=request() if document is None else document)


def by_arm(report):
    return {row["arm_id"]: row for row in report["rows"]}


def assert_refusal(error, code, category):
    assert error.code == error.problem.code == code
    assert error.problem.category == category
    assert error.problem.retryable is False


def expected_position(event_id=EVENT_ID, ticker="TEST"):
    return {"trade_id": event_id, "ticker": ticker, "entry_date": "2024-04-12",
            "legs": tuple({"expiry": "2024-05-17", "strike": 100.0, "right": right,
                           "side": "buy", "qty": 1} for right in ("C", "P"))}


def test_exact_hand_calculated_pairing_and_identity(source):
    snapshot = commit(source)
    report = run(source, snapshot=snapshot)
    rows = by_arm(report)
    assert report["counts"] == {"requested_count": 1, "admitted_count": 1,
                                "selected_count": 1, "compared_count": 1, "arm_count": 3}
    assert rows["fixed_day"]["pnl"] == 4.0
    assert rows["fixed_day"]["exit_date"] == "2024-05-02"
    assert rows["early"]["pnl"] == 2.0
    assert rows["early"]["exit_value"] == 8.0
    assert rows["early"]["pnl_difference"] == -2.0
    assert rows["early"]["exit_date"] == "2024-04-15"
    assert rows["early"]["reason"] == "target"
    assert rows["fallback"]["pnl_difference"] == 0.0
    assert rows["fallback"]["reason"] == "fixed_day"
    assert len(rows["fixed_day"]["visited_dates"]) == 15
    for row in rows.values():
        assert row["entry_cost"] == 6.0
        assert row["trade_id"] == EVENT_ID
        assert row["snapshot_id"] == snapshot.snapshot_id
        assert row["pnl_basis"] == "mark_based"
        assert row["pnl_units"] == comparison.UNITS
        assert row["fill_alpha"] == row["exit_fill_alpha"] == 0.5
        assert row["fixed_day_pnl"] == 4.0
        assert row["fixed_day_exit_date"] == "2024-05-02"
    binding = report["binding"]
    assert binding["request"] == request()
    assert binding["positions"] == [expected_position()]
    assert binding["baseline"] == {"kind": "fixed_day", "trading_days": 14}
    assert binding["cost_policy"]["commissions"] == "not_modeled"
    assert binding["cost_policy"]["contract_multiplier"] == "not_applied"
    assert set(binding["source_closure"]) >= {"experiments/runup_exit_report.py",
                                              "engine/v2/research/experiment_exits.py"}
    assert report["comparison_id"] == content_hash(binding)
    assert report["report_hash"] == content_hash({k: v for k, v in report.items() if k != "report_hash"})
    assert canonical_json(report) == canonical_json(run(source, snapshot=snapshot))


@pytest.mark.parametrize("alpha,cost,baseline_pnl", [(0, 8.0, 0.0), (0.5, 6.0, 4.0), (1, 4.0, 8.0)])
def test_common_fill_controls_all_arms_without_multiplier(source, alpha, cost, baseline_pnl):
    report = run(source, document=request(alpha))
    assert by_arm(report)["fixed_day"]["pnl"] == baseline_pnl
    assert {row["entry_cost"] for row in report["rows"]} == {cost}
    assert {row["fill_alpha"] for row in report["rows"]} == {alpha}


def test_ambiguous_crossing_conservatively_stops_at_worst_quote(source):
    rows = chain_rows()
    for row in rows:
        if row["obs_date"] == DATES[1]:
            row.update(bid=0.0, ask=8.0)
    early = by_arm(run(source, rows=rows))["early"]
    assert early["reason"] == "stop"
    assert early["ambiguous_exit"] is True
    assert early["exit_fill_alpha"] == 0.0
    assert early["pnl"] == -6.0  # Actual adverse quote mark, not the -2 trigger.
    assert early["pnl_difference"] == -10.0


def test_stop_equality_and_first_crossing_are_consumed(source):
    rows = chain_rows()
    for row in rows:
        if row["obs_date"] == DATES[1]:
            row.update(bid=2.0, ask=2.0)
    early = by_arm(run(source, rows=rows))["early"]
    assert early["reason"] == "stop"
    assert early["ambiguous_exit"] is False
    assert early["pnl"] == -2.0
    assert early["exit_date"] == "2024-04-15"


def test_bmo_control_exits_before_announcement_day(source):
    rows = chain_rows()
    for row in rows:
        row["obs_date"] -= pd.offsets.BDay(1)
        row["dte"] = (row["expiry"] - row["obs_date"]).days
    report = run(source, rows=rows, dates=pd.bdate_range("2024-04-11", "2024-05-03"),
                 events=[{**event_rows()[0], "session": "BMO"}])
    assert report["binding"]["positions"][0]["entry_date"] == "2024-04-11"
    assert by_arm(report)["fixed_day"]["exit_date"] == "2024-05-01"


@pytest.mark.parametrize("missing", [0, 1, 8, 14])
def test_missing_held_leg_refuses_even_after_early_target(source, missing):
    rows = [row for row in chain_rows() if not (
        row["obs_date"] == DATES[missing] and row["strike"] == 100 and row["right"] == "P")]
    with pytest.raises((DataError, OpsError)) as caught:
        run(source, rows=rows)
    assert_refusal(caught.value, "EXPERIMENT_VARIANT_FAILED", "internal")
    assert caught.value.problem.details["compared_count"] == 0
    if missing:
        assert caught.value.problem.details["stage"] == "common_horizon_coverage"
        assert caught.value.problem.details["session"] == DATES[missing].date().isoformat()


def test_coverage_preflight_finishes_before_any_threshold_arm(source, monkeypatch):
    original = comparison.walk_exit
    called = []

    def observe(*args, **kwargs):
        called.append(kwargs["economic_params"]["exit"]["kind"])
        return original(*args, **kwargs)

    monkeypatch.setattr(comparison, "walk_exit", observe)
    rows = [row for row in chain_rows() if row["obs_date"] != DATES[12]]
    with pytest.raises(DataError) as caught:
        run(source, rows=rows)
    assert_refusal(caught.value, "EXPERIMENT_VARIANT_FAILED", "internal")
    assert called == ["fixed_day"]


def test_no_future_reselection_and_input_permutation_is_stable(source):
    rows = chain_rows()
    for row in rows:
        # Tied entry spot resolves to lower strike; huge later ATM prices do not reselect.
        row["spot"] = 102.5 if row["obs_date"] == DATES[0] else 105.0
        if row["obs_date"] != DATES[0] and row["strike"] == 105:
            row.update(bid=100.0, ask=102.0)
    report = run(source, rows=rows)
    assert {leg["strike"] for leg in report["binding"]["positions"][0]["legs"]} == {100.0}
    assert by_arm(report)["fixed_day"]["pnl"] == 4.0
    permuted = run(source, rows=list(reversed(rows)), scope="permuted")
    assert permuted["binding"]["positions"] == report["binding"]["positions"]
    assert [row["pnl"] for row in permuted["rows"]] == [row["pnl"] for row in report["rows"]]


def test_selects_first_expiry_meeting_thirty_dte(source):
    rows = chain_rows()
    for expiry in (pd.Timestamp("2024-05-10"), pd.Timestamp("2024-05-24")):
        rows.extend({**row, "expiry": expiry, "dte": (expiry - row["obs_date"]).days,
                     "bid": row["bid"] * 2, "ask": row["ask"] * 2} for row in chain_rows())
    report = run(source, rows=rows)
    assert report["binding"]["positions"] == [expected_position()]
    assert by_arm(report)["fixed_day"]["entry_cost"] == 6.0
    assert by_arm(report)["fixed_day"]["pnl"] == 4.0


def test_request_order_does_not_change_report(source):
    snapshot = commit(source)
    reordered = request()
    reordered["arms"].reverse()
    assert run(source, snapshot=snapshot, document=reordered) == run(source, snapshot=snapshot)


def test_changed_recipe_fill_or_snapshot_changes_identity(source):
    snapshot = commit(source)
    first = run(source, snapshot=snapshot)
    changed = request()
    changed["arms"][0]["target_pnl"] = 3
    altered_recipe = run(source, snapshot=snapshot, document=changed)
    assert altered_recipe["binding"]["request"]["arms"][0]["target_pnl"] == 3.0
    assert by_arm(altered_recipe)["early"]["exit_date"] == "2024-04-16"
    assert by_arm(altered_recipe)["early"]["pnl"] == 4.0
    assert by_arm(altered_recipe)["early"]["pnl_difference"] == 0.0
    altered_fill = run(source, snapshot=snapshot, document=request(0.25))
    rows = chain_rows()
    rows[-1]["ask"] += 1
    altered_snapshot = run(source, rows=rows, scope="new-head")
    assert len({r["comparison_id"] for r in (first, altered_recipe, altered_fill, altered_snapshot)}) == 4
    assert run(source, snapshot=snapshot) == first


@pytest.mark.parametrize("dates", [DATES[1:], DATES[:14]])
def test_insufficient_observed_calendar_refuses(source, dates):
    with pytest.raises(OpsError) as caught:
        run(source, dates=dates)
    assert_refusal(caught.value, "EXPERIMENT_VARIANT_FAILED", "internal")
    assert caught.value.problem.details["compared_count"] == 0


@pytest.mark.parametrize("expiry", [DATES[5], DATES[15]])
def test_invalid_entry_dte_refuses_even_when_expiry_survives_horizon(source, expiry):
    rows = chain_rows()
    for row in rows:
        row["expiry"] = expiry
        row["dte"] = 30
    with pytest.raises(OpsError) as caught:
        run(source, rows=rows)
    assert_refusal(caught.value, "EXPERIMENT_VARIANT_FAILED", "internal")
    assert caught.value.problem.details["stage"] == "entry_selection"
    assert caught.value.problem.details["compared_count"] == 0


def test_whole_explicit_cohort_refuses_missing_sibling(source):
    events = event_rows() + [{**event_rows()[0], "ticker": "MISSING", "event_id": "MISSING_2024-05-02"}]
    document = request()
    document["event_ids"].append("MISSING_2024-05-02")
    with pytest.raises(OpsError) as caught:
        run(source, events=events, document=document)
    assert_refusal(caught.value, "EXPERIMENT_VARIANT_FAILED", "internal")
    assert caught.value.problem.details["requested_count"] == 2
    assert caught.value.problem.details["admitted_count"] == 2
    assert caught.value.problem.details["compared_count"] == 0


def test_two_event_population_remains_paired_under_reordered_request(source):
    second_id = "MISSING_2024-05-02"
    events = event_rows() + [{**event_rows()[0], "ticker": "SECOND", "event_id": second_id}]
    rows = chain_rows() + [{**row, "ticker": "SECOND", "bid": row["bid"] * 2,
                           "ask": row["ask"] * 2} for row in chain_rows()]
    snapshot = commit(source, rows, events=events)
    document = request()
    document["event_ids"].append(second_id)
    report = run(source, snapshot=snapshot, document=document)
    document["event_ids"].reverse()
    document["arms"].reverse()
    assert report == run(source, snapshot=snapshot, document=document)
    assert report["counts"]["compared_count"] == 2
    assert len(report["rows"]) == 6
    assert report["binding"]["positions"] == [expected_position(second_id, "SECOND"), expected_position()]
    for row in report["rows"]:
        scale = 2 if row["trade_id"] == second_id else 1
        early = row["arm_id"] == "early"
        assert row["entry_cost"] == 6.0 * scale
        assert row["exit_value"] == (8.0 if early else 10.0) * scale
        assert row["pnl"] == (2.0 if early else 4.0) * scale
        assert row["pnl_difference"] == (-2.0 if early else 0.0) * scale
        assert row["fixed_day_pnl"] == 4.0 * scale
        assert row["fixed_day_exit_date"] == "2024-05-02"
        assert row["exit_date"] == ("2024-04-15" if early else "2024-05-02")


@pytest.mark.parametrize("bad_quote", [{"bid": None}, {"bid": 5.0, "ask": 4.0}, {"ask": 0.0}])
def test_unusable_daily_held_quote_after_target_refuses(source, bad_quote):
    rows = chain_rows()
    for row in rows:
        if row["obs_date"] == DATES[10] and row["strike"] == 100 and row["right"] == "P":
            row.update(bad_quote)
    with pytest.raises(DataError) as caught:
        run(source, rows=rows)
    assert_refusal(caught.value, "EXPERIMENT_VARIANT_FAILED", "internal")
    assert caught.value.problem.details["stage"] == "common_horizon_coverage"
    assert caught.value.problem.details["compared_count"] == 0


def test_random_holdout_refuses_before_quote_reads(source, monkeypatch):
    event_id = next(f"held-{number}" for number in range(1000)
                    if ExperimentHoldouts.random_membership(f"held-{number}"))
    snapshot = commit(source, events=[{**event_rows()[0], "event_id": event_id}])
    document = request()
    document["event_ids"] = [event_id]

    def forbidden(*args, **kwargs):
        pytest.fail("random holdout reached quotes")

    monkeypatch.setattr(comparison, "_positions", forbidden)
    with pytest.raises(DataError) as caught:
        run(source, snapshot=snapshot, document=document)
    assert_refusal(caught.value, "HOLDOUT_ACCESS_DENIED", "validation")
    assert caught.value.problem.details["stage"] == "population"


def test_holdout_refuses_before_selection(source, monkeypatch):
    snapshot = commit(source)

    def forbidden(*args, **kwargs):
        pytest.fail("quotes opened before holdout admission")

    monkeypatch.setattr(comparison, "_positions", forbidden)
    with pytest.raises(DataError) as caught:
        comparison.compare_runup_exits(Repository(source[0], source[2]), snapshot,
                                       as_of_month="2024-05", document=request())
    assert_refusal(caught.value, "HOLDOUT_ACCESS_DENIED", "validation")
    assert caught.value.problem.details["stage"] == "population"
    assert caught.value.problem.details["admitted_count"] == 0


@pytest.mark.parametrize("change", [
    {"fill_alpha": True}, {"fill_alpha": float("nan")}, {"fill_alpha": 10**1000},
    {"fill_alpha": -0.1}, {"fill_alpha": 1.1}, {"threshold_units": "dollars"},
    {"event_ids": []}, {"event_ids": [EVENT_ID, EVENT_ID]}, {"event_ids": [None]},
    {"arms": []}, {"arms": None}, {"extra": "ignored?"},
    {"arms": [{"arm_id": "fixed_day", "target_pnl": 1, "stop_pnl": -1}]},
    {"arms": [{"arm_id": "x", "target_pnl": 0, "stop_pnl": -1}]},
    {"arms": [{"arm_id": "x", "target_pnl": 1, "stop_pnl": 0}]},
    {"arms": [{"arm_id": "x", "target_pnl": 1, "stop_pnl": -1, "trading_days": 1}]},
    {"event_ids": [str(n) for n in range(10001)]},
    {"arms": request()["arms"] * 2},
    *[{"arms": [{"arm_id": name, "target_pnl": 1, "stop_pnl": -1}]}
      for name in ("", " ", None)],
    *[{"arms": [{"arm_id": "x", "target_pnl": target, "stop_pnl": stop}]}
      for target, stop in ((float("nan"), -1), (float("inf"), -1), (1, -float("inf")), (True, -1))],
    {"arms": [{"arm_id": str(n), "target_pnl": 1, "stop_pnl": -1} for n in range(9)]},
])
def test_bad_request_refuses_before_reads(change):
    with pytest.raises(OpsError) as caught:
        comparison.compare_runup_exits(None, None, as_of_month="2025-01", document={**request(), **change})
    assert_refusal(caught.value, "INVALID_EXPERIMENT_SPEC", "validation")
    assert caught.value.problem.details == {}


@pytest.mark.parametrize("document", [None, [], "request", 1])
def test_nonobject_request_refuses_before_reads(document):
    with pytest.raises(OpsError) as caught:
        comparison.compare_runup_exits(None, None, as_of_month="2025-01", document=document)
    assert_refusal(caught.value, "INVALID_EXPERIMENT_SPEC", "validation")


def test_maximum_declaration_bounds_are_accepted():
    document = {**request(), "event_ids": [str(n) for n in range(10000)],
                "arms": [{"arm_id": str(n), "target_pnl": 1, "stop_pnl": -1} for n in range(8)]}
    result = comparison.validate_request(document)
    assert result["event_ids"] == sorted(document["event_ids"])
    assert result["arms"] == document["arms"]


def cli_args(tmp_path, snapshot):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request()))
    return ["--catalog", str(tmp_path / "catalog.sqlite"), "--store-root", str(tmp_path / "store"),
            "--snapshot-id", snapshot.snapshot_id, "--as-of-month", "2025-01", "--request", str(path)]


def test_real_cli_is_read_only_and_exactly_one_json_report(source, tmp_path, capsys, monkeypatch):
    snapshot = commit(source)
    args = cli_args(tmp_path, snapshot)
    before = list(source[0].iterdump())
    files = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("exploratory comparison attempted a database write")

    original = sqlite3.connect

    def observe(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.set_authorizer(lambda action, *unused: forbidden() if action in {
            sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE} else sqlite3.SQLITE_OK)
        return connection

    monkeypatch.setattr(cli.sqlite3, "connect", observe)
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "complete"
    assert report["counts"]["compared_count"] == 1
    assert cli.main([*args, "--no-ledger"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert list(source[0].iterdump()) == before
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == files


def test_cli_refusal_has_counts_and_no_partial_metrics(source, tmp_path, capsys):
    rows = [row for row in chain_rows() if row["obs_date"] != DATES[8]]
    snapshot = commit(source, rows)
    assert cli.main(cli_args(tmp_path, snapshot)) == 2
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"status", "refused", "details"}
    assert report["refused"] == "EXPERIMENT_VARIANT_FAILED"
    assert report["details"]["compared_count"] == 0
    assert report["details"]["stage"] == "common_horizon_coverage"


@pytest.mark.parametrize("contents", [b"x" * (1024 * 1024 + 1), b"{", b"\xff", b"null",
                                      b"[" * 100000 + b"0" + b"]" * 100000,
                                      json.dumps(request()).replace('"fill_alpha": 0.5',
                                                                   '"fill_alpha": 0, "fill_alpha": 1').encode()])
def test_cli_invalid_json_refuses_without_catalog_creation(tmp_path, capsys, contents):
    path = tmp_path / "request.json"
    path.write_bytes(contents)
    assert cli.main(["--catalog", str(tmp_path / "absent.sqlite"), "--store-root", str(tmp_path / "store"),
                     "--snapshot-id", "missing", "--as-of-month", "2025-01", "--request", str(path)]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "refused", "refused": "INVALID_EXPERIMENT_SPEC", "details": {}}
    assert not (tmp_path / "absent.sqlite").exists()


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_cli_nonregular_request_refuses_without_waiting_or_catalog_creation(tmp_path, kind):
    path = tmp_path / "request"
    os.mkfifo(path) if kind == "fifo" else path.mkdir()
    result = subprocess.run([sys.executable, "-m", "experiments.v2_runup_exits",
                             "--catalog", str(tmp_path / "absent.sqlite"),
                             "--store-root", str(tmp_path / "store"), "--snapshot-id", "missing",
                             "--as-of-month", "2025-01", "--request", str(path)],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 2
    assert json.loads(result.stdout) == {"status": "refused", "refused": "INVALID_EXPERIMENT_SPEC", "details": {}}
    assert not (tmp_path / "absent.sqlite").exists()


def test_regular_symlink_request_retains_supported_file_behavior(tmp_path):
    target = tmp_path / "request.json"
    target.write_text(json.dumps(request()))
    link = tmp_path / "request-link.json"
    link.symlink_to(target)
    assert cli._request(link) == request()


def fractional_request():
    return {**request(), "threshold_units": "entry_debit_fraction",
            "arms": [{"arm_id": "ratio", "target_pnl": 0.5, "stop_pnl": -0.5}]}


def test_fractional_multi_event_comparison_uses_each_premium(source):
    second_id = "MISSING_2024-05-02"
    events = event_rows() + [{**event_rows()[0], "ticker": "SECOND", "event_id": second_id}]
    rows = chain_rows() + [{**row, "ticker": "SECOND", "bid": row["bid"] * 3,
                           "ask": row["ask"] * 3} for row in chain_rows()]
    document = fractional_request()
    document["event_ids"].append(second_id)
    snapshot = commit(source, rows, events=events)
    report = run(source, snapshot=snapshot, document=document)
    assert report["counts"]["compared_count"] == 2
    assert report["binding"]["request"]["threshold_units"] == "entry_debit_fraction"
    for row in report["rows"]:
        scale = 3 if row["trade_id"] == second_id else 1
        assert row["entry_cost"] == 6.0 * scale
        assert row["pnl"] == 4.0 * scale
        assert row["pnl_units"] == comparison.UNITS
        if row["arm_id"] == "ratio":
            assert row["exit_date"] == "2024-04-16"
            assert row["reason"] == "target"
            assert row["thresholds"] == {"units": comparison.UNITS, "entry_debit": 6.0 * scale,
                                          "target_pnl": 3.0 * scale, "stop_pnl": -3.0 * scale}
        else:
            assert "thresholds" not in row
            assert row["exit_date"] == "2024-05-02"
    assert report == run(source, snapshot=snapshot, document=document)
    absolute = run(source, snapshot=snapshot, document={**document, "threshold_units": comparison.UNITS})
    assert absolute["comparison_id"] != report["comparison_id"]
    assert all("thresholds" not in row for row in absolute["rows"])
    assert {row["exit_date"] for row in absolute["rows"] if row["arm_id"] == "ratio"} == {"2024-04-15"}


def test_fractional_request_is_consumed_by_real_cli(source, tmp_path, capsys):
    snapshot = commit(source)
    args = cli_args(tmp_path, snapshot)
    (tmp_path / "request.json").write_text(json.dumps(fractional_request()))
    before = list(source[0].iterdump())
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    ratio = by_arm(report)["ratio"]
    assert ratio["exit_date"] == "2024-04-16"
    assert ratio["thresholds"]["target_pnl"] == 3.0
    assert list(source[0].iterdump()) == before


def test_fractional_comparison_still_requires_full_common_coverage(source):
    rows = [row for row in chain_rows() if row["obs_date"] != DATES[12]]
    with pytest.raises(DataError) as caught:
        run(source, rows=rows, document=fractional_request())
    assert caught.value.problem.details["stage"] == "common_horizon_coverage"
    assert caught.value.problem.details["compared_count"] == 0


def test_fractional_threshold_overflow_refuses_without_partial_metrics(source):
    document = fractional_request()
    document["arms"][0]["target_pnl"] = 1e308
    with pytest.raises(DataError) as caught:
        run(source, document=document)
    assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"
    assert caught.value.problem.details["stage"] == "exit_comparison"
    assert caught.value.problem.details["compared_count"] == 0


@pytest.mark.parametrize("units", [None, "percent", [], {}, True])
def test_invalid_fraction_units_refuse_before_reads(units):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        comparison.compare_runup_exits(None, None, as_of_month="2025-01",
                                       document={**fractional_request(), "threshold_units": units})


def test_fraction_request_refuses_event_specific_thresholds_before_reads():
    document = fractional_request()
    document["arms"][0]["event_thresholds"] = {EVENT_ID: 0.5}
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        comparison.compare_runup_exits(None, None, as_of_month="2025-01", document=document)
