"""Smoke runs exercise real evaluation and the real, temporary CSV ledger."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine import paths
from engine.evaluate import PreregistrationError
from experiments import lib
from experiments.new_experiment import scaffold


def _spec(grid=True):
    return {
        "id": "EXP-901", "title": "synthetic ledger probe",
        "primary_spec": {"probe": "primary"},
        "grid": {"probe": ["second", "third"]} if grid else {},
        "walk_forward": {"min_train_years": 1},
        "preregistered_at": "2019-01-01T00:00:00+00:00",
    }


def _trades():
    dates = pd.date_range("2020-01-01", periods=12, freq="90D")
    returns = [-0.1, 0.2] * 6
    return pd.DataFrame({
        "event_id": [f"probe-{i}" for i in range(len(dates))], "ticker": "TEST",
        "event_date": dates, "entry_date": dates - pd.Timedelta(days=1),
        "exit_date": dates + pd.Timedelta(days=1), "fill_alpha": 0.5,
        "entry_cost": 1.0, "exit_value": [1 + ret for ret in returns], "ret": returns,
    })


def _run(spec, run_dir, **kwargs):
    return lib.evaluate_with_grid(
        spec, _trades(), run_dir, alphas=(0.5,), fractions=(0.05,),
        mc_paths=2, stress=False, **kwargs,
    )


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_no_ledger_leaves_missing_or_existing_ledger_unchanged(tmp_root, grid, existing):
    ledger = tmp_root / "experiments" / "LEDGER.csv"
    assert lib.LEDGER_PATH == ledger
    spec = _spec(grid)
    if existing:
        lib.ledger_append([{
            "id": spec["id"], "spec_hash": lib.spec_hash(spec), "date": "2019-01-01",
            "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "",
            "promoted": "False",
        }])
    before = ledger.read_bytes() if existing else None
    run_dir = tmp_root / "run"
    for _ in range(2):
        result = _run(spec, run_dir, record=False, write_report=False)
        assert result.results["backtest"]["n_events"] == 12
        assert (ledger.read_bytes() if ledger.exists() else None) == before
    arms = 3 if grid else 1
    assert len(list((run_dir / "results").glob("metrics_*.json"))) == arms
    assert len((run_dir / "results" / "run_log.jsonl").read_text().splitlines()) == 2 * arms
    assert (run_dir / "ARMS.md").read_text().count("**secondary**") == arms - 1


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("record_kwargs", [{}, {"record": True}])
def test_recording_appends_each_arm_after_smoke(tmp_root, grid, record_kwargs):
    spec = _spec(grid)
    run_dir = tmp_root / "run"
    ledger = tmp_root / "explicit-ledger.csv"
    _run(spec, run_dir, ledger_path=ledger, record=False, write_report=False)
    assert not ledger.exists()
    _run(spec, run_dir, ledger_path=ledger, write_report=False, **record_kwargs)
    rows = lib.ledger_read(ledger)
    expected = [lib.spec_hash(spec)]
    for value in spec["grid"].get("probe", []):
        cell = {**spec, "primary_spec": {"probe": value}, "grid_cell": True}
        expected.append(lib.spec_hash(cell))
    assert rows["spec_hash"].tolist() == expected
    assert rows["stage"].tolist() == ["ran"] * len(expected)
    before = ledger.read_bytes()
    _run(spec, run_dir, ledger_path=ledger, write_report=False, **record_kwargs)
    assert ledger.read_bytes().startswith(before)
    assert lib.ledger_read(ledger)["spec_hash"].tolist() == expected * 2
    assert not lib.LEDGER_PATH.exists()


def test_no_ledger_keeps_real_reports_and_run_artifacts(tmp_root):
    run_dir = tmp_root / "run"
    result = _run(_spec(), run_dir, record=False)
    assert result.report_path == run_dir / "REPORT.md"
    assert result.report_path.is_file()
    arm_dirs = list((run_dir / "arms").iterdir())
    assert len(arm_dirs) == 2
    for folder in [run_dir, *arm_dirs]:
        assert (folder / "REPORT.md").is_file()
        assert list((folder / "figures").glob("*.png"))
    assert len(list((run_dir / "results").glob("transactions_*.csv"))) == 3
    logs = [json.loads(line) for line in (run_dir / "results" / "run_log.jsonl").read_text().splitlines()]
    assert len(logs) == 3
    assert not lib.LEDGER_PATH.exists()


def test_no_ledger_still_enforces_preregistration(tmp_root):
    spec = _spec()
    del spec["preregistered_at"]
    with pytest.raises(PreregistrationError):
        _run(spec, tmp_root / "run", record=False, write_report=False)
    assert not lib.LEDGER_PATH.exists()
    assert not (tmp_root / "run").exists()


@pytest.mark.parametrize("record", [False, True])
def test_failed_secondary_retains_completed_artifacts_without_index(tmp_root, monkeypatch, record):
    import engine.evaluate as evaluator

    real_evaluate = evaluator.evaluate

    def fail_secondary(spec, trades, **kwargs):
        if spec.get("grid_cell"):
            raise RuntimeError("synthetic secondary failure")
        return real_evaluate(spec, trades, **kwargs)

    run_dir = tmp_root / "run"
    run_dir.mkdir()
    (run_dir / "ARMS.md").write_text("stale index")
    monkeypatch.setattr(evaluator, "evaluate", fail_secondary)
    with pytest.raises(RuntimeError, match="synthetic secondary failure"):
        _run(_spec(), run_dir, record=record)
    assert not (run_dir / "ARMS.md").exists()
    assert (run_dir / "REPORT.md").is_file()
    assert len(list((run_dir / "results").glob("metrics_*.json"))) == 1
    assert len((run_dir / "results" / "run_log.jsonl").read_text().splitlines()) == 1
    assert lib.ledger_read()["stage"].tolist() == (["ran"] if record else [])
    assert lib.LEDGER_PATH.exists() == record


def _runner(tmp_root, monkeypatch):
    monkeypatch.setattr(paths, "SNAPSHOT_FILE", tmp_root / "snapshot.json")
    monkeypatch.syspath_prepend(str(tmp_root))  # Restore run.py's sys.path edit, too.
    folder = scaffold("synthetic runner", "synthetic hypothesis", root=tmp_root / "experiments")
    return runpy.run_path(str(folder / "run.py"))


@pytest.mark.parametrize("argv, expected", [([], True), (["--no-ledger"], False)])
def test_scaffold_forwards_record_flag(tmp_root, monkeypatch, argv, expected):
    runner = _runner(tmp_root, monkeypatch)
    before = lib.LEDGER_PATH.read_bytes()
    trades = _trades()
    calls = []
    monkeypatch.setitem(runner["main"].__globals__, "build_trades", lambda: trades)

    def evaluate(spec, frame, run_dir, **kwargs):
        calls.append((spec, frame, run_dir, kwargs))
        return SimpleNamespace(report_path=run_dir / "REPORT.md")

    monkeypatch.setattr(lib, "evaluate_with_grid", evaluate)
    runner["main"](argv)
    assert len(calls) == 1
    assert calls[0][1] is trades
    assert calls[0][3] == {"record": expected}
    assert lib.LEDGER_PATH.read_bytes() == before
    assert lib.ledger_read()["stage"].tolist() == ["planned"]


ROOT = Path(__file__).resolve().parents[3]


def _exp118_runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))  # Mirror run.py's own sys.path edit.
    return runpy.run_path(
        str(ROOT / "experiments" / "EXP-118_gate_retrain_on_the_expanded_replay_univ" / "run.py")
    )


def test_exp118_refuses_unregistered_spec_before_baseline_or_arms(monkeypatch):
    runner = _exp118_runner(monkeypatch)
    spec = _spec()
    del spec["preregistered_at"]
    monkeypatch.setattr(lib, "load_spec", lambda _path: spec)
    baseline_calls = []

    def refuse_baseline():
        baseline_calls.append("champion_baseline reached")
        raise AssertionError("baseline setup must not run before preregistration validation")

    monkeypatch.setitem(runner, "champion_baseline", refuse_baseline)
    monkeypatch.setattr(sys, "argv", ["run.py", "--no-ledger"])
    with pytest.raises(PreregistrationError):
        runner["main"]()
    assert baseline_calls == []


def test_scaffold_help_exposes_no_ledger_before_trade_build(tmp_root, monkeypatch, capsys):
    runner = _runner(tmp_root, monkeypatch)
    before = lib.LEDGER_PATH.read_bytes()
    with pytest.raises(SystemExit) as exc:
        runner["main"](["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--no-ledger" in help_text and "smoke/subset" in help_text
    assert lib.LEDGER_PATH.read_bytes() == before
