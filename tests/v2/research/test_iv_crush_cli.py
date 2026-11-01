"""Real contained IV CLI lifecycle, synthetic native data and read-only replay."""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.foundation import canonical_json, to_document
from engine.v2.ops.experiments import SPEC_FIELDS
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments import iv_crush_run as cli
from experiments import size_search
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_search_execution import oracle_case, search_case  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_magnitude_search_request import request as size_request
from tests.v2.research.test_native_registration import artifact_object_bytes
from tests.v2.research.test_size_review_contract import panel_request


def request():
    document = size_request()
    panel = panel_request()
    panel.update(feature_snapshot_id=document["feature_snapshot_id"], history_snapshot=document["history_snapshot_id"],
                 price_history_snapshot="snap_" + "4" * 32)
    document.update(schema_version="regression_search_request.v1", panel_request=panel,
                    feature_dictionary_hash=recipe("HMO")["dictionary_hash"])
    document["comparison"].update(reference="iv_crush_causal_refit", sanity_reference="training_mean")
    return document


def arguments(tmp_path, action="report"):
    return [action, "--request", str(tmp_path / "request.json"), "--spec", str(tmp_path / "spec.json"),
        "--catalog", str(tmp_path / "catalog.sqlite"), "--store-root", str(tmp_path / "store"),
        "--plan-ref", str(tmp_path / "ref.json"), "--no-ledger"]


@pytest.mark.parametrize("mutation", ["schema", "shape", "child_memory", "parent_wall", "grid", "selection"])
def test_bad_requests_refuse_before_catalog_and_containment(tmp_path, monkeypatch, capsys, mutation):
    document = request()
    if mutation == "schema":
        document["schema_version"] = "magnitude_search_request.v2"
    elif mutation == "shape":
        del document["limits"]
    elif mutation == "child_memory":
        document["limits"]["child"]["memory_bytes"] = 5
    elif mutation == "parent_wall":
        document["limits"]["parent"]["wall_seconds"] = True
    elif mutation == "grid":
        document["space"]["ridge"]["alpha"] = [1.]
    else:
        document["selection"] = {}
    (tmp_path / "request.json").write_text(json.dumps(document))
    def forbidden(*a, **kw):
        pytest.fail("invalid request reached containment or catalog")
    monkeypatch.setattr(cli, "_contain", forbidden)
    monkeypatch.setattr(cli, "connect", forbidden)
    assert cli.main(arguments(tmp_path)) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == "INVALID_EXPERIMENT_SPEC"


def test_parent_alarm_and_memory_are_installed_before_catalog(tmp_path, monkeypatch, capsys):
    from contextlib import contextmanager

    from engine.v2.ops.errors import fail
    document, observed = request(), []
    (tmp_path / "request.json").write_text(json.dumps(document))
    (tmp_path / "spec.json").write_text(json.dumps({"hypothesis": "synthetic", "primary_arm_id": "candidate",
        "economic_params": {}, "price_source": "iv_crush_targets", "runner": "experiments/iv_crush_search.py"}))
    (tmp_path / "store").mkdir()
    monkeypatch.setattr(size_search.signal, "signal", lambda *a: observed.append("alarm_handler"))
    monkeypatch.setattr(size_search.signal, "alarm", lambda seconds: observed.append(("alarm", seconds)))
    @contextmanager
    def watch(memory):
        observed.append(("rss", memory))
        yield
        observed.append("rss_closed")
    monkeypatch.setattr(cli, "rss_watchdog", watch)
    def connect(*a, **kw):
        observed.append("catalog")
        raise fail("RESOURCE_UNAVAILABLE", "synthetic catalog unavailable")
    monkeypatch.setattr(cli, "connect", connect)
    assert cli.main(arguments(tmp_path, "seal")) == 2
    assert observed[:4] == ["alarm_handler", ("alarm", 5), ("rss", 4), "catalog"]
    assert "rss_closed" in observed and observed[-1] == ("alarm", 0)
    assert json.loads(capsys.readouterr().out)["refused"] == "RESOURCE_UNAVAILABLE"


def test_completed_run_routes_once_without_deadline_or_execution(monkeypatch):
    plan = {"plan_id": "synthetic", "folds": [{"outer": {"year": "2024"}}]}
    calls = []
    monkeypatch.setattr(size_search, "_admit_driver", lambda *a: plan)
    monkeypatch.setattr(size_search, "read_record", lambda *a: {})
    def forbidden(*a, **kw):
        pytest.fail("completed operator began execution or deadline")
    monkeypatch.setattr(size_search, "remaining", forbidden)
    monkeypatch.setattr(size_search, "run_search", forbidden)
    def collect(*context):
        calls.append(context)
        return "verified IV report"
    context = (1, 2, 3, SimpleNamespace(run_id="synthetic"), 5, 6)
    assert size_search._run(context, collector=collect) == "verified IV report"
    assert calls == [context]


@pytest.mark.test_timeout(1800)
def test_real_cli_preflight_register_seal_run_and_guarded_replay(oracle_case, tmp_path):  # noqa: F811
    data = oracle_case
    root = Path(cli.__file__).resolve().parents[1]
    directory = tmp_path / "iv-cli"
    directory.mkdir()
    spec = {name: getattr(data.spec, name) for name in SPEC_FIELDS}
    spec["experiment_id"] = "synthetic-iv-contained-cli"
    paths = {name: directory / (name + ".json") for name in ("spec", "request", "plan-ref")}
    paths["spec"].write_text(json.dumps(spec))
    paths["request"].write_text(canonical_json(data.plan["request"]))
    catalog = data.conn.execute("PRAGMA database_list").fetchone()[2]
    common = ["--spec", str(paths["spec"]), "--request", str(paths["request"]), "--catalog", catalog,
              "--store-root", str(data.store.root), "--no-ledger"]
    fresh = ["--scope", "panel-source", "--target-snapshot-id", data.targets.snapshot_id,
             "--training-start", "2022-06-01", "--labels-before", "2025-01-01"]
    env = {**os.environ, "PYTHONPATH": str(root), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    def invoke(action, extra, *, guarded=False):
        command = [sys.executable, "-m", "experiments.iv_crush_search"]
        if guarded:
            script = """import sys
from engine.models.no_fit import no_fit_guard as legacy
from engine.v2.models.no_fit import no_fit_guard
from experiments import regression_search_binding, size_comparison_run, size_search
from experiments import iv_crush_run
from experiments.iv_crush_run import main
def forbidden(*a, **kw): raise AssertionError('completed CLI reopened inputs or executed')
regression_search_binding.fold_inputs = forbidden
size_comparison_run.resolved_limits = forbidden
size_search.remaining = forbidden
iv_crush_run.connect = forbidden
with legacy(), no_fit_guard(): result = main(sys.argv[1:])
raise SystemExit(result)
"""
            command = [sys.executable, "-c", script]
        process = subprocess.run([*command, action, *common, *extra], cwd=root, env=env,
                                 capture_output=True, text=True, timeout=1700)
        assert process.returncode == 0, process.stdout + process.stderr[-6000:]
        document = json.loads(process.stdout)
        assert process.stdout == canonical_json(document) + "\n"
        assert f"stage={action} start" in process.stderr and f"stage={action} end" in process.stderr
        return document
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    preflight = invoke("preflight", fresh)
    assert preflight["preflight"]["maximum_child_invocations"] == 35
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == artifacts
    registered = invoke("register", fresh)
    paths["plan-ref"].write_text(canonical_json(registered["plan_ref"]))
    remaining = ["--plan-ref", str(paths["plan-ref"])]
    seal = invoke("seal", remaining)
    assert seal["sealed"] == registered["plan"]["plan_id"]
    report = invoke("run", remaining)
    assert report["search"]["metrics"] == {"count": 3, "mae": 10., "rmse": 10.}
    assert report["paired"]["paired_events"] == 0
    assert report["paired"]["broad_coverage_metrics"]["training_mean"]["mae"] == 10.
    assert sum(row["completed_fit_count"] for row in report["inner_fits"]) == 99
    assert report["preflight"]["counts"]["total_component_fits"] == 105
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    assert invoke("report", remaining, guarded=True) == report
    assert invoke("run", remaining, guarded=True) == report
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == artifacts
    assert registered["plan"]["target"] == to_document(data.plan["target"])
    # A real completed catalog copied into DELETE mode must remain physically unchanged.
    replay_catalog = directory / "replay.sqlite"
    with sqlite3.connect(replay_catalog) as destination:
        data.conn.backup(destination)
        assert destination.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    common[common.index("--catalog") + 1] = str(replay_catalog)
    before_file = replay_catalog.read_bytes(), replay_catalog.stat().st_mtime_ns
    assert invoke("report", remaining, guarded=True) == report
    assert invoke("run", remaining, guarded=True) == report
    assert (replay_catalog.read_bytes(), replay_catalog.stat().st_mtime_ns) == before_file
    assert not Path(str(replay_catalog) + "-wal").exists()
    assert not Path(str(replay_catalog) + "-shm").exists()
