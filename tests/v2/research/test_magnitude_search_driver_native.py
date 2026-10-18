"""Fresh native source seals and actual bounded children over tiny synthetic Parquet."""
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest

from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.data.repository import Repository
from engine.v2.foundation import content_hash
from engine.v2.models.no_fit import no_fit_guard
from experiments import magnitude_search_execution as execution
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_trials as trials
from experiments import v2_magnitude_search_run as driver
from experiments.magnitude_search_plan import seal_search_plan
from experiments.magnitude_search_records import read_record
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_magnitude_search_execution import LIMITS
from tests.v2.research.test_magnitude_search_plan import source  # noqa: F401


def test_fresh_registered_native_driver_executes_and_replays_read_only(source, tmp_path, monkeypatch, capsys):  # noqa: F811
    request = {"schema_version": "magnitude_search_request.v1",
        "inner_years": {"2024": ["2023"], "2025": ["2023", "2024"]},
        "space": {"history_ridge": {"alpha": [0.5, 2]}}, "max_candidates": 2, "max_fits": 24}
    sealed = seal_search_plan(source.conn, source.store, source.registration, request)
    request_path, ref_path = tmp_path / "request.json", tmp_path / "plan-ref.json"
    request_path.write_text(json.dumps(request))
    ref_path.write_text(json.dumps(sealed["plan_ref"]))
    common = ["--spec", str(source.spec_path), "--request", str(request_path), "--plan-ref", str(ref_path),
              "--catalog", source.catalog_path, "--store-root", str(source.store.root)]
    limits = [word for key, value in LIMITS.items() for word in ("--" + key.replace("_", "-"), str(value))]
    assert driver.main(["seal", *common]) == 0
    binding = json.loads(capsys.readouterr().out)
    assert binding["sealed"] == sealed["plan"]["plan_id"]
    assert binding["run_id"] == source.registration.run_id
    def journal(slot):
        return read_record(source.conn, source.store, source.registration.run_id, slot)
    root = Path(driver.__file__).resolve().parents[1]
    sources = journal("driver-seal")["driver_sources"]
    assert "experiments/v2_magnitude_search_run.py" in sources
    assert "experiments/magnitude_search_worker.py" in journal("seal")["recorder_sources"]
    for closure in (source.registration.document["source_closure"],
                    sealed["plan"]["implementation"]["source_closure"], journal("seal")["recorder_sources"], sources):
        assert closure == {name: "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest() for name in closure}
    children = []
    original = execution.subprocess.Popen
    def launch(args, **kwargs):
        child = original(args, **kwargs)
        children.append((child, Path(args[3]).parent))
        return child
    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    assert driver.main(["run", *common, *limits]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(children) == 8  # Six inner candidates/folds plus two selected outer folds.
    assert all(child.poll() == 0 and not path.exists() for child, path in children)
    assert result["schema_version"] == "magnitude_search_outer_report.v1"
    assert result["plan_id"] == sealed["plan"]["plan_id"]
    assert [fold["outer_year"] for fold in result["folds"]] == ["2024", "2025"]
    assert result["forecast_count"] == 30
    errors = [abs(row["predicted_absolute_move_pct"] - row["absolute_move_pct"]) for row in result["predictions"]]
    assert result["metrics"] == {"count": 30, "mae": pytest.approx(math.fsum(errors) / 30),
                                 "rmse": pytest.approx(math.sqrt(math.fsum(error ** 2 for error in errors) / 30))}
    assert result["result_id"] == content_hash({key: value for key, value in result.items() if key != "result_id"})
    assert result["final_read_authority"] == "none"
    assert result["prospective_candidate_id"] == result["folds"][-1]["candidate"]["candidate_id"]
    before = catalog_rows(source.conn)
    def forbidden(*args, **kwargs):
        raise AssertionError("completed driver replay started a fit child")
    monkeypatch.setattr(execution.subprocess, "Popen", forbidden)
    with no_fit_guard(), legacy_guard():
        assert driver.main(["run", *common, *limits]) == 0
        assert json.loads(capsys.readouterr().out) == result
        assert driver.main(["collect", *common]) == 0
        assert json.loads(capsys.readouterr().out) == result
        assert driver.main(["seal", *common]) == 0
        assert json.loads(capsys.readouterr().out) == binding
    assert catalog_rows(source.conn) == before
    monkeypatch.setattr(Repository, "scan", forbidden)
    monkeypatch.setattr(trials, "load_magnitude_targets", forbidden)
    monkeypatch.setattr(outer, "load_magnitude_targets", forbidden)
    changed = tmp_path / "changed-code"
    for name in sources:
        target = changed / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    entry = changed / "experiments/v2_magnitude_search_run.py"
    entry.write_text(entry.read_text() + "\n# synthetic changed driver source\n")
    monkeypatch.setattr(driver, "__file__", str(entry))
    with no_fit_guard(), legacy_guard():
        for action in (["run", *common, *limits], ["collect", *common]):
            assert driver.main(action) == 2
            assert json.loads(capsys.readouterr().out)["refused"]["code"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert catalog_rows(source.conn) == before
