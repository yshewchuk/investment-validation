"""P6 slice 5 (training half): recipes, frozen states and artifacts as jobs."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactStore, SystemClock
from engine.v2.models.training import current_recipes
from engine.v2.ops import cli, executor, stages, training
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.plans import request_from_plan
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, catalog, run_until

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _params(**changes):
    fields = {"expected_ids": ("training",), "mode": "state",
              "state": "paired_residual_pool"}
    fields.update(changes)
    return training.TrainingParameters(**fields)


def _problems(**changes):
    return training.training_parameter_problems(None, _params(**changes))


def _recipe_key(calibration):
    for key, recipe in current_recipes().items():
        if (recipe.folds.kind == "request_cutoff") == calibration:
            return key.label()
    raise AssertionError("no matching recipe in the registry")


def test_training_kind_is_registered_with_experiment_heavy_profile():
    kind = stages.registry().get("training")
    assert kind.resource_classes == frozenset({"experiment_heavy"})
    assert kind.checkpoint_contract == "training_job_result.v1.0"
    assert kind.store_domains == (("legacy_store", "read"),)
    assert kind.retry.max_attempts == 1
    with pytest.raises(OpsError, match="unsupported job kind"):
        stages.registry().get("bogus_training_kind")


def test_training_parameters_reject_unknown_mode():
    problems = _problems(mode="forecast")
    assert any("mode" in problem for problem in problems)


def test_training_parameters_reject_expected_ids_mismatch():
    problems = _problems(expected_ids=("training", "extra"))
    assert any("expected_ids" in problem for problem in problems)


def test_recipe_mode_rejects_unknown_recipe_key():
    problems = _problems(mode="recipe", recipe="nope:*:champion", state="")
    assert any("recipe" in problem for problem in problems)


def test_recipe_mode_rejects_malformed_recipe_string():
    problems = _problems(mode="recipe", recipe="size", state="")
    assert any("recipe" in problem for problem in problems)


def test_calibration_recipe_requires_alpha_and_cutoff():
    key = _recipe_key(calibration=True)
    problems = _problems(mode="recipe", recipe=key, state="")
    assert any("alpha" in problem for problem in problems)
    assert any("cutoff" in problem for problem in problems)
    assert _problems(mode="recipe", recipe=key, state="", alpha=0.5,
                     cutoffs=("2026-09-18",)) == []


def test_non_calibration_recipe_rejects_alpha_or_cutoff():
    key = _recipe_key(calibration=False)
    assert _problems(mode="recipe", recipe=key, state="", alpha=0.5)
    assert _problems(mode="recipe", recipe=key, state="", cutoffs=("2026-09-18",))
    assert _problems(mode="recipe", recipe=key, state="") == []


def test_state_mode_rejects_board_analog_and_trailing_cutoff_state_names():
    for name in ("board_analog_matcher", "trailing_pnl_cutoff"):
        problems = _problems(mode="state", state=name)
        assert any("state" in problem for problem in problems), name


def test_state_mode_rejects_extra_cutoff_on_non_paired_pool_member():
    assert _problems(mode="state", state="driver_residual_pool:size",
                     cutoffs=("2026-09-18",))
    assert _problems(mode="state", state="paired_residual_pool",
                     cutoffs=("2026-09-18",)) == []


def test_training_plan_names_experiment_heavy_and_refuses_invalid_params():
    plan = training.training_plan(mode="state", state="paired_residual_pool")
    assert plan["kind"] == "training"
    assert plan["resource_class"] == "experiment_heavy"
    assert plan["effects"] == ["staged"]
    assert plan["parameters"]["mode"] == "state"
    with pytest.raises(OpsError, match="training plan is invalid"):
        training.training_plan(mode="nope")


def test_training_worker_routes_each_mode(tmp_path, monkeypatch):
    from tools import phase5_training_job as job

    calls = []

    def record(name):
        def run(*args, **kwargs):
            calls.append((name, kwargs.get("plan_only")))
            if name == "run_training_job":
                return SimpleNamespace(outcomes=(
                    SimpleNamespace(fold_id="fold-1", status="fitted"),))
            return {"mode": name}
        return run

    for name in ("run_training_job", "run_state_job", "run_board_analog_job",
                 "run_trailing_cutoff_job"):
        monkeypatch.setattr(job, name, record(name))
    monkeypatch.setattr(job, "build_dataset", lambda recipe, *, pairs_path=None: "dataset")
    monkeypatch.setattr(job, "_replay_strategies", lambda: (["STR-THRU"], 10))

    cases = {
        "recipe": ("run_training_job", {"recipe": "size:*:champion"}),
        "state": ("run_state_job", {"state": "driver_residual_pool:size"}),
        "board_analog": ("run_board_analog_job", {"alpha": 0.5, "cutoffs": ["2026-09-18"]}),
        "trailing_cutoff": ("run_trailing_cutoff_job", {"cutoffs": ["2026-01-05"]}),
    }
    base = {"expected_ids": ["training"], "recipe": "", "state": "", "alpha": None,
            "cutoffs": [], "strategies": [], "pairs_path": "", "ticker_chunk": 1000}
    for mode, (expected, extra) in cases.items():
        calls.clear()
        training.run_training_worker({**base, "mode": mode, **extra}, tmp_path)
        assert calls == [(expected, False)], mode
    assert (tmp_path / "training_result.json").is_file()


def test_board_analog_mode_refuses_over_budget(tmp_path, monkeypatch):
    from tools import phase5_training_job as job

    monkeypatch.setattr(job, "_replay_strategies", lambda: (["STR-THRU"], 10_000_000))

    def reached(*args, **kwargs):
        raise AssertionError("run_board_analog_job was reached")

    monkeypatch.setattr(job, "run_board_analog_job", reached)
    params = {"expected_ids": ["training"], "mode": "board_analog", "alpha": 0.5,
              "cutoffs": ["2026-09-18"], "strategies": [], "pairs_path": "",
              "ticker_chunk": 1000}
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(params, tmp_path)
    assert excinfo.value.code == "RESOURCE_LIMIT_EXCEEDED"


def _install_tiny_state_worker(monkeypatch):
    """Swap the worker subprocess's argv for a stub that binds a tiny
    ``run_state_job`` (the seam ``test_v2_ops_native_refresh_e2e.py`` uses)."""
    stub = (
        "import sys, types\n"
        "from engine.v2.ops import worker\n"
        "tools = types.ModuleType('tools')\n"
        "job = types.ModuleType('tools.phase5_training_job')\n"
        "def tiny_state(member, out, *, plan_only=False, **kwargs):\n"
        "    out.mkdir(parents=True, exist_ok=True)\n"
        "    (out / 'state.json').write_text('{\"state\": \"' + member + '\"}')\n"
        "    return {'state': member, 'plan_only': plan_only, 'status': 'written'}\n"
        "job.run_state_job = tiny_state\n"
        "tools.phase5_training_job = job\n"
        "sys.modules['tools'] = tools\n"
        "sys.modules['tools.phase5_training_job'] = job\n"
        "raise SystemExit(worker.main())\n"
    )
    real = subprocess.Popen

    def popen(args, **kwargs):
        if list(args[-2:]) == ["-m", "engine.v2.ops.worker"]:
            args = [sys.executable, "-u", "-c", stub]
        return real(args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "Popen", popen)


def test_e2e_submit_training_through_supervisor(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    try:
        _install_tiny_state_worker(monkeypatch)
        # The kind leases legacy_store/read but carries no pinned
        # ``legacy_manifest.json`` binding (its parameters are scalar only),
        # so bypass the launch-time read-set barrier the same way
        # ``test_v2_ops_snapshot_stages.py`` does; claim-time leasing and the
        # worker dispatch stay real.
        monkeypatch.setattr(Service, "_pin_read_set", lambda self, claim: None)
        plan = training.training_plan(mode="state", state="driver_residual_pool:size")
        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "training-e2e"), clock=clock)
        service = Service(conn, tmp_path, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO)
        try:
            service.start()
            state = run_until(service, conn, receipt.job_id, timeout=90)
            if state != "succeeded":
                failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id = ?",
                                       (receipt.job_id,)).fetchone()[0]
                attempt = conn.execute(
                    "SELECT attempt_id FROM attempts WHERE job_id = ? "
                    "ORDER BY attempt_number DESC LIMIT 1", (receipt.job_id,)).fetchone()
                diagnostic = service.store.staging_dir(attempt["attempt_id"]) / "diagnostics"
                stderr = (diagnostic / "worker.stderr")
                raise AssertionError(f"training attempt did not succeed: {failure}\n"
                                     f"{stderr.read_text() if stderr.is_file() else '(no stderr)'}")
        finally:
            service.close()
        rows = conn.execute(
            "SELECT ao.name, ao.artifact_id FROM attempt_outputs ao JOIN attempts a "
            "ON a.attempt_id = ao.attempt_id WHERE a.job_id = ?", (receipt.job_id,)).fetchall()
        names = sorted(item["name"] for item in rows)
        assert "training_result" in names
        assert "training/state.json" in names
        result_id = next(item["artifact_id"] for item in rows
                         if item["name"] == "training_result")
        document = json.loads(store.read_verified(artifact(conn, store, result_id)))
        assert document["state"] == "driver_residual_pool:size"
    finally:
        conn.close()


def test_cli_plan_and_submit_training_never_runs_training_inline(tmp_path, capsys, monkeypatch):
    from tools import phase5_training_job as job

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not run during plan or submit")

    monkeypatch.setattr(job, "run_training_job", forbidden)
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "training",
                     "--training-mode", "state",
                     "--state", "driver_residual_pool:size"]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "cli-training-1"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["kind"] == "training"
    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        row = conn.execute("SELECT state FROM jobs WHERE job_id=?",
                           (receipt["job_id"],)).fetchone()
        assert row is not None
        assert row["state"] == "queued"
    finally:
        conn.close()
