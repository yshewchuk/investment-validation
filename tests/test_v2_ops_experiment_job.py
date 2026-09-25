"""P6 slice 10: smoke-mode experiments run as supervised v2 jobs."""
import csv
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops import cli, effects_graph, experiments, stages, supervisor, worker
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import experiment_plan
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, catalog, run_until

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _spec_document(**changes):
    document = {"experiment_id": "x", "hypothesis": "plumbing", "primary_arm_id": "fixture",
                "arms": ["fixture"], "seed": 7, "folds": ["fold-1"],
                "economic_params": {"fill": "mid"}, "price_source": "synthetic",
                "runner": "synthetic"}
    document.update(changes)
    return document


def test_experiment_kind_is_registered_with_experiment_heavy_profile():
    kind = stages.registry().get("experiment")
    assert kind.resource_classes == frozenset({"experiment_heavy"})
    assert kind.checkpoint_contract == "experiment_receipt.v1.0"
    with pytest.raises(OpsError, match="unsupported job kind"):
        stages.registry().get("bogus_experiment_kind")


def test_experiment_worker_runs_synthetic_runner_and_writes_receipt(tmp_path):
    (tmp_path / "spec.json").write_text(json.dumps(_spec_document()))
    result = worker.dispatch("experiment", {"expected_ids": ["experiment:x"],
                                            "runner": "synthetic", "no_ledger": True}, tmp_path)
    assert result["completed_ids"] == ["experiment:x"]
    assert result["outputs"][0]["name"] == "experiment_receipt"
    assert (tmp_path / "experiment_receipt.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()

    broken = _spec_document()
    del broken["hypothesis"]
    (tmp_path / "spec.json").write_text(json.dumps(broken))
    with pytest.raises(OpsError, match="required field"):
        worker.dispatch("experiment", {"expected_ids": ["experiment:x"],
                                       "runner": "synthetic", "no_ledger": True}, tmp_path)


def _planned_ledger(path, experiment_id="x", stage="planned"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted\n"
                    f"{experiment_id},deadbeef,2026-01-01,{stage},,,False\n")


def _ledger_rows(path):
    with open(path, newline="") as fh:
        return [{"id": row["id"], "stage": row["stage"]} for row in csv.DictReader(fh)]


def test_activation_refuses_unregistered_experiment_ledger_untouched(tmp_path):
    conn, _, _ = catalog(tmp_path)
    try:
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(_spec_document()))

        with pytest.raises(OpsError) as excinfo:
            experiment_plan(spec_path, smoke=False, root=tmp_path)
        assert excinfo.value.code == "INVALID_REQUEST"
        assert not (experiments_dir / "LEDGER.csv").exists()
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0

        ledger = experiments_dir / "LEDGER.csv"
        _planned_ledger(ledger, experiment_id="EXP-OTHER")
        before = ledger.read_bytes()
        with pytest.raises(OpsError):
            experiment_plan(spec_path, smoke=False, root=tmp_path)
        assert ledger.read_bytes() == before
    finally:
        conn.close()


def test_activation_succeeds_with_planned_row(tmp_path):
    _planned_ledger(tmp_path / "experiments" / "LEDGER.csv")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))

    plan = experiment_plan(spec_path, smoke=False, root=tmp_path)
    assert plan["mode"] == "primary"
    assert plan["parameters"]["no_ledger"] is False
    assert plan["parameters"]["runner"] == "synthetic"
    assert plan["parameters"]["expected_ids"] == ["experiment:x"]


def test_cli_activate_ledger_and_no_ledger_are_mutually_exclusive(tmp_path, capsys):
    assert cli.main(["--root", str(tmp_path), "init"]) == 0
    capsys.readouterr()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))

    assert cli.main(["--root", str(tmp_path), "plan", "experiment",
                     "--spec", str(spec_path), "--no-ledger", "--activate-ledger"]) == 2
    problem = json.loads(capsys.readouterr().out)
    assert problem["code"] == "INVALID_REQUEST"


def test_cli_experiment_plan_without_either_flag_still_refuses(tmp_path, capsys):
    """Slice 10 callers who omit both flags keep their old refusal; the real
    activation path is the explicit --activate-ledger opt-in only."""
    assert cli.main(["--root", str(tmp_path), "init"]) == 0
    capsys.readouterr()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))

    assert cli.main(["--root", str(tmp_path), "plan", "experiment",
                     "--spec", str(spec_path)]) == 2
    problem = json.loads(capsys.readouterr().out)
    assert problem["code"] == "INVALID_REQUEST"
    assert "production experiment activation is disabled" in problem["message"]


def test_worker_dispatch_primary_mode_never_grants_runner_ledger_writes(tmp_path,
                                                                        monkeypatch):
    runner_id = "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py"
    runner_path = tmp_path / runner_id
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("if __name__ == '__main__':\n    pass\n")
    (runner_path.parent / "spec.yaml").write_text("id: EXP-182\n")
    (tmp_path / "spec.json").write_text(json.dumps(_spec_document(runner=runner_id)))

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (tmp_path / "REPORT.md").write_text(
            "# Synthetic infrastructure report\n\n*Generated by engine.report v1.0.*\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = worker.dispatch("experiment",
                             {"expected_ids": ["experiment:x"], "runner": runner_id,
                              "no_ledger": False}, tmp_path)
    assert result["completed_ids"] == ["experiment:x"]
    assert len(commands) == 1
    assert "--no-ledger" in commands[0]


def test_experiment_effect_appends_ledger_row_once_on_retry(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    calls = []
    real_effect = effects_graph.experiment_effect

    def spy(conn, store, claim, refs, *, clock):
        calls.append((conn, store, claim, refs, clock))
        return real_effect(conn, store, claim, refs, clock=clock)

    monkeypatch.setattr(supervisor, "experiment_effect", spy)
    try:
        job = _submit_experiment(conn, tmp_path, clock, "primary-1", no_ledger=False)
        service = Service(conn, tmp_path, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO)
        try:
            service.start()
            assert run_until(service, conn, job.job_id, timeout=90) == "succeeded"
        finally:
            service.close()
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
        ledger = tmp_path / "experiments" / "LEDGER.csv"
        assert _ledger_rows(ledger) == [{"id": "x", "stage": "ran"}]

        conn_, store, claim, refs, clock_ = calls[-1]
        real_effect(conn_, store, claim, refs, clock=clock_)
        real_effect(conn_, store, claim, refs, clock=clock_)
        assert _ledger_rows(ledger) == [{"id": "x", "stage": "ran"}]
    finally:
        conn.close()


def test_experiment_plan_names_a_runner_and_experiment_heavy_profile(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))
    plan = experiment_plan(spec_path, smoke=True)
    assert plan["kind"] == "experiment"
    assert plan["resource_class"] == "experiment_heavy"
    assert plan["effects"] == ["staged"]
    assert plan["parameters"]["runner"] == "synthetic"
    assert plan["parameters"]["no_ledger"] is True

    broken = _spec_document()
    del broken["runner"]
    spec_path.write_text(json.dumps(broken))
    with pytest.raises(OpsError, match="runner"):
        experiment_plan(spec_path, smoke=True)


def _submit_experiment(conn, root, clock, key, *, document=None, no_ledger=True):
    document = document or _spec_document()
    store = ArtifactStore(root)
    spec_ref = store.publish_bytes(json.dumps(document, sort_keys=True).encode(),
                                   schema_ref="experiment_spec.v1.0")
    with transaction(conn):
        register_artifact(conn, spec_ref, None, clock)
    profile = profile_named(DEFAULT_POLICY, "experiment_heavy")
    job = JobSpec(
        kind="experiment",
        implementation_ref=content_hash(worker_source_manifest(REPO)),
        spec_hash=content_hash(document),
        environment_ref=content_hash(
            environment_identity(profile.thread_count or profile.cpu_count)),
        parameters={"expected_ids": ["experiment:x"],
                    "input_bindings": {"spec.json": spec_ref.artifact_id},
                    "runner": "synthetic", "no_ledger": no_ledger},
        input_refs=(spec_ref.artifact_id,), output_namespace="shadow",
        resource_class="experiment_heavy", retry_policy_ref="bounded",
        checkpoint_contract_ref="experiment_receipt.v1.0")
    return submit(conn, stages.registry(), POLICY,
                  SubmitRequest(namespace="shadow", idempotency_key=key,
                                principal="operator", job=job), clock=clock)


def test_experiment_effect_registers_a_durable_run(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    try:
        first = _submit_experiment(conn, tmp_path, clock, "experiment-1")
        service = Service(conn, tmp_path, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO)
        try:
            service.start()
            assert run_until(service, conn, first.job_id, timeout=90) == "succeeded"
            attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                      (first.job_id,)).fetchone()[0]
            assert conn.execute("SELECT COUNT(*) FROM experiment_runs WHERE run_id=?",
                                (attempt_id,)).fetchone()[0] == 1
            # Smoke mode holds no promotion authority: no hypotheses row.
            assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0

            second = _submit_experiment(conn, tmp_path, clock, "experiment-2")
            assert run_until(service, conn, second.job_id, timeout=90) == "succeeded"
        finally:
            service.close()
        # Same spec_hash/input_hash: register_hypothesis' early return keeps
        # exactly one durable run row.
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
    finally:
        conn.close()


def _cli_plan_and_submit(root, spec_path, capsys, key="cli-experiment-1"):
    """The operator's real path: ``ops plan experiment`` then ``ops submit``."""
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "experiment",
                     "--spec", str(spec_path), "--no-ledger"]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", key]) == 0
    return json.loads(capsys.readouterr().out)["job_id"]


def test_cli_experiment_plan_and_submit_run_under_the_planned_profile(tmp_path, capsys):
    """End-to-end through the real CLI: the plan's ``environment_ref`` must
    match what ``Service._launch`` fingerprints for the ``experiment_heavy``
    profile the job actually runs under (the bug this pins: the plan used the
    hard-coded 1-thread identity and every CLI submission was refused)."""
    root = tmp_path / "ops"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))
    job_id = _cli_plan_and_submit(root, spec_path, capsys)

    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        service = Service(conn, root, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO)
        try:
            service.start()
            assert run_until(service, conn, job_id, timeout=90) == "succeeded"
        finally:
            service.close()
        row = conn.execute("SELECT state, failure_json FROM attempts WHERE job_id=?",
                           (job_id,)).fetchone()
        assert row["state"] == "succeeded"
        assert row["failure_json"] is None
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
    finally:
        conn.close()


def test_cli_experiment_with_stale_thread_count_is_refused_input_changed(
        tmp_path, capsys, monkeypatch):
    """Negative control: force the pre-fix 1-thread environment into the plan
    and the same CLI flow is refused at launch with INPUT_CHANGED."""
    real_profile_named = experiments.profile_named
    monkeypatch.setattr(
        experiments, "profile_named",
        lambda policy, name: replace(real_profile_named(policy, name), thread_count=1))

    root = tmp_path / "ops"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))
    job_id = _cli_plan_and_submit(root, spec_path, capsys, key="cli-experiment-stale")

    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        service = Service(conn, root, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO)
        try:
            service.start()
            service.tick()
        finally:
            service.close()
        failure = json.loads(conn.execute("SELECT failure_json FROM attempts WHERE job_id=?",
                                          (job_id,)).fetchone()[0])
        assert failure["code"] == "INPUT_CHANGED"
        assert conn.execute("SELECT state FROM jobs WHERE job_id=?",
                            (job_id,)).fetchone()[0] != "succeeded"
    finally:
        conn.close()
