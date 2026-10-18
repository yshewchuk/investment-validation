"""Native D14 source verification through the existing supervised training lifecycle."""
# packages: ops, models/training
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.v2.data.reference_catalog import insert_reference_inputs
from engine.v2.foundation import ArtifactStore
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops import stages, training
from engine.v2.ops.errors import OpsError
from engine.v2.ops.plans import request_from_plan
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.data_scan_support import commit_tables
from tests.ops_support import TEST_POLICY, catalog, run_until
from tests.v2.models.training.d14_support import forbidden
from tests.v2.models.training.d14_support import pinned as pinned
from tools import phase5_d14_inputs

REPO = Path(__file__).resolve().parents[3]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def plan(pinned, role="implied_t1"):
    source = pinned["inputs"]
    return training.training_plan(
        mode="recipe", recipe=f"{role}:*:tier4_monthly", input_mode="native_d14",
        catalog_path=str(source["catalog"]), objects_root=str(source["artifact_root"]),
        snapshot_id=source["snapshot_id"], receipt_id=source["receipt_id"])


@pytest.mark.parametrize("role", ("implied_t1", "runup_move"))
def test_worker_and_standalone_cli_have_same_resume_identity(pinned, role, monkeypatch, tmp_path):
    monkeypatch.setattr("tools.phase5_training_job.build_dataset", forbidden)
    monkeypatch.setattr("engine.v2.models.training.job._fit_fold", forbidden)
    root = tmp_path / "worker"
    root.mkdir()
    parameters = plan(pinned, role)["parameters"]
    training.run_training_worker(parameters, root)
    identity = json.loads((root / "training/job.json").read_text())
    assert identity["input_identity"]["receipt_id"] == "r1"
    assert identity["input_identity"]["snapshot_id"] == pinned["snapshot"].snapshot_id
    args = ["--role", role, "--out", str(tmp_path / "cli")]
    for name, value in pinned["inputs"].items():
        args += ["--" + name.replace("_", "-"), str(value)]
    assert phase5_d14_inputs.main(args) == 0
    assert json.loads((tmp_path / "cli/job.json").read_text()) == identity
    training.run_training_worker(parameters, root)
    summary = json.loads((root / "training_result.json").read_text())
    assert summary["folds"] and set(summary["folds"].values()) == {"resumed"}

    other = commit_tables(pinned["conn"], pinned["clock"], pinned["records"], pinned["contracts"],
                          store=pinned["store"], scope="other", receipt_id="r2", attempt_id="att-2")
    assert other.snapshot_id == pinned["snapshot"].snapshot_id
    insert_reference_inputs(pinned["conn"], "r2", (pinned["calendar_ref"],))
    pinned["conn"].commit()
    with pytest.raises(OpsError) as err:
        training.run_training_worker({**parameters, "receipt_id": "r2"}, root)
    assert err.value.problem.code == "CHECKPOINT_INCOMPATIBLE"
    assert err.value.problem.details == {"issues": ["RESUME_MISMATCH"]}
    assert json.loads((root / "training/job.json").read_text()) == identity


def test_supervisor_ignores_legacy_barrier_and_reverifies_each_submission(pinned, tmp_path):
    root = tmp_path / "ops"
    root.mkdir()
    conn, clock, _ = catalog(root)
    conn.execute("INSERT INTO store_domains VALUES ('legacy_store', 1)")
    conn.commit()
    source_plan = plan(pinned)
    service = Service(conn, root, stages.registry(), TEST_POLICY, clock=clock,
                      code_source=REPO, store_root=root)
    try:
        service.start()
        first = submit(conn, stages.registry(), POLICY,
                       request_from_plan(source_plan, "native-first"), clock=clock)
        assert run_until(service, conn, first.job_id, timeout=90) == "succeeded", conn.execute(
            "SELECT failure_json FROM jobs WHERE job_id=?", (first.job_id,)).fetchone()[0]
        assert conn.execute("SELECT COUNT(*) FROM store_leases").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
        outputs = conn.execute("SELECT name FROM attempt_outputs").fetchall()
        assert any(row[0] == "training/job.json" for row in outputs)

        # Corrupt only the synthetic content-addressed calendar. A second job
        # must verify bytes again instead of restoring the first job's output.
        digest = pinned["calendar_ref"].content_hash.removeprefix("sha256:")
        obj = pinned["store"].root / "objects" / digest[:2] / digest
        obj.chmod(0o600)
        obj.write_bytes(b"corrupt synthetic calendar")
        second = submit(conn, stages.registry(), POLICY,
                        request_from_plan(source_plan, "native-second"), clock=clock)
        assert run_until(service, conn, second.job_id, timeout=90) == "failed"
        failure = json.loads(conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                                          (second.job_id,)).fetchone()[0])
        assert failure["code"] == "INPUT_CHANGED"
        assert failure["retryable"] is False
        store = ArtifactStore(root)
        details = json.loads(store.read_verified(artifact(conn, store, failure["diagnostic_ref"])))
        assert details == {"data_code": "OBJECT_CORRUPT"}
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM store_leases").fetchone()[0] == 0
    finally:
        service.close()
        conn.close()
