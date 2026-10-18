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
# Independent oracle from the five event months in the synthetic input fixture.
FOLDS = ("m-2024-01", "m-2024-02", "m-2024-07", "m-2024-12", "m-2025-01")
FOLD_FILES = ("COMPLETE.json", "label_availability_receipt.json", "membership_receipt.json")
TRAINING_FILES = ("job.json", "summary.json", *(
    f"folds/{fold}/{name}" for fold in FOLDS for name in FOLD_FILES))


def expected_outputs():
    return [{"name": "training_result", "path": "training_result.json", "schema": "training_job_result.v1.0"}, *[
        {"name": "training/" + rel.replace("/", "__"), "path": "training/" + rel,
         "schema": "training_job_output.v1.0"} for rel in sorted(TRAINING_FILES)]]



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
    result = training.run_training_worker(parameters, root)
    assert result == {"completed_ids": ["training"], "no_work": False, "outputs": expected_outputs()}
    assert json.loads((root / "training_result.json").read_text()) == {
        "mode": "recipe", "recipe_id": f"{role}.tier4_monthly.v1",
        "folds": dict.fromkeys(FOLDS, "skipped")}
    identity = json.loads((root / "training/job.json").read_text())
    assert identity["folds"] == list(FOLDS)
    assert identity["input_identity"]["receipt_id"] == "r1"
    assert identity["input_identity"]["snapshot_id"] == pinned["snapshot"].snapshot_id
    args = ["--role", role, "--out", str(tmp_path / "cli")]
    for name, value in pinned["inputs"].items():
        args += ["--" + name.replace("_", "-"), str(value)]
    assert phase5_d14_inputs.main(args) == 0
    assert json.loads((tmp_path / "cli/job.json").read_text()) == identity
    assert {p.relative_to(root / "training").as_posix() for p in (root / "training").rglob("*") if p.is_file()} == set(TRAINING_FILES)
    for rel in TRAINING_FILES:
        assert (root / "training" / rel).read_bytes() == (tmp_path / "cli" / rel).read_bytes()
    assert training.run_training_worker(parameters, root) == result
    summary = json.loads((root / "training_result.json").read_text())
    assert summary == {"mode": "recipe", "recipe_id": f"{role}.tier4_monthly.v1",
                       "folds": dict.fromkeys(FOLDS, "resumed")}

    other = commit_tables(pinned["conn"], pinned["clock"], pinned["records"], pinned["contracts"],
                          store=pinned["store"], scope="other", receipt_id="r2", attempt_id="att-2")
    assert other.snapshot_id == pinned["snapshot"].snapshot_id
    insert_reference_inputs(pinned["conn"], "r2", (pinned["calendar_ref"],))
    pinned["conn"].commit()
    with pytest.raises(OpsError) as err:
        training.run_training_worker({**parameters, "receipt_id": "r2"}, root)
    assert err.value.problem.code == "CHECKPOINT_INCOMPATIBLE"
    assert err.value.problem.message == "training receipt refused"
    assert err.value.problem.retryable is False
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
        outputs = dict(conn.execute("SELECT name, artifact_id FROM attempt_outputs").fetchall())
        assert set(outputs) == {entry["name"] for entry in expected_outputs()}
        store = ArtifactStore(root)
        published = {}
        for entry in expected_outputs():
            ref = artifact(conn, store, outputs[entry["name"]])
            assert ref.schema_ref == entry["schema"]
            published[entry["path"]] = json.loads(store.read_verified(ref))
        assert published["training/job.json"]["folds"] == list(FOLDS)
        assert published["training_result.json"] == {
            "mode": "recipe", "recipe_id": "implied_t1.tier4_monthly.v1",
            "folds": dict.fromkeys(FOLDS, "skipped")}
        assert published["training/summary.json"] == {
            "recipe_id": "implied_t1.tier4_monthly.v1",
            "recipe_fingerprint": published["training/job.json"]["recipe_fingerprint"],
            "folds": dict.fromkeys(FOLDS, "skipped")}
        for fold in FOLDS:
            for name in FOLD_FILES:
                assert published[f"training/folds/{fold}/{name}"]["fold_id"] == fold
            assert published[f"training/folds/{fold}/COMPLETE.json"]["status"] == "skipped"

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
        assert failure["message"] == "native D14 inputs refused"
        assert failure["details"] == {}
        assert failure["retryable"] is False
        store = ArtifactStore(root)
        details = json.loads(store.read_verified(artifact(conn, store, failure["diagnostic_ref"])))
        assert details == {"data_code": "OBJECT_CORRUPT"}
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM store_leases").fetchone()[0] == 0
    finally:
        service.close()
        conn.close()
