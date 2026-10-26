import json
from types import SimpleNamespace

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.contracts import ArtifactMember, ModelBinding, ModelRelease
from engine.v2.models.release_layout import (
    PHASE5_RELEASE_SCHEMA,
    catalog_path,
)
from engine.v2.models.release_objects import write_object
from engine.v2.ops import retrain_check
from engine.v2.ops.errors import OpsError
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import RetryPolicy


def _catalog(release_root, release_id):
    path = catalog_path(release_root, release_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": PHASE5_RELEASE_SCHEMA, "release_id": release_id,
            "deployment_id": "deployment-test", "members": [], "sources": {}}
    body["manifest_hash"] = content_hash(body)
    path.write_text(json.dumps(body), encoding="utf-8")
    return body


def _acceptance(root, release_id, catalog, *, release_ok=True):
    evidence = {"schema_version": retrain_check.ACCEPTANCE_SCHEMA,
                "release_id": release_id, "manifest_hash": catalog["manifest_hash"],
                "release_ok": release_ok,
                "status": "RELEASE_PASS" if release_ok else "FAIL",
                "findings": [] if release_ok else [{"code": "P5_MEMBER_MISSING"}]}
    evidence["evidence_hash"] = content_hash(evidence)
    (root / retrain_check.ACCEPTANCE_FILE).write_text(
        json.dumps(evidence), encoding="utf-8")
    return evidence


def _model_release(release_root, release_id, *, payload=b"fitted-model", model_id="m1"):
    digest, path = write_object(release_root, payload)
    binding = ModelBinding(
        binding_id="binding-1", model_id=model_id, role="size", strategy_id="*",
        decision_clock_id="close", adapter="json_linear", feature_order=("x",),
        output_names=("y",), members=(ArtifactMember(
            name="estimator", path=path, content_hash=digest),))
    release = ModelRelease(release_id=release_id, deployment_id="deployment-test",
                           bindings=(binding,))
    return SimpleNamespace(
        release=release,
        release_hash=retrain_check.deployment.semantic_release_hash(release),
        release_hash_version="semantic_manifest.v2",
    )


def _run(tmp_path, monkeypatch, *, candidate_model_id="m1", candidate_payload=b"fitted-model",
         acceptance_ok=True):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    candidate_id, incumbent_id = "candidate", "incumbent"
    catalog = _catalog(release_root, candidate_id)
    evidence = _acceptance(attempt, candidate_id, catalog, release_ok=acceptance_ok)
    manifests = {
        candidate_id: _model_release(release_root, candidate_id, payload=candidate_payload,
                                     model_id=candidate_model_id),
        incumbent_id: _model_release(release_root, incumbent_id),
    }
    monkeypatch.setattr(retrain_check.deployment, "read_manifest",
                        lambda _root, release_id: manifests.get(release_id))
    params = {"expected_ids": ("retrain_check",), "release_root": str(release_root),
              "candidate_release_id": candidate_id, "incumbent_release_id": incumbent_id,
              "input_bindings": {"acceptance.json": "acceptance-job"}}
    return attempt, params, evidence


def test_pool_only_check_binds_acceptance_and_model_hashes_repeatably(tmp_path, monkeypatch):
    attempt, params, evidence = _run(tmp_path, monkeypatch)

    first = retrain_check.run_retrain_check_worker(params, attempt)
    first_bytes = (attempt / "retrain_check_receipt.json").read_bytes()
    second = retrain_check.run_retrain_check_worker(params, attempt)

    assert second["receipt"] == first["receipt"]
    assert first["receipt"]["acceptance_evidence_hash"] == evidence["evidence_hash"]
    assert first["receipt"]["candidate_release_hash"] != first["receipt"]["incumbent_release_hash"]
    assert first["receipt"]["check_candidate_release_hash"] == first["receipt"]["candidate_release_hash"]
    assert first["receipt"]["result"] == "pool_only_noop"
    assert first_bytes == (attempt / "retrain_check_receipt.json").read_bytes()
    assert not (tmp_path / "store" / "deployment" / "DEPLOYED").exists()


def test_model_change_refuses_with_open_rule_code_and_no_success_receipt(tmp_path, monkeypatch):
    attempt, params, _ = _run(tmp_path, monkeypatch, candidate_model_id="m2")

    with pytest.raises(OpsError) as excinfo:
        retrain_check.run_retrain_check_worker(params, attempt)

    assert excinfo.value.code == "ACCEPTANCE_RULE_UNDEFINED"
    assert not (attempt / "retrain_check_receipt.json").exists()


def test_failed_pool_acceptance_refuses_before_receipt(tmp_path, monkeypatch):
    attempt, params, _ = _run(tmp_path, monkeypatch, acceptance_ok=False)

    with pytest.raises(OpsError) as excinfo:
        retrain_check.run_retrain_check_worker(params, attempt)

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["code"] == "POOL_ACCEPTANCE_FAILED"
    assert not (attempt / "retrain_check_receipt.json").exists()


def test_acceptance_evidence_drift_refuses_before_receipt(tmp_path, monkeypatch):
    attempt, params, _ = _run(tmp_path, monkeypatch)
    evidence = json.loads((attempt / retrain_check.ACCEPTANCE_FILE).read_text())
    evidence["manifest_hash"] = "sha256:stale"
    (attempt / retrain_check.ACCEPTANCE_FILE).write_text(json.dumps(evidence))

    with pytest.raises(OpsError) as excinfo:
        retrain_check.run_retrain_check_worker(params, attempt)

    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.details["code"] == "POOL_ACCEPTANCE_MISMATCH"
    assert not (attempt / "retrain_check_receipt.json").exists()


def test_object_drift_refuses_before_receipt(tmp_path, monkeypatch):
    attempt, params, _ = _run(tmp_path, monkeypatch)
    candidate_manifest = retrain_check.deployment.read_manifest(
        tmp_path / "store" / "deployment", "candidate")
    object_path = tmp_path / "store" / "deployment" / candidate_manifest.release.bindings[0].members[0].path
    object_path.write_bytes(b"changed-after-acceptance")

    with pytest.raises(OpsError) as excinfo:
        retrain_check.run_retrain_check_worker(params, attempt)

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["code"] == "MODEL_OBJECT_HASH_MISMATCH"
    assert not (attempt / "retrain_check_receipt.json").exists()


def test_staged_manifest_hash_drift_refuses_before_receipt(tmp_path, monkeypatch):
    attempt, params, _ = _run(tmp_path, monkeypatch)
    candidate = retrain_check.deployment.read_manifest(
        tmp_path / "store" / "deployment", "candidate")
    candidate.release_hash = "sha256:stale"

    with pytest.raises(OpsError) as excinfo:
        retrain_check.run_retrain_check_worker(params, attempt)

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["code"] == "STAGED_MANIFEST_HASH_MISMATCH"
    assert not (attempt / "retrain_check_receipt.json").exists()


@pytest.mark.parametrize("missing_id", ["candidate", "incumbent"])
def test_missing_staged_manifest_refuses_before_receipt(tmp_path, monkeypatch, missing_id):
    attempt, params, _ = _run(tmp_path, monkeypatch)
    read_manifest = retrain_check.deployment.read_manifest

    def read_manifest_with_missing(_root, release_id):
        if release_id == missing_id:
            return None
        return read_manifest(_root, release_id)

    monkeypatch.setattr(retrain_check.deployment, "read_manifest",
                        read_manifest_with_missing)
    with pytest.raises(OpsError) as excinfo:
        retrain_check.run_retrain_check_worker(params, attempt)

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["code"] == "RELEASE_NOT_STAGED"
    assert not (attempt / "retrain_check_receipt.json").exists()


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        {"expected_ids": None, "release_root": "/release",
         "candidate_release_id": "candidate", "incumbent_release_id": "incumbent",
         "input_bindings": {"acceptance.json": "acceptance-job"}},
        retrain_check.RetrainCheckParameters(
            expected_ids=None, release_root="/release",
            candidate_release_id="candidate", incumbent_release_id="incumbent",
            input_bindings={"acceptance.json": "acceptance-job"}),
        {"expected_ids": ["retrain_check"], "release_root": "/release",
         "candidate_release_id": "same", "incumbent_release_id": "same",
         "input_bindings": {"acceptance.json": "acceptance-job"}},
        {"expected_ids": ["retrain_check"], "release_root": "/release",
         "candidate_release_id": "candidate", "incumbent_release_id": "incumbent"},
        retrain_check.RetrainCheckParameters(
            expected_ids=("retrain_check",), release_root="/release",
            candidate_release_id="candidate", incumbent_release_id="incumbent",
            input_bindings={"acceptance.json": None}),
        retrain_check.RetrainCheckParameters(
            expected_ids=("retrain_check",), release_root="/release",
            candidate_release_id="candidate", incumbent_release_id="incumbent",
            input_bindings={"acceptance.json": "acceptance-job", "extra.json": 1}),
    ],
)
def test_worker_refuses_invalid_parameters_before_reads_or_receipt(tmp_path, parameters):
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    with pytest.raises(OpsError) as excinfo:
        retrain_check.run_retrain_check_worker(parameters, attempt)

    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.details["code"] == "RETRAIN_CHECK_PARAMETERS_INVALID"
    assert not (attempt / "retrain_check_receipt.json").exists()


@pytest.mark.parametrize("wrong_side", ["candidate", "incumbent"])
def test_manifest_release_id_mismatch_refuses_before_receipt(tmp_path, monkeypatch, wrong_side):
    attempt, params, _ = _run(tmp_path, monkeypatch)
    store = tmp_path / "store" / "deployment"
    manifests = {
        "candidate": retrain_check.deployment.read_manifest(store, "candidate"),
        "incumbent": retrain_check.deployment.read_manifest(store, "incumbent"),
    }

    def read_manifest(_root, release_id):
        if wrong_side == "candidate" and release_id == "candidate":
            return manifests["incumbent"]
        if wrong_side == "incumbent" and release_id == "incumbent":
            return manifests["candidate"]
        return manifests[release_id]

    monkeypatch.setattr(retrain_check.deployment, "read_manifest", read_manifest)
    with pytest.raises(OpsError) as excinfo:
        retrain_check.run_retrain_check_worker(params, attempt)

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["code"] == "STAGED_RELEASE_ID_MISMATCH"
    assert not (attempt / "retrain_check_receipt.json").exists()


def test_symlinked_model_object_refuses_before_receipt(tmp_path, monkeypatch):
    attempt, params, _ = _run(tmp_path, monkeypatch)
    deployment_root = tmp_path / "store" / "deployment"
    candidate = retrain_check.deployment.read_manifest(deployment_root, "candidate")
    member_path = deployment_root / candidate.release.bindings[0].members[0].path
    target_path = member_path.with_name("same-bytes-through-symlink")
    target_path.write_bytes(member_path.read_bytes())
    member_path.unlink()
    member_path.symlink_to(target_path)

    with pytest.raises(OpsError) as excinfo:
        retrain_check.run_retrain_check_worker(params, attempt)

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["code"] == "MODEL_OBJECT_INVALID"
    assert not (attempt / "retrain_check_receipt.json").exists()


def test_job_kind_is_registered_and_non_retrying():
    kind = registry().get("retrain_check")
    params = retrain_check.RetrainCheckParameters(
        expected_ids=("retrain_check",), release_root="/release-store",
        candidate_release_id="candidate", incumbent_release_id="incumbent",
        input_bindings={"acceptance.json": "acceptance-job"})

    assert kind.worker == "retrain_check"
    assert kind.retry == RetryPolicy("none", 1)
    assert kind.validate(None, params) == []
