import json
from types import SimpleNamespace

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.release_layout import PHASE5_RELEASE_SCHEMA, catalog_path
from engine.v2.ops import retrain_check as phase5_accept
from engine.v2.ops.errors import OpsError
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import RetryPolicy
from tools import phase5_acceptance_job


def _catalog(release_root, release_id):
    path = catalog_path(release_root, release_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": PHASE5_RELEASE_SCHEMA, "release_id": release_id,
            "deployment_id": "deployment-test", "members": [], "sources": {}}
    body["manifest_hash"] = content_hash(body)
    path.write_text(json.dumps(body), encoding="utf-8")
    return body


def _evidence(release_id, catalog, *, release_ok=True):
    evidence = {"schema_version": phase5_accept.ACCEPTANCE_SCHEMA,
                "release_id": release_id, "manifest_hash": catalog["manifest_hash"],
                "release_ok": release_ok,
                "status": "RELEASE_PASS" if release_ok else "FAIL",
                "findings": [] if release_ok else [{"code": "P5_MEMBER_MISSING"}]}
    evidence["evidence_hash"] = content_hash(evidence)
    return evidence


def _manifest(release_id):
    return SimpleNamespace(release=SimpleNamespace(release_id=release_id))


def _params(release_root, candidate="candidate"):
    return {"expected_ids": ("phase5_accept",), "release_root": str(release_root),
            "candidate_release_id": candidate}


def _patch_manifest(monkeypatch, manifest):
    monkeypatch.setattr(phase5_accept.deployment, "read_manifest",
                        lambda _root, _release_id: manifest)


def _patch_evidence(monkeypatch, evidence):
    monkeypatch.setattr(phase5_acceptance_job, "build_evidence",
                        lambda _release_root: evidence)


def test_success_writes_acceptance_and_returns_output(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    catalog = _catalog(release_root, "candidate")
    evidence = _evidence("candidate", catalog)
    _patch_manifest(monkeypatch, _manifest("candidate"))
    _patch_evidence(monkeypatch, evidence)

    result = phase5_accept.run_phase5_accept_worker(_params(release_root), attempt)

    assert result["outputs"] == [{"name": "acceptance", "path": "acceptance.json",
                                  "schema": "phase5_acceptance.v1.0"}]
    assert result["completed_ids"] == ("phase5_accept",)
    assert json.loads((attempt / "acceptance.json").read_text()) == evidence
    assert phase5_accept._acceptance_evidence(attempt, release_root, "candidate") == evidence


def test_failed_evidence_is_written_then_refused(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    catalog = _catalog(release_root, "candidate")
    evidence = _evidence("candidate", catalog, release_ok=False)
    _patch_manifest(monkeypatch, _manifest("candidate"))
    _patch_evidence(monkeypatch, evidence)

    with pytest.raises(OpsError) as excinfo:
        phase5_accept.run_phase5_accept_worker(_params(release_root), attempt)

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["code"] == "PHASE5_ACCEPTANCE_FAILED"
    assert json.loads((attempt / "acceptance.json").read_text()) == evidence


@pytest.mark.parametrize("manifest", [None, _manifest("other")])
def test_missing_or_mismatched_manifest_refuses_before_shim(tmp_path, monkeypatch, manifest):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    _patch_manifest(monkeypatch, manifest)
    called = []
    monkeypatch.setattr(phase5_acceptance_job, "run_acceptance",
                        lambda *_args: called.append(True))

    with pytest.raises(OpsError) as excinfo:
        phase5_accept.run_phase5_accept_worker(_params(release_root), attempt)

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["code"] == "RELEASE_NOT_STAGED"
    assert called == []
    assert not (attempt / "acceptance.json").exists()


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        {"expected_ids": None, "release_root": "/release",
         "candidate_release_id": "candidate"},
        {"expected_ids": ("retrain_check",), "release_root": "/release",
         "candidate_release_id": "candidate"},
        {"expected_ids": ("phase5_accept",), "release_root": "  ",
         "candidate_release_id": "candidate"},
        {"expected_ids": ("phase5_accept",), "release_root": "/release",
         "candidate_release_id": "../escape"},
        phase5_accept.Phase5AcceptParameters(
            expected_ids=("phase5_accept",), release_root="/release",
            candidate_release_id=""),
    ],
)
def test_invalid_parameters_refuse_before_reads(tmp_path, parameters):
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    with pytest.raises(OpsError) as excinfo:
        phase5_accept.run_phase5_accept_worker(parameters, attempt)

    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.details["code"] == "PHASE5_ACCEPT_PARAMETERS_INVALID"
    assert not (attempt / "acceptance.json").exists()


def test_changed_evidence_identity_fails_closed(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    catalog = _catalog(release_root, "candidate")
    evidence = _evidence("other", catalog)
    _patch_manifest(monkeypatch, _manifest("candidate"))
    _patch_evidence(monkeypatch, evidence)

    with pytest.raises(OpsError) as excinfo:
        phase5_accept.run_phase5_accept_worker(_params(release_root), attempt)

    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.details["code"] == "PHASE5_ACCEPTANCE_MISMATCH"
    assert (attempt / "acceptance.json").exists()


def test_job_kind_is_registered_non_retrying_and_primary():
    kind = registry().get("phase5_accept")
    params = phase5_accept.Phase5AcceptParameters(
        expected_ids=("phase5_accept",), release_root="/release-store",
        candidate_release_id="candidate")

    assert kind.worker == "phase5_accept"
    assert kind.retry == RetryPolicy("none", 1)
    assert "primary" in kind.namespaces
    assert kind.validate(None, params) == []


def test_worker_leaves_deployed_pointer_unchanged(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    pointer = release_root / "deployment" / "DEPLOYED"
    pointer.parent.mkdir(parents=True)
    pointer.write_bytes(b"sentinel-pointer\n")
    catalog = _catalog(release_root, "candidate")
    evidence = _evidence("candidate", catalog)
    _patch_manifest(monkeypatch, _manifest("candidate"))
    _patch_evidence(monkeypatch, evidence)

    phase5_accept.run_phase5_accept_worker(_params(release_root), attempt)

    assert pointer.read_bytes() == b"sentinel-pointer\n"
