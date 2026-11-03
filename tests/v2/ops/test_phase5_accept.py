import json
from types import SimpleNamespace

import pytest

from checks import phase5_acceptance as acceptance_gate
from engine.v2.foundation import content_hash
from engine.v2.models import frozen_state
from engine.v2.models.release_layout import PHASE5_RELEASE_SCHEMA, StateSpec, catalog_path
from engine.v2.ops import retrain_check as phase5_accept
from engine.v2.ops.errors import OpsError
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import RetryPolicy
from engine.v2.ops.supervisor import Service
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


def _rehash(evidence):
    unhashed = {key: value for key, value in evidence.items()
                if key not in ("runtime_ms", "report", "evidence_hash")}
    evidence["evidence_hash"] = content_hash(unhashed)
    return evidence


def _manifest(release_id):
    return SimpleNamespace(release=SimpleNamespace(release_id=release_id))


def _params(release_root, candidate="candidate"):
    return {"expected_ids": ("phase5_accept",), "release_root": str(release_root),
            "candidate_release_id": candidate}


def _patch_manifest(monkeypatch, manifest, *, hash_matches=True):
    monkeypatch.setattr(phase5_accept.deployment, "read_manifest",
                        lambda _root, _release_id: manifest)
    monkeypatch.setattr(phase5_accept.deployment, "manifest_hash_matches",
                        lambda _manifest: hash_matches)


def _patch_evidence(monkeypatch, evidence):
    def build(_release_root, *, candidate_release_id=None):
        return evidence

    monkeypatch.setattr(phase5_acceptance_job, "build_evidence", build)


def _write_root_selector(release_root, body):
    (release_root / "phase5_release.json").write_text(json.dumps(body), encoding="utf-8")


def _build_for_candidate(release_root, candidate_release_id):
    return acceptance_gate.build_evidence(
        release_root, candidate_release_id=candidate_release_id, state_specs=(),
        consumers={}, watch_dirs=[])


def test_build_evidence_selects_explicit_candidate_over_root_selector(tmp_path):
    release_root = tmp_path / "store"
    candidate = _catalog(release_root, "candidate")
    other = _catalog(release_root, "other")
    _write_root_selector(release_root, other)

    evidence = _build_for_candidate(release_root, "candidate")
    default = acceptance_gate.build_evidence(
        release_root, state_specs=(), consumers={}, watch_dirs=[])

    assert evidence["release_id"] == "candidate"
    assert evidence["manifest_hash"] == candidate["manifest_hash"]
    assert default["release_id"] == other["release_id"]


def test_build_evidence_selects_explicit_candidate_when_root_selector_absent(tmp_path):
    release_root = tmp_path / "store"
    candidate = _catalog(release_root, "candidate")

    evidence = _build_for_candidate(release_root, "candidate")

    assert evidence["release_id"] == "candidate"
    assert evidence["manifest_hash"] == candidate["manifest_hash"]


def test_build_evidence_refuses_missing_candidate_without_root_fallback(tmp_path):
    release_root = tmp_path / "store"
    other = _catalog(release_root, "other")
    _write_root_selector(release_root, other)

    evidence = _build_for_candidate(release_root, "missing")

    assert evidence["status"] == "FAIL"
    assert "release_id" not in evidence
    assert any(row["code"] == acceptance_gate.RELEASE_LAYOUT
               for row in evidence["findings"])


def test_build_evidence_refuses_mismatched_candidate_catalog(tmp_path):
    release_root = tmp_path / "store"
    other = _catalog(release_root, "other")
    path = catalog_path(release_root, "candidate")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(other), encoding="utf-8")

    evidence = _build_for_candidate(release_root, "candidate")

    assert evidence["status"] == "FAIL"
    assert "release_id" not in evidence
    assert any(row["code"] == acceptance_gate.RELEASE_LAYOUT
               for row in evidence["findings"])


def test_worker_threads_candidate_id_to_acceptance_job(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    catalog = _catalog(release_root, "candidate")
    evidence = _evidence("candidate", catalog)
    _patch_manifest(monkeypatch, _manifest("candidate"))
    seen = []

    def build(_release_root, *, candidate_release_id=None):
        seen.append(candidate_release_id)
        return evidence

    monkeypatch.setattr(phase5_acceptance_job, "build_evidence", build)

    phase5_accept.run_phase5_accept_worker(_params(release_root), attempt)

    assert seen == ["candidate"]


def test_verify_object_uses_bounded_reader(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    target = release_root / "deployment" / "objects" / "member"
    target.parent.mkdir(parents=True)
    payload = b"member-bytes"
    target.write_bytes(payload)
    calls = []
    real = acceptance_gate.read_regular_file

    def spy(path, limit):
        calls.append((path, limit))
        return real(path, limit)

    monkeypatch.setattr(acceptance_gate, "read_regular_file", spy)

    assert acceptance_gate._verify_object(
        release_root, "objects/member", acceptance_gate.sha256_bytes(payload)) is None
    assert calls == [(target.resolve(), acceptance_gate.MAX_MEMBER_BYTES)]


def test_typed_load_caps_frozen_state_loader(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    spec = StateSpec(member_id="driver_residual_pool:size", kind="driver_residual_pool",
                     strategies=(), modules=("engine.v2.models.frozen_state",),
                     consumer=None, source="")
    obj = {"path": "objects/member", "content_hash": "sha256:abc"}
    calls = []

    class SpyLoader:
        def __init__(self, root, max_bytes=None):
            calls.append(("init", root, max_bytes))

        def load(self, ref):
            calls.append(("load", ref))
            return "artifact"

    monkeypatch.setattr(frozen_state, "FrozenStateLoader", SpyLoader)

    assert acceptance_gate._typed_load(spec, release_root, obj) == "artifact"
    assert calls[0] == ("init", acceptance_gate.deployment_root(release_root),
                        acceptance_gate.MAX_MEMBER_BYTES)
    assert calls[1][0] == "load"
    assert calls[1][1].path == obj["path"]
    assert calls[1][1].content_hash == obj["content_hash"]


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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda evidence: evidence.update(release_ok=False),
        lambda evidence: evidence.update(status="FAIL"),
        lambda evidence: evidence.update(findings=[{"code": "P5_MEMBER_MISSING"}]),
    ],
)
def test_failed_evidence_is_written_then_refused(tmp_path, monkeypatch, mutate):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    catalog = _catalog(release_root, "candidate")
    evidence = _evidence("candidate", catalog)
    mutate(evidence)
    _rehash(evidence)
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


def test_staged_manifest_hash_mismatch_refuses_before_shim(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    _patch_manifest(monkeypatch, _manifest("candidate"), hash_matches=False)
    called = []
    monkeypatch.setattr(phase5_acceptance_job, "run_acceptance",
                        lambda *_args: called.append(True))

    with pytest.raises(OpsError) as excinfo:
        phase5_accept.run_phase5_accept_worker(_params(release_root), attempt)

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["code"] == "STAGED_MANIFEST_HASH_MISMATCH"
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
        {"expected_ids": ("phase5_accept",), "release_root": "/release",
         "candidate_release_id": ".hidden"},
        {"expected_ids": ("phase5_accept",), "release_root": "/release",
         "candidate_release_id": "a\\b"},
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


def test_catalog_drift_after_evidence_production_is_input_changed(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    catalog = _catalog(release_root, "candidate")
    evidence = _evidence("candidate", catalog)
    _patch_manifest(monkeypatch, _manifest("candidate"))

    def build_then_drift(_release_root, *, candidate_release_id=None):
        drifted = {"schema_version": PHASE5_RELEASE_SCHEMA, "release_id": "candidate",
                   "deployment_id": "deployment-drifted", "members": [], "sources": {}}
        drifted["manifest_hash"] = content_hash(drifted)
        catalog_path(release_root, "candidate").write_text(
            json.dumps(drifted), encoding="utf-8")
        return evidence

    monkeypatch.setattr(phase5_acceptance_job, "build_evidence", build_then_drift)

    with pytest.raises(OpsError) as excinfo:
        phase5_accept.run_phase5_accept_worker(_params(release_root), attempt)

    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.details["code"] == "POOL_ACCEPTANCE_MISMATCH"
    assert (attempt / "acceptance.json").exists()


def test_job_kind_is_registered_non_retrying_with_shadow_and_smoke_namespaces():
    kind = registry().get("phase5_accept")
    params = phase5_accept.Phase5AcceptParameters(
        expected_ids=("phase5_accept",), release_root="/release-store",
        candidate_release_id="candidate")

    assert kind.worker == "phase5_accept"
    assert kind.retry == RetryPolicy("none", 1)
    assert kind.effects == ("staged",)
    assert kind.resource_classes == frozenset({"validation"})
    assert kind.checkpoint_contract == "phase5_acceptance.v1.0"
    assert kind.namespaces == frozenset({"shadow", "smoke"})
    assert kind.validate(None, params) == []


def test_job_kind_does_not_request_deterministic_checkpointing():
    kind = registry().get("phase5_accept")
    service = object.__new__(Service)
    service.registry = registry()
    claim = SimpleNamespace(spec=SimpleNamespace(kind="phase5_accept", parameters={}))
    retrain_claim = SimpleNamespace(spec=SimpleNamespace(kind="retrain_check", parameters={}))

    assert service._cache_allowed(claim) is False
    assert service._cache_allowed(retrain_claim) is True
    assert kind.checkpoint_contract == phase5_accept.ACCEPTANCE_SCHEMA
    assert kind.checkpoint_contract == "phase5_acceptance.v1.0"


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


def test_promote_rehearsal_uses_scratch_copy(tmp_path, monkeypatch):
    release_root = tmp_path / "store"
    store = release_root / "deployment"
    (store / "releases" / "candidate").mkdir(parents=True)
    (store / "history").mkdir()
    pointer = store / "DEPLOYED"
    pointer.write_bytes(b"sentinel-pointer\n")
    roots = []
    copies = []
    real_copy = acceptance_gate._copy_pointer_state

    def copy_spy(source, dest):
        real_copy(source, dest)
        copies.append((source, dest, (dest / "DEPLOYED").read_bytes()))

    monkeypatch.setattr(acceptance_gate, "_copy_pointer_state", copy_spy)
    monkeypatch.setattr(acceptance_gate.deployment, "current_pointer", lambda _root: None)

    def mark(root, _candidate):
        roots.append(("mark", root))

    def promote(root, candidate):
        roots.append(("promote", root))
        (root / "DEPLOYED").write_bytes(candidate.encode())
        return SimpleNamespace(release_id=candidate)

    observed = []

    def current_release(root):
        observed.append((root, (root / "DEPLOYED").read_bytes()))
        return {"release_id": (root / "DEPLOYED").read_text()}

    monkeypatch.setattr(acceptance_gate.deployment, "mark_staging_succeeded", mark)
    monkeypatch.setattr(acceptance_gate.deployment, "promote", promote)
    monkeypatch.setattr(acceptance_gate.deployment, "current_release", current_release)
    monkeypatch.setattr(acceptance_gate.deployment, "resolve_release",
                        lambda _root, candidate: {"release_id": candidate})
    findings = acceptance_gate._Findings()

    result = acceptance_gate._promote_rehearsal(release_root, "candidate", findings)

    assert result["status"] == "ok"
    assert findings.rows == []
    assert pointer.read_bytes() == b"sentinel-pointer\n"
    assert [name for name, _root in roots] == ["mark", "promote"]
    assert len(copies) == 1
    source, scratch, copied_pointer = copies[0]
    assert source == store
    assert scratch != store
    assert copied_pointer == b"sentinel-pointer\n"
    assert len(observed) == 1
    observed_root, observed_pointer = observed[0]
    assert observed_root == scratch
    assert observed_root != store
    assert observed_pointer == b"candidate"
