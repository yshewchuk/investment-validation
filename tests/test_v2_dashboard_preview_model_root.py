"""Phase 6 C4 (partial): the launcher's ``--model-release-root`` wiring.

Drives ``preview.run`` — the real startup boundary — on a loopback ephemeral
port and reads ``/models/release.json`` back over HTTP, so the acceptance
covers argv parsing, the ``create_server`` hand-off and the route together.
No Playwright: everything here is the in-process HTTP surface only. Refresh
and what-if callbacks remain unconfigured by design in this slice.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

from engine.v2.dashboard import preview
from engine.v2.models import (
    ArtifactInventoryMember,
    ArtifactMember,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
    promote,
    stage_release,
)

TOKEN = "launcher-secret"


# --------------------------------------------------------------------------
# model release fixture (mirrors tests/test_v2_ops_serving_model_release.py)
# --------------------------------------------------------------------------


def _hash(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _stage_and_promote(models_root: Path, release_id: str) -> Path:
    payload = json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
                          "outputs": [{"name": "prediction", "intercept": 1.0,
                                       "coefficients": [2.0]}]}, sort_keys=True).encode()
    member_hash = _hash(payload)
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(
        binding_id="b1", model_id="m1", role="size", strategy_id="*",
        decision_clock_id="entry-close", adapter="json-linear.v1",
        feature_order=("x",), output_names=("prediction",), members=(member,),
    )
    release = ModelRelease(release_id=release_id, deployment_id="d1", bindings=(binding,))
    inv_member = ArtifactInventoryMember(member_id="m1:estimator", kind="estimator",
                                         artifact_ref="artifact://m1", content_hash=member_hash)
    artifact = ModelArtifactInventory(
        artifact_id="m1", role="size", strategy_ids=("*",), compatible_clock_ids=("entry-close",),
        target_contract_ref="return.v1", ordered_features=("x",), members=(inv_member,),
    )
    inv_binding = ReleaseBinding(
        role="size", strategy_id="*", clock_id="entry-close", artifact_id="m1",
        ordered_features=("x",), required_member_kinds=("estimator",),
    )
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(artifact,), bindings=(inv_binding,),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",),
    )
    stage_release(models_root, release, inventory, {member_hash: payload})
    promote(models_root, release_id)
    return models_root


# --------------------------------------------------------------------------
# launcher helpers
# --------------------------------------------------------------------------


def _dashboard_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    release_dir = bundle / "releases" / "r1"
    release_dir.mkdir(parents=True)
    (release_dir / "index.html").write_text("<!doctype html><title>legacy</title>")
    (bundle / "CURRENT").write_text("r1\n")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0"}))
    return bundle, health


def _run_launcher(tmp_path: Path, monkeypatch, extra=()):
    monkeypatch.setenv(preview.TOKEN_ENV_VAR, TOKEN)
    bundle, health = _dashboard_bundle(tmp_path)
    return preview.run(["--host", "127.0.0.1", "--port", "0",
                        "--release-root", str(bundle), "--health-path", str(health), *extra])


def _release_json(server) -> dict:
    request = Request(f"http://127.0.0.1:{server.server_port}/models/release.json")
    request.add_header("Authorization", "Bearer " + TOKEN)
    return json.loads(urlopen(request, timeout=5).read())


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


# --------------------------------------------------------------------------
# acceptance
# --------------------------------------------------------------------------


def test_launch_with_model_release_root_serves_deployed_release(tmp_path, monkeypatch):
    models_root = _stage_and_promote(tmp_path / "models", "mr1")
    server, thread, release_id = _run_launcher(
        tmp_path, monkeypatch, ["--model-release-root", str(models_root)])
    try:
        assert release_id == "r1"  # the rest of the launcher still starts and pins normally
        body = _release_json(server)
        assert body["status"] == "deployed"
        assert body["release_id"] == "mr1"
        assert body["deployment_id"] == "d1"
        assert [m["model_id"] for m in body["members"]] == ["m1"]
        assert body["dependencies"]["promotion_action"] == "promote"
        assert TOKEN not in json.dumps(body)  # the refusal/success body never carries the token
    finally:
        _stop(server, thread)


def test_launch_without_model_release_root_keeps_not_configured(tmp_path, monkeypatch):
    # Omitting the option must not guess a model root from --release-root: the
    # bundle store also holds releases/r1/, and picking it up would be wrong.
    server, thread, release_id = _run_launcher(tmp_path, monkeypatch)
    try:
        assert release_id == "r1"
        body = _release_json(server)
        assert body["status"] == "refused"
        assert body["reason_code"] == "MODEL_RELEASE_NOT_CONFIGURED"
        assert body["release_id"] is None
    finally:
        _stop(server, thread)


def test_launch_with_empty_model_release_root_reports_not_deployed(tmp_path, monkeypatch):
    # An explicitly configured but empty store is a DISTINCT state from the
    # not-configured refusal: NOT_CONFIGURED must turn into NOT_DEPLOYED.
    models_root = tmp_path / "models"
    models_root.mkdir()
    server, thread, _ = _run_launcher(
        tmp_path, monkeypatch, ["--model-release-root", str(models_root)])
    try:
        body = _release_json(server)
        assert body["status"] == "refused"
        assert body["reason_code"] == "MODEL_RELEASE_NOT_DEPLOYED"
        assert body["release_id"] is None
    finally:
        _stop(server, thread)
