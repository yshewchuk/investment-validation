"""HTTP-level coverage for the P6-4 model-release view (no v2 view read the
Phase 5 deployment pointer before this)."""
from __future__ import annotations

import hashlib
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

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
from engine.v2.serving.operations import create_server


def _get(url, token=None):
    request = Request(url)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    return urlopen(request)


def _hash(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _fixture(release_id):
    """A one-binding release plus its matching completeness inventory (mirrors
    tests/test_v2_models_deployment.py's own fixture)."""
    payload = json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
                          "outputs": [{"name": "prediction", "intercept": 1.0, "coefficients": [2.0]}]},
                         sort_keys=True).encode()
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
    return release, inventory, {member_hash: payload}, member_hash


def _serve(tmp_path, **kwargs):
    health = tmp_path / "health.json"
    health.write_text('{"schema_version":"operations_health.v1.0"}')
    (tmp_path / "releases").mkdir(exist_ok=True)
    server = create_server(("127.0.0.1", 0), token="secret", health_path=health,
                           release_root=tmp_path, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, "http://127.0.0.1:" + str(server.server_port)


def _stop(server, thread):
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def test_json_route_requires_auth(tmp_path):
    server, thread, base = _serve(tmp_path, model_release_root=tmp_path / "models")
    try:
        with pytest.raises(HTTPError) as error:
            _get(base + "/models/release.json")
        assert error.value.code == 401
    finally:
        _stop(server, thread)


def test_not_configured_is_an_explicit_refusal(tmp_path):
    server, thread, base = _serve(tmp_path)
    try:
        body = json.loads(_get(base + "/models/release.json", "secret").read())
        assert body["status"] == "refused"
        assert body["reason_code"] == "MODEL_RELEASE_NOT_CONFIGURED"
        assert body["release_id"] is None
        assert "members" not in body
    finally:
        _stop(server, thread)


def test_no_release_deployed_is_an_explicit_refusal_not_empty_success(tmp_path):
    server, thread, base = _serve(tmp_path, model_release_root=tmp_path / "models")
    try:
        body = json.loads(_get(base + "/models/release.json", "secret").read())
        assert body["status"] == "refused"
        assert body["reason_code"] == "MODEL_RELEASE_NOT_DEPLOYED"
        assert body["release_id"] is None
        assert "members" not in body
        assert "dependencies" not in body
    finally:
        _stop(server, thread)


def test_deployed_release_is_visible_end_to_end(tmp_path):
    models_root = tmp_path / "models"
    release, inventory, payloads, member_hash = _fixture("r1")
    stage_release(models_root, release, inventory, payloads)
    promote(models_root, "r1")
    server, thread, base = _serve(tmp_path, model_release_root=models_root)
    try:
        body = json.loads(_get(base + "/models/release.json", "secret").read())
        assert body["status"] == "deployed"
        assert body["release_id"] == "r1"
        assert body["deployment_id"] == "d1"
        assert len(body["members"]) == 1
        member = body["members"][0]
        assert member["binding_id"] == "b1"
        assert member["model_id"] == "m1"
        assert member["role"] == "size"
        assert member["strategy_id"] == "*"
        assert member["decision_clock_id"] == "entry-close"
        assert member["adapter"] == "json-linear.v1"
        assert member["artifacts"] == [{"name": "estimator", "content_hash": member_hash}]
        assert body["dependencies"]["previous_release_id"] is None
        assert body["dependencies"]["promotion_action"] == "promote"
    finally:
        _stop(server, thread)


def test_rollback_lineage_is_reported_as_a_dependency(tmp_path):
    models_root = tmp_path / "models"
    release1, inventory1, payloads1, _ = _fixture("r1")
    release2, inventory2, payloads2, _ = _fixture("r2")
    stage_release(models_root, release1, inventory1, payloads1)
    stage_release(models_root, release2, inventory2, payloads2)
    promote(models_root, "r1")
    promote(models_root, "r2")
    server, thread, base = _serve(tmp_path, model_release_root=models_root)
    try:
        body = json.loads(_get(base + "/models/release.json", "secret").read())
        assert body["release_id"] == "r2"
        assert body["dependencies"]["previous_release_id"] == "r1"
    finally:
        _stop(server, thread)


def test_unresolvable_pointer_is_unavailable_not_empty_success(tmp_path):
    models_root = tmp_path / "models"
    release, inventory, payloads, _ = _fixture("r1")
    stage_release(models_root, release, inventory, payloads)
    promote(models_root, "r1")
    (models_root / "releases" / "r1" / "manifest.json").unlink()
    server, thread, base = _serve(tmp_path, model_release_root=models_root)
    try:
        request = Request(base + "/models/release.json")
        request.add_header("Authorization", "Bearer secret")
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 503
        body = json.loads(error.value.read())
        assert body["status"] == "unavailable"
        assert body["reason_code"] == "MODEL_RELEASE_POINTER_UNRESOLVED"
        assert body["release_id"] == "r1"
    finally:
        _stop(server, thread)


def test_page_route_serves_html_without_auth_and_reads_from_the_json_route(tmp_path):
    server, thread, base = _serve(tmp_path, model_release_root=tmp_path / "models")
    try:
        response = _get(base + "/models/release")
        assert response.read().startswith(b"<!doctype html>")
    finally:
        _stop(server, thread)