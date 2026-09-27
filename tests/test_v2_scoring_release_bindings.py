"""Failure semantics and happy-path resolution of ``release_bindings``."""
import hashlib
import json
import unittest.mock
from pathlib import Path

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models import (
    ArtifactInventoryMember,
    ArtifactMember,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
)
from engine.v2.models import deployment
from engine.v2.models.analog_artifact import (
    BoardAnalogPoolArtifact,
    make_board_analog_pool_artifact,
)
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.lineage import Lineage
from engine.v2.models.loader import FrozenInference
from engine.v2.models.payoff_artifact import (
    PayoffArtifactLoader,
    PayoffArtifactRef,
    PayoffLineArtifact,
    make_payoff_line_artifact,
    serialize_payoff_artifact,
)
from engine.v2.models.recalibration_artifact import (
    make_recalibration_map_artifact,
    serialize_recalibration_artifact,
)
from engine.v2.scoring.release_bindings import (
    ModelNotReady,
    NoCurrentRelease,
    resolve_release_binding,
)

_RELEASE_ID = "r1"
_SCHEMA = "phase5_staged_release.v1.0"


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _dep_root(root: Path) -> Path:
    return root / "deployment"


def _object_path(content_hash_value: str) -> str:
    return "objects/" + content_hash_value.removeprefix("sha256:")


def _linear_payload(intercept=1.0, coefficient=2.0) -> bytes:
    return json.dumps({
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": intercept,
                     "coefficients": [coefficient]}],
    }, sort_keys=True).encode()


def _stage_and_promote(root: Path, *, release_id: str = _RELEASE_ID) -> ModelRelease:
    """Stage one gate/STR-THRU binding and promote it; return the staged release."""
    payload = _linear_payload()
    member_hash = _sha(payload)
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(
        binding_id="b1", model_id="m1", role="gate", strategy_id="STR-THRU",
        decision_clock_id="entry-close", adapter="json-linear.v1",
        feature_order=("x",), output_names=("prediction",), members=(member,))
    release = ModelRelease(release_id=release_id, deployment_id="d1", bindings=(binding,))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(ModelArtifactInventory(
            artifact_id="m1", role="gate", strategy_ids=("STR-THRU",),
            compatible_clock_ids=("entry-close",), target_contract_ref="return.v1",
            ordered_features=("x",), members=(ArtifactInventoryMember(
                member_id="m1:estimator", kind="estimator", artifact_ref="artifact://m1",
                content_hash=member_hash),),),),
        bindings=(ReleaseBinding(
            role="gate", strategy_id="STR-THRU", clock_id="entry-close", artifact_id="m1",
            ordered_features=("x",), required_member_kinds=("estimator",),),),
        requirements=(ReleaseRequirement(
            role="gate", strategy_id="STR-THRU", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",),)
    dep_root = _dep_root(root)
    deployment.stage_release(dep_root, release, inventory, {member_hash: payload})
    deployment.promote(dep_root, release_id)
    return deployment.resolve_release(dep_root, release_id)


def _write_object(dep_root: Path, payload: bytes) -> str:
    path = _object_path(_sha(payload))
    target = dep_root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return path


def _payoff() -> tuple[PayoffLineArtifact, bytes]:
    artifact = make_payoff_line_artifact(
        {"n": 2, "intercept": 0.1, "slope": 0.2, "resid_sd": 0.01, "r": 0.5,
         "residuals": [0.01, -0.01]},
        strategy="STR-THRU", driver="driver_prediction", alpha=0.55)
    return artifact, serialize_payoff_artifact(artifact)


def _recalibration() -> tuple[object, bytes]:
    artifact = make_recalibration_map_artifact(
        {"n": 3, "base_rate": 0.4, "x_thresholds": [0.3, 0.7], "y_thresholds": [0.35, 0.65]},
        strategy="STR-THRU", alpha=0.55, min_pairs=2)
    return artifact, serialize_recalibration_artifact(artifact)


def _analog() -> tuple[BoardAnalogPoolArtifact, bytes]:
    artifact = make_board_analog_pool_artifact(
        strategy="STR-THRU", alpha=0.55, cutoff=None, population_edges=[0.0, 1.0],
        causal_edges=None, lineage=Lineage(),
        rows=[["row-1", "2026-01-01", "2026-01-05", 0.02, 0.03, None, None, None]])
    return artifact, serialize_frozen_state(artifact)


def _row(member_id: str, objects: list, status: str = "STAGED") -> dict:
    return {"member_id": member_id, "status": status, "objects": objects}


def _write_catalog(root: Path, *, release_id: str = _RELEASE_ID, schema: str = _SCHEMA,
                   rows: list | None = None) -> None:
    body: dict = {"schema_version": schema, "release_id": release_id,
                  "members": [] if rows is None else rows}
    body["manifest_hash"] = content_hash(body)
    (root / "phase5_release.json").write_text(json.dumps(body, indent=2, sort_keys=True))


def _happy_catalog(root: Path) -> str:
    """Stage every state family's object; return the payoff object's path."""
    dep_root = _dep_root(root)
    payoff, payoff_bytes = _payoff()
    payoff_path = _write_object(dep_root, payoff_bytes)
    recal, recal_bytes = _recalibration()
    recal_path = _write_object(dep_root, recal_bytes)
    analog, analog_bytes = _analog()
    analog_path = _write_object(dep_root, analog_bytes)
    _write_catalog(root, rows=[
        _row("payoff_line:STR-THRU", [
            {"path": payoff_path, "content_hash": payoff.content_hash}]),
        _row("recalibration_map:STR-THRU", [
            {"path": recal_path, "content_hash": recal.content_hash}]),
        _row("board_analog_matcher", [
            {"path": analog_path, "content_hash": analog.content_hash}]),
    ])
    return payoff_path


def test_no_current_release_raises_typed_refusal(tmp_path):
    with pytest.raises(NoCurrentRelease):
        resolve_release_binding(tmp_path)


def test_corrupt_deployed_pointer_invalid_utf8_raises_model_not_ready(tmp_path):
    deployed = _dep_root(tmp_path) / "DEPLOYED"
    deployed.parent.mkdir(parents=True)
    deployed.write_bytes(b"\xff\xfe not utf8")
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == "DEPLOYED"


def test_missing_model_binding_object_raises_model_not_ready(tmp_path):
    release = _stage_and_promote(tmp_path)
    (_dep_root(tmp_path) / release.bindings[0].members[0].path).unlink()
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id.startswith("model:")


def test_unreadable_model_binding_object_raises_model_not_ready_without_leaking_path(tmp_path):
    release = _stage_and_promote(tmp_path)
    member_path = _dep_root(tmp_path) / release.bindings[0].members[0].path
    real_read = Path.read_bytes

    def failing_read(self):
        if self == member_path:
            raise OSError("simulated read failure")
        return real_read(self)

    with unittest.mock.patch.object(Path, "read_bytes", failing_read):
        with pytest.raises(ModelNotReady) as error:
            resolve_release_binding(tmp_path)
    assert error.value.member_id.startswith("model:")
    assert str(member_path) not in error.value.detail
    assert "simulated read failure" not in error.value.detail


def test_model_binding_hash_mismatch_raises_model_not_ready_never_falls_back(tmp_path):
    release = _stage_and_promote(tmp_path)
    member_path = _dep_root(tmp_path) / release.bindings[0].members[0].path
    member_path.write_bytes(_linear_payload(intercept=99.0, coefficient=98.0))
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id.startswith("model:")


def test_missing_state_catalog_raises_model_not_ready(tmp_path):
    _stage_and_promote(tmp_path)
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == "phase5_release.json"


def test_state_catalog_release_id_mismatch_raises_model_not_ready(tmp_path):
    _stage_and_promote(tmp_path)
    _write_catalog(tmp_path, release_id="some-other-release")
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == "phase5_release.json"


def test_state_catalog_wrong_schema_version_raises_model_not_ready(tmp_path):
    _stage_and_promote(tmp_path)
    _write_catalog(tmp_path, schema="phase5_staged_release.v9.9")
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == "phase5_release.json"


def test_payoff_member_pending_status_raises_model_not_ready(tmp_path):
    _stage_and_promote(tmp_path)
    _write_catalog(tmp_path, rows=[_row("payoff_line:STR-THRU", [], status="PENDING")])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == "payoff_line:STR-THRU"


def test_recalibration_member_hash_mismatch_raises_model_not_ready(tmp_path):
    _stage_and_promote(tmp_path)
    _recal, recal_bytes = _recalibration()
    path = _write_object(_dep_root(tmp_path), recal_bytes)
    _write_catalog(tmp_path, rows=[_row("recalibration_map:STR-THRU", [
        {"path": path, "content_hash": _sha(b"declared-but-wrong")}])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == "recalibration_map:STR-THRU"


def test_malformed_object_reference_in_catalog_raises_model_not_ready_without_leaking_content(tmp_path):
    _stage_and_promote(tmp_path)
    _write_catalog(tmp_path, rows=[_row("payoff_line:STR-THRU", [
        {"path": "objects/whatever", "content_hash": None}])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == "payoff_line:STR-THRU"
    assert "whatever" not in error.value.detail


def test_analog_member_unloadable_raises_model_not_ready(tmp_path):
    _stage_and_promote(tmp_path)
    junk = json.dumps({"schema_version": "not_a_frozen_state.v9.9", "rows": []}).encode()
    path = _write_object(_dep_root(tmp_path), junk)
    _write_catalog(tmp_path, rows=[_row("board_analog_matcher", [
        {"path": path, "content_hash": _sha(junk)}])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == "board_analog_matcher"


def test_happy_path_resolves_every_field(tmp_path):
    release = _stage_and_promote(tmp_path)
    payoff_path = _happy_catalog(tmp_path)

    binding = resolve_release_binding(tmp_path)

    assert binding.release_id == release.release_id
    declared = release.bindings[0]
    assert set(binding.model_identity) == {"gate:STR-THRU"}
    identity = binding.model_identity["gate:STR-THRU"]
    assert identity.binding_id == declared.binding_id
    assert identity.model_id == declared.model_id
    assert identity.role == declared.role
    assert identity.strategy_id == declared.strategy_id
    assert identity.decision_clock_id == declared.decision_clock_id
    assert identity.adapter == declared.adapter
    assert identity.feature_order == declared.feature_order
    assert identity.output_names == declared.output_names
    assert dict(binding.model_artifact_refs) == {declared.binding_id: identity.artifact_hash}
    assert binding.model_release == release
    assert isinstance(binding.frozen_inference, FrozenInference)

    expected_payoff, _payoff_bytes = _payoff()
    assert binding.payoff_artifacts["STR-THRU"] == (expected_payoff,)
    expected_recal, _recal_bytes = _recalibration()
    assert binding.recalibration_artifacts["STR-THRU"] == (expected_recal,)
    expected_analog, _analog_bytes = _analog()
    assert binding.analog_artifacts["STR-THRU"] == (expected_analog,)
    assert isinstance(binding.analog_artifacts["STR-THRU"][0], BoardAnalogPoolArtifact)

    independently = PayoffArtifactLoader(_dep_root(tmp_path)).load(
        PayoffArtifactRef(path=payoff_path, content_hash=expected_payoff.content_hash))
    assert isinstance(independently, PayoffLineArtifact)
    assert independently == expected_payoff


def test_never_reads_data_directory_or_legacy_path(tmp_path):
    _stage_and_promote(tmp_path)
    _happy_catalog(tmp_path)
    real_read = Path.read_bytes

    def guarded_read(self):
        text = str(self)
        if "/data/" in text or "engine/score.py" in text or "engine.score" in text:
            raise AssertionError(f"release_bindings read a legacy path: {text}")
        return real_read(self)

    with unittest.mock.patch.object(Path, "read_bytes", guarded_read):
        assert resolve_release_binding(tmp_path).release_id == _RELEASE_ID


def _snapshot(root: Path) -> list:
    return sorted((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                  for path in root.rglob("*") if path.is_file())


def test_read_only_no_files_change(tmp_path):
    _stage_and_promote(tmp_path)
    _happy_catalog(tmp_path)
    before = _snapshot(tmp_path)
    resolve_release_binding(tmp_path)
    assert _snapshot(tmp_path) == before
    resolve_release_binding(tmp_path)
    assert _snapshot(tmp_path) == before


def test_repeated_calls_do_not_share_a_cache_across_release_changes(tmp_path):
    _stage_and_promote(tmp_path)
    payoff_path = _happy_catalog(tmp_path)
    resolve_release_binding(tmp_path)
    payoff, _payoff_bytes = _payoff()
    (_dep_root(tmp_path) / payoff_path).write_bytes(
        serialize_payoff_artifact(payoff) + b" tampered")
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == "payoff_line:STR-THRU"
