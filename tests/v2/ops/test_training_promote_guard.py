"""Issue 192 slice 1: models_promote worker and CLI propagation of expected_previous_release_id."""
import hashlib
import json

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
    deployment,
)
from engine.v2.ops import cli, training
from engine.v2.ops.errors import OpsError


def _release_fixture(release_id, *, intercept, coefficient):
    """A one-binding synthetic ``ModelRelease`` plus its matching inventory."""
    payload = json.dumps({
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": intercept,
                     "coefficients": [coefficient]}],
    }, sort_keys=True).encode()
    member_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(
        binding_id="b1", model_id="m1", role="size", strategy_id="*",
        decision_clock_id="entry-close", adapter="json-linear.v1",
        feature_order=("x",), output_names=("prediction",), members=(member,))
    release = ModelRelease(release_id=release_id, deployment_id="d1", bindings=(binding,))

    inv_member = ArtifactInventoryMember(member_id="m1:estimator", kind="estimator",
                                         artifact_ref="artifact://m1", content_hash=member_hash)
    artifact = ModelArtifactInventory(
        artifact_id="m1", role="size", strategy_ids=("*",), compatible_clock_ids=("entry-close",),
        target_contract_ref="return.v1", ordered_features=("x",), members=(inv_member,))
    inv_binding = ReleaseBinding(
        role="size", strategy_id="*", clock_id="entry-close", artifact_id="m1",
        ordered_features=("x",), required_member_kinds=("estimator",))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(artifact,), bindings=(inv_binding,),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",))
    return release, inventory, {member_hash: payload}


def _stage_pair(root):
    """Stage and mark two distinct synthetic releases r1 and r2 under ``root``."""
    for release_id, intercept, coefficient in (("r1", 1.0, 2.0), ("r2", 10.0, 20.0)):
        release, inventory, payloads = _release_fixture(release_id, intercept=intercept,
                                                        coefficient=coefficient)
        deployment.stage_release(root, release, inventory, payloads)
        deployment.mark_staging_succeeded(root, release_id)


def test_promote_worker_refuses_concurrent_promote_without_touching_deployment(tmp_path):
    _stage_pair(tmp_path)
    deployment.promote(tmp_path, "r1")
    history_before = deployment.pointer_history(tmp_path)

    with pytest.raises(OpsError) as excinfo:
        training.run_promote_worker(
            {"expected_ids": ["models_promote"], "release_root": str(tmp_path),
             "release_id": "r2", "expected_previous_release_id": "r2"},
            tmp_path / "staging")

    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details == {"exception_class": "ConcurrentPromote"}
    assert not (tmp_path / "staging" / "pointer_state.json").exists()
    assert deployment.current_pointer(tmp_path).release_id == "r1"
    assert deployment.pointer_history(tmp_path) == history_before


def test_promote_worker_guard_reaches_deployment_from_plan_parameters(tmp_path):
    _stage_pair(tmp_path)
    deployment.promote(tmp_path, "r1")
    (tmp_path / "staging").mkdir()

    plan = training.promote_plan(release_root=str(tmp_path), release_id="r2",
                                 expected_previous_release_id="r1")
    assert plan["parameters"]["expected_previous_release_id"] == "r1"
    result = training.run_promote_worker(plan["parameters"], tmp_path / "staging")

    assert result["completed_ids"] == ["models_promote"]
    document = json.loads((tmp_path / "staging" / "pointer_state.json").read_text())
    assert document["release_id"] == "r2"
    assert deployment.current_pointer(tmp_path).release_id == "r2"
    assert [item.release_id for item in deployment.pointer_history(tmp_path)] == ["r1", "r2"]


def test_promote_worker_without_the_guard_key_keeps_previous_behavior(tmp_path):
    _stage_pair(tmp_path)
    deployment.promote(tmp_path, "r1")
    (tmp_path / "staging").mkdir()

    result = training.run_promote_worker(
        {"expected_ids": ["models_promote"], "release_root": str(tmp_path),
         "release_id": "r2"}, tmp_path / "staging")

    assert result["completed_ids"] == ["models_promote"]
    assert deployment.current_pointer(tmp_path).release_id == "r2"


def test_promote_worker_same_target_noop_with_stale_guard(tmp_path):
    _stage_pair(tmp_path)
    deployment.promote(tmp_path, "r1")
    deployment.promote(tmp_path, "r2")
    history_before = deployment.pointer_history(tmp_path)
    (tmp_path / "staging").mkdir()

    result = training.run_promote_worker(
        {"expected_ids": ["models_promote"], "release_root": str(tmp_path),
         "release_id": "r2", "expected_previous_release_id": "r1"},
        tmp_path / "staging")

    assert result["completed_ids"] == ["models_promote"]
    assert len(deployment.pointer_history(tmp_path)) == len(history_before) == 2
    assert deployment.current_pointer(tmp_path).release_id == "r2"


def test_cli_plan_promote_carries_expected_previous_release_id(tmp_path, capsys):
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()

    assert cli.main(["--root", str(root), "plan", "promote",
                     "--release-root", str(tmp_path), "--release-id", "r2",
                     "--expected-previous-release-id", "r1"]) == 0
    parameters = json.loads(capsys.readouterr().out)["plan"]["parameters"]
    assert parameters["expected_previous_release_id"] == "r1"

    assert cli.main(["--root", str(root), "plan", "promote",
                     "--release-root", str(tmp_path), "--release-id", "r2"]) == 0
    parameters = json.loads(capsys.readouterr().out)["plan"]["parameters"]
    assert parameters["expected_previous_release_id"] is None
