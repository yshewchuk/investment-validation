from dataclasses import replace

import pytest

from engine.v2.foundation import from_document, to_document
from engine.v2.models import (
    ArtifactInventoryMember,
    ModelArtifactInventory,
    ModelReleaseInventory,
    ModelReleaseRefusal,
    ReleaseBinding,
    ReleaseRequirement,
    release_issues,
    require_complete_release,
)


def _member(name, kind):
    return ArtifactInventoryMember(member_id=name, kind=kind, artifact_ref=f"artifact://{name}", content_hash="sha256:" + "a" * 64)


def _artifact(**changes):
    value = ModelArtifactInventory(
        artifact_id="gate-v1", role="gate", strategy_ids=("STR-THRU",),
        compatible_clock_ids=("entry-close",), target_contract_ref="return.v1",
        ordered_features=("forecast", "analog_mean", "entry_cost"),
        members=tuple(_member(kind, kind) for kind in ("estimator", "transform", "residual", "calibration")),
    )
    return replace(value, **changes)


def _release(**changes):
    binding = ReleaseBinding(
        role="gate", strategy_id="STR-THRU", clock_id="entry-close",
        artifact_id="gate-v1", ordered_features=("forecast", "analog_mean", "entry_cost"),
        required_member_kinds=("estimator", "transform", "residual", "calibration"),
    )
    value = ModelReleaseInventory(
        release_id="r1", deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(_artifact(),), bindings=(binding,),
        requirements=(ReleaseRequirement(role="gate", strategy_id="STR-THRU", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r1", evidence_refs=("evidence://r1",),
    )
    return replace(value, **changes)


def _codes(value):
    return {item.code for item in release_issues(value)}


def test_complete_release_round_trips_with_feature_order():
    value = _release()
    assert require_complete_release(value) is value
    assert from_document(ModelReleaseInventory, to_document(value)) == value
    assert value.artifacts[0].ordered_features == ("forecast", "analog_mean", "entry_cost")


def test_missing_binding_is_a_stable_refusal():
    with pytest.raises(ModelReleaseRefusal) as error:
        require_complete_release(_release(bindings=()))
    assert error.value.code == "MODEL_RELEASE_INCOMPLETE"
    assert [item.code for item in error.value.issues] == ["MISSING_BINDING"]


@pytest.mark.parametrize("kind", ["transform", "residual", "calibration"])
def test_missing_required_state_member_refuses(kind):
    artifact = _artifact()
    artifact = replace(artifact, members=tuple(item for item in artifact.members if item.kind != kind))
    assert f"MISSING_{kind.upper()}_MEMBER" in _codes(_release(artifacts=(artifact,)))


def test_feature_order_and_clock_mismatches_refuse():
    value = _release()
    binding = replace(value.bindings[0], ordered_features=("analog_mean", "forecast", "entry_cost"), clock_id="pre-open")
    requirement = ReleaseRequirement(role="gate", strategy_id="STR-THRU", clock_id="pre-open")
    codes = _codes(replace(value, bindings=(binding,), requirements=(requirement,)))
    assert {"FEATURE_ORDER_MISMATCH", "UNKNOWN_CLOCK", "CLOCK_MISMATCH"} <= codes
