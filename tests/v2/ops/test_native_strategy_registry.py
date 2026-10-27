"""Registry routing over real synthetic releases; only THRU and RUNUP enabled."""
from dataclasses import FrozenInstanceError, replace

import pytest

from engine.v2.ops import native_score_batch as batch
from tests.v2.ops.test_native_score_batch_driver_alias import (
    DRIVER,
    GATE,
    SIZE,
    _assemble,
    _event_inputs,
    _stage,
)


def test_registry_is_immutable_and_twin_are_enabled():
    registry = batch._STRATEGY_REGISTRY
    assert tuple(registry) == ("STR-THRU", "STR-RUNUP", "TWIN-P", "TWIN-P5", "BFLY-P", "BFLY-P5", "CTR5", "RAMP7", "CND-PS")
    assert registry["STR-THRU"] == batch._StrategySpec(
        drivers=(batch._DriverSpec("driver_prediction", "driver:STR-THRU", "size:*"),),
        gate_role="gate:STR-THRU", gate_policy_key="STR-THRU",
        bundle_strategy="STR-THRU")
    with pytest.raises(TypeError):
        registry["STR-RUNUP"] = registry["STR-THRU"]
    with pytest.raises(FrozenInstanceError):
        registry["STR-THRU"].drivers[0].alias = "implied_t1:*"


def test_unregistered_rows_are_all_retained_as_refusals(tmp_path):
    binding = _stage(tmp_path, [DRIVER, SIZE, GATE])
    unsupported = ["DYN-SV", "UNKNOWN"]
    events = [_event_inputs(strategy=name) for name in ["STR-THRU", *unsupported]]
    assembled, refusals = _assemble(binding, events)
    assert list(assembled) == [events[0].key]
    assert [refusal.key for refusal in refusals] == [event.key for event in events[1:]]
    assert [refusal.code for refusal in refusals] == ["UNSUPPORTED_STRATEGY"] * 2
    assert [refusal.detail for refusal in refusals] == [
        f"only STR-THRU, STR-RUNUP, TWIN-P, TWIN-P5, BFLY-P, BFLY-P5, CTR5, RAMP7, CND-PS is supported, got {name!r}" for name in unsupported]


def test_admission_comes_from_registry_not_a_second_allowlist(tmp_path, monkeypatch):
    binding = _stage(tmp_path, [DRIVER, GATE])
    monkeypatch.setattr(batch, "_STRATEGY_REGISTRY", {})
    assembled, refusals = _assemble(binding, [_event_inputs()])
    assert assembled == {}
    assert [refusal.code for refusal in refusals] == ["UNSUPPORTED_STRATEGY"]


def test_registered_exact_driver_key_is_used(tmp_path, monkeypatch):
    binding = _stage(tmp_path, [DRIVER, SIZE, GATE])
    spec = replace(batch._STRATEGY_REGISTRY["STR-THRU"],
                   drivers=(batch._DriverSpec("driver_prediction", "size:*"),))
    monkeypatch.setattr(batch, "_STRATEGY_REGISTRY", {"STR-THRU": spec})
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    request, inputs = assembled[event.key]
    assert set(inputs.features["model_identity"]) == {"size:*", "gate:STR-THRU"}
    assert request.model_artifact_refs[0] == binding.model_identity["size:*"].artifact_hash


@pytest.mark.parametrize("roles,driver_key,bundle_strategy", [
    ([DRIVER, SIZE, GATE], "driver:STR-THRU", "BFLY-P"),
    ([SIZE, GATE], "size:*", "BFLY-P"),
    ([DRIVER, GATE], "driver:STR-THRU", "STR-RUNUP"),
])
def test_registered_keys_reach_real_bundle_and_preserve_downstream_refusal(
        tmp_path, monkeypatch, roles, driver_key, bundle_strategy):
    """Test-only registration proves routing, not BFLY-P readiness.

    The test-only row/bundle mismatch proves explicit bundle routing. Its
    single-driver recipe is insufficient for either real lower builder, which
    must still refuse instead of inventing the missing output.
    """
    binding = _stage(tmp_path, roles)
    spec = replace(batch._STRATEGY_REGISTRY["STR-THRU"],
                   gate_policy_key="test-policy", bundle_strategy=bundle_strategy)
    monkeypatch.setattr(batch, "_STRATEGY_REGISTRY", {"BFLY-P": spec})
    observed = []
    real_builder = batch.assemble_nightly_source_bundle

    def inspect_then_build(**kwargs):
        bundle = real_builder(**kwargs)
        observed.append(bundle)
        return bundle

    monkeypatch.setattr(batch, "assemble_nightly_source_bundle", inspect_then_build)
    event = _event_inputs(strategy="BFLY-P")
    # The test-only RUNUP bundle route requires its calendar-owned horizon.
    event = replace(event, calendar_row={**event.calendar_row, "days_before_print": 5})
    assembled, refusals = _assemble(
        binding, [event], gate_policy={"test-policy": {"threshold": 0.0}})
    assert assembled == {}
    expected_code = "MODEL_NOT_READY" if bundle_strategy == "STR-RUNUP" else "NATIVE_INPUT_BUILD_FAILED"
    assert [refusal.code for refusal in refusals] == [expected_code]
    assert [refusal.key for refusal in refusals] == [event.key]
    assert len(observed) == 1
    bundle = observed[0]
    assert bundle.strategy == bundle_strategy
    assert bundle.source_ref == "native-score-batch:TEST:2026-01-15:BFLY-P"
    assert set(bundle.model_identity) == {driver_key, "gate:STR-THRU"}
    assert bundle.model_identity[driver_key]["binding_id"] == binding.model_identity[
        driver_key].binding_id
    assert bundle.forecast_recipes == {
        "driver_prediction": {"binding_id": binding.model_identity[driver_key].binding_id}}
    assert bundle.gate_recipe == {"threshold": 0.0, "binding_id": "b-gate-STR-THRU"}


@pytest.mark.parametrize("field,value,roles,policy,code,detail", [
    ("role", "driver:missing", [GATE], {"STR-THRU": {}},
     "RELEASE_MISSING_ROLE", "driver:missing"),
    ("alias", "size:missing", [SIZE, GATE], {"STR-THRU": {}},
     "RELEASE_MISSING_ROLE", "driver:STR-THRU"),
    ("gate_role", "gate:missing", [DRIVER, GATE], {"STR-THRU": {}},
     "RELEASE_MISSING_ROLE", "gate:missing"),
    ("gate_policy_key", "missing-policy", [DRIVER, GATE], {"STR-THRU": {}},
     "GATE_POLICY_NOT_STAGED", "no gate policy staged for 'STR-THRU'"),
])
def test_missing_registered_dependency_refuses_without_bundle(
        tmp_path, monkeypatch, field, value, roles, policy, code, detail):
    binding = _stage(tmp_path, roles)
    spec = batch._STRATEGY_REGISTRY["STR-THRU"]
    spec = (replace(spec, drivers=(replace(spec.drivers[0], **{field: value}),))
            if field in {"role", "alias"} else replace(spec, **{field: value}))
    monkeypatch.setattr(batch, "_STRATEGY_REGISTRY", {"STR-THRU": spec})

    def never_build(**kwargs):
        pytest.fail("missing registered dependency reached source-bundle assembly")

    monkeypatch.setattr(batch, "assemble_nightly_source_bundle", never_build)
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event], gate_policy=policy)
    assert assembled == {}
    assert len(refusals) == 1
    assert (refusals[0].key, refusals[0].code, refusals[0].detail) == (event.key, code, detail)


def test_registered_alias_clock_mismatch_refuses_before_bundle(tmp_path):
    binding = _stage(tmp_path, [SIZE, GATE])
    identities = dict(binding.model_identity)
    identities["size:*"] = replace(identities["size:*"], decision_clock_id="other-clock")
    binding = replace(binding, model_identity=identities)
    assembled, refusals = _assemble(binding, [_event_inputs()])
    assert assembled == {}
    assert [refusal.code for refusal in refusals] == ["AMBIGUOUS_DECISION_CLOCK"]
    assert refusals[0].detail == (
        "driver:STR-THRU='other-clock' gate:STR-THRU='entry-close'")
