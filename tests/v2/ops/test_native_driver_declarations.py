"""Ordered driver declarations over real synthetic releases and model state."""
import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops import native_score_batch as batch
from engine.v2.scoring.frozen_executor import FrozenStageRefusal
from engine.v2.scoring.identity import request_hash
from tests.v2.ops.runup_helpers import stage_model_release as _stage
from tests.v2.ops.test_native_score_batch_driver_alias import (
    GATE,
    SIZE,
    _assemble,
    _event_doc,
    _event_inputs,
    _staged_driver_pool,
    _worker_parameters,
)
from tests.v2.ops.test_native_score_batch_driver_alias import (
    _stage as _stage_without_model_state,
)

IMPLIED = ("implied_t1", "*", ("days_before_print", "implied_x", "shared"))
EXACT = ("driver", "STR-RUNUP", ("days_before_print", "exact_x", "shared"))
MOVE = ("runup_move", "*", ("move_x", "shared"))
RUNUP_GATE = ("gate", "STR-RUNUP", ("gate_x", "pred_abs_move", "analog_n"))
POLICY = {"STR-RUNUP": {"threshold": 0.0}, "STR-THRU": {"threshold": 0.0}}


def _enable_test_runup(monkeypatch):
    monkeypatch.setattr(batch, "_STRATEGY_REGISTRY", {
        **batch._STRATEGY_REGISTRY, "STR-RUNUP": batch._RUNUP_SPEC})


def _event():
    event = _event_inputs(strategy="STR-RUNUP")
    return replace(event,
                   calendar_row={**event.calendar_row, "entry_date": "2026-01-09",
                                 "exit_date": "2026-01-14", "expiry": "2026-02-20", "days_before_print": 3},
                   quote_rows=[{**quote, "expiry": "2026-02-20"} for quote in event.quote_rows],
                   panel_row={**event.panel_row, "implied_x": 2.0, "exact_x": 7.0,
                              "move_x": 11.0, "shared": 5.0, "gate_x": 13.0})


def _build(binding, event=None, **kwargs):
    event = event or _event()
    rows, refusals = _assemble(binding, [event], gate_policy=POLICY, **kwargs)
    return rows, refusals, event


def test_runup_descriptor_is_frozen_and_registered():
    assert tuple(batch._STRATEGY_REGISTRY) == ("STR-THRU", "STR-RUNUP", "TWIN-P", "TWIN-P5", "BFLY-P", "BFLY-P5", "CTR5", "RAMP7")
    spec = batch._RUNUP_SPEC
    assert batch._STRATEGY_REGISTRY["STR-RUNUP"] is spec
    assert spec.drivers == (
        batch._DriverSpec("driver_prediction", "driver:STR-RUNUP", "implied_t1:*"),
        batch._DriverSpec("runup_move_prediction", "runup_move:*"))
    assert (spec.gate_role, spec.gate_policy_key, spec.bundle_strategy, spec.driver_name) == (
        "gate:STR-RUNUP", "STR-RUNUP", "STR-RUNUP", "im_t1")
    with pytest.raises(FrozenInstanceError):
        spec.driver_name = "abs_move"
    with pytest.raises(FrozenInstanceError):
        spec.drivers[1].output = "driver_prediction"


@pytest.mark.parametrize("drivers", [
    (), [], [batch._DriverSpec("driver_prediction", "one")], (None,),
    (batch._DriverSpec("driver_prediction", "one"), batch._DriverSpec("driver_prediction", "two")),
])
def test_malformed_code_owned_declarations_cannot_silently_drop_outputs(drivers):
    with pytest.raises(ValueError):
        replace(batch._RUNUP_SPEC, drivers=drivers)


def test_driver_output_name_is_declarative_without_enabling_another_strategy():
    spec = replace(batch._STRATEGY_REGISTRY["STR-THRU"],
                   drivers=(batch._DriverSpec("forecast_abs_move", "size:*"),))
    assert spec.drivers[0].output == "forecast_abs_move"
    assert tuple(batch._STRATEGY_REGISTRY) == ("STR-THRU", "STR-RUNUP", "TWIN-P", "TWIN-P5", "BFLY-P", "BFLY-P5", "CTR5", "RAMP7")


@pytest.mark.parametrize("exact,driver_key,expected", [
    (False, "implied_t1:*", 11.0), (True, "driver:STR-RUNUP", 16.0),
])
def test_every_recipe_identity_reference_and_frozen_consumer_survives(
        tmp_path, monkeypatch, exact, driver_key, expected):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [IMPLIED, MOVE, RUNUP_GATE, *([EXACT] if exact else [])])
    rows, refusals, event = _build(binding, driver_name="caller-override")
    assert refusals == ()
    request, inputs = rows[event.key]
    assert inputs.context["strategy"] == "STR-RUNUP"
    assert inputs.forecast["driver_name"] == "im_t1"
    assert set(inputs.forecast["executors"]) == {"driver_prediction", "runup_move_prediction"}
    # Independent dataclass/JSON projection, not the production document serializer.
    assert inputs.features["model_identity"] == {
        key: json.loads(json.dumps(asdict(binding.model_identity[key])))
        for key in (driver_key, "runup_move:*", "gate:STR-RUNUP")}
    selected = [binding.model_identity[key] for key in (driver_key, "runup_move:*", "gate:STR-RUNUP")]
    assert request.model_artifact_refs == tuple(identity.artifact_hash for identity in selected)
    assert inputs.forecast["model_artifact_refs"] == {
        "driver_prediction": selected[0].artifact_hash,
        "runup_move_prediction": selected[1].artifact_hash}
    assert inputs.forecast["artifact_hashes"] == request.model_artifact_refs[:2]
    expected_features = {"days_before_print": 3, "shared": 5.0, "move_x": 11.0,
                         "gate_x": 13.0, ("exact_x" if exact else "implied_x"): 7.0 if exact else 2.0}
    assert inputs.features["model_inputs"] == expected_features
    assert tuple(inputs.features["model_inputs"]) == tuple(sorted(expected_features))
    with no_fit_guard():
        for output, value in (("driver_prediction", expected), ("runup_move_prediction", 17.0)):
            executor = inputs.forecast["executors"][output]
            identity = selected[0 if output == "driver_prediction" else 1]
            assert executor.binding_id == identity.binding_id
            assert executor.predict(inputs.features["model_inputs"]) == {output: value}


@pytest.mark.parametrize("reverse,driver_keys", [
    (False, ("implied_t1:*", "runup_move:*")),
    (True, ("runup_move:*", "implied_t1:*")),
])
def test_declaration_order_controls_references_not_runup_residual_slot_meaning(
        tmp_path, monkeypatch, reverse, driver_keys):
    pools, pool_rows = {}, []
    for role in ("implied_t1", "runup_move"):
        pools[role], row = _staged_driver_pool(tmp_path, role=role, model_id=f"m-{role}-*")
        pool_rows.append(row)
    assert pools["implied_t1"].content_hash != pools["runup_move"].content_hash
    binding = _stage(tmp_path, [IMPLIED, MOVE, RUNUP_GATE], rows=pool_rows)
    spec = batch._RUNUP_SPEC
    if reverse:
        spec = replace(spec, drivers=spec.drivers[::-1])
    monkeypatch.setattr(batch, "_STRATEGY_REGISTRY", {"STR-RUNUP": spec})
    rows, refusals, event = _build(binding)
    assert refusals == ()
    request, inputs = rows[event.key]
    assert request.model_artifact_refs == tuple(
        binding.model_identity[key].artifact_hash for key in (*driver_keys, "gate:STR-RUNUP"))
    assert inputs.model["model_residual_artifacts"] == {
        "driver": pools["implied_t1"], "runup_move": pools["runup_move"]}
    assert inputs.model["model_residual_artifact_recipe"] == {
        slot: {"role": role, "model_id": f"m-{role}-*", "fold": None,
               "content_hash": pools[role].content_hash}
        for slot, role in (("driver", "implied_t1"), ("runup_move", "runup_move"))}


@pytest.mark.parametrize("missing,detail", [
    (IMPLIED, "driver:STR-RUNUP"), (MOVE, "runup_move:*"), (RUNUP_GATE, "gate:STR-RUNUP"),
])
def test_missing_role_refuses_only_its_row(tmp_path, monkeypatch, missing, detail):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [SIZE, GATE, *[role for role in (IMPLIED, MOVE, RUNUP_GATE) if role != missing]])
    runup, through = _event(), _event_inputs()
    rows, refusals = _assemble(binding, [runup, through], gate_policy=POLICY)
    assert list(rows) == [through.key]
    assert [(r.key, r.code, r.detail) for r in refusals] == [(runup.key, "RELEASE_MISSING_ROLE", detail)]


@pytest.mark.parametrize("missing,detail", [(MOVE, "runup_move:*"), (RUNUP_GATE, "gate:STR-RUNUP")])
def test_later_missing_dependency_precedes_earlier_driver_clock_mismatch(
        tmp_path, monkeypatch, missing, detail):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [role for role in (IMPLIED, MOVE, RUNUP_GATE) if role != missing])
    identities = dict(binding.model_identity)
    identities["implied_t1:*"] = replace(identities["implied_t1:*"], decision_clock_id="wrong")
    rows, refusals, event = _build(replace(binding, model_identity=identities))
    assert rows == {}
    assert [(r.key, r.code, r.detail) for r in refusals] == [(event.key, "RELEASE_MISSING_ROLE", detail)]


@pytest.mark.parametrize("role", ["implied_t1:*", "runup_move:*", "gate:STR-RUNUP"])
def test_every_required_role_must_share_the_decision_clock(tmp_path, monkeypatch, role):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [IMPLIED, MOVE, RUNUP_GATE])
    identities = dict(binding.model_identity)
    identities[role] = replace(identities[role], decision_clock_id="different")
    binding = replace(binding, model_identity=identities)
    monkeypatch.setattr(batch, "assemble_nightly_source_bundle", lambda **k: pytest.fail("clock mismatch built bundle"))
    rows, refusals, event = _build(binding)
    assert rows == {}
    expected_role = "runup_move:*" if role == "runup_move:*" else "driver:STR-RUNUP"
    clocks = ("entry-close", "different") if role == "gate:STR-RUNUP" else ("different", "entry-close")
    assert [(r.key, r.code, r.detail) for r in refusals] == [(
        event.key, "AMBIGUOUS_DECISION_CLOCK",
        f"{expected_role}={clocks[0]!r} gate:STR-RUNUP={clocks[1]!r}")]


@pytest.mark.parametrize("role", ["implied_t1:*", "runup_move:*", "gate:STR-RUNUP"])
def test_derived_feature_order_requires_every_role(tmp_path, monkeypatch, role):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [IMPLIED, MOVE, RUNUP_GATE])
    identities = dict(binding.model_identity)
    identities[role] = replace(identities[role], feature_order=())
    rows, refusals, event = _build(replace(binding, model_identity=identities))
    assert rows == {}
    assert [(r.key, r.code) for r in refusals] == [(event.key, "RELEASE_MISSING_FEATURE_ORDER")]


def test_explicit_features_keep_existing_override_and_executor_missing_feature_refusal(tmp_path, monkeypatch):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [IMPLIED, MOVE, RUNUP_GATE])
    rows, refusals, event = _build(binding, feature_names=("shared",))
    assert refusals == ()
    inputs = rows[event.key][1]
    assert inputs.features["model_inputs"] == {"shared": 5.0}
    with pytest.raises(FrozenStageRefusal) as error:
        inputs.forecast["executors"]["runup_move_prediction"].predict(inputs.features["model_inputs"])
    assert error.value.code == "MISSING_FEATURES"
    assert error.value.missing_features == ("move_x",)


@pytest.mark.parametrize("changes", [
    {"role": "size"}, {"output_names": ("left", "right")},
    {"output_names": ("runup_move_prediction",)}, {"output_names": ("unknown_raw",)},
])
def test_wrong_secondary_binding_is_sanitized_input_build_refusal(tmp_path, monkeypatch, changes):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [IMPLIED, MOVE, RUNUP_GATE])
    release = replace(binding.model_release, bindings=tuple(
        replace(item, **changes) if item.role == "runup_move" else item
        for item in binding.model_release.bindings))
    rows, refusals, event = _build(replace(binding, model_release=release))
    assert rows == {}
    assert [(r.key, r.code, r.detail) for r in refusals] == [(
        event.key, "NATIVE_INPUT_BUILD_FAILED", "native input build failed: ValueError")]


def test_changing_second_driver_artifact_changes_request_identity(tmp_path, monkeypatch):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path / "first", [IMPLIED, MOVE, RUNUP_GATE])
    rows, refusals, event = _build(binding)
    assert refusals == ()
    different_move = ("runup_move", "*", ("move_x",))
    changed_binding = _stage(tmp_path / "second", [IMPLIED, different_move, RUNUP_GATE])
    changed, refused, _ = _build(changed_binding, event)
    assert refused == ()
    before, after = rows[event.key][0], changed[event.key][0]
    assert before.model_artifact_refs[::2] == after.model_artifact_refs[::2]
    assert before.model_artifact_refs[1] != after.model_artifact_refs[1]
    assert request_hash(before) != request_hash(after)
    assert after.model_artifact_refs[1] == changed_binding.model_identity["runup_move:*"].artifact_hash


def test_through_preserves_public_driver_name_override(tmp_path):
    binding = _stage(tmp_path, [SIZE, GATE])
    rows, refusals = _assemble(binding, [_event_inputs()], driver_name="custom-driver")
    assert refusals == ()
    assert next(iter(rows.values()))[1].forecast["driver_name"] == "custom-driver"


def test_registered_runup_requires_verified_model_state(tmp_path):
    binding = _stage_without_model_state(tmp_path, [IMPLIED, MOVE, RUNUP_GATE])
    rows, refusals, event = _build(binding)
    assert rows == {}
    assert [(r.key, r.code) for r in refusals] == [(event.key, "MODEL_NOT_READY")]


def test_missing_runup_policy_refuses_before_bundle(tmp_path, monkeypatch):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [IMPLIED, MOVE, RUNUP_GATE])
    monkeypatch.setattr(batch, "assemble_nightly_source_bundle", lambda **k: pytest.fail("missing policy built bundle"))
    event = _event()
    rows, refusals = _assemble(binding, [event], gate_policy={"STR-THRU": {}})
    assert rows == {}
    assert [(r.key, r.code, r.detail) for r in refusals] == [(
        event.key, "GATE_POLICY_NOT_STAGED", "no gate policy staged for 'STR-RUNUP'")]


def test_bad_exact_driver_never_falls_back_to_healthy_alias(tmp_path, monkeypatch):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [EXACT, IMPLIED, MOVE, RUNUP_GATE])
    identities = dict(binding.model_identity)
    identities["driver:STR-RUNUP"] = replace(identities["driver:STR-RUNUP"], decision_clock_id="wrong")
    rows, refusals, event = _build(replace(binding, model_identity=identities))
    assert rows == {}
    assert [(r.key, r.code, r.detail) for r in refusals] == [(
        event.key, "AMBIGUOUS_DECISION_CLOCK", "driver:STR-RUNUP='wrong' gate:STR-RUNUP='entry-close'")]


def test_same_artifact_hash_deduplicates_request_without_dropping_either_output(tmp_path, monkeypatch):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [("implied_t1", "*", ("shared",)),
                               ("runup_move", "*", ("shared",)), RUNUP_GATE])
    rows, refusals, event = _build(binding)
    assert refusals == ()
    request, inputs = rows[event.key]
    driver_hash = binding.model_identity["implied_t1:*"].artifact_hash
    assert binding.model_identity["runup_move:*"].artifact_hash == driver_hash
    assert request.model_artifact_refs == (driver_hash, binding.model_identity["gate:STR-RUNUP"].artifact_hash)
    assert inputs.forecast["artifact_hashes"] == (driver_hash, driver_hash)
    assert set(inputs.forecast["executors"]) == {"driver_prediction", "runup_move_prediction"}
    for output, executor in inputs.forecast["executors"].items():
        assert executor.predict(inputs.features["model_inputs"]) == {output: 6.0}


def test_derived_secondary_feature_gap_is_not_defaulted_or_dropped(tmp_path, monkeypatch):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [IMPLIED, MOVE, RUNUP_GATE])
    event = _event()
    panel = dict(event.panel_row)
    del panel["move_x"]
    rows, refusals, event = _build(binding, replace(event, panel_row=panel))
    assert refusals == ()
    inputs = rows[event.key][1]
    assert inputs.features["missing_mask"]["move_x"] is True
    assert "move_x" not in inputs.features["model_inputs"]
    with pytest.raises(FrozenStageRefusal) as error:
        inputs.forecast["executors"]["runup_move_prediction"].predict(inputs.features["model_inputs"])
    assert (error.value.code, error.value.missing_features) == ("MISSING_FEATURES", ("move_x",))


def test_worker_executes_both_forecasts_but_refuses_unavailable_gate_features(tmp_path, monkeypatch):
    _enable_test_runup(monkeypatch)
    binding = _stage(tmp_path, [IMPLIED, MOVE, RUNUP_GATE])
    event = _event()
    document = _event_doc(strategy="STR-RUNUP")
    document.update(calendar_row=dict(event.calendar_row), panel_row=dict(event.panel_row),
                    quote_rows=event.quote_rows)
    root = tmp_path / "work"
    root.mkdir()
    (root / "events.json").write_text(json.dumps([document]))
    parameters = _worker_parameters(tmp_path, expected_ids=["evt-1"], feature_names=[])
    parameters["gate_policy"] = POLICY
    batch.run_native_score_batch_worker(parameters, root)
    refused = json.loads((root / "refusals.json").read_text())
    assert refused["refusals"] == {}
    records = json.loads((root / "records.json").read_text())
    assert len(records["records"]) == 1
    record = next(iter(records["records"].values()))
    assert record["readiness"] == "refused"
    assert "MISSING_GATE_INPUT:forecast" in record["reason_codes"]
    assert "MISSING_SIMULATION_INPUT" not in record["reason_codes"]
    assert record["forecasts"]["driver_prediction"] == 11.0
    assert record["forecasts"]["runup_move_raw_d14"] == 17.0
    assert record["forecasts"]["runup_move_prediction"] == pytest.approx(51.0 / 14.0)
    roles = ("implied_t1:*", "runup_move:*", "gate:STR-RUNUP")
    assert record["model_artifact_ids"] == [binding.model_identity[role].artifact_hash for role in roles]
    assert record["resolved_request"]["model_identity"] == {
        role: json.loads(json.dumps(asdict(binding.model_identity[role]))) for role in roles}
