"""Native runup recipes accept raw D14 points, never pretransformed answers.

Hash-verified JSON-linear releases exercise the real source builder and recipe
executor without fitting models or reading production data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.models import (
    ArtifactMember,
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
)
from engine.v2.scoring import application
from engine.v2.scoring.frozen_executor import FrozenRecipeExecutor
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs

RAW_OUTPUTS = (
    "prediction", "pred_runup_abs_move_d14", "runup_move_d14", "runup_move_raw_d14",
)
FINAL_OUTPUTS = (
    "runup_move_prediction", "runup_move_p10", "runup_move_p90", "runup_move_sd",
)
RUNUP_TARGETS = ("runup_move_prediction", "pred_runup_abs_move_d14")
CHOOSER_PRODUCERS = ("pred_im_t1_d14", "pred_runup_abs_move_d14")


def _binding(root, output_names, role, binding_id="producer"):
    document = {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [
            {"name": name, "intercept": 3.0 + index, "coefficients": [2.0]}
            for index, name in enumerate(output_names)
        ],
    }
    raw = json.dumps(document, sort_keys=True).encode()
    path = root / f"{binding_id}.json"
    path.write_bytes(raw)
    return ModelBinding(
        binding_id=binding_id, model_id=f"synthetic-{role}", role=role,
        strategy_id="*", decision_clock_id="entry-close", adapter="json-linear.v1",
        feature_order=("x",), output_names=tuple(output_names),
        members=(ArtifactMember(
            name="estimator", path=path.name,
            content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
        ),),
    )


def _bundle(root, output_names, target, *, declared=None, role="runup_move"):
    binding = _binding(root, output_names, role)
    bindings = [binding]
    recipe = {"binding_id": binding.binding_id}
    if declared is not None:
        recipe["output"] = declared
    strategy = "STR-THRU"
    forecast = {"driver_prediction": {"intercept": 2.0}}
    chooser = {}
    gate = {"model": {"intercept": 1.0}, "threshold": 0.5}
    if target in CHOOSER_PRODUCERS or target == "chooser_score":
        strategy = "TWIN-P"
        forecast = {"forecast_abs_move": {"intercept": 2.0}}
        if target == "chooser_score":
            chooser = recipe
        else:
            champion = _binding(root, ("chooser_score",), "chooser", "champion")
            bindings.append(champion)
            chooser = {
                "binding_id": champion.binding_id, "producers": {target: recipe},
            }
    elif target == "gate_score":
        gate = {**recipe, "threshold": 0.5}
    else:
        if target == "runup_move_prediction":
            strategy = "STR-RUNUP"
        forecast[target] = recipe
    return SourceBundle(
        source_ref="synthetic:raw-runup", strategy=strategy,
        context={"ticker": "AAA", "event_date": "2026-09-16", "spot": 100.0,
                 "entry_date": "2026-09-02", "exit_date": "2026-09-15",
                 "expiry": "2026-09-18", "days_before_print": 7.0},
        raw_quotes={("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 1.2}},
        feature_vector={"x": 3.0}, feature_missing_mask={"x": False},
        model_identity={}, forecast_recipes=forecast,
        model_artifact_refs={name: f"synthetic:{name}" for name in forecast},
        residual_recipe={}, analog_recipe={}, gate_recipe=gate, chooser_recipe=chooser,
        frozen_inference=FrozenInference(root),
        model_release=ModelRelease(
            release_id="synthetic-release", deployment_id="synthetic-deployment",
            bindings=tuple(bindings),
        ),
    )


def _executor(bundle, target):
    inputs = build_native_score_inputs(bundle)
    if target in CHOOSER_PRODUCERS:
        executor = inputs.chooser["producers"][target]
    elif target == "chooser_score":
        executor = inputs.chooser["executors"][target]
    elif target == "gate_score":
        executor = inputs.gate["executors"][target]
    else:
        executor = inputs.forecast["executors"][target]
    assert isinstance(executor, FrozenRecipeExecutor)
    return executor


@pytest.mark.parametrize("target", RUNUP_TARGETS)
@pytest.mark.parametrize("output", RAW_OUTPUTS)
@pytest.mark.parametrize("explicit", (False, True), ids=("implicit", "explicit"))
def test_raw_aliases_execute_without_scaling_or_renaming_leaks(tmp_path, target, output, explicit):
    bundle = _bundle(tmp_path, (output,), target, declared=output if explicit else None)
    executor = _executor(bundle, target)

    # Seven days is half D14, but recipe executors must return the raw point.
    assert executor.predict(bundle.feature_vector) == {target: 9.0}
    assert executor.target == target
    assert executor.role == "runup_move"
    assert executor.artifact_hashes == (
        bundle.model_release.bindings[0].members[0].content_hash,
    )


@pytest.mark.parametrize("target", RUNUP_TARGETS)
@pytest.mark.parametrize("output", FINAL_OUTPUTS)
@pytest.mark.parametrize("explicit", (False, True), ids=("implicit", "explicit"))
def test_final_names_are_rejected_at_build_time(tmp_path, target, output, explicit):
    bundle = _bundle(tmp_path, (output,), target, declared=output if explicit else None)

    with pytest.raises(ValueError, match="pretransformed forecast output"):
        build_native_score_inputs(bundle)


@pytest.mark.parametrize("target", RUNUP_TARGETS)
@pytest.mark.parametrize("final", FINAL_OUTPUTS)
@pytest.mark.parametrize("raw_first", (False, True))
def test_final_sibling_is_rejected_even_with_explicit_raw_selection(
    tmp_path, target, final, raw_first,
):
    names = ("prediction", final) if raw_first else (final, "prediction")
    bundle = _bundle(tmp_path, names, target, declared="prediction")

    with pytest.raises(ValueError, match="pretransformed forecast output"):
        build_native_score_inputs(bundle)


@pytest.mark.parametrize("target", RUNUP_TARGETS)
@pytest.mark.parametrize("explicit", (False, True), ids=("implicit", "explicit"))
def test_unknown_singleton_is_not_assumed_to_be_a_raw_point(tmp_path, target, explicit):
    name = "unrecognized_d14"
    bundle = _bundle(tmp_path, (name,), target, declared=name if explicit else None)

    with pytest.raises(ValueError, match="must select a raw D14 point output"):
        build_native_score_inputs(bundle)


@pytest.mark.parametrize("target", RUNUP_TARGETS)
@pytest.mark.parametrize("suffix", ("p10", "p90", "sd"))
def test_explicit_raw_band_selection_cannot_stand_in_for_point(tmp_path, target, suffix):
    band = f"runup_move_raw_d14_{suffix}"
    bundle = _bundle(tmp_path, ("prediction", band), target, declared=band)

    with pytest.raises(ValueError, match="must select a raw D14 point output"):
        build_native_score_inputs(bundle)


@pytest.mark.parametrize("target", RUNUP_TARGETS)
@pytest.mark.parametrize("output", RAW_OUTPUTS)
def test_explicit_raw_selection_in_multi_output_binding_executes(tmp_path, target, output):
    bundle = _bundle(tmp_path, ("runup_move_raw_d14_p10", output), target, declared=output)

    assert _executor(bundle, target).predict(bundle.feature_vector) == {target: 10.0}


def test_chooser_target_retains_implicit_selection_in_multi_output_binding(tmp_path):
    target = "pred_runup_abs_move_d14"
    bundle = _bundle(tmp_path, ("runup_move_raw_d14_p10", target), target)

    assert _executor(bundle, target).predict(bundle.feature_vector) == {target: 10.0}


@pytest.mark.parametrize("target", RUNUP_TARGETS)
def test_raw_name_must_still_be_declared_by_binding(tmp_path, target):
    bundle = _bundle(tmp_path, ("prediction",), target, declared="runup_move_d14")

    with pytest.raises(ValueError, match="is not in"):
        build_native_score_inputs(bundle)


@pytest.mark.parametrize("target", RUNUP_TARGETS)
def test_role_suffix_does_not_bypass_final_output_rejection(tmp_path, target):
    bundle = _bundle(
        tmp_path, ("prediction", "runup_move_sd"), target,
        declared="prediction", role="runup_move:STR-RUNUP",
    )

    with pytest.raises(ValueError, match="pretransformed forecast output"):
        build_native_score_inputs(bundle)


@pytest.mark.parametrize(("target", "role"), (
    ("driver_prediction", "size"), ("driver_prediction", "implied_t1"),
    ("pred_iv_crush", "iv_crush"), ("model_fair_pct", "fair_value"),
    ("gate_score", "gate"), ("chooser_score", "chooser"),
    ("pred_im_t1_d14", "implied_t1"),
))
@pytest.mark.parametrize("output", ("unrecognized_d14", "runup_move_prediction"))
def test_other_roles_keep_singleton_output_mapping(tmp_path, target, role, output):
    bundle = _bundle(tmp_path, (output,), target, role=role)

    assert _executor(bundle, target).predict(bundle.feature_vector) == {target: 9.0}


def _historical_score(root, output):
    bundle = _bundle(root, (output,), "runup_move_prediction")
    # Historical callers already carry native inputs, rather than resolving a
    # source-bound recipe; their singleton compatibility must remain separate.
    inputs = build_native_score_inputs(replace(bundle, forecast_recipes={
        "driver_prediction": {"intercept": 2.0},
        "runup_move_prediction": {"intercept": 0.0},
    }))
    release = bundle.model_release
    request = ScoreRequest(
        event_id="synthetic-event", calendar_revision="synthetic-calendar",
        strategy_version="STR-RUNUP", deployment_id=release.deployment_id,
        decision_clock_id="entry-close", requested_decision_at="2026-09-02",
        snapshot_id="synthetic-snapshot", mode="replay", fill_model={"alpha": 0.5},
    )
    inference_request = InferenceRequest(
        release_id=release.release_id, binding_id="producer",
        feature_order=("x",), rows=((3.0,),),
    )
    return application.score_frozen(
        request, bundle.frozen_inference, release, inference_request,
        {"_native_inputs": inputs},
    )


@pytest.mark.parametrize("output", (*RAW_OUTPUTS, "unrecognized_d14"))
def test_historical_score_frozen_retains_raw_and_unknown_singleton_compatibility(tmp_path, output):
    record = _historical_score(tmp_path, output)

    assert record.forecasts["runup_move_raw_d14"] == 9.0
    assert record.forecasts["runup_move_prediction"] == 4.5
    assert "PRETRANSFORMED_FROZEN_OUTPUT:runup_move" not in record.reason_codes


@pytest.mark.parametrize("output", FINAL_OUTPUTS)
def test_historical_score_frozen_still_refuses_final_outputs(tmp_path, output):
    record = _historical_score(tmp_path, output)

    assert record.forecasts["runup_move_prediction"] is None
    assert "PRETRANSFORMED_FROZEN_OUTPUT:runup_move" in record.reason_codes
