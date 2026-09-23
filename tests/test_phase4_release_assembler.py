import copy
from dataclasses import replace

import pytest

from checks import phase4_real
from engine.v2.contracts import SCORE_REQUEST_V1
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring.stages import StageObservation, receipt
from tools.phase4_release_assembler import (
    NATIVE_INPUT_KEYS,
    OPTIONAL_STAGES,
    REQUIRED_STAGES,
    ReleaseAssemblyError,
    assemble_input_trace,
)
from tools.phase4_request_translation import canonical_request_from_legacy


#: A pair's saved LEGACY request (payload.request); ``_request()`` is its
#: canonical V2 translation, which the strict verifier re-derives.
LEGACY_REQUEST = {
    "ticker": "ABC", "strategy": "STR-THRU", "as_of": "2026-09-16",
    "event_date": "2026-09-17", "session": "AMC",
    "fill": {"policy_id": "legacy.fill_alpha.v1", "alpha": 0.5},
}


def _request():
    bound = to_document(canonical_request_from_legacy(
        LEGACY_REQUEST, event_id="event-1", snapshot="snapshot-1",
    ))
    return {
        "event_id": "event-1",
        "calendar_revision": bound["calendar_revision"],
        "strategy_version": "STR-THRU",
        "deployment_id": "deployment-1",
        "decision_clock_id": "entry-close-1",
        "requested_decision_at": "2026-09-16",
        "snapshot_id": "snapshot-1",
        "mode": "replay",
        "fill_model": bound["fill_model"],
        "event_revision": bound["event_revision"],
        "contract_override": None,
        "geometry_override": None,
        "dependency_refs": [],
        "model_artifact_refs": [],
        "residual_state_ref": None,
        "analog_state_ref": None,
        "calibration_state_ref": None,
        "schema_version": SCORE_REQUEST_V1,
    }


def _native_inputs(request):
    inputs = {
        "context": {"strategy": "STR-THRU"},
        "features": {"model_inputs": {"spot": 100.0}},
        "forecast": {
            "models": {
                "driver_prediction": {
                    "intercept": 0.0,
                    "coefficients": {"spot": 0.01},
                },
            },
            "model_artifact_refs": {"driver_prediction": "model:driver"},
        },
        "geometry": None,
        "pricing": None,
        "model": {},
        "analogs": {},
        "simulation": {},
        "gate": {},
        "chooser": {},
        "diagnostics": {},
        "source_ref": "pending",
    }
    shared = {
        "request": copy.deepcopy(request),
        "native_inputs": {
            key: copy.deepcopy(value)
            for key, value in inputs.items()
            if key != "source_ref"
        },
    }
    inputs["source_ref"] = content_hash(shared)
    return inputs, shared


def _observations(*, include_model=False):
    stages = (*REQUIRED_STAGES[:-1], "diagnostics", REQUIRED_STAGES[-1])
    if include_model:
        # "model" executes between "pricing" and "analogs" in
        # engine.v2.scoring.stages.STAGE_NAMES; insert it there so a
        # captured observation sequence looks like a real one.
        pricing_index = stages.index("pricing")
        stages = (
            *stages[:pricing_index + 1], "model", *stages[pricing_index + 1:],
        )
    rows = []
    for stage in stages:
        inputs = {"stage": stage, "side": "input"}
        output = {"stage": stage, "side": "output"}
        rows.append(StageObservation(inputs, output, receipt(stage, inputs, output)))
    return rows


def _assembled():
    request = _request()
    native_inputs, shared_inputs = _native_inputs(request)
    trace = assemble_input_trace(
        request=request,
        shared_inputs=shared_inputs,
        native_inputs=native_inputs,
        observations=_observations(),
        resources=[],
    )
    return request, native_inputs, shared_inputs, trace


def test_assembled_trace_passes_strict_verifier(tmp_path):
    request, native_inputs, shared_inputs, trace = _assembled()
    pair = {
        "payload": {
            "request": copy.deepcopy(LEGACY_REQUEST),
            "record": {},
            "legacy_input_hash": content_hash(shared_inputs),
            "input_trace": trace,
            "input_trace_hash": trace["trace_hash"],
        },
    }

    verified = phase4_real._verified_trace_bundle(pair, tmp_path)

    assert set(trace["native_inputs"]) == NATIVE_INPUT_KEYS
    assert tuple(trace["stages"]) == REQUIRED_STAGES
    assert verified["inputs"].source_ref == content_hash(shared_inputs)
    assert verified["native_input_hash"] == content_hash(native_inputs)
    assert all(
        row["native_path"] != ["native_inputs", "source_ref"]
        for row in trace["input_translation"]["mappings"]
    )


def test_model_stage_is_known_and_validated_when_present(tmp_path):
    # A fresh capture always runs the payoff-calibration/model stage, so its
    # observation must be accepted -- not refused as "unknown" -- and its
    # real hash/owner evidence checked exactly like every other stage.
    request = _request()
    native_inputs, shared_inputs = _native_inputs(request)
    trace = assemble_input_trace(
        request=request,
        shared_inputs=shared_inputs,
        native_inputs=native_inputs,
        observations=_observations(include_model=True),
        resources=[],
    )

    assert "model" in trace["stages"]
    assert set(trace["stages"]) == set(REQUIRED_STAGES) | set(OPTIONAL_STAGES)

    pair = {
        "payload": {
            "request": copy.deepcopy(LEGACY_REQUEST),
            "record": {},
            "legacy_input_hash": content_hash(shared_inputs),
            "input_trace": trace,
            "input_trace_hash": trace["trace_hash"],
        },
    }
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    assert "model" in {row["stage"] for row in verified["captured_receipts"]}


def test_model_stage_absence_is_not_an_error_for_an_old_capture():
    # A trace captured before the model stage existed never recorded it,
    # and the trace schema carries no version field to key a requirement
    # on -- so a trace with no "model" observation at all must keep working
    # exactly as before.
    request, native_inputs, shared_inputs, trace = _assembled()
    assert "model" not in trace["stages"]
    assert tuple(trace["stages"]) == REQUIRED_STAGES


def test_a_genuinely_unknown_stage_is_still_refused():
    request = _request()
    native_inputs, shared_inputs = _native_inputs(request)
    observations = list(_observations())
    bogus = {"stage": "bogus", "side": "input"}
    bogus_out = {"stage": "bogus", "side": "output"}
    observations.append(
        StageObservation(
            bogus, bogus_out,
            replace(receipt(REQUIRED_STAGES[0], bogus, bogus_out), stage="bogus"),
        )
    )

    with pytest.raises(ReleaseAssemblyError, match="unknown native observation stage"):
        assemble_input_trace(
            request=request,
            shared_inputs=shared_inputs,
            native_inputs=native_inputs,
            observations=observations,
            resources=[],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.pop("diagnostics"), "missing=.*diagnostics"),
        (lambda value: value.update({"unexpected": {}}), "unknown=.*unexpected"),
    ),
)
def test_native_input_keys_must_match_verifier_exactly(mutation, message):
    request = _request()
    native_inputs, shared_inputs = _native_inputs(request)
    mutation(native_inputs)

    with pytest.raises(ReleaseAssemblyError, match=message):
        assemble_input_trace(
            request=request,
            shared_inputs=shared_inputs,
            native_inputs=native_inputs,
            observations=_observations(),
            resources=[],
        )


def test_translation_requires_complete_leaf_coverage():
    request = _request()
    native_inputs, shared_inputs = _native_inputs(request)

    with pytest.raises(ReleaseAssemblyError, match="leaf coverage mismatch"):
        assemble_input_trace(
            request=request,
            shared_inputs=shared_inputs,
            native_inputs=native_inputs,
            observations=_observations(),
            resources=[],
            mappings=[{
                "shared_path": ["request", "event_id"],
                "native_path": ["request", "event_id"],
            }],
        )


def test_stage_receipt_hash_must_match_observed_content():
    request = _request()
    native_inputs, shared_inputs = _native_inputs(request)
    observations = _observations()
    observations[0] = replace(
        observations[0],
        input_document={"stage": "resolve_context", "tampered": True},
    )

    with pytest.raises(ReleaseAssemblyError, match="input hash mismatch"):
        assemble_input_trace(
            request=request,
            shared_inputs=shared_inputs,
            native_inputs=native_inputs,
            observations=observations,
            resources=[],
        )


def test_calculated_answers_are_rejected_but_forecast_roles_are_allowed():
    request = _request()
    native_inputs, shared_inputs = _native_inputs(request)
    native_inputs["context"]["nested"] = {"entry_cost": 12.5}

    with pytest.raises(ReleaseAssemblyError, match="calculated answer"):
        assemble_input_trace(
            request=request,
            shared_inputs=shared_inputs,
            native_inputs=native_inputs,
            observations=_observations(),
            resources=[],
        )


def test_role_model_input_feature_named_like_an_answer_is_allowed():
    request = _request()
    inputs, shared = _native_inputs(request)
    inputs["features"]["role_model_inputs"] = {
        "gate": {"entry_cost_pct": 4.2, "spot": 100.0},
    }
    shared["native_inputs"]["features"] = copy.deepcopy(inputs["features"])
    inputs["source_ref"] = content_hash(shared)

    trace = assemble_input_trace(
        request=request,
        shared_inputs=shared,
        native_inputs=inputs,
        observations=_observations(),
        resources=[],
    )

    role_gate = trace["native_inputs"]["features"]["role_model_inputs"]["gate"]
    assert role_gate["entry_cost_pct"] == 4.2


def test_answer_field_directly_under_features_is_still_rejected():
    request = _request()
    inputs, shared = _native_inputs(request)
    inputs["features"]["entry_cost_pct"] = 4.2
    shared["native_inputs"]["features"] = copy.deepcopy(inputs["features"])
    inputs["source_ref"] = content_hash(shared)

    with pytest.raises(ReleaseAssemblyError, match="calculated answer"):
        assemble_input_trace(
            request=request,
            shared_inputs=shared,
            native_inputs=inputs,
            observations=_observations(),
            resources=[],
        )


#: R4-checkpoint gap: ``driver_p10``/``driver_p90`` (the driver/implied_t1
#: quantile band, legacy ``engine/score.py:2874-2875``/``3053-3054``) and
#: ``win_model_raw`` (the payoff-model layer's raw win rate,
#: ``engine/score.py:2914``/``3087``) are three fields legacy publishes that
#: no stage in ``engine/v2/scoring/stages.py`` has ever owned or computed --
#: unlike their siblings (``forecast_p10``/``forecast_p90``/``forecast_sd``,
#: ``exp_pnl_model``/``win_model``), which were already in
#: ``source_inputs._ANSWER_FIELDS`` and so already refused if smuggled in as
#: a captured legacy value. These three were not: a capture could carry
#: legacy's own number for them straight into ``native_inputs.context`` (or
#: ``.features``) and it would pass through unfiltered
#: (``stages._merge_stage`` only strips names in ``_OWNED_OUTPUTS``, and
#: none of these three is), making a Phase 4 row "agree" on a field native
#: never actually computed. Each parametrized field below used to pass
#: silently through this exact path before being added to
#: ``_ANSWER_FIELDS``; this test plants it as a captured answer under
#: ``context`` (mirroring ``test_answer_field_directly_under_features_is_
#: still_rejected``'s pattern for ``features``) and asserts the assembler
#: now refuses it as a calculated answer.
@pytest.mark.parametrize("field", ["driver_p10", "driver_p90", "win_model_raw"])
def test_previously_unprotected_answer_fields_are_now_rejected(field):
    request = _request()
    inputs, shared = _native_inputs(request)
    inputs["context"][field] = 1.23
    shared["native_inputs"]["context"] = copy.deepcopy(inputs["context"])
    inputs["source_ref"] = content_hash(shared)

    with pytest.raises(ReleaseAssemblyError, match="calculated answer"):
        assemble_input_trace(
            request=request,
            shared_inputs=shared,
            native_inputs=inputs,
            observations=_observations(),
            resources=[],
        )
