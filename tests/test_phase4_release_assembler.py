import copy
from dataclasses import replace

import pytest

from checks import phase4_real
from engine.v2.contracts import SCORE_REQUEST_V1
from engine.v2.foundation import content_hash
from engine.v2.scoring.stages import StageObservation, receipt
from tools.phase4_release_assembler import (
    NATIVE_INPUT_KEYS,
    REQUIRED_STAGES,
    ReleaseAssemblyError,
    assemble_input_trace,
)


def _request():
    return {
        "event_id": "event-1",
        "calendar_revision": "calendar-1",
        "strategy_version": "STR-THRU",
        "deployment_id": "deployment-1",
        "decision_clock_id": "entry-close-1",
        "requested_decision_at": "2026-09-16",
        "snapshot_id": "snapshot-1",
        "mode": "replay",
        "fill_model": {"alpha": 0.5},
        "event_revision": "event-revision-1",
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


def _observations():
    rows = []
    for stage in (*REQUIRED_STAGES[:-1], "diagnostics", REQUIRED_STAGES[-1]):
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
            "request": request,
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
