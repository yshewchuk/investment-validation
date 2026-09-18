import hashlib
import json

import pytest

from checks.phase4_frozen_bridge import (
    FROZEN_RELEASE_SCHEMA,
    FROZEN_TRACE_SCHEMA,
    FrozenBridgeError,
    prepare_frozen_replay,
)
from engine.v2.contracts import ScoreRequest
from engine.v2.scoring.stages import NativeScoreInputs, STAGE_NAMES, receipt


def _case(tmp_path):
    estimator = {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{
            "name": "forecast_abs_move",
            "intercept": 1.0,
            "coefficients": [2.0],
        }],
    }
    raw = json.dumps(estimator, sort_keys=True).encode()
    (tmp_path / "estimator.json").write_bytes(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    release = {
        "schema_version": FROZEN_RELEASE_SCHEMA,
        "release_id": "release-1",
        "deployment_id": "deployment-1",
        "bindings": [{
            "binding_id": "size-1",
            "model_id": "size-model-1",
            "request_ref": "model:size:1",
            "role": "size",
            "strategy_id": "STR-THRU",
            "decision_clock_id": "entry-close",
            "adapter": "json-linear.v1",
            "feature_order": ["x"],
            "output_names": ["forecast_abs_move"],
            "members": [{"name": "estimator", "resource_id": "size-estimator"}],
        }],
    }
    resources = [
        {
            "resource_id": "frozen-release",
            "ref": "model:size:1",
            "kind": "sidecar",
            "path": "release.json",
        },
        {
            "resource_id": "size-estimator",
            "ref": "artifact:size:1",
            "kind": "artifact",
            "path": "estimator.json",
            "sha256": digest,
        },
    ]
    metadata = {
        "frozen_inference": {
            "schema_version": FROZEN_TRACE_SCHEMA,
            "release_resource_id": "frozen-release",
            "binding_ids": ["size-1"],
        },
    }
    request = ScoreRequest(
        event_id="event-1",
        event_revision="event-revision-1",
        calendar_revision="calendar-1",
        strategy_version="STR-THRU",
        deployment_id="deployment-1",
        decision_clock_id="entry-close",
        requested_decision_at="2026-09-16",
        snapshot_id="snapshot-1",
        mode="replay",
        fill_model={"alpha": 0.5},
        model_artifact_refs=("model:size:1",),
    )
    inputs = NativeScoreInputs(
        context={"strategy": "STR-THRU"},
        features={"model_inputs": {"x": 3.0}},
        forecast={},
        geometry=None,
        pricing=None,
        analogs={},
        simulation={},
        gate={},
        chooser={},
        diagnostics={},
        source_ref="sha256:" + "1" * 64,
        stage_receipts=tuple(
            receipt(stage, {"source": "test"}, {"execution": "pending"})
            for stage in STAGE_NAMES if stage != "diagnostics"
        ),
    )
    return resources, {"frozen-release": release}, metadata, request, inputs


def _prepare(tmp_path):
    resources, documents, metadata, request, inputs = _case(tmp_path)
    return prepare_frozen_replay(
        release_root=tmp_path,
        resource_rows=resources,
        verified_documents=documents,
        metadata=metadata,
        request=request,
        inputs=inputs,
    )


def test_verified_resources_construct_frozen_inference_request(tmp_path):
    plan = _prepare(tmp_path)

    assert plan is not None
    assert plan.requests[0].rows == ((3.0,),)
    result = plan.inference.infer(plan.release, plan.requests[0])
    assert result.status == "READY"
    assert result.predictions == ((7.0,),)
    assert plan.receipt.startswith("sha256:")


def test_calculated_forecast_answer_is_rejected(tmp_path):
    resources, documents, metadata, request, inputs = _case(tmp_path)
    inputs = NativeScoreInputs(
        **{
            **inputs.__dict__,
            "forecast": {"forecast_abs_move": 999.0},
        }
    )

    with pytest.raises(FrozenBridgeError, match="calculated answers"):
        prepare_frozen_replay(
            release_root=tmp_path,
            resource_rows=resources,
            verified_documents=documents,
            metadata=metadata,
            request=request,
            inputs=inputs,
        )


def test_artifact_byte_change_is_rejected_before_inference(tmp_path):
    resources, documents, metadata, request, inputs = _case(tmp_path)
    (tmp_path / "estimator.json").write_bytes(b"corrupt")

    with pytest.raises(FrozenBridgeError, match="sha256: mismatch"):
        prepare_frozen_replay(
            release_root=tmp_path,
            resource_rows=resources,
            verified_documents=documents,
            metadata=metadata,
            request=request,
            inputs=inputs,
        )
