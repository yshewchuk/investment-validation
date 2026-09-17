from dataclasses import replace

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.foundation import DocumentError, from_document, to_document
from engine.v2.scoring.identity import request_hash


def request(**changes):
    value = ScoreRequest(
        event_id="evt-1", event_revision="rev-1", calendar_revision="cal-1",
        strategy_version="STR-THRU", deployment_id="dep-1",
        decision_clock_id="entry-close", requested_decision_at="2026-09-16",
        snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5, "policy_id": "fill.v1"},
        geometry_override={"strike": 10.123456789},
        model_artifact_refs=("model-a",), residual_state_ref="res-a",
    )
    return replace(value, **changes)


def test_request_round_trip_preserves_null_zero_and_empty_tuple():
    value = request(contract_override=None, geometry_override={"width": 0.0},
                    dependency_refs=())
    assert from_document(ScoreRequest, to_document(value)) == value
    assert value.contract_override is None
    assert value.geometry_override["width"] == 0.0
    assert value.dependency_refs == ()


def test_identity_changes_for_all_numerical_dependencies_but_not_schema_version():
    baseline = request()
    assert request_hash(baseline) != request_hash(replace(baseline, geometry_override={"strike": 10.123457}))
    assert request_hash(baseline) != request_hash(replace(baseline, decision_clock_id="d-1"))
    assert request_hash(baseline) != request_hash(replace(baseline, fill_model={"alpha": 0.0}))
    assert request_hash(baseline) != request_hash(replace(baseline, residual_state_ref="res-b"))
    assert request_hash(baseline) == request_hash(replace(baseline, schema_version="score_request.v1.0"))


def test_contract_decoder_refuses_unknown_field_and_bad_version():
    document = to_document(request())
    document["unexpected"] = True
    with pytest.raises(DocumentError) as error:
        from_document(ScoreRequest, document)
    assert error.value.code == "UNKNOWN_FIELD"
    document = to_document(request())
    document["schema_version"] = "score_request.v2.0"
    with pytest.raises(DocumentError) as error:
        from_document(ScoreRequest, document)
    assert error.value.code == "UNSUPPORTED_VERSION"
