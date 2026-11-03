"""Pin the fit identity hashes produced by the current evidence assembly."""
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments.magnitude_search_evidence import fit_evidence


def test_fit_evidence_pins_matrix_identity_hashes():
    inputs = {
        "features": [
            {"event_id": "train", "context_id": "ctx-t", "horizon_sessions": 5, "x": 1.25},
            {"event_id": "eval", "context_id": "ctx-e", "horizon_sessions": 5, "x": 2.5},
        ],
        "training": [{"event_id": "train", "absolute_move_pct": 0.75}],
    }
    fold = {"evaluation_event_ids": ["eval"]}
    plan = {"horizons": [5]}
    execution = {"features": ["x"]}

    evidence = fit_evidence(inputs, fold, plan, execution)

    assert evidence == [{
        "horizon_sessions": 5,
        "training_event_ids": ["train"],
        "evaluation_event_ids": ["eval"],
        "training_matrix_id": (
            "sha256:bfd4b7d466045c69589d99da95da03320a7e34c1c4f0f904e8f887a5c41590c8"
        ),
        "prediction_matrix_id": (
            "sha256:0125b449b8bfce98cb25585176d9704b3eda2d13c83f290658a55720c535b36a"
        ),
    }]

    inputs["features"][1]["x"] = 3.5
    changed = fit_evidence(inputs, fold, plan, execution)
    assert changed[0]["prediction_matrix_id"] != evidence[0]["prediction_matrix_id"]


def test_fit_evidence_dict_inputs_include_event_dates():
    inputs = {
        "features": [
            {"event_id": "train", "context_id": "ctx-t", "horizon_sessions": 5, "x": 1.25,
             "event_date": "2024-01-02"},
            {"event_id": "eval", "context_id": "ctx-e", "horizon_sessions": 5, "x": 2.5},
        ],
        "training": [{"event_id": "train", "absolute_move_pct": 0.75}],
    }
    fold = {"evaluation_event_ids": ["eval"]}
    plan = {"horizons": [5]}
    execution = {"features": ["x"], "fit_recipe": {"requires_event_dates": True}}

    evidence = fit_evidence(inputs, fold, plan, execution)

    assert len(evidence) == 1
    assert evidence[0]["training_matrix_id"] == content_hash({
        "columns": ["x"], "row_keys": [["train", "ctx-t", 5]],
        "values": [[1.25]], "absolute_move_pct": [0.75], "event_dates": ["2024-01-02"],
    })
    assert evidence[0]["prediction_matrix_id"] == (
        "sha256:0125b449b8bfce98cb25585176d9704b3eda2d13c83f290658a55720c535b36a"
    )


@pytest.mark.parametrize("event_date, message", [
    (None, "search feature event_date is missing or not a string"),
    ("2024-02-30", "search feature event_date is malformed"),
    ("2024-1-2", "search feature event_date is not canonical ISO"),
])
def test_fit_evidence_refuses_missing_or_malformed_opted_in_dates(event_date, message):
    feature = {"event_id": "train", "context_id": "ctx-t", "horizon_sessions": 5, "x": 1.25}
    if event_date is not None:
        feature["event_date"] = event_date
    inputs = {
        "features": [feature],
        "training": [{"event_id": "train", "absolute_move_pct": 0.75}],
    }
    execution = {"features": ["x"], "fit_recipe": {"requires_event_dates": True}}

    with pytest.raises(OpsError) as caught:
        fit_evidence(inputs, {"evaluation_event_ids": []}, {"horizons": [5]}, execution)

    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", message).problem
