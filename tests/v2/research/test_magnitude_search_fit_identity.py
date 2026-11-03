"""Pin the fit identity hashes produced by the current evidence assembly."""
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

    assert len(evidence) == 1
    assert evidence[0]["training_matrix_id"] == (
        "sha256:bfd4b7d466045c69589d99da95da03320a7e34c1c4f0f904e8f887a5c41590c8"
    )
    assert evidence[0]["prediction_matrix_id"] == (
        "sha256:0125b449b8bfce98cb25585176d9704b3eda2d13c83f290658a55720c535b36a"
    )
