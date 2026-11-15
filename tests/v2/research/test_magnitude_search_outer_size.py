"""Outer terminals bind lineage by reference instead of inlining every row."""
from experiments import magnitude_search_outer as outer
from experiments.magnitude_search_records import MAX_BYTES
from engine.v2.foundation import canonical_json

LINEAGE_ROWS = 120_000


def lineage_row(index):
    digest = f"{index:064x}"
    return {"event_id": f"SYNTH:{index:06d}", "fragment_id": f"fragment-{index:06d}",
            "object_id": "sha256:" + digest, "content_hash": "sha256:" + digest[::-1]}


def test_terminal_omits_lineage_and_stays_within_the_manifest_cap(monkeypatch):
    inputs = {"preflight": {}, "training_inputs": {
        "schema_version": "research_magnitude_training_targets.v1",
        "read_inputs": [lineage_row(index) for index in range(LINEAGE_ROWS)]}}
    validation = {"evaluation_inputs": {"schema_version": "research_magnitude_targets.v1",
                    "labels_before": "2025-02-01"},
        "targets": [{"event_id": "SYNTH:000000", "absolute_move_pct": 2.0}]}
    forecasts = [{"event_id": "SYNTH:000000", "horizon_sessions": 1,
                  "predicted_absolute_move_pct": 1.5}]
    started = {"outer_id": "outer", "plan_id": "plan", "candidate": {"candidate_id": "candidate"},
        "selection_id": "selection", "prospective_candidate_id": "candidate"}
    monkeypatch.setattr(outer, "_validate_targets", lambda *args: None)
    result = outer._result(started, inputs, "inputs-id", {"fit_attempt_count": 1}, forecasts,
                           validation, {"year": "2024"}, [1])
    assert len(canonical_json(inputs["training_inputs"]).encode()) > MAX_BYTES
    assert "training_inputs" not in result
    assert len(canonical_json(result).encode()) < MAX_BYTES
