import pytest

from engine.v2.features import FeatureContextError, FeatureContextPlanner


def test_context_preserves_analog_population_and_zero_values():
    planner = FeatureContextPlanner.default()
    request = planner.request(
        event_refs=({"event_id": "e1", "decision_at": "2026-09-16"},),
        snapshot_ref="snap-1",
        recipe_refs=("legacy.market_context.v1", "legacy.bucket_analogs.v1"),
        visible_event_ids=("e1",),
        analog_population_ref="all-history-v1",
    )
    frame = planner.frame(
        request,
        ({"event_id": "e1", "observed_at": "2026-09-15", "x": 0.0},),
        ordered_columns=("x", "missing"), coverage_receipt_ref="coverage-1",
    )
    assert frame.values[0]["x"] == 0.0
    assert frame.values[0]["missing"] is None if "missing" in frame.values[0] else True
    assert frame.lineage_refs == request.feature_recipe_refs
    assert request.decision_contexts[0]["analog_population_ref"] == "all-history-v1"


def test_context_rejects_future_observation():
    planner = FeatureContextPlanner.default()
    request = planner.request(
        event_refs=({"event_id": "e1", "decision_at": "2026-09-16"},),
        snapshot_ref="snap-1", recipe_refs=("legacy.market_context.v1",),
    )
    with pytest.raises(FeatureContextError, match="after decision cutoff"):
        planner.frame(request, ({"event_id": "e1", "observed_at": "2026-09-17"},),
                      ordered_columns=("x",), coverage_receipt_ref="coverage-1")
