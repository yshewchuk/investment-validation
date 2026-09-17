import pytest

from engine.v2.features import FeatureContextError, FeatureContextPlanner


def test_context_preserves_analog_population_and_zero_values():
    planner = FeatureContextPlanner.default()
    request = planner.request(
        event_refs=({"event_id": "e1", "decision_at": "2026-09-16T20:00:00Z"},),
        snapshot_ref="snap-1",
        recipe_refs=("legacy.market_context.v1", "legacy.bucket_analogs.v1"),
        visible_event_ids=("e1",),
        analog_population_ref="all-history-v1",
    )
    frame = planner.frame(
        request,
        ({"event_id": "e1", "observed_at": "2026-09-16T19:00:00Z", "x": 0.0},),
        ordered_columns=("x", "missing"), coverage_receipt_ref="coverage-1",
    )
    assert frame.values[0]["x"] == 0.0
    assert frame.values[0]["missing"] is None if "missing" in frame.values[0] else True
    assert frame.lineage_refs == request.feature_recipe_refs
    assert request.decision_contexts[0]["analog_population_ref"] == "all-history-v1"


def test_context_rejects_future_observation():
    planner = FeatureContextPlanner.default()
    request = planner.request(
        event_refs=({"event_id": "e1", "decision_at": "2026-09-16T20:00:00Z"},),
        snapshot_ref="snap-1", recipe_refs=("legacy.market_context.v1",),
    )
    with pytest.raises(FeatureContextError, match="after decision cutoff"):
        planner.frame(request, ({"event_id": "e1", "observed_at": "2026-09-16T21:00:00Z"},),
                      ordered_columns=("x",), coverage_receipt_ref="coverage-1")


def test_context_compares_timezone_offsets_as_instants():
    planner = FeatureContextPlanner.default()
    request = planner.request(
        event_refs=({"event_id": "e1", "decision_at": "2026-09-16T20:00:00Z"},),
        snapshot_ref="snap-1", recipe_refs=("legacy.market_context.v1",),
    )
    with pytest.raises(FeatureContextError, match="after decision cutoff"):
        planner.frame(
            request,
            ({"event_id": "e1", "observed_at": "2026-09-16T19:00:00-04:00"},),
            ordered_columns=("x",),
            coverage_receipt_ref="coverage-1",
        )


@pytest.mark.parametrize(
    ("decision_at", "observed_at", "message"),
    (
        (None, "2026-09-16T19:00:00Z", "decision cutoff timestamp is required"),
        ("2026-09-16T20:00:00Z", None, "observed_at timestamp is required"),
        ("2026-09-16T20:00:00", "2026-09-16T19:00:00Z", "must be timezone-aware"),
        ("2026-09-16T20:00:00Z", "2026-09-16T19:00:00", "must be timezone-aware"),
    ),
)
def test_context_refuses_missing_or_naive_causal_timestamps(
    decision_at, observed_at, message
):
    planner = FeatureContextPlanner.default()
    request = planner.request(
        event_refs=({"event_id": "e1", "decision_at": decision_at},),
        snapshot_ref="snap-1", recipe_refs=("legacy.market_context.v1",),
    )
    with pytest.raises(FeatureContextError, match=message):
        planner.frame(
            request,
            ({"event_id": "e1", "observed_at": observed_at},),
            ordered_columns=("x",),
            coverage_receipt_ref="coverage-1",
        )
