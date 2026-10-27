"""SIZE's shared research-refit history for challengers and the legacy comparator."""
from engine.v2.data.errors import fail
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.eligible_history_table import FEATURES
from engine.v2.research.magnitude_folds import magnitude_window, plan_magnitude_folds
from engine.v2.research.snapshot_inputs import snapshot_reference
from experiments import eligible_history


def _policy(policy):
    if policy != eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30):
        raise fail("HOLDOUT_ACCESS_DENIED", "SIZE requires the frozen 30-day eligibility policy")


def build(repository, snapshot, **request):
    """No fits or legacy parity promise: both arms consume this same clean table."""
    if "producer" in request:
        raise fail("HOLDOUT_ACCESS_DENIED", "installed producer identity cannot be supplied by callers")
    request = {"event_ids": None, **request}
    if set(request) != {"as_of_month", "evaluation_years", "training_start",
                        "history_start", "policy", "limits", "event_ids"}:
        raise fail("CONTRACT_MISMATCH", "explicit closed SIZE history request required")
    window = magnitude_window(request.pop("evaluation_years"), request.pop("training_start"))
    request["horizons"] = [1, 3, 5]
    eligible_history.validate_request(request, allow_all_events=True)
    _policy(request["policy"])
    snapshot_reference(snapshot)
    eligible_history.admit_source(repository, snapshot, request["limits"])
    plan = plan_magnitude_folds(repository, snapshot, as_of_month=request["as_of_month"],
        **window, event_ids=request["event_ids"])
    request["event_ids"] = [event["event_id"] for event in plan["events"]]
    artifact = eligible_history.materialize(repository, snapshot, **request)
    return {**artifact, "temporal_plan": plan, "request": request}


def comparison_inputs(repository, snapshot, artifact):
    """A single admitted input bundle; caller shares it across both model arms."""
    if (type(artifact) is not dict or not {"request", "feature_snapshot"} <= set(artifact)
            or set(artifact) - {"request", "feature_snapshot", "context", "temporal_plan"}
            or any(type(artifact[key]) is not dict for key in ("context", "temporal_plan") if key in artifact)):
        raise fail("CONTRACT_MISMATCH", "closed SIZE history artifact envelope required")
    eligible_history.validate_request(artifact["request"])
    _policy(artifact["request"]["policy"])
    return eligible_history.read(repository, snapshot, artifact["feature_snapshot"],
        columns=list(FEATURES), **artifact["request"])
