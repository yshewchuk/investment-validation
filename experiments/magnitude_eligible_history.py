"""SIZE's shared research-refit history for challengers and the legacy comparator."""
from engine.v2.data.errors import fail
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.eligible_history_table import FEATURES
from engine.v2.research.magnitude_folds import plan_magnitude_folds
from experiments import eligible_history


def _policy(policy):
    if policy != eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30):
        raise fail("HOLDOUT_ACCESS_DENIED", "SIZE requires the frozen 30-day eligibility policy")


def build(repository, snapshot, *, as_of_month, evaluation_years, training_start,
          history_start, policy, limits, event_ids=None):
    """No fits or legacy parity promise: both arms consume this same clean table."""
    _policy(policy)
    eligible_history.admit_source(repository, snapshot, limits)
    plan = plan_magnitude_folds(repository, snapshot, as_of_month=as_of_month,
        evaluation_years=evaluation_years, training_start=training_start, event_ids=event_ids)
    request = dict(as_of_month=as_of_month, policy=policy, history_start=history_start,
        event_ids=[event["event_id"] for event in plan["events"]], horizons=[1, 3, 5], limits=limits)
    artifact = eligible_history.materialize(repository, snapshot, **request)
    return {**artifact, "temporal_plan": plan, "request": request}


def comparison_inputs(repository, snapshot, artifact):
    """A single admitted input bundle; caller shares it across both model arms."""
    _policy(artifact["request"]["policy"])
    return eligible_history.read(repository, snapshot, artifact["feature_snapshot"],
        columns=list(FEATURES), **artifact["request"])
