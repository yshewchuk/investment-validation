"""Metadata-only magnitude folds; no label access or experiment admission."""
from __future__ import annotations

from engine.v2.data import errors
from engine.v2.foundation import content_hash
from engine.v2.foundation.experiment_holdouts import _calendar_date
from engine.v2.models.regression_target import validate_target
from engine.v2.research.experiment_population import load_population
from engine.v2.research.magnitude_features import _calendar_input
from engine.v2.research.magnitude_inputs import forecast_cutoff, magnitude_recipe

_HORIZONS = (5, 3, 1)


def _window(evaluation_years, training_start):
    """Reject implicit/ambiguous fold declarations before repository access."""
    try:
        if (not isinstance(evaluation_years, (list, tuple)) or not evaluation_years
                or any(not isinstance(year, str) or len(year) != 4 or not year.isascii()
                       or not year.isdecimal() or not 2 <= int(year) <= 9998
                       for year in evaluation_years)
                or list(evaluation_years) != sorted(set(evaluation_years))):
            raise ValueError
        start = _calendar_date(training_start).isoformat()
        if start >= evaluation_years[0] + "-01-01":
            raise ValueError
        return start, f"{int(evaluation_years[-1]) + 1:04d}-01-01"
    except (ValueError, TypeError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "explicit ordered years and earlier training start required") from None


def magnitude_window(evaluation_years, training_start):
    """Canonical detached request shared by planning and native preregistration."""
    start, _ = _window(evaluation_years, training_start)
    return {"training_start": start, "evaluation_years": list(evaluation_years)}


def _event_plan(event, calendar, clocks):
    """Bind one canonical event to all horizons, with no outcome or feature read."""
    day, forecasts = event.event_date.date().isoformat(), []
    for horizon in _HORIZONS:
        key = (day, event.session, horizon)
        if key not in clocks:
            clocks[key] = forecast_cutoff(event.event_date, event.session, horizon, calendar)
        forecasts.append(dict(clocks[key]))
    return {"event_id": event.event_id, "ticker": event.ticker, "event_date": day,
            "session": event.session, "population_use": event.population_use, "forecasts": forecasts}


def annual_fold(year, events):
    """Use one conservative fit boundary for every horizon of an annual fold."""
    test = [event for event in events if event["event_date"][:4] == year]
    if not test:
        raise errors.fail("POPULATION_COLLAPSED", "every magnitude fold needs evaluation events")
    before = min(year + "-01-01", *(cutoff["source_session_ceiling"]
                 for event in test for cutoff in event["forecasts"]))
    train = [event["event_id"] for event in events if event["event_date"] < before]
    if not train:
        raise errors.fail("POPULATION_COLLAPSED", "every magnitude fold needs training candidates")
    return {"year": year, "training_labels_before": before,
            "training_candidate_event_ids": train,
            "evaluation_event_ids": [event["event_id"] for event in test]}


def plan_magnitude_folds(repository, snapshot, *, as_of_month, evaluation_years,
                         training_start, event_ids=None):
    """Plan annual groups from admitted metadata; preserve the legacy target identity."""
    return _plan_folds(repository, snapshot, as_of_month=as_of_month, evaluation_years=evaluation_years,
        training_start=training_start, event_ids=event_ids, target=magnitude_recipe(),
        schema="magnitude_temporal_plan.v1")


def plan_regression_folds(repository, snapshot, *, target, as_of_month, evaluation_years,
                          training_start, event_ids=None):
    """Plan target-bound clocks, not target availability or sealed search admission."""
    target = validate_target(target)
    return _plan_folds(repository, snapshot, as_of_month=as_of_month, evaluation_years=evaluation_years,
        training_start=training_start, event_ids=event_ids, target=target,
        schema="regression_temporal_plan.v1")


def _plan_folds(repository, snapshot, *, target, schema, as_of_month, evaluation_years,
                training_start, event_ids):
    start, end = _window(evaluation_years, training_start)
    population = load_population(repository, snapshot, as_of_month=as_of_month,
                                 purpose="selection", event_ids=event_ids)
    days = population.event_date.dt.strftime("%Y-%m-%d")
    inside = (days >= start) & (days < end)
    if event_ids is not None and not inside.all():
        raise errors.fail("CONTRACT_MISMATCH", "explicit magnitude events must be inside the declared window")
    selected = population[inside].sort_values(["event_date", "event_id"], kind="stable")
    if selected.empty:
        raise errors.fail("POPULATION_COLLAPSED", "magnitude temporal window is empty")
    calendar, source = _calendar_input(repository, snapshot)
    clocks = {}
    events = [_event_plan(event, calendar, clocks) for event in selected.itertuples(index=False)]
    folds = [annual_fold(year, events) for year in evaluation_years]
    records = repository.fragment_records(snapshot, "earnings_events")
    context = {name: selected.iloc[0][name] for name in (
        "snapshot_id", "holdout_as_of_month", "random_membership_version", "rolling_membership_version")}
    identity = {"schema_version": schema, **context,
        "training_start": start, "evaluation_years": list(evaluation_years),
        "horizons": list(_HORIZONS), "target_recipe": target,
        "fold_policy": {"kind": "expanding_event_year", "group_key": "event_id",
            "fit_boundary": "min_year_start_and_evaluation_forecast_ceilings",
            "training_label_availability": "strictly_before_fit_boundary_required_later",
            "issuer_overlap": "allowed_across_time", "inner_search": "not_defined"},
        "population_input": {"table": "earnings_events",
            "columns": ["event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"],
            "dataset_version": snapshot.table_versions["earnings_events"].dataset_version_id,
            "objects": [{"fragment_id": record.fragment_id, "object_id": record.object_ref.object_id,
                         "content_hash": record.object_ref.content_hash} for record in records]},
        "calendar_input": source, "events": events, "folds": folds,
        "availability": {"basis": "reconstructed_date_bounds", "event_schedule_as_known": "unverified",
                         "source_publication": "unverified"}}
    return {**identity, "plan_id": content_hash(identity)}
