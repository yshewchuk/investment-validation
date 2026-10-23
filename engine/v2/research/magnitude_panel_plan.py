"""Metadata-only panel context planning; never admission to raw market payloads."""
import json
import math
import re
from datetime import date

from engine.v2.contracts.data import SnapshotRef
from engine.v2.data import errors
from engine.v2.foundation import (
    canonical_json,
    content_hash,
    from_document,
    market_calendar,
    to_document,
)
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.research import magnitude_history_store as history
from engine.v2.research.magnitude_features import calendar_input
from engine.v2.research.magnitude_folds import annual_fold, magnitude_window
from engine.v2.research.magnitude_inputs import (
    forecast_cutoff,
    magnitude_recipe,
    target_object_identity,
)
from engine.v2.research.magnitude_panel_table import canonical_day, recipe
from engine.v2.research.progress import Progress, stage
from engine.v2.research.snapshot_inputs import snapshot_reference

_KEYS = {"schema_version", "source_snapshot", "history_snapshot", "history_context",
         "feature_recipe", "availability_lane", "producer", "admission", "context_id", "forecast_calendar",
         "forecast_calendar_source"}
_POLICY = {"raw_payload_access": "not_authorized",
           "raw_policy_status": "not_authorized_by_this_plan",
           "required_later": ["permitted_raw_read_contract", "approved_protected_event_policy",
                              "per_observation_availability", "bounded_batched_materializer"]}


def _digest(document, key):
    return document[key] == content_hash({k: v for k, v in document.items() if k != key})


def _calendar(document):
    """Validate the complete observed calendar, independently of supplied cutoff hashes."""
    if type(document) is not dict or set(document) != {"days", "observed_through"}:
        raise ValueError
    days = document["days"]
    if (type(days) is not list or not days or days != sorted(set(days))
            or document["observed_through"] != days[-1]):
        raise ValueError
    for day in days:
        canonical_day(day)
    return market_calendar.CalendarSessions(tuple(days), document["observed_through"])


def _cutoff(cutoff, event, calendar, clocks):
    """A closed v1 clock cannot be upgraded by merely recomputing its digest."""
    key = event["event_date"], event["session"], cutoff["horizon_sessions"]
    if key not in clocks:
        clocks[key] = forecast_cutoff(*key, calendar)
    if (set(cutoff) != {"schema_version", "event_date", "session", "horizon_sessions",
            "pre_announcement_close", "source_session_ceiling", "calendar_id", "cutoff_id"}
            or cutoff["schema_version"] != "earnings_forecast_cutoff.v1"
            or not _digest(cutoff, "cutoff_id")
            or cutoff != clocks[key]):
        raise ValueError


def _json_document(value):
    """Accept JSON types only; canonical hashing also supports non-JSON Python values."""
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError
        for item in value.values():
            _json_document(item)
    elif type(value) is list:
        for item in value:
            _json_document(item)
    elif type(value) not in (str, int, float, bool, type(None)) or (
            type(value) is float and not math.isfinite(value)):
        raise ValueError


def _text(value):
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError


def _population_source(source, snapshot):
    """Pin the exact native population projection and detached fragment shapes."""
    if (type(source) is not dict or set(source) != {"table", "columns", "dataset_version", "objects"}
            or source["table"] != "earnings_events"
            or source["columns"] != ["event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"]
            or source["dataset_version"] != snapshot["table_versions"]["earnings_events"]["dataset_version_id"]):
        raise ValueError
    _fragments(source["objects"])


def _plan_metadata(plan, snapshot):
    """Closed native v1 request and derived folds, not repository authentication."""
    magnitude_window(plan["evaluation_years"], canonical_day(plan["training_start"]))
    policy = ExperimentHoldouts(plan["holdout_as_of_month"])
    if (type(plan["evaluation_years"]) is not list
            or plan["random_membership_version"] != policy.random_version
            or plan["rolling_membership_version"] != policy.rolling_version
            or plan["horizons"] != [5, 3, 1]
            or any(type(horizon) is not int for horizon in plan["horizons"])
            or plan["target_recipe"] != magnitude_recipe()
            or plan["fold_policy"] != {"kind": "expanding_event_year", "group_key": "event_id",
                "fit_boundary": "min_year_start_and_evaluation_forecast_ceilings",
                "training_label_availability": "strictly_before_fit_boundary_required_later",
                "issuer_overlap": "allowed_across_time", "inner_search": "not_defined"}
            or plan["availability"] != {"basis": "reconstructed_date_bounds",
                "event_schedule_as_known": "unverified", "source_publication": "unverified"}):
        raise ValueError
    _population_source(plan["population_input"], snapshot)
    events = plan["events"]
    end = f"{int(plan['evaluation_years'][-1]) + 1:04d}-01-01"
    if (events != sorted(events, key=lambda event: (event["event_date"], event["event_id"]))
            or any(not plan["training_start"] <= event["event_date"] < end for event in events)
            or plan["folds"] != [annual_fold(year, events) for year in plan["evaluation_years"]]):
        raise ValueError


def _event_schema(event):
    """Canonical scalar identity and exact native forecast container/order."""
    if type(event) is not dict or set(event) != {"event_id", "ticker", "event_date", "session", "population_use", "forecasts"}:
        raise ValueError
    for field in ("event_id", "ticker"):
        _text(event[field])
    if (event["population_use"] != "post-release selection" or event["session"] not in ("BMO", "AMC")
            or type(event["forecasts"]) is not list
            or [cutoff["horizon_sessions"] for cutoff in event["forecasts"]] != [5, 3, 1]):
        raise ValueError


def _clocks(binding, calendar):
    """Reject duplicate targets and invalid or future clocks even on a rehashed plan."""
    plan = binding["temporal_plan"]
    keys = {"schema_version", "snapshot_id", "holdout_as_of_month", "random_membership_version",
            "rolling_membership_version", "training_start", "evaluation_years", "horizons",
            "target_recipe", "fold_policy", "population_input", "calendar_input", "events", "folds",
            "availability", "plan_id"}
    if (set(plan) != keys or not _digest(plan, "plan_id") or plan["schema_version"] != "magnitude_temporal_plan.v1"
            or plan["snapshot_id"] != binding["source_snapshot"]["snapshot_id"]):
        raise ValueError
    events = plan["events"]
    if type(events) is not list or not events:
        raise ValueError
    clocks = {}
    progress = Progress("panel clocks", len(events))
    with stage("panel clocks", report_refusals=True):
        try:
            for event in events:
                _event_schema(event)
                day = canonical_day(event["event_date"])
                history.validate_event_forecasts(event, binding["history_start"], date.fromisoformat(day))
                for cutoff in event["forecasts"]:
                    _cutoff(cutoff, event, calendar, clocks)
                progress.tick()
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError, errors.DataError):
            raise errors.fail("CONTRACT_MISMATCH", "invalid panel clocks") from None
    if len({event["event_id"] for event in events}) != len(events):
        raise ValueError
    _plan_metadata(plan, binding["source_snapshot"])


def _identity_parts(context):
    """Validate nested closed history identity and explicit producer/admission policies."""
    binding = context["history_context"]
    history.validate_context_identity(binding)
    if (set(binding) != {"schema_version", "source_snapshot", "temporal_plan", "history_start",
                         "feature_recipe", "producer", "context_id"}
            or binding["schema_version"] != "magnitude_materialization.v1"
            or binding["source_snapshot"] != context["source_snapshot"]
            or binding["feature_recipe"] != history.feature_recipe()):
        raise ValueError
    canonical_day(binding["history_start"])
    _producer(binding["producer"])
    _clocks(binding, _calendar(context["forecast_calendar"]))
    _calendar_source(context)
    _producer(context["producer"])
    admission = context["admission"]
    if (set(admission) != {*_POLICY, "history_fragments", "prior_outcome_fragments"}
            or any(admission[k] != value for k, value in _POLICY.items())):
        raise ValueError
    _fragments(admission["history_fragments"])
    _fragments(admission["prior_outcome_fragments"], event=True)


def _fragments(items, *, event=False):
    """Closed detached identities only; repository membership needs preflight."""
    keys = {"fragment_id", "object_id", "content_hash"} | ({"event_id"} if event else set())
    if type(items) is not list:
        raise ValueError
    for item in items:
        if (type(item) is not dict or set(item) != keys
                or any(not isinstance(value, str) or not value.strip() for value in item.values())
                or re.fullmatch(r"sha256:[0-9a-f]{64}", item["content_hash"]) is None):
            raise ValueError


def _calendar_source(context):
    """Check detached calendar provenance against the history and snapshot bindings."""
    source, snapshot = context["forecast_calendar_source"], context["source_snapshot"]
    if (type(source) is not dict
            or set(source) != {"snapshot_id", "calendar_version", "table", "columns", "dataset_version", "objects"}
            or source != context["history_context"]["temporal_plan"]["calendar_input"]
            or source["snapshot_id"] != snapshot["snapshot_id"]
            or source["calendar_version"] != snapshot["calendar_version"]
            or source["dataset_version"] != snapshot["table_versions"]["daily_market"]["dataset_version_id"]
            or source["table"] != "daily_market" or source["columns"] != ["date"]):
        raise ValueError
    _fragments(source["objects"])


def _producer(producer):
    """Require explicit source hashes and an environment document, not truthy sentinels."""
    if (type(producer) is not dict or set(producer) != {"source_closure", "environment"}
            or any(type(producer[key]) is not dict or not producer[key] for key in producer)):
        raise ValueError
    for path, digest in producer["source_closure"].items():
        if (not isinstance(path, str) or not path.strip() or not isinstance(digest, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None):
            raise ValueError


def validate_context(context):
    """Closed detached identity and clocks; repository admission is a separate step."""
    try:
        _json_document(context)
        if (type(context) is not dict or set(context) != _KEYS
                or context["schema_version"] != "magnitude_panel_context.v1"
                or not _digest(context, "context_id") or canonical_json(context["feature_recipe"]) != canonical_json(recipe())
                or context["availability_lane"] != "reconstructed"):
            raise ValueError
        for key in ("source_snapshot", "history_snapshot"):
            _panel_snapshot(from_document(SnapshotRef, context[key]))
        _identity_parts(context)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError, errors.DataError):
        raise errors.fail("CONTRACT_MISMATCH", "invalid research panel context") from None


def _panel_snapshot(snapshot):
    try:
        snapshot_reference(snapshot)
    except errors.DataError:
        raise errors.fail("CONTRACT_MISMATCH", "invalid panel snapshot reference") from None


def build_context(repository, source_snapshot, history_snapshot, history_context, *,
                  producer, availability_lane="reconstructed"):
    """Bind exact artifacts, protected population and isolated fragments before payloads.

    Only population/calendar metadata is scanned. No history, outcomes, price or
    daily-state values are decoded. The returned plan expressly grants no raw reads.
    """
    try:
        _json_document(producer)
        _producer(producer)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
        raise errors.fail("CONTRACT_MISMATCH", "invalid panel producer metadata") from None
    for snapshot in (source_snapshot, history_snapshot):
        _panel_snapshot(snapshot)
    if type(history_context) is not dict:
        raise errors.fail("CONTRACT_MISMATCH", "explicit panel snapshot/context types required")
    if (repository.resolve(source_snapshot.snapshot_id) != source_snapshot
            or repository.resolve(history_snapshot.snapshot_id) != history_snapshot
            or history_context.get("source_snapshot") != to_document(source_snapshot)):
        raise errors.fail("IDENTITY_CONFLICT", "panel requires exact source and history snapshots")
    with stage("panel source preflight", report_refusals=True):
        _, plans = history.preflight_source(repository, source_snapshot, history_context)
    with stage("panel stored preflight", report_refusals=True):
        history.preflight_stored(repository, history_snapshot, history_context)
    with stage("panel calendar metadata", report_refusals=True):
        calendar, calendar_source = calendar_input(repository, source_snapshot)
    if calendar_source != history_context["temporal_plan"]["calendar_input"]:
        raise errors.fail("IDENTITY_CONFLICT", "panel calendar source differs from history")
    records = repository.fragment_records(history_snapshot, history.TABLE)
    admission = {**_POLICY,
        "history_fragments": [{"fragment_id": r.fragment_id, "object_id": r.object_ref.object_id,
                               "content_hash": r.object_ref.content_hash} for r in records],
        "prior_outcome_fragments": [target_object_identity(event, record) for event, _, record in plans]}
    identity = {"schema_version": "magnitude_panel_context.v1",
        "source_snapshot": to_document(source_snapshot), "history_snapshot": to_document(history_snapshot),
        "history_context": history_context, "feature_recipe": recipe(), "producer": producer,
        "availability_lane": availability_lane, "admission": admission,
        "forecast_calendar": to_document(calendar), "forecast_calendar_source": calendar_source}
    # Detach nested caller objects: mutating a request cannot mutate a returned identity.

    try:
        identity = json.loads(canonical_json(identity))
        context = {**identity, "context_id": content_hash(identity)}
    except RecursionError:
        raise errors.fail("CONTRACT_MISMATCH", "panel metadata nesting exceeds serialization limits") from None
    validate_context(context)
    return context


def preflight_context(repository, source_snapshot, history_snapshot, context):
    """Recheck the complete metadata plan; no cache or implicit refresh is allowed."""
    validate_context(context)
    for snapshot in (source_snapshot, history_snapshot):
        _panel_snapshot(snapshot)
    try:
        if (context["source_snapshot"] != to_document(source_snapshot)
                or context["history_snapshot"] != to_document(history_snapshot)):
            raise errors.fail("IDENTITY_CONFLICT", "panel snapshots differ from context")
    except RecursionError:
        raise errors.fail("CONTRACT_MISMATCH", "panel snapshot nesting exceeds serialization limits") from None
    expected = build_context(repository, source_snapshot, history_snapshot, context["history_context"],
        producer=context["producer"], availability_lane=context["availability_lane"])
    if expected != context:
        raise errors.fail("IDENTITY_CONFLICT", "panel admission metadata changed")
    return context["admission"]
