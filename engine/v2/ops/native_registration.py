"""Verify native registration evidence owned by the existing catalog reservation."""
import json
import re
from datetime import date

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    artifact_reference,
    canonical_json,
    content_hash,
    from_document,
)
from engine.v2.ops.errors import fail


def verified_registration_bytes(conn, store, run_id, *, expected_ref=None):
    """Return exact verified registration bytes, or None for an absent reservation."""
    row = conn.execute(
        "SELECT e.*, h.input_hash AS registered_input, h.payload_hash AS registered_payload, "
        "h.payload_json AS registered_document FROM experiment_runs e "
        "LEFT JOIN hypotheses h ON h.run_id=e.run_id AND h.spec_hash=e.spec_hash "
        "WHERE e.run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    try:
        evidence = json.loads(row["evidence_json"])
        ref = from_document(ArtifactRef, evidence["native_registration"])
        if (ref.schema_ref not in ("native_experiment_registration.v1.0", "native_experiment_registration.v1.1",
                                  "native_experiment_registration.v1.2")
                or (expected_ref is not None and ref != expected_ref)):
            raise ValueError
        binding = store.read_verified(ref)
        document = json.loads(binding)
        valid = _matches(row, evidence, ref, binding, document)
    except (ArtifactError, ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration differs from its immutable binding")
    return binding


def _matches(row, evidence, ref, binding, document):
    variant = content_hash(document)
    return (_magnitude_binding(document) and document["schema_version"] == ref.schema_ref
            and canonical_json(document).encode() == binding
            and artifact_reference(binding, ref.schema_ref) == ref
            and row["spec_hash"] == document["spec_hash"] and row["mode"] == "primary"
            and row["input_hash"] == row["registered_input"] == variant
            and row["registered_payload"] == content_hash({"spec": row["spec_hash"], "input": variant})
            and row["registered_document"] == "{}" and evidence["variant_id"] == variant
            and type(evidence["planned_variants"]) is int and evidence["planned_variants"] == 1)


def _magnitude_binding(document):
    """Check stored plan integrity and cross-links, never reconstruct or authorize."""
    if document["schema_version"] != "native_experiment_registration.v1.2":
        return not {"magnitude_request", "magnitude_plan"}.intersection(document)
    if set(document) != {"schema_version", "spec_hash", "execution_plan", "snapshot", "scope",
                         "source_closure", "environment", "event_ids", "holdouts",
                         "magnitude_request", "magnitude_plan"}:
        return False
    plan, request = document["magnitude_plan"], document["magnitude_request"]
    if not isinstance(plan, dict) or not isinstance(request, dict):
        return False
    if set(plan) != {"schema_version", "snapshot_id", "holdout_as_of_month", "random_membership_version",
                     "rolling_membership_version", "training_start", "evaluation_years", "horizons",
                     "target_recipe", "fold_policy", "population_input", "calendar_input", "events",
                     "folds", "availability", "plan_id"}:
        return False
    return (_magnitude_metadata(plan) and _magnitude_sources(plan, document["snapshot"])
            and _magnitude_context(document, plan, request))


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _day(value):
    """Validate the JSON wire date, without accepting timestamps or compact ISO dates."""
    return (isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is not None
            and date.fromisoformat(value).isoformat() == value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _magnitude_metadata(plan):
    """Validate fixed v1 wire descriptors, not live calendar or fold semantics."""
    recipe = {"schema_version": "earnings_magnitude.v1", "units": "percentage_points",
        "target": "absolute_earnings_close_to_close_return", "price_basis": "split_neutral_non_dividend_adjusted",
        "estimand": "conditional_mean", "primary_loss": "pooled_rmse",
        "secondary_diagnostics": ["mae", "quantile_calibration"]}
    policy = {"kind": "expanding_event_year", "group_key": "event_id",
        "fit_boundary": "min_year_start_and_evaluation_forecast_ceilings",
        "training_label_availability": "strictly_before_fit_boundary_required_later",
        "issuer_overlap": "allowed_across_time", "inner_search": "not_defined"}
    availability = {"basis": "reconstructed_date_bounds", "event_schedule_as_known": "unverified",
                    "source_publication": "unverified"}
    return (plan["target_recipe"] == recipe and plan["fold_policy"] == policy
            and plan["availability"] == availability and _magnitude_window(plan)
            and plan["random_membership_version"] == "canonical-event-sha256.v1"
            and plan["rolling_membership_version"] == "calendar-months.v1"
            and _text(plan["snapshot_id"]) and _text(plan["holdout_as_of_month"])
            and len(plan["holdout_as_of_month"]) == 7 and _day(plan["holdout_as_of_month"] + "-01"))


def _magnitude_window(plan):
    years = plan["evaluation_years"]
    return (isinstance(years, list) and bool(years)
            and all(isinstance(year, str) and re.fullmatch(r"[0-9]{4}", year)
                    and 2 <= int(year) <= 9998 for year in years)
            and years == sorted(set(years)) and _day(plan["training_start"])
            and plan["training_start"] < years[0] + "-01-01")


def _magnitude_sources(plan, snapshot):
    """Bind closed provenance records to the already sealed snapshot, without I/O."""
    columns = ["event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"]
    population, calendar = plan["population_input"], plan["calendar_input"]
    if (not _magnitude_source(population, "earnings_events", columns, snapshot)
            or not _magnitude_source(calendar, "daily_market", ["date"], snapshot)):
        return False
    return (calendar["snapshot_id"] == plan["snapshot_id"]
            and _text(calendar["calendar_version"]) and calendar["calendar_version"] == snapshot["calendar_version"])


def _magnitude_source(source, table, columns, snapshot):
    fields = {"table", "columns", "dataset_version", "objects"}
    if table == "daily_market":
        fields |= {"snapshot_id", "calendar_version"}
    if not isinstance(source, dict) or set(source) != fields:
        return False
    return (source["table"] == table and source["columns"] == columns and _text(source["dataset_version"])
            and source["dataset_version"] == snapshot["table_versions"][table]["dataset_version_id"]
            and _magnitude_objects(source["objects"]))


def _magnitude_objects(objects):
    if not isinstance(objects, list) or not objects:
        return False
    return (all(isinstance(obj, dict) and set(obj) == {"fragment_id", "object_id", "content_hash"}
                and _text(obj["fragment_id"]) and _text(obj["object_id"]) and _digest(obj["content_hash"])
                for obj in objects)
            and len({obj["fragment_id"] for obj in objects}) == len(objects))


def _magnitude_context(document, plan, request):
    """Cross-link the sealed request and selected membership to the outer identity."""
    return (plan["schema_version"] == "magnitude_temporal_plan.v1"
            and plan["plan_id"] == content_hash({k: v for k, v in plan.items() if k != "plan_id"})
            and request == {"training_start": plan["training_start"], "evaluation_years": plan["evaluation_years"]}
            and plan["evaluation_years"] == document["execution_plan"]["folds"]
            and document["execution_plan"]["price_source"] == "computed_moves"
            and plan["snapshot_id"] == document["snapshot"]["snapshot_id"]
            and document["holdouts"] == {name: plan[name] for name in (
                "holdout_as_of_month", "random_membership_version", "rolling_membership_version")}
            and _magnitude_events(plan["events"], document["event_ids"])
            and _magnitude_folds(plan))


def _magnitude_events(events, expected):
    """Enclosing IDs must equal the complete nonempty unique canonical plan IDs."""
    fields = {"event_id", "ticker", "event_date", "session", "population_use", "forecasts"}
    if (not isinstance(events, list) or any(not isinstance(event, dict) or set(event) != fields
                                          for event in events)):
        return False
    ids = [event["event_id"] for event in events]
    return (bool(ids) and all(_magnitude_event(event) for event in events)
            and len(set(ids)) == len(ids) and sorted(ids) == expected
            and events == sorted(events, key=lambda event: (event["event_date"], event["event_id"]))
            and len({clock["calendar_id"] for event in events for clock in event["forecasts"]}) == 1)


def _magnitude_event(event):
    return (all(_text(event[key]) for key in ("event_id", "ticker", "event_date", "session", "population_use"))
            and _day(event["event_date"]) and event["session"] in ("AMC", "BMO")
            and event["population_use"] == "post-release selection" and _magnitude_forecasts(event))


def _magnitude_forecasts(event):
    """Each closed clock binds its canonical event and fixed horizon with an intact digest."""
    clocks = event["forecasts"]
    if not isinstance(clocks, list) or len(clocks) != 3:
        return False
    return (all(_magnitude_clock(clock, horizon, event) for clock, horizon in zip(clocks, (5, 3, 1)))
            and len({clock["pre_announcement_close"] for clock in clocks}) == 1)


def _magnitude_clock(clock, horizon, event):
    fields = {"schema_version", "event_date", "session", "horizon_sessions", "pre_announcement_close",
              "source_session_ceiling", "calendar_id", "cutoff_id"}
    return (isinstance(clock, dict) and set(clock) == fields
        and clock["schema_version"] == "earnings_forecast_cutoff.v1"
        and clock["event_date"] == event["event_date"] and clock["session"] == event["session"]
        and type(clock["horizon_sessions"]) is int and clock["horizon_sessions"] == horizon
        and _day(clock["pre_announcement_close"]) and _day(clock["source_session_ceiling"])
        and _digest(clock["calendar_id"])
        and clock["cutoff_id"] == content_hash({k: v for k, v in clock.items() if k != "cutoff_id"}))


def _magnitude_folds(plan):
    """Stored folds must cover the declared years and refer only to their sealed population."""
    if plan["horizons"] != [5, 3, 1] or any(type(value) is not int for value in plan["horizons"]):
        return False
    folds = plan["folds"]
    if not isinstance(folds, list) or len(folds) != len(plan["evaluation_years"]):
        return False
    events = {event["event_id"]: event["event_date"][:4] for event in plan["events"]}
    return all(_magnitude_fold(fold, year, events) for fold, year in zip(folds, plan["evaluation_years"]))


def _magnitude_fold(fold, year, events):
    """Validate structural membership; fresh reconstruction owns the actual date boundary."""
    if not isinstance(fold, dict) or set(fold) != {
            "year", "training_labels_before", "training_candidate_event_ids", "evaluation_event_ids"}:
        return False
    train, test = fold["training_candidate_event_ids"], fold["evaluation_event_ids"]
    return (fold["year"] == year and _day(fold["training_labels_before"])
            and all(_magnitude_member_order(ids, events) for ids in (train, test))
            and not set(train).intersection(test)
            and set(test) == {key for key, event_year in events.items() if event_year == year})


def _magnitude_member_order(ids, events):
    """One linear pass preserves the canonical order and rejects duplicates/foreign IDs."""
    if not isinstance(ids, list) or not ids:
        return False
    members = set(ids)
    return ids == [key for key in events if key in members]
