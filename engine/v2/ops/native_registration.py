"""Verify native registration evidence owned by the existing catalog reservation."""
import json

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
    return _magnitude_context(document, plan, request)


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
    return (bool(ids) and all(isinstance(event[key], str) and event[key].strip()
                             for event in events for key in fields - {"forecasts"})
            and len(set(ids)) == len(ids) and sorted(ids) == expected
            and all(_magnitude_forecasts(event) for event in events))


def _magnitude_forecasts(event):
    """Each closed clock binds its canonical event and fixed horizon with an intact digest."""
    clocks = event["forecasts"]
    if not isinstance(clocks, list) or len(clocks) != 3:
        return False
    fields = {"schema_version", "event_date", "session", "horizon_sessions", "pre_announcement_close",
              "source_session_ceiling", "calendar_id", "cutoff_id"}
    return all(isinstance(clock, dict) and set(clock) == fields
        and clock["schema_version"] == "earnings_forecast_cutoff.v1"
        and clock["event_date"] == event["event_date"] and clock["session"] == event["session"]
        and type(clock["horizon_sessions"]) is int and clock["horizon_sessions"] == horizon
        and clock["cutoff_id"] == content_hash({k: v for k, v in clock.items() if k != "cutoff_id"})
        for clock, horizon in zip(clocks, (5, 3, 1)))


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
    return (fold["year"] == year and isinstance(fold["training_labels_before"], str)
            and bool(fold["training_labels_before"])
            and all(isinstance(ids, list) and bool(ids) and len(set(ids)) == len(ids)
                    and set(ids) <= events.keys() for ids in (train, test))
            and not set(train).intersection(test)
            and set(test) == {key for key, event_year in events.items() if event_year == year})
