"""EXP-190 Q2a: read the sealed v14 SIZE panel, dictionary and labels by content hash (design
sections 3, 4, 7). No production caller yet (Q2b folds and Q6 driver are the first)."""
import json
import sys
import time

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import content_hash, to_document
from engine.v2.ops.errors import fail
from engine.v2.research import _magnitude_panel_store
from engine.v2.research.magnitude_inputs import qualify_magnitude_training_targets
from engine.v2.research.magnitude_panel_qualified_table import recipe
from engine.v2.research.progress import stage
from experiments import size_search_binding
from experiments.derived_rows import with_derived
from experiments.native_registration import read_native_registration
from experiments.size_pnl_inputs import LEAD
from experiments.size_pnl_readers import admit_events
from experiments.size_pnl_sources import (
    MAX_OBJECT_BYTES, _bound_snapshot, _derived, _snapshot, _snapshot_mismatch, derive_panel_id, load_plan,
)

_MALFORMED = (KeyError, TypeError, AttributeError, ValueError)
_PANEL_REFUSALS = ("HOLDOUT_ACCESS_DENIED", "CONTRACT_MISMATCH", "IDENTITY_CONFLICT")


def _refuse(reason, message):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", f"{reason}: {message}", details={"reason": reason})


def _panel_rows(repository, feature, context):
    try:
        return _magnitude_panel_store.read(repository, feature, context, recipe("HMOSX")["model_columns"],
                                           lineage=False)
    except DataError as exc:
        if exc.code not in _PANEL_REFUSALS:
            raise
        raise _refuse("panel_mismatch", "the stored panel is not the sealed artifact") from None


def _load(conn, store, spec, plan_ref, max_object_bytes):
    registration = read_native_registration(conn, store, spec)
    document = registration.document
    if document["schema_version"] != size_search_binding.SCHEMA:
        raise _refuse("registration_schema_unknown", "the registration is not a v14 native registration")
    plan = load_plan(store, registration, plan_ref)
    panel_id = _derived(derive_panel_id, plan)
    if content_hash(json.loads(registration.binding_json)["size_panel"]) != panel_id:
        raise _refuse("panel_binding_mismatch", "the compact panel binding is not the plan's panel")
    dictionary = recipe("HMOSXLP")
    matches = [c for c in plan["candidates"] if c["feature_recipe"].get("subset") == "HMOSXLP"]
    if not matches or any(c["feature_recipe"]["dictionary_hash"] != dictionary["dictionary_hash"]
                          or list(c["feature_recipe"]["model_columns"]) != dictionary["model_columns"]
                          for c in matches):
        raise _refuse("dictionary_mismatch", "the plan's HMOSXLP candidates do not match the v14 dictionary")
    pinned = document["snapshot"]
    repository = Repository(conn, store, pin_snapshots=True)
    _bound_snapshot(conn, pinned["snapshot_id"])
    snapshot = _snapshot(repository, pinned["snapshot_id"])
    if to_document(snapshot) != pinned:
        raise _snapshot_mismatch()
    month = document["holdouts"]["holdout_as_of_month"]
    with stage("quantile admission"):
        admitted = admit_events(repository, snapshot, month, max_object_bytes=max_object_bytes)
    ids = set(admitted["events"]["event_id"])
    panel = document["size_panel"]
    context = panel["context"]
    events = sorted((e for e in context["target_events"] if e["event_id"] in ids),
                    key=lambda e: (e["event_date"], e["event_id"]))
    if not events:
        raise _refuse("admission_empty", "no panel event is admitted")
    feature_pin = panel["feature_snapshot"]
    _bound_snapshot(conn, feature_pin["snapshot_id"])
    feature = _snapshot(repository, feature_pin["snapshot_id"])
    if to_document(feature) != feature_pin:
        raise _refuse("panel_mismatch", "the stored panel snapshot is not the sealed one")
    with stage("quantile sealed panel"):
        rows = _panel_rows(repository, feature, context)
    rows = [row for row in rows if row["horizon_sessions"] == LEAD and row["event_id"] in ids]
    rows = with_derived(rows, dictionary["model_columns"])
    event_ids = [e["event_id"] for e in events]
    binding = content_hash({"plan_id": plan["plan_id"], "panel_id": panel_id, "snapshot_id": pinned["snapshot_id"],
                            "dictionary_hash": dictionary["dictionary_hash"], "event_ids": event_ids})
    return {"plan": plan, "repository": repository, "snapshot": snapshot, "as_of_month": month, "events": events,
            "rows": rows, "excluded_events": len(context["target_events"]) - len(events),
            "model_columns": list(dictionary["model_columns"]), "dictionary_hash": dictionary["dictionary_hash"],
            "binding": binding, "context": context, "fold_events": document["magnitude_plan"]["events"],
            "evaluation_labels_before": document["magnitude_execution"]["evaluation_labels_before"]}


def load_sealed_panel(conn, store, spec, plan_ref, *, max_object_bytes=MAX_OBJECT_BYTES):
    """Return the sealed panel for one plan: admitted events, lead-1 rows and the HMOSXLP binding."""
    started = time.monotonic()
    try:
        sealed = _load(conn, store, spec, plan_ref, max_object_bytes)
    except _MALFORMED as exc:
        raise _refuse("sealed_malformed", f"a sealed panel document is malformed ({type(exc).__name__})") from None
    print(f"[quantile_sources] events={len(sealed['events'])} excluded={sealed['excluded_events']} "
          f"rows={len(sealed['rows'])} dictionary={sealed['dictionary_hash']} "
          f"elapsed={time.monotonic() - started:.1f}s", file=sys.stderr)
    return sealed


def training_labels(sealed, event_ids, labels_before):
    """Qualified training labels for admitted events; a label counts only if available before labels_before."""
    if not (isinstance(event_ids, (list, tuple)) and event_ids and all(isinstance(e, str) for e in event_ids)
            and len(set(event_ids)) == len(event_ids)):
        raise fail("INVALID_EXPERIMENT_SPEC", "event_ids must be a non-empty list of unique strings",
                   details={"reason": "event_ids_invalid"})
    if not set(event_ids) <= {e["event_id"] for e in sealed["events"]}:
        raise fail("HOLDOUT_ACCESS_DENIED", "requested event ids are not admitted",
                   details={"reason": "key_not_admitted"})
    with stage("quantile training labels"):
        frame = qualify_magnitude_training_targets(sealed["repository"], sealed["snapshot"],
                                                   as_of_month=sealed["as_of_month"], labels_before=labels_before,
                                                   event_ids=list(event_ids))
    omitted = len(frame.attrs.get("omitted_labels", []))
    print(f"[quantile_labels] requested={len(event_ids)} qualified={len(frame)} omitted={omitted} "
          f"labels_before={labels_before}", file=sys.stderr)
    return frame
