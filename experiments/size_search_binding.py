"""Explicit v2 SIZE source/fold binding; installed panel admission, no hidden builds."""
import copy
import json
import sys
import time
from datetime import date
from hashlib import sha256
from pathlib import Path

from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import artifact_reference, canonical_json, content_hash, from_document, to_document
from engine.v2.models.frozen_state_io import BoundedStateReadError, read_bounded_state
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.magnitude_inputs import (
    preflight_magnitude_targets,
    qualify_magnitude_training_targets,
)
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_search_request_versions import compile_request
from experiments.size_registration_artifacts import (
    pack_size_registration,
    size_registration_artifact_bytes,
)
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    NativeRegistration,
    prepare_native_registration,
    verify_native_registration,
)
from experiments.size_comparator import recipe as comparison_recipe

SCHEMA = "native_experiment_registration.v1.7"


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "qualified SIZE source/fold binding differs")


class _PreviewStore:
    def publish_bytes(self, payload, schema_ref=None):
        return artifact_reference(payload, schema_ref)


def _preview_payload(binding):
    compact, _ = pack_size_registration(_PreviewStore(), binding)
    size_registration_artifact_bytes(compact)
    return canonical_json(compact).encode("utf-8")


def validate_panel_request(value):
    """Closed local wire envelope only; installed readers retain source admission."""
    try:
        strings = {"feature_snapshot_id", "history_snapshot", "price_history_snapshot",
                   "as_of_month", "history_start", "source_start"}
        limits = {"limits": {"max_source_rows", "max_fragments", "max_object_bytes", "max_total_bytes", "max_output_rows"},
            "metadata_limits": {"max_catalog_tables", "max_catalog_fragments", "max_catalog_bytes",
                "max_fragments", "max_partitions", "max_tickers", "max_scan_rows", "max_object_bytes",
                "max_total_object_bytes", "max_batch_rows", "max_retained_rows",
                "max_retained_metadata_bytes", "max_proof_bytes"}}
        if (type(value) is not dict or set(value) != strings | set(limits) | {"policy", "event_ids", "horizons"}
                or any(type(value[k]) is not str or not value[k] or value[k] != value[k].strip() for k in strings)):
            raise ValueError
        for day in (value["history_start"], value["source_start"], value["as_of_month"] + "-01"):
            if date.fromisoformat(day).isoformat() != day:
                raise ValueError
        events, horizons = value["event_ids"], value["horizons"]
        if (type(events) is not list or not events or any(type(e) is not str or not e or e != e.strip() for e in events)
                or len(set(events)) != len(events) or type(horizons) is not list or horizons != [1, 3, 5]
                or any(type(h) is not int for h in horizons)
                or type(value["policy"]) is not dict or type(value["policy"].get("embargo_days")) is not int
                or value["policy"] != eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)):
            raise ValueError
        for key, fields in limits.items():
            if (type(value[key]) is not dict or set(value[key]) != fields
                    or any(type(n) is not int or not 0 < n < sys.maxsize for n in value[key].values())):
                raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid closed SIZE panel request") from None
    return value


def _read_panel(repository, source, request, *, columns):
    from experiments.size_panel import read
    args = dict(request)
    feature = repository.resolve(args.pop("feature_snapshot_id"))
    for name in ("history_snapshot", "price_history_snapshot"):
        args[name] = repository.resolve(args[name])
    return read(repository, source, feature, columns=columns, lineage=False, **args)


_PANEL_MEMO = {}


def _stored_panel(repository, source, binding, columns):
    """Full admission on a process-fresh key; later calls reuse the memo after a hash check."""
    validate_panel_request(binding["size_panel_request"])
    key = (id(repository), content_hash(to_document(source)),
           content_hash(binding["size_panel_request"]), tuple(columns))
    entry = _PANEL_MEMO.get(key)
    if entry is not None and entry["repository"] is repository:
        panel = entry["panel"]
        fingerprint = content_hash({"context": panel["context"], "feature_snapshot": panel["feature_snapshot"]})
        _require(fingerprint == content_hash(binding["size_panel"]))
    else:
        panel = _read_panel(repository, source, binding["size_panel_request"], columns=columns)
        _require({"context": panel["context"], "feature_snapshot": panel["feature_snapshot"]}
                 == binding["size_panel"])
        _PANEL_MEMO.clear()
        _PANEL_MEMO[key] = {"repository": repository, "panel": panel}
    return copy.deepcopy(panel)


def prepare(repository, spec, request, panel_request, *, scope, training_start, labels_before,
            publish_artifacts=True):
    """Prepare a new development anchor; caller explicitly reserves it separately."""
    from engine.v2.research.magnitude_panel_qualified_table import recipe
    sources = _sources()
    panel_request = validate_panel_request(panel_request)
    schedule = compile_request(request)
    if request["limits"]["child"]["memory_bytes"] > request["limits"]["parent"]["memory_bytes"]:
        raise fail("INVALID_EXPERIMENT_SPEC", "child memory must not exceed inherited parent limit")
    _require(request["feature_dictionary_hash"] == recipe("HMO")["dictionary_hash"])
    _require(panel_request["feature_snapshot_id"] == request["feature_snapshot_id"]
             and panel_request["history_snapshot"] == request["history_snapshot_id"]
             and panel_request["horizons"] == [1, 3, 5])
    base, spec = prepare_native_registration(repository, spec, code_root=Path(__file__).resolve().parents[1],
        scope=scope, as_of_month=panel_request["as_of_month"], event_ids=panel_request["event_ids"],
        magnitude_training_start=training_start, magnitude_labels_before=labels_before, defer_serialization=True)
    binding = base.document
    _require(binding["execution_plan"]["runner"] == "experiments/size_search.py")
    binding["source_closure"] = sources
    _require(binding["magnitude_plan"]["evaluation_years"] == schedule["request"]["outer_years"])
    panel = _read_panel(repository, from_document(SnapshotRef, binding["snapshot"]), panel_request,
                        columns=recipe("HMO")["model_columns"])
    context = panel["context"]
    _require(set(binding["event_ids"]) == {e["event_id"] for e in context["target_events"]})
    _require(request["eligibility_policy_ref"] == {"schema_ref": "group_eligibility_plan.v1",
        "plan_id": context["eligibility_plan"]["plan_id"], "policy_id": context["eligibility_plan"]["policy"]["policy_id"]})
    binding.update(schema_version=SCHEMA, size_request=schedule["request"], size_panel_request=panel_request,
        size_panel={"context": context, "feature_snapshot": panel["feature_snapshot"]},
        size_comparison=comparison_recipe())
    preview = _preview_payload(binding)
    if len(preview) > MAX_NATIVE_EVIDENCE_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "native registration exceeds its UTF-8 byte budget",
            details={"registration_bytes": len(preview), "max_registration_bytes": MAX_NATIVE_EVIDENCE_BYTES})
    _folds(binding, schedule["request"])
    if not publish_artifacts:
        return NativeRegistration(base.run_id, b"", binding), spec
    compact, logical = pack_size_registration(repository._store, binding)
    payload = canonical_json(compact).encode("utf-8")
    _require(payload == preview)
    return NativeRegistration(base.run_id, payload, logical), spec


def _source_files():
    root = Path(__file__).resolve().parents[1]
    registry = read_bounded_state(root / "engine/models/registry.json", 16 * 1024 * 1024)
    return {**source_closure(root, ["experiments/size_search.py", "experiments/magnitude_search_worker.py"]), "engine/models/registry.json": "sha256:" + sha256(registry).hexdigest()}


try:
    _IMPORTED_SOURCES = _source_files()
except (BoundedStateReadError, OSError):
    _IMPORTED_SOURCES = None
_IMPORTED_ENVIRONMENT = environment_identity(1)


def _sources():
    try:
        current = _source_files()
    except (BoundedStateReadError, OSError) as error:
        raise fail("RESOURCE_LIMIT_EXCEEDED" if getattr(error, "code", None) == "SIZE_LIMIT" else "RESOURCE_UNAVAILABLE", "installed registry unavailable") from None
    _require(current == _IMPORTED_SOURCES and environment_identity(1) == _IMPORTED_ENVIRONMENT)
    return current


def verify_installation(binding):
    _require(binding["schema_version"] in ("native_experiment_registration.v1.6", SCHEMA)
        and binding["environment"] == environment_identity(1)
        and binding["source_closure"] == _sources())
    _require(binding["size_comparison"] == comparison_recipe())


def _folds(binding, request):
    from experiments.magnitude_search_plan import _nested
    context = binding["size_panel"]["context"]
    events = {e["event_id"]: e for e in context["target_events"]}
    full_start, full_end = date.min.isoformat(), date.max.isoformat()
    embargoed_tickers = {w["ticker"] for w in context.get("eligibility_plan", {}).get("random_windows", [])
        if w["start"] == full_start and w["end"] == full_end}
    def hit(event_id):
        return bool(embargoed_tickers) and events[event_id]["ticker"] in embargoed_tickers
    folds = _nested(binding, request)
    for group in folds:
        for fold in [group["outer"], *group["inner"]]:
            fold["training_candidate_event_ids"] = [e for e in fold["training_candidate_event_ids"]
                if events[e]["label_end"] < fold["training_labels_before"]
                and not hit(e)]
            _require(bool(fold["training_candidate_event_ids"]))
            retained = [e for e in fold["evaluation_event_ids"] if not hit(e)]
            if embargoed_tickers and not retained:
                raise fail("INVALID_EXPERIMENT_SPEC", "ambiguous-random ticker embargo removed every fold evaluation event")
            excluded = [e for e in fold["evaluation_event_ids"] if hit(e)]
            fold["evaluation_event_ids"] = retained
            if excluded:
                fold["evaluation_exclusions"] = [{"event_id": eid, "reasons": ["embargo_ticker"]}
                    for eid in excluded]
            cutoff = fold.get("evaluation_labels_before", binding["magnitude_execution"]["evaluation_labels_before"])
            _require(all(events[e]["label_end"] < cutoff for e in fold["evaluation_event_ids"]))
    return folds


def _shared_inner(folds):
    """Repeated v3 inner years must describe one immutable trial after eligibility."""
    seen = {}
    for owner in folds:
        for inner in owner["inner"]:
            prior = seen.setdefault(inner["year"], inner)
            if (prior["training_candidate_event_ids"] != inner["training_candidate_event_ids"]
                    or prior["evaluation_event_ids"] != inner["evaluation_event_ids"]):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT",
                    "repeated inner-year folds differ after eligibility filtering")


def prepare_plan(conn, store, registration, request, *, plan_schema=None):
    """Expand the fixed dictionary grid into fully bound executable recipes."""
    from experiments.magnitude_search_plan import PLAN_V3
    plan_schema = plan_schema or PLAN_V3
    verify_native_registration(conn, store, registration, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
    binding = registration.document
    verify_installation(binding)
    schedule = compile_request(request)
    _require(schedule["request"] == binding["size_request"])
    folds = _folds(binding, request)
    if plan_schema == PLAN_V3:
        _shared_inner(folds)
    candidates = []
    source_binding_id = (content_hash(json.loads(registration.binding_json)["size_panel"])
                         if binding["schema_version"] == "native_experiment_registration.v1.7"
                         else content_hash(binding["size_panel"]))
    for spec in schedule["candidate_specs"]:
        recipe = candidate_recipe(spec["family"], spec["parameters"], columns=spec["feature_recipe"]["model_columns"],
                                  feature_dictionary_hash=spec["feature_recipe"]["dictionary_hash"])
        identity = {"recipe": recipe, "feature_recipe": spec["feature_recipe"],
                    "source_binding_id": source_binding_id, "candidate_spec_id": spec["candidate_spec_id"]}
        candidates.append({**identity, "candidate_id": content_hash(identity)})
    identity = {"schema_version": plan_schema, "request": schedule["request"],
        "anchor": {"run_id": registration.run_id, "variant_id": registration.variant_id},
        "implementation": {"source_closure": binding["source_closure"], "environment": binding["environment"]},
        "candidates": sorted(candidates, key=lambda c: c["candidate_id"]), "folds": folds,
        "horizons": [5, 3, 1], "candidate_count": schedule["counts"]["candidates"],
        "planned_fit_count": schedule["counts"]["search_fits"], "counts": schedule["counts"],
        "selection": {"metric": "inner_pooled_rmse", "ties": "candidate_id_ascending", "retries": 0,
                      "outer": "frozen_selected_only", "final_read_authority": "none"}}
    return {**identity, "plan_id": content_hash(identity)}


def execution_for_candidate(binding, candidate):
    execution = {**binding["magnitude_execution"], "fit_recipe": candidate["recipe"]}
    if binding.get("schema_version") in ("native_experiment_registration.v1.6", SCHEMA):
        execution.update(schema_version="size_candidate_execution.v1", features=candidate["recipe"]["features"],
                         model=candidate["recipe"]["model"])
    return execution


def fold_inputs(repository, binding, fold):
    """Revalidate installed producer before opening any training outcome."""
    from engine.v2.research.magnitude_panel_qualified_table import recipe
    verify_installation(binding)
    _require(any(fold in group["inner"] or fold == {**group["outer"],
        "evaluation_labels_before": binding["magnitude_execution"]["evaluation_labels_before"]}
        for group in _folds(binding, binding["size_request"])))
    source = from_document(SnapshotRef, binding["snapshot"])
    started = time.monotonic()
    panel = _stored_panel(repository, source, binding, recipe("HMO")["model_columns"])
    participants = set(fold["training_candidate_event_ids"] + fold["evaluation_event_ids"])
    events = {e["event_id"]: e for e in panel["context"]["target_events"]}
    _require(participants <= set(events) and all(events[e]["label_end"] < fold["training_labels_before"]
             for e in fold["training_candidate_event_ids"]))
    print(f"[stage] read_panel done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    month = binding["holdouts"]["holdout_as_of_month"]
    started = time.monotonic()
    preflight = preflight_magnitude_targets(repository, source, as_of_month=month, event_ids=sorted(participants))
    print(f"[stage] preflight done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    started = time.monotonic()
    train = qualify_magnitude_training_targets(repository, source, as_of_month=month,
        labels_before=fold["training_labels_before"], event_ids=fold["training_candidate_event_ids"])
    print(f"[stage] qualify done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    started = time.monotonic()
    result = {"preflight": preflight, "stored_inputs_id": content_hash(binding["size_panel"]),
        "training_inputs": dict(train.attrs), "training": train[["event_id", "absolute_move_pct"]].to_dict("records"),
        "features": [r for r in panel["rows"] if r["event_id"] in participants]}
    print(f"[stage] publish done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return result


def validator_identity():
    """Source closure of the SIZE entrypoint plus environment; the receipt's validator binding."""
    return {"source_closure": _sources(), "environment": environment_identity(1)}
