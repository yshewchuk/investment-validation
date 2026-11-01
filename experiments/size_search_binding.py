"""Explicit v2 SIZE source/fold binding; installed panel admission, no hidden builds."""
import copy
import json
import sys
import time
from datetime import date
from hashlib import sha256
from pathlib import Path

from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import (
    artifact_reference,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.models.frozen_state_io import BoundedStateReadError, read_bounded_state
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.research.magnitude_inputs import (
    preflight_magnitude_targets,
    qualify_magnitude_training_targets,
)
from engine.v2.research.magnitude_panel_qualified_table import METADATA, admit_columns
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_search_request_versions import compile_request
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    NativeRegistration,
    prepare_native_registration,
    verify_native_registration,
)
from experiments.size_comparator import COLUMNS
from experiments.size_comparator import candidate_recipe as size_candidate_recipe
from experiments.size_comparator import recipe as comparison_recipe
from experiments.size_panel_request import validate_panel_request
from experiments.size_registration_artifacts import (
    pack_size_registration,
    size_registration_artifact_bytes,
)

SCHEMA = "native_experiment_registration.v1.7"


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "qualified SIZE source/fold binding differs")


def _panel_columns(schema_version, model_columns):
    if schema_version == "magnitude_search_request.v5":
        return list(dict.fromkeys(c for c in COLUMNS + list(model_columns) if c not in METADATA))
    return model_columns


class _PreviewStore:
    def publish_bytes(self, payload, schema_ref=None):
        return artifact_reference(payload, schema_ref)


def _preview_payload(binding):
    compact, _ = pack_size_registration(_PreviewStore(), binding)
    size_registration_artifact_bytes(compact)
    return canonical_json(compact).encode("utf-8")


def read_panel(repository, source, request, *, columns):
    """Resolve the pinned request and delegate installed qualified-panel admission."""
    from experiments.size_panel import read
    args = dict(request)
    feature = repository.resolve(args.pop("feature_snapshot_id"))
    for name in ("history_snapshot", "price_history_snapshot"):
        args[name] = repository.resolve(args[name])
    return read(repository, source, feature, columns=columns, lineage=False, **args)


_PANEL_MEMO = {}


def read_stored_panel(repository, source, request, *, columns, expected_panel=None):
    """Share one admitted panel per process; validate each request and return detached copies."""
    validate_panel_request(request)
    admit_columns(columns)
    key = (id(repository), content_hash(to_document(source)),
           content_hash(request), tuple(columns))
    entry = _PANEL_MEMO.get(key)
    if entry is not None and entry["repository"] is repository:
        panel = entry["panel"]
        fingerprint = content_hash({"context": panel["context"], "feature_snapshot": panel["feature_snapshot"]})
        _require(fingerprint == entry["fingerprint"])
        if expected_panel is not None:
            _require(fingerprint == content_hash(expected_panel))
    else:
        panel = read_panel(repository, source, request, columns=columns)
        metadata = {"context": panel["context"], "feature_snapshot": panel["feature_snapshot"]}
        _require(expected_panel is None or metadata == expected_panel)
        _PANEL_MEMO.clear()
        _PANEL_MEMO[key] = {"repository": repository, "panel": panel, "fingerprint": content_hash(metadata)}
    return copy.deepcopy(panel)


def _stored_panel(repository, source, binding, columns):
    """Preserve SIZE's expected metadata admission over the shared one-panel memo."""
    return read_stored_panel(repository, source, binding["size_panel_request"],
                             columns=columns, expected_panel=binding["size_panel"])


def prepare(repository, spec, request, panel_request, *, scope, training_start, labels_before,
            publish_artifacts=True):
    """Prepare a new development anchor; caller explicitly reserves it separately."""
    from engine.v2.research.magnitude_panel_qualified_table import recipe
    sources = _sources()
    panel_request = validate_panel_request(panel_request)
    schedule = compile_request(request, horizons=tuple(h for h in (5, 3, 1) if h in panel_request["horizons"]))
    if request["limits"]["child"]["memory_bytes"] > request["limits"]["parent"]["memory_bytes"]:
        raise fail("INVALID_EXPERIMENT_SPEC", "child memory must not exceed inherited parent limit")
    _require(request["feature_dictionary_hash"] == recipe("HMO")["dictionary_hash"])
    _require(panel_request["feature_snapshot_id"] == request["feature_snapshot_id"]
             and panel_request["history_snapshot"] == request["history_snapshot_id"]
             and set(panel_request["horizons"]) == set(schedule["horizons"]))
    base, spec = prepare_native_registration(repository, spec, code_root=Path(__file__).resolve().parents[1],
        scope=scope, as_of_month=panel_request["as_of_month"], event_ids=panel_request["event_ids"],
        magnitude_training_start=training_start, magnitude_labels_before=labels_before, defer_serialization=True)
    binding = base.document
    _require(binding["execution_plan"]["runner"] == "experiments/size_search.py")
    binding["source_closure"] = sources
    _require(binding["magnitude_plan"]["evaluation_years"] == schedule["request"]["outer_years"])
    panel_columns = _panel_columns(request["schema_version"], recipe("HMO")["model_columns"])
    panel = read_panel(repository, from_document(SnapshotRef, binding["snapshot"]), panel_request,
                        columns=panel_columns)
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
    from experiments.magnitude_search_plan import PLAN_V2, PLAN_V3
    request_version = request.get("schema_version") if isinstance(request, dict) else None
    if plan_schema is None:
        plan_schema = PLAN_V2 if request_version == "magnitude_search_request.v3" else PLAN_V3
    if request_version == "magnitude_search_request.v3" and plan_schema != PLAN_V2:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search plan schema does not match request version")
    if plan_schema not in (PLAN_V2, PLAN_V3):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "unsupported SIZE search plan schema")
    verify_native_registration(conn, store, registration, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
    binding = registration.document
    verify_installation(binding)
    if binding["schema_version"] == "native_experiment_registration.v1.6":
        horizons = (5, 3, 1)
    else:
        panel_request = binding.get("size_panel_request")
        if not isinstance(panel_request, dict):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "qualified SIZE registration is missing its panel request")
        validate_panel_request(panel_request)
        horizons = tuple(h for h in (5, 3, 1) if h in panel_request["horizons"])
    schedule = compile_request(request, horizons=horizons)
    _require(schedule["request"] == binding["size_request"])
    folds = _folds(binding, request)
    if plan_schema == PLAN_V3:
        _shared_inner(folds)
    candidates = []
    source_binding_id = (content_hash(json.loads(registration.binding_json)["size_panel"])
                         if binding["schema_version"] == "native_experiment_registration.v1.7"
                         else content_hash(binding["size_panel"]))
    for spec in schedule["candidate_specs"]:
        recipe = (size_candidate_recipe(binding["size_comparison"]) if spec["family"] == "size_v1_4" else
            candidate_recipe(spec["family"], spec["parameters"], columns=spec["feature_recipe"]["model_columns"],
                             feature_dictionary_hash=spec["feature_recipe"]["dictionary_hash"]))
        identity = {"recipe": recipe, "feature_recipe": spec["feature_recipe"],
                    "source_binding_id": source_binding_id, "candidate_spec_id": spec["candidate_spec_id"]}
        candidates.append({**identity, "candidate_id": content_hash(identity)})
    requested_selection = schedule["request"].get("selection")
    if requested_selection is None:
        selection_contract = {"metric": "inner_pooled_rmse", "ties": "candidate_id_ascending", "retries": 0,
                              "outer": "frozen_selected_only", "final_read_authority": "none"}
    else:
        selection_contract = {
            "metric": "inner_pooled_history_mean_normalized_event_blend",
            "metric_weights": requested_selection["metric_weights"],
            "lead_weights": requested_selection["lead_weights"],
            "large_move_threshold_pct": requested_selection["large_move_threshold_pct"],
            "pooling": "all_inner_year_rows",
            "comparator": "history_mean_report_only",
            "tie_band": "max_sample_se_of_best_pooled_candidate_yearly_blends_or_floor",
            "tie_band_floor": requested_selection["tie_band_floor"],
            "ties": "candidate_id_ascending",
            "retries": 0,
            "outer": "frozen_selected_only",
            "final_read_authority": "none",
        }
    identity = {"schema_version": plan_schema, "request": schedule["request"],
        "anchor": {"run_id": registration.run_id, "variant_id": registration.variant_id},
        "implementation": {"source_closure": binding["source_closure"], "environment": binding["environment"]},
        "candidates": sorted(candidates, key=lambda c: c["candidate_id"]), "folds": folds,
        "horizons": schedule["horizons"], "candidate_count": schedule["counts"]["candidates"],
        "planned_fit_count": schedule["counts"]["search_fits"], "counts": schedule["counts"],
        "selection": selection_contract}
    return {**identity, "plan_id": content_hash(identity)}


def execution_for_candidate(binding, candidate):
    if binding.get("schema_version") == "native_experiment_registration.v1.8":
        return {**binding["regression_execution"], "schema_version": "regression_candidate_execution.v1",
            "target": binding["regression_plan"]["target_recipe"], "fit_recipe": candidate["recipe"],
            "features": candidate["recipe"]["features"], "model": candidate["recipe"]["model"]}
    execution = {**binding["magnitude_execution"], "fit_recipe": candidate["recipe"]}
    if binding.get("schema_version") in ("native_experiment_registration.v1.6", SCHEMA):
        size = candidate["recipe"].get("schema_version") == "size_history_mean_candidate.v1"
        features = COLUMNS if size else candidate["recipe"]["features"]
        model = "size_v1_4" if size else candidate["recipe"]["model"]
        execution.update(schema_version="size_candidate_execution.v1", features=features,
                         model=model)
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
    panel_columns = _panel_columns(binding["size_request"].get("schema_version"), recipe("HMO")["model_columns"])
    panel = _stored_panel(repository, source, binding, panel_columns)
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
