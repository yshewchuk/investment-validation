"""Target-aware nested plan binding using the existing native search mechanisms."""
import sys
from collections import Counter

from engine.v2.contracts import SnapshotRef
from engine.v2.data.repository import Repository
from engine.v2.foundation import content_hash, from_document
from engine.v2.models.regression_candidate import regression_candidate_recipe
from engine.v2.ops.errors import fail
from engine.v2.research.iv_crush_inputs import skipped_iv_crush_targets
from engine.v2.research.magnitude_panel_qualified_table import recipe as panel_recipe
from experiments.magnitude_search_request_schema import regression_request_version
from experiments.magnitude_search_request_versions import compile_request
from experiments.native_registration import MAX_NATIVE_EVIDENCE_BYTES, verify_native_registration
from experiments.size_panel_request import validate_panel_request
from experiments.size_search_binding import read_stored_panel

_STORED = {"HMO": 0, "HMOS": 1, "HMOSX": 2, "HMOSXL": 2, "HMOSXLP": 2}
_STORED_NAME = ("HMO", "HMOS", "HMOSX")
_SKIP_SCANS = {}


def _skipped_targets(repository, snapshot, month, event_ids):
    """Plan reads recompute the scan; one process reads each (snapshot, month, ids) once."""
    key = (snapshot.snapshot_id, month, tuple(sorted(event_ids)))
    if key not in _SKIP_SCANS:
        if len(_SKIP_SCANS) >= 4:
            _SKIP_SCANS.pop(next(iter(_SKIP_SCANS)))
        _SKIP_SCANS[key] = skipped_iv_crush_targets(repository, snapshot, as_of_month=month,
                                                    event_ids=list(key[2]))
    return dict(_SKIP_SCANS[key])


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "regression search differs from its native binding")


def panel_coverage(event_ids, panel_event_ids):
    """Measure how many IV events the SIZE panel holds; a shortfall is reported, never dropped."""
    ids, panel = set(event_ids), set(panel_event_ids)
    return {"iv_events": len(ids), "in_size_panel": len(ids & panel), "outside": len(ids - panel)}


def _apply_label_exclusions(folds, excluded):
    """Drop skipped-label events from each fold's training and evaluation cohorts."""
    def filtered(fold):
        result, removed = dict(fold), []
        for key in ("training_candidate_event_ids", "evaluation_event_ids"):
            original = fold[key]
            kept = [event_id for event_id in original if event_id not in excluded]
            if len(kept) != len(original):
                result[key] = kept
                removed.extend(event_id for event_id in original if event_id in excluded)
        if removed:
            result["label_exclusions"] = [{"event_id": event_id, "reasons": [excluded[event_id]]}
                                          for event_id in sorted(set(removed))]
            if not result["evaluation_event_ids"] or not result["training_candidate_event_ids"]:
                raise fail("INVALID_EXPERIMENT_SPEC",
                           "regression fold has no events left after IV label exclusions")
        return result
    return [{**group, "outer": filtered(group["outer"]),
             "inner": [filtered(fold) for fold in group["inner"]]} for group in folds]


def stored_columns(request):
    """Stored panel columns by request version: HMO for v1, the widest declared stored subset for v2."""
    if regression_request_version(request["schema_version"]) == 1:
        return panel_recipe("HMO")["model_columns"]
    return panel_recipe(_STORED_NAME[max(_STORED[name] for name in request["feature_sets"])])["model_columns"]


def prepare_plan(conn, store, registration, request, *, target, comparison):
    """Bind metadata only; installed fold readers later qualify causal payloads."""
    from experiments.magnitude_search_plan import _implementation, _nested
    version = regression_request_version(request.get("schema_version")) if type(request) is dict else None
    if version not in (1, 2):
        raise fail("INVALID_EXPERIMENT_SPEC", "regression search requires its explicit request")
    schedule = compile_request(request)
    request = schedule["request"]
    panel = validate_panel_request(request["panel_request"])
    if request["limits"]["child"]["memory_bytes"] > request["limits"]["parent"]["memory_bytes"]:
        raise fail("INVALID_EXPERIMENT_SPEC", "child memory must not exceed inherited parent limit")
    verify_native_registration(conn, store, registration, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
    binding = registration.document
    _require(binding["schema_version"] == "native_experiment_registration.v1.8")
    temporal = binding["regression_plan"]
    _require(temporal["target_recipe"] == target
        and temporal["evaluation_years"] == request["outer_years"]
        and request["feature_dictionary_hash"] == panel_recipe("HMO" if version == 1 else "HMOSXLP")["dictionary_hash"]
        and request["feature_snapshot_id"] == panel["feature_snapshot_id"] == binding["feature_snapshot"]["snapshot_id"]
        and request["history_snapshot_id"] == panel["history_snapshot"]
        and panel["as_of_month"] == binding["holdouts"]["holdout_as_of_month"]
        and (version == 2 or set(panel["event_ids"]) == set(binding["event_ids"]))
        and panel["policy"]["policy_id"] == request["eligibility_policy_ref"]["policy_id"])
    if version == 2:
        coverage = panel_coverage(binding["event_ids"], panel["event_ids"])
        print(f"[iv] panel_coverage iv_events={coverage['iv_events']} in_size_panel={coverage['in_size_panel']} outside={coverage['outside']}", file=sys.stderr)
        _require(coverage["outside"] == 0)
    implementation = _implementation(binding, entrypoints=[binding["execution_plan"]["runner"],
        "experiments/magnitude_search_worker.py"])
    folds = _nested(binding, request, temporal=temporal)
    label_exclusions = None
    if "target_snapshot" in binding:
        repository = Repository(conn, store, pin_snapshots=True)
        excluded = _skipped_targets(repository,
            from_document(SnapshotRef, binding["target_snapshot"]),
            binding["holdouts"]["holdout_as_of_month"], binding["event_ids"])
        folds = _apply_label_exclusions(folds, excluded)
        label_exclusions = {"counts": dict(sorted(Counter(excluded.values()).items())),
            "events": [{"event_id": event_id, "reason": reason}
                       for event_id, reason in sorted(excluded.items())]}
        print(f"[iv] label_exclusions target_skipped_excluded={len(excluded)} by_reason="
              + ", ".join(f"{reason}={count}" for reason, count in label_exclusions["counts"].items()),
              file=sys.stderr)
    candidates = []
    for spec in schedule["candidate_specs"]:
        recipe = regression_candidate_recipe(spec["family"], spec["parameters"],
            target_domain=target["domain"], columns=spec["feature_recipe"]["model_columns"],
            feature_dictionary_hash=spec["feature_recipe"]["dictionary_hash"])
        identity = {"recipe": recipe, "feature_recipe": spec["feature_recipe"],
            "source_binding_id": registration.variant_id, "target_id": target["target_id"],
            "candidate_spec_id": spec["candidate_spec_id"]}
        candidates.append({**identity, "candidate_id": content_hash(identity)})
    identity = {"schema_version": "regression_search_plan.v1", "comparison": comparison, "request": request,
        "anchor": {"run_id": registration.run_id, "variant_id": registration.variant_id},
        "implementation": implementation, "target": target, "folds": folds,
        "candidates": sorted(candidates, key=lambda candidate: candidate["candidate_id"]),
        "horizons": schedule["horizons"], "candidate_count": schedule["counts"]["candidates"],
        "planned_fit_count": schedule["counts"]["search_fits"], "counts": schedule["counts"],
        "selection": {"metric": "inner_pooled_rmse" if version == 1 else "blend_rmse_mae",
            "ties": "candidate_id_ascending" if version == 1 else "lowest_score_then_candidate_id_ascending", "retries": 0,
            "outer": "frozen_selected_only", "final_read_authority": "none"}}
    if label_exclusions is not None and label_exclusions["events"]:
        identity["label_exclusions"] = label_exclusions
    return {**identity, "plan_id": content_hash(identity)}


def fold_inputs(repository, binding, fold, *, plan):
    """Admit causal native inputs for a caller-verified sealed IV plan, without evaluation labels."""
    from engine.v2.research.iv_crush_inputs import load_iv_crush_targets, preflight_iv_crush_targets
    from experiments.magnitude_search_plan import _implementation, _nested
    _require(type(plan) is dict and plan.get("schema_version") == "regression_search_plan.v1"
        and plan.get("target") == binding["regression_plan"]["target_recipe"])
    _require(plan["implementation"] == _implementation(binding, entrypoints=[
        binding["execution_plan"]["runner"], "experiments/magnitude_search_worker.py"]))
    folds = _nested(binding, plan["request"], temporal=binding["regression_plan"])
    folds = _apply_label_exclusions(folds, {entry["event_id"]: entry["reason"]
        for entry in plan.get("label_exclusions", {}).get("events", [])})
    _require(plan["folds"] == folds and any(fold in group["inner"] or fold == {**group["outer"],
        "evaluation_labels_before": binding["regression_execution"]["evaluation_labels_before"]}
        for group in folds))
    participants = set(fold["training_candidate_event_ids"] + fold["evaluation_event_ids"])
    month = binding["holdouts"]["holdout_as_of_month"]
    snapshot = from_document(SnapshotRef, binding["target_snapshot"])
    preflight = preflight_iv_crush_targets(repository, snapshot, as_of_month=month, event_ids=sorted(participants))
    train = load_iv_crush_targets(repository, snapshot, as_of_month=month,
        labels_before=fold["training_labels_before"], event_ids=fold["training_candidate_event_ids"])
    panel = read_stored_panel(repository, from_document(SnapshotRef, binding["snapshot"]),
        plan["request"]["panel_request"], columns=stored_columns(plan["request"]))
    context = panel["context"]
    _require(panel["feature_snapshot"] == binding["feature_snapshot"]
        and plan["request"]["eligibility_policy_ref"] == {"schema_ref": "group_eligibility_plan.v1",
            "plan_id": context["eligibility_plan"]["plan_id"], "policy_id": context["eligibility_plan"]["policy"]["policy_id"]}
        and participants <= {event["event_id"] for event in context["target_events"]})
    return {"preflight": preflight, "target": plan["target"],
        "stored_inputs_id": content_hash({"context": context, "feature_snapshot": panel["feature_snapshot"]}),
        "training_inputs": dict(train.attrs), "training": train[["event_id", plan["target"]["column"]]].to_dict("records"),
        "features": [row for row in panel["rows"] if row["event_id"] in participants]}
