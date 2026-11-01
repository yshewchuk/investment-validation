"""Target-aware nested plan binding using the existing native search mechanisms."""
from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import content_hash, from_document
from engine.v2.models.regression_candidate import regression_candidate_recipe
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_panel_qualified_table import recipe as panel_recipe
from experiments.magnitude_search_request_versions import compile_request
from experiments.native_registration import MAX_NATIVE_EVIDENCE_BYTES, verify_native_registration
from experiments.size_panel_request import validate_panel_request
from experiments.size_search_binding import read_stored_panel


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "regression search differs from its native binding")


def prepare_plan(conn, store, registration, request, *, target, comparison):
    """Bind metadata only; installed fold readers later qualify causal payloads."""
    from experiments.magnitude_search_plan import _implementation, _nested
    if type(request) is not dict or request.get("schema_version") != "regression_search_request.v1":
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
        and request["feature_dictionary_hash"] == panel_recipe("HMO")["dictionary_hash"]
        and request["feature_snapshot_id"] == panel["feature_snapshot_id"] == binding["feature_snapshot"]["snapshot_id"]
        and request["history_snapshot_id"] == panel["history_snapshot"]
        and panel["as_of_month"] == binding["holdouts"]["holdout_as_of_month"]
        and set(panel["event_ids"]) == set(binding["event_ids"])
        and panel["policy"]["policy_id"] == request["eligibility_policy_ref"]["policy_id"])
    implementation = _implementation(binding, entrypoints=[binding["execution_plan"]["runner"],
        "experiments/magnitude_search_worker.py"])
    folds = _nested(binding, request, temporal=temporal)
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
        "selection": {"metric": "inner_pooled_rmse", "ties": "candidate_id_ascending", "retries": 0,
            "outer": "frozen_selected_only", "final_read_authority": "none"}}
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
    _require(plan["folds"] == folds and any(fold in group["inner"] or fold == {**group["outer"],
        "evaluation_labels_before": binding["regression_execution"]["evaluation_labels_before"]}
        for group in folds))
    participants = set(fold["training_candidate_event_ids"] + fold["evaluation_event_ids"])
    month = binding["holdouts"]["holdout_as_of_month"]
    snapshot = from_document(SnapshotRef, binding["target_snapshot"])
    preflight = preflight_iv_crush_targets(repository, snapshot, as_of_month=month, event_ids=sorted(participants))
    panel = read_stored_panel(repository, from_document(SnapshotRef, binding["snapshot"]),
        plan["request"]["panel_request"], columns=panel_recipe("HMO")["model_columns"])
    context = panel["context"]
    _require(panel["feature_snapshot"] == binding["feature_snapshot"]
        and plan["request"]["eligibility_policy_ref"] == {"schema_ref": "group_eligibility_plan.v1",
            "plan_id": context["eligibility_plan"]["plan_id"], "policy_id": context["eligibility_plan"]["policy"]["policy_id"]}
        and participants <= {event["event_id"] for event in context["target_events"]})
    train = load_iv_crush_targets(repository, snapshot, as_of_month=month,
        labels_before=fold["training_labels_before"], event_ids=fold["training_candidate_event_ids"])
    return {"preflight": preflight, "target": plan["target"],
        "stored_inputs_id": content_hash({"context": context, "feature_snapshot": panel["feature_snapshot"]}),
        "training_inputs": dict(train.attrs), "training": train[["event_id", plan["target"]["column"]]].to_dict("records"),
        "features": [row for row in panel["rows"] if row["event_id"] in participants]}
