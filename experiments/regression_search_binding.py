"""Target-aware nested plan binding using the existing native search mechanisms."""
from engine.v2.foundation import content_hash
from engine.v2.models.regression_candidate import regression_candidate_recipe
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_panel_qualified_table import recipe as panel_recipe
from experiments.magnitude_search_request_versions import compile_request
from experiments.native_registration import MAX_NATIVE_EVIDENCE_BYTES, verify_native_registration
from experiments.size_panel_request import validate_panel_request


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "regression search differs from its native binding")


def prepare_plan(conn, store, registration, request, *, target):
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
    identity = {"schema_version": "regression_search_plan.v1", "request": request,
        "anchor": {"run_id": registration.run_id, "variant_id": registration.variant_id},
        "implementation": implementation, "target": target, "folds": folds,
        "candidates": sorted(candidates, key=lambda candidate: candidate["candidate_id"]),
        "horizons": schedule["horizons"], "candidate_count": schedule["counts"]["candidates"],
        "planned_fit_count": schedule["counts"]["search_fits"], "counts": schedule["counts"],
        "selection": {"metric": "inner_pooled_rmse", "ties": "candidate_id_ascending", "retries": 0,
            "outer": "frozen_selected_only", "final_read_authority": "none"}}
    return {**identity, "plan_id": content_hash(identity)}
