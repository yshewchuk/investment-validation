"""Distribution-owned admission plan over verified native SIZE source/fold metadata.

No registration, feature build, SIZE execution, fitting or holdout authority.
"""
from pathlib import Path

from engine.v2.foundation import artifact_reference, content_hash, to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import environment_identity, source_closure
from experiments.distribution_candidate_recipe import fit_budget, validate_candidate_recipe
from experiments.magnitude_search_plan import read_search_plan
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    NativeRegistration,
    verify_native_registration,
)

__all__ = ["prepare_plan"]


def prepare_plan(conn, store, own_registration, source_registration, source_request,
                 source_plan_ref, recipe, budget_caps):
    """Return (small detached plan, verified native plan); never copy rows into identity.

    ``budget_caps`` has exactly max_fits, max_trees and child_limits. The source
    plan supplies folds only: its 33-candidate execution budget is not adopted.
    """
    for registration in (own_registration, source_registration):
        if not isinstance(registration, NativeRegistration) or type(registration.binding_json) is not bytes:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution requires native registrations")
        if len(registration.binding_json) > MAX_NATIVE_EVIDENCE_BYTES:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution registration exceeds native byte budget")
    recipe = validate_candidate_recipe(recipe)
    if type(budget_caps) is not dict or set(budget_caps) != {"max_fits", "max_trees", "child_limits"}:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution budget caps")
    verify_native_registration(conn, store, own_registration, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
    # The public source-plan reader verifies the stored source registration too.
    native = read_search_plan(conn, store, source_registration, source_request, source_plan_ref)
    own, source = own_registration.document, source_registration.document
    execution = own["execution_plan"]
    if (own["schema_version"] != "native_experiment_registration.v1.0"
            or source["schema_version"] != "native_experiment_registration.v1.7"
            or own_registration.run_id == source_registration.run_id
            or any(own[key] != source[key] for key in ("snapshot", "scope", "holdouts", "event_ids"))
            or execution["runner"] == "experiments/size_search.py"
            or type(execution["seed"]) is not int or execution["seed"] != 0
            or execution["price_source"] != "computed_moves" or execution["economic_params"]
            or execution["folds"] != [group["outer"]["year"] for group in native["folds"]]
            or "final_read" in own):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution source and owner bindings differ")
    sources = source_closure(Path(__file__).resolve().parents[1], [execution["runner"],
        "experiments/distribution_plan.py", "experiments/distribution_trials.py",
        "experiments/distribution_class_support.py", "experiments/distribution_fitted.py",
        "experiments/distribution_oof.py"])
    environment = environment_identity(1)
    if (execution["runner"] not in own["source_closure"] or own["environment"] != environment or any(sources.get(key) != digest
            for key, digest in own["source_closure"].items())):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution installation differs from its registration")
    budget = fit_budget(recipe, inner_folds=sum(len(group["inner"]) for group in native["folds"]),
                        outer_folds=len(native["folds"]), **budget_caps)
    registrations = {}
    for name, registration, binding in (("own", own_registration, own), ("source", source_registration, source)):
        registrations[name] = {"run_id": registration.run_id, "variant_id": registration.variant_id,
            "registration_ref": to_document(artifact_reference(registration.binding_json,
                binding["schema_version"]))}
    identity = {"schema_version": "distribution_plan.v1", "registrations": registrations,
        "source_plan_ref": to_document(source_plan_ref), "source_plan_id": native["plan_id"],
        "context_id": source["size_panel"]["context"]["context_id"],
        "implementation": {"source_closure": sources, "environment": environment},
        "recipe": recipe, "budget": budget, "final_read_authority": "none"}
    return {**identity, "plan_id": content_hash(identity)}, native
