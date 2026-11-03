"""Prepare a distinct prediction registration over a fresh qualified SIZE anchor."""
from pathlib import Path

from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import from_document
from engine.v2.ops.errors import fail
from experiments.distribution_candidate_recipe import fit_budget
from experiments.distribution_prediction_plan import require
from experiments.magnitude_search_plan import read_search_plan
from experiments.native_registration import MAX_NATIVE_EVIDENCE_BYTES, prepare_native_registration


def require_spec(spec):
    """The prediction lifecycle cannot adopt another runner's owner registration."""
    require(spec.runner == "experiments/distribution_prediction_run.py" and spec.seed == 0
            and type(spec.seed) is int and not spec.economic_params
            and spec.price_source == "computed_moves" and not spec.input_files)


def _prepare(conn, store, repository, spec, source, source_request, ref, recipe, caps):
    """Prepare only the distinct distribution registration, never rebuild or run SIZE."""
    require_spec(spec)
    native = read_search_plan(conn, store, source, source_request, ref)
    binding = source.document
    require(binding["schema_version"] == "native_experiment_registration.v1.7"
            and set(native["horizons"]) >= {5, 3, 1}
            and list(spec.folds) == [group["outer"]["year"] for group in native["folds"]])
    require(type(caps) is dict and set(caps) == {"max_fits", "max_trees", "child_limits"})
    budget = fit_budget(recipe, inner_folds=sum(len(group["inner"]) for group in native["folds"]),
        outer_folds=len(native["folds"]), **caps)
    registration, spec = prepare_native_registration(repository, spec,
        code_root=Path(__file__).resolve().parents[1],
        as_of_month=binding["holdouts"]["holdout_as_of_month"], scope=binding["scope"],
        event_ids=binding["event_ids"], snapshot=from_document(SnapshotRef, binding["snapshot"]))
    if len(registration.binding_json) > MAX_NATIVE_EVIDENCE_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution registration exceeds native byte budget")
    own = registration.document
    require(registration.run_id != source.run_id and own["schema_version"] == "native_experiment_registration.v1.0"
            and "final_read" not in own and all(own[key] == binding[key] for key in ("snapshot", "scope", "event_ids", "holdouts")))
    result = {"schema_version": "distribution_prediction_preparation.v1", "run_id": registration.run_id,
        "variant_id": registration.variant_id, "source_plan_id": native["plan_id"],
        "fit_budget": budget, "final_read_authority": "none"}
    return registration, spec, result
