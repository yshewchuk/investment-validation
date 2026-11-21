"""EXP-188 S2a: bind stored v14 trial predictions to the SIZE selection population.

Only ``event_id`` and the lead-horizon predicted move are read from each stored
prediction. The whole stored object is hashed (bytes, not values); stored outcome values are never read or used. Run-level identity failures
refuse the whole run; per-candidate gaps become exclusions (no imputation).
A blend binds its component candidates' inner arrays; averaging is left to the reader slice.

No production entrypoint calls these functions yet; the S2b reader slice and the S9 driver are their first callers.
"""
import sys

import numpy as np

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments.magnitude_search_plan import trial_key
from experiments.magnitude_search_refits import blend_components
from experiments.size_search_winners import LOG1P_SCHEMAS

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_PRED = "predicted_absolute_move_pct"
_SEAL_SLOT = "driver-seal"

PAIRS = {"H": ("H-sq", "H-abs"), "E": ("E-raw", "E-log"), "N": ("N-raw", "N-log")}
PAIR_ROLES = ("H-sq", "H-abs", "E-raw", "E-log", "N-raw", "N-log")
PAIR_TRANSFORM = {"E-raw": "none", "N-raw": "none", "E-log": "log1p", "N-log": "log1p"}
_HGB_PAIR = {"learning_rate": 0.03, "l2_regularization": 10, "max_iter": 3000,
             "max_leaf_nodes": 15, "min_samples_leaf": 300}
_EN_PAIR = {"alpha": 0.03, "l1_ratio": 0.75, "n_knots": 6}
_MLP_PAIR = {"profile": "chrono15_relu64x32_l2_1"}
# Pre-registered candidate recipe per pair role (design section 3). "family" is a schema_version
# prefix; "params" are the built recipe's declared fields (estimator params, spline n_knots, MLP profile).
# The HMOSXLP feature set is NOT checked here: feature columns are bound by the plan hash.
PAIR_RECIPES = {
    "H-sq": {"family": "magnitude_ridge_hgb_recipe", "params": {**_HGB_PAIR, "loss": "squared_error"}},
    "H-abs": {"family": "magnitude_ridge_hgb_recipe", "params": {**_HGB_PAIR, "loss": "absolute_error"}},
    "E-raw": {"family": "magnitude_elasticnet_spline_recipe", "params": _EN_PAIR},
    "E-log": {"family": "magnitude_elasticnet_spline_recipe", "params": _EN_PAIR},
    "N-raw": {"family": "magnitude_mlp_recipe", "params": _MLP_PAIR},
    "N-log": {"family": "magnitude_mlp_recipe", "params": _MLP_PAIR},
}
ROLES = ("M0", "M1", "M2", "M3", "M4", "M5") + PAIR_ROLES
MAX_MODELS = 12
LEAD = 1


def _conflict(reason, message):
    return fail(_CONFLICT, message, details={"reason": reason})


def bind_run(plan, read, *, expected, panel_id):
    """Whole-run identity check (R1). Returns the actual identities."""
    seal = read(_SEAL_SLOT)
    if seal is None:
        raise _conflict("seal_unavailable", "the driver seal is not recorded")
    actual = {"plan_id": plan["plan_id"], "registration_id": plan["anchor"]["run_id"],
              "seal_id": content_hash(seal), "panel_id": panel_id}
    for key in ("plan_id", "registration_id", "seal_id", "panel_id"):
        if key not in expected or expected[key] != actual[key]:
            raise _conflict(key + "_mismatch", "run identity does not match the expected registration")
    if LEAD not in plan["horizons"]:
        raise _conflict("lead_unavailable", "the plan has no lead-horizon (1) predictions")
    return dict(actual)


def population_admit(repository, snapshot, as_of_month, *, max_object_bytes, load=None):
    """Return ``admit(event_ids)`` backed by the canonical population reader."""
    if isinstance(max_object_bytes, bool) or not isinstance(max_object_bytes, int) or max_object_bytes <= 0:
        raise fail("INVALID_EXPERIMENT_SPEC", "max_object_bytes must be a positive integer",
                   details={"reason": "max_object_bytes_invalid"})
    if load is None:
        from engine.v2.research.experiment_population import load_population
        load = load_population

    def admit(event_ids):
        ids = sorted(event_ids)
        frame = load(repository, snapshot, as_of_month=as_of_month,
                     purpose="selection", event_ids=ids, max_object_bytes=max_object_bytes)
        return {"admitted": sorted(frame["event_id"].tolist()), "dropped": []}

    return admit


def bind_array(record, sealed_ids, admit):
    """Bind one stored prediction object at the lead (LEAD) to its sealed event ids."""
    rows = [row for row in record["predictions"] if row["horizon_sessions"] == LEAD]
    if not rows:
        raise _conflict("lead_unavailable", "no predictions at the requested horizon")
    ids = [row["event_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise _conflict("duplicate_event_id", "a prediction repeats an event id")
    sealed = set(sealed_ids)
    if not set(ids) <= sealed:
        raise _conflict("event_ids_outside_population", "event ids fall outside the sealed population")
    if set(ids) < sealed:
        raise fail(_CONFLICT, "event ids are a strict subset of the sealed population",
                   details={"reason": "partial_event_ids"})
    result = admit(ids)
    if sorted(result["admitted"]) != sorted(ids) or result["dropped"]:
        raise _conflict("admission_dropped", "admission dropped or changed event ids")
    try:
        pred = np.asarray([row[_PRED] for row in rows], dtype=float)
    except (TypeError, ValueError):
        raise _conflict("nonfinite_prediction", "a prediction is not a finite number") from None
    if not np.all(np.isfinite(pred)):
        raise _conflict("nonfinite_prediction", "a prediction is not a finite number")
    return {"pred": pred, "event_ids": np.asarray(ids, dtype=object),
            "binding": content_hash({"array_object": content_hash(record), "admission": result})}


def bind_candidate(plan, read, candidate_id, admit):
    """Bind one candidate across every outer group and inner year (R1, no imputation)."""
    years, seen = {}, set()
    for group in plan["folds"]:
        outer_year = group["outer"]["year"]
        for inner in group["inner"]:
            key = trial_key(plan, outer_year, inner["year"], candidate_id)
            if key in seen:
                continue
            seen.add(key)
            year = str(inner["year"])
            if year in years:
                raise fail("INVALID_EXPERIMENT_SPEC", "an inner year repeats under distinct trials",
                           details={"reason": "repeated_inner_year"})
            record = read("terminal:" + key)
            if record is None:
                return {"status": "excluded", "reason": "missing"}
            if not isinstance(record, dict) or record.get("status") != "succeeded":
                return {"status": "excluded", "reason": "failed"}
            try:
                years[year] = bind_array(record, inner["evaluation_event_ids"], admit)
            except OpsError as exc:
                if exc.problem.details.get("reason") != "partial_event_ids":
                    raise
                return {"status": "excluded", "reason": "partial"}
    pairs = [[year, years[year]["binding"]] for year in sorted(years)]
    return {"status": "bound", "years": years, "binding": content_hash(pairs)}


def _transform(candidate):
    return "log1p" if candidate["recipe"]["schema_version"] in LOG1P_SCHEMAS else "none"


def _pick(cands, candidate_id, missing_reason, blends=frozenset()):
    if candidate_id is None:
        return None, missing_reason
    if candidate_id in blends:
        return ("v14", candidate_id, "mixed"), None
    if candidate_id not in cands:
        return None, "not_in_plan"
    return ("v14", candidate_id, _transform(cands[candidate_id])), None


def _blend(blends, blend_ids, role):
    blend_id = blend_ids.get(role)
    if blend_id is None:
        return None, "unmapped"
    if blend_id not in blends:
        return None, "not_in_plan"
    return ("v14", blend_id, "mixed"), None


def _recipe_fields(recipe):
    """Declared fields of a built recipe under their real names: estimator params, MLP profile, spline n_knots."""
    fields = {}
    estimator = recipe.get("estimator")
    if isinstance(estimator, dict) and isinstance(estimator.get("params"), dict):
        fields.update(estimator["params"])
    if "profile" in recipe:
        fields["profile"] = recipe["profile"]
    preprocessing = recipe.get("preprocessing")
    spline = preprocessing.get("spline") if isinstance(preprocessing, dict) else None
    if isinstance(spline, dict) and "n_knots" in spline:
        fields["n_knots"] = spline["n_knots"]
    return fields


def _recipe_matches(recipe, role):
    spec = PAIR_RECIPES[role]
    schema = recipe.get("schema_version") if isinstance(recipe, dict) else None
    if not isinstance(schema, str) or not (
            schema == spec["family"] or schema.startswith(spec["family"] + ".")):
        return False
    fields = _recipe_fields(recipe)
    return all(key in fields and fields[key] == value for key, value in spec["params"].items())


def _pair(cands, extra, pair_candidates, role):
    candidate_id = pair_candidates.get(role)
    if candidate_id is None:
        return None, "unmapped"
    if candidate_id not in extra:
        return None, "not_in_extra_refits"
    if candidate_id not in cands:
        return None, "not_in_plan"
    transform = _transform(cands[candidate_id])
    required = PAIR_TRANSFORM.get(role)
    if required is not None and transform != required:
        return None, "transform_mismatch"
    if not _recipe_matches(cands[candidate_id]["recipe"], role):
        return None, "recipe_mismatch"
    return ("v14", candidate_id, transform), None


def resolve_roster(plan, selection, *, outer_year, pair_candidates, blend_ids):
    """Pre-registered roster for one outer year. No P&L is read."""
    cands = {c["candidate_id"]: c for c in plan["candidates"]}
    blends = {b["blend_id"] for b in plan.get("blends", [])}
    selection_block = plan.get("selection")
    extra = set(selection_block.get("extra_refits") or ()) if isinstance(selection_block, dict) else set()
    items = [s for s in selection["selections"] if str(s["outer_year"]) == str(outer_year)]
    item = items[0] if items else {}
    winners = item.get("winners", {})
    found = {
        "M0": (("production", "size_v1_4", "none"), None),
        "M1": _pick(cands, item.get("selected_candidate_id"), "winner_unresolved", blends),
        "M2": _pick(cands, winners.get("rmse", {}).get("candidate_id"), "winner_unresolved", blends),
        "M3": _pick(cands, winners.get("mae", {}).get("candidate_id"), "winner_unresolved", blends),
        "M4": _blend(blends, blend_ids, "M4"),
        "M5": _blend(blends, blend_ids, "M5"),
    }
    for role in PAIR_ROLES:
        found[role] = _pair(cands, extra, pair_candidates, role)

    models, not_testable, role_model = {}, [], {}
    for role in ROLES:
        spec, reason = found[role]
        if spec is None:
            not_testable.append({"role": role, "reason": reason})
            continue
        source, candidate_id, transform = spec
        model_id = content_hash({"source": source, "candidate_id": candidate_id,
                                 "target_transform": transform})
        role_model[role] = model_id
        if model_id in models:
            models[model_id]["roles"].append(role)
        else:
            if source == "production":
                kind = "production"
            elif transform == "mixed":
                kind = "blend"
            else:
                kind = "candidate"
            models[model_id] = {"model_id": model_id, "source": source, "candidate_id": candidate_id,
                                "target_transform": transform, "kind": kind, "roles": [role]}
    if len(models) > MAX_MODELS:
        raise fail("INVALID_EXPERIMENT_SPEC", "the roster exceeds the model budget",
                   details={"reason": "too_many_models"})
    dead = {item["role"] for item in not_testable}
    pairs = {}
    for pid, members in PAIRS.items():
        ids = [role_model.get(role) for role in members]
        blocked = bool(dead & set(members)) or ids[0] == ids[1]
        pairs[pid] = "not_testable" if blocked else "testable"
    return {"models": list(models.values()), "not_testable": not_testable, "pairs": pairs}


def _bind_blend(plan, read, blend_id, admit):
    components = blend_components(plan, blend_id)
    if components is None:
        return {"status": "excluded", "reason": "blend_unknown"}
    bound = {cid: bind_candidate(plan, read, cid, admit) for cid in components}
    for cid in components:
        if bound[cid]["status"] != "bound":
            return {"status": "excluded", "reason": "component_" + bound[cid]["reason"]}
    pairs = [[cid, bound[cid]["binding"]] for cid in components]
    return {"status": "bound", "kind": "blend", "components": bound,
            "binding": content_hash(pairs)}


def bind_models(plan, read, roster, admit):
    """Bind every v14 model in the roster; production is deferred to S2b."""
    out = {}
    for model in roster["models"]:
        if model["kind"] == "production":
            out[model["model_id"]] = {"status": "deferred", "reason": "production_bound_by_s2b"}
        elif model["kind"] == "blend":
            out[model["model_id"]] = _bind_blend(plan, read, model["candidate_id"], admit)
        else:
            out[model["model_id"]] = bind_candidate(plan, read, model["candidate_id"], admit)
    statuses = [result["status"] for result in out.values()]
    print(f"[pnl_inputs] models={len(out)} bound={statuses.count('bound')} "
          f"excluded={statuses.count('excluded')} deferred={statuses.count('deferred')}",
          file=sys.stderr)
    return out
