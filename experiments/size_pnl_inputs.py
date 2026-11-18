"""EXP-188 S2a: bind stored v14 trial predictions to the SIZE selection population.

Only ``event_id`` and the lead-horizon predicted move are read from each stored
prediction. Stored outcome values are never read. Run-level identity failures
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
ROLES = ("M0", "M1", "M2", "M3", "M4", "M5") + PAIR_ROLES
MAX_MODELS = 12


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
    if 1 not in plan["horizons"]:
        raise _conflict("lead_unavailable", "the plan has no lead-horizon (1) predictions")
    return dict(actual)


def population_admit(repository, snapshot, as_of_month, load=None):
    """Return ``admit(event_ids)`` backed by the canonical population reader."""
    if load is None:
        from engine.v2.research.experiment_population import load_population
        load = load_population

    def admit(event_ids):
        ids = sorted(event_ids)
        frame = load(repository, snapshot, as_of_month=as_of_month,
                     purpose="selection", event_ids=ids)
        return {"admitted": sorted(frame["event_id"].tolist()), "dropped": []}

    return admit


def bind_array(record, sealed_ids, admit, *, horizon=1):
    """Bind one stored prediction object at ``horizon`` to its sealed event ids."""
    rows = [row for row in record["predictions"] if row["horizon_sessions"] == horizon]
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


def bind_candidate(plan, read, candidate_id, admit, *, horizon=1):
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
                years[year] = bind_array(
                    record, inner["evaluation_event_ids"], admit, horizon=horizon)
            except OpsError as exc:
                if exc.problem.details.get("reason") != "partial_event_ids":
                    raise
                return {"status": "excluded", "reason": "partial"}
    pairs = [[year, years[year]["binding"]] for year in sorted(years)]
    return {"status": "bound", "years": years, "binding": content_hash(pairs)}


def _transform(candidate):
    return "log1p" if candidate["recipe"]["schema_version"] in LOG1P_SCHEMAS else "none"


def _pick(cands, candidate_id, missing_reason):
    if candidate_id is None:
        return None, missing_reason
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


def _pair(cands, extra, pair_candidates, role):
    candidate_id = pair_candidates.get(role)
    if candidate_id is None:
        return None, "unmapped"
    if candidate_id not in extra:
        return None, "not_in_extra_refits"
    if candidate_id not in cands:
        return None, "not_in_plan"
    return ("v14", candidate_id, _transform(cands[candidate_id])), None


def resolve_roster(plan, selection, *, outer_year, pair_candidates, blend_ids):
    """Pre-registered roster for one outer year. No P&L is read."""
    cands = {c["candidate_id"]: c for c in plan["candidates"]}
    blends = {b["blend_id"] for b in plan.get("blends", [])}
    extra = set(plan["selection"]["extra_refits"])
    items = [s for s in selection["selections"] if str(s["outer_year"]) == str(outer_year)]
    item = items[0] if items else {}
    winners = item.get("winners", {})
    found = {
        "M0": (("production", "size_v1_4", "none"), None),
        "M1": _pick(cands, item.get("selected_candidate_id"), "winner_unresolved"),
        "M2": _pick(cands, winners.get("rmse", {}).get("candidate_id"), "winner_unresolved"),
        "M3": _pick(cands, winners.get("mae", {}).get("candidate_id"), "winner_unresolved"),
        "M4": _blend(blends, blend_ids, "M4"),
        "M5": _blend(blends, blend_ids, "M5"),
    }
    for role in PAIR_ROLES:
        found[role] = _pair(cands, extra, pair_candidates, role)

    models, not_testable = {}, []
    for role in ROLES:
        spec, reason = found[role]
        if spec is None:
            not_testable.append({"role": role, "reason": reason})
            continue
        source, candidate_id, transform = spec
        model_id = content_hash({"source": source, "candidate_id": candidate_id,
                                 "target_transform": transform})
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
    pairs = {pid: "not_testable" if dead & set(members) else "testable"
             for pid, members in PAIRS.items()}
    return {"models": list(models.values()), "not_testable": not_testable, "pairs": pairs}


def _bind_blend(plan, read, blend_id, admit, horizon):
    components = blend_components(plan, blend_id)
    if components is None:
        return {"status": "excluded", "reason": "blend_unknown"}
    bound = {cid: bind_candidate(plan, read, cid, admit, horizon=horizon) for cid in components}
    for cid in components:
        if bound[cid]["status"] != "bound":
            return {"status": "excluded", "reason": "component_" + bound[cid]["reason"]}
    pairs = [[cid, bound[cid]["binding"]] for cid in components]
    return {"status": "bound", "kind": "blend", "components": bound,
            "binding": content_hash(pairs)}


def bind_models(plan, read, roster, admit, *, horizon=1):
    """Bind every v14 model in the roster; production is deferred to S2b."""
    out = {}
    for model in roster["models"]:
        if model["kind"] == "production":
            out[model["model_id"]] = {"status": "deferred", "reason": "production_bound_by_s2b"}
        elif model["kind"] == "blend":
            out[model["model_id"]] = _bind_blend(plan, read, model["candidate_id"], admit, horizon)
        else:
            out[model["model_id"]] = bind_candidate(
                plan, read, model["candidate_id"], admit, horizon=horizon)
    statuses = [result["status"] for result in out.values()]
    print(f"[pnl_inputs] models={len(out)} bound={statuses.count('bound')} "
          f"excluded={statuses.count('excluded')} deferred={statuses.count('deferred')}",
          file=sys.stderr)
    return out
