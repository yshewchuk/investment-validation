"""EXP-188 S9b: bind the 2025 refit forecasts (outer-forecasts slots) to the sealed outer cohort (design section 12).

Reads only the outer-start, outer-execution and outer-forecasts slots of each refit unit; never a 2025 target slot.
A missing or failed refit excludes its model; an identity or hash mismatch refuses the whole run.
"""
import math
import sys
import time

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import size_pnl_inputs as inputs
from experiments.magnitude_search_refits import blend_components, per_candidate, refit_key

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_FORECAST_KEYS = {"event_id", "horizon_sessions", "predicted_absolute_move_pct"}


def _conflict(reason, message):
    return fail(_CONFLICT, message, details={"reason": reason})


def _check_start(plan, start, cid, year, key):
    """A start record must name this plan, outer year, candidate and refit key; a malformed record is a mismatch too."""
    try:
        identity_ok = (start["plan_id"] == plan["plan_id"] and start["outer_year"] == year
                       and start["candidate"]["candidate_id"] == cid and start["outer_id"] == key)
    except (KeyError, TypeError) as exc:
        raise _conflict("outer_identity_mismatch", "the outer start record is malformed") from exc
    if not identity_ok:
        raise _conflict("outer_identity_mismatch", "the outer refit does not match the sealed plan")


def _valid_forecast(row):
    """A forecast row names a str event and a finite predicted move, and carries no target field."""
    if not (isinstance(row, dict) and _FORECAST_KEYS <= row.keys() and "absolute_move_pct" not in row):
        return False
    pred = row["predicted_absolute_move_pct"]
    return (isinstance(row["event_id"], str) and isinstance(pred, (int, float)) and not isinstance(pred, bool)
            and math.isfinite(pred))


def _bind_unit(plan, read, cid, sealed_ids, admit, year):
    """One candidate's 2025 refit: an excluded or bound unit, with the identity checks of design section 12.
    Every present record is validated before any exclusion."""
    key = refit_key(plan, year, cid)
    start, execution, forecasts = read("outer-start:" + key), read("outer-execution:" + key), read("outer-forecasts:" + key)
    if start is not None and not isinstance(start, dict):
        raise _conflict("outer_identity_mismatch", "the outer start record is malformed")
    if execution is not None and not isinstance(execution, dict):
        raise _conflict("outer_identity_mismatch", "the outer execution record is malformed")
    if forecasts is not None and not isinstance(forecasts, list):
        raise _conflict("outer_identity_mismatch", "the outer forecasts record is malformed")
    if isinstance(execution, dict) and not ({"status", "forecasts_id"} <= execution.keys()
                                            and execution["status"] in ("succeeded", "failed")):
        raise _conflict("outer_identity_mismatch", "the outer execution record is malformed")
    if isinstance(execution, dict) and execution["status"] == "succeeded" \
            and not (isinstance(execution["forecasts_id"], str) and execution["forecasts_id"].startswith("sha256:")):
        raise _conflict("outer_identity_mismatch", "a succeeded outer refit has no forecasts hash")  # before any exclusion
    if isinstance(forecasts, list) and not all(_valid_forecast(row) for row in forecasts):
        raise _conflict("outer_identity_mismatch", "an outer forecast row is malformed")
    if isinstance(start, dict): _check_start(plan, start, cid, year, key)  # before the exclusions: a failed run must still name this plan
    # The producer writes start, then execution, then forecasts: a later slot without an earlier one is corruption, not a gap.
    if start is None and (execution is not None or forecasts is not None):
        raise _conflict("outer_identity_mismatch", "the outer start record is missing under a later outer slot")
    if execution is None and forecasts is not None:
        raise _conflict("outer_identity_mismatch", "the outer forecasts are present without an execution record")
    if isinstance(execution, dict) and execution["status"] == "succeeded" and isinstance(forecasts, list) \
            and execution["forecasts_id"] != content_hash(forecasts):
        raise _conflict("outer_forecasts_hash_mismatch", "the outer forecasts differ from the recorded hash")
    if isinstance(execution, dict) and execution["status"] == "failed":
        # The producer records a failed run with forecasts_id None and writes no forecasts slot.
        if execution["forecasts_id"] is not None or forecasts is not None:
            raise _conflict("outer_identity_mismatch", "a failed outer refit carries forecasts")
        return {"status": "excluded", "reason": "outer_failed"}
    if start is None or execution is None or forecasts is None:
        return {"status": "excluded", "reason": "outer_missing"}
    try:
        array = inputs.bind_array({"predictions": forecasts}, sealed_ids, admit)
    except OpsError as exc:
        if exc.problem.details.get("reason") != "partial_event_ids":
            raise
        return {"status": "excluded", "reason": "partial"}
    return {"status": "bound", "array": array}


def _candidate(entry, unit, outer_year):
    """A candidate entry with the outer year bound, or excluded with the unit's reason."""
    if unit["status"] == "excluded":
        return {"status": "excluded", "reason": unit["reason"]}
    years, key = dict(entry["years"]), str(outer_year)
    if key in years:
        raise fail("INVALID_EXPERIMENT_SPEC", "the outer year is already bound", details={"reason": "repeated_inner_year"})
    years[key] = unit["array"]
    return {**entry, "years": years, "binding": content_hash([[y, years[y]["binding"]] for y in sorted(years)])}


def _blend(entry, plan, blend_id, unit, outer_year):
    """A blend entry with every component's outer year bound; excluded if any component is."""
    cids = blend_components(plan, blend_id)
    if cids is None or sorted(entry["components"]) != cids:
        raise _conflict("outer_identity_mismatch", "the bound blend components differ from the plan")
    units = {cid: unit(cid) for cid in cids}
    for cid in cids:
        if units[cid]["status"] == "excluded":
            return {"status": "excluded", "reason": "component_" + units[cid]["reason"]}
    components = {cid: _candidate(entry["components"][cid], units[cid], outer_year) for cid in cids}
    return {**entry, "components": components,
            "binding": content_hash([[cid, components[cid]["binding"]] for cid in cids])}


def bind_outer(plan, read, bound, roster, admit, *, outer_year):
    """Return a new bound-models dict with the sealed outer year added to every bound v14 model (input not mutated)."""
    started = time.monotonic()
    models = {m["model_id"]: m for m in roster["models"]}
    if not any(entry.get("status") == "bound" and models[model_id]["kind"] != "production"
               for model_id, entry in bound.items()):
        return dict(bound)
    if not per_candidate(plan):
        raise _conflict("outer_plan_unsupported", "the outer binder needs a per-candidate plan")
    fold = next((g for g in plan["folds"] if g["outer"]["year"] == outer_year), None)
    if fold is None:
        raise _conflict("outer_identity_mismatch", "the plan seals no fold for the outer year")
    sealed, cache = fold["outer"]["evaluation_event_ids"], {}

    def unit(cid):
        if cid not in cache:
            cache[cid] = _bind_unit(plan, read, cid, sealed, admit, outer_year)
        return cache[cid]

    out = {}
    for model_id, entry in bound.items():
        if entry.get("status") != "bound" or models[model_id]["kind"] == "production":
            out[model_id] = entry
        elif models[model_id]["kind"] == "blend":
            out[model_id] = _blend(entry, plan, models[model_id]["candidate_id"], unit, outer_year)
        else:
            out[model_id] = _candidate(entry, unit(models[model_id]["candidate_id"]), outer_year)
    for model_id, entry in out.items():
        if entry.get("status") == "excluded":
            print(f"[pnl_outer] excluded {model_id} {entry['reason']}", file=sys.stderr)
    statuses = [unit_result["status"] for unit_result in cache.values()]
    print(f"[pnl_outer] units={len(cache)} bound={statuses.count('bound')} excluded={statuses.count('excluded')} "
          f"elapsed={time.monotonic() - started:.1f}s", file=sys.stderr)
    return out
