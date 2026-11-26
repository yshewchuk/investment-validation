"""EXP-190 Q4b: baseline B3 (design guides/size_quantile_move_design.md sections 5 and 7).

B3 is B0's construction on the sealed lead-1 predictions of the model bind_b3 chose. A year without a bound
prediction is not_testable, not a refusal. Pure over caller-injected read and admit. No production caller yet.
"""
import numpy as np
import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from engine.v2.research.progress import decision, stage
from experiments.magnitude_search_refits import blend_components
from experiments.size_pnl_inputs import bind_candidate
from experiments.size_pnl_outer_inputs import bind_outer
from experiments.size_pnl_readers import _blend_series, _candidate_series
from experiments.size_quantile_baselines import pool_quantiles

INNER_YEARS = (2023, 2024)
OUTER_YEAR = 2025
_MALFORMED = (KeyError, TypeError, AttributeError, ValueError)
_EXTRA = ("bucket", "fallback", "n_rows", "reason")


def _refuse(reason, message):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", f"{reason}: {message}", details={"reason": reason})


def _inner(plan, read, cids, kind, year, admit):
    """(status, entry) for one inner year; a blend names its first failing component in sorted order."""
    one = {**plan, "folds": [{**g, "inner": [i for i in g["inner"] if str(i["year"]) == str(year)]}
                             for g in plan["folds"]]}
    parts = {cid: bind_candidate(one, read, cid, admit) for cid in cids}
    for cid in cids:
        entry = parts[cid]
        status = entry.get("reason") or ("bound" if str(year) in entry.get("years", {}) else "missing")
        if status != "bound":
            return (f"component_{status}" if kind == "blend" else status), None
    if kind == "candidate":
        return "bound", parts[cids[0]]
    entry = {"status": "bound", "kind": "blend", "components": parts,
             "binding": content_hash([[cid, parts[cid]["binding"]] for cid in cids])}
    return "bound", entry


def _outer(plan, read, cid, kind, cids, admit):
    """(status, entry) for the sealed outer year; missing when the plan seals no fold for it."""
    if not any(str(g["outer"]["year"]) == str(OUTER_YEAR) for g in plan["folds"]):
        return "missing", None
    if kind == "candidate":
        bound = {cid: {"status": "bound", "years": {}}}
    else:
        bound = {cid: {"status": "bound", "kind": "blend",
                       "components": {c: {"status": "bound", "years": {}} for c in cids}}}
    roster = {"models": [{"model_id": cid, "kind": kind, "candidate_id": cid}]}
    entry = bind_outer(plan, read, bound, roster, admit, outer_year=OUTER_YEAR)[cid]
    return ("bound", entry) if entry["status"] == "bound" else (entry["reason"], None)


def _bind(plan, read, binding, admit):
    if binding["plan_id"] != plan["plan_id"]:
        raise _refuse("b3_plan_mismatch", "the B3 binding names another plan")
    cid, kind = binding["b3_candidate_id"], binding["kind"]
    cids = [cid] if kind == "candidate" else blend_components(plan, cid) if kind == "blend" else None
    if not cids:
        raise _refuse("b3_series_unbound", f"unknown or empty B3 model {kind!r}")
    results = [(str(y), *_inner(plan, read, cids, kind, y, admit)) for y in INNER_YEARS]
    results.append((str(OUTER_YEAR), *_outer(plan, read, cid, kind, cids, admit)))
    parts = [_blend_series(e) if e.get("kind") == "blend" else _candidate_series(e)
             for _, status, e in results if status == "bound"]
    series = pd.concat(parts) if parts else pd.Series([], dtype=float, index=pd.Index([], name="event_id"))
    series.index = series.index.rename("event_id")
    if series.index.has_duplicates:
        raise _refuse("duplicate_event_id", "an event id repeats across years")
    years = {year: status for year, status, _ in results}
    rows = [[year, status, entry["binding"] if status == "bound" else None] for year, status, entry in results]
    digest = content_hash({"plan_id": plan["plan_id"], "b3_candidate_id": cid, "years": rows})
    decision("quantile B3", f"id={cid} years={years} rows={len(series)}")
    return {"b3_candidate_id": cid, "years": years, "series": series, "binding": digest}


def bind_series(plan, read, binding, admit):
    """Bind the sealed lead-1 predictions of B3's model for 2023-2025; malformed documents refuse."""
    with stage("quantile B3 binding"):
        try:
            return _bind(plan, read, binding, admit)
        except _MALFORMED as exc:
            raise _refuse("b3_series_unbound", f"malformed sealed document ({type(exc).__name__})") from None


def b3_quantiles(table, labels, events, bound):
    """B0's construction with donors whose pred_move is the bound B3 series; crush forecast and date stay B0's."""
    series = bound["series"]
    bound_years = [year for year, status in bound["years"].items() if status == "bound"]
    unbound = ~pd.to_datetime(events["event_date"]).dt.year.astype(str).isin(bound_years).to_numpy()
    donors = table.loc[table.index.isin(series.index)]
    donors = donors.assign(pred_move=series.reindex(donors.index).to_numpy())
    no_forecast = int((~series.index.isin(table.index)).sum())
    scored = events.assign(pred_move=series.reindex(events.index).to_numpy())
    out = pool_quantiles(donors, labels, scored, name="B3")
    quant = [col for col in out.columns if col not in _EXTRA]
    out.loc[unbound, quant + ["bucket"]] = np.nan
    out.loc[unbound, "fallback"] = False
    out.loc[unbound, "n_rows"] = 0
    out.loc[unbound, "reason"] = "not_testable"
    decision("quantile B3", f"events={len(events)} not_testable={int(unbound.sum())} no_crush_forecast={no_forecast}")
    return out
