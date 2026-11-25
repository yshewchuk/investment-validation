"""Pure v14 bindings for EXP-190 (slice Q1). The Q2 reader loads sealed documents by content hash and passes them in.
No production caller yet. Never calls read_search_plan, prepare_plan or verify_installation; reads no files."""
import itertools
import math
import sys

from engine.v2.ops.errors import fail

_HGB_PREFIX = "magnitude_ridge_hgb_recipe"
_SUBSET = "HMOSXLP"
_MAE_RULE = "normalized.mae"
_MALFORMED = (KeyError, TypeError, AttributeError, ValueError)
_BAND_KEYS = ("pinball_p50", "pinball_mean", "coverage_p90", "coverage_p10_p90")


def _refuse(reason, message):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", f"{reason}: {message}", details={"reason": reason})


def _key(value):
    """Equal by value; a bool matches only a bool."""
    return type(value) is bool, value


def _subset(candidate):
    return candidate["feature_recipe"]["subset"]


def _grid_body(spec, plan):
    fixed = spec["grid"]["fixed"]
    leaves = spec["grid"]["axes"]["max_leaf_nodes"]
    leaf_sizes = spec["grid"]["axes"]["min_samples_leaf"]
    cells = list(itertools.product(leaves, leaf_sizes))
    grid = {(_key(a), _key(b)) for a, b in cells}
    hgb = [c for c in plan["candidates"] if str(c["recipe"].get("schema_version", "")).startswith(_HGB_PREFIX)
           and c["recipe"].get("model") == "history_hgb"]
    covered = set()
    for candidate in hgb:
        params = candidate["recipe"]["estimator"]["params"]
        for key, value in fixed.items():
            if key not in params or _key(value) != _key(params[key]):
                raise _refuse("v14_grid_mismatch", f"{candidate['candidate_id']}: {key} differs from the spec")
        pair = (_key(params["max_leaf_nodes"]), _key(params["min_samples_leaf"]))
        if pair not in grid:
            raise _refuse("v14_grid_mismatch", f"{candidate['candidate_id']}: outside the grid axes")
        if _subset(candidate) == _SUBSET:
            covered.add(pair)
    for a, b in cells:
        if (_key(a), _key(b)) not in covered:
            raise _refuse("v14_grid_mismatch", f"no HMOSXLP HGB candidate for pair {[a, b]}")
    return {"plan_id": plan["plan_id"], "hgb_candidates": len(hgb),
            "configurations": sorted([a, b] for a, b in cells)}


def bind_grid(spec, plan):
    """Return the v14 grid binding, or refuse ``v14_grid_mismatch`` (malformed documents included).

    HGB candidates must carry the fixed parameters and an in-grid pair; HMOSXLP ones cover the axis product.
    """
    try:
        return _grid_body(spec, plan)
    except _MALFORMED as exc:
        raise _refuse("v14_grid_mismatch", f"malformed document ({type(exc).__name__})") from None


def _band_entry(row):
    pooled = row["pooled"]
    values = {key: None for key in _BAND_KEYS} if pooled is None else {key: pooled[key] for key in _BAND_KEYS}
    return {"inner_year": row["inner_year"], "status": row["status"], **values}


def _band_status(band, distribution_status):
    statuses = [entry["status"] for entry in band]
    if band and all(status == "ok" for status in statuses):
        return "ok"
    if distribution_status is not None and distribution_status != "ok":
        return distribution_status
    return next((status for status in statuses if status != "ok"), "no_rows")


def _report_band(plan_id, m3, report):
    """Return ``(band, band_status)`` for M3's 2025 inner rows; a malformed report refuses v14_report_mismatch."""
    if report is None:
        return [], "report_not_given"
    try:
        if report["plan_id"] != plan_id:
            raise _refuse("v14_report_mismatch", "report plan_id differs from the plan")
        distribution = report["distribution"]
        band = sorted((_band_entry(r) for r in distribution.get("inner", [])
                       if str(r["outer_year"]) == "2025" and r["candidate_id"] == m3),
                      key=lambda entry: entry["inner_year"])
        return band, _band_status(band, distribution.get("status"))
    except _MALFORMED:
        raise _refuse("v14_report_mismatch", "report is malformed") from None


def _fallback(rows, candidates):
    """Lowest normalized MAE among HMOSXLP candidates with metrics; ties go to the lowest candidate id."""
    eligible = []
    for row in rows:
        cid = row["candidate_id"]
        if row["metrics"] is None or cid not in candidates or _subset(candidates[cid]) != _SUBSET:
            continue
        mae = row["normalized"]["mae"]
        if isinstance(mae, bool) or not isinstance(mae, (int, float)) or not math.isfinite(mae):
            raise _refuse("b3_unbound", f"{cid}: normalized.mae is not a finite number")
        eligible.append((mae, cid))
    if not eligible:
        raise _refuse("b3_unbound", "no HMOSXLP candidate with metrics for the fallback")
    return min(eligible)[1]


def _b3_body(plan, selection, report):
    plan_id = plan["plan_id"]
    if selection.get("plan_id") != plan_id or plan["selection"]["rule_metrics"]["mae"] != _MAE_RULE:
        raise _refuse("v14_selection_mismatch", "selection record or plan rule does not bind to this plan")
    years = [item for item in selection["selections"] if str(item["outer_year"]) == "2025"]
    if len(years) != 1:
        raise _refuse("m3_unbound", f"{len(years)} selection items for outer year 2025, expected 1")
    rows, m3 = years[0]["scores"], years[0]["winners"]["mae"]["candidate_id"]
    m3_rows = [row for row in rows if row["candidate_id"] == m3 and row["metrics"] is not None]
    if len(m3_rows) != 1:
        raise _refuse("m3_unbound", f"{m3} needs exactly one score row with metrics")
    candidates = {c["candidate_id"]: c for c in plan["candidates"]}
    blends = {b["blend_id"]: b for b in plan.get("blends", [])}
    if m3 in candidates:
        kind, uses = "candidate", _subset(candidates[m3]) == _SUBSET
    elif m3 in blends:
        parts = blends[m3]["component_candidate_ids"]
        kind, uses = "blend", bool(parts) and all(p in candidates and _subset(candidates[p]) == _SUBSET for p in parts)
    else:
        raise _refuse("m3_unbound", f"{m3} is neither a plan candidate nor a plan blend")
    if uses:
        b3, rule = m3, "v14_mae_winner"
    else:
        b3, rule, kind = _fallback(rows, candidates), "lowest_inner_mae_hmosxlp", "candidate"
    pinball = m3_rows[0]["metrics"]["mae"] / 2
    band, band_status = _report_band(plan_id, m3, report)
    return {"plan_id": plan_id, "m3_candidate_id": m3, "b3_candidate_id": b3, "rule": rule, "kind": kind,
            "m3_pinball_p50_from_mae": pinball, "m3_band_status": band_status, "m3_band": band}


def bind_b3(plan, selection, report=None):
    """Bind B3 to the v14 MAE winner (or its fallback) and report M3's inner band; stderr log only.

    Which years of B3 predictions exist is decided in Q4b.
    """
    try:
        result = _b3_body(plan, selection, report)
    except _MALFORMED as exc:
        raise _refuse("m3_unbound", f"malformed v14 document ({type(exc).__name__})") from None
    band = ",".join(f"{e['inner_year']}:{e['pinball_mean']}/{e['coverage_p10_p90']}" for e in result["m3_band"])
    print(f"[quantile_bind] m3={result['m3_candidate_id']} b3={result['b3_candidate_id']} "
          f"rule={result['rule']} p50_pinball={result['m3_pinball_p50_from_mae']} "
          f"band_status={result['m3_band_status']} band={band or '-'}", file=sys.stderr)
    return result
