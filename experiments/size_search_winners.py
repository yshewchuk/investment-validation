"""Report-only rows for every distinct winner under ``selection.winner_rules``.

Builds report rows from already-read records; logs one stderr line. The primary (``overall``) winner stays in ``paired``
elsewhere; each row here is one candidate that any rule picked in some outer year.
"""
import sys

PRIMARY_RULE = "overall"
LOG1P_SCHEMAS = frozenset({
    "magnitude_elasticnet_recipe.v2",
    "magnitude_elasticnet_spline_recipe.v2",
    "magnitude_mlp_recipe.v2",
})


def _score(row):
    metrics = row.get("metrics") or {}
    return {"rmse": metrics.get("rmse"), "mae": metrics.get("mae"),
            "blend": row.get("blend_score"), "normalized": row.get("normalized")}


def _share(lowest, forecast_rows, targets):
    scored = [r for r in forecast_rows if r["horizon_sessions"] == lowest and r["event_id"] in targets]
    if not scored:
        return None
    above = sum(1 for r in scored if targets[r["event_id"]] > r["predicted_absolute_move_pct"])
    return above / len(scored)


def winner_rows(plan, selection, refits, forecasts, events, comparisons, targets, *, paired, paired_kwargs):
    if not plan.get("selection", {}).get("winner_rules"):
        return None
    candidates = {c["candidate_id"]: c for c in plan["candidates"]}
    fold_events = {g["outer"]["year"]: set(g["outer"]["evaluation_event_ids"]) for g in plan["folds"]}
    refit_index = {(r["outer_year"], r["candidate_id"]): r for r in refits}
    lowest = min(plan["horizons"])

    # candidate_id -> {"rules": set, "years": {outer_year: {"rules", "scores", "tie_band"}}}
    won = {}
    for item in selection["selections"]:
        year = item["outer_year"]
        score_by = {row["candidate_id"]: row for row in item["scores"]}
        for rule, win in item["winners"].items():
            cid = win["candidate_id"]
            entry = won.setdefault(cid, {"rules": set(), "years": {}})
            entry["rules"].add(rule)
            per_year = entry["years"].setdefault(year, {"rules": set(), "scores": {}, "tie_band": {}})
            per_year["rules"].add(rule)
            per_year["scores"][rule] = _score(score_by.get(cid, {}))
            per_year["tie_band"][rule] = win["tie_band"]

    rows = []
    for cid, entry in won.items():
        years = sorted(entry["years"])
        error_code = None
        for year in years:
            refit = refit_index.get((year, cid))
            if refit is None:
                error_code = "missing_refit"
            elif refit["status"] == "failed":
                error_code = (refit.get("result") or {}).get("code") or "failed"
            elif refit["status"] != "completed" or (year, cid) not in forecasts:
                error_code = "missing_refit"
            if error_code is not None:
                break
        completed = error_code is None

        paired_result, share = None, None
        if completed:
            ids = set().union(*(fold_events[y] for y in years))
            fc = [row for y in years for row in forecasts[(y, cid)]]
            fc_ids = [row for row in fc if row["event_id"] in ids]
            label_reasons = paired_kwargs.get("label_reasons")
            implied = paired_kwargs.get("implied_moves")
            paired_result = paired(
                [e for e in events if e["event_id"] in ids],
                fc_ids,
                [row for row in comparisons if row["event_id"] in ids],
                {k: v for k, v in targets.items() if k in ids},
                horizons=paired_kwargs["horizons"],
                implied_moves=None if implied is None else {k: v for k, v in implied.items() if k[0] in ids},
                implied_scales=paired_kwargs["implied_scales"],
                label_reasons=None if label_reasons is None else {k: v for k, v in label_reasons.items() if k in ids})
            share = _share(lowest, fc, targets)

        rules = sorted(entry["rules"])
        inner = [{"outer_year": year, "rules": sorted(entry["years"][year]["rules"]),
                  "scores": entry["years"][year]["scores"], "tie_band": entry["years"][year]["tie_band"]}
                 for year in years]
        candidate = candidates.get(cid) or {}
        recipe = candidate.get("recipe")
        rows.append({
            "candidate_id": cid, "rules": rules, "primary": PRIMARY_RULE in rules,
            "kind": "candidate", "components": [],
            "target_transform": "log1p" if (recipe or {}).get("schema_version") in LOG1P_SCHEMAS else "none",
            "outer_years": years, "inner": inner,
            "status": "completed" if completed else "failed",
            "error_code": None if completed else error_code,
            "paired": paired_result, "actual_above_forecast_share": share})

    rows.sort(key=lambda row: (not row["primary"], row["candidate_id"]))
    failed = sum(1 for row in rows if row["status"] == "failed")
    print(f"[winners] rows={len(rows)} failed={failed}", file=sys.stderr)
    return rows
