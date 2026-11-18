"""Pure blend inner scoring: average aligned component forecasts per event, then score them."""
import math

from engine.v2.ops.errors import fail
from experiments.magnitude_search_selection import ZeroComparatorError


def _lead1(rows):
    """event_id -> (horizon-1 row); rows with horizon_sessions != 1 dropped."""
    return {row["event_id"]: row for row in rows if row.get("horizon_sessions") == 1}


def _conflict():
    return fail("EXPERIMENT_IDENTITY_CONFLICT", "blend component rows do not align on events and targets")


def average_component_rows(component_rows):
    """component_rows: list (sorted component-id order) of row lists for one year.
    Keep lead-1 rows (horizon_sessions == 1) per component keyed by event_id. All components must
    have identical event-id sets, no duplicate event ids, and equal absolute_move_pct per event, else
    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "blend component rows do not align on events and targets").
    Return rows sorted by event_id: {"event_id", "horizon_sessions": 1, "absolute_move_pct": <target>,
    "predicted_absolute_move_pct": math.fsum(preds in component order)/len(components)}."""
    if not component_rows:
        raise _conflict()
    maps = []
    for rows in component_rows:
        by_id = _lead1(rows)
        if len(by_id) != sum(row.get("horizon_sessions") == 1 for row in rows):
            raise _conflict()
        maps.append(by_id)
    ids = set(maps[0])
    if any(set(by_id) != ids for by_id in maps[1:]):
        raise _conflict()
    averaged = []
    for event_id in sorted(ids):
        target = maps[0][event_id]["absolute_move_pct"]
        if any(by_id[event_id]["absolute_move_pct"] != target for by_id in maps[1:]):
            raise _conflict()
        preds = [by_id[event_id]["predicted_absolute_move_pct"] for by_id in maps]
        averaged.append({"event_id": event_id, "horizon_sessions": 1, "absolute_move_pct": target,
                         "predicted_absolute_move_pct": math.fsum(preds) / len(maps)})
    return averaged


def score_blends(blends, component_years, inner_years, history_rows_by_year, score_rows, failed_components):
    """blends: plan["blends"] list of records (keys name, blend_id, component_candidate_ids).
    component_years: {candidate_id: {year: lead-1 or full rows}} for succeeded components.
    failed_components: set of candidate ids that failed any inner year.
    inner_years: ordered list of inner years (same order the candidate loop pools in).
    history_rows_by_year: {year: history rows}. score_rows(candidate_rows, baseline_rows) -> score dict.
    Returns a list of score rows, one per blend, in blend order:
      failed blend (any component in failed_components): {"candidate_id": blend_id, "metrics": None, "kind": "blend", "blend_name": name}
      else {"candidate_id": blend_id, **pooled, "yearly": yearly, "kind": "blend", "blend_name": name}
    where per year rows = average_component_rows([component_years[c][year] for c in sorted ids]),
    pooled = score_rows(concat of yearly averaged rows in inner_years order, concat of history rows in same order),
    yearly[year] = score_rows(avg rows, history_rows_by_year[year]) with ZeroComparatorError -> None
    (mirrors the candidate path in magnitude_search_trials._freeze). Do not catch other errors."""
    scores = []
    for blend in blends:
        name, blend_id = blend["name"], blend["blend_id"]
        ids = sorted(blend["component_candidate_ids"])
        if any(component in failed_components for component in ids):
            scores.append({"candidate_id": blend_id, "metrics": None, "kind": "blend", "blend_name": name})
            continue
        averaged = {year: average_component_rows([component_years[c][year] for c in ids])
                    for year in inner_years}
        pooled = score_rows([row for year in inner_years for row in averaged[year]],
                            [row for year in inner_years for row in history_rows_by_year[year]])
        yearly = {}
        for year in inner_years:
            try:
                yearly[year] = score_rows(averaged[year], history_rows_by_year[year])
            except ZeroComparatorError:
                yearly[year] = None
        scores.append({"candidate_id": blend_id, **pooled, "yearly": yearly, "kind": "blend", "blend_name": name})
    return scores


def best_non_blend_id(eligible):
    """candidate_id of the lowest (blend_score, candidate_id) row whose kind is not "blend"."""
    non_blend = [score for score in eligible if score.get("kind") != "blend"]
    if not non_blend:
        raise fail("EXPERIMENT_VARIANT_FAILED", "all candidates failed for an outer fold")
    return min(non_blend, key=lambda score: (score["blend_score"], score["candidate_id"]))["candidate_id"]
