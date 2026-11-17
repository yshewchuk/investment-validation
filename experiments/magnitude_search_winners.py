"""Per-rule winner adapters for the three-rule SIZE freeze."""
from experiments.magnitude_search_selection import selection_tie_band

__all__ = ["rule_winners"]

_RULE_PATHS = {
    "overall": ("blend_score",),
    "rmse": ("normalized", "rmse"),
    "mae": ("normalized", "mae"),
}


def _score(row, path):
    value = row
    for key in path:
        value = value[key]
    return value


def _adapted_rows(rows, path):
    return [{**row, "blend_score": _score(row, path)} for row in rows]


def _adapted_yearly(rows, path, best_id, best_missing):
    adapted = []
    for row in rows:
        yearly = {year: {**value, "blend_score": _score(value, path)}
                  for year, value in row["yearly"].items()}
        adapted.append({**row, "yearly": yearly})
    if best_missing and not any(row["candidate_id"] == best_id for row in adapted):
        adapted.insert(0, {"candidate_id": best_id, "yearly": {},
                           "diagnostic_unavailable": True})
    return adapted


def rule_winners(eligible, yearly_scores, *, floor, tie_rule, rules):
    """Select one winner per rule without mutating the supplied score rows."""
    winners = {}
    for rule in rules:
        path = _RULE_PATHS[rule]
        candidates = _adapted_rows(eligible, path)
        best = min(candidates, key=lambda row: (row["blend_score"], row["candidate_id"]))
        best_id = best["candidate_id"]
        original = next(row for row in eligible if row["candidate_id"] == best_id)
        best_missing = any(value is None for value in original["yearly"].values())
        selected_id, tie_band = selection_tie_band(
            candidates, _adapted_yearly(yearly_scores, path, best_id, best_missing),
            floor=floor, tie_rule=tie_rule)
        winners[rule] = {"candidate_id": selected_id, "tie_band": tie_band}
    return {"winners": winners,
            "distinct_winner_ids": sorted({winner["candidate_id"]
                                           for winner in winners.values()})}
