import copy

from experiments.magnitude_search_selection import CURRENT_TIE_RULE, selection_tie_band
from experiments.magnitude_search_winners import rule_winners


def _score(overall, rmse, mae):
    return {"blend_score": overall, "normalized": {"rmse": rmse, "mae": mae}}


def _row(candidate_id, overall, rmse, mae, yearly):
    return {"candidate_id": candidate_id, **_score(overall, rmse, mae), "yearly": yearly}


def _yearly_scores(eligible):
    return [{"candidate_id": row["candidate_id"], "yearly": row["yearly"]}
            for row in eligible
            if all(value is not None for value in row["yearly"].values())]


def test_each_rule_can_pick_its_own_candidate():
    eligible = [
        _row("a", 0.1, 0.9, 0.5, {"2023": _score(0.1, 0.9, 0.5)}),
        _row("b", 0.2, 0.1, 0.9, {"2023": _score(0.2, 0.1, 0.9)}),
        _row("c", 0.3, 0.5, 0.1, {"2023": _score(0.3, 0.5, 0.1)}),
    ]
    result = rule_winners(eligible, _yearly_scores(eligible), floor=0.005,
                          tie_rule=CURRENT_TIE_RULE,
                          rules=["overall", "rmse", "mae"])
    assert {rule: winner["candidate_id"]
            for rule, winner in result["winners"].items()} == {
                "overall": "a", "rmse": "b", "mae": "c"}
    assert result["distinct_winner_ids"] == ["a", "b", "c"]


def test_shared_winner_is_reported_once():
    eligible = [
        _row("a", 0.1, 0.2, 0.2, {"2023": _score(0.1, 0.2, 0.2)}),
        _row("b", 0.2, 0.9, 0.9, {"2023": _score(0.2, 0.9, 0.9)}),
    ]
    result = rule_winners(eligible, _yearly_scores(eligible), floor=0.005,
                          tie_rule=CURRENT_TIE_RULE, rules=["overall", "rmse"])
    assert set(result["winners"]) == {"overall", "rmse"}
    assert all(winner["candidate_id"] == "a"
               for winner in result["winners"].values())
    assert result["distinct_winner_ids"] == ["a"]


def test_exact_rule_tie_picks_lowest_candidate_id():
    eligible = [
        _row("b", 0.2, 0.1, 0.1, {"2023": _score(0.2, 0.1, 0.1)}),
        _row("a", 0.3, 0.1, 0.2, {"2023": _score(0.3, 0.1, 0.2)}),
    ]
    result = rule_winners(eligible, _yearly_scores(eligible), floor=0.005,
                          tie_rule=CURRENT_TIE_RULE, rules=["rmse"])
    assert result["winners"]["rmse"]["candidate_id"] == "a"


def test_overall_matches_unadapted_selection_and_inputs_are_not_mutated():
    eligible = [
        _row("z", 0.1, 0.4, 0.4, {"2022": _score(0.1, 0.4, 0.4),
                                  "2023": _score(0.1, 0.4, 0.4)}),
        _row("a", 0.2, 0.3, 0.3, {"2022": _score(0.2, 0.3, 0.3),
                                  "2023": _score(0.2, 0.3, 0.3)}),
    ]
    yearly = _yearly_scores(eligible)
    eligible_before, yearly_before = copy.deepcopy(eligible), copy.deepcopy(yearly)
    expected_id, expected_band = selection_tie_band(
        eligible, yearly, floor=0.005, tie_rule=CURRENT_TIE_RULE)
    result = rule_winners(eligible, yearly, floor=0.005,
                          tie_rule=CURRENT_TIE_RULE, rules=["overall"])
    assert result["winners"]["overall"] == {
        "candidate_id": expected_id, "tie_band": expected_band}
    assert eligible == eligible_before
    assert yearly == yearly_before


def test_rule_leader_with_missing_yearly_is_a_diagnostic():
    eligible = [
        _row("a", 0.1, 0.5, 0.5, {"2023": _score(0.1, 0.5, 0.5)}),
        _row("b", 0.2, 0.1, 0.9, {"2023": None}),
    ]
    result = rule_winners(eligible, _yearly_scores(eligible), floor=0.005,
                          tie_rule=CURRENT_TIE_RULE, rules=["rmse"])
    winner = result["winners"]["rmse"]
    assert winner["candidate_id"] == "b"
    assert winner["tie_band"]["standard_error"] is None
