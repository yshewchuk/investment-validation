"""Unit tests for report-only winner rows (fake paired callable, hand-built records)."""
import pytest

from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe
from experiments.size_paired_report import paired_report
from experiments.size_search_winners import winner_rows

_HASH = "sha256:" + "0" * 64


class FakePaired:
    def __init__(self):
        self.calls = []

    def __call__(self, ev, fc, cmp, tg, **kwargs):
        self.calls.append({"events": [e["event_id"] for e in ev], "forecasts": fc, "targets": tg, "kwargs": kwargs})
        return {"fake": True, "n_events": len(ev)}


def _score(cid, rmse, blend):
    return {"candidate_id": cid, "metrics": {"rmse": rmse, "mae": rmse / 2},
            "normalized": {"rmse": rmse}, "blend_score": blend}


def _plan(rules=("overall", "rmse", "mae")):
    plan = {"plan_id": "p", "horizons": [1, 3],
            "candidates": [
                {"candidate_id": "a", "recipe": elasticnet_recipe(
                    columns=["a"], feature_dictionary_hash=_HASH, alpha=0.1, l1_ratio=0.5,
                    target_transform="log1p")},
                {"candidate_id": "b", "recipe": elasticnet_recipe(
                    columns=["a"], feature_dictionary_hash=_HASH, alpha=0.1, l1_ratio=0.5)},
                {"candidate_id": "c", "recipe": mlp_recipe(
                    columns=["a"], feature_dictionary_hash=_HASH, profile="relu32_l2_0p01")}],
            "folds": [{"outer": {"year": "2024", "evaluation_event_ids": ["e1", "e2", "e3"]}},
                      {"outer": {"year": "2025", "evaluation_event_ids": ["e4"]}}]}
    if rules:
        plan["selection"] = {"winner_rules": list(rules)}
    return plan


SELECTION = {"selections": [
    {"outer_year": "2024",
     "scores": [_score("a", 2.0, 0.5), _score("b", 1.0, 0.4), _score("c", 3.0, 0.9)],
     "winners": {"overall": {"candidate_id": "b", "tie_band": 0.1},
                 "rmse": {"candidate_id": "b", "tie_band": 0.1},
                 "mae": {"candidate_id": "a", "tie_band": 0.2}},
     "selected_candidate_id": "b"},
    {"outer_year": "2025",
     "scores": [_score("a", 2.0, 0.5), _score("b", 1.5, 0.6), _score("c", 1.0, 0.3)],
     "winners": {rule: {"candidate_id": "c", "tie_band": 0.1} for rule in ("overall", "rmse", "mae")},
     "selected_candidate_id": "c"}]}

REFITS = [{"outer_year": "2024", "candidate_id": "b", "status": "completed", "result": {}},
          {"outer_year": "2024", "candidate_id": "a", "status": "completed", "result": {}},
          {"outer_year": "2025", "candidate_id": "c", "status": "completed", "result": {}}]

EVENTS = [{"event_id": e} for e in ("e1", "e2", "e3", "e4")]
TARGETS = {"e1": 2.0, "e2": 2.0, "e3": 1.0, "e4": 1.0}
COMPARISONS = [{"event_id": e, "horizon_sessions": h} for e in ("e1", "e2", "e3", "e4") for h in (1, 3)]
PAIRED_KWARGS = {"horizons": [1, 3],
                 "implied_moves": {("e1", 1): 0.5, ("e4", 1): 0.7},
                 "implied_scales": {"2024": {"scale": 1.0, "fit_n": 3}},
                 "label_reasons": {}}


def _fc(preds):
    """Horizon-1 predictions from ``preds``; horizon-3 rows are all 0.0 so they cannot change the share."""
    return ([{"event_id": e, "horizon_sessions": 1, "predicted_absolute_move_pct": p} for e, p in preds.items()]
            + [{"event_id": e, "horizon_sessions": 3, "predicted_absolute_move_pct": 0.0} for e in preds])


FORECASTS = {
    ("2024", "a"): _fc({"e1": 1.5, "e2": 1.5, "e3": 1.5}),
    ("2024", "b"): _fc({"e1": 1.0, "e2": 2.0, "e3": 3.0}),
    ("2025", "c"): _fc({"e4": 0.5}),
}


def _run(fake, *, plan=None, refits=REFITS, forecasts=FORECASTS):
    return winner_rows(plan or _plan(), SELECTION, refits, forecasts, EVENTS, COMPARISONS, TARGETS,
                       paired=fake, paired_kwargs=PAIRED_KWARGS)


def _by_id(rows):
    return {row["candidate_id"]: row for row in rows}


def test_no_winner_rules_returns_none():
    assert _run(FakePaired(), plan=_plan(rules=())) is None


def test_two_rules_same_candidate_give_one_row_with_both_rules():
    rows = _run(FakePaired())
    b_rows = [row for row in rows if row["candidate_id"] == "b"]
    assert len(b_rows) == 1
    assert b_rows[0]["rules"] == ["overall", "rmse"]
    assert b_rows[0]["primary"] is True


def test_rows_primary_first_then_by_id_and_inner_scores_match_candidate():
    rows = _run(FakePaired())
    assert [row["candidate_id"] for row in rows] == ["b", "c", "a"]
    assert [row["primary"] for row in rows] == [True, True, False]
    by_id = _by_id(rows)
    assert by_id["a"]["rules"] == ["mae"]
    assert by_id["a"]["inner"][0]["scores"]["mae"]["rmse"] == 2.0
    assert by_id["a"]["inner"][0]["scores"]["mae"]["blend"] == 0.5
    assert by_id["b"]["inner"][0]["scores"]["rmse"]["rmse"] == 1.0
    assert by_id["b"]["inner"][0]["scores"]["rmse"]["blend"] == 0.4
    assert by_id["a"]["target_transform"] == "log1p"
    assert by_id["b"]["target_transform"] == "none"
    assert by_id["c"]["target_transform"] == "none"


def test_missing_refit_marks_row_failed_with_error_code():
    refits = [r for r in REFITS if not (r["candidate_id"] == "b" and r["outer_year"] == "2024")]
    rows = _by_id(_run(FakePaired(), refits=refits))
    assert rows["b"]["status"] == "failed"
    assert rows["b"]["error_code"] == "missing_refit"
    assert rows["b"]["paired"] is None
    assert rows["b"]["actual_above_forecast_share"] is None
    assert rows["c"]["status"] == "completed"


def test_failed_refit_marks_its_row_and_leaves_others_unaffected():
    failed_c = {"outer_year": "2025", "candidate_id": "c", "status": "failed", "result": {"code": "FIT_FAILED"}}
    refits = [r for r in REFITS if r["candidate_id"] != "c"] + [failed_c]
    forecasts = {k: v for k, v in FORECASTS.items() if k != ("2025", "c")}
    rows = _by_id(_run(FakePaired(), refits=refits, forecasts=forecasts))
    assert rows["c"]["status"] == "failed"
    assert rows["c"]["error_code"] == "FIT_FAILED"
    assert rows["c"]["paired"] is None
    assert rows["c"]["actual_above_forecast_share"] is None
    assert rows["b"]["status"] == "completed"
    assert rows["b"]["error_code"] is None
    assert rows["b"]["paired"]["fake"] is True
    assert rows["b"]["actual_above_forecast_share"] == pytest.approx(1 / 3)


def test_share_uses_lowest_horizon_and_strict_greater():
    rows = _by_id(_run(FakePaired()))
    # b: predictions [1, 2, 3] against actuals [2, 2, 1] at horizon 1 -> only the first is above.
    assert rows["b"]["actual_above_forecast_share"] == pytest.approx(1 / 3)
    # a: predictions [1.5, 1.5, 1.5] against actuals [2, 2, 1] -> two are above.
    # Distinct shares across candidates rule out a constant output.
    assert rows["a"]["actual_above_forecast_share"] == pytest.approx(2 / 3)
    assert rows["a"]["actual_above_forecast_share"] != rows["b"]["actual_above_forecast_share"]


def test_candidate_winning_one_year_gets_only_that_years_events():
    fake = FakePaired()
    rows = _by_id(_run(fake))
    assert rows["c"]["outer_years"] == ["2025"]
    assert rows["c"]["paired"] == {"fake": True, "n_events": 1}
    assert ["e4"] in [call["events"] for call in fake.calls]
    c_call = next(call for call in fake.calls if call["events"] == ["e4"])
    assert c_call["kwargs"]["implied_moves"] == {("e4", 1): 0.7}
    assert c_call["kwargs"]["implied_scales"] == PAIRED_KWARGS["implied_scales"]
    assert rows["b"]["paired"] == {"fake": True, "n_events": 3}


# Real paired_report oracle: two outer years, one event pair per year.
REAL_EVENTS = [{"event_id": e, "ticker": t, "event_date": d} for e, t, d in (
    ("e0", "A", "2024-01-18"), ("e1", "B", "2024-04-18"),
    ("e2", "A", "2025-01-16"), ("e3", "B", "2025-04-17"))]
REAL_TARGETS = {"e0": 2.0, "e1": 6.0, "e2": 10.0, "e3": 14.0}
REAL_SELECTION = {"selections": [
    {"outer_year": "2024", "scores": [],
     "winners": {"overall": {"candidate_id": "a", "tie_band": 0.1},
                 "rmse": {"candidate_id": "b", "tie_band": 0.1}},
     "selected_candidate_id": "a"},
    {"outer_year": "2025", "scores": [],
     "winners": {"overall": {"candidate_id": "a", "tie_band": 0.1}},
     "selected_candidate_id": "a"}]}
REAL_REFITS = [{"outer_year": "2024", "candidate_id": "a", "status": "completed", "result": {}},
               {"outer_year": "2024", "candidate_id": "b", "status": "completed", "result": {}},
               {"outer_year": "2025", "candidate_id": "a", "status": "completed", "result": {}}]


def _real_plan():
    return {"plan_id": "p", "horizons": [1, 3],
            "candidates": [{"candidate_id": "a"}, {"candidate_id": "b"}],
            "folds": [{"outer": {"year": "2024", "evaluation_event_ids": ["e0", "e1"]}},
                      {"outer": {"year": "2025", "evaluation_event_ids": ["e2", "e3"]}}],
            "selection": {"winner_rules": ["overall", "rmse"]}}


def _real_forecasts(preds):
    return [{"event_id": e, "horizon_sessions": h, "predicted_absolute_move_pct": p}
            for e, p in preds.items() for h in (1, 3)]


REAL_FORECASTS = {
    ("2024", "a"): _real_forecasts({"e0": 2.5, "e1": 6.5}),
    ("2025", "a"): _real_forecasts({"e2": 10.5, "e3": 14.5}),
    ("2024", "b"): _real_forecasts({"e0": 3.0, "e1": 7.0}),
}
# Incumbent is target - 2 at every horizon, so its absolute error is 2 everywhere.
REAL_COMPARISONS = [{"event_id": e, "horizon_sessions": h, "predicted_absolute_move_pct": REAL_TARGETS[e] - 2,
                     "abstention": None, "history_mean": 4.0, "quote_status": "present"}
                    for e in REAL_TARGETS for h in (1, 3)]


def test_real_paired_report_scores_each_winner_over_its_own_outer_years():
    rows = _by_id(winner_rows(_real_plan(), REAL_SELECTION, REAL_REFITS, REAL_FORECASTS, REAL_EVENTS,
                              REAL_COMPARISONS, REAL_TARGETS, paired=paired_report,
                              paired_kwargs={"horizons": [1, 3], "implied_moves": None,
                                             "implied_scales": {}, "label_reasons": {}}))
    assert rows["a"]["status"] == "completed" and rows["b"]["status"] == "completed"
    # a wins in both years: all four events, error 0.5 vs incumbent error 2.0.
    assert rows["a"]["paired"]["paired_events"] == 4
    assert rows["a"]["paired"]["difference"]["mae"] == pytest.approx(0.5 - 2)
    # b wins only rmse in 2024: only e0 and e1 are paired.
    assert rows["b"]["paired"]["paired_events"] == 2
    assert rows["b"]["paired"]["difference"]["mae"] == pytest.approx(1.0 - 2)
    assert {r["event_id"] for r in rows["b"]["paired"]["paired_rows"]} == {"e0", "e1"}
