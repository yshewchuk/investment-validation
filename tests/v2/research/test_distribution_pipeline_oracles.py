"""Hand-derived score oracles over published pipeline sections, with live mutants."""
from math import log
from types import SimpleNamespace

import pytest

from experiments import distribution_pipeline as pipeline
from experiments.magnitude_search_shards import iter_section
from tests.v2.research.test_distribution_pipeline import pipeline_case as pipeline_case
from tests.v2.research.test_distribution_run_plan import plan_case as plan_case
from tests.v2.research.test_distribution_scores import destination, receipt

# p U[1, 5/4] + (1-p) U[5/4, 3/2], observed y=5/4:
# CRPS = integral F^2 below y + (1-F)^2 above y = (p^2+(1-p)^2)/12.
# Only the bin-4 cumulative probability differs from truth: RPS=p^2;
# observed bin 5 has probability 1-p: class loss=-log(1-p).
# Zero-residual same_mean is the atom mu=11/8-p/4, so CRPS=|mu-5/4|.
# The matched point is the atom 1: CRPS=1/4, RPS=1, class loss=infinite.
EXPECTED = {
    1: {"histogram": (5 / 96, 1 / 16, log(4 / 3)), "same_mean": (1 / 16, 0, 0)},
    3: {"histogram": (1 / 24, 1 / 4, log(2)), "same_mean": (0, 0, 0)},
    5: {"histogram": (5 / 96, 9 / 16, log(4)), "same_mean": (1 / 16, 1, "infinite")},
}


@pytest.fixture
def score_case(pipeline_case, monkeypatch):
    case = pipeline_case
    original_decisions, original_outcomes = pipeline.decision_quotes, pipeline.outcome_quotes
    def decisions(*args, **kwargs):
        quoted = original_decisions(*args, **kwargs)
        return {alpha: {key: (row, SimpleNamespace(spot=100, cost=1.25))
            for key, (row, _) in rows.items()} for alpha, rows in quoted.items()}
    def outcomes(*args, **kwargs):
        return {key: {**row, "entry_cash": 125, "exit_cash": 150 if key.endswith("AAA") else 100}
            for key, row in original_outcomes(*args, **kwargs).items()}
    # At spot 100 this bridge gives per-share exit=A. Cost=5/4, zero fees:
    # h1 histogram P(profit)=3/4; its point controls have probabilities 1 and 0.
    # The supplied cash outcomes include both wins and losses, regardless of selection.
    monkeypatch.setattr(pipeline, "decision_quotes", decisions)
    monkeypatch.setattr(pipeline, "outcome_quotes", outcomes)
    monkeypatch.setattr(pipeline, "read_bridge", lambda *args: SimpleNamespace(intercept=0, slope=.01, residuals=[0]))
    return case


def exercise(case):
    for group in case.groups:
        results = {}
        for horizon, p in ((1, 1 / 4), (3, 1 / 2), (5, 3 / 4)):
            rows = [{"event_id": key, "probabilities": [0] * 4 + [p, 1 - p] + [0] * 75,
                "point_prediction": 1} for key in group["outer"]["evaluation_event_ids"]]
            results[horizon] = {"rows": rows, "receipt": receipt(destination(outer_year=group["outer"]["year"], horizon=horizon)),
                "tail": {"policy": "empirical_excess", "sample_count": 1, "excesses": [0]}}
        pipeline.evaluate_outer(case.run, group, case.fits, results, case.plan, case.books, case.quotes,
            case.opportunities, case.sections, "frozen-calendar")
    return pipeline.finish_reports(case.run, case.books, case.quotes, case.opportunities, case.sections)


def published_rows(case, prefix):
    return [row for section in case.sections if section["name"].startswith(prefix)
        for row in iter_section(case.store, section["name"], section["descriptor"], lambda row: (row["position"],))]


def assert_values(row, expected):
    for field, value in expected.items():
        assert row[field] == (value if isinstance(value, str) else pytest.approx(value)), f"{field} numeric oracle"


def assert_score_oracles(case, report):
    profits = published_rows(case, "profit-scores:")
    assert len(profits) == 36
    for row in profits:
        won = row["event_id"].endswith("AAA")
        probability = {"histogram": 3 / 4, "same_mean": 1, "matched_point": 0}[row["arm"]]
        # Brier=(p-y)^2: histogram win/loss=1/16,9/16; controls=0/1,1/0.
        losses = {"histogram": (1 / 16, 9 / 16), "same_mean": (0, 1), "matched_point": (1, 0)}
        assert row["observed_profit"] is won
        assert_values(row, {"probability_profit": probability, "brier": losses[row["arm"]][0 if won else 1]})
    forecasts = published_rows(case, "scores:")
    assert len(forecasts) == 36
    for row in forecasts:
        assert row["observation"] == 1.25 and row["status"] == "available"
        expected = EXPECTED[row["horizon"]].get(row["arm"], (1 / 4, 1, "infinite"))
        assert_values(row, dict(zip(("crps", "rps", "log_loss"), expected)))
    for horizon, expected in EXPECTED.items():
        result = report["forecast"]["horizons"][str(horizon)]
        assert_values(result["histogram"], dict(zip(("crps_mean", "rps_mean", "log_loss_mean"), expected["histogram"])))
        # Nonzero paired improvements: 1/96, -1/24, 1/96 at h1,h3,h5.
        assert_values(result, {"mean_crps_improvement": -1 / 24 if horizon == 3 else 1 / 96})


def test_published_profit_and_forecast_scores_match_hand_derived_oracles(score_case):
    assert_score_oracles(score_case, exercise(score_case))


@pytest.mark.parametrize("field", ["brier", "crps", "rps", "log_loss"])
@pytest.mark.parametrize("constant", [0., .5])
def test_score_oracles_kill_constant_output_mutants(score_case, monkeypatch, field, constant):
    # Plant the defect on the live call edge BEFORE publication. The original
    # scorer supplies unrelated fields only; expected values above never call it.
    name = "profit_scores" if field == "brier" else "score_forecasts"
    original, changed = getattr(pipeline, name), []
    def constant_output(*args, **kwargs):
        for row in original(*args, **kwargs):
            changed.append(row["event_id"])
            yield {**row, field: constant}
    monkeypatch.setattr(pipeline, name, constant_output)
    report = exercise(score_case)
    assert len(changed) == 36
    with pytest.raises(AssertionError, match=rf"\b{field}\b numeric oracle"):
        assert_score_oracles(score_case, report)
