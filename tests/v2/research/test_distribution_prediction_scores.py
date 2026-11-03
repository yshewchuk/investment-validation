"""Neutral score imports and legacy export identity, without a study or fits."""
import json
import subprocess
import sys

from experiments import distribution_prediction_scores as neutral
from experiments import distribution_scores as legacy
from tests.v2.research.test_distribution_scores import forecast, pool


def test_strategy_forecast_exports_are_the_identical_neutral_functions():
    assert legacy.arm_forecasts is neutral.arm_forecasts
    assert legacy.score_forecasts is neutral.score_forecasts


def test_forecast_scoring_imports_and_executes_without_strategy_dependencies():
    payload = json.dumps({"result": forecast(), "pool": pool()})
    script = r'''
import importlib.abc
import json
import sys
blocked = {"experiments.distribution_scores", "experiments.distribution_run_request",
           "experiments.distribution_run_plan", "experiments.distribution_book",
           "experiments.distribution_market", "engine.v2.research.magnitude_distribution_payoff"}
class ForbidStrategy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname not in blocked, "strategy dependency: " + fullname
sys.meta_path.insert(0, ForbidStrategy())
from engine.v2.ops.errors import OpsError, fail
from experiments.distribution_prediction_scores import arm_forecasts, score_forecasts
data = json.loads(sys.stdin.read())
result, pool = data["result"], data["pool"]
values = arm_forecasts(result["rows"][0], result["tail"], pool, receipt=result["receipt"])
assert values["matched_point"] == [0, 1]
rows = list(score_forecasts(result, pool, {"e": .125}, outer_year="2026", horizon=1))
assert [row["arm"] for row in rows] == ["histogram", "same_mean", "matched_point"]
assert abs(rows[0]["crps"] - 1/48) < 1e-15
assert rows[0]["rps"] == rows[0]["log_loss"] == 0
assert rows[2]["crps"] == .25
missing = list(score_forecasts(result, pool, {"e": {"omitted": "no_label"}}, outer_year="2026", horizon=1))
assert len(missing) == 3 and all(row["status"] == "missing_target" for row in missing)
assert all(row["reason"] == "no_label" and "crps" not in row for row in missing)
result["receipt"]["claim"]["plan_id"] = "foreign"
try:
    next(score_forecasts(result, pool, None, outer_year="2026", horizon=1))
except OpsError as error:
    assert error.problem == fail("INVALID_EXPERIMENT_SPEC", "invalid distribution run contract").problem
else:
    raise AssertionError("foreign evidence reached target access")
assert not blocked.intersection(sys.modules)
print("neutral-forecast-scores-ok")
'''
    child = subprocess.run([sys.executable, "-c", script], input=payload,
        capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.strip() == "neutral-forecast-scores-ok"
