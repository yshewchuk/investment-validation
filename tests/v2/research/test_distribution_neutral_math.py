"""Hand-derived neutral-control/CRPS oracles; no fit, quote or source admission."""
import builtins
import subprocess
import sys
from copy import deepcopy

import pytest

from engine.v2.data.errors import DataError, fail
from engine.v2.research import magnitude_distribution_math as math
from engine.v2.research import magnitude_distribution_payoff as payoff
from experiments import distribution_oof as oof


@pytest.mark.parametrize("samples,observation,expected", [
    ([0, 1], 0, .25), ([0, 1], .125, .25), ([0, 1], .5, .25),
    ([0, 1], 1, .25), ([0, 1], 2, 1.25),
    ([0, 0, 2], 1, 5 / 9), ([20, 20, 22], 21, 5 / 9),
    ([0, 5, 10, 20], 5, 35 / 16), ([0, 2, 6], 3, 1),
    ([0, 0, 0], 0, 0), ([0, 0, 0], 2, 2),
])
def test_prepared_empirical_crps_matches_hand_derived_atom_oracles(samples, observation, expected):
    # CRPS = mean absolute observation error - sum(unordered pair distances)/n².
    # E.g. [0,5,10,20], y=5: 25/4 - 65/16 = 35/16; no production oracle.
    assert math.prepare_empirical_crps(samples)(observation) == pytest.approx(expected)


@pytest.mark.parametrize("samples", [[0], [2, 0, 0], [0, 1, 5, 20, 25],
    [1e-300, 2e-300], [1e308, 1e308]])
@pytest.mark.parametrize("observation", [0, .125, 5, 20, 100])
def test_prepared_path_preserves_existing_empirical_score_exactly(samples, observation):
    assert math.prepare_empirical_crps(samples)(observation) == math.empirical_crps(samples, observation)


def test_samples_validate_sort_and_detach_once_then_reuse_immutable_atoms(monkeypatch):
    calls, seen = [], []
    scalar, integrate = math._scalar, math._crps_components
    def validated(value, **kwargs):
        calls.append(value)
        return scalar(value, **kwargs)
    def sorted_once(values):
        seen.append("sorted")
        return builtins.sorted(values)
    parts = []
    def integrated(atoms, observation):
        parts.append(atoms)
        return integrate(atoms, observation)
    monkeypatch.setattr(math, "_scalar", validated)
    monkeypatch.setattr(math, "sorted", sorted_once, raising=False)
    monkeypatch.setattr(math, "_crps_components", integrated)
    samples = [2, 0, 0]
    score = math.prepare_empirical_crps(samples)
    assert calls == samples and seen == ["sorted"] and parts == []
    samples[:] = [999, -1]  # Caller mutation must not change the prepared law.
    monkeypatch.setattr(math, "sorted", lambda *args: pytest.fail("shared baseline sorted again"))
    assert [score(value) for value in (1, 0, 1)] == pytest.approx([5 / 9, 2 / 9, 5 / 9])
    assert calls == [2, 0, 0, 1, 0, 1]  # Only observations are revalidated.
    assert parts[0] is parts[1] is parts[2]
    assert parts[0] == ((0., 0., 1 / 3), (0., 0., 1 / 3), (2., 2., 1 / 3))


@pytest.mark.parametrize("samples", [None, (), [], [True], ["1"], [-1], [None],
    [float("nan")], [float("inf")], [float("-inf")], [10 ** 1000]])
def test_preparation_refuses_invalid_empirical_samples_with_existing_problem(samples):
    with pytest.raises(DataError) as caught:
        math.prepare_empirical_crps(samples)
    assert caught.value.problem == fail("CONTRACT_MISMATCH", "invalid magnitude distribution arithmetic").problem


@pytest.mark.parametrize("observation", [None, True, "1", -1, float("nan"), float("inf"), 10 ** 1000])
def test_prepared_scorer_preserves_observation_refusals_and_survives_bad_call(observation):
    score = math.prepare_empirical_crps([0, 1])
    with pytest.raises(DataError) as caught:
        score(observation)
    assert caught.value.problem == fail("CONTRACT_MISMATCH", "invalid magnitude distribution arithmetic").problem
    assert score(.5) == .25


@pytest.mark.parametrize("target,residuals,expected", [
    (2, [-3, 0, 2], {"magnitudes": [0., 2., 4.], "target_mean": 2., "raw_mean": 2.,
        "scale": 1., "zero_count": 1, "anchored_mean": 2.}),
    (2, [-3, 0, 4], {"magnitudes": [0., 1.5, 4.5], "target_mean": 2., "raw_mean": 8 / 3,
        "scale": .75, "zero_count": 1, "anchored_mean": 2.}),
    (.5, [-1, .5, .5], {"magnitudes": [0., .75, .75], "target_mean": .5, "raw_mean": 2 / 3,
        "scale": .75, "zero_count": 1, "anchored_mean": .5}),
])
def test_neutral_mean_control_has_exact_hand_derived_support_repair(target, residuals, expected):
    before = deepcopy(residuals)
    result = math.mean_matched_control(target, residuals)
    assert result == expected and residuals == before
    result["magnitudes"].clear()
    assert math.mean_matched_control(target, residuals) == expected


@pytest.mark.parametrize("target,residuals", [
    (0, [0]), (-1, [0]), (True, [0]), ("1", [0]), (None, [0]), (10 ** 1000, [0]),
    (float("nan"), [0]), (float("inf"), [0]), (1, None), (1, (0,)), (1, []),
    (1, [True]), (1, ["1"]), (1, [float("nan")]), (1, [float("inf")]),
    (1, [10 ** 1000]), (1, [-1, -2]), (1e308, [1e308]), (1e308, [-1e308, 0]),
    (5e-324, [0, 1e308]),
])
def test_neutral_mean_control_retains_the_full_legacy_payoff_problem(target, residuals):
    expected = fail("CONTRACT_MISMATCH", "invalid magnitude payoff inputs or arithmetic").problem
    for operation in (math.mean_matched_control, payoff.mean_matched_control, oof.mean_matched_control):
        with pytest.raises(DataError) as caught:
            operation(target, residuals)
        assert caught.value.problem == expected


def test_old_payoff_api_is_an_identity_preserving_reexport():
    assert payoff.__all__ == ["bridge_summary", "empirical_bridge_summary", "mean_matched_control"]
    assert payoff.mean_matched_control is oof.mean_matched_control is math.mean_matched_control
    assert {"mean_matched_control", "prepare_empirical_crps"} <= set(math.__all__)


def test_neutral_math_and_oof_import_and_execute_without_strategy_modules():
    script = r'''
import importlib.abc
import sys
blocked = {"engine.v2.research.magnitude_distribution_payoff", "experiments.distribution_book",
           "experiments.distribution_run_request", "experiments.distribution_market", "experiments.distribution_scores"}
class ForbidStrategy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname not in blocked, "strategy dependency: " + fullname
sys.meta_path.insert(0, ForbidStrategy())
from engine.v2.foundation import content_hash
from engine.v2.research.magnitude_distribution import histogram
from engine.v2.research.magnitude_distribution_math import mean_matched_control, prepare_empirical_crps
from experiments.distribution_oof import control_for_prediction
assert prepare_empirical_crps([0, 1])(.5) == .25
assert mean_matched_control(2, [-3, 0, 2])["magnitudes"] == [0., 2., 4.]
pool = {"schema_version": "distribution_oof_pool.v1", "policy": "earlier_inner_equal_weight_per_horizon.v1",
        "point_status": "ready", "histogram_status": "ready", "point_residuals": [0], "histogram_residuals": [0]}
pool["pool_id"] = content_hash(pool)
document = histogram([.5, .5] + [0] * 79, tail_excesses=[0])
result = control_for_prediction(pool, document, .5)
assert result["point"]["magnitudes"] == [.5] and result["histogram"]["magnitudes"] == [.25]
assert not blocked.intersection(sys.modules)
print("neutral-import-ok")
'''
    child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.strip() == "neutral-import-ok"
