"""Independent synthetic fixed-bridge/control oracles; no execution or fits."""
import math
import random
import socket
import subprocess

import pytest

from engine.v2.data.errors import DataError
from engine.v2.research import magnitude_distribution as distribution
from engine.v2.research import magnitude_distribution_payoff as payoff


def _histogram(masses, *, tail=None):
    probabilities = [0.0] * 81
    for index, mass in masses.items():
        probabilities[index] = mass
    return distribution.histogram(probabilities, tail_excesses=[0] if tail is None else tail)


def _summary(document, **changes):
    arguments = {"spot": 100, "entry_cost": 4, "intercept": 0, "slope": 0.01,
                 "payoff_residuals": [0], "reconstruction": "lower"}
    arguments.update(changes)
    return payoff.bridge_summary(document, **arguments)


def _refuses(call, *args, expected_message="invalid magnitude payoff inputs or arithmetic", **kwargs):
    with pytest.raises(DataError) as error:
        call(*args, **kwargs)
    assert error.value.code == "CONTRACT_MISMATCH"
    assert error.value.problem.retryable is False
    assert error.value.problem.message == expected_message
    return error.value


def test_same_mean_linear_bridge_same_ev_different_profit_probability():
    # A=3 certainly versus A=0/6 equally: E[A]=3, not a profit guarantee.
    concentrated = _summary(_histogram({12: 1}))
    dispersed = _summary(_histogram({0: 0.5, 24: 0.5}))
    assert concentrated == {"expected_profit": -1.0, "probability_profit": 0.0,
                            "probability_negative_exit": 0.0, "mean_exit_value": 3.0}
    assert dispersed == {"expected_profit": -1.0, "probability_profit": 0.5,
                         "probability_negative_exit": 0.0, "mean_exit_value": 3.0}


def test_active_clip_can_change_ev_at_fixed_mean():
    arguments = {"intercept": -0.04, "entry_cost": 0.5}
    concentrated = _summary(_histogram({12: 1}), **arguments)
    dispersed = _summary(_histogram({0: 0.5, 24: 0.5}), **arguments)
    assert concentrated["expected_profit"] == -0.5
    assert concentrated["probability_profit"] == 0
    assert dispersed["expected_profit"] == pytest.approx(0.5)
    assert dispersed["probability_profit"] == 0.5


def test_nonlinear_illustrative_option_marks_are_not_linear_bridge_evidence():
    # Explicit hypothetical marks, NOT an option repricing implementation/data.
    marks, cost = {0: 1.60, 3: 3.12, 6: 6.00}, 3.50
    concentrated, dispersed = [(3, 1.0)], [(0, 0.5), (6, 0.5)]
    assert sum(move * weight for move, weight in concentrated) == 3
    assert sum(move * weight for move, weight in dispersed) == 3
    assert sum((marks[move] - cost) * weight for move, weight in concentrated) == pytest.approx(-0.38)
    assert sum((marks[move] - cost) * weight for move, weight in dispersed) == pytest.approx(0.30)
    assert sum(weight for move, weight in concentrated if marks[move] > cost) == 0
    assert sum(weight for move, weight in dispersed if marks[move] > cost) == 0.5
    # The corresponding inactive linear bridge correctly gives equal EV.
    assert _summary(_histogram({12: 1}))["expected_profit"] == -1
    assert _summary(_histogram({0: 0.5, 24: 0.5}))["expected_profit"] == -1


@pytest.mark.parametrize("slope,intercept", [(0.01, -0.03875), (-0.01, 0.03875)])
def test_uniform_clip_crossing_has_exact_atom_and_integrated_positive_part(slope, intercept):
    # A~U[3.75,4): line ranges -0.125..0.125 dollars. Half the mass clips to 0.
    result = _summary(_histogram({15: 1}), slope=slope, intercept=intercept,
                      entry_cost=0, reconstruction="uniform")
    assert result["mean_exit_value"] == pytest.approx(0.03125)
    assert result["probability_profit"] == pytest.approx(0.5)
    assert result["probability_negative_exit"] == 0


def test_negative_slope_uniform_profit_inequality_points_the_other_way():
    # V=5-A for A~U[0,.25); V>4.9 iff A<.1.
    result = _summary(_histogram({0: 1}), intercept=0.05, slope=-0.01,
                      entry_cost=4.9, reconstruction="uniform")
    assert result["mean_exit_value"] == pytest.approx(4.875)
    assert result["expected_profit"] == pytest.approx(-0.025)
    assert result["probability_profit"] == pytest.approx(0.4)


@pytest.mark.parametrize("reconstruction,mean,probability", [
    ("lower", 3.0, 0.0), ("uniform", 3.125, 0.5), ("upper", 3.25, 1.0),
])
def test_finite_bin_endpoint_sensitivity_is_explicit(reconstruction, mean, probability):
    result = _summary(_histogram({12: 1}), entry_cost=3.125, reconstruction=reconstruction)
    assert result["mean_exit_value"] == pytest.approx(mean)
    assert result["probability_profit"] == pytest.approx(probability)


def test_residual_is_after_clip_and_negative_exit_is_not_repaired():
    result = _summary(_histogram({12: 1}), intercept=-1, slope=0, entry_cost=1,
                      payoff_residuals=[-0.01, 0.02])
    assert result == {"expected_profit": -0.5, "probability_profit": 0.5,
                      "probability_negative_exit": 0.5, "mean_exit_value": 0.5}


@pytest.mark.parametrize("intercept,residuals,cost,win,negative", [
    (0.01, [0], 1, 0, 0),  # Exact break-even is not profit.
    (0, [0], 0, 0, 0),     # Zero exit value is not negative.
    (0, [-0.01, 0.01], 1, 0, 0.5),
    (0, [-0.01, 0.02], 0, 0.5, 0.5),
    (0.01, [-0.01], 0, 0, 0),  # Exact cancellation remains zero.
])
def test_strict_profit_and_negative_exit_atom_boundaries(intercept, residuals, cost, win, negative):
    result = _summary(_histogram({80: 1}, tail=[0, 100]), intercept=intercept,
                      slope=0, entry_cost=cost, payoff_residuals=residuals)
    assert result["probability_profit"] == win
    assert result["probability_negative_exit"] == negative


def test_decimal_break_even_atom_uses_direct_addition_and_float_neighbors():
    assert 16.19 + 82.81 == 99.0
    for cost in (math.nextafter(99.0, -math.inf), 99.0, math.nextafter(99.0, math.inf)):
        for shift in (math.nextafter(82.81, -math.inf), 82.81, math.nextafter(82.81, math.inf)):
            result = payoff.empirical_bridge_summary([0], spot=1, entry_cost=cost,
                intercept=16.19, slope=0, payoff_residuals=[shift])
            assert result["expected_profit"] == (16.19 + shift) - cost
            assert result["probability_profit"] == float(16.19 + shift > cost)
    result = payoff.empirical_bridge_summary([0], spot=1, entry_cost=99,
        intercept=16.19, slope=0, payoff_residuals=[82.81])
    assert result["expected_profit"] == result["probability_profit"] == 0


def test_decimal_zero_exit_atom_and_neighbors_use_direct_comparisons():
    for shift in (math.nextafter(-16.19, -math.inf), -16.19, math.nextafter(-16.19, math.inf)):
        result = payoff.empirical_bridge_summary([0, 0], spot=1, entry_cost=0,
            intercept=16.19, slope=0, payoff_residuals=[shift])
        assert result["probability_negative_exit"] == float(16.19 + shift < 0)
        assert result["probability_profit"] == float(16.19 + shift > 0)


def test_empirical_tail_zero_duplicate_atoms_and_no_cap():
    result = _summary(_histogram({80: 1}, tail=[0, 0, 20, 80]), entry_cost=20)
    assert result["mean_exit_value"] == 45
    assert result["expected_profit"] == 25
    assert result["probability_profit"] == 0.5
    assert result["probability_negative_exit"] == 0


def test_supplied_residual_mean_changes_ev_without_changing_move_distribution():
    document = _histogram({0: 0.5, 24: 0.5})
    result = _summary(document, payoff_residuals=[0.01, 0.03])
    assert result["expected_profit"] == pytest.approx(1)
    assert result["mean_exit_value"] == pytest.approx(5)
    assert result["probability_profit"] == 0.5


def test_resolved_tail_required_even_when_its_probability_is_zero():
    probabilities = [1] + [0] * 80
    document = distribution.histogram(probabilities)
    _refuses(_summary, document, expected_message="unresolved magnitude distribution tail")
    assert _summary(_histogram({0: 1}))["mean_exit_value"] == 0


@pytest.mark.parametrize("target,residuals,draws,raw_mean,scale", [
    (3, [-6, 0, 6], [0, 2.25, 6.75], 4, 0.75),
    (2, [-4, 0, 4], [0, 1.5, 4.5], 8 / 3, 0.75),
    (0.5, [-1, 0, 1], [0, 0.375, 1.125], 2 / 3, 0.75),
    (5, [-5, 0, 5], [0, 5, 10], 5, 1),
])
def test_mean_matched_control_records_support_repair_and_rescaling(target, residuals, draws, raw_mean, scale):
    before = residuals.copy()
    result = payoff.mean_matched_control(target, residuals)
    assert result == {"magnitudes": draws, "target_mean": target, "raw_mean": raw_mean,
                      "scale": scale, "zero_count": 1, "anchored_mean": target}
    assert residuals == before
    result["magnitudes"][1] = 99
    assert payoff.mean_matched_control(target, residuals)["magnitudes"] == draws


def test_control_preserves_order_duplicate_draws_and_unchanged_valid_mean():
    result = payoff.mean_matched_control(3, [-3, 3, -3, 3])
    assert result["magnitudes"] == [0, 6, 0, 6]
    assert result["scale"] == 1
    assert result["zero_count"] == 2
    assert result["anchored_mean"] == 3


def test_mean_control_atoms_reuse_identical_bridge_without_histogramming():
    control = payoff.mean_matched_control(3, [-3, 3])
    arguments = {"spot": 100, "entry_cost": 4, "intercept": 0, "slope": 0.01,
                 "payoff_residuals": [0]}
    result = payoff.empirical_bridge_summary(control["magnitudes"], **arguments)
    assert result == _summary(_histogram({0: 0.5, 24: 0.5}))
    assert result["probability_profit"] == 0.5
    assert result["expected_profit"] == -1
    zero = payoff.empirical_bridge_summary([0, 0], **{**arguments, "entry_cost": 0})
    assert zero["mean_exit_value"] == 0
    assert zero["probability_profit"] == 0
    for bad in ([], [-1], [True], [float("nan")], (0,)):
        _refuses(payoff.empirical_bridge_summary, bad, **arguments)


@pytest.mark.parametrize("target,residuals", [(0, [0]), (-1, [0]), (3, [-3]),
                                           (3, [-6, -4]), (1e308, [1e308]),
                                           (5e-324, [1e308])])
def test_control_undefined_or_overflowing_anchor_refuses(target, residuals):
    _refuses(payoff.mean_matched_control, target, residuals)


class _Float(float):
    pass


class _List(list):
    pass


@pytest.mark.parametrize("bad", [True, False, None, "secret", float("nan"), float("inf"),
                               -float("inf"), 10 ** 1000, _Float(1)])
def test_invalid_scalar_fields_and_sample_elements_refuse(bad):
    for field in ("spot", "entry_cost", "intercept", "slope"):
        error = _refuses(_summary, _histogram({0: 1}), **{field: bad})
        assert error.problem.message == "invalid magnitude payoff inputs or arithmetic"
    _refuses(_summary, _histogram({0: 1}), payoff_residuals=[bad])
    _refuses(payoff.mean_matched_control, bad, [0])
    _refuses(payoff.mean_matched_control, 1, [bad])


@pytest.mark.parametrize("bad", [[], (), (0,), {}, None, "secret", _List([0])])
def test_nonempty_plain_sample_lists_required(bad):
    _refuses(_summary, _histogram({0: 1}), payoff_residuals=bad)
    _refuses(payoff.mean_matched_control, 1, bad)


@pytest.mark.parametrize("changes", [
    {"spot": 0}, {"spot": -1}, {"entry_cost": -1},
    {"spot": 1e308, "intercept": 1e308}, {"slope": 1e308},
    {"spot": 1e308, "payoff_residuals": [1e308]},
    {"entry_cost": 1e308, "spot": 1, "payoff_residuals": [-1e308]},
    {"reconstruction": "midpoint"},
])
def test_bad_support_policy_or_nonfinite_arithmetic_refuses(changes):
    message = "invalid magnitude distribution arithmetic" if "reconstruction" in changes else "invalid magnitude payoff inputs or arithmetic"
    _refuses(_summary, _histogram({80: 1}, tail=[100]), expected_message=message, **changes)


def test_public_required_argument_type_errors_are_preserved():
    with pytest.raises(TypeError):
        payoff.bridge_summary()
    with pytest.raises(TypeError):
        payoff.mean_matched_control()
    with pytest.raises(TypeError):
        payoff.empirical_bridge_summary()
    assert payoff.__all__ == ["bridge_summary", "empirical_bridge_summary", "mean_matched_control"]


def test_bounded_tail_residual_queries_not_cartesian_product(monkeypatch):
    document = _histogram({0: 0.5, 80: 0.5}, tail=list(range(200)))
    calls, component_calls = [], []
    original_cdf, original_components = payoff._cdf, payoff.components

    def count_cdf(*args, **kwargs):
        calls.append(1)
        assert len(args[0][0]) <= 80
        return original_cdf(*args, **kwargs)

    def count_components(*args, **kwargs):
        component_calls.append(1)
        return original_components(*args, **kwargs)

    monkeypatch.setattr(payoff, "_cdf", count_cdf)
    monkeypatch.setattr(payoff, "components", count_components)
    residuals = [0.01] * 101
    result = _summary(document, payoff_residuals=residuals, reconstruction="uniform")
    assert len(calls) == 2 * len(residuals)
    assert len(component_calls) == 1
    assert result == pytest.approx({"mean_exit_value": 60.8125, "expected_profit": 56.8125,
                                   "probability_profit": 0.5, "probability_negative_exit": 0})


def test_seeded_empirical_mixtures_match_independent_cartesian_oracle():
    generator = random.Random(197)
    for _ in range(50):
        magnitudes = [generator.randrange(120) / 4 for _ in range(11)]
        residuals = [generator.choice([-0.04, -0.01, 0, 0.02]) for _ in range(7)]
        intercept = generator.choice([-0.05, 0, 0.03])
        slope = generator.choice([-0.01, 0, 0.01])
        spot, cost = generator.choice([1, 10, 100]), generator.choice([0, 0.5, 2])
        exits = [max(0, (intercept + slope * magnitude) * spot) + residual * spot
                 for magnitude in magnitudes for residual in residuals]
        expected_mean = math.fsum(exits) / len(exits)
        result = payoff.empirical_bridge_summary(magnitudes, spot=spot, entry_cost=cost,
            intercept=intercept, slope=slope, payoff_residuals=residuals)
        assert result["mean_exit_value"] == pytest.approx(expected_mean, abs=1e-12)
        assert result["expected_profit"] == pytest.approx(expected_mean - cost, abs=1e-12)
        assert result["probability_profit"] == pytest.approx(sum(value > cost for value in exits) / len(exits))
        assert result["probability_negative_exit"] == pytest.approx(sum(value < 0 for value in exits) / len(exits))


def test_nonbinary_decimal_mixtures_match_direct_strict_oracle():
    generator = random.Random(991)
    for _ in range(100):
        magnitudes = [generator.randrange(4000) / 100 for _ in range(7)]
        residuals = [generator.randrange(-10000, 10000) / 100 for _ in range(9)]
        intercept, slope = generator.randrange(5000) / 100, generator.choice([-0.31, 0, 0.17])
        exits = [max(0, intercept + slope * magnitude) + residual
                 for magnitude in magnitudes for residual in residuals]
        cost = max(0, generator.choice(exits))  # Includes exact atom ties.
        result = payoff.empirical_bridge_summary(magnitudes, spot=1, entry_cost=cost,
            intercept=intercept, slope=slope, payoff_residuals=residuals)
        assert result["probability_profit"] == pytest.approx(sum(value > cost for value in exits) / len(exits))
        assert result["probability_negative_exit"] == pytest.approx(sum(value < 0 for value in exits) / len(exits))


def test_no_io_no_input_mutation_and_deterministic_replay(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("payoff helper attempted external I/O")

    document, residuals = _histogram({0: 0.25, 12: 0.75}), [-0.01, 0.02]
    expected = {"expected_profit": -1.25, "probability_profit": 0.375,
                "probability_negative_exit": 0.125, "mean_exit_value": 2.75}
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for _ in range(2):
        assert _summary(document, payoff_residuals=residuals) == expected
    assert document == _histogram({0: 0.25, 12: 0.75})
    assert residuals == [-0.01, 0.02]
    assert payoff.mean_matched_control(3, [-3, 3])["magnitudes"] == [0, 6]
