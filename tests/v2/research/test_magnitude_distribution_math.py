"""Independent exact-score/atom oracles; no market payloads or fitted models."""
import copy
import math
import random
import socket
import subprocess

import pytest

from engine.v2.data.errors import DataError
from engine.v2.research import magnitude_distribution_math as scoring
from engine.v2.research.magnitude_distribution import histogram


def _distribution(weights=None, tail=None):
    masses = [0.0] * 81
    for index, mass in (weights or {0: 1.0}).items():
        masses[index] = mass
    return histogram(masses, tail_excesses=tail)


def _refuses(function, *args, expected_message="invalid magnitude distribution arithmetic", **kwargs):
    with pytest.raises(DataError) as error:
        function(*args, **kwargs)
    assert error.value.code == "CONTRACT_MISMATCH"
    assert error.value.problem.retryable is False
    assert error.value.problem.message == expected_message


@pytest.mark.parametrize("mode,expected", [
    ("uniform", ((0.0, 0.25, 1.0),)),
    ("lower", ((0.0, 0.0, 1.0),)),
    ("upper", ((0.25, 0.25, 1.0),)),
])
def test_literal_components_and_endpoint_sensitivity(mode, expected):
    document = _distribution(tail=[0])
    assert scoring.components(document, mode) == expected
    assert scoring.mean(document, reconstruction=mode) == {"uniform": 0.125, "lower": 0, "upper": 0.25}[mode]
    assert scoring.crps(document, 0, reconstruction=mode) == pytest.approx(
        {"uniform": 1 / 12, "lower": 0, "upper": 0.25}[mode])
    assert scoring.bin_scores(document, 0.25) == {"rps": 1, "log_loss": math.inf}


@pytest.mark.parametrize("observation,expected", [(0, 1 / 12), (0.125, 1 / 48),
                                                (0.25, 1 / 12), (2, 11 / 6)])
def test_uniform_crps_closed_form(observation, expected):
    assert scoring.crps(_distribution(tail=[0]), observation) == pytest.approx(expected)


def test_separated_uniform_mixture_crps_and_mean():
    document = _distribution({0: 0.5, 4: 0.5}, [0])
    assert scoring.mean(document) == 0.625
    assert scoring.crps(document, 0.5) == pytest.approx(11 / 48)
    assert scoring.crps(document, 0.5, reconstruction="lower") == 0.25
    assert scoring.crps(document, 0.5, reconstruction="upper") == 0.25


def test_tail_zero_atom_duplicates_and_strict_cdf():
    document = _distribution({80: 1}, [0, 0, 2])
    assert scoring.components(document) == ((20.0, 20.0, 1 / 3),) * 2 + ((22.0, 22.0, 1 / 3),)
    assert scoring.cdf(document, 20) == pytest.approx(2 / 3)
    assert scoring.cdf(document, 20, strict=True) == 0
    assert scoring.cdf(document, 22, strict=True) == pytest.approx(2 / 3)
    assert scoring.cdf(document, 22) == 1
    assert scoring.crps(document, 21) == pytest.approx(5 / 9)
    assert scoring.mean(document) == pytest.approx(62 / 3)
    assert scoring.bin_scores(document, 20) == {"rps": 0, "log_loss": 0}


def test_mixed_tail_and_uniform_threshold_oracles():
    document = _distribution({0: 0.25, 80: 0.75}, [0, 4])
    for threshold, expected in [(-1, 0), (0, 0), (0.125, 0.125), (0.25, 0.25),
                                (19.9, 0.25), (20, 0.625), (24, 1), (1e308, 1)]:
        assert scoring.cdf(document, threshold) == expected
    assert scoring.cdf(document, 20, strict=True) == 0.25
    assert scoring.cdf(document, 24, strict=True) == 0.625
    assert scoring.mean(document) == 16.53125


@pytest.mark.parametrize("function,args", [
    (scoring.components, ()), (scoring.mean, ()), (scoring.cdf, (0,)),
    (scoring.crps, (0,)),
])
@pytest.mark.parametrize("weights", [{0: 1}, {80: 1}])
def test_unresolved_tail_refuses_full_distribution_even_with_zero_overflow(function, args, weights):
    document = _distribution(weights)
    _refuses(function, document, *args, expected_message="unresolved magnitude distribution tail")
    expected = {"rps": 80, "log_loss": math.inf} if 0 in weights else {"rps": 0, "log_loss": 0}
    assert scoring.bin_scores(document, 100) == expected


@pytest.mark.parametrize("observation,rps,log_loss", [(0, 45, math.log(4)),
                                                     (20, 5, -math.log(0.75)),
                                                     (100, 5, -math.log(0.75))])
def test_bin_score_literal_probability_oracles(observation, rps, log_loss):
    result = scoring.bin_scores(_distribution({0: 0.25, 80: 0.75}), observation)
    assert result == pytest.approx({"rps": rps, "log_loss": log_loss})


def test_perfect_and_impossible_bin_forecasts():
    first, last = _distribution(), _distribution({80: 1})
    assert scoring.bin_scores(first, 0) == {"rps": 0, "log_loss": 0}
    assert scoring.bin_scores(first, 20) == {"rps": 80, "log_loss": math.inf}
    assert scoring.bin_scores(last, 0) == {"rps": 80, "log_loss": math.inf}
    for index in range(81):
        document = _distribution({index: 1})
        assert scoring.bin_scores(document, index / 4) == {"rps": 0, "log_loss": 0}


@pytest.mark.parametrize("probability,outcome,expected", [(0.25, True, 0.5625),
    (0.25, False, 0.0625), (0, False, 0), (1, True, 0), (1, False, 1)])
def test_binary_brier_literal_oracles(probability, outcome, expected):
    assert scoring.brier(probability, outcome) == expected


def test_admitted_roundoff_is_explicitly_normalized_without_mutation():
    document = _distribution({0: 0.5 - 5e-13, 80: 0.5}, [0])
    before = copy.deepcopy(document)
    denominator = (0.5 - 5e-13) + 0.5
    expected = ((0.5 - 5e-13) * 0.125 + 0.5 * 20) / denominator
    assert scoring.mean(document) == pytest.approx(expected, abs=1e-14)
    assert scoring.cdf(document, 20) == 1
    assert math.fsum(weight for _, _, weight in scoring.components(document)) == 1
    assert scoring.bin_scores(document, 20)["log_loss"] == -math.log(0.5 / denominator)
    assert document == before


def test_huge_finite_tail_and_observation_are_uncapped():
    point = _distribution({80: 1}, [1e308])
    assert scoring.mean(point) == 1e308
    assert scoring.crps(point, 0) == 1e308
    assert scoring.crps(point, 1e308) == 0
    assert scoring.cdf(point, -1e308) == 0
    assert scoring.crps(_distribution(tail=[0]), 1e308, reconstruction="lower") == 1e308


def test_positive_tail_mass_cannot_silently_underflow_during_reconstruction():
    document = _distribution({0: 1, 80: math.nextafter(0, 1)}, [0, 1])
    _refuses(scoring.components, document, expected_message="unrepresentable magnitude distribution tail weight")
    assert math.isfinite(scoring.bin_scores(document, 20)["log_loss"])


def test_large_empirical_tail_matches_independent_sorted_distance_formula():
    rng = random.Random(8021)
    samples = sorted(20 + rng.randrange(0, 10000) / 100 for _ in range(2000))
    observation, size = 47.25, len(samples)
    expected = math.fsum(abs(value - observation) for value in samples) / size
    expected -= math.fsum((2 * index - size + 1) * value
                          for index, value in enumerate(samples)) / size ** 2
    rng.shuffle(samples)
    assert samples != sorted(samples)
    document = _distribution({80: 1}, [value - 20 for value in samples])
    assert scoring.crps(document, observation) == pytest.approx(expected, abs=1e-12)
    assert scoring.empirical_crps(samples, observation) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("samples,observation,expected", [([0], 0, 0), ([0], 1, 1),
    ([0, 2], 1, 0.5), ([0, 0, 2], 1, 5 / 9), ([0, 0], 2, 2), ([1e308], 0, 1e308)])
def test_empirical_control_score_retains_zero_atoms(samples, observation, expected):
    before = samples.copy()
    assert scoring.empirical_crps(samples, observation) == pytest.approx(expected)
    assert samples == before


@pytest.mark.parametrize("samples", [None, [], (), (0,), "0", [-1], [True],
    [float("nan")], [float("inf")], [10 ** 1000]])
def test_empirical_score_invalid_samples_refuse(samples):
    _refuses(scoring.empirical_crps, samples, 0)


def test_mixed_crps_matches_independent_pair_distance_identity():
    # Nonoverlapping uniforms and atoms: cross-component distance is the
    # distance between means; within one uniform it is width/3.
    terms = [(0, 0.25, 0.2), (1, 1.25, 0.3), (20, 20, 1 / 6),
             (21, 21, 1 / 6), (23, 23, 1 / 6)]
    document = _distribution({0: 0.2, 4: 0.3, 80: 0.5}, [0, 1, 3])
    for observation in (0, 0.1, 0.25, 1.125, 7, 20, 21.5, 100):
        absolute = 0
        for lo, hi, mass in terms:
            distance = abs((lo + hi) / 2 - observation)
            if lo < observation < hi:
                distance = ((observation - lo) ** 2 + (hi - observation) ** 2) / (2 * (hi - lo))
            absolute += mass * distance
        pair = 0
        for index, (lo, hi, mass) in enumerate(terms):
            pair += mass ** 2 * (hi - lo) / 3
            for other_lo, other_hi, other_mass in terms[index + 1:]:
                pair += 2 * mass * other_mass * ((other_lo + other_hi - lo - hi) / 2)
        assert scoring.crps(document, observation) == pytest.approx(absolute - pair / 2, abs=1e-13)


@pytest.mark.parametrize("value", [True, False, None, "0", float("nan"), float("inf"),
                                  -float("inf"), 10 ** 1000])
def test_invalid_numeric_inputs_refuse(value):
    document = _distribution(tail=[0])
    _refuses(scoring.cdf, document, value)
    _refuses(scoring.crps, document, value)
    _refuses(scoring.bin_scores, document, value, expected_message="invalid magnitude distribution")
    _refuses(scoring.brier, value, False)
    _refuses(scoring.empirical_crps, [0], value)


@pytest.mark.parametrize("mode", [None, True, "", "midpoint", [], {}])
def test_invalid_reconstruction_refuses(mode):
    document = _distribution(tail=[0])
    for function, args in [(scoring.components, ()), (scoring.mean, ()),
                           (scoring.cdf, (0,)), (scoring.crps, (0,))]:
        _refuses(function, document, *args, reconstruction=mode)


def test_invalid_flags_negative_targets_and_probabilities_refuse():
    document = _distribution(tail=[0])
    for bad in (0, 1, None, "true"):
        _refuses(scoring.cdf, document, 0, strict=bad)
        _refuses(scoring.brier, 0.5, bad)
    for bad in (-0.01, 1.01):
        _refuses(scoring.brier, bad, False)
    _refuses(scoring.crps, document, -1)
    _refuses(scoring.bin_scores, document, -1, expected_message="invalid magnitude distribution")
    document["probabilities"][0] = 0.9
    for function, args in [(scoring.components, ()), (scoring.mean, ()),
                           (scoring.cdf, (0,)), (scoring.crps, (0,)), (scoring.bin_scores, (0,))]:
        _refuses(function, document, *args, expected_message="invalid magnitude distribution")


def test_public_required_arguments_raise_type_error():
    for name in scoring.__all__:
        with pytest.raises(TypeError):
            getattr(scoring, name)()


def test_numeric_and_container_subclasses_are_not_plain_inputs():
    class Number(float):
        pass

    class Samples(list):
        pass

    class Mode(str):
        pass

    document = _distribution(tail=[0])
    _refuses(scoring.cdf, document, Number(0))
    _refuses(scoring.empirical_crps, Samples([0]), 0)
    _refuses(scoring.components, document, Mode("uniform"))


def test_no_io_and_detached_components(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source-free score attempted external I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    document = _distribution({80: 1}, [0, 2])
    parts = scoring.components(document)
    assert scoring.mean(document) == 21
    assert scoring.crps(document, 21) == 0.5
    assert scoring.cdf(document, 21) == 0.5
    assert scoring.bin_scores(document, 21) == {"rps": 0, "log_loss": 0}
    assert scoring.brier(0.5, True) == 0.25
    document["tail"]["excesses"][0] = 999
    assert parts == ((20.0, 20.0, 0.5), (22.0, 22.0, 0.5))
