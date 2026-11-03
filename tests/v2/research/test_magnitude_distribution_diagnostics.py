"""Literal interval/atom oracles; no source data, fitted calibration or economics."""
import copy
import math
import socket
import tracemalloc

import pytest

from engine.v2.data.errors import DataError
from engine.v2.research import magnitude_distribution_diagnostics as diagnostics
from engine.v2.research.magnitude_distribution import histogram


def _law(weights=None, tail=(0,)):
    masses = [0.0] * 81
    for index, mass in (weights or {0: 1}).items():
        masses[index] = mass
    return histogram(masses, tail_excesses=None if tail is None else list(tail))


def _refuses(function, *args, message=None):
    with pytest.raises(DataError) as error:
        function(*args)
    assert error.value.code == "CONTRACT_MISMATCH"
    assert error.value.problem.retryable is False
    if message is not None:
        assert error.value.problem.message == message


@pytest.mark.parametrize("probability,expected", [
    (0, 0), (0.125, 0.125), (0.25, 0.25), (0.375, 1.125),
    (0.5, 1.25), (math.nextafter(0.5, 1), 20), (2 / 3, 20),
    (math.nextafter(2 / 3, 1), 24), (0.9, 24), (1, 24),
])
def test_uniform_gap_and_duplicate_tail_inverse_oracles(probability, expected):
    document = _law({0: 0.25, 4: 0.25, 80: 0.5}, (4, 0, 4))
    assert diagnostics.quantile(document, probability) == expected


def test_support_endpoints_ignore_empty_bins_and_zero_overflow_mass():
    document = _law({4: 1}, (1000,))
    assert diagnostics.quantile(document, 0) == 1
    assert diagnostics.quantile(document, 1) == 1.25
    assert diagnostics.quantile(_law({80: 1}, (1e308,)), 1) == 1e308
    assert diagnostics.pit_bracket(_law({80: 1}, (1e308,)), 1e308) == (0, 1)


def test_fixed_equal_tail_uniform_intervals_are_inclusive_and_report_width():
    document = _law({4: 1})
    expected = [(0.5, 1.0625, 1.1875), (0.8, 1.025, 1.225),
                (0.9, 1.0125, 1.2375), (0.95, 1.00625, 1.24375)]
    result = diagnostics.interval_diagnostics(document, 1.1)
    assert [row["level"] for row in result] == [0.5, 0.8, 0.9, 0.95]
    for row, (level, lower, upper) in zip(result, expected, strict=True):
        assert row == pytest.approx({"level": level, "lower": lower, "upper": upper,
                                     "width": upper - lower, "covered": True})
        index = [0.5, 0.8, 0.9, 0.95].index(level)
        assert diagnostics.interval_diagnostics(document, row["lower"])[index]["covered"] is True
        assert diagnostics.interval_diagnostics(document, row["upper"])[index]["covered"] is True
        below = math.nextafter(row["lower"], 0)
        above = math.nextafter(row["upper"], math.inf)
        assert diagnostics.interval_diagnostics(document, below)[index]["covered"] is False
        assert diagnostics.interval_diagnostics(document, above)[index]["covered"] is False


@pytest.mark.parametrize("observation,expected", [
    (0, (0, 0)), (0.125, (0.125, 0.125)), (19, (0.25, 0.25)),
    (20, (0.25, 0.75)), (23, (0.75, 0.75)), (24, (0.75, 1)), (100, (1, 1)),
])
def test_pit_brackets_retain_tail_atoms(observation, expected):
    assert diagnostics.pit_bracket(_law({0: 0.25, 80: 0.75}, (4, 0, 0)), observation) == expected


@pytest.mark.parametrize("probability,expected", [
    (0, 0), (0.25, 0), (0.75, 0), (math.nextafter(0.75, 1), 2), (1, 2),
])
def test_empirical_inverse_zero_and_duplicate_atoms(probability, expected):
    assert diagnostics.empirical_quantile([2, 0, 0, 0], probability) == expected


def test_empirical_cdf_rank_boundaries_do_not_round_probability_times_count():
    samples = [0, 1, 2]
    probability = math.nextafter(1 / 3, 1)
    assert probability * 3 == 1  # Multiplication erases the adjacent float.
    assert diagnostics.empirical_quantile(samples, probability) == 1
    assert diagnostics.empirical_quantile(samples, 1 / 3) == 0
    assert diagnostics.empirical_quantile(samples, math.nextafter(1 / 3, 0)) == 0
    assert diagnostics.empirical_quantile(list(range(20)), 0.05) == 0


def test_prepared_empirical_coverage_and_pit_oracles():
    diagnose = diagnostics.prepare_empirical_diagnostics([2, 0, 0, 0])
    result = diagnose(0)
    assert result["pit_lower"] == 0
    assert result["pit_upper"] == 0.75
    assert result["intervals"][0] == {"level": 0.5, "lower": 0, "upper": 0,
                                       "width": 0, "covered": True}
    for row in result["intervals"][1:]:
        assert (row["lower"], row["upper"], row["width"], row["covered"]) == (0, 2, 2, True)
    assert diagnose(1)["pit_lower"] == diagnose(1)["pit_upper"] == 0.75
    assert diagnose(1)["intervals"][0]["covered"] is False
    assert diagnose(2)["pit_lower"] == 0.75
    assert diagnose(2)["pit_upper"] == 1
    assert diagnose(3)["pit_lower"] == diagnose(3)["pit_upper"] == 1
    assert all(row["covered"] is False for row in diagnose(3)["intervals"])


def test_degenerate_and_huge_empirical_atoms_are_not_smoothed_or_capped():
    diagnose = diagnostics.prepare_empirical_diagnostics([0, 0])
    assert all(row["width"] == 0 for row in diagnose(0)["intervals"])
    assert (diagnose(0)["pit_lower"], diagnose(0)["pit_upper"]) == (0, 1)
    assert diagnostics.empirical_quantile([1e308], 0.5) == 1e308
    huge = diagnostics.prepare_empirical_diagnostics([0, 1e308])(1e308)
    assert all(row["width"] == 1e308 for row in huge["intervals"])


@pytest.mark.parametrize("weights", [{0: 1}, {80: 1}])
@pytest.mark.parametrize("function,args", [
    (diagnostics.quantile, (0,)), (diagnostics.quantile, (1,)),
    (diagnostics.interval_diagnostics, (0,)), (diagnostics.pit_bracket, (0,)),
])
def test_unresolved_tails_refuse_without_partial_diagnostics(function, args, weights):
    _refuses(function, _law(weights, tail=None), *args,
             message="unresolved magnitude distribution tail")


@pytest.mark.parametrize("value", [True, False, None, "0", -1, float("nan"),
                                  float("inf"), -float("inf"), 10 ** 1000])
def test_invalid_numbers_refuse(value):
    document = _law()
    for function in (diagnostics.quantile, diagnostics.interval_diagnostics, diagnostics.pit_bracket):
        _refuses(function, document, value, message="invalid magnitude diagnostic input")
    _refuses(diagnostics.empirical_quantile, [0], value)
    _refuses(diagnostics.prepare_empirical_diagnostics([0]), value)
    _refuses(diagnostics.prepare_empirical_diagnostics, [value])


@pytest.mark.parametrize("samples", [None, [], (), (0,), "0", {}, {0}])
def test_invalid_sample_containers_refuse(samples):
    _refuses(diagnostics.empirical_quantile, samples, 0.5)
    _refuses(diagnostics.prepare_empirical_diagnostics, samples)


def test_invalid_probabilities_and_histogram_schema_refuse():
    for probability in (1.01, 2):
        _refuses(diagnostics.quantile, _law(), probability)
        _refuses(diagnostics.empirical_quantile, [0], probability)
    document = _law()
    document["probabilities"][0] = 0.9
    for function, argument in ((diagnostics.quantile, 0),
                               (diagnostics.interval_diagnostics, 0), (diagnostics.pit_bracket, 0)):
        _refuses(function, document, argument)


def test_plain_input_types_and_required_arguments():
    class Number(float):
        pass

    class Samples(list):
        pass

    _refuses(diagnostics.quantile, _law(), Number(0.5))
    _refuses(diagnostics.prepare_empirical_diagnostics, Samples([0]))
    _refuses(diagnostics.prepare_empirical_diagnostics, [Number(0)])
    for name in diagnostics.__all__:
        with pytest.raises(TypeError):
            getattr(diagnostics, name)()


def test_document_and_prepared_input_output_mutation_isolation():
    document = _law({0: 0.25, 80: 0.75}, (4, 0, 0))
    before = copy.deepcopy(document)
    diagnostics.quantile(document, 0.5)
    diagnostics.pit_bracket(document, 20)
    diagnostics.interval_diagnostics(document, 20)[0]["lower"] = -1
    assert document == before
    samples = [2, 0, 0, 0]
    diagnose = diagnostics.prepare_empirical_diagnostics(samples)
    original = diagnose(0)
    samples[:] = [-1, float("nan")]
    changed = diagnose(0)
    changed["intervals"][0]["lower"] = -1
    changed["intervals"].clear()
    changed["pit_lower"] = -1
    assert diagnose(0) == original
    assert diagnostics.prepare_empirical_diagnostics([4, 4])(0)["pit_upper"] == 0


def test_histogram_intervals_reconstruct_only_once(monkeypatch):
    original = diagnostics.components
    calls = []

    def counted(document):
        calls.append(1)
        return original(document)

    monkeypatch.setattr(diagnostics, "components", counted)
    diagnostics.interval_diagnostics(_law(), 0)
    assert calls == [1]


def test_prepared_calls_do_not_resort_revalidate_or_randomize(monkeypatch):
    diagnose = diagnostics.prepare_empirical_diagnostics(list(range(1000)))

    def forbidden(*args, **kwargs):
        pytest.fail("prepared diagnostic unexpectedly touched input or an external operation")

    monkeypatch.setattr(diagnostics, "_samples", forbidden)
    monkeypatch.setattr("builtins.sorted", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("random.random", forbidden)
    for observation in (0, 499, 999, 1000):
        assert diagnose(observation) == diagnose(observation)


def test_prepared_large_empirical_law_has_linear_memory_and_bounded_search(monkeypatch):
    samples = list(range(20000, 0, -1))
    tracemalloc.start()
    try:
        diagnose = diagnostics.prepare_empirical_diagnostics(samples)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8_000_000  # A 20k x 20k pair matrix cannot fit this bound.
    assert samples[0] == 20000
    calls = []
    original = diagnostics.bisect_left

    def counted(values, value, **kwargs):
        calls.append(len(values))
        return original(values, value, **kwargs)

    monkeypatch.setattr(diagnostics, "bisect_left", counted)
    result = diagnose(10000)
    assert calls == [20000]
    assert result["pit_lower"] == 9999 / 20000
    assert result["pit_upper"] == 0.5
    assert [row["width"] for row in result["intervals"]] == [10000, 16000, 18000, 19000]


def test_mass_roundoff_and_many_tail_atoms_preserve_inverse_boundaries():
    document = _law({0: 0.5 - 5e-13, 80: 0.5}, range(1000))
    probability = (0.5 - 5e-13) / (1 - 5e-13)
    assert diagnostics.quantile(document, probability) == 0.25
    assert diagnostics.quantile(document, math.nextafter(probability, 1)) == 20
    assert diagnostics.quantile(document, 1) == 1019
    assert diagnostics.pit_bracket(document, 1019)[1] == 1
