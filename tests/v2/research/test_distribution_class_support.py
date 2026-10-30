"""Synthetic class-count/mapping oracles, without fitting or source admission."""
import copy
import itertools
import math
import socket
import subprocess

import pytest

from engine.v2.data.errors import DataError
from engine.v2.ops.errors import OpsError
from experiments import distribution_class_support as support


def _counts(entries):
    result = [0] * 81
    for label, count in entries.items():
        result[label] = count
    return result


def _refuses(function, *args, code="INVALID_EXPERIMENT_SPEC", **kwargs):
    with pytest.raises(OpsError) as error:
        function(*args, **kwargs)
    assert error.value.code == code
    assert error.value.problem.retryable is False
    assert error.value.problem.details == {}
    return error.value.problem.message


@pytest.mark.parametrize("entries,labels,trees", [
    ({0: 5, 80: 7}, [0, 80], 1), ({6: 1, 9: 2}, [6, 9], 1),
    ({0: 5, 9: 2, 80: 7}, [0, 9, 80], 3),
    ({index: 1 for index in range(81)}, list(range(81)), 81),
])
def test_training_count_and_tree_multiplicity_oracles(entries, labels, trees):
    counts = _counts(entries)
    assert support.training_support(counts) == {
        "bin_counts": counts, "observed_class_labels": labels,
        "sample_count": sum(entries.values()), "trees_per_iteration": trees}


def test_count_outputs_are_detached_and_integer_counts_are_not_converted_to_float():
    counts = _counts({0: 10 ** 400, 80: 1})
    first = support.training_support(counts)
    second = support.training_support(counts)
    counts[0] = 0
    first["bin_counts"][80] = 99
    first["observed_class_labels"].clear()
    assert second["bin_counts"] == _counts({0: 10 ** 400, 80: 1})
    assert second["sample_count"] == 10 ** 400 + 1
    assert second["observed_class_labels"] == [0, 80]


@pytest.mark.parametrize("counts", [[0] * 81, _counts({0: 1}), _counts({80: 10})])
def test_zero_or_one_observed_class_is_a_typed_candidate_failure(counts):
    message = _refuses(support.training_support, counts, code="EXPERIMENT_VARIANT_FAILED")
    assert message == "distribution candidate requires two observed classes"


class _Int(int):
    pass


class _List(list):
    pass


@pytest.mark.parametrize("counts", [None, (), {}, "private-input", [1] * 80, [1] * 82,
    _List([1] * 81), [True] * 81, [False] * 81, [1.0] * 81, [-1] * 81,
    [math.inf] * 81, [math.nan] * 81, ["1"] * 81, [_Int(1)] * 81])
def test_invalid_counts_refuse_with_safe_fixed_message(counts):
    assert _refuses(support.training_support, counts) == "invalid distribution class support"


def test_all_class_permutations_preserve_alignment_and_absent_zero_mass():
    pairs = [(0, 0.25), (7, 0.5), (80, 0.25)]
    expected = [0.0] * 81
    expected[0], expected[7], expected[80] = 0.25, 0.5, 0.25
    for permutation in itertools.permutations(pairs):
        labels, probabilities = map(list, zip(*permutation))
        result = support.class_histogram(labels, probabilities)
        assert result["probabilities"] == expected
        assert result["tail"] == {"policy": "unresolved", "sample_count": 0, "excesses": []}


def test_binary_all_classes_and_zero_observed_class_probabilities():
    result = support.class_histogram([80, 0], [1, 0], tail_excesses=[0, 2])
    assert result["probabilities"] == [0] * 80 + [1]
    labels = list(reversed(range(81)))
    probabilities = [1 / 81] * 81
    assert support.class_histogram(labels, probabilities)["probabilities"] == probabilities


def test_mapping_returns_detached_samples_and_preserves_admitted_mass_roundoff():
    labels, probabilities, tail = [80, 0], [0.5, 0.5 - 5e-13], [0, 2, 2]
    before = copy.deepcopy((labels, probabilities, tail))
    result = support.class_histogram(labels, probabilities, tail_excesses=tail)
    assert (labels, probabilities, tail) == before
    assert result["probabilities"][0] == 0.5 - 5e-13
    assert sum(result["probabilities"]) != 1
    labels.clear()
    probabilities.clear()
    tail.clear()
    assert result["probabilities"][80] == 0.5
    assert result["tail"] == {"policy": "empirical_excess", "sample_count": 3, "excesses": [0, 2, 2]}


@pytest.mark.parametrize("labels,probabilities", [
    (None, []), ([], []), ([0], [1]), ([0, 0], [0.5, 0.5]),
    ([0, 81], [0.5, 0.5]), ([-1, 80], [0.5, 0.5]), ([False, 80], [0.5, 0.5]),
    ([0, True], [0.5, 0.5]), ([0.0, 80], [0.5, 0.5]), ([math.nan, 80], [0.5, 0.5]),
    ([math.inf, 80], [0.5, 0.5]), ([{}, 80], [0.5, 0.5]), ([_Int(0), 80], [0.5, 0.5]),
    ((0, 80), [0.5, 0.5]), (_List([0, 80]), [0.5, 0.5]),
    ([0, 80], None), ([0, 80], (0.5, 0.5)), ([0, 80], _List([0.5, 0.5])),
    ([0, 80], [1]), (list(range(82)), [1 / 82] * 82),
])
def test_structural_mapping_errors_refuse_before_native_histogram(labels, probabilities, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid mapping reached numerical validator")

    monkeypatch.setattr(support, "histogram", forbidden)
    assert _refuses(support.class_histogram, labels, probabilities) == "invalid distribution class mapping"


@pytest.mark.parametrize("probabilities,tail", [
    ([True, 0], None), ([math.nan, 0.5], None), ([math.inf, 0], None),
    ([-0.1, 1.1], None), ([0.25, 0.25], None), (["secret", 0], None),
    ([1, 0], []), ([1, 0], [-1]), ([1, 0], [math.inf]), ([1, 0], (0,)),
])
def test_native_numerical_and_tail_refusals_are_not_reclassified(probabilities, tail):
    with pytest.raises(DataError) as error:
        support.class_histogram([0, 80], probabilities, tail_excesses=tail)
    assert error.value.code == "CONTRACT_MISMATCH"
    assert error.value.problem.message == "invalid magnitude distribution"
    assert error.value.problem.retryable is False


def test_native_error_object_propagates_unchanged(monkeypatch):
    from engine.v2.data.errors import fail

    sentinel = fail("CONTRACT_MISMATCH", "safe synthetic refusal")
    def refuse(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(support, "histogram", refuse)
    with pytest.raises(DataError) as error:
        support.class_histogram([0, 80], [0.5, 0.5])
    assert error.value is sentinel


def test_public_arity_and_source_free_operations(monkeypatch):
    assert support.__all__ == ["training_support", "class_histogram"]
    for function in (support.training_support, support.class_histogram):
        with pytest.raises(TypeError):
            function()
    def forbidden(*args, **kwargs):
        raise AssertionError("source-free class policy attempted external I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert support.training_support(_counts({0: 1, 80: 1}))["sample_count"] == 2
    assert support.class_histogram([80, 0], [0.5, 0.5])["probabilities"][80] == 0.5
