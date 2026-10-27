"""Synthetic numerical-format oracles; no data, models or source admission."""
import copy
import json
import math
import socket
import subprocess

import pytest

from engine.v2.data.errors import DataError
from engine.v2.research import magnitude_distribution as distribution


def _mass(index=0):
    values = [0.0] * 81
    values[index] = 1.0
    return values


def _document():
    # Independent wire-format oracle, not implementation recipe/constants.
    return {
        "schema_version": "research_magnitude_distribution.v1",
        "target": "absolute_simple_return",
        "unit": "percentage_points",
        "finite_bin_edges": [index * 0.25 for index in range(81)],
        "finite_bin_policy": "uniform",
        "overflow_policy": "uncapped",
        "probabilities": _mass(),
        "tail": {"policy": "unresolved", "sample_count": 0, "excesses": []},
    }


def _refuses(call, *args, **kwargs):
    with pytest.raises(DataError) as error:
        call(*args, **kwargs)
    assert error.value.code == "CONTRACT_MISMATCH"
    assert error.value.problem.message == "invalid magnitude distribution"
    assert error.value.problem.retryable is False


def test_literal_recipe_constructor_and_json_round_trip():
    expected = _document()
    recipe = {key: value for key, value in expected.items() if key not in ("probabilities", "tail")}
    assert distribution.recipe() == recipe
    assert distribution.histogram(_mass()) == expected
    assert distribution.validate_distribution(json.loads(json.dumps(expected, allow_nan=False))) == expected
    assert distribution.__all__ == ["recipe", "histogram", "validate_distribution", "bin_index"]


@pytest.mark.parametrize("index", range(81))
def test_exact_quarter_point_boundaries_and_one_hot_distributions(index):
    boundary = index * 0.25
    assert distribution.bin_index(boundary) == index
    assert distribution.bin_index(math.nextafter(boundary, math.inf)) == index
    if index:
        assert distribution.bin_index(math.nextafter(boundary, -math.inf)) == index - 1
    assert distribution.histogram(_mass(index))["probabilities"] == _mass(index)


@pytest.mark.parametrize("magnitude", [20, 20.25, 100, 1000, 1e308, float.fromhex("0x1.fffffffffffffp1023")])
def test_overflow_not_capped_or_divided_into_infinity(magnitude):
    assert distribution.bin_index(magnitude) == 80


def test_empirical_tail_preserves_zero_duplicates_order_and_count():
    excesses = [0, 80.0, 2.5, 2.5, 1e308]
    result = distribution.histogram(_mass(80), tail_excesses=excesses)
    assert result["tail"] == {"policy": "empirical_excess", "sample_count": 5, "excesses": excesses}
    assert result["probabilities"][-1] == 1
    assert distribution.validate_distribution(result) == result
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_zero_overflow_does_not_erase_or_invent_tail():
    assert distribution.histogram(_mass())["tail"]["policy"] == "unresolved"
    assert distribution.histogram(_mass(), tail_excesses=[0])["tail"] == {
        "policy": "empirical_excess", "sample_count": 1, "excesses": [0]}
    assert distribution.histogram(_mass(80))["tail"]["policy"] == "unresolved"
    _refuses(distribution.histogram, _mass(), tail_excesses=[])


def test_detachment_on_input_output_and_recipe_mutation():
    values, excesses = _mass(), [0, 3]
    first = distribution.histogram(values, tail_excesses=excesses)
    second = distribution.validate_distribution(first)
    values[0], excesses[0] = 0, 99
    assert first["probabilities"][0] == 1
    assert first["tail"]["excesses"] == [0, 3]
    first["probabilities"][0] = 0
    first["tail"]["excesses"][0] = 99
    first["finite_bin_edges"][0] = 100
    assert second["probabilities"][0] == 1
    assert second["tail"]["excesses"] == [0, 3]
    assert second["finite_bin_edges"][0] == 0
    recipe = distribution.recipe()
    recipe["finite_bin_edges"].clear()
    assert len(distribution.recipe()["finite_bin_edges"]) == 81


def test_mass_tolerance_without_smoothing_or_renormalization():
    probabilities = [0.1] * 10 + [0] * 71
    probabilities[0] -= 5e-13
    before = probabilities.copy()
    result = distribution.histogram(probabilities)
    assert result["probabilities"] == before
    assert math.fsum(result["probabilities"]) != 1
    assert result["probabilities"][-1] == 0
    probabilities[0] -= 2e-12
    _refuses(distribution.histogram, probabilities)
    _refuses(distribution.histogram, [0.5, 0.5 + 2e-12] + [0] * 79)


def test_adjacent_representable_normalization_boundary():
    accepted = 1.0 - 1e-12
    refused = math.nextafter(accepted, 0)
    assert abs(accepted - 1) <= 1e-12 < abs(refused - 1)
    assert distribution.histogram([accepted] + [0] * 80)["probabilities"][0] == accepted
    _refuses(distribution.histogram, [refused] + [0] * 80)


class _Float(float):
    pass


class _List(list):
    pass


class _Dict(dict):
    pass


class _String(str):
    pass


_BAD_NUMBERS = [True, False, None, "0", -0.001, float("nan"), float("inf"),
                -float("inf"), complex(1), 10 ** 1000, _Float(0)]


@pytest.mark.parametrize("value", _BAD_NUMBERS)
def test_invalid_numeric_elements_refuse_consistently(value):
    _refuses(distribution.bin_index, value)
    masses = _mass()
    masses[1] = value
    _refuses(distribution.histogram, masses)
    _refuses(distribution.histogram, _mass(), tail_excesses=[value])
    document = _document()
    document["finite_bin_edges"][0] = value
    _refuses(distribution.validate_distribution, document)


@pytest.mark.parametrize("values", [None, (), (1,) * 81, {}, "secret", _List(_mass()),
                                   [], [1], [0] * 81, [1] * 81, [0] * 80, [0] * 82,
                                   [1 + 5e-13] + [0] * 80])
def test_probability_shape_support_and_normalization_refusals(values):
    _refuses(distribution.histogram, values)


@pytest.mark.parametrize("value", [None, True, {}, "secret", (), _List([0]), (0,)])
def test_plain_tail_list_required(value):
    if value is not None:
        _refuses(distribution.histogram, _mass(), tail_excesses=value)
    document = _document()
    document["tail"]["excesses"] = value
    _refuses(distribution.validate_distribution, document)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "research_magnitude_distribution.v2"),
    ("schema_version", _String("research_magnitude_distribution.v1")),
    ("target", "signed_simple_return"), ("unit", "ratio"),
    ("finite_bin_policy", "midpoint"), ("overflow_policy", "capped"),
    ("finite_bin_edges", [index / 4 for index in range(80)]),
    ("finite_bin_edges", tuple(index / 4 for index in range(81))),
    ("finite_bin_edges", [0, 0] + [index / 4 for index in range(2, 81)]),
    ("finite_bin_edges", [index / 4 for index in range(80)] + [21]),
    ("probabilities", [0.5] * 81), ("tail", None),
])
def test_fixed_semantics_cannot_be_overridden(field, value):
    document = _document()
    document[field] = value
    _refuses(distribution.validate_distribution, document)


@pytest.mark.parametrize("field", _document())
def test_missing_required_field_refuses(field):
    document = _document()
    del document[field]
    _refuses(distribution.validate_distribution, document)


def test_unknown_fields_subclasses_and_non_documents_refuse():
    for document in [None, [], (), "secret", _Dict(_document()), {**_document(), "mean": 0}]:
        _refuses(distribution.validate_distribution, document)
    document = _document()
    document[_String("tail")] = document.pop("tail")
    _refuses(distribution.validate_distribution, document)


def test_nonplain_mapping_key_refuses_before_custom_equality():
    class ExplosiveKey(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            raise AssertionError("untrusted key equality was executed")

    document = _document()
    tail = document.pop("tail")
    document[ExplosiveKey("tail")] = tail
    _refuses(distribution.validate_distribution, document)


@pytest.mark.parametrize("change", [
    {"policy": "fitted"}, {"policy": _String("unresolved")}, {"sample_count": True},
    {"sample_count": -1}, {"sample_count": 0.0}, {"sample_count": 1},
    {"excesses": [0]}, {"policy": "empirical_excess"}, {"provenance": "admitted"},
])
def test_tail_schema_policy_and_count_refuse(change):
    document = _document()
    document["tail"].update(change)
    _refuses(distribution.validate_distribution, document)


def test_tail_missing_fields_and_nested_subclass_refuse_without_mutation():
    for field in ("policy", "sample_count", "excesses"):
        document = _document()
        del document["tail"][field]
        before = copy.deepcopy(document)
        _refuses(distribution.validate_distribution, document)
        assert document == before
    document = _document()
    document["tail"] = _Dict(document["tail"])
    _refuses(distribution.validate_distribution, document)


def test_unresolved_policy_refuses_nonempty_tail_even_with_matching_count():
    document = _document()
    document["tail"] = {"policy": "unresolved", "sample_count": 1, "excesses": [0]}
    _refuses(distribution.validate_distribution, document)


@pytest.mark.parametrize("function", [distribution.histogram, distribution.validate_distribution, distribution.bin_index])
def test_omitted_required_argument_is_python_type_error(function):
    with pytest.raises(TypeError):
        function()


def test_public_operations_are_source_free(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("numerical representation attempted external I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    document = distribution.histogram(_mass(80), tail_excesses=[0, 1])
    assert distribution.validate_distribution(document) == document
    assert distribution.recipe()["unit"] == "percentage_points"
    assert distribution.bin_index(12345) == 80
