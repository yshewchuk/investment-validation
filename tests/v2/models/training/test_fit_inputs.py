"""Frozen FitInputs transport: guards first, then ordered columns and row counts."""
import dataclasses

import pytest

from engine.models import no_fit as legacy_no_fit
from engine.v2.models import no_fit
from engine.v2.models.training.fit_inputs import FitInputs

RECIPE = {"features": ["signal", "constant"]}


class Poison:
    def __array__(self, *args, **kwargs):
        pytest.fail("array value inspected during construction")


class InspectedRecipe(dict):
    def __getitem__(self, key):
        pytest.fail("recipe inspected before guard")


class LabelsByLength:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return self.rows


def test_stores_caller_objects_by_identity_and_stays_frozen():
    train, labels, prediction = [[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0], [[5.0, 6.0]]
    inputs = FitInputs.from_recipe(
        RECIPE, train, labels, prediction, ["signal", "constant"],
        column_error="columns refused", row_error="rows refused")
    assert inputs.train_features is train
    assert inputs.train_labels is labels
    assert inputs.prediction_features is prediction
    assert inputs.columns == ["signal", "constant"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.columns = []


def test_accepts_tuple_columns_in_exact_recipe_order():
    inputs = FitInputs.from_recipe(
        RECIPE, [[1.0, 2.0]], [1.0], [[3.0, 4.0]], ("signal", "constant"),
        column_error="columns refused", row_error="rows refused")
    assert inputs.columns == ("signal", "constant")


@pytest.mark.parametrize("columns", [
    ["constant", "signal"], ["signal"], ["signal", "constant", "extra"],
    "signal", None, 7])
def test_column_mismatch_or_wrong_type_refuses_before_value_access(columns):
    with pytest.raises(ValueError, match="columns refused"):
        FitInputs.from_recipe(
            RECIPE, Poison(), Poison(), Poison(), columns,
            column_error="columns refused", row_error="rows refused")


def test_row_count_mismatch_refuses_without_conversion():
    with pytest.raises(ValueError, match="rows refused"):
        FitInputs.from_recipe(
            RECIPE, [[1.0, 2.0], [3.0, 4.0]], LabelsByLength(5), [[7.0, 8.0]],
            ["signal", "constant"],
            column_error="columns refused", row_error="rows refused")


@pytest.mark.parametrize("guard", [legacy_no_fit, no_fit])
def test_both_no_fit_guards_precede_recipe_and_array_access(guard):
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden) as caught:
        FitInputs.from_recipe(
            InspectedRecipe(), Poison(), Poison(), Poison(), ["signal", "constant"],
            column_error="columns refused", row_error="rows refused")
    assert type(caught.value) is guard.RuntimeFitForbidden
    assert str(caught.value) == "no-fit guard active: FitInputs.from_recipe"