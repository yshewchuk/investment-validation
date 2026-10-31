"""Tiny authorized synthetic fits and isolated numerical boundary oracles."""
import copy
import json
import socket
import subprocess

import numpy as np
import pytest
import sklearn.ensemble
from threadpoolctl import threadpool_info

from engine.models import no_fit as legacy_guard
from engine.v2.models import no_fit as native_guard
from engine.v2.models.training import magnitude_ridge_hgb as point_owner
from engine.v2.ops.errors import OpsError, fail
from experiments import distribution_candidate_fit as fitting
from experiments.distribution_candidate_recipe import candidate_recipe


def _inputs(labels=(0, 0.5, 0, 0.5), *, feature_set="H"):
    recipe = candidate_recipe(feature_set)
    columns = recipe["point_recipe"]["features"]
    return recipe, np.zeros((len(labels), len(columns))), np.array(labels), np.ones((2, len(columns))), columns


def _fit(inputs):
    recipe, train, labels, prediction, columns = inputs
    return fitting.fit_candidate(recipe, train, labels, prediction, columns=columns)


def _refuses(inputs, code="INVALID_EXPERIMENT_SPEC"):
    with pytest.raises(OpsError) as error:
        _fit(inputs)
    assert error.value.code == code
    assert error.value.problem.retryable is False
    assert error.value.problem.details == {}
    return error.value


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError("numeric input inspected before refusal")

    def __getitem__(self, key):
        raise AssertionError("recipe inspected before refusal")

    def __iter__(self):
        raise AssertionError("columns inspected before refusal")


@pytest.fixture
def observed(monkeypatch):
    calls = {"constructed": [], "fit": [], "predict": [], "point": [], "behavior": {}}
    behavior = calls["behavior"]

    class Classifier:
        def __init__(self, **params):
            calls["constructed"].append(params)

        def fit(self, train, labels):
            calls["fit"].append((train.copy(), labels.copy(), threadpool_info()))
            if "classifier_error" in behavior:
                raise behavior["classifier_error"]
            self.classes_ = np.array(behavior.get("classes", sorted(set(labels))))
            self.n_iter_ = behavior.get("iterations", 100)
            self.n_trees_per_iteration_ = behavior.get("trees", 1 if len(self.classes_) == 2 else len(self.classes_))
            return self

        def predict_proba(self, prediction):
            calls["predict"].append(prediction.copy())
            return behavior.get("probabilities", np.full((len(prediction), len(self.classes_)), 1 / len(self.classes_)))

    def point(recipe, train, labels, prediction, *, columns):
        calls["point"].append((copy.deepcopy(recipe), train.copy(), labels.copy(), prediction.copy(), list(columns)))
        if "point_error" in behavior:
            raise behavior["point_error"]
        return behavior.get("point_result", {"predictions": np.full(len(prediction), 3.0)})

    monkeypatch.setattr(sklearn.ensemble, "HistGradientBoostingClassifier", Classifier)
    monkeypatch.setattr(point_owner, "fit_ridge_hgb_predictions", point)
    return calls


@pytest.mark.parametrize("guard", [legacy_guard, native_guard])
def test_both_guards_precede_every_input_inspection_and_estimator(guard, observed):
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden, match="distribution_candidate_fit"):
        fitting.fit_candidate(Poison(), Poison(), Poison(), Poison(), columns=Poison())
    assert not observed["constructed"] and not observed["point"]


def test_declared_parameters_shared_preprocessing_and_actual_diagnostics(observed):
    inputs = list(_inputs((0, 0.25, 20, 22)))
    recipe, train, labels, prediction, columns = inputs
    train[:, 1] = np.nan
    train[0, 2] = np.nan
    prediction[:, 1] = [123, np.nan]
    prediction[0, 2] = np.nan
    before = copy.deepcopy(inputs)
    observed["behavior"].update(classes=[80, 0, 1], probabilities=[[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]])
    result = _fit(inputs)
    assert set(result) == {"schema_version", "recipe_id", "probabilities", "tail", "point_predictions", "diagnostics"}
    assert result["schema_version"] == "distribution_candidate_fit.v1"
    assert result["recipe_id"] == recipe["recipe_id"]
    assert observed["constructed"] == [recipe["classifier"]["params"]]
    assert type(observed["constructed"][0]["max_features"]) is float
    fitted, binned, threads = observed["fit"][0]
    assert binned.tolist() == [0, 1, 80, 80]
    assert np.all(fitted[:, 1] == 0) and np.isnan(fitted[0, 2])
    assert np.all(observed["predict"][0][:, 1] == 0)
    assert np.isnan(observed["predict"][0][0, 2])
    assert all(pool["num_threads"] == 1 for pool in threads)
    point_recipe, point_train, point_labels, point_prediction, point_columns = observed["point"][0]
    assert point_recipe == recipe["point_recipe"] and point_columns == columns
    np.testing.assert_array_equal(point_train, fitted)
    np.testing.assert_array_equal(point_labels, labels)
    np.testing.assert_array_equal(point_prediction, observed["predict"][0])
    assert result["probabilities"][0] == [0.25, 0.25] + [0.0] * 78 + [0.5]
    assert result["probabilities"][1] == [0.5, 0.25] + [0.0] * 78 + [0.25]
    assert result["tail"] == {"policy": "empirical_excess", "sample_count": 2, "excesses": [0, 2]}
    assert result["point_predictions"] == [3.0, 3.0]
    diagnostics = result["diagnostics"]
    assert diagnostics["classifier"] == {"class_labels": [80, 0, 1], "iterations": 100,
        "trees_per_iteration": 3, "total_trees": 300}
    assert diagnostics["point"] == {"iterations_declared": 100, "trees_per_iteration_declared": 1}
    assert diagnostics["all_null_training_columns"] == [columns[1]]
    assert diagnostics["training_rows"] == 4 and diagnostics["prediction_rows"] == 2
    assert diagnostics["training_support"]["bin_counts"] == [1, 1] + [0] * 78 + [2]
    assert recipe == before[0] and columns == before[4]
    for index in (1, 2, 3):
        np.testing.assert_array_equal(inputs[index], before[index])
    json.dumps(result, allow_nan=False)


def test_train_only_tail_is_shared_once_and_detached(observed, monkeypatch):
    inputs = list(_inputs((0, 20, 22, 22)))
    inputs[3] = np.full((3, len(inputs[4])), 999)
    original, tails = fitting.class_histogram, []
    def inspect(labels, probabilities, *, tail_excesses=None):
        tails.append(tail_excesses)
        return original(labels, probabilities, tail_excesses=tail_excesses)

    monkeypatch.setattr(fitting, "class_histogram", inspect)
    result = _fit(inputs)
    assert tails == [[0.0, 2.0, 2.0], None, None]
    assert result["tail"]["excesses"] == [0, 2, 2]
    inputs[2][:] = 100
    tails[0].clear()
    result["probabilities"][0][0] = 99
    assert result["tail"]["excesses"] == [0, 2, 2]
    assert result["probabilities"][1][0] == 0.5


@pytest.mark.parametrize("labels", [(0, 0), (0.01, 0.02), (20, 22)])
def test_degenerate_binned_support_refuses_before_either_estimator(labels, observed):
    _refuses(_inputs(labels), "EXPERIMENT_VARIANT_FAILED")
    assert not observed["constructed"] and not observed["point"]


@pytest.mark.parametrize("feature_set", ["H", "HM", "HMO"])
def test_all_declared_feature_subsets_use_the_exact_ordered_width(feature_set, observed):
    inputs = _inputs(feature_set=feature_set)
    result = _fit(inputs)
    assert observed["fit"][0][0].shape[1] == len(inputs[4])
    assert observed["point"][0][-1] == inputs[4]
    assert len(result["probabilities"]) == len(inputs[3])


@pytest.mark.parametrize("kind", ["recipe", "columns", "train-width", "predict-width", "train-empty",
    "predict-empty", "train-flat", "predict-flat", "train-inf", "predict-inf", "labels-length",
    "labels-nested", "labels-inf", "labels-nan", "labels-negative", "conversion", "overflow"])
def test_invalid_inputs_are_typed_and_never_fit(kind, observed):
    inputs = list(_inputs())
    if kind == "recipe":
        inputs[0]["classifier"]["params"]["early_stopping"] = True
        inputs[1] = Poison()
    elif kind == "columns":
        inputs[4] = list(reversed(inputs[4]))
        inputs[1] = Poison()
    elif kind in ("train-width", "predict-width", "train-empty", "predict-empty", "train-flat", "predict-flat"):
        index = 1 if kind.startswith("train") else 3
        shape = (2, 10) if kind.endswith("width") else ((0, 11) if kind.endswith("empty") else (11,))
        inputs[index] = np.zeros(shape)
    elif kind in ("train-inf", "predict-inf"):
        inputs[1 if kind == "train-inf" else 3][0, 0] = np.inf
    elif kind.startswith("labels"):
        inputs[2] = {"labels-length": [0], "labels-nested": [[0], [0.5], [0], [0.5]],
            "labels-inf": [0, 0.5, np.inf, 0], "labels-nan": [0, 0.5, np.nan, 0],
            "labels-negative": [0, 0.5, -1, 0]}[kind]
    else:
        inputs[1] = [["sensitive-value" if kind == "conversion" else 10 ** 1000] * 11] * 4
    error = _refuses(inputs)
    assert "sensitive-value" not in str(error)
    assert not observed["constructed"] and not observed["point"]


@pytest.mark.parametrize("index", [1, 2, 3])
def test_complex_inputs_refuse_instead_of_silently_dropping_imaginary_parts(index, observed):
    inputs = list(_inputs())
    inputs[index] = inputs[index].astype(complex) + 1j
    _refuses(inputs)
    assert not observed["constructed"] and not observed["point"]


@pytest.mark.parametrize("key,value", [("classes", [0, 3]), ("classes", [0, 0]),
    ("classes", [0.0, 2.0]), ("classes", [[0, 2]]), ("classes", [True, False]),
    ("iterations", 99), ("iterations", True), ("trees", 2), ("trees", 1.0),
    ("probabilities", [[0.5, 0.5]]), ("probabilities", [[np.nan, 0.5]] * 2),
    ("probabilities", [[np.inf, 0]] * 2), ("probabilities", [[-0.1, 1.1]] * 2),
    ("probabilities", [[0.2, 0.2]] * 2), ("probabilities", [["bad", 1]] * 2),
    ("probabilities", np.full((2, 2), 0.5 + 1j)),
    ("point_result", None), ("point_result", {}), ("point_result", {"predictions": [1]}),
    ("point_result", {"predictions": [np.inf, 1]}), ("point_result", {"predictions": [-1, 1]}),
    ("point_result", {"predictions": np.array([1 + 1j, 2 + 1j])})])
def test_invalid_estimator_results_refuse_without_successful_output(key, value, observed):
    observed["behavior"][key] = value
    _refuses(_inputs(), "EXPERIMENT_VARIANT_FAILED")
    assert len(observed["fit"]) == 1
    assert len(observed["point"]) <= 1


@pytest.mark.parametrize("stage", ["classifier_error", "point_error"])
@pytest.mark.parametrize("error", [ValueError("sensitive-value"), FloatingPointError("sensitive-value")])
def test_numeric_estimator_errors_are_sanitized_without_retry(stage, error, observed):
    observed["behavior"][stage] = error
    result = _refuses(_inputs(), "EXPERIMENT_VARIANT_FAILED")
    assert result.problem.message == "distribution numeric fit failed"
    assert result.__suppress_context__ is True
    assert len(observed["fit"]) == 1 and len(observed["point"]) <= 1


@pytest.mark.parametrize("error", [MemoryError("resource"), OSError("resource"), TypeError("programming"),
    RuntimeError("programming"), fail("RESOURCE_LIMIT_EXCEEDED", "resource"),
    legacy_guard.RuntimeFitForbidden("guard"), native_guard.RuntimeFitForbidden("guard")])
@pytest.mark.parametrize("stage", ["classifier_error", "point_error"])
def test_resources_programming_and_guard_failures_propagate_unchanged(error, stage, observed):
    observed["behavior"][stage] = error
    with pytest.raises(type(error)) as raised:
        _fit(_inputs())
    assert raised.value is error
    assert len(observed["fit"]) == 1 and len(observed["point"]) <= 1


@pytest.mark.parametrize("labels,classes,counts,trees", [
    ((0, 0.5, 0, 0.5), [0, 2], [2, 2], 1),
    ((0, 0.25, 20, 22), [0, 1, 80], [1, 1, 2], 3),
])
def test_tiny_real_fixed_estimators_match_constant_feature_oracles(labels, classes, counts, trees):
    inputs = _inputs(labels)
    result = _fit(inputs)
    for row in result["probabilities"]:
        assert [row[label] for label in classes] == pytest.approx(np.array(counts) / len(labels), abs=1e-12)
        assert all(value == 0 for index, value in enumerate(row) if index not in classes)
    assert result["point_predictions"] == pytest.approx([sum(labels) / len(labels)] * 2)
    assert result["diagnostics"]["classifier"]["iterations"] == 100
    assert result["diagnostics"]["classifier"]["trees_per_iteration"] == trees


def test_tiny_real_all_null_case_is_deterministic_and_no_tail_is_unresolved():
    inputs = list(_inputs())
    inputs[1][:] = np.nan
    inputs[3][:] = np.arange(11)
    first, second = _fit(inputs), _fit(inputs)
    assert first == second
    assert first["tail"] == {"policy": "unresolved", "sample_count": 0, "excesses": []}
    assert first["diagnostics"]["all_null_training_columns"] == inputs[4]
    assert first["diagnostics"]["point"] == {"iterations_declared": 100, "trees_per_iteration_declared": 1}


def test_no_data_or_network_io_and_progress_is_stderr_only(observed, monkeypatch, capsys):
    original_open = open
    def runtime_only(path, *args, **kwargs):
        # threadpoolctl inspects loaded libraries, not experiment/source data.
        if path == "/proc/self/maps":
            return original_open(path, *args, **kwargs)
        raise AssertionError("numeric fit attempted data I/O")

    def forbidden(*args, **kwargs):
        raise AssertionError("numeric fit attempted external I/O")

    monkeypatch.setattr("builtins.open", runtime_only)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = _fit(_inputs())
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "distribution classifier fit/predict" in captured.err
    assert "distribution matched point fit" in captured.err
    assert "distribution probability mapping" in captured.err
    assert result["point_predictions"] == [3, 3]


def test_public_export_and_required_arguments():
    assert fitting.__all__ == ["fit_candidate"]
    with pytest.raises(TypeError):
        fitting.fit_candidate()
