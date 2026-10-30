"""Synthetic shared horizon orchestration and unchanged SIZE failure policy."""

import numpy as np
import pandas as pd
import pytest

from engine.models import no_fit as legacy_guard
from engine.v2.foundation import content_hash
from engine.v2.models import no_fit
from engine.v2.models.regression_candidate import regression_candidate_recipe
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research import horizon_fitted
from engine.v2.research.horizon_fit_inputs import prepare_horizon_inputs
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_fitted import fitted_forecasts as magnitude_forecasts

FEATURES = dict(columns=["x"], feature_dictionary_hash="sha256:" + "a" * 64)
FAMILIES = [("ridge", {"alpha": 1.}), ("hgb", {"max_leaf_nodes": 7}),
            ("elasticnet", {"alpha": .1, "l1_ratio": .75})]


def indexed():
    return {(event, horizon): dict(event_id=event, context_id="synthetic", horizon_sessions=horizon,
        x=value, forbidden_evaluation_label=object()) for horizon in (5, 3, 1)
        for event, value in (("a", 0.), ("b", 1.), ("evaluation", 0.))}


def run(*, source=None, recipe=None, labels=(-140., -140.), progress=None, diagnostics=None, **changes):
    policy = recipe if recipe is not None else regression_candidate_recipe(*FAMILIES[0], target_domain="signed", **FEATURES)
    args = dict(target_column="crush_pct_iv30", target_domain="signed", route="regression_candidate", recipe=policy,
                progress=progress, diagnostics=diagnostics)
    return horizon_fitted.fitted_forecasts(source if source is not None else indexed(), ["a", "b"], labels,
        ["evaluation"], [5, 3, 1], ["x"], **{**args, **changes})


def legacy(*, progress=None, strict=False):
    return magnitude_forecasts(indexed(), pd.DataFrame({"event_id": ["a", "b"], "absolute_move_pct": [1., 2.]}),
        {"evaluation_event_ids": ["evaluation"]}, {"horizons": [5, 3, 1]},
        {"schema_version": "size_candidate_execution.v1", "features": ["x"],
         "fit_recipe": candidate_recipe(*FAMILIES[0], **FEATURES)}, progress=progress, strict_failures=strict)


@pytest.mark.parametrize("family,parameters", FAMILIES)
@pytest.mark.parametrize("value", [-140., 25.])
def test_signed_fits_preserve_both_signs_and_horizon_order(family, parameters, value):
    recipe = regression_candidate_recipe(family, parameters, target_domain="signed", **FEATURES)
    progress, diagnostics = [], []
    predictions, evidence = run(recipe=recipe, labels=(value, value), diagnostics=diagnostics,
        progress=lambda state, fit: progress.append((state, fit["horizon_sessions"])))
    assert list(predictions.values()) == pytest.approx([value] * 3)
    assert [fit["horizon_sessions"] for fit in evidence] == [5, 3, 1]
    assert progress == [(state, h) for h in (5, 3, 1) for state in ("started", "completed")]
    assert [d["horizon_sessions"] for d in diagnostics] == [5, 3, 1]
    assert all("recipe_id" in d and "predictions" not in d for d in diagnostics)


def test_estimator_mutation_cannot_change_future_inputs_or_original_hashes(monkeypatch):
    source = indexed()
    original = {key: {k: v for k, v in row.items() if k != "forbidden_evaluation_label"} for key, row in source.items()}
    labels = np.array([-140., -140.])
    def mutate(route, recipe, train, y, prediction, *, columns):
        assert y.tolist() == [-140., -140.]
        train[:] = 999.
        y[:] = 999.
        prediction[:] = 999.
        return {"predictions": [-140.]}
    monkeypatch.setattr(horizon_fitted, "fit_execution", mutate)
    _, evidence = run(source=source, labels=labels)
    assert labels.tolist() == [-140., -140.]
    assert {key: {k: v for k, v in row.items() if k != "forbidden_evaluation_label"} for key, row in source.items()} == original
    for h, fit in zip((5, 3, 1), evidence):
        assert fit == prepare_horizon_inputs(source, ["a", "b"], ["evaluation"], h, ["x"], labels,
                                            target_column="crush_pct_iv30")[2]


@pytest.mark.parametrize("guard", [no_fit, legacy_guard])
def test_guards_precede_all_core_input_inspection(guard):
    class Poison:
        def __getattribute__(self, name):
            pytest.fail("input inspected before guard")
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        horizon_fitted.fitted_forecasts(*([Poison()] * 6), target_column=Poison(),
            target_domain=Poison(), route=Poison(), recipe=Poison())


@pytest.mark.parametrize("changes", [dict(target_domain="other"), dict(route="unknown"),
    dict(route="magnitude"), dict(route="magnitude_candidate"), dict(target_domain="nonnegative")])
def test_domain_and_route_drift_refuse_before_feature_access(changes):
    class Poison(dict):
        def __getitem__(self, key):
            pytest.fail("feature access preceded domain/route check")
    with pytest.raises(ValueError):
        run(source=Poison(), **changes)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("error", [ValueError("value"), ArithmeticError("arithmetic"), RuntimeError("runtime"),
    MemoryError("memory"), OSError("os"), ImportError("import"), fail("RESOURCE_UNAVAILABLE", "synthetic"),
    no_fit.RuntimeFitForbidden("native"), legacy_guard.RuntimeFitForbidden("legacy")])
def test_size_compatibility_preserves_exact_error_policy(monkeypatch, strict, error):
    calls, progress = [], []
    def broken(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(horizon_fitted, "fit_execution", broken)
    propagate = isinstance(error, (no_fit.RuntimeFitForbidden, legacy_guard.RuntimeFitForbidden)) or (
        strict and not isinstance(error, (ValueError, ArithmeticError)))
    with pytest.raises(type(error) if propagate else OpsError) as caught:
        legacy(strict=strict, progress=lambda state, fit: progress.append(state))
    if propagate:
        assert caught.value is error
    else:
        assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"
        assert caught.value.problem.message == "fitted magnitude estimator failed"
    assert calls == [1] and progress == ["started"]


@pytest.mark.parametrize("phase", ["started", "completed"])
def test_progress_failure_is_not_mislabeled_as_fit_failure(phase):
    error = ValueError("synthetic callback failure")
    def progress(state, fit):
        if state == phase:
            raise error
    with pytest.raises(ValueError) as caught:
        legacy(progress=progress)
    assert caught.value is error


@pytest.mark.parametrize("values", [[], [True], [float("nan")], [float("inf")]])
def test_bad_forecasts_fail_without_partial_return(monkeypatch, values):
    monkeypatch.setattr(horizon_fitted, "fit_execution", lambda *a, **k: {"predictions": values})
    with pytest.raises(horizon_fitted.HorizonFitFailure) as caught:
        run()
    assert isinstance(caught.value.__cause__, ValueError)


def test_one_shot_ids_columns_and_horizons_are_materialized_once():
    source = indexed()
    recipe = regression_candidate_recipe(*FAMILIES[0], target_domain="signed", **FEATURES)
    predictions, evidence = horizon_fitted.fitted_forecasts(source, iter(["a", "b"]), [-140., -140.],
        iter(["evaluation"]), iter([5, 3, 1]), iter(["x"]), target_column="crush_pct_iv30",
        target_domain="signed", route="regression_candidate", recipe=recipe)
    assert (predictions, evidence) == run(source=source, recipe=recipe)
    assert all(fit["training_event_ids"] == ["a", "b"] and fit["evaluation_event_ids"] == ["evaluation"] for fit in evidence)

    for h, fit in zip((5, 3, 1), evidence):
        assert fit["training_matrix_id"] == content_hash({"columns": ["x"],
            "row_keys": [[event, "synthetic", h] for event in ("a", "b")],
            "values": [[0.], [1.]], "crush_pct_iv30": [-140., -140.]})


@pytest.mark.parametrize("horizons", [[], [3], [5, 3, 1]])
def test_zero_one_and_multiple_horizon_callback_counts(horizons):
    observed = []
    predictions, evidence = horizon_fitted.fitted_forecasts(indexed(), ["a", "b"], [-140., -140.],
        ["evaluation"], horizons, ["x"], target_column="crush_pct_iv30", target_domain="signed",
        route="regression_candidate", recipe=regression_candidate_recipe(*FAMILIES[0], target_domain="signed", **FEATURES),
        progress=lambda state, fit: observed.append((state, fit["horizon_sessions"])))
    assert observed == [(state, h) for h in horizons for state in ("started", "completed")]
    assert len(predictions) == len(evidence) == len(horizons)


def test_late_fit_failure_has_only_provisional_progress_and_diagnostics(monkeypatch):
    calls, observed, diagnostics = [], [], []
    def fitted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("second horizon failed")
        return {"predictions": [-140.], "synthetic": "diagnostic"}
    monkeypatch.setattr(horizon_fitted, "fit_execution", fitted)
    with pytest.raises(horizon_fitted.HorizonFitFailure):
        run(diagnostics=diagnostics, progress=lambda state, fit: observed.append((state, fit["horizon_sessions"])))
    assert observed == [("started", 5), ("completed", 5), ("started", 3)]
    assert diagnostics == [{"horizon_sessions": 5, "synthetic": "diagnostic"}]


@pytest.mark.parametrize("guard", [legacy_guard, no_fit])
def test_size_guard_now_precedes_matrix_preparation_and_progress(guard, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("active guard permitted matrix preparation or progress")
    monkeypatch.setattr(horizon_fitted, "prepare_horizon_inputs", forbidden)
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        legacy(progress=forbidden)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("candidate", [False, True])
def test_size_empty_horizons_do_not_require_or_execute_a_recipe(monkeypatch, strict, candidate):
    def forbidden(*args, **kwargs):
        pytest.fail("empty horizon plan prepared, fitted or reported progress")
    monkeypatch.setattr(horizon_fitted, "prepare_horizon_inputs", forbidden)
    monkeypatch.setattr(horizon_fitted, "fit_execution", forbidden)
    execution = {"features": []}
    if candidate:
        execution["schema_version"] = "size_candidate_execution.v1"
    diagnostics = []
    assert magnitude_forecasts({}, pd.DataFrame({"event_id": ["train"], "absolute_move_pct": [2.]}),
        {"evaluation_event_ids": []}, {"horizons": []}, execution, progress=forbidden,
        strict_failures=strict, diagnostics=diagnostics) == ({}, [])
    assert diagnostics == []


@pytest.mark.parametrize("guard", [legacy_guard, no_fit])
def test_size_empty_horizons_still_honor_both_no_fit_guards(guard):
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        magnitude_forecasts({}, pd.DataFrame({"event_id": ["train"], "absolute_move_pct": [2.]}),
            {"evaluation_event_ids": []}, {"horizons": []}, {"features": []})


@pytest.mark.parametrize("phase", ["started", "completed"])
@pytest.mark.parametrize("deferred", [False, True])
def test_callback_mutation_cannot_change_returned_evidence(phase, deferred):
    expected = []
    for horizon in (5, 3, 1):
        training = {"columns": ["x"], "row_keys": [[e, "synthetic", horizon] for e in ("a", "b")],
                    "values": [[0.], [1.]], "crush_pct_iv30": [-140., -140.]}
        prediction = {"columns": ["x"], "row_keys": [["evaluation", "synthetic", horizon]], "values": [[0.]]}
        expected.append({"horizon_sessions": horizon, "training_event_ids": ["a", "b"],
            "evaluation_event_ids": ["evaluation"], "training_matrix_id": content_hash(training),
            "prediction_matrix_id": content_hash(prediction)})
    retained, observed = [], []
    def mutate(fit):
        fit["training_matrix_id"] = "changed"
        fit["prediction_matrix_id"] = "changed"
        fit["training_event_ids"].append("injected")
        fit["evaluation_event_ids"].clear()
    def progress(state, fit):
        assert fit == expected[len(observed) // 2]
        observed.append(state)
        if state == phase:
            retained.append(fit)
            if not deferred:
                mutate(fit)
    predictions, evidence = run(progress=progress)
    for fit in retained:
        mutate(fit)
    assert observed == [s for _ in (5, 3, 1) for s in ("started", "completed")]
    assert predictions == {("evaluation", h): -140. for h in (5, 3, 1)}
    assert evidence == expected
