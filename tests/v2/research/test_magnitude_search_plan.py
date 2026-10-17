"""Native metadata-only search sealing with independent folds, grids and refusals."""
import copy
import errno
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    artifact_reference,
    canonical_json,
    from_document,
    to_document,
)
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.native_registration import (
    validate_magnitude_fitted_execution,
    validate_magnitude_history_execution,
)
from engine.v2.research import magnitude_folds
from engine.v2.research.magnitude_folds import annual_fold
from experiments import magnitude_search_plan as search
from experiments import native_registration as native
from experiments import v2_magnitude_search as cli
from tests.v2.models.training.test_magnitude_estimators import FEATURES, HGB_PARAMS, RIDGE_PARAMS
from tests.v2.research.magnitude_trial_support import (
    CUTOFF,
    HISTORY,
    MONTH,
    ROOT,
    build_history,
    catalog_rows,
    forbid_access,
    forbid_history_rebuild,
    history_event,
    magnitude_dataset,
    magnitude_source,
    outcome_move,
)
from tests.v2.research.test_native_registration import artifact_object_bytes


def _sha(value):
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _request():
    return {"schema_version": "magnitude_search_request.v1",
        "inner_years": {"2024": ["2023"], "2025": ["2023", "2024"]},
        "space": {"history_ridge": {"alpha": [0.5, 2]}, "history_hgb": {
            "learning_rate": [0.05, 0.2], "max_iter": [10], "max_leaf_nodes": [3],
            "min_samples_leaf": [2], "l2_regularization": [0, 1]}},
        "max_candidates": 6, "max_fits": 60}


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("search-native")
    events, moves = magnitude_dataset()
    for event in events:
        if event["event_date"].date().isoformat() in ("2023-01-03", "2025-01-07"):
            event["session"] = "BMO"
    for ticker in ("SYNTH_A", "SYNTH_B"):
        for day in ("2022-12-22", "2022-12-23", "2024-12-27", "2024-12-30"):
            event = history_event(day, ticker=ticker)
            events.append(event)
            moves.append(outcome_move(event, -3))
    value = magnitude_source(root, register=False, build=False, events=events, moves=moves)
    value.build, value.feature_snapshot = build_history(value, training_start="2022-01-01",
        evaluation_years=["2024", "2025"])
    value.registration = native.register_native(value.conn, value.store, value.repository, value.spec,
        code_root=ROOT, as_of_month=MONTH, magnitude_training_start="2022-01-01",
        magnitude_labels_before=CUTOFF, magnitude_feature_snapshot_id=value.feature_snapshot.snapshot_id,
        magnitude_history_start=HISTORY, magnitude_fit_model="history_ridge")
    value.catalog_path = value.conn.execute("PRAGMA database_list").fetchone()[2]
    yield value
    value.conn.close()


def _metadata_only(monkeypatch, source):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    forbid_history_rebuild(monkeypatch)
    for estimator in (Ridge, HistGradientBoostingRegressor):
        monkeypatch.setattr(estimator, "__init__", forbid_access)
    for name in ("scan", "resolve", "resolve_pinned", "fragment_records", "table_contract"):
        monkeypatch.setattr(Repository, name, forbid_access)
    from engine.v2.models.training import magnitude
    monkeypatch.setattr(magnitude, "fit_magnitude_predictions", forbid_access)
    original = ArtifactStore.read_verified
    opened = []

    def read(store, ref, *args, **kwargs):
        assert ref.schema_ref in ("native_experiment_registration.v1.5", "magnitude_search_plan.v1")
        opened.append(ref)
        return original(store, ref, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "read_verified", read)
    return opened


def _ids(source, dates):
    return [event["event_id"] for event in sorted(source.events,
        key=lambda event: (event["event_date"], event["event_id"]))
        if event["event_date"].date().isoformat() in dates]


def _expected_folds(source):
    first_train = _ids(source, ("2022-06-02",))
    outer_2024 = _ids(source, ("2022-06-02", "2022-12-22", "2022-12-23", "2023-01-03", "2023-06-02"))
    inner_2023 = {"year": "2023", "training_labels_before": "2022-12-22",
        "training_candidate_event_ids": first_train,
        "evaluation_event_ids": _ids(source, ("2023-01-03", "2023-06-02"))}
    inner_2024 = _ids(source, ("2024-06-12", "2024-06-17"))
    eval_2024 = _ids(source, ("2024-06-12", "2024-06-17", "2024-12-27", "2024-12-30"))
    return [
        {"outer": {"year": "2024", "training_labels_before": "2024-01-01",
            "training_candidate_event_ids": outer_2024, "evaluation_event_ids": eval_2024},
         "inner": [{**inner_2023, "evaluation_labels_before": "2024-01-01"}]},
        {"outer": {"year": "2025", "training_labels_before": "2024-12-27",
            "training_candidate_event_ids": outer_2024 + inner_2024,
            "evaluation_event_ids": _ids(source, ("2025-01-07",))},
         "inner": [{**inner_2023, "evaluation_labels_before": "2024-12-27"},
            {"year": "2024", "training_labels_before": "2024-01-01",
             "training_candidate_event_ids": outer_2024, "evaluation_event_ids": inner_2024,
             "evaluation_labels_before": "2024-12-27"}]}]


def _assert_folds(source, plan):
    assert plan["folds"] == _expected_folds(source)


def _recipe(kind, params):
    ridge = kind == "history_ridge"
    return {"schema_version": "magnitude_fitted_recipe.v1", "model": kind, "features": FEATURES,
        "estimator": {"class": "Ridge" if ridge else "HistGradientBoostingRegressor", "params": params},
        "preprocessing": {"imputer": {"strategy": "median", "keep_empty_features": True},
            "missing_indicator": {"features": "all", "sparse": False},
            "scaler": {"with_mean": True, "with_std": True}, "output_features": 22} if ridge else {
            "missing_values": "native", "scaling": "none", "all_empty_training_columns": "constant_zero_train_and_predict"},
        "target_transform": "none", "prediction_projection": "clip_zero", "fit_scope": "fold_and_horizon",
        "numerical_threads": 1}


def _assert_candidates(plan):
    recipes = [_recipe("history_ridge", {**RIDGE_PARAMS, "alpha": alpha}) for alpha in (0.5, 2)]
    for rate, penalty in ((0.05, 0), (0.05, 1), (0.2, 0), (0.2, 1)):
        recipes.append(_recipe("history_hgb", {**HGB_PARAMS, "learning_rate": rate,
            "l2_regularization": penalty, "max_iter": 10, "max_leaf_nodes": 3, "min_samples_leaf": 2}))
    expected = sorted([{"candidate_id": _sha(recipe), "recipe": recipe} for recipe in recipes],
                      key=lambda candidate: candidate["candidate_id"])
    assert plan["candidates"] == expected
    assert plan["candidate_count"] == 6 and plan["planned_fit_count"] == 60


def test_public_fold_rule_is_shared_with_native_planning_and_all_source_bindings(source, monkeypatch):
    original, calls = magnitude_folds.annual_fold, []
    def traced(year, events):
        calls.append(year)
        return original(year, events)
    monkeypatch.setattr(magnitude_folds, "annual_fold", traced)
    native_plan = magnitude_folds.plan_magnitude_folds(source.repository, source.snapshot,
        as_of_month=MONTH, evaluation_years=["2024", "2025"], training_start="2022-01-01")
    assert native_plan["folds"] == [fold["outer"] for fold in _expected_folds(source)]
    assert calls == ["2024", "2025"]
    _metadata_only(monkeypatch, source)
    plan = search.seal_search_plan(source.conn, source.store, source.registration, _request())["plan"]
    _assert_folds(source, plan)
    assert calls == ["2024", "2025", "2024", "2025", "2023", "2023", "2024"]
    public = "engine/v2/research/magnitude_folds.py"
    digest = "sha256:" + hashlib.sha256((ROOT / public).read_bytes()).hexdigest()
    assert plan["implementation"]["source_closure"][public] == digest
    assert source.registration.document["source_closure"][public] == digest
    assert source.build["context"]["producer"]["source_closure"][public] == digest


def test_public_fold_rule_preserves_input_order_with_independent_boundary():
    events = [{"event_id": event_id, "event_date": day,
               "forecasts": [] if ceiling is None else [{"source_session_ceiling": ceiling}]}
              for event_id, day, ceiling in [
                  ("eval-z", "2024-12-02", "2024-11-25"),
                  ("train-z", "2023-11-01", None),
                  ("future", "2025-01-01", None),
                  ("eval-a", "2024-01-08", "2023-12-28"),
                  ("boundary", "2023-12-28", None),
                  ("train-a", "2022-01-01", None)]]
    result = annual_fold("2024", events)
    events.clear()
    assert result == {"year": "2024", "training_labels_before": "2023-12-28",
                      "training_candidate_event_ids": ["train-z", "train-a"],
                      "evaluation_event_ids": ["eval-z", "eval-a"]}


@pytest.mark.parametrize("events,message", [
    ([], "every magnitude fold needs evaluation events"),
    ([{"event_id": "evaluation", "event_date": "2024-06-12",
       "forecasts": [{"source_session_ceiling": "2024-06-03"}]}],
     "every magnitude fold needs training candidates"),
])
def test_public_fold_rule_preserves_exact_native_population_refusal(events, message):
    with pytest.raises(DataError) as caught:
        annual_fold("2024", events)
    assert to_document(caught.value.problem) == _problem("POPULATION_COLLAPSED", message)


@pytest.mark.parametrize("missing", ["evaluation", "training"])
def test_shared_population_refusal_is_normalized_at_outer_reconstruction(source, missing):
    binding = source.registration.document
    events = binding["magnitude_plan"]["events"]
    binding["magnitude_plan"]["events"] = [event for event in events
        if (event["event_date"][:4] != "2024" if missing == "evaluation" else event["event_date"] >= "2024-01-01")]
    with pytest.raises(OpsError) as caught:
        search._nested(binding, _request())
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "outer fold metadata differs from its declared boundaries")


def test_real_seal_verify_is_complete_metadata_only_idempotent_and_preserves_anchor(source, monkeypatch):
    opened = _metadata_only(monkeypatch, source)
    catalog, objects, anchor = catalog_rows(source.conn), artifact_object_bytes(source.store), source.registration.binding_json
    request = _request()
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, request)
    plan, ref = sealed["plan"], from_document(ArtifactRef, sealed["plan_ref"])
    assert set(plan) == {"schema_version", "request", "anchor", "implementation", "candidates", "folds",
                         "horizons", "candidate_count", "planned_fit_count", "selection", "plan_id"}
    assert plan["schema_version"] == "magnitude_search_plan.v1"
    assert plan["request"] == request and plan["horizons"] == [5, 3, 1]
    assert plan["anchor"] == {"run_id": source.registration.run_id, "variant_id": _sha(json.loads(anchor)),
        "registration_ref": to_document(artifact_reference(anchor, "native_experiment_registration.v1.5"))}
    assert plan["selection"] == {"primary_metric": "inner_event_horizon_pooled_rmse", "secondary_metrics": ["mae"],
        "tie_break": "candidate_id_ascending", "cohort": "shared_qualified_labels_across_candidates",
        "training_labels": "strictly_before_inner_fit", "validation_labels": "complete_before_outer_fit",
        "attempts": "all_candidates_before_selection", "failed_candidate": "exclude_entire_score",
        "all_candidates_failed": "refuse", "retries": 0,
        "outer_evaluation": "selected_candidate_only_after_freezing", "outer_fit_failure": "refuse",
        "winner": "latest_outer_inner_selection_before_its_evaluation", "final_read_authority": "none"}
    _assert_folds(source, plan)
    _assert_candidates(plan)
    assert plan["plan_id"] == _sha({key: value for key, value in plan.items() if key != "plan_id"})
    assert ref == artifact_reference(canonical_json(plan).encode(), "magnitude_search_plan.v1")
    sources = plan["implementation"]["source_closure"]
    for path in ("experiments/magnitude_search_plan.py", "experiments/v2_magnitude_search.py",
                 "experiments/v2_prediction.py", "engine/v2/ops/native_registration.py",
                 "engine/v2/foundation/experiment_holdouts.py", "engine/v2/research/magnitude_folds.py"):
        assert sources[path] == "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert plan["implementation"]["environment"] == json.loads(anchor)["environment"]
    assert all(sources[path] == digest for path, digest in json.loads(anchor)["source_closure"].items())
    assert "experiments/magnitude_search_plan.py" not in json.loads(anchor)["source_closure"]
    assert search.read_search_plan(source.conn, source.store, source.registration, _request(), ref) == plan
    assert search.seal_search_plan(source.conn, source.store, source.registration, request) == sealed
    assert catalog_rows(source.conn) == catalog and source.registration.binding_json == anchor
    after = artifact_object_bytes(source.store)
    assert all(after[key] == value for key, value in objects.items())
    assert set(after) - set(objects) <= {ref.storage_key}
    assert opened and all(item.schema_ref in (search.SCHEMA, "native_experiment_registration.v1.5") for item in opened)
    request["space"]["history_ridge"]["alpha"].clear()
    plan["candidates"][0]["recipe"]["features"].clear()
    reread = search.read_search_plan(source.conn, source.store, source.registration, _request(), ref)
    _assert_candidates(reread)


@pytest.mark.parametrize("years", [{"0003": ["0002"]}, {"9998": ["9997"]}])
def test_request_admits_both_year_endpoints(years):
    request = _request()
    request["inner_years"] = years
    admitted, count = search._request(request)
    assert admitted == request and count == 6
    admitted["inner_years"].clear()
    assert request["inner_years"] == years


def test_minimum_integer_leaf_count_has_exact_published_recipes(source, monkeypatch):
    _metadata_only(monkeypatch, source)
    request = _request()
    request.update(space={"history_hgb": {"learning_rate": [0.05], "max_iter": [10],
        "max_leaf_nodes": [2, 4], "min_samples_leaf": [2], "l2_regularization": [0]}},
        max_candidates=2, max_fits=24)
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, request)
    recipes = [_recipe("history_hgb", {**HGB_PARAMS, "learning_rate": 0.05,
        "max_iter": 10, "max_leaf_nodes": leaves, "min_samples_leaf": 2, "l2_regularization": 0})
        for leaves in (2, 4)]
    assert sealed["plan"]["candidates"] == sorted([
        {"candidate_id": _sha(recipe), "recipe": recipe} for recipe in recipes],
        key=lambda candidate: candidate["candidate_id"])
    assert sealed["plan"]["candidate_count"] == 2
    assert sealed["plan"]["planned_fit_count"] == 24
    assert source.store.read_verified(from_document(ArtifactRef, sealed["plan_ref"])) == canonical_json(sealed["plan"]).encode()


@pytest.mark.parametrize("field,value", [
    ("schema_version", "wrong"), ("extra", 1), ("max_candidates", 1), ("max_candidates", 4097),
    ("max_candidates", True), ("max_candidates", 6.0), ("max_candidates", "6"),
    ("max_fits", 0), ("max_fits", True), ("max_fits", 60.0), ("max_fits", "60"),
    ("space", {}), ("space", []), ("space", {"ridge": {"alpha": [1, 2]}}),
    ("inner_years", {}), ("inner_years", []), ("inner_years", {"2024": []}),
    ("inner_years", {"2024": ["2024"]}), ("inner_years", {"2024": ["2025"]}),
    ("inner_years", {"2024": ["2023", "2023"]}), ("inner_years", {"2024": ["2023", "2022"]}),
    ("inner_years", {"2024": [2023]}), ("inner_years", {"２０２４": ["2023"]}),
    ("inner_years", {"0001": ["0000"]}), ("inner_years", {"9999": ["9998"]}),
    pytest.param("inner_years", {"2024": ["0001"]}, id="inner-year-below-minimum"),
])
def test_invalid_request_refuses_before_artifact_access(field, value, monkeypatch):
    request = _request()
    request[field] = value
    monkeypatch.setattr(search, "verify_native_registration", forbid_access)
    monkeypatch.setattr(ArtifactStore, "read_verified", forbid_access)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(object(), object(), object(), request)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "search requires explicit valid axes, inner years and count caps")


@pytest.mark.parametrize("missing", list(_request()))
def test_missing_request_fields_refuse_before_access(missing, monkeypatch):
    request = _request()
    del request[missing]
    monkeypatch.setattr(search, "verify_native_registration", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(object(), object(), object(), request)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "search requires explicit valid axes, inner years and count caps")


@pytest.mark.parametrize("axis,value", [
    ("alpha", []), ("alpha", [0]), ("alpha", [-1]), ("alpha", [True]), ("alpha", ["1"]),
    ("alpha", [float("nan")]), ("alpha", [float("inf")]), ("alpha", [1, 1.0]),
    ("alpha", [2, 1]), ("alpha", [1, 2, 3, 4, 5, 6, 7]), ("alpha", [[1]]),
    ("learning_rate", [0]), ("learning_rate", [-0.1]), ("learning_rate", [False]),
    ("l2_regularization", [-1]), ("l2_regularization", [float("-inf")]),
    ("max_iter", [0]), ("max_iter", [1.0]), ("max_iter", [True]),
    ("max_leaf_nodes", [1]), ("max_leaf_nodes", [2.0]), ("min_samples_leaf", [0]),
    ("min_samples_leaf", [2.5]),
])
def test_invalid_axes_and_duplicate_numeric_values_refuse_before_expansion(axis, value, monkeypatch):
    request = _request()
    kind = "history_ridge" if axis == "alpha" else "history_hgb"
    request["space"][kind][axis] = value
    monkeypatch.setattr(search, "verify_native_registration", forbid_access)
    monkeypatch.setattr(search.itertools, "product", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(object(), object(), object(), request)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "search requires explicit valid axes, inner years and count caps")


@pytest.mark.parametrize("kind,axes", [
    ("history_ridge", {"alpha": [1, 2], "positive": [False]}),
    ("history_ridge", {}), ("history_hgb", {"learning_rate": [0.1]}),
])
def test_unknown_and_incomplete_axes_refuse(kind, axes):
    request = _request()
    request["space"][kind] = axes
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(object(), object(), object(), request)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "search requires explicit valid axes, inner years and count caps")


@pytest.mark.parametrize("cap,value", [("max_candidates", 5), ("max_fits", 59)])
def test_budget_refusal_precedes_cartesian_expansion_and_publication(source, monkeypatch, cap, value):
    request = _request()
    request[cap] = value
    monkeypatch.setattr(search.itertools, "product", forbid_access)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, request)
    message = {"max_candidates": "search requires explicit valid axes, inner years and count caps",
               "max_fits": "planned search fits exceed the explicit cap"}[cap]
    assert to_document(caught.value.problem) == _problem("INVALID_EXPERIMENT_SPEC", message)


def test_singleton_grid_with_admissible_caps_refuses_before_access(monkeypatch):
    request = _request()
    request.update(space={"history_ridge": {"alpha": [1]}}, max_candidates=2)
    monkeypatch.setattr(search, "verify_native_registration", forbid_access)
    monkeypatch.setattr(search.itertools, "product", forbid_access)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(object(), object(), object(), request)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "search requires explicit valid axes, inner years and count caps")


def test_intact_catalog_owned_v14_anchor_refuses_before_nested_planning_or_publication(source, monkeypatch):
    spec = replace(source.spec, experiment_id=source.spec.experiment_id + "-mean-anchor")
    registration = native.register_native(source.conn, source.store, source.repository, spec,
        code_root=ROOT, as_of_month=MONTH, magnitude_training_start="2022-01-01",
        magnitude_labels_before=CUTOFF, magnitude_feature_snapshot_id=source.feature_snapshot.snapshot_id,
        magnitude_history_start=HISTORY)
    assert registration.document["schema_version"] == "native_experiment_registration.v1.4"
    validate_magnitude_history_execution(registration.document)
    assert native.read_native_registration(source.conn, source.store, spec).binding_json == registration.binding_json
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(search, "_nested", forbid_access)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, registration, _request())
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "fitted magnitude execution requires a sealed v1.5 plan")
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("field", ["training_labels_before", "training_candidate_event_ids"])
def test_outer_reconstruction_refuses_independently_changed_metadata(source, field):
    binding = source.registration.document
    if field == "training_labels_before":
        binding["magnitude_plan"]["folds"][0][field] = "2024-01-02"
    else:
        binding["magnitude_plan"]["folds"][0][field].pop()
    with pytest.raises(OpsError) as caught:
        search._nested(binding, _request())
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "outer fold metadata differs from its declared boundaries")


@pytest.mark.parametrize("years,message", [
    ({"2024": ["2023"]}, "search inner years must cover every outer fold"),
    ({"2024": ["2023"], "2026": ["2024"]}, "search inner years must cover every outer fold"),
    ({"2024": ["2022"], "2025": ["2023", "2024"]},
     "every search fold needs earlier training and evaluation candidates"),
    ({"2024": ["2021"], "2025": ["2023", "2024"]},
     "every search fold needs earlier training and evaluation candidates"),
])
def test_missing_or_empty_inner_folds_never_borrow_history_or_skip(source, monkeypatch, years, message):
    request = _request()
    request["inner_years"] = years
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, request)
    assert to_document(caught.value.problem) == _problem("INVALID_EXPERIMENT_SPEC", message)


def test_canonical_order_detachment_and_distinct_declarations_are_not_a_reservation(source):
    request = _request()
    first = search.seal_search_plan(source.conn, source.store, source.registration, request)
    reordered = dict(reversed(list(request.items())))
    reordered["space"] = dict(reversed(list(request["space"].items())))
    reordered["inner_years"] = dict(reversed(list(request["inner_years"].items())))
    assert search.seal_search_plan(source.conn, source.store, source.registration, reordered) == first
    request["space"]["history_ridge"]["alpha"][0] = 0.25
    before = catalog_rows(source.conn)
    second = search.seal_search_plan(source.conn, source.store, source.registration, request)
    assert first["plan_ref"] != second["plan_ref"]
    for sealed in (first, second):
        assert search.read_search_plan(source.conn, source.store, source.registration, sealed["plan"]["request"],
            from_document(ArtifactRef, sealed["plan_ref"])) == sealed["plan"]
    assert catalog_rows(source.conn) == before


@pytest.mark.parametrize("kind", ["history_ridge", "history_hgb"])
def test_two_candidate_single_family_request_is_supported_with_exact_refit_budget(source, kind):
    request = _request()
    axes = request["space"][kind]
    if kind == "history_hgb":
        axes["l2_regularization"] = [0]
    request.update(space={kind: axes}, max_candidates=2, max_fits=24)
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, request)
    plan = sealed["plan"]
    assert plan["candidate_count"] == 2 and plan["planned_fit_count"] == 24
    assert {item["recipe"]["model"] for item in plan["candidates"]} == {kind}
    assert len({item["candidate_id"] for item in plan["candidates"]}) == 2
    assert search.read_search_plan(source.conn, source.store, source.registration, request,
        from_document(ArtifactRef, sealed["plan_ref"])) == plan


@pytest.mark.parametrize("kind", ["history_ridge", "history_hgb"])
def test_changed_axes_produce_independent_complete_recipes_and_hashes(source, monkeypatch, kind):
    _metadata_only(monkeypatch, source)
    request = _request()
    if kind == "history_ridge":
        request.update(space={kind: {"alpha": [0.25, 4, 8]}}, max_candidates=3, max_fits=33)
        recipes = [_recipe(kind, {**RIDGE_PARAMS, "alpha": value}) for value in (0.25, 4, 8)]
        expected_count, expected_fits = 3, 33
    else:
        request.update(space={kind: {"learning_rate": [0.1], "max_iter": [7, 11],
            "max_leaf_nodes": [5, 9], "min_samples_leaf": [1, 4], "l2_regularization": [0.25]}},
            max_candidates=8, max_fits=78)
        recipes = [_recipe(kind, {**HGB_PARAMS, "learning_rate": 0.1, "l2_regularization": 0.25,
            "max_iter": iterations, "max_leaf_nodes": leaves, "min_samples_leaf": samples})
            for iterations, leaves, samples in (
                (7, 5, 1), (7, 5, 4), (7, 9, 1), (7, 9, 4),
                (11, 5, 1), (11, 5, 4), (11, 9, 1), (11, 9, 4))]
        expected_count, expected_fits = 8, 78
    expected = sorted([{"candidate_id": _sha(recipe), "recipe": recipe} for recipe in recipes],
                      key=lambda candidate: candidate["candidate_id"])
    before = catalog_rows(source.conn)
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, request)
    plan = sealed["plan"]
    assert plan["candidates"] == expected
    assert plan["candidate_count"] == expected_count and plan["planned_fit_count"] == expected_fits
    assert plan["request"] == request
    assert search.read_search_plan(source.conn, source.store, source.registration, request,
        from_document(ArtifactRef, sealed["plan_ref"])) == plan
    assert catalog_rows(source.conn) == before


def test_alternate_inner_map_has_exact_fits_and_adjacent_cap_refuses_before_publication(source, monkeypatch):
    _metadata_only(monkeypatch, source)
    request = _request()
    request.update(inner_years={"2024": ["2023"], "2025": ["2024"]}, max_fits=42)
    expected = _expected_folds(source)
    expected[1]["inner"].pop(0)
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, request)
    plan = sealed["plan"]
    assert plan["folds"] == expected
    assert plan["candidate_count"] == 6 and plan["planned_fit_count"] == 42
    assert search.read_search_plan(source.conn, source.store, source.registration, request,
        from_document(ArtifactRef, sealed["plan_ref"])) == plan
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    request["max_fits"] = 41
    monkeypatch.setattr(search.itertools, "product", forbid_access)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, request)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "planned search fits exceed the explicit cap")
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_single_outer_anchor_has_exact_refit_budget_and_adjacent_cap_refusal(source, monkeypatch):
    _, feature_snapshot = build_history(source, training_start="2022-01-01", evaluation_years=["2024"])
    spec = replace(source.spec, experiment_id=source.spec.experiment_id + "-one-outer", folds=("2024",))
    registration = native.register_native(source.conn, source.store, source.repository, spec,
        code_root=ROOT, as_of_month=MONTH, magnitude_training_start="2022-01-01",
        magnitude_labels_before=CUTOFF, magnitude_feature_snapshot_id=feature_snapshot.snapshot_id,
        magnitude_history_start=HISTORY, magnitude_fit_model="history_ridge")
    validate_magnitude_fitted_execution(registration.document)
    assert native.read_native_registration(source.conn, source.store, spec).binding_json == registration.binding_json
    assert registration.document["magnitude_plan"]["evaluation_years"] == ["2024"]
    assert registration.document["magnitude_plan"]["folds"] == [_expected_folds(source)[0]["outer"]]
    _metadata_only(monkeypatch, source)
    request = _request()
    request.update(inner_years={"2024": ["2023"]}, max_fits=21)
    catalog = catalog_rows(source.conn)
    sealed = search.seal_search_plan(source.conn, source.store, registration, request)
    plan = sealed["plan"]
    assert plan["folds"] == _expected_folds(source)[:1]
    # Six candidates times three horizons, plus one three-horizon outer refit.
    assert plan["candidate_count"] == 6 and plan["planned_fit_count"] == 21
    assert plan["horizons"] == [5, 3, 1]
    assert search.read_search_plan(source.conn, source.store, registration, request,
        from_document(ArtifactRef, sealed["plan_ref"])) == plan
    assert catalog_rows(source.conn) == catalog
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    request["max_fits"] = 20
    monkeypatch.setattr(search.itertools, "product", forbid_access)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, registration, request)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "planned search fits exceed the explicit cap")
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_declared_candidate_ceiling_is_admitted_without_expanding_or_executing(monkeypatch):
    request = _request()
    request.update(space={"history_ridge": {"alpha": list(range(1, 4097))}}, max_candidates=4096)
    monkeypatch.setattr(search.itertools, "product", forbid_access)
    detached, count = search._request(request)
    assert count == 4096 and detached == request
    request["space"]["history_ridge"]["alpha"].clear()
    assert len(detached["space"]["history_ridge"]["alpha"]) == 4096


@pytest.mark.parametrize("field", ["schema", "candidate", "candidate_id", "train", "validation", "cutoff",
    "label_cutoff", "fit_count", "horizons", "selection", "anchor", "request", "source", "environment", "extra"])
def test_rehashed_plan_drift_refuses_complete_reconstruction(source, field):
    plan = search.seal_search_plan(source.conn, source.store, source.registration, _request())["plan"]
    if field == "schema":
        plan["schema_version"] = "magnitude_search_plan.v2"
    elif field in ("candidate", "candidate_id"):
        if field == "candidate":
            plan["candidates"][0]["recipe"]["features"] = ["n_prior"]
        else:
            plan["candidates"][0]["candidate_id"] = "sha256:" + "0" * 64
    elif field in ("train", "validation"):
        key = "training_candidate_event_ids" if field == "train" else "evaluation_event_ids"
        plan["folds"][0]["inner"][0][key] = plan["folds"][0]["outer"]["evaluation_event_ids"]
    elif field in ("cutoff", "label_cutoff"):
        key = "training_labels_before" if field == "cutoff" else "evaluation_labels_before"
        plan["folds"][0]["inner"][0][key] = "2025-02-01"
    elif field == "fit_count":
        plan["planned_fit_count"] = 54
    elif field == "horizons":
        plan["horizons"] = [1]
    elif field == "selection":
        plan["selection"]["primary_metric"] = "outer_rmse"
    elif field == "anchor":
        plan["anchor"]["run_id"] = "different-native-run"
    elif field == "request":
        del plan["request"]["max_fits"]
    elif field == "source":
        plan["implementation"]["source_closure"]["experiments/magnitude_search_plan.py"] = "sha256:" + "0" * 64
    elif field == "environment":
        plan["implementation"]["environment"]["unbound"] = "value"
    else:
        plan["extra"] = "unbound"
    plan["plan_id"] = _sha({key: value for key, value in plan.items() if key != "plan_id"})
    ref = source.store.publish_bytes(canonical_json(plan).encode(), schema_ref="magnitude_search_plan.v1")
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(), ref)
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "search plan reference differs from its complete binding")
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("drift", ["anchor_source", "fold_source", "planner_source", "environment"])
def test_installed_identity_drift_refuses_without_publication(source, monkeypatch, drift):
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, _request())
    if drift == "environment":
        monkeypatch.setattr(search, "environment_identity", lambda _: {"changed": True})
    else:
        original = search.source_closure
        def closure(*args):
            sources = original(*args)
            path = {"anchor_source": "experiments/v2_prediction.py", "fold_source":
                    "engine/v2/research/magnitude_folds.py", "planner_source": "experiments/magnitude_search_plan.py"}[drift]
            return {**sources, path: "sha256:" + "0" * 64}
        monkeypatch.setattr(search, "source_closure", closure)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(),
            from_document(ArtifactRef, sealed["plan_ref"]))
    message = ("search plan reference differs from its complete binding" if drift == "planner_source"
               else "search anchor sources or environment differ from this installation")
    assert to_document(caught.value.problem) == _problem("EXPERIMENT_IDENTITY_CONFLICT", message)


@pytest.mark.parametrize("drift", ["anchor_source", "fold_source", "environment"])
def test_prepare_rejects_installed_anchor_drift_before_publication(source, monkeypatch, drift):
    _metadata_only(monkeypatch, source)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    if drift == "environment":
        monkeypatch.setattr(search, "environment_identity", lambda _: {"changed": True})
    else:
        path = "engine/v2/research/magnitude_folds.py" if drift == "fold_source" else "experiments/v2_prediction.py"
        assert path in source.registration.document["source_closure"]
        original = search.source_closure
        def closure(*args):
            return {**original(*args), path: "sha256:" + "0" * 64}
        monkeypatch.setattr(search, "source_closure", closure)
    monkeypatch.setattr(search, "_candidates", forbid_access)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, _request())
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "search anchor sources or environment differ from this installation")
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_wrong_reference_schema_refuses_before_open(source, monkeypatch):
    ref = artifact_reference(b"protected-payload", "native_magnitude_fitted_result.v1")
    monkeypatch.setattr(ArtifactStore, "read_verified", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(), ref)
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "search plan reference has the wrong schema")


@pytest.mark.parametrize("kind", ["none", "document"])
def test_nonreference_values_refuse_before_preparation_or_artifacts(source, monkeypatch, kind):
    ref = None if kind == "none" else to_document(artifact_reference(b"metadata", search.SCHEMA))
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(search, "_prepare", forbid_access)
    monkeypatch.setattr(ArtifactStore, "read_verified", forbid_access)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(), ref)
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "search plan reference has the wrong schema")
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("field", ["artifact_id", "storage_key", "byte_size", "schema_version"])
def test_same_hash_reference_metadata_drift_refuses_before_plan_open(source, monkeypatch, field):
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, _request())
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    payload = source.store.read_verified(ref)
    assert payload == canonical_json(sealed["plan"]).encode()
    changes = {"artifact_id": ref.artifact_id + "-changed", "storage_key": ref.storage_key + "-changed",
               "byte_size": ref.byte_size + 1, "schema_version": "artifact_ref.v9.0"}
    changed = replace(ref, **{field: changes[field]})
    assert changed.content_hash == ref.content_hash and changed != ref
    original, opened = ArtifactStore.read_verified, []
    def read(store, candidate):
        assert candidate.schema_ref == "native_experiment_registration.v1.5", "rejected plan opened"
        opened.append(candidate)
        return original(store, candidate)
    monkeypatch.setattr(ArtifactStore, "read_verified", read)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(), changed)
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "search plan reference differs from its complete binding")
    assert opened and (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    assert (source.store.root / ref.storage_key).read_bytes() == payload


def test_forged_metadata_schema_cannot_open_arbitrary_payload(source, monkeypatch):
    ref = source.store.publish_bytes(b"synthetic protected outcome", schema_ref="magnitude_search_plan.v1")
    original, opened = ArtifactStore.read_verified, []
    def read(store, candidate):
        assert candidate != ref, "unadmitted object opened"
        opened.append(candidate)
        return original(store, candidate)
    monkeypatch.setattr(ArtifactStore, "read_verified", read)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(), ref)
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "search plan reference differs from its complete binding")
    assert opened and all(candidate.schema_ref == "native_experiment_registration.v1.5" for candidate in opened)


@pytest.mark.parametrize("failure", [fail("INPUT_CHANGED", "missing source"), SyntaxError("invalid source"),
    OSError("unreadable source"), ArtifactError("UNSAFE_PATH", "invalid source path"), ValueError("invalid source")])
def test_unverifiable_installed_source_is_identity_refusal(source, monkeypatch, failure):
    _metadata_only(monkeypatch, source)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    def broken(*args):
        raise failure
    monkeypatch.setattr(search, "source_closure", broken)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, _request())
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "search implementation sources cannot be verified")
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def _problem(code, message, *, details=None):
    return {"code": code, "category": "validation", "retryable": False, "message": message,
        "stage": None, "trace_id": None, "dependency_refs": [], "retry_after_seconds": None,
        "diagnostic_ref": None, "details": details or {}, "schema_version": "problem.v1.0"}


def test_real_plan_byte_corruption_is_exact_nonretryable_library_and_cli_refusal(source, tmp_path, capsys):
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, _request())
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    path = source.store.root / ref.storage_key
    original = path.read_bytes()
    request, reference = tmp_path / "request.json", tmp_path / "reference.json"
    request.write_text(json.dumps(_request()))
    reference.write_text(json.dumps(sealed["plan_ref"]))
    before = catalog_rows(source.conn)
    try:
        path.chmod(0o644)
        path.write_bytes(b"!" * len(original))
        expected = _problem("EXPERIMENT_IDENTITY_CONFLICT", "stored search plan failed integrity verification")
        with pytest.raises(OpsError) as caught:
            search.read_search_plan(source.conn, source.store, source.registration, _request(), ref)
        assert to_document(caught.value.problem) == expected
        assert cli.main(_args(source, "verify", request, reference)) == 2
        assert json.loads(capsys.readouterr().out) == {"refused": expected}
        with pytest.raises(OpsError) as caught:
            search.seal_search_plan(source.conn, source.store, source.registration, _request())
        assert to_document(caught.value.problem) == expected
        assert cli.main(_args(source, "prepare", request)) == 2
        assert json.loads(capsys.readouterr().out) == {"refused": expected}
        assert catalog_rows(source.conn) == before
    finally:
        path.write_bytes(original)
        path.chmod(0o444)


def test_missing_plan_object_has_exact_resource_problem_without_publication(source, tmp_path, capsys, monkeypatch):
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, _request())
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    store = ArtifactStore(tmp_path / "registration-only")
    store.publish_bytes(source.registration.binding_json, schema_ref="native_experiment_registration.v1.5")
    assert not (store.root / ref.storage_key).exists()
    before = catalog_rows(source.conn), artifact_object_bytes(store)
    request, reference = tmp_path / "request.json", tmp_path / "reference.json"
    request.write_text(json.dumps(_request()))
    reference.write_text(json.dumps(sealed["plan_ref"]))
    expected = _problem("RESOURCE_UNAVAILABLE", "search planning request could not be completed")
    expected.update(category="resource", retryable=True)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, store, source.registration, _request(), ref)
    assert to_document(caught.value.problem) == expected
    args = _args(source, "verify", request, reference)
    args[args.index("--store-root") + 1] = str(store.root)
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert not captured.err and json.loads(captured.out) == {"refused": expected}
    assert (catalog_rows(source.conn), artifact_object_bytes(store)) == before


@pytest.mark.parametrize("code", ["MISSING", "NOT_A_REGULAR_FILE", "UNSAFE_PATH"])
def test_other_publication_artifact_failures_have_exact_resource_problem(source, monkeypatch, code):
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    def broken(*args, **kwargs):
        raise ArtifactError(code, "synthetic private detail")
    monkeypatch.setattr(ArtifactStore, "publish_bytes", broken)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, _request())
    expected = _problem("RESOURCE_UNAVAILABLE", "search planning request could not be completed")
    expected.update(category="resource", retryable=True)
    assert to_document(caught.value.problem) == expected
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_changed_catalog_anchor_refuses_before_plan_publication(source, monkeypatch):
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbid_access)
    before = catalog_rows(source.conn)
    source.conn.execute("BEGIN")
    try:
        source.conn.execute("UPDATE experiment_runs SET input_hash=? WHERE run_id=?",
                            ("sha256:" + "0" * 64, source.registration.run_id))
        with pytest.raises(OpsError) as caught:
            search.seal_search_plan(source.conn, source.store, source.registration, _request())
        assert to_document(caught.value.problem) == _problem(
            "EXPERIMENT_IDENTITY_CONFLICT", "native registration differs from its immutable binding")
    finally:
        source.conn.rollback()
    assert catalog_rows(source.conn) == before


def test_fixed_v15_cannot_execute_search_recipe_or_sidecar(source):
    plan = search.seal_search_plan(source.conn, source.store, source.registration, _request())["plan"]
    expected = _problem("EXPERIMENT_IDENTITY_CONFLICT", "fitted magnitude execution requires a sealed v1.5 plan")
    with pytest.raises(OpsError) as caught:
        validate_magnitude_fitted_execution(plan)
    assert to_document(caught.value.problem) == expected
    for candidate in plan["candidates"]:
        binding = source.registration.document
        binding["magnitude_execution"]["fit_recipe"] = candidate["recipe"]
        with pytest.raises(OpsError) as caught:
            validate_magnitude_fitted_execution(binding)
        assert to_document(caught.value.problem) == expected


def test_planning_preserves_fresh_v15_registration_and_history_producer_bytes(source):
    original = source.registration.binding_json
    producer = canonical_json(source.build["context"])
    search.seal_search_plan(source.conn, source.store, source.registration, _request())
    admitted = native.require_native_registration(source.conn, source.store, source.repository, source.spec,
        code_root=ROOT, as_of_month=MONTH, magnitude_training_start="2022-01-01",
        magnitude_labels_before=CUTOFF, magnitude_feature_snapshot_id=source.feature_snapshot.snapshot_id,
        magnitude_history_start=HISTORY, magnitude_fit_model="history_ridge")
    assert admitted.binding_json == original and canonical_json(source.build["context"]) == producer
    assert admitted.document["magnitude_history"]["context"] == source.build["context"]


def _args(source, action, path, reference=None):
    return [action, "--spec", str(source.spec_path), "--catalog", str(source.catalog_path),
            "--store-root", str(source.store.root), "--request", str(path),
            *([] if reference is None else ["--plan-ref", str(reference)])]


def test_real_cli_prepare_verify_preserves_catalog_and_no_trial_state(source, tmp_path, capsys, monkeypatch):
    _metadata_only(monkeypatch, source)
    request, reference = tmp_path / "request.json", tmp_path / "reference.json"
    request.write_text(json.dumps(_request()))
    before = catalog_rows(source.conn)
    paths_before = {str(path.relative_to(source.store.root)) for path in source.store.root.rglob("*") if path.is_file()}
    assert cli.main(_args(source, "prepare", request)) == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"plan_ref", "plan_id", "candidate_count", "planned_fit_count"}
    assert output["candidate_count"] == 6 and output["planned_fit_count"] == 60
    after_prepare = artifact_object_bytes(source.store)
    paths_after = {str(path.relative_to(source.store.root)) for path in source.store.root.rglob("*") if path.is_file()}
    assert paths_after - paths_before <= {output["plan_ref"]["storage_key"]}
    reference.write_text(json.dumps(output["plan_ref"]))
    assert cli.main(_args(source, "verify", request, reference)) == 0
    assert json.loads(capsys.readouterr().out) == {"verified": output["plan_id"]}
    assert catalog_rows(source.conn) == before
    assert artifact_object_bytes(source.store) == after_prepare
    assert {str(path.relative_to(source.store.root)) for path in source.store.root.rglob("*") if path.is_file()} == paths_after
    assert cli.main([*_args(source, "prepare", request), "--no-ledger"]) == 0
    assert json.loads(capsys.readouterr().out) == output
    assert cli.main([*_args(source, "verify", request, reference), "--no-ledger"]) == 0
    assert json.loads(capsys.readouterr().out) == {"verified": output["plan_id"]}
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == after_prepare
    assert {str(path.relative_to(source.store.root)) for path in source.store.root.rglob("*") if path.is_file()} == paths_after


@pytest.mark.parametrize("action,options,diagnostic", [
    ("prepare", [], "the following arguments are required: --request"),
    ("verify", [], "the following arguments are required: --request"),
    ("verify", ["--plan-ref", "unused"], "the following arguments are required: --request"),
    ("verify", ["--request", "unused"], "verify requires --plan-ref in addition to --request; prepare accepts no --plan-ref"),
    ("prepare", ["--request", "unused", "--plan-ref", "unused"],
     "verify requires --plan-ref in addition to --request; prepare accepts no --plan-ref"),
])
def test_cli_options_refuse_before_effects(action, options, diagnostic, monkeypatch, capsys):
    monkeypatch.setattr(cli, "connect", forbid_access)
    monkeypatch.setattr(Path, "is_dir", forbid_access)
    with pytest.raises(SystemExit) as caught:
        cli.main([action, "--spec", "unused", "--catalog", "unused", "--store-root", "unused", *options])
    assert caught.value.code == 2
    output = capsys.readouterr()
    assert not output.out and output.err.splitlines()[-1].endswith("error: " + diagnostic)


def test_cli_forwards_exact_typed_refusal_and_closes_catalog(source, tmp_path, monkeypatch, capsys):
    request = tmp_path / "request.json"
    request.write_text(json.dumps(_request()))
    problem = fail("EXPERIMENT_IDENTITY_CONFLICT", "independent synthetic refusal", details={"marker": "native"})
    def refuse(*args):
        raise problem
    monkeypatch.setattr(cli, "seal_search_plan", refuse)
    connections, original = [], cli.connect
    def connect(*args, **kwargs):
        conn = original(*args, **kwargs)
        connections.append(conn)
        return conn
    monkeypatch.setattr(cli, "connect", connect)
    assert cli.main(_args(source, "prepare", request)) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "independent synthetic refusal", details={"marker": "native"})}
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize("kind", ["bad_json", "invalid_request", "missing_catalog", "missing_store"])
def test_cli_malformed_or_missing_inputs_have_exact_envelopes_and_no_effects(source, tmp_path, capsys, kind):
    request = tmp_path / "request.json"
    request.write_text("{" if kind == "bad_json" else json.dumps({} if kind == "invalid_request" else _request()))
    args = _args(source, "prepare", request)
    message = "search planning request could not be completed"
    code = "INVALID_EXPERIMENT_SPEC"
    if kind == "invalid_request":
        message = "search requires explicit valid axes, inner years and count caps"
    elif kind in ("missing_catalog", "missing_store"):
        option = "--catalog" if kind == "missing_catalog" else "--store-root"
        args[args.index(option) + 1] = str(tmp_path / "missing")
        code = "RESOURCE_UNAVAILABLE"
        if kind == "missing_store":
            message = "search requires an existing artifact store"
    expected = _problem(code, message)
    if code == "RESOURCE_UNAVAILABLE":
        expected.update(category="resource", retryable=True)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert not captured.err and json.loads(captured.out) == {"refused": expected}
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    assert not (tmp_path / "missing").exists()


def test_independent_oracles_reject_hardcoded_grid_and_temporal_leak(source):
    plan = search.seal_search_plan(source.conn, source.store, source.registration, _request())["plan"]
    broken = copy.deepcopy(plan)
    broken["candidates"][0]["recipe"]["estimator"]["params"]["alpha"] = 1
    with pytest.raises(AssertionError):
        _assert_candidates(broken)
    broken = copy.deepcopy(plan)
    broken["folds"][0]["inner"][0]["training_candidate_event_ids"].extend(_ids(source, ("2022-12-22",)))
    with pytest.raises(AssertionError):
        _assert_folds(source, broken)
    broken = copy.deepcopy(plan)
    broken["folds"][1]["inner"][1]["evaluation_event_ids"].extend(_ids(source, ("2024-12-27", "2024-12-30")))
    with pytest.raises(AssertionError):
        _assert_folds(source, broken)


@pytest.mark.parametrize("option", ["--spec", "--request", "--plan-ref"])
@pytest.mark.parametrize("kind,message", [
    ("oversized", "search metadata requires bounded regular files"),
    ("directory", "search metadata requires bounded regular files"),
    ("symlink", "search metadata must not follow leaf symlinks"),
    ("invalid_utf8", "search planning request could not be completed"),
    ("invalid_json", "search planning request could not be completed"),
])
def test_cli_bounds_each_external_document_before_catalog(
        source, tmp_path, monkeypatch, capsys, option, kind, message):
    request, reference, bad = (tmp_path / name for name in ("request", "reference", "bad"))
    request.write_text(json.dumps(_request()))
    reference.write_text(json.dumps(to_document(artifact_reference(b"unused", search.SCHEMA))))
    if kind == "directory":
        bad.mkdir()
    elif kind == "symlink":
        bad.symlink_to(request)
    else:
        bad.write_bytes({"oversized": b" " * (1024 * 1024 + 1),
                         "invalid_utf8": b"\xff", "invalid_json": b"{"}[kind])
    args = _args(source, "verify", request, reference)
    args[args.index(option) + 1] = str(bad)
    monkeypatch.setattr(cli, "connect", forbid_access)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": _problem("INVALID_EXPERIMENT_SPEC", message)}
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_cli_fifo_refuses_without_blocking(source, tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    result = subprocess.run([sys.executable, "-m", "experiments.v2_magnitude_search",
        *_args(source, "prepare", path)], cwd=ROOT, capture_output=True, text=True, timeout=15,
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")})
    assert result.returncode == 2 and not result.stderr
    assert json.loads(result.stdout) == {"refused": _problem(
        "INVALID_EXPERIMENT_SPEC", "search metadata requires bounded regular files")}
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_bounded_reader_accepts_exact_limit_and_closes_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "metadata"
    path.write_bytes(b"{}" + b" " * (1024 * 1024 - 2))
    opened, original = [], os.open
    def capture(*args):
        descriptor = original(*args)
        opened.append(descriptor)
        return descriptor
    monkeypatch.setattr(cli.os, "open", capture)
    assert cli.MAX_METADATA_BYTES == 1024 * 1024
    assert cli._read_document(path) == {}
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_bounded_reader_checks_growth_overshoot_before_parse_and_closes(tmp_path, monkeypatch):
    path = tmp_path / "metadata"
    path.write_bytes(b"{}")
    original, original_stream, opened, consumed = os.fstat, os.fdopen, [], []
    def earlier_size(descriptor):
        opened.append(descriptor)
        info = original(descriptor)
        with path.open("ab") as stream:
            stream.write(b" " * (2 * 1024 * 1024))
        return info
    class BoundedStream:
        def __init__(self, descriptor, mode, **kwargs):
            assert kwargs == {"buffering": 0, "closefd": False}
            self.stream = original_stream(descriptor, mode, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def read(self, requested):
            assert requested == 1024 * 1024 + 1 - sum(consumed)
            data = self.stream.read(requested)
            consumed.append(len(data))
            return data
    monkeypatch.setattr(cli.os, "fstat", earlier_size)
    monkeypatch.setattr(cli.os, "fdopen", BoundedStream)
    monkeypatch.setattr(cli.json, "loads", forbid_access)
    with pytest.raises(OpsError) as caught:
        cli._read_document(path)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "search metadata exceeds its byte limit")
    assert sum(consumed) == 1024 * 1024 + 1
    with pytest.raises(OSError):
        original(opened[0])


def test_bounded_reader_keeps_opened_descriptor_across_path_replacement(tmp_path, monkeypatch):
    path, replacement = tmp_path / "metadata", tmp_path / "replacement"
    path.write_text('{"opened":1}')
    replacement.write_text('{"opened":2}')
    original = os.open
    def replace_after_open(*args):
        descriptor = original(*args)
        replacement.replace(path)
        return descriptor
    monkeypatch.setattr(cli.os, "open", replace_after_open)
    assert cli._read_document(path) == {"opened": 1}
    assert json.loads(path.read_text()) == {"opened": 2}


@pytest.mark.parametrize("stage", ["connect", "read_native_registration", "seal_search_plan"])
@pytest.mark.parametrize("error_type", [KeyError, TypeError, ValueError, RecursionError])
def test_cli_does_not_misclassify_unexpected_builtins(source, tmp_path, monkeypatch, capsys, stage, error_type):
    request = tmp_path / "request"
    request.write_text(json.dumps(_request()))
    error = error_type("independent internal defect")
    connections, original = [], cli.connect
    def connect(*args, **kwargs):
        conn = original(*args, **kwargs)
        connections.append(conn)
        return conn
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(cli, "connect", connect)
    monkeypatch.setattr(cli, stage, broken)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(error_type) as caught:
        cli.main(_args(source, "prepare", request))
    assert caught.value is error and not capsys.readouterr().out
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


@pytest.mark.parametrize("kind", ["ref_shape", "spec_input_files", "huge_integer", "deep_json"])
def test_cli_typed_input_conversions_still_refuse_before_catalog(source, tmp_path, monkeypatch, capsys, kind):
    request, reference = tmp_path / "request", tmp_path / "reference"
    request.write_text("9" * 5000 if kind == "huge_integer" else json.dumps(_request()))
    if kind == "deep_json":
        request.write_text("[" * 100_000 + "0" + "]" * 100_000)
        with pytest.raises(RecursionError):
            json.loads(request.read_text())
    reference.write_text("{}")
    args = _args(source, "verify", request, reference) if kind == "ref_shape" else _args(source, "prepare", request)
    if kind == "spec_input_files":
        spec = json.loads(source.spec_path.read_text())
        spec["input_files"] = None
        path = tmp_path / "spec"
        path.write_text(json.dumps(spec))
        args[args.index("--spec") + 1] = str(path)
    monkeypatch.setattr(cli, "connect", forbid_access)
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": _problem(
        "INVALID_EXPERIMENT_SPEC", "search planning request could not be completed")}


def test_cli_unopenable_socket_is_resource_failure_before_catalog(source, tmp_path, monkeypatch, capsys):
    path = tmp_path / "sock"
    original = os.open
    def open_error(candidate, flags, *args, **kwargs):
        if str(candidate) == str(path):
            raise OSError(errno.ENXIO, "independent unopenable socket")
        return original(candidate, flags, *args, **kwargs)
    monkeypatch.setattr(cli.os, "open", open_error)
    monkeypatch.setattr(cli, "connect", forbid_access)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    assert cli.main(_args(source, "prepare", path)) == 2
    expected = _problem("RESOURCE_UNAVAILABLE", "search planning request could not be completed")
    expected.update(category="resource", retryable=True)
    assert json.loads(capsys.readouterr().out) == {"refused": expected}
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("kind,code,message,details", [
    ("missing", "INVALID_REQUEST", "experiment specification is missing a required field", {"field": "hypothesis"}),
    ("shape", "INVALID_EXPERIMENT_SPEC", "arms and folds must be JSON arrays", {"field": "arms", "type": "str"}),
    ("unknown", "INVALID_EXPERIMENT_SPEC", "experiment specification declares fields this resolver does not know",
     {"fields": ["unexpected"]}),
])
def test_cli_preserves_native_spec_conversion_refusals(
        source, tmp_path, monkeypatch, capsys, kind, code, message, details):
    spec = json.loads(source.spec_path.read_text())
    if kind == "missing":
        del spec["hypothesis"]
    elif kind == "shape":
        spec["arms"] = "invalid"
    else:
        spec["unexpected"] = True
    path = tmp_path / "spec"
    path.write_text(json.dumps(spec))
    args = _args(source, "prepare", tmp_path / "never-read-request")
    args[args.index("--spec") + 1] = str(path)
    monkeypatch.setattr(cli, "connect", forbid_access)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": _problem(code, message, details=details)}
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_bounded_reader_accumulates_short_unbuffered_reads(tmp_path, monkeypatch):
    path = tmp_path / "metadata"
    path.write_text('{"independent": [1,2,3]}')
    original, lengths = os.fdopen, []
    class ShortReader:
        def __init__(self, descriptor, mode, **kwargs):
            assert kwargs == {"buffering": 0, "closefd": False}
            self.stream = original(descriptor, mode, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def read(self, requested):
            assert requested == 1024 * 1024 + 1 - sum(lengths)
            chunk = self.stream.read(min(3, requested))
            lengths.append(len(chunk))
            return chunk
    monkeypatch.setattr(cli.os, "fdopen", ShortReader)
    assert cli._read_document(path) == {"independent": [1, 2, 3]}
    assert lengths[-1] == 0 and len(lengths) > 2


@pytest.mark.parametrize("kind", ["oversized", "directory"])
def test_bounded_reader_rejects_descriptor_before_read_or_parse(tmp_path, monkeypatch, kind):
    path = tmp_path / "metadata"
    if kind == "oversized":
        path.write_bytes(b" " * (1024 * 1024 + 1))
    else:
        path.mkdir()
    monkeypatch.setattr(cli.os, "fdopen", forbid_access)
    monkeypatch.setattr(cli.json, "loads", forbid_access)
    with pytest.raises(OpsError) as caught:
        cli._read_document(path)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "search metadata requires bounded regular files")
