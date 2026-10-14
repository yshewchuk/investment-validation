"""Native metadata-only search sealing with independent folds, grids and refusals."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
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
from experiments import magnitude_search_plan as search
from experiments import native_registration as native
from experiments import v2_magnitude_search as cli
from tests.v2.models.training.test_magnitude_estimators import FEATURES, HGB_PARAMS, RIDGE_PARAMS
from tests.v2.research.test_event_history import _event
from tests.v2.research.test_magnitude_history_model import (
    CUTOFF,
    HISTORY,
    MONTH,
    ROOT,
    _dataset,
    _no_rebuild,
    _source,
)
from tests.v2.research.test_magnitude_history_store import _build
from tests.v2.research.test_magnitude_registration import _forbid
from tests.v2.research.test_native_registration import _catalog, _objects
from tests.v2.research.test_prediction_inputs import _move


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
    events, moves = _dataset()
    for event in events:
        if event["event_date"].date().isoformat() in ("2023-01-03", "2025-01-07"):
            event["session"] = "BMO"
    for ticker in ("SYNTH_A", "SYNTH_B"):
        for day in ("2022-12-22", "2022-12-23", "2024-12-27", "2024-12-30"):
            event = _event(day, ticker=ticker)
            events.append(event)
            moves.append(_move(event, -3))
    value = _source(root, register=False, build=False, events=events, moves=moves)
    value.build, value.feature_snapshot = _build(value, training_start="2022-01-01",
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

    _no_rebuild(monkeypatch)
    for estimator in (Ridge, HistGradientBoostingRegressor):
        monkeypatch.setattr(estimator, "__init__", _forbid)
    for name in ("scan", "resolve", "resolve_pinned", "fragment_records", "table_contract"):
        monkeypatch.setattr(Repository, name, _forbid)
    from engine.v2.models.training import magnitude
    monkeypatch.setattr(magnitude, "fit_magnitude_predictions", _forbid)
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


def test_real_seal_verify_is_complete_metadata_only_idempotent_and_preserves_anchor(source, monkeypatch):
    opened = _metadata_only(monkeypatch, source)
    catalog, objects, anchor = _catalog(source.conn), _objects(source.store), source.registration.binding_json
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
    assert _catalog(source.conn) == catalog and source.registration.binding_json == anchor
    after = _objects(source.store)
    assert all(after[key] == value for key, value in objects.items())
    assert set(after) - set(objects) <= {ref.storage_key}
    assert opened and all(item.schema_ref in (search.SCHEMA, "native_experiment_registration.v1.5") for item in opened)
    request["space"]["history_ridge"]["alpha"].clear()
    plan["candidates"][0]["recipe"]["features"].clear()
    reread = search.read_search_plan(source.conn, source.store, source.registration, _request(), ref)
    _assert_candidates(reread)


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
])
def test_invalid_request_refuses_before_artifact_access(field, value, monkeypatch):
    request = _request()
    request[field] = value
    monkeypatch.setattr(search, "verify_native_registration", _forbid)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(object(), object(), object(), request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


@pytest.mark.parametrize("missing", list(_request()))
def test_missing_request_fields_refuse_before_access(missing, monkeypatch):
    request = _request()
    del request[missing]
    monkeypatch.setattr(search, "verify_native_registration", _forbid)
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
    monkeypatch.setattr(search, "verify_native_registration", _forbid)
    monkeypatch.setattr(search.itertools, "product", _forbid)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(object(), object(), object(), request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


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
    monkeypatch.setattr(search.itertools, "product", _forbid)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


def test_singleton_grid_with_admissible_caps_refuses_before_access(monkeypatch):
    request = _request()
    request.update(space={"history_ridge": {"alpha": [1]}}, max_candidates=2)
    monkeypatch.setattr(search, "verify_native_registration", _forbid)
    monkeypatch.setattr(search.itertools, "product", _forbid)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
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
    before = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(search, "_nested", _forbid)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, registration, _request())
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "fitted magnitude execution requires a sealed v1.5 plan")
    assert (_catalog(source.conn), _objects(source.store)) == before


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


@pytest.mark.parametrize("years", [
    {"2024": ["2023"]}, {"2024": ["2023"], "2026": ["2024"]},
    {"2024": ["2022"], "2025": ["2023", "2024"]},
    {"2024": ["2021"], "2025": ["2023", "2024"]},
])
def test_missing_or_empty_inner_folds_never_borrow_history_or_skip(source, monkeypatch, years):
    request = _request()
    request["inner_years"] = years
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


def test_canonical_order_detachment_and_distinct_declarations_are_not_a_reservation(source):
    request = _request()
    first = search.seal_search_plan(source.conn, source.store, source.registration, request)
    reordered = dict(reversed(list(request.items())))
    reordered["space"] = dict(reversed(list(request["space"].items())))
    reordered["inner_years"] = dict(reversed(list(request["inner_years"].items())))
    assert search.seal_search_plan(source.conn, source.store, source.registration, reordered) == first
    request["space"]["history_ridge"]["alpha"][0] = 0.25
    before = _catalog(source.conn)
    second = search.seal_search_plan(source.conn, source.store, source.registration, request)
    assert first["plan_ref"] != second["plan_ref"]
    for sealed in (first, second):
        assert search.read_search_plan(source.conn, source.store, source.registration, sealed["plan"]["request"],
            from_document(ArtifactRef, sealed["plan_ref"])) == sealed["plan"]
    assert _catalog(source.conn) == before


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
    before = _catalog(source.conn)
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, request)
    plan = sealed["plan"]
    assert plan["candidates"] == expected
    assert plan["candidate_count"] == expected_count and plan["planned_fit_count"] == expected_fits
    assert plan["request"] == request
    assert search.read_search_plan(source.conn, source.store, source.registration, request,
        from_document(ArtifactRef, sealed["plan_ref"])) == plan
    assert _catalog(source.conn) == before


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
    before = _catalog(source.conn), _objects(source.store)
    request["max_fits"] = 41
    monkeypatch.setattr(search.itertools, "product", _forbid)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, request)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "planned search fits exceed the explicit cap")
    assert (_catalog(source.conn), _objects(source.store)) == before


def test_single_outer_anchor_has_exact_refit_budget_and_adjacent_cap_refusal(source, monkeypatch):
    _, feature_snapshot = _build(source, training_start="2022-01-01", evaluation_years=["2024"])
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
    catalog = _catalog(source.conn)
    sealed = search.seal_search_plan(source.conn, source.store, registration, request)
    plan = sealed["plan"]
    assert plan["folds"] == _expected_folds(source)[:1]
    # Six candidates times three horizons, plus one three-horizon outer refit.
    assert plan["candidate_count"] == 6 and plan["planned_fit_count"] == 21
    assert plan["horizons"] == [5, 3, 1]
    assert search.read_search_plan(source.conn, source.store, registration, request,
        from_document(ArtifactRef, sealed["plan_ref"])) == plan
    assert _catalog(source.conn) == catalog
    before = _catalog(source.conn), _objects(source.store)
    request["max_fits"] = 20
    monkeypatch.setattr(search.itertools, "product", _forbid)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, registration, request)
    assert to_document(caught.value.problem) == _problem(
        "INVALID_EXPERIMENT_SPEC", "planned search fits exceed the explicit cap")
    assert (_catalog(source.conn), _objects(source.store)) == before


def test_declared_candidate_ceiling_is_admitted_without_expanding_or_executing(monkeypatch):
    request = _request()
    request.update(space={"history_ridge": {"alpha": list(range(1, 4097))}}, max_candidates=4096)
    monkeypatch.setattr(search.itertools, "product", _forbid)
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
    before = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(), ref)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not caught.value.problem.retryable
    assert (_catalog(source.conn), _objects(source.store)) == before


@pytest.mark.parametrize("drift", ["anchor_source", "planner_source", "environment"])
def test_installed_identity_drift_refuses_without_publication(source, monkeypatch, drift):
    sealed = search.seal_search_plan(source.conn, source.store, source.registration, _request())
    if drift == "environment":
        monkeypatch.setattr(search, "environment_identity", lambda _: {"changed": True})
    else:
        original = search.source_closure
        def closure(*args):
            sources = original(*args)
            path = "experiments/v2_prediction.py" if drift == "anchor_source" else "experiments/magnitude_search_plan.py"
            return {**sources, path: "sha256:" + "0" * 64}
        monkeypatch.setattr(search, "source_closure", closure)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(),
            from_document(ArtifactRef, sealed["plan_ref"]))
    message = ("search plan reference differs from its complete binding" if drift == "planner_source"
               else "search anchor sources or environment differ from this installation")
    assert to_document(caught.value.problem) == _problem("EXPERIMENT_IDENTITY_CONFLICT", message)


@pytest.mark.parametrize("drift", ["anchor_source", "environment"])
def test_prepare_rejects_installed_anchor_drift_before_publication(source, monkeypatch, drift):
    _metadata_only(monkeypatch, source)
    before = _catalog(source.conn), _objects(source.store)
    if drift == "environment":
        monkeypatch.setattr(search, "environment_identity", lambda _: {"changed": True})
    else:
        path = "experiments/v2_prediction.py"
        assert path in source.registration.document["source_closure"]
        original = search.source_closure
        def closure(*args):
            return {**original(*args), path: "sha256:" + "0" * 64}
        monkeypatch.setattr(search, "source_closure", closure)
    monkeypatch.setattr(search, "_candidates", _forbid)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, _request())
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "search anchor sources or environment differ from this installation")
    assert (_catalog(source.conn), _objects(source.store)) == before


def test_wrong_reference_schema_refuses_before_open(source, monkeypatch):
    ref = artifact_reference(b"protected-payload", "native_magnitude_fitted_result.v1")
    monkeypatch.setattr(ArtifactStore, "read_verified", _forbid)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(), ref)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not caught.value.problem.retryable


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
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    before = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError) as caught:
        search.read_search_plan(source.conn, source.store, source.registration, _request(), changed)
    assert to_document(caught.value.problem) == _problem(
        "EXPERIMENT_IDENTITY_CONFLICT", "search plan reference differs from its complete binding")
    assert opened and (_catalog(source.conn), _objects(source.store)) == before
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
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not caught.value.problem.retryable
    assert opened and all(candidate.schema_ref == "native_experiment_registration.v1.5" for candidate in opened)


@pytest.mark.parametrize("failure", [fail("INPUT_CHANGED", "missing source"), SyntaxError("invalid source")])
def test_unverifiable_installed_source_is_identity_refusal(source, monkeypatch, failure):
    def broken(*args):
        raise failure
    monkeypatch.setattr(search, "source_closure", broken)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        search.seal_search_plan(source.conn, source.store, source.registration, _request())
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not caught.value.problem.retryable
    assert caught.value.problem.message == "search implementation sources cannot be verified"


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
    before = _catalog(source.conn)
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
        assert _catalog(source.conn) == before
    finally:
        path.write_bytes(original)
        path.chmod(0o444)


def test_changed_catalog_anchor_refuses_before_plan_publication(source, monkeypatch):
    monkeypatch.setattr(ArtifactStore, "publish_bytes", _forbid)
    before = _catalog(source.conn)
    source.conn.execute("BEGIN")
    try:
        source.conn.execute("UPDATE experiment_runs SET input_hash=? WHERE run_id=?",
                            ("sha256:" + "0" * 64, source.registration.run_id))
        with pytest.raises(OpsError) as caught:
            search.seal_search_plan(source.conn, source.store, source.registration, _request())
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not caught.value.problem.retryable
    finally:
        source.conn.rollback()
    assert _catalog(source.conn) == before


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
    before = _catalog(source.conn)
    assert cli.main(_args(source, "prepare", request)) == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"plan_ref", "plan_id", "candidate_count", "planned_fit_count"}
    assert output["candidate_count"] == 6 and output["planned_fit_count"] == 60
    reference.write_text(json.dumps(output["plan_ref"]))
    assert cli.main(_args(source, "verify", request, reference)) == 0
    assert json.loads(capsys.readouterr().out) == {"verified": output["plan_id"]}
    assert _catalog(source.conn) == before
    assert not list(tmp_path.rglob("REPORT.md")) and not list(tmp_path.rglob("LEDGER.csv"))


@pytest.mark.parametrize("action,options,diagnostic", [
    ("prepare", [], "the following arguments are required: --request"),
    ("verify", [], "the following arguments are required: --request"),
    ("verify", ["--plan-ref", "unused"], "the following arguments are required: --request"),
    ("verify", ["--request", "unused"], "verify requires --plan-ref in addition to --request; prepare accepts no --plan-ref"),
    ("prepare", ["--request", "unused", "--plan-ref", "unused"],
     "verify requires --plan-ref in addition to --request; prepare accepts no --plan-ref"),
])
def test_cli_options_refuse_before_effects(action, options, diagnostic, monkeypatch, capsys):
    monkeypatch.setattr(cli, "connect", _forbid)
    monkeypatch.setattr(Path, "is_dir", _forbid)
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
    import sqlite3
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
    before = _catalog(source.conn), _objects(source.store)
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert not captured.err and json.loads(captured.out) == {"refused": expected}
    assert (_catalog(source.conn), _objects(source.store)) == before
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
