"""regression_search_request.v2 compile, dispatch, binding by version (P1)."""
import json
from types import SimpleNamespace

import pytest

from engine.models.no_fit import no_fit_guard as legacy_no_fit
from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import content_hash, from_document, to_document
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_registration import regression_request_version
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments import iv_crush_run
from experiments import magnitude_search_plan as plans
from experiments import magnitude_search_request
from experiments import magnitude_search_request_v3
from experiments import native_registration as native
from experiments import size_search
from experiments import v2_magnitude_search_run
from experiments.magnitude_search_request_versions import compile_request
from experiments.magnitude_search_selection import CURRENT_TIE_RULE
from experiments.regression_search_binding import panel_coverage, stored_columns
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_registration import prepared  # noqa: F401
from tests.v2.research.test_iv_crush_search_plan import _preview_request, request as v1_request
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_magnitude_search_request import request as size_request
from tests.v2.research.test_native_registration import artifact_object_bytes
from tests.v2.research.test_size_review_contract import panel_request as wire_panel_request

_DELETE = object()


def _selection():
    return {"metric_weights": {"rmse": 0.67, "mae": 0.33},
            "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0}, "tie_band_floor": 0.005}


def v2_request(**over):
    value = size_request()
    panel = {**wire_panel_request(), "feature_snapshot_id": value["feature_snapshot_id"],
        "history_snapshot": value["history_snapshot_id"],
        "price_history_snapshot": "snap_" + "4" * 32, "horizons": [1]}
    value.update(schema_version="regression_search_request.v2", panel_request=panel,
        feature_dictionary_hash=recipe("HMOSXLP")["dictionary_hash"],
        feature_sets=["HMO", "HMOS", "HMOSX", "HMOSXL", "HMOSXLP"],
        space={"ridge": {"alpha": [0.1, 1.0, 10.0]},
               "elasticnet": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.25, 0.75]},
               "hgb": {"max_leaf_nodes": [7, 15]}},
        selection=_selection(), max_candidates=55, max_fits=2000,
        comparison={"reference": "iv_crush_causal_refit", "sanity_reference": "training_mean",
                    "max_logical_fits": 9, "max_component_fits": 2000})
    value.update(over)
    return value


def _put(document, path, value):
    target = document
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value


def refuses(request):
    with pytest.raises(OpsError) as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False


def test_v2_compiles_five_sets_with_counts_and_specs():
    result = compile_request(v2_request())
    counts = result["counts"]
    assert result["schema_version"] == "regression_search_schedule.v1"
    assert result["horizons"] == [1]
    assert counts == {"candidates": 55, "inner_folds": 6, "outer_folds": 3,
        "search_fits": 333, "baseline_logical_fits": 3, "baseline_component_fits": 3,
        "total_logical_fits": 336, "total_component_fits": 336}
    assert {spec["feature_recipe"]["subset"] for spec in result["candidate_specs"]} == {
        "HMO", "HMOS", "HMOSX", "HMOSXL", "HMOSXLP"}
    for spec in result["candidate_specs"]:
        subset = spec["feature_recipe"]["subset"]
        assert spec["feature_recipe"]["dictionary_hash"] == recipe(subset)["dictionary_hash"]
    assert result["request"]["feature_dictionary_hash"] == recipe("HMOSXLP")["dictionary_hash"]
    assert result["request_id"] == content_hash(result["request"])
    assert result["request"]["selection"]["tie_band_floor"] == 0.005


@pytest.mark.parametrize("workers", [1, 8])
def test_workers_are_not_identity_and_default_equals_omitted(workers):
    baseline = compile_request(v2_request())
    document = v2_request(workers=workers)
    result = compile_request(document)
    assert result["request_id"] == baseline["request_id"]
    assert result["candidate_specs"] == baseline["candidate_specs"]
    assert "workers" not in result["request"]
    assert json.dumps(result) == json.dumps(compile_request(document))


def test_explicit_lead_selector_and_panel_horizon_derivation():
    document = v2_request()
    assert compile_request(document, horizons=(1,)) == compile_request(document)
    assert compile_request(document, horizons=(5, 3, 1)) == compile_request(document)
    for horizons in ((3,), (1,)):
        request = v2_request()
        if horizons == (1,):
            request["panel_request"]["horizons"] = [1, 3, 5]
        with pytest.raises(OpsError) as caught:
            compile_request(request, horizons=horizons)
        assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    late = v2_request()
    late["panel_request"]["horizons"] = [3, 5]
    with pytest.raises(OpsError) as caught:
        compile_request(late)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("path,value", [
    (("selection",), _DELETE), (("selection", "large_move_threshold_pct"), 10.0),
    (("selection", "metric_weights", "large_move_underprediction"), 0.1),
    (("selection", "metric_weights", "rmse"), 0.7),
    (("selection", "lead_weights"), {"1": 0.5, "3": 0.5, "5": 0.0}),
    (("selection", "tie_band_floor"), -1), (("selection", "tie_band_floor"), True),
    (("selection", "tie_band_floor"), "x"), (("selection", "tie_band_floor"), float("inf")),
    (("feature_sets",), ["H"]), (("feature_sets",), ["HM", "HMO"]),
    (("feature_sets",), ["HMOS", "HMO"]), (("feature_sets",), ["HMO", "HMO"]),
    (("feature_sets",), []), (("feature_dictionary_hash",), recipe("HMO")["dictionary_hash"]),
    (("space", "ridge", "alpha"), [10.0, 1.0]), (("space", "ridge", "alpha"), [0.1, 0.1]),
    (("space", "ridge", "alpha"), [1e-9]), (("space", "ridge", "alpha"), [20000.0]),
    (("space", "ridge", "alpha"), ["0.1"]), (("space", "ridge", "alpha"), [True]),
    (("space", "elasticnet", "l1_ratio"), [2.0]), (("space", "hgb", "max_leaf_nodes"), [7.5]),
    (("space", "hgb", "max_leaf_nodes"), [1]), (("space", "hgb", "max_leaf_nodes"), [65]),
    (("space", "hgb", "max_leaf_nodes"), [8]), (("space", "hgb", "max_leaf_nodes"), [7, 16]),
    (("space", "hgb"), _DELETE), (("space", "hgb", "loss"), ["squared_error"]),
    (("space", "elasticnet", "spline_knots"), [0]),
    (("workers",), 0), (("workers",), 9), (("workers",), True), (("workers",), "2"),
    (("extra",), 1), (("schema_version",), "regression_search_request.v3"),
    (("max_candidates",), 54), (("max_fits",), 332),
    (("comparison", "max_component_fits"), 335),
    (("comparison", "reference"), "size_v1_4_causal_refit"),
    (("limits", "child", "memory_bytes"), 5),
    (("panel_request", "feature_snapshot_id"), "snap_" + "a" * 32),
    (("panel_request", "price_history_snapshot"), "snap_bad"), (("max_fits",), float("inf")),
])
def test_refusals(path, value):
    document = v2_request()
    _put(document, path, value)
    refuses(document)


@pytest.mark.parametrize("path,value", [
    (("selection", "large_move_threshold_pct"), 10.0),
    (("feature_sets",), ["H"]), (("space", "hgb"), _DELETE)])
def test_refusals_precede_candidate_expansion(monkeypatch, path, value):
    document = v2_request()
    _put(document, path, value)

    def no_expansion(*args, **kwargs):
        pytest.fail("invalid regression v2 request reached candidate expansion")

    monkeypatch.setattr(magnitude_search_request, "_specifications", no_expansion)
    refuses(document)


def test_v1_and_magnitude_compilers_never_accept_v2_and_vice_versa():
    document = v2_request()
    for compiler in (magnitude_search_request.compile_request,
                     magnitude_search_request_v3.compile_request):
        with pytest.raises(OpsError) as caught:
            compiler(document)
        assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    legacy = _preview_request()
    legacy["selection"] = _selection()
    with pytest.raises(OpsError) as caught:
        compile_request(legacy)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False


@pytest.mark.parametrize("value,expected", [
    ("regression_search_request.v1", 1), ("regression_search_request.v2", 2),
    ("regression_search_request.v10", 10), ("regression_search_request.v0", None),
    ("regression_search_request.v01", None), ("magnitude_search_request.v2", None),
    (None, None), (3, None), ("", None)])
def test_regression_request_version(value, expected):
    assert regression_request_version(value) == expected


def test_stored_columns_by_version():
    def columns(feature_sets):
        return stored_columns({"schema_version": "regression_search_request.v2",
                               "feature_sets": feature_sets})

    assert stored_columns({"schema_version": "regression_search_request.v1"}) == recipe("HMO")["model_columns"]
    assert columns(["HMO"]) == recipe("HMO")["model_columns"]
    assert columns(["HMO", "HMOS"]) == recipe("HMOS")["model_columns"]
    for feature_sets in (["HMOS", "HMOSXL"], ["HMOSXLP"],
                         ["HMO", "HMOS", "HMOSX", "HMOSXL", "HMOSXLP"]):
        assert columns(feature_sets) == recipe("HMOSX")["model_columns"]
    widest = columns(["HMOSXLP"])
    assert not any(name.startswith("log1p_") or "_x_" in name for name in widest)
    assert panel_coverage(["a", "b", "c"], ["b", "c", "d"]) == {
        "iv_events": 3, "in_size_panel": 2, "outside": 1}
    assert panel_coverage([], []) == {"iv_events": 0, "in_size_panel": 0, "outside": 0}


def _v2_plan_request(data):
    document = v1_request(data)
    document.update(schema_version="regression_search_request.v2",
        feature_dictionary_hash=recipe("HMOSXLP")["dictionary_hash"],
        feature_sets=["HMO", "HMOS"], selection=_selection(), max_candidates=22, max_fits=23)
    document["comparison"].update(max_logical_fits=1, max_component_fits=24)
    document["panel_request"]["horizons"] = [1]
    return document


def _seal(data, registered, document):
    with no_fit_guard(), legacy_no_fit():
        return plans.seal_search_plan(data.conn, data.store, registered, document)


def test_binding_on_real_plan_path(prepared, capsys):  # noqa: F811
    data = prepared
    registered = native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    document = _v2_plan_request(data)
    v1_plan = _seal(data, registered, v1_request(data))["plan"]
    sealed = _seal(data, registered, document)
    plan = sealed["plan"]
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    with no_fit_guard(), legacy_no_fit():
        with pytest.raises(OpsError) as refused:
            plans.read_search_plan(data.conn, data.store, registered, document, ref)
    assert refused.value.code == "INVALID_EXPERIMENT_SPEC"
    assert _seal(data, registered, document) == sealed
    assert v1_plan["selection"]["metric"] == "inner_pooled_rmse"
    assert plan["selection"]["metric"] == "blend_rmse_mae"
    assert plan["selection"]["ties"] == CURRENT_TIE_RULE
    assert plan["horizons"] == [1] and plan["counts"]["candidates"] == 22
    assert {candidate["feature_recipe"]["subset"] for candidate in plan["candidates"]} == {"HMO", "HMOS"}
    assert all(candidate["recipe"]["target_domain"] == "signed" for candidate in plan["candidates"])
    ids = list(registered.document["event_ids"])
    superset = _v2_plan_request(data)
    superset["panel_request"]["event_ids"] = ids + ["extra_event"]
    _seal(data, registered, superset)
    expected = len(set(ids))
    assert f"panel_coverage iv_events={expected} in_size_panel={expected} outside=0" in capsys.readouterr().err
    outside = _v2_plan_request(data)
    outside["panel_request"]["event_ids"] = ["only_other_event"]
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with pytest.raises(OpsError) as caught:
        _seal(data, registered, outside)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert len(set(ids)) > 0 and f"outside={len(set(ids))}" in capsys.readouterr().err
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


def test_read_search_plan_refuses_v2_before_any_read():
    with pytest.raises(OpsError) as refused:
        plans.read_search_plan(None, None, None, v2_request(), None)
    assert refused.value.code == "INVALID_EXPERIMENT_SPEC"
    assert "awaits its signed selection scorer" in refused.value.problem.message


def test_v2_requests_refused_by_size_cli_and_v2_run_cli(tmp_path, capsys):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(v2_request()))
    common = ["--request", str(request_path), "--catalog", str(tmp_path / "catalog.sqlite"),
              "--store-root", str(tmp_path / "store"), "--no-ledger"]
    assert size_search.main(["report", "--spec", "spec", "--plan-ref", "ref", *common]) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INVALID_EXPERIMENT_SPEC",
        "message": "INVALID_EXPERIMENT_SPEC: IV requires its dedicated CLI; use the admitted library path"}
    (tmp_path / "store").mkdir()
    (tmp_path / "spec.json").write_text(json.dumps({"hypothesis": "synthetic",
        "primary_arm_id": "candidate", "economic_params": {}, "price_source": "iv_crush_targets",
        "runner": "experiments/iv_crush_search.py"}))
    (tmp_path / "ref.json").write_text(json.dumps(to_document(ArtifactRef(artifact_id="a",
        content_hash="sha256:" + "0" * 64, schema_ref="regression_search_plan.v1", byte_size=1,
        storage_key="objects/00/" + "0" * 64))))
    assert v2_magnitude_search_run.main(["seal", "--spec", str(tmp_path / "spec.json"),
        "--plan-ref", str(tmp_path / "ref.json"), *common]) == 2
    refusal = json.loads(capsys.readouterr().out)["refused"]
    assert refusal["code"] == "INVALID_EXPERIMENT_SPEC"
    assert refusal["message"] == "IV requires dedicated CLI containment; use the admitted library path"


@pytest.mark.parametrize("action", ["seal", "run"])
def test_iv_cli_refuses_v2_seal_and_run_before_any_fit(tmp_path, capsys, action):
    (tmp_path / "request.json").write_text(json.dumps(v2_request()))
    (tmp_path / "spec.json").write_text(json.dumps({}))
    (tmp_path / "ref.json").write_text(json.dumps(to_document(ArtifactRef(artifact_id="a",
        content_hash="sha256:" + "0" * 64, schema_ref="regression_search_plan.v1", byte_size=1,
        storage_key="objects/00/" + "0" * 64))))
    (tmp_path / "store").mkdir()
    assert iv_crush_run.main([action, "--spec", str(tmp_path / "spec.json"),
        "--request", str(tmp_path / "request.json"), "--catalog", str(tmp_path / "catalog.sqlite"),
        "--store-root", str(tmp_path / "store"), "--plan-ref", str(tmp_path / "ref.json"),
        "--no-ledger"]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert refusal["message"] == "INVALID_EXPERIMENT_SPEC: regression v2 execution awaits its signed selection scorer"
    assert not (tmp_path / "catalog.sqlite").exists()


def test_panel_coverage_helper_is_v2_only():
    registration = SimpleNamespace(document={"event_ids": ["a", "b"]})
    request = v2_request(
        panel_request={**v2_request()["panel_request"], "event_ids": ["b", "c"]})
    assert iv_crush_run._panel_coverage(registration, request) == {
        "iv_events": 2, "in_size_panel": 1, "outside": 1}
    assert iv_crush_run._panel_coverage(
        registration, {"schema_version": "regression_search_request.v1"}) is None


def test_register_refuses_coverage_shortfall_before_reserving(tmp_path, monkeypatch, capsys):
    request = v2_request(
        panel_request={**v2_request()["panel_request"], "event_ids": ["b", "c"]})
    request_path, spec_path = tmp_path / "request.json", tmp_path / "spec.json"
    (tmp_path / "store").mkdir()

    def read_input(path):
        return request if path == str(request_path) else {}

    def registration(*args, **kwargs):
        return SimpleNamespace(document={"event_ids": ["a"]}, run_id="r", variant_id="v"), "spec"

    monkeypatch.setattr(iv_crush_run, "_input", read_input)
    monkeypatch.setattr(iv_crush_run, "experiment_spec_from_document", lambda document: SimpleNamespace(
        runner="experiments/iv_crush_search.py", price_source="iv_crush_targets"))
    monkeypatch.setattr(iv_crush_run, "connect", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(iv_crush_run, "ArtifactStore", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(iv_crush_run, "Repository", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(iv_crush_run, "_prepare", registration)
    monkeypatch.setattr(iv_crush_run, "reserve_native_registration",
        lambda *args, **kwargs: pytest.fail("reserved"))
    monkeypatch.setattr(iv_crush_run, "seal_search_plan", lambda *args, **kwargs: pytest.fail("sealed"))
    assert iv_crush_run.main(["register", "--spec", str(spec_path), "--request", str(request_path),
        "--catalog", str(tmp_path / "catalog.sqlite"), "--store-root", str(tmp_path / "store"),
        "--scope", "scope", "--target-snapshot-id", "target", "--training-start", "2020-01",
        "--labels-before", "2021-01", "--no-ledger"]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["refused"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert refusal["message"] == "EXPERIMENT_IDENTITY_CONFLICT: regression search differs from its native binding"
