"""Review regressions: closed wire input, actual source closure, explicit child jobs."""
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from experiments import size_search_binding as binding
from experiments.magnitude_search_request_versions import compile_request
from experiments.size_search_report import resource_report
from tests.v2.research.test_magnitude_confirmed_metadata import LIMITS as METADATA_LIMITS
from tests.v2.research.test_magnitude_search_request import request as search_request


def panel_request():
    return {"feature_snapshot_id": "snap_feature", "history_snapshot": "snap_history",
        "price_history_snapshot": "snap_prices", "as_of_month": "2025-01",
        "history_start": "2010-01-01", "source_start": "2015-01-01",
        "policy": eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30),
        "event_ids": ["event_one"], "horizons": [1, 3, 5],
        "limits": dict.fromkeys(["max_source_rows", "max_fragments", "max_object_bytes", "max_total_bytes", "max_output_rows"], 1000),
        "metadata_limits": dict(METADATA_LIMITS)}


def assert_invalid(value):
    with pytest.raises(OpsError) as error:
        binding.validate_panel_request(value)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert str(error.value) == "INVALID_EXPERIMENT_SPEC: invalid closed SIZE panel request"


@pytest.mark.parametrize("key", list(panel_request()))
def test_every_missing_or_malformed_panel_field_is_typed_before_source_work(monkeypatch, key):
    monkeypatch.setattr(binding, "prepare_native_registration", lambda *a, **k: pytest.fail("source read"))
    for bad in (None, [], {}, False, 3.5, ""):
        value = panel_request()
        value[key] = bad
        assert_invalid(value)
        with pytest.raises(OpsError) as error:
            binding.prepare(None, None, {}, value, scope="unused", training_start="2010-01-01", labels_before="2025-01-01")
        assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    value = panel_request()
    del value[key]
    assert_invalid(value)


@pytest.mark.parametrize("key,bad", [
    ("as_of_month", "2025-13"), ("as_of_month", "202501"), ("history_start", "2025-02-30"),
    ("source_start", "20250101"), ("event_ids", ["x", "x"]), ("event_ids", [None]),
    ("event_ids", [" "]), ("horizons", [True, 3, 5]), ("horizons", [1, 5, 3]),
    ("policy", eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=29)),
    ("policy", {**eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30), "embargo_days": 30.0}),
])
def test_panel_value_poison_is_typed(key, bad):
    value = panel_request()
    value[key] = bad
    assert_invalid(value)


@pytest.mark.parametrize("group", ["limits", "metadata_limits"])
@pytest.mark.parametrize("change", ["missing", "extra", "bool", "negative", "float"])
def test_closed_positive_limit_maps_refuse(group, change):
    value = panel_request()
    limits = value[group]
    first = next(iter(limits))
    if change == "missing":
        del limits[first]
    elif change == "extra":
        limits["caller_producer"] = 1
    else:
        limits[first] = {"bool": True, "negative": -1, "float": 2.5}[change]
    assert_invalid(value)


def test_valid_panel_request_is_not_rewritten():
    value = panel_request()
    before = deepcopy(value)
    assert binding.validate_panel_request(value) is value
    assert value == before
    value["producer"] = "caller"
    assert_invalid(value)


@pytest.mark.parametrize("action", ["preflight", "register"])
@pytest.mark.parametrize("bad", [{}, None, {"feature_snapshot_id": 3},
    {**panel_request(), "policy": {**panel_request()["policy"], "embargo_days": 30.0}}])
def test_cli_panel_refusal_precedes_catalog_store_or_containment(monkeypatch, capsys, action, bad):
    from experiments import size_search
    monkeypatch.setattr(size_search, "_read_document", lambda path: bad if path == "panel" else {})
    for name in ("connect", "Repository", "ArtifactStore", "_contain"):
        monkeypatch.setattr(size_search, name, lambda *a, **k: pytest.fail("source/write/containment"))
    status = size_search.main([action, "--spec", "spec", "--request", "request", "--catalog", "catalog",
        "--store-root", "store", "--no-ledger", "--panel-request", "panel", "--scope", "development",
        "--training-start", "2010-01-01", "--labels-before", "2025-01-01", "--receipt-ref", "receipt"])
    assert status == 2
    assert capsys.readouterr().out == '{"refused": "INVALID_EXPERIMENT_SPEC", "message": "INVALID_EXPERIMENT_SPEC: invalid closed SIZE panel request"}\n'


IMPLEMENTATION = ["experiments/size_search_binding.py", "experiments/size_comparator.py",
    "experiments/magnitude_fitted.py", "experiments/magnitude_candidate_fit.py",
    "experiments/size_search_report.py", "experiments/size_paired_report.py",
    "experiments/magnitude_search_worker.py", "engine/models/registry.json"]


@pytest.fixture(scope="module")
def code_copy(tmp_path_factory):
    root = Path(binding.__file__).resolve().parents[1]
    target = tmp_path_factory.mktemp("bounded-code-copy")
    sealed = binding._source_files()
    assert set(IMPLEMENTATION) <= set(sealed)
    for name in sealed:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, path)
    return target, sealed


@pytest.mark.parametrize("path", IMPLEMENTATION)
@pytest.mark.parametrize("fresh_process", [False, True])
def test_real_file_drift_refuses_sealed_replay_even_after_fresh_import(code_copy, monkeypatch, path, fresh_process):
    root, sealed = code_copy
    monkeypatch.setattr(binding, "__file__", str(root / "experiments/size_search_binding.py"))
    registered = {"schema_version": binding.SCHEMA, "source_closure": sealed,
        "environment": binding.environment_identity(1), "size_comparison": binding.comparison_recipe()}
    target = root / path
    original = target.read_bytes()
    try:
        target.write_bytes(original + (b"\n " if path.endswith(".json") else b"\n# deployment drift\n"))
        assert binding._source_files()[path] != sealed[path]
        if fresh_process:
            monkeypatch.setattr(binding, "_IMPORTED_SOURCES", binding._source_files())
        with pytest.raises(OpsError) as error:
            binding.verify_installation(registered)
        assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    finally:
        target.write_bytes(original)


def test_environment_drift_refuses_before_preparation_source_reads(monkeypatch):
    monkeypatch.setattr(binding, "environment_identity", lambda workers: {"changed": "runtime"})
    monkeypatch.setattr(binding, "prepare_native_registration", lambda *a, **k: pytest.fail("source read"))
    with pytest.raises(OpsError) as error:
        binding.prepare(None, None, {}, panel_request(), scope="unused",
                        training_start="2010-01-01", labels_before="2025-01-01")
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_603_search_fits_have_204_children_not_201_or_207():
    request = search_request()
    schedule = compile_request(request)
    assert schedule["counts"]["search_fits"] == 603
    result = resource_report(schedule)
    assert result["child_jobs"] == {"inner_candidates": 198, "outer_challengers": 3, "outer_comparators": 3}
    assert result["maximum_child_invocations"] == 204
    assert result["configured_child_wall_envelope_seconds"] == 204 * request["limits"]["child"]["wall_seconds"]


@pytest.mark.parametrize("anchor,request_version", [("1.6", "v1"), ("1.5", "v2"),
    ("1.4", "v2"), ("1.6", None), ("1.5", None), ("1.6", "v9"), ("1.5", "v9")])
def test_cross_version_plan_rejected_before_reads_or_publication(monkeypatch, anchor, request_version):
    from types import SimpleNamespace
    from experiments import magnitude_search_plan as planner
    registration = SimpleNamespace(document={"schema_version": "native_experiment_registration.v" + anchor})
    request = {} if request_version is None else {"schema_version": "magnitude_search_request." + request_version}
    monkeypatch.setattr(planner, "verify_native_registration", lambda *a: pytest.fail("read before version admission"))
    store = SimpleNamespace(publish_bytes=lambda *a, **k: pytest.fail("published cross-version plan"))
    with pytest.raises(OpsError) as caught:
        planner.seal_search_plan(None, store, registration, request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "search requires explicit valid axes, inner years and count caps"


def test_previous_native_validator_already_refused_size_anchor_as_legacy():
    from engine.v2.ops.native_registration import validate_magnitude_fitted_execution
    with pytest.raises(OpsError) as caught:
        validate_magnitude_fitted_execution({"schema_version": "native_experiment_registration.v1.6"})
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
