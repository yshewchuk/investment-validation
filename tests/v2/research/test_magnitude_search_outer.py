"""Native selected outer fits, temporal ordering and pooled independent oracles."""
import math
import shutil
from types import SimpleNamespace

import pytest

from engine.models.no_fit import RuntimeFitForbidden as LegacyRuntimeFitForbidden
from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.contracts import ArtifactRef
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, from_document
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_trials as trials
from experiments.magnitude_search_plan import seal_search_plan
from tests.v2.research.magnitude_trial_support import poison_outcomes, trial_cases
from tests.v2.research.test_magnitude_search_execution import LIMITS, inject_child
from tests.v2.research.test_magnitude_search_plan import source as base_source  # noqa: F401


@pytest.fixture(scope="module")
def native_outer_source(base_source, tmp_path_factory):  # noqa: F811
    path = tmp_path_factory.mktemp("outer-native")
    conn = connect(path / "catalog.sqlite")
    base_source.conn.backup(conn)
    shutil.copytree(base_source.store.root, path / "store")
    store = ArtifactStore(path / "store")
    request = {"schema_version": "magnitude_search_request.v1",
        "inner_years": {"2024": ["2023"], "2025": ["2023", "2024"]},
        "space": {"history_ridge": {"alpha": [0.5, 2]}}, "max_candidates": 2, "max_fits": 24}
    sealed = seal_search_plan(conn, store, base_source.registration, request)
    recorder = trials.SearchTrials(conn, store, Repository(conn, store), base_source.registration,
                                  request, from_document(ArtifactRef, sealed["plan_ref"]))
    recorder.seal(clock=SystemClock())
    case = SimpleNamespace(recorder=recorder, plan=sealed["plan"])
    for args in trial_cases(case):
        recorder.execute(*args, limits=LIMITS)
    selected = recorder.freeze()
    yield SimpleNamespace(**vars(case), conn=conn, store=store, request=request, selected=selected,
                          snapshot=base_source.snapshot)
    conn.close()


@pytest.fixture
def outer_case(native_outer_source, tmp_path):
    original = native_outer_source
    conn = connect(tmp_path / "catalog.sqlite")
    original.conn.backup(conn)
    shutil.copytree(original.store.root, tmp_path / "store")
    store = ArtifactStore(tmp_path / "store")
    recorder = trials.SearchTrials(conn, store, Repository(conn, store), original.recorder.registration,
                                  original.request, original.recorder.plan_ref)
    yield SimpleNamespace(**{**vars(original), "conn": conn, "store": store, "recorder": recorder})
    conn.close()


def run(case, year="2024", **kwargs):
    r = case.recorder
    return outer.execute_outer_fold(case.conn, case.store, r.repository, r.registration,
                                    r.request, r.plan_ref, year, limits=kwargs.get("limits", LIMITS))


def collect(case):
    r = case.recorder
    return outer.collect_outer_results(case.conn, case.store, r.repository, r.registration, r.request, r.plan_ref)


def slot(case, name, year="2024"):
    key = content_hash([case.plan["plan_id"], year])
    return case.recorder._read("outer-" + name + ":" + key)


def forbid_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("replay attempted fitting or native input/label access")
    for module in (outer, trials):
        monkeypatch.setattr(module, "run_fitted_trial", forbidden)
        monkeypatch.setattr(module, "load_magnitude_targets", forbidden)
    monkeypatch.setattr(outer, "_inputs", forbidden)
    monkeypatch.setattr(outer, "execution_limits", forbidden)


def test_real_outer_selected_only_forecasts_precede_labels_and_replay(outer_case, monkeypatch):
    case = outer_case
    poison_outcomes(monkeypatch, case.recorder.repository, case.snapshot, ("2023-04-03", "2025-12-03"))
    original = outer.load_magnitude_targets
    def labels(*args, **kwargs):
        assert slot(case, "forecasts") is not None
        assert slot(case, "execution")["completed_fit_count"] == 3
        assert kwargs["labels_before"] == "2025-02-01"
        assert not case.conn.in_transaction
        return original(*args, **kwargs)
    monkeypatch.setattr(outer, "load_magnitude_targets", labels)
    result = run(case)
    scores = case.selected["selections"][0]["scores"]
    assert scores[0]["metrics"] == scores[1]["metrics"]
    assert result["candidate"]["candidate_id"] == min(item["candidate_id"] for item in scores)
    assert result["candidate"]["candidate_id"] == case.selected["selections"][0]["selected_candidate_id"]
    assert result["prospective_candidate_id"] == case.selected["prospective_candidate_id"]
    assert result["evaluation_authority"] == "selected_outer_fold_only"
    assert result["final_read_authority"] == "none"
    assert result["event_count"] == 8 and result["forecast_count"] == 24
    assert result["training_inputs"]["labels_before"] == "2024-01-01"
    assert result["evaluation_inputs"]["labels_before"] == "2025-02-01"
    errors = [abs(row["predicted_absolute_move_pct"] - row["absolute_move_pct"]) for row in result["predictions"]]
    assert result["metrics"] == {"count": 24, "mae": pytest.approx(sum(errors) / 24),
                                 "rmse": pytest.approx(math.sqrt(sum(e * e for e in errors) / 24))}
    before = case.conn.total_changes
    forbid_work(monkeypatch)
    with no_fit_guard(), legacy_guard():
        assert run(case) == result
    assert case.conn.total_changes == before
    with pytest.raises(OpsError) as error:
        run(case, limits={**LIMITS, "cpu_seconds": 31})
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_collect_requires_all_folds_and_pools_errors_without_outer_selection(outer_case, monkeypatch):
    case = outer_case
    first = run(case)
    with pytest.raises(OpsError, match="every completed outer fold"):
        collect(case)
    assert slot(case, "start", "2025") is None
    second = run(case, "2025")
    forbid_work(monkeypatch)
    before = case.conn.total_changes
    with no_fit_guard(), legacy_guard():
        report = collect(case)
    assert report["folds"] == [first, second]
    assert report["event_count"] == 10
    errors = [abs(row["predicted_absolute_move_pct"] - row["absolute_move_pct"]) for row in report["predictions"]]
    expected = math.sqrt(sum(error ** 2 for error in errors) / len(errors))
    assert report["metrics"]["rmse"] == pytest.approx(expected)
    assert report["metrics"]["rmse"] != pytest.approx((first["metrics"]["rmse"] + second["metrics"]["rmse"]) / 2)
    for metrics in report["per_horizon"]:
        selected = [row for row in report["predictions"] if row["horizon_sessions"] == metrics["horizon_sessions"]]
        assert metrics["count"] == 10
        assert metrics["rmse"] == pytest.approx(math.sqrt(sum((row["predicted_absolute_move_pct"] - row["absolute_move_pct"]) ** 2
                                                             for row in selected) / 10))
    assert report["prospective_candidate_id"] == case.selected["selections"][-1]["selected_candidate_id"]
    assert case.conn.total_changes == before


@pytest.mark.parametrize("guard, exception", [(no_fit_guard, RuntimeFitForbidden), (legacy_guard, LegacyRuntimeFitForbidden)])
def test_no_fit_guards_before_outer_claim(outer_case, guard, exception):
    before = outer_case.conn.total_changes
    with guard(), pytest.raises(exception, match="execute_outer_fold"):
        run(outer_case)
    assert outer_case.conn.total_changes == before
    assert slot(outer_case, "start") is None


@pytest.mark.parametrize("stage", ["fit", "labels", "input"])
def test_outer_failure_never_replaces_candidate_or_retries(outer_case, monkeypatch, stage):
    case = outer_case
    if stage == "fit":
        inject_child(monkeypatch, "def injected(*args): raise ValueError('synthetic estimator failure')")
    else:
        def refused(*args, **kwargs):
            raise fail("RESOURCE_UNAVAILABLE", "synthetic shared failure")
        monkeypatch.setattr(outer, "load_magnitude_targets" if stage == "labels" else "_inputs", refused)
    with pytest.raises(OpsError) as error:
        run(case)
    assert error.value.code == ("EXPERIMENT_VARIANT_FAILED" if stage == "fit" else "RESOURCE_UNAVAILABLE")
    assert slot(case, "start") is not None and slot(case, "terminal") is None
    if stage == "fit":
        assert slot(case, "execution")["fit_attempt_count"] == 1
        assert slot(case, "execution")["completed_fit_count"] == 0
        assert slot(case, "forecasts") is None
    elif stage == "labels":
        assert slot(case, "forecasts") is not None
    monkeypatch.setattr(outer, "run_fitted_trial", lambda *a: pytest.fail("retried outer fit"))
    with pytest.raises(OpsError, match="already committed"):
        run(case)
    assert case.recorder.freeze() == case.selected


@pytest.mark.parametrize("year, limits", [("2026", LIMITS), (2024, LIMITS), ("2024", {}),
    ("2024", {**LIMITS, "wall_seconds": True})])
def test_invalid_request_never_claims(outer_case, year, limits):
    before = outer_case.conn.total_changes
    with pytest.raises(OpsError) as error:
        run(outer_case, year, limits=limits)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert outer_case.conn.total_changes == before
