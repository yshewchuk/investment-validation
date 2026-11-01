"""Synthetic native SIZE fallback lifecycle; storage/selection stay real, rivals are numeric fakes."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import SystemClock, content_hash, from_document
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_plan as planner
from experiments import magnitude_search_trials as trials
from experiments import size_comparison_run as comparison
from experiments import size_search_binding as binding
from experiments.magnitude_search_evidence import fit_evidence
from experiments.magnitude_search_inputs import load_fit_inputs
from experiments.native_registration import reserve_native_registration
from experiments.size_search_report import collect
from experiments.v2_magnitude_search_run import run_search, seal_search_run
from tests.v2.research.magnitude_trial_support import poison_outcomes, prediction_spec
from tests.v2.research.test_magnitude_search_request_v5 import v5_request
from tests.v2.research.test_size_search_integration import lead_one_case  # noqa: F401

pytestmark = pytest.mark.xdist_group("size_search_integration")


@pytest.fixture
def candidate_case(request):
    base = request.getfixturevalue("lead_one_case")
    document = v5_request()
    for key in ("feature_dictionary_hash", "feature_snapshot_id", "history_snapshot_id",
                "eligibility_policy_ref", "outer_years", "inner_years", "limits"):
        document[key] = deepcopy(base.request[key])
    # A legal small declared grid, not a truncated plan or relaxed execution cap.
    for family, axes in document["space"].items():
        document["space"][family] = {name: values[:1] for name, values in axes.items()}
    document.update(max_candidates=10, max_fits=11)
    spec = prediction_spec(experiment_id=request.node.name, folds=("2024",),
                           runner="experiments/size_search.py")
    registration, spec = binding.prepare(base.repository, spec, document, base.panel_request, **base.args)
    reserve_native_registration(base.conn, base.store, registration, spec)
    sealed = planner.seal_search_plan(base.conn, base.store, registration, document)
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    context = (base.conn, base.store, base.repository, registration, document, ref)
    seal_search_run(*context, clock=SystemClock())
    return SimpleNamespace(**{**vars(base), "registration": registration, "request": document,
        "context": context, "plan": sealed["plan"], "ref": ref,
        "recorder": trials.SearchTrials(*context)})


def _forecasts(fold, value):
    return [{"event_id": event, "horizon_sessions": 1, "predicted_absolute_move_pct": value}
            for event in sorted(fold["evaluation_event_ids"])]


def _generic_result(inputs, fold, plan, execution, progress):
    """Cheap rivals with explicit valid diagnostics, never fake storage or selection."""
    recipe = execution["fit_recipe"]
    elastic = recipe["model"] == "elasticnet"
    diagnostics = []
    for fit in fit_evidence(inputs, fold, plan, execution):
        progress("started", fit)
        progress("completed", {**fit, "fit_seconds": 0.0})
        detail = {"schema_version": "magnitude_" + ("elasticnet" if elastic else "ridge_hgb") + "_fit.v1",
                  "recipe_id": content_hash(recipe), "horizon_sessions": fit["horizon_sessions"]}
        if elastic:
            detail.update(coefficients=[{"kind": kind, "column": name, "coefficient": 0.0}
                for kind in ("value", "missing_indicator") for name in recipe["features"]],
                coefficient_scale="standardized_inputs", selected_columns=[], intercept=1000.0,
                converged=True, n_iter=1, dual_gap=0.0)
        diagnostics.append(detail)
    return {"event": "succeeded", "forecasts": _forecasts(fold, 1000.0),
            "score_seconds": 0.0, "rss_delta_kib": 0, "diagnostics": diagnostics}


def test_native_fallback_records_reconciles_selects_and_reports_same_grid(candidate_case, monkeypatch):
    case, recorder = candidate_case, candidate_case.recorder
    plan, limits = case.plan, case.request["limits"]["child"]
    group = plan["folds"][0]
    size = next(c for c in plan["candidates"] if c["recipe"]["schema_version"] == "size_history_mean_candidate.v1")
    args = ("2024", "2023", size["candidate_id"])
    key = planner.trial_key(plan, *args)
    assert plan["candidate_count"] == 10 and plan["counts"]["total_component_fits"] == 15
    assert size["recipe"]["missing"] == "history_mean" and size["recipe"]["prediction_domain"] == "signed"
    poison_outcomes(monkeypatch, case.repository, case.panel_source,
                    ["2023-03-20", "2023-04-03", "2023-05-15"])
    jobs = {"native_size": 0, "numeric_rivals": 0, "native_comparator": 0}
    native = trials.run_fitted_trial

    def worker(source, fold, plan, execution, resolved, progress):
        assert set(source) == {"manifest", "inputs_id", "store_root"}
        inputs = load_fit_inputs(case.store, source["manifest"], source["inputs_id"],
            fold["evaluation_event_ids"], plan["horizons"], execution["features"])
        assert set(inputs) == {"training", "features"}
        assert {r["event_id"] for r in inputs["training"]}.isdisjoint(fold["evaluation_event_ids"])
        assert all("absolute_move_pct" not in row for row in inputs["features"])
        if execution.get("model") == "size_v1_4":
            jobs["native_size"] += 1
            assert all(row["n_prior"] < 4 for row in inputs["features"])
            return native(source, fold, plan, execution, resolved, progress)
        if execution["schema_version"] == "size_comparison_execution.v1":
            jobs["native_comparator"] += 1
            return native(source, fold, plan, execution, resolved, progress)
        jobs["numeric_rivals"] += 1
        return _generic_result(inputs, fold, plan, execution, progress)

    for module in (trials, outer, comparison):
        monkeypatch.setattr(module, "run_fitted_trial", worker)
    # Crash after durable child result/accounting, before forecast/label publication.
    def interrupt_publication(*a, **k):
        raise OSError("publication stopped")

    with monkeypatch.context() as patch:
        patch.setattr(recorder, "_record", interrupt_publication)
        with pytest.raises(OSError, match="publication stopped"):
            recorder.execute(*args, limits=limits)
    assert recorder._read("execution-result:" + key)["result"]["event"] == "succeeded"
    assert recorder._read("terminal:" + key) is None
    assert recorder._read("forecasts:" + key) is None
    admitted = recorder._read("inputs:" + key)
    assert recorder._read("validation:" + admitted["cohort"]) is None
    receipt = recorder.reconcile(*args, limits=limits)
    assert jobs["native_size"] == 1
    assert sorted(r["predicted_absolute_move_pct"] for r in receipt["terminal"]["predictions"]) == [2.0, 3.0]
    assert receipt["diagnostics"][0]["fallback_counts"] == {"insufficient_history": 2}
    assert receipt["diagnostics"][0]["component_fits"] == 0
    inner_keys = {(e, 1) for e in group["inner"][0]["evaluation_event_ids"]}
    for candidate in plan["candidates"]:
        result = recorder.execute("2024", "2023", candidate["candidate_id"], limits=limits)
        assert {(r["event_id"], r["horizon_sessions"]) for r in result["terminal"]["predictions"]} == inner_keys
    selection = recorder.freeze()
    assert selection["selections"][0]["selected_candidate_id"] == size["candidate_id"]
    assert {score["metrics"]["count"] for score in selection["selections"][0]["scores"]} == {2}
    outer_result = outer.execute_outer_fold(*case.context, "2024", limits=limits)
    assert [r["predicted_absolute_move_pct"] for r in outer_result["predictions"]] == [4.0]
    report = collect(*case.context)
    assert report["paired"]["v1_4_fallbacks"] == [{"outer_year": "2024", "horizon_sessions": 1,
        "fallback_counts": {"insufficient_history": 1}}]
    assert report["paired"]["challenger_abstentions"] == 0
    assert report["paired"]["broad_coverage_metrics"]["challenger"]["count"] == 1
    assert report["paired"]["comparison_status"] == "unavailable_no_complete_matched_events"
    assert jobs == {"native_size": 2, "numeric_rivals": 9, "native_comparator": 1}
    slots = [row[0] for row in case.conn.execute(
        "SELECT slot FROM magnitude_search_records WHERE run_id = ?", (case.registration.run_id,))]
    assert not any(slot.startswith("forecast-submission:") for slot in slots)

    def no_fit(*a, **k):
        pytest.fail("completed native lifecycle launched another fit")
    for module in (trials, outer, comparison):
        monkeypatch.setattr(module, "run_fitted_trial", no_fit)
    before = case.conn.total_changes
    assert recorder.execute(*args, limits=limits) == receipt
    assert recorder.reconcile(*args, limits=limits) == receipt
    assert run_search(*case.context, limits=limits) == report["search"]
    assert collect(*case.context) == report
    assert case.conn.total_changes == before
    original_read = recorder._read
    for mutation in ("missing", "none", "changed"):
        def altered(slot):
            value = original_read(slot)
            if slot == "execution-result:" + key:
                value = deepcopy(value)
                rows = value["result"]["forecasts"]
                if mutation == "missing":
                    rows.pop()
                else:
                    rows[0]["predicted_absolute_move_pct"] = None if mutation == "none" else 777.0
            return value
        with monkeypatch.context() as patch:
            patch.setattr(recorder, "_read", altered)
            with pytest.raises(OpsError) as caught:
                recorder.reconcile(*args, limits=limits)
            assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert case.conn.total_changes == before


def test_signed_size_submissions_require_full_grid_and_cannot_claim_native_authority(candidate_case, monkeypatch):
    case, recorder = candidate_case, candidate_case.recorder
    plan, fold = case.plan, case.plan["folds"][0]["inner"][0]
    size = next(c for c in plan["candidates"] if c["recipe"]["schema_version"] == "size_history_mean_candidate.v1")
    args = ("2024", "2023", size["candidate_id"])
    key = planner.trial_key(plan, *args)
    recorder.start(*args)
    negative = _forecasts(fold, -0.25)
    for invalid in (negative[:-1], [{**negative[0], "predicted_absolute_move_pct": None}, *negative[1:]]):
        with pytest.raises(OpsError) as caught:
            recorder.record(*args, forecasts=invalid)
        assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
        assert recorder._read("forecasts:" + key) is None
    terminal = recorder.record(*args, forecasts=negative)
    assert [r["predicted_absolute_move_pct"] for r in terminal["predictions"]] == [-0.25, -0.25]
    assert recorder.record(*args, forecasts=negative) == terminal
    with pytest.raises(OpsError) as caught:
        recorder.record(*args, forecasts=_forecasts(fold, -0.5))
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    for candidate in plan["candidates"]:
        if candidate != size:
            other = ("2024", "2023", candidate["candidate_id"])
            recorder.start(*other)
            recorder.record(*other, forecasts=_forecasts(fold, 1000.0))
    assert recorder.freeze()["prospective_candidate_id"] == size["candidate_id"]
    monkeypatch.setattr(outer, "run_fitted_trial", lambda *a, **k: pytest.fail("submitted forecasts claimed native work"))
    with pytest.raises(OpsError) as caught:
        outer.execute_outer_fold(*case.context, "2024", limits=case.request["limits"]["child"])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert "completed native inner trials" in caught.value.problem.message
