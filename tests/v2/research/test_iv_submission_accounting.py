"""Signed submissions cannot bypass durable native fit accounting."""
from copy import deepcopy

import pytest

from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.contracts import Problem
from engine.v2.foundation import SystemClock, content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_trials as trials_module
from experiments.magnitude_search_records import read_record
from experiments.magnitude_search_trials import SearchTrials
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_search_execution import search_case  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes


def context(data):
    plan = data.plan
    trials = SearchTrials(data.conn, data.store, data.repository, data.registration,
                          plan["request"], data.ref)
    trials.seal(clock=SystemClock())
    group = plan["folds"][0]
    args = (group["outer"]["year"], group["inner"][0]["year"], plan["candidates"][0]["candidate_id"])
    forecasts = [{"event_id": event, "horizon_sessions": horizon, "predicted_crush_pct_iv30": -140.}
        for event in group["inner"][0]["evaluation_event_ids"] for horizon in plan["horizons"]]
    return trials, args, forecasts


@pytest.mark.parametrize("submission", ["forecasts", "failure"])
def test_manual_iv_submission_refuses_before_labels_or_publication(search_case, monkeypatch, submission):  # noqa: F811
    data = search_case
    trials, args, forecasts = context(data)
    trials.start(*args)
    key = content_hash([data.plan["plan_id"], *args])
    assert read_record(data.conn, data.store, data.registration.run_id, "execution-start:" + key) is None
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    def forbidden(*args, **kwargs):
        pytest.fail("manual IV submission opened evaluation targets")
    monkeypatch.setattr(trials_module, "evaluation_targets", forbidden)
    supplied = ({"forecasts": forecasts} if submission == "forecasts" else
                {"failure": fail("EXPERIMENT_VARIANT_FAILED", "synthetic failure").problem})
    with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
        trials.record(*args, **supplied)
    assert caught.value.problem == Problem(code="EXPERIMENT_IDENTITY_CONFLICT",
        category="validation", retryable=False,
        message="regression recording requires native execution accounting")
    assert catalog_rows(data.conn) == before
    assert artifact_object_bytes(data.store) == artifacts


def test_native_iv_reuses_validation_and_refuses_corruption_without_reload(search_case, monkeypatch):  # noqa: F811
    from experiments.magnitude_search_plan import cohort_key, trial_key
    from tests.v2.research.test_magnitude_search_validation_reuse import corrupt_stored_record
    data = search_case
    trials, args, _ = context(data)
    limits = data.plan["request"]["limits"]["child"]
    first = trials.execute(*args, limits=limits)["terminal"]
    def forbidden(*a, **k):
        pytest.fail("signed validation reuse reopened target payloads")
    monkeypatch.setattr(trials_module, "evaluation_targets", forbidden)
    candidates = data.plan["candidates"]
    second_args = (*args[:2], candidates[1]["candidate_id"])
    second = trials.execute(*second_args, limits=limits)["terminal"]
    for key in ("inputs_id", "validation_id", "evaluation_inputs", "predictions", "metrics"):
        assert second[key] == first[key]
    cohort = cohort_key(data.plan, *args[:2])
    assert data.conn.execute(
        "SELECT count(*) FROM magnitude_search_records WHERE slot LIKE 'validation:%'").fetchone()[0] == 1
    corrupt_stored_record(data, "validation:" + cohort)
    third_args = (*args[:2], candidates[2]["candidate_id"])
    with pytest.raises(OpsError) as caught:
        trials.execute(*third_args, limits=limits)
    assert caught.value.problem == fail(
        "EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified").problem
    assert trials._read("terminal:" + trial_key(data.plan, *third_args)) is None


def test_completed_native_iv_submission_replays_without_labels_or_fits(search_case, monkeypatch):  # noqa: F811
    data = search_case
    trials, args, _ = context(data)
    result = trials.execute(*args, limits=data.plan["request"]["limits"]["child"])
    assert result["fit_attempt_count"] == result["completed_fit_count"] == 3
    assert not {"score_seconds", "rss_delta_kib", "inner_metrics"} & result.keys()
    assert not {"score_seconds", "rss_delta_kib", "inner_metrics"} & result["terminal"].keys()
    key = content_hash([data.plan["plan_id"], *args])
    forecasts = read_record(data.conn, data.store, data.registration.run_id, "forecasts:" + key)
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    def forbidden(*args, **kwargs):
        pytest.fail("completed IV submission reopened targets or launched fitting")
    monkeypatch.setattr(trials_module, "evaluation_targets", forbidden)
    monkeypatch.setattr(trials_module, "run_fitted_trial", forbidden)
    with no_fit_guard(), legacy_guard():
        assert trials.record(*args, forecasts=forecasts) == result["terminal"]
        assert trials.reconcile(*args, limits=data.plan["request"]["limits"]["child"]) == result
    original_read = trials._read
    for prefix in ("execution-result:", "execution:", "fit-completed:"):
        def altered(slot):
            value = deepcopy(original_read(slot))
            if value is not None and slot.startswith(prefix):
                if prefix == "fit-completed:":
                    value["fit_seconds"] = 0.1
                else:
                    target = value["result"] if prefix == "execution-result:" else value
                    target.update(score_seconds=0.1, rss_delta_kib=0)
            return value
        with monkeypatch.context() as patch:
            patch.setattr(trials, "_read", altered)
            with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
                trials.execute(*args, limits=data.plan["request"]["limits"]["child"])
            assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert catalog_rows(data.conn) == before
    assert artifact_object_bytes(data.store) == artifacts


@pytest.mark.parametrize("kind", ["completed", "succeeded", "failed", "untimed_failure"])
def test_iv_worker_timing_contract_precedes_evaluation(search_case, monkeypatch, kind):  # noqa: F811
    from experiments.magnitude_search_evidence import fit_evidence
    from experiments.magnitude_search_inputs import load_inputs
    data = search_case
    trials, args, forecasts = context(data)
    def forbidden(*a, **k):
        pytest.fail("refused or failed IV fit opened evaluation targets")
    monkeypatch.setattr(trials_module, "evaluation_targets", forbidden)
    def child(source, inner, plan, execution, limits, progress):
        inputs = load_inputs(data.store, source["manifest"], source["inputs_id"])
        for fit in fit_evidence(inputs, inner, plan, execution):
            progress("started", fit)
            if kind in ("failed", "untimed_failure"):
                return {"event": "failed", **({} if kind == "untimed_failure" else
                        {"score_seconds": 0.1, "rss_delta_kib": 0})}
            progress("completed", {**fit, **({"fit_seconds": 0.1} if kind == "completed" else {})})
        return {"event": "succeeded", "forecasts": forecasts, "diagnostics": [],
                "score_seconds": 0.1, "rss_delta_kib": 0}
    monkeypatch.setattr(trials_module, "run_fitted_trial", child)
    limits = data.plan["request"]["limits"]["child"]
    if kind == "untimed_failure":
        result = trials.execute(*args, limits=limits)
        assert result["status"] == result["terminal"]["status"] == "failed"
        assert result["fit_attempt_count"] == 1 and result["completed_fit_count"] == 0
        monkeypatch.setattr(trials_module, "run_fitted_trial", forbidden)
        with no_fit_guard(), legacy_guard():
            assert trials.execute(*args, limits=limits) == result
            assert trials.reconcile(*args, limits=limits) == result
        original_read = trials._read
        def timed_failure(slot):
            value = deepcopy(original_read(slot))
            if value is not None and slot.startswith("execution-result:"):
                value["result"].update(score_seconds=0.1, rss_delta_kib=0)
            return value
        monkeypatch.setattr(trials, "_read", timed_failure)
        with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
            trials.reconcile(*args, limits=limits)
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    else:
        with pytest.raises(OpsError) as caught:
            trials.execute(*args, limits=limits)
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
        assert caught.value.problem.message == ("search fit progress differs from admitted inputs"
            if kind == "completed" else "search fit completion differs from admitted attempts")
        key = content_hash([data.plan["plan_id"], *args])
        for prefix in ("execution-result:", "execution:", "terminal:", "forecasts:"):
            assert trials._read(prefix + key) is None
