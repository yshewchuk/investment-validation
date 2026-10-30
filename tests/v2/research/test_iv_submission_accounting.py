"""Signed submissions cannot bypass durable native fit accounting."""
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


def test_completed_native_iv_submission_replays_without_labels_or_fits(search_case, monkeypatch):  # noqa: F811
    data = search_case
    trials, args, _ = context(data)
    result = trials.execute(*args, limits=data.plan["request"]["limits"]["child"])
    assert result["fit_attempt_count"] == result["completed_fit_count"] == 3
    key = content_hash([data.plan["plan_id"], *args])
    forecasts = read_record(data.conn, data.store, data.registration.run_id, "forecasts:" + key)
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    def forbidden(*args, **kwargs):
        pytest.fail("completed IV submission reopened targets or launched fitting")
    monkeypatch.setattr(trials_module, "evaluation_targets", forbidden)
    monkeypatch.setattr(trials_module, "run_fitted_trial", forbidden)
    with no_fit_guard(), legacy_guard():
        assert trials.record(*args, forecasts=forecasts) == result["terminal"]
    assert catalog_rows(data.conn) == before
    assert artifact_object_bytes(data.store) == artifacts
