"""Signed comparator dispatch and native zero-retry evidence, without real datasets."""
from copy import deepcopy

import pytest

from engine.models.no_fit import RuntimeFitForbidden as LegacyFitForbidden
from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.models.no_fit import RuntimeFitForbidden, no_fit_guard
from engine.v2.ops.errors import OpsError, fail
from experiments import iv_crush_comparator as comparator
from experiments import iv_crush_comparison_run as policy
from experiments import magnitude_search_execution as worker
from experiments import size_comparison_run as journal
from experiments import v2_magnitude_search_run as driver
from experiments.iv_crush_plan import target
from experiments.magnitude_search_inputs import publish_inputs
from experiments.magnitude_search_records import read_record
from experiments.size_job_budget import remaining
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_comparator import case
from tests.v2.research.test_iv_crush_search_execution import search_case  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes


def numeric_case(n):
    indexed, training, fold, plan, _ = case(n)
    training["crush_pct_iv30"] = -140.
    plan.update(target=target(), comparison=comparator.recipe())
    inputs = {"features": sorted(indexed.values(), key=lambda row: (row["event_id"], row["horizon_sessions"])),
        "training": training.to_dict("records"), "target": target(),
        "preflight": {"schema_version": "research_iv_crush_preflight.v1", "event_ids": [], "planned_inputs": []},
        "training_inputs": {"consumed_inputs": [], "excluded_future_events": [], "holdout_exclusions": []}}
    return indexed, training, inputs, fold, plan


@pytest.mark.parametrize("n", [499, 500])
def test_real_fixed_child_signed_fit_abstention_and_exact_ack(tmp_path, n):
    _, _, inputs, fold, plan = numeric_case(n)
    store = ArtifactStore(tmp_path / "store")
    manifest, inputs_id = publish_inputs(store, inputs)
    markers = []
    result = worker.run_fitted_trial({"manifest": manifest, "inputs_id": inputs_id, "store_root": str(store.root)},
        fold, plan, policy.execution(plan), worker.execution_limits(
            {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 60}),
        lambda state, fit: markers.append((state, fit)))
    assert set(result) == {"event", "forecasts", "fit_details"}  # Untimed IV protocol stays closed.
    assert result["forecasts"] == [{"event_id": "test", "horizon_sessions": h,
        "predicted_crush_pct_iv30": -140. if n == 500 else None,
        "abstention": None if n == 500 else "insufficient_training", "training_mean": -140.} for h in (1, 3, 5)]
    assert all(row["logical_fits"] == row["component_fits"] == int(n == 500) for row in result["fit_details"])
    expected = comparator.fit_evidence(inputs, fold, plan)
    assert markers == [(state, fit) for fit in expected for state in ("started", "completed")]
    policy.validate_result(result, inputs, fold, plan)


@pytest.mark.parametrize("mutation", ["none", "list", "missing", "extra", "timing", "row_none",
    "forecasts_none", "duplicate", "wrong_mean", "wrong_reason", "prediction", "boolean", "fit_count"])
def test_result_refuses_shape_and_accounting_mutations(mutation):
    indexed, training, inputs, fold, plan = numeric_case(1)
    result = comparator.forecasts(indexed, training, fold, plan, policy.execution(plan), progress=lambda *a: None)
    if mutation == "none":
        result = None
    elif mutation == "list":
        result = []
    elif mutation == "missing":
        del result["fit_details"]
    elif mutation == "extra":
        result["unexpected"] = 0
    elif mutation == "timing":
        result.update(score_seconds=0.1, rss_delta_kib=0)
    elif mutation == "row_none":
        result["forecasts"][0] = None
    elif mutation == "forecasts_none":
        result["forecasts"] = None
    elif mutation == "duplicate":
        result["forecasts"][1] = deepcopy(result["forecasts"][0])
    elif mutation == "wrong_mean":
        result["forecasts"][0]["training_mean"] = 0.
    elif mutation == "wrong_reason":
        result["forecasts"][0]["abstention"] = "unavailable_input"
    elif mutation == "prediction":
        result["forecasts"][0]["predicted_crush_pct_iv30"] = -140.
    elif mutation == "boolean":
        result["forecasts"][0]["training_mean"] = True
    else:
        result["fit_details"][0]["component_fits"] = 1
    with pytest.raises(OpsError) as caught:
        policy.validate_result(result, inputs, fold, plan)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def invocation(data, *, seal_driver=True):
    args = (data.conn, data.store, data.repository, data.registration, data.plan["request"], data.ref)
    if seal_driver:
        driver.seal_search_run(*args, clock=SystemClock())
    else:
        driver.SearchTrials(*args).seal(clock=SystemClock())
    fold = {**data.plan["folds"][0]["outer"], "evaluation_labels_before": "2025-01-01"}
    return lambda **kw: journal.execute(*args[:4], data.plan, fold,
        limits=data.plan["request"]["limits"]["child"], **kw)


@pytest.mark.parametrize("guard,error", [(no_fit_guard, RuntimeFitForbidden), (legacy_guard, LegacyFitForbidden)])
def test_guards_precede_inputs_claims_and_job_deadline(search_case, monkeypatch, guard, error):  # noqa: F811
    data = search_case
    invoke = invocation(data)
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    monkeypatch.setattr(policy, "fold_inputs", lambda *a, **kw: pytest.fail("input admission before guard"))
    with guard(), pytest.raises(error):
        invoke()
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == artifacts


def test_fresh_admission_can_recover_but_claim_failure_cannot_retry(search_case, monkeypatch):  # noqa: F811
    data = search_case
    invoke = invocation(data)
    remaining(data.conn, data.store, data.registration, data.plan)
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    error = fail("RESOURCE_UNAVAILABLE", "synthetic admission failure")
    def refuse(*a, **kw):
        raise error
    with monkeypatch.context() as patch:
        patch.setattr(policy, "fold_inputs", refuse)
        for _ in range(2):
            with pytest.raises(OpsError) as caught:
                invoke()
            assert caught.value.problem == error.problem
        assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == artifacts
    monkeypatch.setattr(journal, "publish_inputs", refuse)
    with pytest.raises(OpsError) as caught:
        invoke()
    assert caught.value.problem == error.problem
    key = content_hash([data.plan["plan_id"], "2024", "comparison"])
    assert read_record(data.conn, data.store, data.registration.run_id, "comparison-start:" + key)
    monkeypatch.setattr(policy, "fold_inputs", lambda *a, **kw: pytest.fail("claimed inputs re-admitted"))
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with pytest.raises(OpsError) as caught:
        invoke()
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == artifacts


def test_native_terminal_replay_is_read_only_and_checks_all_markers(search_case, monkeypatch):  # noqa: F811
    data = search_case
    invoke = invocation(data)
    result = invoke()
    assert all(row["training_mean"] == -50. for row in result["forecasts"])
    key = content_hash([data.plan["plan_id"], "2024", "comparison"])
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    def forbidden(*a, **kw):
        pytest.fail("completed comparator reopened source inputs or launched a child")
    monkeypatch.setattr(policy, "fold_inputs", forbidden)
    monkeypatch.setattr(journal, "run_fitted_trial", forbidden)
    monkeypatch.setattr(journal, "resolved_limits", forbidden)
    with no_fit_guard(), legacy_guard():
        assert invoke(replay_only=True) == result
    original = journal.read_record
    for suffix in ("start", "started:5", "completed:1", "terminal"):
        def changed(conn, store, run_id, slot):
            value = deepcopy(original(conn, store, run_id, slot))
            if slot == "comparison-" + suffix + ":" + key:
                if suffix == "terminal":
                    value["result"]["forecasts"][0]["training_mean"] = 0.
                else:
                    value["unexpected"] = True
            return value
        with monkeypatch.context() as patch:
            patch.setattr(journal, "read_record", changed)
            with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
                invoke(replay_only=True)
            assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == artifacts


def test_driver_cannot_retroactively_seal_comparison_claim(search_case, monkeypatch):  # noqa: F811
    data = search_case
    invoke = invocation(data, seal_driver=False)
    def interrupted(*a, **kw):
        raise fail("RESOURCE_UNAVAILABLE", "synthetic publication interruption")
    monkeypatch.setattr(journal, "publish_inputs", interrupted)
    with pytest.raises(OpsError):
        invoke()
    args = (data.conn, data.store, data.repository, data.registration, data.plan["request"], data.ref)
    with pytest.raises(OpsError, match="already started"):
        driver.seal_search_run(*args, clock=SystemClock())


def test_competing_comparison_claim_cannot_publish_or_fit(search_case, monkeypatch):  # noqa: F811
    data = search_case
    invoke = invocation(data)
    original = journal.append_record
    claimed = []
    def competing(conn, store, run_id, slot, body, **kwargs):
        if slot.startswith("comparison-start:"):
            assert kwargs.get("replay") is False
            original(conn, store, run_id, slot, body, **kwargs)
            claimed.append(slot)
        return original(conn, store, run_id, slot, body, **kwargs)
    def forbidden(*a, **kw):
        pytest.fail("losing comparison claim published or fitted")
    monkeypatch.setattr(journal, "append_record", competing)
    monkeypatch.setattr(journal, "publish_inputs", forbidden)
    monkeypatch.setattr(journal, "run_fitted_trial", forbidden)
    with pytest.raises(OpsError, match="already committed"):
        invoke()
    assert len(claimed) == 1


def test_comparison_failure_prevents_outer_claim_and_evaluation_read(search_case, monkeypatch):  # noqa: F811
    from experiments import magnitude_search_outer as outer
    data = search_case
    invocation(data)
    candidate = data.plan["candidates"][0]["candidate_id"]
    selection = {"selections": [{"outer_year": "2024", "selected_candidate_id": candidate}],
                 "prospective_candidate_id": candidate}
    monkeypatch.setattr(outer, "_native_selection", lambda *a: selection)
    def failed(*a, **kw):
        raise fail("RESOURCE_UNAVAILABLE", "synthetic comparison interruption")
    def forbidden(*a, **kw):
        pytest.fail("outer claim, child or evaluation read followed comparison failure")
    monkeypatch.setattr(journal, "run_fitted_trial", failed)
    monkeypatch.setattr(outer, "evaluation_targets", forbidden)
    monkeypatch.setattr(outer, "publish_inputs", forbidden)
    monkeypatch.setattr(outer, "run_fitted_trial", forbidden)
    args = (data.conn, data.store, data.repository, data.registration, data.plan["request"], data.ref)
    with pytest.raises(OpsError, match="synthetic comparison"):
        outer.execute_outer_fold(*args, "2024", limits=data.plan["request"]["limits"]["child"])
    key = content_hash([data.plan["plan_id"], "2024"])
    assert read_record(data.conn, data.store, data.registration.run_id, "outer-start:" + key) is None
