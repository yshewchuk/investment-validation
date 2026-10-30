"""Complete tiny native IV search with protected-byte ordering and no-refit replay."""
import json
from copy import deepcopy
from statistics import mean

import pytest

from engine.models.no_fit import RuntimeFitForbidden as LegacyFitForbidden
from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.contracts import ArtifactRef
from engine.v2.data import objects
from engine.v2.foundation import SystemClock, content_hash, from_document
from engine.v2.models.no_fit import RuntimeFitForbidden, no_fit_guard
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_execution as execution
from experiments import magnitude_search_plan as plans
from experiments import v2_magnitude_search_run as driver
from experiments.magnitude_search_records import read_record
from tests.v2.research.magnitude_trial_support import catalog_rows, poison_outcomes
from tests.v2.research.test_iv_crush_fold_inputs import fold_case
from tests.v2.research.test_iv_crush_inputs import target
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_magnitude_search_driver_cli import _invoke, source  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes

VALUES = {"2022-06-02": -140., "2023-01-03": -20., "2023-04-03": 40.,
          "2023-06-02": 10., "2024-06-17": -60.}


@pytest.fixture
def search_case(panel_data, monkeypatch, request):  # noqa: F811
    with monkeypatch.context() as setup:
        setup.setattr("tests.v2.research.test_iv_crush_fold_inputs.target",
            lambda event, **changes: target(event,
                value=VALUES.get(event["event_date"].date().isoformat(), -40.)))
        data = fold_case.__wrapped__(panel_data, setup, request)
    document = deepcopy(data.plan["request"])
    document["limits"] = {"child": {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 60},
        "parent": {"memory_bytes": 3 << 30, "wall_seconds": 1800},
        "build": {"memory_bytes": 1 << 30, "wall_seconds": 60}, "job": {"wall_seconds": 1800}}
    sealed = plans.seal_search_plan(data.conn, data.store, data.registration, document)
    data.plan, data.ref = sealed["plan"], from_document(ArtifactRef, sealed["plan_ref"])
    poison_outcomes(monkeypatch, data.repository, data.panel_source, (), all_outcomes=True)
    return data


def test_all_candidates_selected_outer_and_read_only_replay(search_case, monkeypatch):
    data = search_case
    plan, run_id = data.plan, data.registration.run_id
    args = (data.conn, data.store, data.repository, data.registration, plan["request"], data.ref)
    driver.seal_search_run(*args, clock=SystemClock())
    def read(slot):
        return read_record(data.conn, data.store, run_id, slot)
    group = plan["folds"][0]
    outer_key = content_hash([plan["plan_id"], "2024"])
    allowed = set(data.registration.document["event_ids"])
    by_day = {event["event_date"].date().isoformat(): event["event_id"] for event in data.events}
    target_ids = {record.object_ref.object_id: by_day[record.time_min]
        for record in data.repository.fragment_records(data.targets, "iv_crush_targets")}
    opened = []
    original = objects.read_verified_bytes

    def guarded(store, ref, *, max_bytes):
        event = target_ids.get(ref.object_id)
        if event is not None:
            assert event in allowed, "excluded IV target bytes opened"
            if event in group["outer"]["evaluation_event_ids"]:
                selection = read("selection")
                assert selection is not None and len(selection["attempts"]) == 33
                assert read("outer-forecasts:" + outer_key) is not None
                assert read("outer-execution:" + outer_key)["completed_fit_count"] == 3
            elif event in group["inner"][0]["evaluation_event_ids"]:
                assert any(read("forecasts:" + content_hash([plan["plan_id"], "2024", "2023", c["candidate_id"]]))
                           is not None for c in plan["candidates"]), "inner labels preceded durable forecasts"
            opened.append(event)
        return original(store, ref, max_bytes=max_bytes)

    monkeypatch.setattr(objects, "read_verified_bytes", guarded)
    result = driver.run_search(*args, limits=plan["request"]["limits"]["child"])
    assert result["schema_version"] == "regression_search_outer_report.v1"
    assert result["target"] == plan["target"] and result["final_read_authority"] == "none"
    assert len(plan["candidates"]) == 33
    assert {c["feature_recipe"]["subset"] for c in plan["candidates"]} == {"H", "HM", "HMO"}
    selection = read("selection")
    for candidate in plan["candidates"]:
        key = content_hash([plan["plan_id"], "2024", "2023", candidate["candidate_id"]])
        terminal, fitted = read("terminal:" + key), read("execution:" + key)
        assert fitted["completed_fit_count"] == fitted["fit_attempt_count"] == 3
        assert len(fitted["diagnostics"]) == 3
        errors = []
        for row in terminal["predictions"]:
            day = next(day for day, event in by_day.items() if event == row["event_id"])
            assert row["crush_pct_iv30"] == VALUES[day]
            assert row["predicted_crush_pct_iv30"] == pytest.approx(-140.)
            errors.append(-140. - VALUES[day])
        assert terminal["metrics"]["rmse"] == pytest.approx(mean(e * e for e in errors) ** .5)
        assert terminal["metrics"]["mae"] == pytest.approx(mean(abs(e) for e in errors))
    chosen = min(selection["selections"][0]["scores"], key=lambda item: (item["metrics"]["rmse"], item["candidate_id"]))
    assert result["folds"][0]["candidate"]["candidate_id"] == chosen["candidate_id"]
    assert result["forecast_count"] == 3 and result["event_count"] == 1
    errors = [row["predicted_crush_pct_iv30"] + 60. for row in result["predictions"]]
    assert all(row["crush_pct_iv30"] == -60. for row in result["predictions"])
    assert result["metrics"]["rmse"] == pytest.approx(mean(e * e for e in errors) ** .5)
    assert set(group["outer"]["evaluation_event_ids"]) <= set(opened)
    before, saved = catalog_rows(data.conn), artifact_object_bytes(data.store)
    def forbidden(*args, **kwargs):
        pytest.fail("completed replay opened target payloads or started a child")
    monkeypatch.setattr(execution.subprocess, "Popen", forbidden)
    def replay_bytes(store, ref, *, max_bytes):
        assert ref.object_id not in target_ids, "replay reopened IV target bytes"
        return original(store, ref, max_bytes=max_bytes)
    monkeypatch.setattr(objects, "read_verified_bytes", replay_bytes)
    with no_fit_guard(), legacy_guard():
        assert driver.run_search(*args, limits=plan["request"]["limits"]["child"]) == result
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == saved


@pytest.mark.parametrize("action", ["seal", "run", "collect"])
def test_legacy_cli_refuses_iv_before_catalog(source, capsys, action):  # noqa: F811
    source.paths["request"].write_text(json.dumps({"schema_version": "regression_search_request.v1"}))
    result, _ = _invoke(source, capsys, action, status=2)
    assert result["refused"]["code"] == "INVALID_EXPERIMENT_SPEC"
    assert result["refused"]["message"] == "IV requires dedicated CLI containment; use the admitted library path"
    assert not source.calls and not source.connections


@pytest.mark.parametrize("guard,error", [(no_fit_guard, RuntimeFitForbidden), (legacy_guard, LegacyFitForbidden)])
def test_guards_refuse_native_iv_before_claims_or_inputs(search_case, guard, error):
    data = search_case
    args = (data.conn, data.store, data.repository, data.registration, data.plan["request"], data.ref)
    driver.seal_search_run(*args, clock=SystemClock())
    before, saved = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with guard(), pytest.raises(error):
        driver.run_search(*args, limits=data.plan["request"]["limits"]["child"])
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == saved


def test_unfrozen_outer_and_changed_inner_limits_refuse_without_claims(search_case):
    from experiments.magnitude_search_outer import execute_outer_fold
    data = search_case
    args = (data.conn, data.store, data.repository, data.registration, data.plan["request"], data.ref)
    driver.seal_search_run(*args, clock=SystemClock())
    limits = data.plan["request"]["limits"]["child"]
    before, saved = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with pytest.raises(OpsError) as caught:
        execute_outer_fold(*args, "2024", limits=limits)
    assert (caught.value.code, caught.value.problem.message) == ("EXPERIMENT_IDENTITY_CONFLICT",
        "outer evaluation requires an existing frozen selection")
    with pytest.raises(OpsError) as caught:
        driver.run_search(*args, limits={**limits, "wall_seconds": limits["wall_seconds"] + 1})
    assert (caught.value.code, caught.value.problem.message) == ("INVALID_EXPERIMENT_SPEC",
        "v2 trial limits differ from sealed request")
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == saved
