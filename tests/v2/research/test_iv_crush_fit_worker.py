"""Real fixed-child signed fits and unchanged accounting/ACK containment."""
from copy import deepcopy
import subprocess

import pytest

from engine.models.no_fit import RuntimeFitForbidden as LegacyFitForbidden, no_fit_guard as legacy_guard
from engine.v2.models.no_fit import RuntimeFitForbidden, no_fit_guard
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_execution as execution
from experiments import magnitude_search_worker as worker
from experiments.magnitude_search_evidence import fit_evidence
from experiments.magnitude_search_inputs import publish_inputs
from experiments.size_fit_diagnostics import evidence as diagnostics
from experiments.size_fit_diagnostics import keys as diagnostic_keys
from experiments.size_search_binding import execution_for_candidate
from tests.v2.research.test_iv_crush_fold_inputs import admit, fold_case  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401

LIMITS = {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 60}


@pytest.mark.parametrize("entry", ["trials", "driver", "outer"])
def test_iv_orchestration_refuses_before_claims_reads_or_child(fold_case, monkeypatch, entry):  # noqa: F811
    from engine.v2.contracts import ArtifactRef
    from engine.v2.foundation import SystemClock, from_document
    from engine.v2.ops.migrations import migrate
    from experiments import magnitude_search_outer as outer
    from experiments import magnitude_search_trials as trials
    from experiments import v2_magnitude_search_run as driver
    from experiments.magnitude_search_plan import seal_search_plan
    from experiments.magnitude_search_records import MIGRATIONS, OWNER
    from tests.v2.research.magnitude_trial_support import catalog_rows
    from tests.v2.research.test_native_registration import artifact_object_bytes
    data = fold_case
    request = data.plan["request"]
    sealed = seal_search_plan(data.conn, data.store, data.registration, request)
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    migrate(data.conn, OWNER, MIGRATIONS, clock=SystemClock())
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    def forbidden(*a, **k):
        pytest.fail("inactive IV orchestration reached an effect boundary")
    for module in (trials, outer, driver):
        monkeypatch.setattr(module, "append_record", forbidden)
    for module in (trials, outer):
        monkeypatch.setattr(module, "admit_fold_inputs", forbidden)
        monkeypatch.setattr(module, "run_fitted_trial", forbidden)
        monkeypatch.setattr(module, "load_magnitude_targets", forbidden)
    args = (data.conn, data.store, data.repository, data.registration, request, ref)
    with pytest.raises(OpsError) as caught:
        if entry == "trials":
            trials.SearchTrials(*args)
        elif entry == "driver":
            driver.seal_search_run(*args, clock=SystemClock())
        else:
            outer.execute_outer_fold(*args, data.plan["folds"][0]["outer"]["year"], limits=LIMITS)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "IV trial and outer orchestration is not enabled"
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


@pytest.mark.parametrize("model", ["history_ridge", "elasticnet", "history_hgb"])
def test_native_fold_to_real_signed_child(fold_case, model):  # noqa: F811
    data = fold_case
    assert diagnostic_keys(data.registration.document) == {"diagnostics"}
    candidate = next(c for c in data.plan["candidates"] if c["recipe"]["model"] == model
                     and c["feature_recipe"]["subset"] == "H")
    fit = execution_for_candidate(data.registration.document, candidate)
    assert fit["target"] == data.plan["target"] and fit["fit_recipe"] == candidate["recipe"]
    cohort = admit(data)
    manifest, identity = publish_inputs(data.store, cohort)
    source = {"manifest": manifest, "inputs_id": identity, "store_root": str(data.store.root)}
    events = []
    result = execution.run_fitted_trial(source, data.fold, data.plan, fit,
        execution.execution_limits(LIMITS), lambda state, evidence: events.append((state, evidence)))
    assert result["event"] == "succeeded"
    assert set(result) == {"event", "forecasts", "diagnostics"}
    assert result["forecasts"] == [{"event_id": event, "horizon_sessions": horizon,
        "predicted_crush_pct_iv30": pytest.approx(-140.)}
        for event in sorted(data.fold["evaluation_event_ids"]) for horizon in (1, 3, 5)]
    assert [state for state, _ in events] == ["started", "completed"] * 3
    assert [item["horizon_sessions"] for _, item in events] == [5, 5, 3, 3, 1, 1]
    assert [item for state, item in events if state == "completed"] == fit_evidence(cohort, data.fold, data.plan, fit)
    assert diagnostics(result, fit, [5, 3, 1]) == {"diagnostics": result["diagnostics"]}
    altered = deepcopy(result)
    altered["diagnostics"][0]["schema_version"] = "magnitude_elasticnet_fit.v1"
    with pytest.raises(OpsError) as caught:
        diagnostics(altered, fit, [5, 3, 1])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


class Poison:
    def __iter__(self):
        pytest.fail("guard inspected rows")

    def __getitem__(self, key):
        pytest.fail("guard inspected request")


@pytest.mark.parametrize("guard,error", [(legacy_guard, LegacyFitForbidden), (no_fit_guard, RuntimeFitForbidden)])
def test_both_guards_precede_signed_input_inspection(guard, error):
    with guard(), pytest.raises(error):
        worker.regression_forecasts(Poison(), Poison(), Poison(), None, [])


@pytest.mark.parametrize("error", [ValueError("private"), ArithmeticError("private"),
    MemoryError("private"), OSError("private"), ImportError("private"), RuntimeError("private")])
def test_signed_worker_adapter_preserves_failure_classification(monkeypatch, error):
    from engine.v2.research import horizon_fitted
    def broken(*a, **k):
        raise horizon_fitted.HorizonFitFailure("wrapped") from error
    monkeypatch.setattr(horizon_fitted, "fitted_forecasts", broken)
    request = {"execution": {"target": {"column": "crush_pct_iv30", "domain": "signed"},
        "features": ["x"], "fit_recipe": {}}, "inner": {"evaluation_event_ids": ["test"]}, "plan": {"horizons": [5, 3, 1]}}
    if isinstance(error, (ValueError, ArithmeticError)):
        with pytest.raises(OpsError) as caught:
            worker.regression_forecasts({}, [{"event_id": "train", "crush_pct_iv30": -140.}], request, None, [])
        assert caught.value.code == "EXPERIMENT_VARIANT_FAILED" and "private" not in str(caught.value)
    else:
        with pytest.raises(type(error)) as caught:
            worker.regression_forecasts({}, [{"event_id": "train", "crush_pct_iv30": -140.}], request, None, [])
        assert caught.value is error


@pytest.mark.parametrize("error", ["ValueError", "OSError"])
def test_real_signed_child_late_failure_keeps_only_acknowledged_prefix(fold_case, monkeypatch, error):  # noqa: F811
    data = fold_case
    candidate = next(c for c in data.plan["candidates"] if c["recipe"]["model"] == "history_ridge")
    fit = execution_for_candidate(data.registration.document, candidate)
    manifest, identity = publish_inputs(data.store, admit(data))
    source = {"manifest": manifest, "inputs_id": identity, "store_root": str(data.store.root)}
    original = subprocess.Popen
    body = ("count=0\noriginal=fitted.fit_regression_candidate\n"
        "def injected(*a,**k):\n global count\n count+=1\n"
        f" if count==2: raise {error}('private synthetic failure')\n return original(*a,**k)\n"
        "fitted.fit_regression_candidate=injected")
    script = ("import sys;sys.argv=sys.argv[1:];from engine.v2.models.training import regression_candidate as fitted;"
        + "exec(" + repr(body) + ");from experiments.magnitude_search_worker import main;main()")
    def launch(args, **kwargs):
        return original([args[0], "-c", script, *args[2:]], **kwargs)
    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    events = []
    def run():
        return execution.run_fitted_trial(source, data.fold, data.plan, fit,
            execution.execution_limits(LIMITS), lambda state, evidence: events.append((state, evidence)))
    if error == "ValueError":
        assert run() == {"event": "failed"}
    else:
        with pytest.raises(OpsError) as caught:
            run()
        assert caught.value.code == "RESOURCE_UNAVAILABLE" and "private" not in str(caught.value)
    assert [state for state, _ in events] == ["started", "completed", "started"]
    assert [item["horizon_sessions"] for _, item in events] == [5, 5, 3]
