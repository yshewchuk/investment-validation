"""Fresh IV admission is read-only; a claimed fit still has no retry authority."""
import pytest

from engine.v2.foundation import SystemClock, content_hash
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research import iv_crush_inputs
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_trials as trials_module
from experiments import size_panel, size_search_binding
from experiments import v2_magnitude_search_run as driver
from experiments.magnitude_search_records import read_record
from experiments.size_job_budget import remaining
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_search_execution import search_case  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes


def invocation(data, monkeypatch, stage):
    plan = data.plan
    args = (data.conn, data.store, data.repository, data.registration, plan["request"], data.ref)
    driver.seal_search_run(*args, clock=SystemClock())
    limits = plan["request"]["limits"]["child"]
    group, candidate = plan["folds"][0], plan["candidates"][0]
    trial = (group["outer"]["year"], group["inner"][0]["year"], candidate["candidate_id"])
    recorder = trials_module.SearchTrials(*args)
    if stage == "start":
        return lambda: recorder.start(*trial)
    if stage == "inner":
        return lambda: recorder.execute(*trial, limits=limits)
    if stage == "driver":
        return lambda: driver.run_search(*args, limits=limits)
    # Isolate the already-validated selection boundary; the whole-search oracle
    # separately exercises real selection. Outer plan/admission/journal stay real.
    selection = {"selections": [{"outer_year": group["outer"]["year"],
        "selected_candidate_id": candidate["candidate_id"]}],
        "prospective_candidate_id": candidate["candidate_id"]}
    monkeypatch.setattr(outer, "_native_selection", lambda *a: selection)
    return lambda: outer.execute_outer_fold(*args, group["outer"]["year"], limits=limits)


@pytest.mark.parametrize("stage", ["start", "inner", "driver", "outer"])
@pytest.mark.parametrize("reader", ["panel", "training_targets"])
def test_fresh_iv_admission_failure_has_no_claim_or_cas_effect(search_case, monkeypatch, stage, reader):  # noqa: F811
    data = search_case
    invoke = invocation(data, monkeypatch, stage)
    # Admission may consume the existing whole-job budget, never reset it.
    remaining(data.conn, data.store, data.registration, data.plan)
    size_search_binding._PANEL_MEMO.clear()
    error = fail("RESOURCE_UNAVAILABLE", "synthetic admission read unavailable")
    calls = []
    def refuse(*args, **kwargs):
        calls.append(None)
        raise error
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with monkeypatch.context() as patch:
        if reader == "panel":
            patch.setattr(size_panel, "read", refuse)
        else:
            patch.setattr(iv_crush_inputs, "load_iv_crush_targets", refuse)
        for attempt in (1, 2):
            with pytest.raises(OpsError) as caught:
                invoke()
            assert caught.value.problem == error.problem
            assert len(calls) == attempt
            assert catalog_rows(data.conn) == before
            assert artifact_object_bytes(data.store) == artifacts
    # Removing only the transient read refusal permits an explicit fresh call.
    # Stop at publication, after actual admission, without launching any fit.
    reached = fail("RESOURCE_UNAVAILABLE", "synthetic post-admission publication checkpoint")
    def stop_at_publication(*args, **kwargs):
        raise reached
    monkeypatch.setattr(outer if stage == "outer" else trials_module, "publish_inputs", stop_at_publication)
    with pytest.raises(OpsError) as caught:
        invoke()
    assert caught.value.problem == reached.problem
    assert sum(row[0].startswith("outer-start:" if stage == "outer" else "start:")
        for row in data.conn.execute("SELECT slot FROM magnitude_search_records")) == 1


@pytest.mark.parametrize("stage", ["inner", "outer"])
@pytest.mark.parametrize("failure_at", ["publication", "fit_started"])
def test_iv_post_claim_failure_keeps_claim_and_refuses_before_readmission(search_case, monkeypatch, stage, failure_at):  # noqa: F811
    from experiments.magnitude_search_evidence import fit_evidence
    from experiments.magnitude_search_inputs import load_inputs
    data = search_case
    invoke = invocation(data, monkeypatch, stage)
    module = trials_module if stage == "inner" else outer
    error = fail("RESOURCE_UNAVAILABLE", "synthetic interruption after claim")
    def interrupted(source, fold, plan, execution, limits, progress):
        inputs = load_inputs(data.store, source["manifest"], source["inputs_id"])
        progress("started", fit_evidence(inputs, fold, plan, execution)[0])
        raise error
    def publication(*args, **kwargs):
        raise error
    if failure_at == "publication":
        monkeypatch.setattr(module, "publish_inputs", publication)
    monkeypatch.setattr(module, "run_fitted_trial", interrupted)
    with pytest.raises(OpsError) as caught:
        invoke()
    assert caught.value.problem == error.problem
    group = data.plan["folds"][0]
    key = content_hash([data.plan["plan_id"], group["outer"]["year"],
        group["inner"][0]["year"], data.plan["candidates"][0]["candidate_id"]]) if stage == "inner" else (
        content_hash([data.plan["plan_id"], group["outer"]["year"]]))
    prefix = "" if stage == "inner" else "outer-"
    assert read_record(data.conn, data.store, data.registration.run_id, prefix + "start:" + key)
    slots = [row[0] for row in data.conn.execute("SELECT slot FROM magnitude_search_records")]
    assert sum(slot.startswith(prefix + "fit-started:") for slot in slots) == (failure_at == "fit_started")
    assert not any(slot.startswith(prefix + "fit-completed:") for slot in slots)
    before, artifacts = catalog_rows(data.conn), artifact_object_bytes(data.store)
    def forbidden(*args, **kwargs):
        pytest.fail("claimed IV fit was re-admitted or relaunched")
    monkeypatch.setattr(module, "run_fitted_trial", forbidden)
    monkeypatch.setattr(module, "publish_inputs", forbidden)
    if stage == "inner":
        monkeypatch.setattr(trials_module.SearchTrials, "_cohort_inputs", forbidden)
    else:
        monkeypatch.setattr(outer, "_inputs", forbidden)
    with pytest.raises(OpsError) as caught:
        invoke()
    assert caught.value.problem == fail(
        "EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
    assert catalog_rows(data.conn) == before
    assert artifact_object_bytes(data.store) == artifacts


@pytest.mark.parametrize("stage", ["inner", "outer"])
def test_iv_concurrent_claim_wins_before_publication_or_fit(search_case, monkeypatch, stage):  # noqa: F811
    data = search_case
    invoke = invocation(data, monkeypatch, stage)
    module = trials_module if stage == "inner" else outer
    original = module.append_record
    claimed = []
    def competing(conn, store, run_id, slot, body, **kwargs):
        if slot.startswith("outer-start:" if stage == "outer" else "start:"):
            assert kwargs.get("replay") is False
            original(conn, store, run_id, slot, body, **kwargs)
            claimed.append(slot)
        return original(conn, store, run_id, slot, body, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("losing claim published inputs or launched a fit")
    monkeypatch.setattr(module, "append_record", competing)
    monkeypatch.setattr(module, "publish_inputs", forbidden)
    monkeypatch.setattr(module, "run_fitted_trial", forbidden)
    with pytest.raises(OpsError) as caught:
        invoke()
    assert caught.value.problem == fail(
        "EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
    assert len(claimed) == 1
    slots = [row[0] for row in data.conn.execute("SELECT slot FROM magnitude_search_records")]
    assert not any(slot.startswith(("inputs:", "outer-inputs:", "fit-started:", "outer-fit-started:")) for slot in slots)
