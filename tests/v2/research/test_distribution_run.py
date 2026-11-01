"""Run orchestration with native seams doubled, real immutable journal/CAS/books."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.models import no_fit as legacy_guard
from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import (
    ArtifactStore,
    SystemClock,
    canonical_json,
    content_hash,
    to_document,
)
from engine.v2.models import no_fit as native_guard
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.migrations import migrate
from experiments import distribution_pipeline as pipeline
from experiments import distribution_run as runner
from experiments import distribution_run_plan as metadata
from experiments.distribution_run_request import PRIMARY_ARMS
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record
from experiments.magnitude_search_shards import iter_section, write_section
from experiments.native_registration import prepare_native_registration, reserve_native_registration
from tests.v2.research.magnitude_trial_support import prediction_spec
from tests.v2.research.test_distribution_native_admission import case as native_source_case
from tests.v2.research.test_distribution_native_admission import trials as native_trials
from tests.v2.research.test_distribution_run_request import request
from tests.v2.research.test_distribution_scores import destination, receipt


def _error(action, code="INVALID_EXPERIMENT_SPEC"):
    with pytest.raises(OpsError) as caught:
        action()
    assert caught.value.code == code
    return caught.value


@pytest.fixture
def case(tmp_path, monkeypatch):
    conn, store, clock = connect(tmp_path / "catalog.sqlite"), ArtifactStore(tmp_path / "store"), SystemClock()
    conn.execute("CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO experiment_runs VALUES ('distribution-own')")
    migrate(conn, OWNER, MIGRATIONS, clock=clock)
    snapshot = SnapshotRef(snapshot_id="quotes", manifest_hash=content_hash("snapshot"), table_versions={},
        calendar_version="calendar", source_priority_version="priority", finality_receipt_refs=(), knowledge_mode_by_table={})
    events = [{"event_id": f"{year}-{ticker}", "ticker": ticker, "event_date": f"{year}-06-03",
               "forecasts": [{"horizon_sessions": h, "session": "AMC", "source_session_ceiling": f"{year}-06-02"} for h in (5, 3, 1)]}
              for year in ("2023", "2024") for ticker in ("AAA", "BBB")]
    groups = [{"outer": {"year": year, "evaluation_event_ids": [e["event_id"] for e in events if e["event_date"].startswith(year)],
        "training_labels_before": year + "-01-01"}, "inner": [{"year": str(int(year) - 1)}]} for year in ("2023", "2024")]
    plan = {"plan_id": content_hash("B-plan"), "budget": {"child_limits": {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 30}}}
    binding = {"snapshot": to_document(snapshot), "holdouts": {"holdout_as_of_month": "2025-01"},
        "magnitude_execution": {"evaluation_labels_before": "2025-01-01"}, "size_panel": {"context": {"target_events": events}}}
    calls, state = [], {"missing_pool": False, "failure": None, "worker_hash": content_hash("worker")}
    def slots():
        return [row[0] for row in conn.execute("SELECT slot FROM magnitude_search_records ORDER BY slot")]
    def assert_intents(year):
        assert all(f"distribution:run-intents:{year}:{alpha}" in slots() for alpha in (0, .25, .5)), "future read before all alpha intents"
    def seal(*, clock):
        calls.append(("B-seal",))
        return append_record(conn, store, "distribution-own", "distribution:seal", plan)
    def start(role, year, inner=None):
        calls.append(("admit", role, year, inner))
        if year == "2024":
            assert_intents("2023")
        return append_record(conn, store, "distribution-own", f"distribution:admission:{role}:{year}:{inner}", {}, replay=False)
    trials = SimpleNamespace(conn=conn, store=store, own=SimpleNamespace(run_id="distribution-own",
        document={"execution_plan": {"runner": "experiments/distribution_run.py"}}),
        source=SimpleNamespace(document=binding), repository=SimpleNamespace(resolve=lambda value: snapshot),
        request={}, source_plan_ref=None, recipe={"horizons": [5, 3, 1]}, caps={}, seal=seal, start=start)
    monkeypatch.setattr(metadata, "prepare_plan", lambda *args: (deepcopy(plan), {"folds": deepcopy(groups)}))
    def closure(root, entries):
        assert "experiments/distribution_worker.py" in entries
        return {entry: state["worker_hash"] for entry in entries}
    monkeypatch.setattr(metadata, "source_closure", closure)
    for module in (metadata, pipeline):
        monkeypatch.setattr(module, "read_bridge", lambda *args: SimpleNamespace(intercept=1, slope=1, residuals=[0]))
    @contextmanager
    def contained(limits, seconds):
        calls.append(("contained", deepcopy(limits), seconds))
        yield
    monkeypatch.setattr(runner, "contained", contained)
    class Fits:
        def __init__(self, value):
            assert value is trials
        def fit(self, role, year, horizon, inner=None, *, bounded):
            assert bounded is True
            calls.append(("fit", role, year, horizon, inner))
            if state["failure"] is not None and sum(call[0] == "fit" for call in calls) >= state.get("fail_on", 1):
                raise state["failure"]
            trial = content_hash([role, year, horizon, inner])
            ids = next(g["outer"]["evaluation_event_ids"] for g in groups if g["outer"]["year"] == year)
            scope = destination(outer_year=year, horizon=horizon)
            scope.update(role=role, inner_year=inner)
            evidence = receipt(scope)
            evidence["claim"]["trial"]["trial_id"] = trial
            return {"receipt": {**evidence, "result_id": content_hash([trial, "result"])},
                "rows": [{"event_id": event, "probabilities": [0] * 4 + [1] + [0] * 76, "point_prediction": 1} for event in ids],
                "tail": {"policy": "empirical_excess", "sample_count": 1, "excesses": [0]}}
    monkeypatch.setattr(runner, "DistributionFits", Fits)
    def pool(fits, role, year, horizon):
        body = {"schema_version": "distribution_oof_pool.v1", "policy": "earlier_inner_equal_weight_per_horizon.v1",
            "point_status": "ready", "histogram_status": "no_eligible_rows" if state["missing_pool"] else "ready",
            "point_residuals": [0], "histogram_residuals": [0],
            "destination": destination(outer_year=year, horizon=horizon)}
        return {**body, "pool_id": content_hash(body)}
    monkeypatch.setattr(pipeline, "residual_pool", pool)
    monkeypatch.setattr(runner, "trading_calendar_from_snapshot", lambda *a, **kw: "calendar")
    def decisions(repository, snapshot, selected, alphas, economics, *, calendar=None):
        year = selected[0]["event_date"][:4]
        assert calendar == "calendar"
        calls.append(("decision", year, tuple(alphas)))
        rows = {event["event_id"]: ({"decision_date": pd.Timestamp(year + "-06-02"), "entry_date": pd.Timestamp(year + "-06-03"),
            "exit_date": pd.Timestamp(year + "-06-04"), "ticker": event["ticker"]}, SimpleNamespace(spot=100, cost=1)) for event in selected}
        return {alpha: rows for alpha in alphas}
    def outcomes(repository, snapshot, decisions, alpha, economics, contracts):
        year = next(iter(decisions))[:4]
        assert_intents(year)
        calls.append(("outcome", year, alpha))
        return {event: {"entry_cash": 100, "exit_cash": None if state.get("missing_exit") and event == "2023-AAA" else state.get("exit_cash", 120), "exit_at": year + "-06-04", "contracts": 1, "multiplier": 100}
                for event in decisions}
    def targets(repository, snapshot, **kwargs):
        year = kwargs["event_ids"][0][:4]
        assert_intents(year)
        calls.append(("targets", year))
        assert kwargs["omit_not_computed"] is True
        selected = [event for event in kwargs["event_ids"] if not (state.get("omitted") and event == "2023-AAA")]
        frame = pd.DataFrame({"event_id": selected, "absolute_move_pct": [1.25] * len(selected)})
        frame.attrs["omitted_labels"] = [{"event_id": event, "reason": "skipped"} for event in kwargs["event_ids"] if event not in selected]
        return frame
    monkeypatch.setattr(pipeline, "decision_quotes", decisions)
    monkeypatch.setattr(pipeline, "outcome_quotes", outcomes)
    monkeypatch.setattr(pipeline, "load_magnitude_targets", targets)
    benchmarks = []
    for group in groups:
        fold = group["outer"]
        rows = [{"event_id": event, "incumbent_magnitudes": [1], "gate_selected": True} for event in sorted(fold["evaluation_event_ids"])]
        descriptor = write_section(store, "distribution_benchmark", rows, lambda row: (row["event_id"],))
        body = {"schema_version": "distribution_benchmark.v1", "outer_year": fold["year"],
                "training_labels_before": fold["training_labels_before"], "rows": descriptor,
                "forecast_horizon": 1, "decision_policy": "native_h1_source_session_ceiling"}
        benchmarks.append({"outer_year": fold["year"], "ref": to_document(store.publish_bytes(canonical_json(body).encode(), schema_ref=body["schema_version"]))})
    result = SimpleNamespace(trials=trials, request=request(), calls=calls, state=state, slots=slots, store=store, conn=conn,
        groups=groups, plan=plan, benchmarks=benchmarks, clock=clock)
    yield result
    conn.close()


def _make(case, **changes):
    return runner.DistributionRun(case.trials, {**case.request, **changes})


def _rows(case, section):
    return list(iter_section(case.store, section["name"], section["descriptor"], lambda row: (row["position"],)))


@pytest.mark.parametrize("registered_runner", ["experiments/distribution_run_plan.py", "experiments/v2_prediction.py"])
@pytest.mark.parametrize("operation", ["preflight", "seal"])
def test_wrong_registered_runner_refuses_before_d_seal_attempt_or_fit(case, registered_runner, operation):
    case.trials.own.document["execution_plan"]["runner"] = registered_runner
    run = _make(case)
    call = run.preflight if operation == "preflight" else lambda: run.seal(clock=case.clock)
    expected = fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution run requires its registered runner")
    assert _error(call, expected.code).problem == expected.problem
    assert not any(slot.startswith("distribution:run-") for slot in case.slots())
    assert not any(call[0] in ("fit", "decision", "outcome", "targets", "admit") for call in case.calls)


def test_registered_runner_check_follows_inherited_verification(case, monkeypatch):
    case.trials.own.document = None
    failure = fail("RESOURCE_UNAVAILABLE", "synthetic inherited verification refusal")
    def refused(*args):
        raise failure
    monkeypatch.setattr(metadata.DistributionRunPlan, "preflight", refused)
    assert _error(_make(case).preflight, failure.code) is failure


@pytest.fixture(scope="module")
def native_case(tmp_path_factory):
    yield from native_source_case.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize("registered_runner", ["experiments/distribution_run.py",
    "experiments/distribution_run_plan.py", "experiments/v2_prediction.py"])
def test_actual_native_registration_requires_genuine_distribution_runner(native_case, monkeypatch, registered_runner):
    case = native_case
    source = case.registration.document
    spec = prediction_spec(experiment_id="synthetic-distribution-runner-" + Path(registered_runner).stem,
        primary_arm_id="raw_pair", arms=("raw_pair",), folds=("2024",), runner=registered_runner)
    own, spec = prepare_native_registration(case.repository, spec,
        code_root=Path(__file__).resolve().parents[3], as_of_month=source["holdouts"]["holdout_as_of_month"],
        scope=source["scope"], event_ids=source["event_ids"], snapshot=case.panel_source)
    own = reserve_native_registration(case.conn, case.store, own, spec)
    adapter = native_trials(case, own)
    values = request()
    values["quote_snapshot_id"] = case.panel_source.snapshot_id
    values["bridges"] = [entry for entry in values["bridges"] if entry["outer_year"] == "2024"]
    # Only the frozen bridge seam is doubled; B registration/source-plan/closure checks are real.
    monkeypatch.setattr(metadata, "read_bridge", lambda *args: object())
    run = runner.DistributionRun(adapter, values)
    changes = case.conn.total_changes
    with legacy_guard.no_fit_guard(), native_guard.no_fit_guard():
        if registered_runner == "experiments/distribution_run.py":
            result = run.preflight()
            assert result["opportunities"] == len(case.plan["folds"][0]["outer"]["evaluation_event_ids"])
            assert result["final_read_authority"] == "none"
            assert registered_runner in result["source_closure"]
        else:
            expected = fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution run requires its registered runner")
            assert _error(run.preflight, expected.code).problem == expected.problem
    assert case.conn.total_changes == changes
    assert case.conn.execute("SELECT count(*) FROM magnitude_search_records WHERE run_id=?",
                             (own.run_id,)).fetchone()[0] == 0


def test_preflight_is_metadata_only_and_not_registration_or_fit(case):
    result = _make(case).preflight()
    assert result["opportunities"] == 4
    assert result["available_benchmarks"] == {"incumbent": False, "gate": False}
    assert result["source_closure"]["experiments/distribution_worker.py"] == case.state["worker_hash"]
    assert case.calls == [] and case.slots() == []


@pytest.mark.parametrize("change", [{"strict_benchmarks": True}, {"required_arms": [*PRIMARY_ARMS, "incumbent"]},
                                    {"required_arms": [*PRIMARY_ARMS, "gate"]}])
def test_missing_required_or_strict_benchmarks_refuse_before_effects(case, change):
    _error(lambda: _make(case, **change).preflight())
    assert case.calls == [] and case.slots() == []


def test_existing_benchmark_cas_is_verified_for_every_outer(case):
    result = _make(case, benchmarks=case.benchmarks, strict_benchmarks=True).preflight()
    assert result["available_benchmarks"] == {"incumbent": True, "gate": True}
    _error(lambda: _make(case, benchmarks=case.benchmarks[:1], strict_benchmarks=True).preflight())


@pytest.mark.parametrize("change", [{"max_opportunities": 3}, {"bridges": []}])
def test_preflight_derives_schedule_and_rejects_insufficient_declarations(case, change):
    _error(lambda: _make(case, **change).preflight())
    assert case.calls == []


def test_every_outer_freezes_all_alpha_intents_before_future_reads_or_next_admission(case):
    run = _make(case)
    run.seal(clock=case.clock)
    result = run.run(clock=case.clock)
    fits = [call[1:] for call in case.calls if call[0] == "fit"]
    assert fits == [(role, year, horizon, inner) for year in ("2023", "2024")
        for role, inner in (("inner", str(int(year) - 1)), ("outer", None)) for horizon in (5, 3, 1)]
    assert sum(slot.startswith("distribution:run-fit:") for slot in case.slots()) == 12
    assert case.calls.index(("targets", "2023")) < case.calls.index(("admit", "inner", "2024", "2023"))
    assert result["current_strategy_superiority_claim"] is False
    for item in result["reports"]:
        report = item["report"]
        assert report["opportunities"] == 4
        assert report["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 0
        assert report["contrasts"]["incumbent"]["status"] == "unavailable_arm"
        assert report["contrasts"]["gate"]["status"] == "unavailable_arm"
        assert "incumbent" not in report["arms"] and "gate" not in report["arms"]
    assert sum(section["name"].startswith("scores:") for section in result["sections"]) == 6
    assert all(len(_rows(case, section)) == 6 for section in result["sections"] if section["name"].startswith("scores:"))


def test_common_primary_availability_prevents_arm_specific_selection(case):
    case.state["missing_pool"] = True
    run = _make(case)
    run.seal(clock=case.clock)
    result = run.run(clock=case.clock)
    for section in result["sections"]:
        if section["name"].startswith("intents:"):
            for row in _rows(case, section):
                assert row["primary_common"] is False
                assert all(row["arms"][arm]["selected"] is False for arm in PRIMARY_ARMS)
    assert all(report["report"]["arms"]["histogram"]["funded"] == 0 for report in result["reports"])


def test_failed_attempt_is_recorded_once_and_cannot_retry(case):
    run = _make(case)
    run.seal(clock=case.clock)
    failure = fail("EXPERIMENT_VARIANT_FAILED", "synthetic fit failure")
    case.state["failure"] = failure
    assert _error(lambda: run.run(clock=case.clock), failure.code) is failure
    assert "distribution:run-started" in case.slots() and "distribution:run-failed" in case.slots()
    assert "distribution:run-completed" not in case.slots()
    case.state["failure"] = None
    _error(lambda: run.run(clock=case.clock), "EXPERIMENT_IDENTITY_CONFLICT")
    assert sum(call[0] == "fit" for call in case.calls) == 1


def test_completed_collection_and_run_replay_are_readonly_even_under_guards(case, monkeypatch):
    run = _make(case)
    run.seal(clock=case.clock)
    result = run.run(clock=case.clock)
    before, changes = case.slots(), case.conn.total_changes
    for module, names in ((pipeline, ("decision_quotes", "outcome_quotes", "load_magnitude_targets")), (runner, ("DistributionFits",))):
        for name in names:
            monkeypatch.setattr(module, name, lambda *a, **kw: pytest.fail("completed replay opened an execution boundary"))
    monkeypatch.setattr(case.store, "publish_bytes", lambda *a, **kw: pytest.fail("completed replay wrote CAS"))
    with legacy_guard.no_fit_guard(), native_guard.no_fit_guard():
        assert run.collect() == result and run.run() == result
    assert case.slots() == before and case.conn.total_changes == changes


def test_worker_closure_drift_refuses_before_fits_or_quotes(case):
    run = _make(case)
    run.seal(clock=case.clock)
    case.state["worker_hash"] = content_hash("changed-worker")
    _error(run.run)
    assert not any(call[0] in ("fit", "decision", "outcome", "targets") for call in case.calls)
    assert "distribution:run-started" not in case.slots()


def test_collect_does_not_fallback_to_work_when_completion_missing(case):
    run = _make(case)
    run.seal(clock=case.clock)
    _error(run.collect)
    assert case.calls == [("B-seal",)]


def test_bootstrap_time_consumes_job_budget_before_attempt_claim(case, monkeypatch):
    run = _make(case)
    run.seal(clock=case.clock)
    began = runner.time.monotonic()
    monkeypatch.setattr(runner.time, "monotonic", lambda: began + run.request["limits"]["job_wall_seconds"] + 1)
    _error(lambda: run._run_contained(case.clock, began), "RESOURCE_LIMIT_EXCEEDED")
    assert "distribution:run-started" not in case.slots()
    assert not any(call[0] == "fit" for call in case.calls)


def test_started_claim_is_sql_immutable_and_duplicate_is_refused(case):
    run = _make(case)
    run.seal(clock=case.clock)
    append_record(case.conn, case.store, case.trials.own.run_id, "distribution:run-started", {"interrupted": True}, replay=False)
    _error(run.run, "EXPERIMENT_IDENTITY_CONFLICT")
    assert read_record(case.conn, case.store, case.trials.own.run_id, "distribution:run-started") == {"interrupted": True}
    assert not any(call[0] == "fit" for call in case.calls)


def test_skipped_targets_remain_explicit_score_rows_not_selection_filters(case):
    case.state["omitted"] = True
    run = _make(case)
    run.seal(clock=case.clock)
    result = run.run(clock=case.clock)
    missing = [row for section in result["sections"] if section["name"].startswith("scores:")
               for row in _rows(case, section) if row["event_id"] == "2023-AAA"]
    assert len(missing) == 9
    assert all(row["status"] == "missing_target" and row["observation"] is None and row["reason"] == "skipped" for row in missing)
    assert all(item["report"]["arms"]["histogram"]["funded"] == 4 for item in result["reports"])


def test_missing_future_exit_does_not_change_frozen_intents_or_become_zero_profit(case):
    case.state["missing_exit"] = True
    run = _make(case)
    run.seal(clock=case.clock)
    result = run.run(clock=case.clock)
    for item in result["reports"]:
        report = item["report"]
        assert report["arms"]["histogram"]["funded"] == 4 and report["arms"]["histogram"]["unresolved"] == 1
        assert report["contrasts"]["same_mean"]["net_delta_per_opportunity"] is None
        assert report["risk_checks"]["histogram"]["completeness_met"] is False
        assert report["risk_checks"]["histogram"]["loss_limit_met"] is None
    for section in result["sections"]:
        if section["name"].startswith("intents:"):
            assert all(row["arms"]["histogram"]["selected"] for row in _rows(case, section))


def test_declared_risk_limits_are_computed_not_just_echoed(case):
    case.state["exit_cash"] = 50
    risk = {**case.request["risk"], "maximum_loss_usd": 0, "maximum_settled_drawdown_usd": 0}
    run = _make(case, risk=risk)
    run.seal(clock=case.clock)
    result = run.run(clock=case.clock)
    for item in result["reports"]:
        checks = item["report"]["risk_checks"]["histogram"]
        assert checks == {"loss_limit_met": False, "settled_drawdown_met": False, "completeness_met": True}


def test_completed_collect_verifies_cas_not_just_catalog_metadata(case, monkeypatch):
    run = _make(case)
    run.seal(clock=case.clock)
    result = run.run(clock=case.clock)
    bad = result["sections"][0]["descriptor"]["shards"][0]["ref"]["content_hash"]
    original = case.store.read_verified
    failure = fail("RESOURCE_UNAVAILABLE", "synthetic missing shard")
    def read(ref):
        if ref.content_hash == bad:
            raise failure
        return original(ref)
    monkeypatch.setattr(case.store, "read_verified", read)
    assert _error(run.collect, failure.code) is failure


@pytest.mark.parametrize("attempt", [4, 8])
def test_partial_attempt_prefix_is_preserved_without_skipping_failed_fit(case, attempt):
    run = _make(case)
    run.seal(clock=case.clock)
    case.state.update(failure=fail("RESOURCE_LIMIT_EXCEEDED", "synthetic child exhausted"), fail_on=attempt)
    _error(lambda: run.run(clock=case.clock), "RESOURCE_LIMIT_EXCEEDED")
    assert sum(call[0] == "fit" for call in case.calls) == attempt
    assert sum(slot.startswith("distribution:run-fit:") for slot in case.slots()) == attempt - 1
    failure = read_record(case.conn, case.store, case.trials.own.run_id, "distribution:run-failed")
    assert failure["code"] == "RESOURCE_LIMIT_EXCEEDED"
    assert "distribution:run-completed" not in case.slots()
    _error(run.run, "EXPERIMENT_IDENTITY_CONFLICT")
    assert sum(call[0] == "fit" for call in case.calls) == attempt


def test_metadata_only_seal_cannot_be_reused_by_final_runner(case):
    metadata.DistributionRunPlan(case.trials, case.request).seal(clock=case.clock)
    run = _make(case)
    _error(run.run)
    assert not any(call[0] in ("fit", "decision", "outcome", "targets") for call in case.calls)
    assert "distribution:run-started" not in case.slots()


def test_final_runner_seal_cannot_be_reused_by_metadata_entrypoint(case):
    _make(case).seal(clock=case.clock)
    plan = metadata.DistributionRunPlan(case.trials, case.request)
    _error(plan.admit)
    assert case.calls == [("B-seal",)]


def test_run_admission_occurs_inside_containment(case, monkeypatch):
    run = _make(case)
    run.seal(clock=case.clock)
    admitted = run.admit
    active = []
    @contextmanager
    def containment(*args):
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    def admit():
        assert active, "run admission escaped parent containment"
        return admitted()
    monkeypatch.setattr(runner, "contained", containment)
    monkeypatch.setattr(run, "admit", admit)
    run.run(clock=case.clock)
    assert not active
