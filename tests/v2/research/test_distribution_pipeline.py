"""Standalone outer-decision/report pipeline; real CAS/books, no runner or fits."""
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.models.no_fit import no_fit_guard as legacy_no_fit
from engine.v2.foundation import content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.executor import THREAD_VARIABLES
from experiments import distribution_pipeline as pipeline
from experiments.distribution_run_request import PRIMARY_ARMS
from experiments.magnitude_search_shards import iter_section
from tests.v2.research.test_distribution_run_plan import _error, make_plan
from tests.v2.research.test_distribution_run_plan import plan_case as plan_case
from tests.v2.research.test_distribution_scores import destination, receipt


@pytest.fixture
def pipeline_case(plan_case, monkeypatch):
    case = plan_case
    case.state.update(missing_pool=False, missing_exit=False, omitted=False, wrong_scope=False)
    def pool(fits, role, year, horizon):
        assert fits is case.fits and role == "outer" and horizon in (5, 3, 1)
        body = {"schema_version": "distribution_oof_pool.v1", "policy": "earlier_inner_equal_weight_per_horizon.v1",
            "point_status": "ready", "histogram_status": "no_eligible_rows" if case.state["missing_pool"] else "ready",
            "point_residuals": [0], "histogram_residuals": [0], "destination": destination(outer_year=year, horizon=3 if case.state["wrong_scope"] and horizon == 1 else horizon)}
        return {**body, "pool_id": content_hash(body)}
    def assert_intents(year):
        assert all(f"distribution:run-intents:{year}:{alpha}" in case.slots() for alpha in case.request["alphas"])
    def decisions(repository, snapshot, events, alphas, economics, *, calendar):
        assert calendar == "frozen-calendar"
        assert all("session" not in event and event["forecasts"][-1]["session"] == "AMC" for event in events)
        year = events[0]["event_date"][:4]
        case.calls.append(("decisions", year))
        rows = {event["event_id"]: ({"decision_date": pd.Timestamp(year + "-06-02"),
            "entry_date": pd.Timestamp(year + "-06-03")}, SimpleNamespace(spot=100, cost=1)) for event in events}
        return {alpha: rows for alpha in alphas}
    def outcomes(repository, snapshot, decisions, alpha, economics, contracts):
        year = next(iter(decisions))[:4]
        assert_intents(year)
        case.calls.append(("outcomes", year, alpha))
        return {event: {"entry_cash": 100, "exit_cash": None if case.state["missing_exit"] and event == "2023-AAA" else 120,
            "exit_at": year + "-06-04", "contracts": contracts, "multiplier": 100} for event in decisions}
    def targets(repository, snapshot, **kwargs):
        year = kwargs["event_ids"][0][:4]
        assert_intents(year)
        assert kwargs["omit_not_computed"] is True
        case.calls.append(("targets", year))
        selected = [event for event in kwargs["event_ids"] if not (case.state["omitted"] and event == "2023-AAA")]
        frame = pd.DataFrame({"event_id": selected, "absolute_move_pct": [1.25] * len(selected)})
        frame.attrs["omitted_labels"] = [{"event_id": event, "reason": "skipped"} for event in kwargs["event_ids"] if event not in selected]
        return frame
    monkeypatch.setattr(pipeline, "residual_pool", pool)
    monkeypatch.setattr(pipeline, "read_bridge", lambda *args: SimpleNamespace(intercept=1, slope=1, residuals=[0]))
    monkeypatch.setattr(pipeline, "decision_quotes", decisions)
    monkeypatch.setattr(pipeline, "outcome_quotes", outcomes)
    monkeypatch.setattr(pipeline, "load_magnitude_targets", targets)
    case.fits = object()  # No estimator/admission API exists on this sentinel.
    case.run = make_plan(case)
    case.plan = case.run.seal(clock=case.clock)
    case.books = {alpha: {arm: [] for arm in (*PRIMARY_ARMS, "incumbent", "gate")} for alpha in case.request["alphas"]}
    case.quotes = {alpha: {} for alpha in case.books}
    case.opportunities, case.sections = [], []
    return case


def _evaluate(case, group):
    rows = [{"event_id": event, "probabilities": [0] * 4 + [1] + [0] * 76, "point_prediction": 1}
            for event in group["outer"]["evaluation_event_ids"]]
    results = {horizon: {"rows": rows, "receipt": receipt(destination(outer_year=group["outer"]["year"], horizon=horizon)), "tail": {"policy": "empirical_excess", "sample_count": 1, "excesses": [0]}}
               for horizon in (5, 3, 1)}
    pipeline.evaluate_outer(case.run, group, case.fits, results, case.plan, case.books, case.quotes,
        case.opportunities, case.sections, "frozen-calendar")


def _finish(case):
    return pipeline.finish_reports(case.run, case.books, case.quotes, case.opportunities, case.sections)


def _rows(case, prefix):
    return [row for section in case.sections if section["name"].startswith(prefix)
        for row in iter_section(case.store, section["name"], section["descriptor"], lambda row: (row["position"],))]


def test_pipeline_import_has_no_future_runner_dependency():
    script = """import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name != 'experiments.distribution_run', name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments.distribution_pipeline import evaluate_outer, finish_reports
assert 'experiments.distribution_run' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **dict.fromkeys(THREAD_VARIABLES, "1")})
    assert result.returncode == 0, result.stderr


def test_pipeline_freezes_all_fill_intents_before_any_outcome_without_fitting(pipeline_case):
    case = pipeline_case
    with no_fit_guard(), legacy_no_fit():
        for group in case.groups:
            _evaluate(case, group)
        result = _finish(case)
    assert len(case.opportunities) == 4
    assert len(_rows(case, "scores:")) == 36
    assert len(_rows(case, "profit-scores:")) == 36
    assert not hasattr(case.run, "run")
    for item in result["reports"]:
        report = item["report"]
        assert report["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 0
        assert report["contrasts"]["incumbent"]["status"] == "unavailable_arm"
        assert report["contrasts"]["gate"]["status"] == "unavailable_arm"
        assert report["arms"]["histogram"]["funded"] == 4
    assert result["promotion_authority"] == "none" and result["current_strategy_superiority_claim"] is False


def test_pipeline_common_mask_abstains_all_primary_arms_when_control_unavailable(pipeline_case):
    case = pipeline_case
    case.state["missing_pool"] = True
    for group in case.groups:
        _evaluate(case, group)
    for row in _rows(case, "intents:"):
        assert row["primary_common"] is False
        assert all(not row["arms"][arm]["selected"] for arm in PRIMARY_ARMS)
    assert all(item["report"]["arms"]["histogram"]["funded"] == 0 for item in _finish(case)["reports"])


def test_pipeline_missing_labels_and_exit_preserve_opportunity_and_unknown_loss(pipeline_case):
    case = pipeline_case
    case.state.update(omitted=True, missing_exit=True)
    for group in case.groups:
        _evaluate(case, group)
    missing = [row for row in _rows(case, "scores:") if row["event_id"] == "2023-AAA"]
    assert len(missing) == 9 and all(row["status"] == "missing_target" for row in missing)
    for item in _finish(case)["reports"]:
        report = item["report"]
        assert report["opportunities"] == 4 and report["arms"]["histogram"]["funded"] == 4
        assert report["arms"]["histogram"]["unresolved"] == 1
        assert report["contrasts"]["same_mean"]["net_delta_per_opportunity"] is None
        assert report["risk_checks"]["histogram"]["loss_limit_met"] is None
        assert report["risk_checks"]["histogram"]["completeness_met"] is False


def test_duplicate_outer_intents_refuse_before_any_evaluation_effect(pipeline_case, monkeypatch):
    case = pipeline_case
    _evaluate(case, case.groups[0])
    _assert_no_effects(case, monkeypatch, lambda: _evaluate(case, case.groups[0]), "EXPERIMENT_IDENTITY_CONFLICT")


@pytest.mark.parametrize("alpha", [0, .25, .5, "0.0"])
def test_any_partial_alpha_intent_prevents_accumulator_and_cas_changes(pipeline_case, monkeypatch, alpha):
    case = pipeline_case
    if alpha == "0.0":
        case.books = {float(key): value for key, value in case.books.items()}
    # Occupancy, not truthiness, prevents re-entry even for an empty body.
    case.run.record(f"intents:2023:{alpha}", {}, replay=False)
    _assert_no_effects(case, monkeypatch, lambda: _evaluate(case, case.groups[0]), "EXPERIMENT_IDENTITY_CONFLICT")


def test_float_alpha_book_keys_use_sealed_slot_names_and_refuse_partial_reentry(pipeline_case, monkeypatch):
    case = pipeline_case
    case.books = {float(key): value for key, value in case.books.items()}
    original = case.run.record
    failure = RuntimeError("synthetic interruption before second alpha commits")
    def interrupted(slot, body=None, **kwargs):
        if slot == "intents:2023:0.25" and body is not None:
            raise failure
        return original(slot, body, **kwargs)
    monkeypatch.setattr(case.run, "record", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _evaluate(case, case.groups[0])
    monkeypatch.setattr(case.run, "record", original)
    assert original("intents:2023:0") is not None and original("intents:2023:0.0") is None
    assert original("intents:2023:0.25") is None
    _assert_no_effects(case, monkeypatch, lambda: _evaluate(case, case.groups[0]), "EXPERIMENT_IDENTITY_CONFLICT")


@pytest.mark.parametrize("occupied", [None, 0, .25, .5])
def test_duplicate_refusal_preserves_actual_accumulator_journal_and_cas_bytes(pipeline_case, occupied):
    case = pipeline_case
    if occupied is None:
        _evaluate(case, case.groups[0])
    else:
        case.run.record(f"intents:2023:{occupied}", {}, replay=False)
    before = _effect_snapshot(case)
    _error(lambda: _evaluate(case, case.groups[0]), "EXPERIMENT_IDENTITY_CONFLICT")
    assert _effect_snapshot(case) == before


def test_pipeline_mismatched_receipt_pool_scope_refuses_before_intents_or_outcomes(pipeline_case):
    case = pipeline_case
    case.state["wrong_scope"] = True
    _error(lambda: _evaluate(case, case.groups[0]), "INVALID_EXPERIMENT_SPEC")
    assert not any("run-intents:" in slot for slot in case.slots())
    assert not any(call[0] in ("outcomes", "targets") for call in case.calls)


def _effect_snapshot(case):
    journal = tuple(tuple(row) for row in case.conn.execute("SELECT * FROM magnitude_search_records ORDER BY run_id, slot"))
    objects = {str(path.relative_to(case.store.root)): path.read_bytes()
               for path in case.store.root.rglob("*") if path.is_file()}
    return deepcopy((case.books, case.quotes, case.opportunities, case.sections, journal, objects, case.calls))


def _assert_no_effects(case, monkeypatch, action, code="INVALID_EXPERIMENT_SPEC"):
    before = _effect_snapshot(case)
    def forbidden(*args, **kwargs):
        pytest.fail("refused pipeline call reached evaluation/report effects")
    for name in ("residual_pool", "decision_quotes", "read_bridge", "read_benchmark", "outcome_quotes",
                 "load_magnitude_targets", "funded_book", "paired_report", "forecast_report"):
        monkeypatch.setattr(pipeline, name, forbidden)
    monkeypatch.setattr(case.run, "publish", forbidden)
    _error(action, code)
    assert _effect_snapshot(case) == before


@pytest.mark.parametrize("path,value", [
    (("minimum_net_ev_usd",), 2), (("quote_snapshot_id",), "changed-quotes"),
    (("economics", "commission"), 1), (("book", "initial_cash"), 2000),
    (("risk", "maximum_loss_usd"), 2000), (("uncertainty", "resamples"), 5),
    (("report_end",), "2025-02-01"), (("primary_alpha",), 0),
])
@pytest.mark.parametrize("phase", ["evaluate", "finish"])
def test_pipeline_refuses_mutated_policy_before_effects(pipeline_case, monkeypatch, path, value, phase):
    case = pipeline_case
    if phase == "finish":
        for group in case.groups:
            _evaluate(case, group)
    target = case.run.request
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    action = (lambda: _finish(case)) if phase == "finish" else (lambda: _evaluate(case, case.groups[0]))
    _assert_no_effects(case, monkeypatch, action)


@pytest.mark.parametrize("field,value", [("quote_snapshot", {"snapshot_id": "changed"}),
    ("available_benchmarks", {"incumbent": True, "gate": True}), ("run_plan_id", "changed")])
def test_pipeline_refuses_caller_plan_drift_before_effects(pipeline_case, monkeypatch, field, value):
    case = pipeline_case
    case.plan[field] = value
    _assert_no_effects(case, monkeypatch, lambda: _evaluate(case, case.groups[0]))


@pytest.mark.parametrize("phase", ["evaluate", "finish"])
def test_pipeline_requires_existing_seal_before_effects(pipeline_case, monkeypatch, phase):
    case = pipeline_case
    original = case.run.record
    monkeypatch.setattr(case.run, "record", lambda slot, *args, **kwargs:
        None if slot == "seal" else original(slot, *args, **kwargs))
    action = (lambda: _finish(case)) if phase == "finish" else (lambda: _evaluate(case, case.groups[0]))
    _assert_no_effects(case, monkeypatch, action)


@pytest.mark.parametrize("value", [None, object(), {"invalid": object()}])
def test_pipeline_malformed_caller_plan_uses_typed_refusal(pipeline_case, monkeypatch, value):
    case = pipeline_case
    case.plan = value
    _assert_no_effects(case, monkeypatch, lambda: _evaluate(case, case.groups[0]))


def test_decisions_use_detached_seal_even_if_callback_mutates_live_inputs(pipeline_case, monkeypatch):
    case = pipeline_case
    original = pipeline.decision_quotes
    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        case.run.request["minimum_net_ev_usd"] = 100000
        case.plan["available_benchmarks"] = {"incumbent": True, "gate": True}
        return value
    monkeypatch.setattr(pipeline, "decision_quotes", mutate)
    _evaluate(case, case.groups[0])
    for row in _rows(case, "intents:"):
        assert set(row["arms"]) == set(PRIMARY_ARMS)
        assert all(item["selected"] for item in row["arms"].values())


def test_reports_use_detached_seal_even_if_callback_mutates_live_request(pipeline_case, monkeypatch):
    case = pipeline_case
    for group in case.groups:
        _evaluate(case, group)
    original = pipeline.paired_report
    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        case.run.request["primary_alpha"] = 0
        case.run.request["risk"]["maximum_loss_usd"] = 0
        return value
    monkeypatch.setattr(pipeline, "paired_report", mutate)
    result = _finish(case)
    assert result["primary_alpha"] == case.plan["request"]["primary_alpha"]
    assert result["risk_limits"] == case.plan["request"]["risk"]
