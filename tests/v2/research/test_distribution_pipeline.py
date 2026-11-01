"""Standalone outer-decision/report pipeline; real CAS/books, no runner or fits."""
import os
import subprocess
import sys
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


def test_duplicate_outer_intents_refuse_before_reopening_evaluation_outcomes(pipeline_case):
    case = pipeline_case
    _evaluate(case, case.groups[0])
    outcomes = [call for call in case.calls if call[0] in ("outcomes", "targets")]
    _error(lambda: _evaluate(case, case.groups[0]), "EXPERIMENT_IDENTITY_CONFLICT")
    assert [call for call in case.calls if call[0] in ("outcomes", "targets")] == outcomes


def test_pipeline_mismatched_receipt_pool_scope_refuses_before_intents_or_outcomes(pipeline_case):
    case = pipeline_case
    case.state["wrong_scope"] = True
    _error(lambda: _evaluate(case, case.groups[0]), "INVALID_EXPERIMENT_SPEC")
    assert not any("run-intents:" in slot for slot in case.slots())
    assert not any(call[0] in ("outcomes", "targets") for call in case.calls)
