"""Runner acceptance over fake fit/source seams and real pipeline, journal and CAS."""
from hashlib import sha256

import pytest

from engine.v2.ops.errors import OpsError, fail
from experiments import distribution_run as runner
from experiments.distribution_run_request import PRIMARY_ARMS
from experiments.magnitude_search_shards import iter_section
from tests.v2.research.test_distribution_run import case as case


def observe(monkeypatch, run, before_evaluation=None):
    calls = {"evaluation": [], "finished": [], "published": []}
    evaluate, finish, publish = runner.evaluate_outer, runner.finish_reports, run.publish
    def evaluated(value, group, fits, results, plan, books, quotes, opportunities, sections, calendar):
        calls["evaluation"].append(group["outer"]["year"])
        if before_evaluation is not None:
            before_evaluation()
        assert value is run and set(results) == {1, 3, 5}
        assert tuple(books) == tuple(quotes) == tuple(plan["request"]["alphas"]) == (0, .25, .5)
        for horizon, result in results.items():
            receipt = result["receipt"]
            assert receipt["claim"]["trial"]["horizon"] == horizon
            assert receipt["claim"]["role"] == "outer" and receipt["claim"]["inner_year"] is None
            assert run.record("fit:" + receipt["claim"]["trial"]["trial_id"]) == {
                "result_id": receipt["result_id"], "status": "bounded_completed"}
        return evaluate(value, group, fits, results, plan, books, quotes, opportunities, sections, calendar)
    def finished(*args):
        calls["finished"].append(True)
        return finish(*args)
    def published(name, rows):
        section = publish(name, rows)
        calls["published"].append(section)
        return section
    monkeypatch.setattr(runner, "evaluate_outer", evaluated)
    monkeypatch.setattr(runner, "finish_reports", finished)
    monkeypatch.setattr(run, "publish", published)
    return calls


def snapshot(case):
    records = tuple(tuple(row) for row in case.conn.execute(
        "SELECT slot, ref_json FROM magnitude_search_records ORDER BY slot"))
    objects = tuple(sorted((str(path.relative_to(case.store.root)), sha256(path.read_bytes()).hexdigest())
        for path in case.store.root.rglob("*") if path.is_file()))
    return records, objects, case.conn.total_changes


def rows(case, section):
    return list(iter_section(case.store, section["name"], section["descriptor"], lambda row: (row["position"],)))


def test_actual_runner_supplies_all_horizons_and_sealed_alphas_to_pipeline(case, monkeypatch):
    run = runner.DistributionRun(case.trials, case.request)
    run.seal(clock=case.clock)
    calls = observe(monkeypatch, run)
    attributes = set(vars(run))
    result = run.run(clock=case.clock)
    assert calls["evaluation"] == ["2023", "2024"] and calls["finished"] == [True]
    assert set(vars(run)) == attributes  # No books/quotes/opportunities/results retained on the runner.
    names = {section["name"] for section in result["sections"]}
    assert names == {f"{kind}:{year}:{alpha}" for kind in ("intents", "profit-scores")
        for year in ("2023", "2024") for alpha in (0, .25, .5)} | {
        f"scores:{year}:{horizon}" for year in ("2023", "2024") for horizon in (1, 3, 5)} | {
        f"{kind}:{alpha}:{arm}" for kind in ("book", "curve") for alpha in (0, .25, .5) for arm in PRIMARY_ARMS}
    assert calls["published"] == result["sections"]
    for section in result["sections"]:
        if section["name"].startswith("scores:"):
            _, year, horizon = section["name"].split(":")
            assert {(row["event_id"], row["horizon"], row["arm"]) for row in rows(case, section)} == {
                (year + "-" + ticker, int(horizon), arm) for ticker in ("AAA", "BBB") for arm in PRIMARY_ARMS}
    assert run.record("completed") == result and run.record("failed") is None


@pytest.mark.parametrize("missing", [1, 3, 5])
def test_fault_injected_missing_horizon_refuses_before_any_evaluation_or_report(case, monkeypatch, missing):
    run = runner.DistributionRun(case.trials, case.request)
    run.seal(clock=case.clock)
    calls = observe(monkeypatch, run)
    # Native preparation fixes the recipe. Inject an impossible truncated roster
    # only into the fake post-seal adapter to test the runner's own handoff guard.
    case.trials.recipe["horizons"] = [h for h in (5, 3, 1) if h != missing]
    with pytest.raises(OpsError) as caught:
        run.run(clock=case.clock)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "distribution outer results require horizons 1,3,5").problem
    assert calls == {"evaluation": [], "finished": [], "published": []}
    assert not any(call[0] in ("decision", "outcome", "targets") for call in case.calls)
    assert sum(slot.startswith("distribution:run-fit:") for slot in case.slots()) == 4
    assert run.record("failed")["code"] == "INVALID_EXPERIMENT_SPEC" and run.record("completed") is None


def test_book_roster_uses_detached_admitted_alphas_not_live_request(case, monkeypatch):
    run = runner.DistributionRun(case.trials, case.request)
    sealed = run.seal(clock=case.clock)
    admitted = run.admit
    def admit():
        plan = admitted()
        run.request["alphas"][:] = [0]  # Fault injection after real admission, before roster allocation.
        assert plan["request"]["alphas"] == sealed["request"]["alphas"] == [0, .25, .5]
        return plan
    monkeypatch.setattr(run, "admit", admit)
    # Restore policy before the REAL pipeline checks its seal: this test isolates
    # the intervening roster allocation, not permission to consume mutated policy.
    calls = observe(monkeypatch, run, lambda: run.request["alphas"].__setitem__(slice(None), [0, .25, .5]))
    result = run.run(clock=case.clock)
    assert calls["evaluation"] == ["2023", "2024"]
    assert [item["alpha"] for item in result["reports"]] == sealed["request"]["alphas"]


def test_failed_second_group_preserves_prefix_without_finalization_or_retry_state(case, monkeypatch):
    run = runner.DistributionRun(case.trials, case.request)
    run.seal(clock=case.clock)
    calls = observe(monkeypatch, run)
    attributes = set(vars(run))
    failure = fail("RESOURCE_LIMIT_EXCEEDED", "synthetic second-group fit exhaustion")
    case.state.update(failure=failure, fail_on=8)
    with pytest.raises(OpsError) as caught:
        run.run(clock=case.clock)
    assert caught.value is failure
    assert calls["evaluation"] == ["2023"] and calls["finished"] == []
    assert {section["name"] for section in calls["published"]} == {
        f"{kind}:2023:{alpha}" for kind in ("intents", "profit-scores") for alpha in (0, .25, .5)} | {
        f"scores:2023:{horizon}" for horizon in (1, 3, 5)}
    assert all(rows(case, section) for section in calls["published"])
    assert sum(call[0] == "fit" for call in case.calls) == 8
    assert sum(slot.startswith("distribution:run-fit:") for slot in case.slots()) == 7
    assert run.record("failed")["code"] == failure.code and run.record("completed") is None
    assert set(vars(run)) == attributes
    durable = snapshot(case)
    execution = [call for call in case.calls if call[0] in ("fit", "admit", "decision", "outcome", "targets")]
    case.state["failure"] = None
    fresh = runner.DistributionRun(case.trials, case.request)
    for attempted in (run, fresh):
        with pytest.raises(OpsError) as retry:
            attempted.run(clock=case.clock)
        assert retry.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
        with pytest.raises(OpsError) as collect:
            attempted.collect()
        assert collect.value.code == "INVALID_EXPERIMENT_SPEC"
    assert snapshot(case) == durable
    assert [call for call in case.calls if call[0] in ("fit", "admit", "decision", "outcome", "targets")] == execution
    assert calls["evaluation"] == ["2023"] and calls["finished"] == []
    assert set(vars(run)) == attributes and set(vars(fresh)) == {"trials", "request"}


def test_zero_funded_runner_output_is_diagnostic_not_risk_or_promotion_proof(case):
    case.state["missing_pool"] = True
    run = runner.DistributionRun(case.trials, case.request)
    run.seal(clock=case.clock)
    result = run.run(clock=case.clock)
    for item in result["reports"]:
        report = item["report"]
        assert report["opportunities"] == 4 and report["uplift_met"] is True
        assert report["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 0
        for arm in PRIMARY_ARMS:
            summary = report["arms"][arm]
            assert tuple(summary[key] for key in ("funded", "unresolved", "settled_net_profit", "settlement_drawdown")) == (0, 0, 0, 0)
            assert report["risk_checks"][arm] == {"loss_limit_met": True, "settled_drawdown_met": True, "completeness_met": None}
    books = [section for section in result["sections"] if section["name"].startswith("book:")]
    assert len(books) == 9
    assert all(row["status"] == "no_trade" and row["net_profit"] == 0 for section in books for row in rows(case, section))
    assert result["promotion_authority"] == result["final_read_authority"] == "none"
    assert result["current_strategy_superiority_claim"] is False
    assert result["capacity"] == "not_established_no_depth" and result["calibration_identity"] == "uncalibrated"
