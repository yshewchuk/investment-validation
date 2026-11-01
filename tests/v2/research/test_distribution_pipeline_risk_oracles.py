"""Independent final-report risk arithmetic and live comparison-flip mutants."""
import ast
import inspect
import textwrap

import pytest

from experiments import distribution_pipeline as pipeline
from tests.v2.research.test_distribution_run_plan import make_plan
from tests.v2.research.test_distribution_run_plan import plan_case as plan_case

ARMS = ("histogram", "same_mean", "matched_point")
FLAGS = ("loss_limit_met", "settled_drawdown_met", "completeness_met")
RISK_KEYS = ("minimum_uplift_usd", "maximum_loss_usd", "maximum_settled_drawdown_usd", "completeness_floor")
# Four $100 entries settle chronologically at 120, 50, 110, 100.
# Histogram equity: 1000 -> 1020 -> 970 -> 980 -> 980; net=-20, drawdown=50.
# same_mean selects only the -50 event; matched_point selects only the +20 event.
# Thus histogram-minus-same_mean uplift=(-20-(-50))/4=7.5, not zero.
# Distinct supplied intents exercise the final-report API, not a claim that the
# unchanged linear-bridge EV selector produces distinct same-mean selections.
CASES = {
    # limits (uplift, loss, drawdown, completeness), missing last exit, no funding,
    # uplift verdict, and exact (loss, drawdown, completeness) flags by arm.
    "loose": ((7, 21, 51, .99), False, False, True, ((True, True, True), (False, True, True), (True, True, True))),
    "equal": ((7.5, 20, 50, 1), False, False, True, ((True, True, True), (False, True, True), (True, True, True))),
    "tight": ((8, 19, 49, 1), False, False, False, ((False, False, True), (False, False, True), (True, True, True))),
    "partial_loose": ((0, 20, 51, .5), True, False, None, ((None, True, True), (False, True, True), (True, True, True))),
    "partial_equal": ((0, 20, 50, .75), True, False, None, ((None, True, True), (False, True, True), (True, True, True))),
    "partial_tight": ((0, 20, 49, 1), True, False, None, ((None, False, False), (False, False, True), (True, True, True))),
    "zero": ((0, 0, 0, 1), False, True, True, ((True, True, None),) * 3),
}


def exercise(case, name):
    limits, missing, empty = CASES[name][:3]
    run = make_plan(case, risk=dict(zip(RISK_KEYS, limits)))
    run.seal(clock=case.clock)  # Every scenario's policy is fixed BEFORE sealing.
    books, quotes, opportunities = {}, {}, []
    intents, prices = [], {}
    for event, credit in zip(case.events, (120, 50, 110, 100)):
        key, ticker, year = event["event_id"], event["ticker"], event["event_date"][:4]
        decision = year + "-06-02"
        opportunities.append(dict(event_id=key, ticker=ticker, decision_date=decision))
        intents.append(dict(event_id=key, ticker=ticker, event_date=event["event_date"],
            decision_at=decision, entry_at=event["event_date"], selected=False))
        prices[key] = dict(entry_cash=100, exit_cash=None if missing and key == "2024-BBB" else credit,
            exit_at=year + ("-06-04" if ticker == "AAA" else "-06-05"), contracts=1, multiplier=100)
    for alpha in run.request["alphas"]:
        books[alpha] = {arm: [{**row, "selected": not empty and (arm == "histogram" or
            row["event_id"] == ("2023-BBB" if arm == "same_mean" else "2023-AAA"))} for row in intents] for arm in ARMS}
        quotes[alpha] = prices
    return pipeline.finish_reports(run, books, quotes, opportunities, [])


def assert_risk_oracles(result, name):
    limits, missing, empty, uplift, flags = CASES[name]
    # Missing final credit leaves its $100 debit reserved, not a fabricated loss:
    # histogram funded=4, unresolved=1, completeness=3/4, equity=980, cash=880.
    # Known paired deltas total30 over three scoreable events, but full uplift=None.
    summaries = {
        "histogram": (4, int(missing), -20, 50, 880 if missing else 980, 980),
        "same_mean": (1, 0, -50, 50, 950, 950),
        "matched_point": (1, 0, 20, 0, 1020, 1020),
    } if not empty else dict.fromkeys(ARMS, (0, 0, 0, 0, 1000, 1000))
    assert len(result["reports"]) == 3
    assert result["risk_limits"] == dict(zip(RISK_KEYS, limits))
    assert result["promotion_authority"] == "none" and result["current_strategy_superiority_claim"] is False
    for item in result["reports"]:
        report = item["report"]
        assert report["opportunities"] == 4 and report["final_read_authority"] == "none"
        assert set(report["arms"]) == set(report["risk_checks"]) == set(ARMS)
        for arm, expected_flags in zip(ARMS, flags):
            summary = report["arms"][arm]
            actual = tuple(summary[key] for key in ("funded", "unresolved", "settled_net_profit",
                "settlement_drawdown", "cash", "settlement_equity"))
            assert actual == summaries[arm]
            for flag, expected in zip(FLAGS, expected_flags):
                assert report["risk_checks"][arm][flag] is expected, f"{flag} exact verdict"
        primary = report["contrasts"]["same_mean"]
        assert primary["net_delta_per_opportunity"] == (None if missing else 0 if empty else 7.5)
        assert primary["scoreable_delta"] == (10 if missing else 0 if empty else 7.5)
        assert primary["scoreable_events"] == (3 if missing else 4)
        assert primary["unresolved_events"] == (["2024-BBB"] if missing else [])
        assert report["uplift_met"] is uplift, "uplift_met exact verdict"


@pytest.mark.parametrize("name", CASES)
def test_final_report_risk_verdicts_match_exact_settlement_oracles(plan_case, name):
    assert_risk_oracles(exercise(plan_case, name), name)


def mutate_comparison(monkeypatch, risk_key, replacement):
    # Compile the ACTUAL installed function with one operator changed. Keep its
    # sealed-policy admission, real funded books, report math and CAS writes;
    # no test-side reimplementation of the decision being tested is substituted.
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(pipeline.finish_reports))))
    comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Compare) and any(
        isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name) and child.value.id == "risk"
        and isinstance(child.slice, ast.Constant) and child.slice.value == risk_key for child in ast.walk(node))]
    assert len(comparisons) == 1 and len(comparisons[0].ops) == 1
    original = ast.LtE if risk_key == "maximum_settled_drawdown_usd" else ast.GtE
    assert isinstance(comparisons[0].ops[0], original)
    comparisons[0].ops = [replacement()]
    namespace = vars(pipeline).copy()
    exec(compile(ast.fix_missing_locations(tree), "<finish_reports-comparison-mutant>", "exec"), namespace)
    monkeypatch.setattr(pipeline, "finish_reports", namespace["finish_reports"])


@pytest.mark.parametrize("flag,risk_key,replacement,name", [
    ("uplift_met", "minimum_uplift_usd", ast.LtE, "loose"),
    ("uplift_met", "minimum_uplift_usd", ast.Gt, "equal"),
    ("loss_limit_met", "maximum_loss_usd", ast.LtE, "loose"),
    ("loss_limit_met", "maximum_loss_usd", ast.Gt, "equal"),
    ("settled_drawdown_met", "maximum_settled_drawdown_usd", ast.GtE, "loose"),
    ("settled_drawdown_met", "maximum_settled_drawdown_usd", ast.Lt, "equal"),
    ("completeness_met", "completeness_floor", ast.LtE, "partial_loose"),
    ("completeness_met", "completeness_floor", ast.Gt, "partial_equal"),
])
def test_risk_oracles_kill_live_comparison_flips(plan_case, monkeypatch, flag, risk_key, replacement, name):
    mutate_comparison(monkeypatch, risk_key, replacement)
    result = exercise(plan_case, name)
    with pytest.raises(AssertionError, match=rf"\b{flag} exact verdict\b"):
        assert_risk_oracles(result, name)
