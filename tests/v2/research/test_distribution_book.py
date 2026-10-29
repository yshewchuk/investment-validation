"""Synthetic premium-funded book oracles, without market/source access."""
from copy import deepcopy

import pytest

from engine.v2.ops.errors import OpsError
from experiments.distribution_book import funded_book, validate_book_config


def config(**changes):
    return dict(initial_cash=1000, contracts_per_trade=2, max_trade_debit=1000,
        max_ticker_debit=1000, max_concurrent=10, settlement_delay_days=0,
        same_time_order="settlements_before_entries") | changes


def intent(event="a", *, entry="2026-01-02", decision="2026-01-01", ticker="AAA", selected=True):
    return dict(event_id=event, ticker=ticker, event_date=entry or "2026-01-02",
        decision_at=decision, entry_at=entry, selected=selected)


def quote(*, entry=100, exit=120, day="2026-01-03", contracts=2, multiplier=100):
    return dict(entry_cash=entry, exit_cash=exit, exit_at=day, contracts=contracts, multiplier=multiplier)


def run(intents, quotes, **changes):
    return funded_book(intents, quotes, config=config(**changes), report_end="2026-01-10")


def test_total_cash_already_includes_actual_multiplier_quantity_fees_and_slippage():
    quantity, multiplier, entry_mid, exit_mid = 2, 10, 3, 4
    entry = quantity * multiplier * (entry_mid + .1) + max(5, quantity * .5) + quantity * .2
    exit = quantity * multiplier * (exit_mid - .1) - max(5, quantity * .5) - quantity * .2
    result = run([intent()], {"a": quote(entry=entry, exit=exit, multiplier=multiplier)})
    row, summary = result["per_event"][0], result["summary"]
    assert row["net_profit"] == pytest.approx(exit - entry)
    assert summary["cash"] == pytest.approx(1000 + exit - entry)
    assert summary["capital_days"] == entry
    assert row["quote"]["contracts"] == quantity
    assert row["quote"]["multiplier"] == multiplier


def test_missing_exit_remains_funded_does_not_erase_opportunity_or_free_cash():
    rows = [intent("b"), intent("a")]
    result = run(rows, {"a": quote(exit=None), "b": quote()}, initial_cash=150)
    a, b = result["per_event"]
    assert a["status"] == "unresolved" and a["net_profit"] is None
    assert b["status"] == "cash" and b["net_profit"] == 0
    assert result["summary"] == dict(opportunities=2, funded=1, unresolved=1,
        funded_coverage=.5, settled_net_profit=0, cash=50, settlement_equity=150,
        capital_days=800, settlement_drawdown=0, net_profit_per_opportunity=None)
    assert result["curve"][-1]["date"] == "2026-01-10"


@pytest.mark.parametrize("exit_cash", [120, None], ids=["known-quote", "missing-quote"])
@pytest.mark.parametrize("exit_day,delay", [("2026-01-11", 0), ("2026-01-10", 1)],
                         ids=["exit-after-end", "settlement-after-end"])
def test_complete_report_horizon_refuses_uncovered_exit_or_settlement(exit_cash, exit_day, delay):
    rows, quotes = [intent()], {"a": quote(exit=exit_cash, day=exit_day)}
    before = deepcopy((rows, quotes))
    with pytest.raises(OpsError) as caught:
        run(rows, quotes, settlement_delay_days=delay)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert (rows, quotes) == before


@pytest.mark.parametrize("exit_cash,status,profit", [(120, "settled", 20), (None, "unresolved", None)])
def test_settlement_at_report_end_accepts_known_or_missing_exit_quote(exit_cash, status, profit):
    result = run([intent()], {"a": quote(exit=exit_cash, day="2026-01-08")}, settlement_delay_days=2)
    row, summary = result["per_event"][0], result["summary"]
    assert row["status"] == status and row["net_profit"] == profit
    assert row["debit"] == 100 and row["capital_days"] == 800
    assert summary["funded"] == 1 and summary["unresolved"] == int(exit_cash is None)
    assert summary["net_profit_per_opportunity"] == profit
    assert result["curve"][-1]["date"] == "2026-01-10"


def test_future_exit_value_does_not_select_or_fund_simultaneous_entries():
    rows = [intent("a"), intent("b")]
    outputs = [run(rows, {"a": quote(exit=value), "b": quote()}, initial_cash=150)
        for value in (None, 0, 1_000_000)]
    assert [out["per_event"][1]["status"] for out in outputs] == ["cash"] * 3
    assert all(out["per_event"][0]["debit"] == 100 for out in outputs)


@pytest.mark.parametrize("quotes", [{}, {"a": None}, {"a": quote(entry=None)}])
def test_missing_entry_is_execution_refusal_not_unresolved_funded_trade(quotes):
    result = run([intent()], quotes)
    assert result["per_event"][0]["status"] == "missing_entry"
    assert result["summary"]["funded"] == result["summary"]["unresolved"] == 0
    assert result["summary"]["net_profit_per_opportunity"] == 0


def test_unselected_and_unplanned_opportunities_are_zero_not_funded():
    rows = [intent("a", selected=False), intent("b", selected=False, decision=None, entry=None)]
    result = run(rows, {"a": quote(exit=None)})
    assert [row["status"] for row in result["per_event"]] == ["no_trade", "abstained"]
    assert result["summary"]["opportunities"] == 2
    assert result["summary"]["net_profit_per_opportunity"] == 0


@pytest.mark.parametrize("changes,reason", [
    ({"max_trade_debit": 99}, "trade_cap"), ({"max_ticker_debit": 150}, "ticker_cap"),
    ({"max_concurrent": 1}, "concurrent_cap"), ({"initial_cash": 150}, "cash")])
def test_caps_have_deterministic_same_time_competition(changes, reason):
    result = run([intent("b"), intent("a")], {"a": quote(), "b": quote()}, **changes)
    assert result["per_event"][1]["status"] == reason
    if reason != "trade_cap":
        assert result["per_event"][0]["status"] == "settled"


def test_exact_affordability_and_limit_equality_are_admitted():
    result = run([intent()], {"a": quote()}, initial_cash=100, max_trade_debit=100,
        max_ticker_debit=100, max_concurrent=1)
    assert result["summary"]["funded"] == 1
    assert result["summary"]["cash"] == 120


@pytest.mark.parametrize("order,funded", [("settlements_before_entries", 2), ("entries_before_settlements", 1)])
def test_same_date_settlement_order_is_explicit(order, funded):
    rows = [intent("a"), intent("b", decision="2026-01-02", entry="2026-01-03")]
    result = run(rows, {"a": quote(), "b": quote(day="2026-01-04")}, initial_cash=150,
        same_time_order=order, max_concurrent=1)
    assert result["summary"]["funded"] == funded


def test_calendar_settlement_delay_reserves_capital_and_capacity_past_exit():
    rows = [intent("a"), intent("b", decision="2026-01-03", entry="2026-01-04")]
    result = run(rows, {"a": quote(), "b": quote(day="2026-01-05")}, settlement_delay_days=2,
        max_concurrent=1)
    assert result["per_event"][1]["status"] == "concurrent_cap"
    assert result["per_event"][0]["capital_days"] == 300
    assert result["capacity_policy"] == "reserved_until_settlement"


def test_ticker_cap_is_separate_and_released_after_settlement():
    rows = [intent("a"), intent("b", ticker="BBB"),
        intent("c", decision="2026-01-03", entry="2026-01-04")]
    result = run(rows, {"a": quote(), "b": quote(), "c": quote(day="2026-01-05")}, max_ticker_debit=100)
    assert result["summary"]["funded"] == 3


def test_settlement_equity_drawdown_is_not_entry_debit_or_intraday_mark():
    rows = [intent("a"), intent("b", decision="2026-01-03", entry="2026-01-04")]
    result = run(rows, {"a": quote(exit=150), "b": quote(exit=60, day="2026-01-05")})
    assert result["curve"][0]["cash"] == 900
    assert result["curve"][0]["settlement_equity"] == 1000
    assert result["summary"]["settlement_drawdown"] == 40
    assert result["summary"]["settlement_equity"] == 1010
    assert result["equity_policy"] == "settled_pnl_open_debit_cost_not_mtm"


def test_same_day_settlements_are_one_equity_observation_not_artificial_order_risk():
    result = run([intent("a"), intent("b")], {"a": quote(exit=0), "b": quote(exit=200)})
    assert result["summary"]["settlement_drawdown"] == 0
    assert result["summary"]["settled_net_profit"] == 0


def test_exit_fees_can_exceed_proceeds_without_silent_clipping():
    result = run([intent()], {"a": quote(exit=-10)}, initial_cash=100)
    assert result["summary"]["cash"] == -10
    assert result["per_event"][0]["net_profit"] == -110
    assert result["summary"]["settlement_drawdown"] == 110


def test_empty_book_and_detached_outputs():
    assert run([], {})["summary"]["funded_coverage"] is None
    rows, quotes, cfg = [intent()], {"a": quote()}, config()
    before = deepcopy((rows, quotes, cfg))
    result = funded_book(rows, quotes, config=cfg, report_end="2026-01-10")
    assert (rows, quotes, cfg) == before
    result["per_event"][0]["quote"]["entry_cash"] = -1
    result["config"]["initial_cash"] = -1
    assert (rows, quotes, cfg) == before


@pytest.mark.parametrize("changes", [
    {"initial_cash": 0}, {"initial_cash": float("inf")}, {"initial_cash": 10**1000},
    {"contracts_per_trade": True}, {"contracts_per_trade": 1.5}, {"max_concurrent": 0},
    {"max_trade_debit": -1}, {"max_ticker_debit": float("nan")},
    {"settlement_delay_days": -.5}, {"settlement_delay_days": True}, {"same_time_order": "guess"}])
def test_invalid_explicit_config_refuses(changes):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        validate_book_config(config(**changes))


def test_missing_or_extra_config_no_defaults():
    for cfg in ({}, config() | {"unknown": 1}):
        with pytest.raises(OpsError):
            validate_book_config(cfg)
    with pytest.raises(TypeError):
        funded_book([], {}, config=config())


@pytest.mark.parametrize("change", [
    {"contracts": 1}, {"contracts": True}, {"multiplier": 0}, {"multiplier": 1.5},
    {"entry_cash": -1}, {"entry_cash": 0}, {"exit_cash": float("nan")},
    {"exit_at": "2026-01-02"}, {"exit_at": "2026-01-11"}, {"exit_at": "20260103"}])
def test_invalid_quote_refuses(change):
    with pytest.raises(OpsError):
        run([intent()], {"a": quote() | change})


def test_duplicate_unknown_and_invalid_clock_inputs_refuse():
    for rows, quotes in [([intent(), intent()], {}), ([intent()], {"unknown": quote()}),
            ([intent(decision="2026-01-02")], {}), ([intent(entry=None)], {}),
            ([intent(entry=None, decision=None)], {}), ([intent(entry="20260102")], {})]:
        with pytest.raises(OpsError):
            run(rows, quotes)


def test_nonfinite_derived_arithmetic_refuses():
    with pytest.raises(OpsError):
        run([intent()], {"a": quote(entry=1e308)}, initial_cash=1e308,
            max_trade_debit=1e308, max_ticker_debit=1e308)


def test_decimal_cash_and_ticker_equality_do_not_refuse_from_binary_roundoff():
    rows = [intent("a"), intent("b"), intent("c", decision="2026-01-03", entry="2026-01-04")]
    quotes = {"a": quote(entry=.1, exit=.1), "b": quote(entry=.2, exit=.2),
        "c": quote(entry=.3, exit=.3, day="2026-01-05")}
    result = run(rows, quotes, initial_cash=.3, max_ticker_debit=.3, max_trade_debit=.3)
    assert result["summary"]["funded"] == 3
    assert result["summary"]["cash"] == .3
    assert result["summary"]["settled_net_profit"] == 0


def test_small_realized_profit_does_not_disappear_against_large_initial_cash():
    result = run([intent()], {"a": quote(entry=100, exit=101)}, initial_cash=1e20)
    assert result["summary"]["settled_net_profit"] == 1
    assert result["summary"]["net_profit_per_opportunity"] == 1


def test_exact_integer_money_is_not_rounded_before_settlement():
    debit = 2**53 + 1
    result = run([intent()], {"a": quote(entry=debit, exit=debit + 1)},
        initial_cash=debit, max_trade_debit=debit, max_ticker_debit=debit)
    assert result["summary"]["settled_net_profit"] == 1


def test_order_invariance_and_cash_conservation_random_integer_oracle():
    import random
    from fractions import Fraction

    rng = random.Random(82)
    for _ in range(20):
        rows, quotes = [intent(str(i), ticker=str(i % 3)) for i in range(30)], {}
        for row in rows:
            quotes[row["event_id"]] = quote(entry=rng.randrange(10, 80), exit=rng.choice([None, rng.randrange(-5, 150)]))
        changes = dict(initial_cash=500, max_trade_debit=65, max_ticker_debit=150, max_concurrent=8)
        result = run(rows, quotes, **changes)
        assert result == run(list(reversed(rows)), dict(reversed(list(quotes.items()))), **changes)
        funded = [row for row in result["per_event"] if row["debit"]]
        settled = [row for row in funded if row["status"] == "settled"]
        expected_cash = 500 - sum(row["debit"] for row in funded) + sum(row["quote"]["exit_cash"] for row in settled)
        expected_profit = sum(row["quote"]["exit_cash"] - row["quote"]["entry_cash"] for row in settled)
        assert result["summary"]["cash"] == expected_cash
        assert result["summary"]["settled_net_profit"] == expected_profit
        assert result["summary"]["capital_days"] == float(sum(
            Fraction(str(row["debit"])) * (1 if row["status"] == "settled" else 8) for row in funded))
