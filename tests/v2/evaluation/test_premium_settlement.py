"""Standalone cash oracles and parity with the landed research adapter."""
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from random import Random

import pytest

from engine.v2.data.errors import DataError
from engine.v2.evaluation.premium_settlement import settle_premiums


def record(identifier="a", **changes):
    return dict(id=identifier, bucket="AAA", selected=True, entry_date=date(2026, 1, 2),
                settlement_date=date(2026, 1, 3), debit=100, credit=120) | changes


def policy(**changes):
    return dict(initial_cash=1000, max_trade_debit=1000, max_bucket_debit=1000,
                max_concurrent=10, report_end=date(2026, 1, 10),
                same_time_order="settlements_before_entries") | changes


def run(records, **changes):
    return settle_premiums(records, **policy(**changes))


def assert_refusal(call):
    with pytest.raises(DataError) as caught:
        call()
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == "invalid premium settlement inputs or arithmetic"
    assert caught.value.problem.details == {}


def test_hand_computed_whole_document_and_daily_curve():
    result = run([record()])
    assert result == {"per_opportunity": [dict(id="a", status="settled", net_profit=20,
        debit=100, capital_days=100)], "curve": [
            dict(date="2026-01-02", cash=900, settlement_equity=1000),
            dict(date="2026-01-03", cash=1020, settlement_equity=1020),
            dict(date="2026-01-10", cash=1020, settlement_equity=1020)],
        "summary": dict(opportunities=1, funded=1, unresolved=0, funded_coverage=1,
            settled_net_profit=20, cash=1020, settlement_equity=1020, capital_days=100,
            settlement_drawdown=0, net_profit_per_opportunity=20),
        "equity_policy": "settled_pnl_open_debit_cost_not_mtm", "capacity_policy": "reserved_until_settlement"}


def test_missing_credit_reserves_cash_bucket_and_capacity_through_report_end():
    result = run([record("b"), record("a", credit=None)], initial_cash=150)
    assert result["per_opportunity"] == [
        dict(id="a", status="unresolved", net_profit=None, debit=100, capital_days=800),
        dict(id="b", status="cash", net_profit=0, debit=0, capital_days=0)]
    assert result["summary"] == dict(opportunities=2, funded=1, unresolved=1, funded_coverage=.5,
        settled_net_profit=0, cash=50, settlement_equity=150, capital_days=800,
        settlement_drawdown=0, net_profit_per_opportunity=None)
    for limits, reason in [(dict(max_bucket_debit=100), "bucket_cap"), (dict(max_concurrent=1), "concurrent_cap")]:
        later = record("b", entry_date=date(2026, 1, 4), settlement_date=date(2026, 1, 5))
        assert run([record("a", credit=None), later], **limits)["per_opportunity"][1]["status"] == reason


@pytest.mark.parametrize("credit", [None, 0, -100, 1_000_000])
def test_future_credit_does_not_change_same_entry_affordability(credit):
    result = run([record("a", credit=credit), record("b")], initial_cash=150)
    assert result["per_opportunity"][0]["debit"] == 100
    assert result["per_opportunity"][1] == dict(id="b", status="cash", net_profit=0, debit=0, capital_days=0)


@pytest.mark.parametrize("changes,reason", [(dict(max_trade_debit=99), "trade_cap"),
    (dict(max_bucket_debit=150), "bucket_cap"), (dict(max_concurrent=1), "concurrent_cap"),
    (dict(initial_cash=150), "cash")])
def test_fixed_cap_precedence_and_lexicographic_ties(changes, reason):
    rows = [record("b"), record("a")]
    result = run(rows, **changes)
    assert result == run(rows[::-1], **changes)
    assert result["per_opportunity"][1]["status"] == reason
    assert result["per_opportunity"][0]["status"] == ("trade_cap" if reason == "trade_cap" else "settled")
    all_limits = run(rows, initial_cash=1, max_trade_debit=1, max_bucket_debit=1, max_concurrent=1)
    assert [row["status"] for row in all_limits["per_opportunity"]] == ["trade_cap", "trade_cap"]


@pytest.mark.parametrize("order,funded", [("settlements_before_entries", 2), ("entries_before_settlements", 1)])
def test_same_day_order_and_capacity_release(order, funded):
    rows = [record(), record("b", entry_date=date(2026, 1, 3), settlement_date=date(2026, 1, 4))]
    result = run(rows, initial_cash=100, max_concurrent=1, max_bucket_debit=100, same_time_order=order)
    assert result["summary"]["funded"] == funded
    assert result["summary"]["settled_net_profit"] == funded * 20
    assert result["per_opportunity"][1]["status"] == ("settled" if funded == 2 else "bucket_cap")


def test_bucket_is_a_separate_cap_and_delayed_settlement_retains_it():
    rows = [record(settlement_date=date(2026, 1, 5)),
            record("b", entry_date=date(2026, 1, 4), settlement_date=date(2026, 1, 6)),
            record("c", bucket="BBB", entry_date=date(2026, 1, 4), settlement_date=date(2026, 1, 6))]
    result = run(rows, max_bucket_debit=100)
    assert [row["status"] for row in result["per_opportunity"]] == ["settled", "bucket_cap", "settled"]
    assert result["summary"]["capital_days"] == 500


def test_missing_entry_unselected_and_unplanned_keep_the_opportunity_denominator():
    rows = [record("a", debit=None), record("b", debit=None, credit=None, settlement_date=None),
            record("c", selected=False), record("d", selected=False, entry_date=None,
                settlement_date=None, debit=None, credit=None)]
    result = run(rows)
    assert [row["status"] for row in result["per_opportunity"]] == ["missing_entry", "missing_entry", "no_trade", "abstained"]
    assert result["summary"]["opportunities"] == 4
    assert result["summary"]["funded"] == result["summary"]["unresolved"] == result["summary"]["net_profit_per_opportunity"] == 0
    assert [row["date"] for row in result["curve"]] == ["2026-01-02", "2026-01-03", "2026-01-10"]


def test_negative_credit_is_retained_even_when_it_overdraws_cash():
    result = run([record(credit=-10)], initial_cash=100)
    assert result["per_opportunity"][0]["net_profit"] == -110
    assert result["summary"]["cash"] == -10
    assert result["summary"]["settlement_drawdown"] == 110


def test_drawdown_samples_daily_settlement_equity_not_intraday_action_order():
    result = run([record("a", credit=0), record("b", credit=200)])
    assert result["summary"]["settlement_drawdown"] == result["summary"]["settled_net_profit"] == 0
    assert result["curve"][0]["cash"] == 800 and result["curve"][0]["settlement_equity"] == 1000


def test_decimal_limit_equality_large_integer_profit_and_capital_days_roundtrip():
    result = run([record("a", debit=.1, credit=.1), record("b", debit=.2, credit=.2)],
                 initial_cash=.3, max_trade_debit=.3, max_bucket_debit=.3)
    assert result["summary"]["funded"] == 2 and result["summary"]["cash"] == .3
    debit = 2**53 + 1
    result = run([record(debit=debit, credit=debit + 1)], initial_cash=debit,
                 max_trade_debit=debit, max_bucket_debit=debit)
    assert result["summary"]["settled_net_profit"] == 1
    assert run([record(credit=101)], initial_cash=1e20)["summary"]["settled_net_profit"] == 1
    rows = [record(str(i), debit=1 / (i + 3), credit=None) for i in range(7)]
    result = run(rows)
    expected = float(sum((Fraction(str(float(Fraction(str(row["debit"])) * 8))) for row in rows), Fraction()))
    assert result["summary"]["capital_days"] == expected


@pytest.mark.parametrize("credit,status", [(120, "settled"), (None, "unresolved")])
def test_settlement_at_report_end_is_covered_without_duplicate_final_curve(credit, status):
    result = run([record(credit=credit, settlement_date=date(2026, 1, 10))])
    assert result["per_opportunity"][0]["status"] == status
    assert result["per_opportunity"][0]["capital_days"] == 800
    assert [row["date"] for row in result["curve"]] == ["2026-01-02", "2026-01-10"]


def test_empty_detached_results_and_callback_once_per_scheduled_action():
    empty = run([])
    assert empty["per_opportunity"] == [] and len(empty["curve"]) == 1
    assert empty["summary"]["funded_coverage"] is empty["summary"]["net_profit_per_opportunity"] is None
    rows = [record(), record("b", selected=False), record("c", debit=None, credit=None, settlement_date=None)]
    before, calls = deepcopy(rows), []
    result = run(rows, on_action=lambda: calls.append(1))
    assert len(calls) == 5  # Includes no-op entry/settlement actions, not only funded actions.
    result["per_opportunity"][0]["debit"] = -1
    assert rows == before


def test_callback_errors_propagate_and_callback_mutation_cannot_change_prepared_records():
    failure = RuntimeError("caller telemetry failed")
    def broken():
        raise failure
    with pytest.raises(RuntimeError) as caught:
        run([record()], on_action=broken)
    assert caught.value is failure
    rows = [record()]
    expected = run(rows)
    assert run(rows, on_action=lambda: rows[0].update(credit=99999, selected=False)) == expected


@pytest.mark.parametrize("field", ["initial_cash", "max_trade_debit", "max_bucket_debit"])
@pytest.mark.parametrize("value", [None, True, 0, -1, "100", float("nan"), float("inf"), 10**400])
def test_invalid_cash_caps_refuse_even_for_empty_records(field, value):
    assert_refusal(lambda: run([], **{field: value}))


@pytest.mark.parametrize("changes", [dict(max_concurrent=0), dict(max_concurrent=True), dict(max_concurrent=1.5),
    dict(report_end="2026-01-10"), dict(report_end=datetime(2026, 1, 10)), dict(report_end=None),
    dict(same_time_order="guess"), dict(same_time_order=None), dict(same_time_order=[]), dict(on_action=1)])
def test_invalid_explicit_policy_refuses(changes):
    assert_refusal(lambda: run([], **changes))


@pytest.mark.parametrize("value", [None, {}, (), iter(())])
def test_records_require_materialized_list(value):
    assert_refusal(lambda: run(value))


@pytest.mark.parametrize("value", [None, [], {}, record() | {"extra": 1}])
def test_record_shape_is_closed(value):
    assert_refusal(lambda: run([value]))


@pytest.mark.parametrize("field", ["id", "bucket"])
@pytest.mark.parametrize("value", [None, "", True, [], 1])
def test_plain_nonempty_ids_and_buckets(field, value):
    assert_refusal(lambda: run([record(**{field: value})]))


@pytest.mark.parametrize("field", ["debit", "credit"])
@pytest.mark.parametrize("value", [True, "100", float("nan"), float("inf"), float("-inf"), 10**400, Decimal("1"), 1j])
def test_plain_finite_money_even_when_unselected(field, value):
    assert_refusal(lambda: run([record(selected=False, **{field: value})]))


@pytest.mark.parametrize("changes", [dict(debit=0), dict(debit=-1), dict(selected=1),
    dict(entry_date="2026-01-02"), dict(entry_date=datetime(2026, 1, 2)), dict(entry_date=date(2026, 1, 11)),
    dict(settlement_date="2026-01-03"), dict(settlement_date=datetime(2026, 1, 3)),
    dict(settlement_date=date(2026, 1, 2)), dict(settlement_date=date(2026, 1, 11)),
    dict(settlement_date=None), dict(entry_date=None),
    dict(selected=False, entry_date=None, settlement_date=None, debit=None, credit=0),
    dict(selected=True, entry_date=None, settlement_date=None, debit=None, credit=None)])
def test_date_cash_selection_consistency(changes):
    assert_refusal(lambda: run([record(**changes)]))


def test_duplicate_and_nonfinite_derived_outputs_refuse_without_input_mutation():
    assert_refusal(lambda: run([record(), record()]))
    rows = [record(debit=1e308, credit=None)]
    before = deepcopy(rows)
    assert_refusal(lambda: run(rows, initial_cash=1e308, max_trade_debit=1e308, max_bucket_debit=1e308))
    assert rows == before
    assert_refusal(lambda: run([record(credit=1e308)], initial_cash=1e308))


def test_required_arguments_are_not_hidden_defaults():
    with pytest.raises(TypeError):
        settle_premiums([])


def test_no_source_storage_or_fit_access(monkeypatch):
    from pathlib import Path

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected I/O")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert run([record()])["summary"]["settled_net_profit"] == 20


def test_normalized_parity_with_landed_book_across_multi_event_policies():
    from experiments.distribution_book import funded_book

    rng = Random(931)
    for _ in range(120):
        cfg = dict(initial_cash=rng.choice([.3, 150, 500, 1e20]), contracts_per_trade=2,
            max_trade_debit=rng.choice([.3, 100, 300]), max_ticker_debit=rng.choice([.3, 150, 500]),
            max_concurrent=rng.choice([1, 2, 10]), settlement_delay_days=rng.randrange(3),
            same_time_order=rng.choice(["settlements_before_entries", "entries_before_settlements"]))
        intents, quotes, normalized = [], {}, []
        for index in range(rng.randrange(21)):
            event, ticker = str(index), str(index % 3)
            entry = date(2026, 1, rng.randrange(2, 15)) if rng.randrange(7) else None
            selected = bool(rng.randrange(2)) if entry is not None else False
            intents.append(dict(event_id=event, ticker=ticker, event_date="2026-01-02", selected=selected,
                decision_at=(entry - timedelta(days=1)).isoformat() if entry else None,
                entry_at=entry.isoformat() if entry else None))
            quote = None
            if entry is not None and rng.randrange(5):
                quote = dict(entry_cash=rng.choice([None, .1, .2, 100, 150]),
                    exit_cash=rng.choice([None, -.1, 0, .3, 120, 200]), contracts=2, multiplier=100,
                    exit_at=(entry + timedelta(days=rng.randrange(1, 5))).isoformat())
                quotes[event] = quote
            normalized.append(dict(id=event, bucket=ticker, selected=selected, entry_date=entry,
                settlement_date=date.fromisoformat(quote["exit_at"]) + timedelta(days=cfg["settlement_delay_days"]) if quote else None,
                debit=quote["entry_cash"] if quote else None, credit=quote["exit_cash"] if quote else None))
        original = funded_book(intents, quotes, config=cfg, report_end="2026-01-31")
        expected = {key: original[key] for key in ("curve", "summary", "equity_policy", "capacity_policy")}
        expected["per_opportunity"] = [{"id": row["event_id"],
            **{key: row[key] for key in ("net_profit", "debit", "capital_days")},
            "status": "bucket_cap" if row["status"] == "ticker_cap" else row["status"]}
            for row in original["per_event"]]
        args = dict(initial_cash=cfg["initial_cash"], max_trade_debit=cfg["max_trade_debit"],
            max_bucket_debit=cfg["max_ticker_debit"], max_concurrent=cfg["max_concurrent"],
            same_time_order=cfg["same_time_order"], report_end=date(2026, 1, 31))
        before = deepcopy(normalized)
        assert settle_premiums(normalized, **args) == expected
        assert settle_premiums(normalized[::-1], **args) == expected
        assert normalized == before
