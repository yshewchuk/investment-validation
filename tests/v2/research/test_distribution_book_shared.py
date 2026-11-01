"""Distribution-to-shared-settlement mapping and unchanged refusal boundaries."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import date

import pytest

from engine.v2.data.errors import fail as data_fail
from engine.v2.ops.errors import OpsError
from experiments import distribution_book as book
from tests.v2.research.test_distribution_book import config, intent, quote


def test_shared_call_normalizes_only_admitted_economics_and_rejoins_whole_document(monkeypatch):
    original, calls = book.settle_premiums, []
    def capture(records, **kwargs):
        calls.append((deepcopy(records), kwargs))
        return original(records, **kwargs)
    monkeypatch.setattr(book, "settle_premiums", capture)
    rows, quotes, cfg = [intent()], {"a": quote()}, config(settlement_delay_days=2)
    before = deepcopy((rows, quotes, cfg))
    result = book.funded_book(rows, quotes, config=cfg, report_end="2026-01-10")
    assert len(calls) == 1
    records, kwargs = calls[0]
    assert records == [dict(id="a", bucket="AAA", selected=True, entry_date=date(2026, 1, 2),
        settlement_date=date(2026, 1, 5), debit=100, credit=120)]
    assert {key: value for key, value in kwargs.items() if key != "on_action"} == dict(initial_cash=1000,
        max_trade_debit=1000, max_bucket_debit=1000, max_concurrent=10, report_end=date(2026, 1, 10),
        same_time_order="settlements_before_entries")
    assert callable(kwargs["on_action"])
    assert result == dict(schema_version="distribution_book.v1", config=cfg, report_end="2026-01-10",
        per_event=[rows[0] | dict(quote=quotes["a"], status="settled", net_profit=20, debit=100, capital_days=300)],
        curve=[dict(date="2026-01-02", cash=900, settlement_equity=1000),
               dict(date="2026-01-05", cash=1020, settlement_equity=1020),
               dict(date="2026-01-10", cash=1020, settlement_equity=1020)],
        summary=dict(opportunities=1, funded=1, unresolved=0, funded_coverage=1,
            settled_net_profit=20, cash=1020, settlement_equity=1020, capital_days=300,
            settlement_drawdown=0, net_profit_per_opportunity=20),
        equity_policy="settled_pnl_open_debit_cost_not_mtm", capacity_policy="reserved_until_settlement")
    assert list(result) == ["schema_version", "config", "report_end", "per_event", "curve", "summary", "equity_policy", "capacity_policy"]
    assert list(result["per_event"][0]) == list(rows[0]) + ["quote", "status", "net_profit", "debit", "capital_days"]
    assert (rows, quotes, cfg) == before


@pytest.mark.parametrize("order", ["settlements_before_entries", "entries_before_settlements"])
def test_previously_valid_order_string_subclasses_are_normalized_without_tightening(order):
    class Order(str):
        pass
    cfg = config(same_time_order=Order(order), max_ticker_debit=100)
    result = book.funded_book([intent("a"), intent("b")], {"a": quote(), "b": quote()},
                              config=cfg, report_end="2026-01-10")
    assert type(result["config"]["same_time_order"]) is Order
    assert result["per_event"][1]["status"] == "ticker_cap"
    assert "per_opportunity" not in result


@pytest.mark.parametrize("rows,quotes,cfg,end", [
    ([intent()], {"a": quote()}, config(contracts_per_trade=True), "2026-01-10"),
    ([intent()], {"a": quote(contracts=1)}, config(), "2026-01-10"),
    ([intent()], {"a": quote(multiplier=0)}, config(), "2026-01-10"),
    ([intent(decision="2026-01-02")], {"a": quote()}, config(), "2026-01-10"),
    ([intent()], {"a": quote(day="2026-01-02")}, config(), "2026-01-10"),
    ([intent()], {"a": quote(day="2026-01-10")}, config(settlement_delay_days=1), "2026-01-10"),
    ([intent()], {"a": quote(exit=None, day="2026-01-11")}, config(), "2026-01-10"),
    ([intent(), intent()], {}, config(), "2026-01-10"),
    ([intent()], {"unknown": quote()}, config(), "2026-01-10"),
    ([intent()], {"a": quote(entry=-1)}, config(), "2026-01-10"),
    ([intent()], {"a": quote()}, config(), "20260110"),
])
def test_original_experiment_admission_precedes_shared_accounting(monkeypatch, rows, quotes, cfg, end):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid experiment reached generic accounting")
    monkeypatch.setattr(book, "settle_premiums", forbidden)
    with pytest.raises(OpsError) as caught:
        book.funded_book(rows, quotes, config=cfg, report_end=end)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == "invalid distribution book inputs or arithmetic"


def test_core_numeric_refusal_keeps_the_exact_book_error_envelope(monkeypatch):
    def refused(*args, **kwargs):
        raise data_fail("CONTRACT_MISMATCH", "invalid premium settlement inputs or arithmetic")
    monkeypatch.setattr(book, "settle_premiums", refused)
    with pytest.raises(OpsError) as caught:
        book.funded_book([intent()], {"a": quote()}, config=config(), report_end="2026-01-10")
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False and caught.value.problem.details == {}
    assert caught.value.problem.message == "invalid distribution book inputs or arithmetic"
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize("failure", [MemoryError("synthetic allocation"), RuntimeError("synthetic programming"),
    data_fail("RESOURCE_UNAVAILABLE", "synthetic resource failure")])
def test_unexpected_shared_failures_are_not_reclassified(monkeypatch, failure):
    def failed(*args, **kwargs):
        raise failure
    monkeypatch.setattr(book, "settle_premiums", failed)
    with pytest.raises(type(failure)) as caught:
        book.funded_book([intent()], {"a": quote()}, config=config(), report_end="2026-01-10")
    assert caught.value is failure


def test_existing_progress_brackets_core_and_counts_noop_scheduled_actions(monkeypatch):
    events = []
    class Progress:
        def __init__(self, name, total):
            events.append(("progress", name, total))
        def tick(self):
            events.append("tick")
    @contextmanager
    def stage(name):
        events.append(("start", name))
        yield
        events.append(("done", name))
    monkeypatch.setattr(book, "Progress", Progress)
    monkeypatch.setattr(book, "stage", stage)
    book.funded_book([intent("a", selected=False), intent("b")], {"a": quote()},
                     config=config(), report_end="2026-01-10")
    assert events == [("progress", "distribution book", 3), ("start", "distribution book"),
                      "tick", "tick", "tick", ("done", "distribution book")]


def test_existing_progress_programming_error_propagates(monkeypatch):
    failure = RuntimeError("synthetic progress callback")
    def broken(self):
        raise failure
    monkeypatch.setattr(book.Progress, "tick", broken)
    with pytest.raises(RuntimeError) as caught:
        book.funded_book([intent()], {"a": quote()}, config=config(), report_end="2026-01-10")
    assert caught.value is failure
