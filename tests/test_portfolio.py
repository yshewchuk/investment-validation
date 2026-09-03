"""The hypothetical book — what you'd hold if you took every recommendation.

This module turns the frozen ledger into a P&L, so the ways it can quietly
lie are the things worth testing: counting a withheld gate as a buy, buying
the same trade once per night it was proposed, or dropping the trades it
could not price.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine import portfolio


def _pred(ticker="AAA", strategy="STR-THRU", as_of="2026-08-28",
          event_date="2026-08-31", gate_pass=True, entry_cost=6.0, row_id=None):
    return {
        "row_id": row_id or f"{as_of}|{ticker}|{strategy}|atm|na",
        "as_of": as_of, "ticker": ticker, "strategy": strategy,
        "event_date": event_date,
        "score": {"gate_pass": gate_pass, "exp_pnl_model": 0.1, "win_model": 0.45},
        "intended_prices": {"alpha": 0.5, "entry_cost": entry_cost, "spot": 50.0},
    }


@pytest.fixture
def ledger_stub(monkeypatch):
    def install(preds, outcomes=()):
        monkeypatch.setattr(portfolio.ledger, "read_predictions", lambda **k: list(preds))
        monkeypatch.setattr(portfolio.ledger, "read_outcomes", lambda: list(outcomes))
    return install


class TestWhatCountsAsARecommendation:
    def test_a_withheld_gate_is_not_a_buy(self, ledger_stub):
        """`gate_pass=None` is the gate declining to decide — out-of-domain
        name, missing features. Reading that silence as a buy would put the
        whole board in the book."""
        ledger_stub([_pred(gate_pass=None), _pred(ticker="BBB", gate_pass=False)])
        assert portfolio.build_book().empty

    def test_only_gate_pass_true_is_taken(self, ledger_stub):
        ledger_stub([_pred(ticker="AAA", gate_pass=True),
                     _pred(ticker="BBB", gate_pass=None)])
        book = portfolio.build_book()
        assert list(book["ticker"]) == ["AAA"]


class TestOnePurchasePerTrade:
    def test_a_trade_proposed_five_nights_is_bought_once(self, ledger_stub):
        nights = ["2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28", "2026-08-29"]
        ledger_stub([_pred(as_of=n, event_date="2026-09-30") for n in nights])
        book = portfolio.build_book()
        assert len(book) == 1

    def test_it_is_bought_on_the_first_night_it_was_recommended(self, ledger_stub):
        ledger_stub([_pred(as_of="2026-08-29", event_date="2026-09-30"),
                     _pred(as_of="2026-08-26", event_date="2026-09-30")])
        assert str(portfolio.build_book()["as_of"].iloc[0].date()) == "2026-08-26"

    def test_the_same_ticker_on_two_events_is_two_trades(self, ledger_stub):
        ledger_stub([_pred(event_date="2026-09-30"), _pred(event_date="2026-12-30")])
        assert len(portfolio.build_book()) == 2


class TestStates:
    def _settled(self, row_id, ret, cost=6.0):
        return {"row_id": row_id, "status": "resolved", "realized_pnl": ret,
                "realized_entry_cost": cost, "realized_exit_value": cost * (1 + ret),
                "reason": None}

    def test_a_future_event_is_open_not_zero(self, ledger_stub):
        far = (pd.Timestamp.today() + pd.Timedelta(days=30)).date()
        ledger_stub([_pred(event_date=str(far))])
        book = portfolio.build_book()
        assert book["state"].iloc[0] == "open"
        assert portfolio.summarize(book)["n_settled"] == 0

    def test_a_fresh_print_is_awaiting_its_exit_chain(self, ledger_stub):
        """ORATS posts a session's chains around midnight, so a trade that
        printed today cannot settle yet. Calling it unresolvable would write
        off every fresh trade in the book."""
        today = pd.Timestamp.today().normalize()
        ledger_stub([_pred(event_date=str((today - pd.Timedelta(days=1)).date()))])
        assert portfolio.build_book()["state"].iloc[0] == "awaiting_exit"

    def test_an_old_unpriceable_trade_is_unresolvable_and_kept(self, ledger_stub):
        old = (pd.Timestamp.today() - pd.Timedelta(days=60)).date()
        ledger_stub([_pred(event_date=str(old))])
        book = portfolio.build_book()
        assert book["state"].iloc[0] == "unresolvable"
        assert len(book) == 1, "a trade the book could not follow must not vanish"

    def test_settled_pnl_is_return_times_cost_times_multiplier(self, ledger_stub):
        old = (pd.Timestamp.today() - pd.Timedelta(days=60)).date()
        p = _pred(event_date=str(old))
        ledger_stub([p], [self._settled(p["row_id"], 0.25, cost=6.0)])
        book = portfolio.build_book(contracts=2)
        # 25% on a $6.00 premium, two contracts of 100 shares = $300.
        assert book["pnl"].iloc[0] == pytest.approx(0.25 * 6.0 * 100 * 2)
        assert portfolio.summarize(book)["pnl"] == pytest.approx(300.0)

    def test_capital_counts_every_committed_trade_not_just_settled(self, ledger_stub):
        far = (pd.Timestamp.today() + pd.Timedelta(days=30)).date()
        ledger_stub([_pred(ticker="AAA", event_date=str(far), entry_cost=5.0),
                     _pred(ticker="BBB", event_date=str(far), entry_cost=7.0)])
        s = portfolio.summarize(portfolio.build_book())
        assert s["capital_committed"] == pytest.approx(1200.0)
        assert s["n_settled"] == 0


class TestEmptyBook:
    def test_no_predictions_renders_rather_than_raising(self, ledger_stub):
        ledger_stub([])
        book = portfolio.build_book()
        assert book.empty
        assert "No recommendations" in portfolio.render(book, portfolio.summarize(book))
