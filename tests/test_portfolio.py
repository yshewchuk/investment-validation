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

    def test_the_last_view_before_entry_is_the_one_that_counts(self, ledger_stub):
        """Not the first night it appeared — the last one before the position
        opens. The gate's verdict CHANGES across those nights as chains arrive
        and features fill in, and the verdict you would have acted on is the
        final one, not the first."""
        ledger_stub([_pred(as_of="2026-08-26", event_date="2026-09-30"),
                     _pred(as_of="2026-08-29", event_date="2026-09-30")])
        assert str(portfolio.build_book()["as_of"].iloc[0].date()) == "2026-08-29"

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

    def test_equal_dollars_is_the_default_not_equal_contracts(self, ledger_stub):
        """One contract each is not equal sizing. Premiums in a single week ran
        $2.40 to $69.00, so a one-contract book puts 29x more capital behind the
        expensive name and lets a few premiums decide the P&L."""
        old = (pd.Timestamp.today() - pd.Timedelta(days=60)).date()
        cheap = _pred(ticker="CHEAP", event_date=str(old), entry_cost=2.0)
        dear = _pred(ticker="DEAR", event_date=str(old), entry_cost=60.0)
        ledger_stub([cheap, dear])
        book = portfolio.build_book(capital_per_trade=10_000.0)
        # A premium is per SHARE and a contract is 100 shares: $2.00 is $200 a
        # lot (50 lots = $10,000 exactly), $60.00 is $6,000 a lot (1 lot).
        # Near-equal, not equal — whole lots, rounded down. The dearest names
        # are the ones that fall short, which is why the budget is $10,000 and
        # not $1,000.
        assert book["contracts"].tolist() == [50.0, 1.0]
        assert book["capital"].tolist() == [10_000.0, 6_000.0]

    def test_equal_dollars_gives_each_trade_the_same_pnl_for_the_same_return(
        self, ledger_stub
    ):
        old = (pd.Timestamp.today() - pd.Timedelta(days=60)).date()
        cheap = _pred(ticker="CHEAP", event_date=str(old), entry_cost=2.0)
        dear = _pred(ticker="DEAR", event_date=str(old), entry_cost=60.0)
        ledger_stub([cheap, dear], [
            self._settled(cheap["row_id"], 0.10, cost=2.0),
            self._settled(dear["row_id"], 0.10, cost=60.0),
        ])
        book = portfolio.build_book(capital_per_trade=10_000.0)
        # 10% on $10,000 and on $6,000 — the rounding gap, made visible.
        assert book["pnl"].round(2).tolist() == [1000.0, 600.0]

    def test_contracts_mode_is_still_available_and_is_not_equal(self, ledger_stub):
        old = (pd.Timestamp.today() - pd.Timedelta(days=60)).date()
        ledger_stub([_pred(ticker="CHEAP", event_date=str(old), entry_cost=2.0),
                     _pred(ticker="DEAR", event_date=str(old), entry_cost=60.0)])
        book = portfolio.build_book(contracts=1)
        assert book["capital"].tolist() == [200.0, 6000.0]

    def test_an_expensive_name_is_not_dropped_by_the_budget(self, ledger_stub):
        """Whole lots with a floor of one. Rounding down alone would size a
        premium above the budget at zero, and a book that silently drops its
        priciest recommendations is not measuring the board."""
        old = (pd.Timestamp.today() - pd.Timedelta(days=60)).date()
        ledger_stub([_pred(ticker="HUGE", event_date=str(old), entry_cost=150.0)])
        book = portfolio.build_book(capital_per_trade=10_000.0)
        assert len(book) == 1
        # $15,000 a lot against a $10,000 budget: one lot, over budget, kept.
        assert book["contracts"].iloc[0] == 1.0
        assert book["capital"].iloc[0] == pytest.approx(15_000.0)

    def test_contracts_are_always_whole(self, ledger_stub):
        """The book is meant to be executable. A fraction of a contract is not
        a thing you can buy."""
        old = (pd.Timestamp.today() - pd.Timedelta(days=60)).date()
        ledger_stub([_pred(ticker=f"T{i}", event_date=str(old), entry_cost=c)
                     for i, c in enumerate((1.075, 6.2, 31.225, 69.0))])
        book = portfolio.build_book(capital_per_trade=10_000.0)
        assert (book["contracts"] == book["contracts"].round()).all()
        assert (book["contracts"] >= 1).all()

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
        s = portfolio.summarize(portfolio.build_book(contracts=1))
        assert s["capital_committed"] == pytest.approx(1200.0)
        assert s["n_settled"] == 0
        # and under the default equal-dollar sizing, both carry the same capital
        eq = portfolio.summarize(portfolio.build_book(capital_per_trade=10_000.0))
        # $5.00 -> 20 lots = $10,000; $7.00 -> 14 lots = $9,800.
        assert eq["capital_committed"] == pytest.approx(19_800.0)


class TestEmptyBook:
    def test_no_predictions_renders_rather_than_raising(self, ledger_stub):
        ledger_stub([])
        book = portfolio.build_book()
        assert book.empty
        assert "No recommendations" in portfolio.render(book, portfolio.summarize(book))


class TestOutcomesMissingRealizedColumns:
    """No outcome has resolved into `realized_entry_cost`/`realized_exit_value`.

    Those two fields are written only by a RESOLVED outcome row — an
    unresolvable one has nothing to price yet. A freshly reset ledger, or
    simply no trade's exit session having settled, means every outcome on file
    can be unresolvable at once, and a frame built from those JSON records
    never gets a column no row supplied. This is not a hypothetical: it broke
    `dashboard/earnings/data/book.json` for the whole board (`available: false,
    KeyError: "['realized_entry_cost', 'realized_exit_value'] not in index"`)
    the day the ledger reset and nothing had settled yet.
    """

    @staticmethod
    def _unresolvable(row_id):
        return {"row_id": row_id, "status": "unresolvable",
                "reason": "no priced replay for this event at the intended alpha",
                "realized_pnl": None, "realized_win": None}

    def test_an_all_unresolvable_outcomes_table_does_not_raise(self, ledger_stub):
        ledger_stub([_pred(ticker="AAA")], outcomes=[self._unresolvable("AAA-row")])
        book = portfolio.build_book()
        assert list(book["ticker"]) == ["AAA"]
        assert pd.isna(book.loc[0, "realized_entry_cost"])
        assert pd.isna(book.loc[0, "realized_exit_value"])

    def test_a_mix_of_resolved_and_unresolvable_still_merges_correctly(self, ledger_stub):
        preds = [_pred(ticker="AAA", row_id="A-row"), _pred(ticker="BBB", row_id="B-row")]
        outcomes = [
            {"row_id": "A-row", "status": "resolved", "reason": None,
             "realized_pnl": 0.2, "realized_win": True,
             "realized_entry_cost": 6.0, "realized_exit_value": 7.2},
            self._unresolvable("B-row"),
        ]
        ledger_stub(preds, outcomes=outcomes)
        book = portfolio.build_book().set_index("ticker")
        assert book.loc["AAA", "realized_entry_cost"] == pytest.approx(6.0)
        assert pd.isna(book.loc["BBB", "realized_entry_cost"])


class TestContrarianBook:
    """`include_declined=True` adds the gate's rejections, tagged `recommended`.

    A withheld gate (`gate_pass is None`) is neither a recommendation nor a
    rejection and must appear in neither population.
    """

    def test_declined_rows_are_excluded_by_default(self, ledger_stub):
        ledger_stub([_pred(ticker="AAA", gate_pass=True),
                     _pred(ticker="BBB", gate_pass=False)])
        book = portfolio.build_book()
        assert list(book["ticker"]) == ["AAA"]

    def test_include_declined_adds_them_tagged(self, ledger_stub):
        ledger_stub([_pred(ticker="AAA", gate_pass=True),
                     _pred(ticker="BBB", gate_pass=False)])
        book = portfolio.build_book(include_declined=True).set_index("ticker")
        assert bool(book.loc["AAA", "recommended"]) is True
        assert bool(book.loc["BBB", "recommended"]) is False

    def test_withheld_rows_never_appear_even_with_declined_included(self, ledger_stub):
        ledger_stub([_pred(ticker="AAA", gate_pass=True),
                     _pred(ticker="BBB", gate_pass=False),
                     _pred(ticker="CCC", gate_pass=None)])
        book = portfolio.build_book(include_declined=True)
        assert set(book["ticker"]) == {"AAA", "BBB"}

    def test_by_recommended_splits_settled_pnl(self, ledger_stub):
        preds = [_pred(ticker="AAA", gate_pass=True, row_id="A-row"),
                 _pred(ticker="BBB", gate_pass=False, row_id="B-row")]
        outcomes = [
            {"row_id": "A-row", "status": "resolved", "reason": None,
             "realized_pnl": 0.2, "realized_win": True,
             "realized_entry_cost": 6.0, "realized_exit_value": 7.2},
            {"row_id": "B-row", "status": "resolved", "reason": None,
             "realized_pnl": -0.3, "realized_win": False,
             "realized_entry_cost": 6.0, "realized_exit_value": 4.2},
        ]
        ledger_stub(preds, outcomes=outcomes)
        book = portfolio.build_book(include_declined=True)
        summary = portfolio.summarize(book)
        by_rec = summary["by_recommended"]
        assert by_rec["recommended"]["pnl"] > 0
        assert by_rec["declined"]["pnl"] < 0

    def test_a_single_population_summary_has_no_split(self, ledger_stub):
        """Without include_declined there is only one population — nothing to
        split, and the key should not appear implying a comparison that was
        never measured."""
        ledger_stub([_pred(ticker="AAA", gate_pass=True)])
        book = portfolio.build_book()
        summary = portfolio.summarize(book)
        assert "by_recommended" not in summary
