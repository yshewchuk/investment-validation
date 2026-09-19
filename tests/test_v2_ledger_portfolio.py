"""engine.v2.ledger.portfolio -- P6-3 books-funding.

Proves the v2 catalog-backed build_book()/summarize() match legacy's own
pure accounting (engine.portfolio.build_book/summarize) over the SAME
synthetic decisions -- book totals, capital per trade and funding agree
with the existing accounting by construction.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine import portfolio as legacy_portfolio
from engine.v2.ledger import portfolio as v2_portfolio
from engine.v2.ledger.decisions import DecisionConflict, import_lines, set_authority
from engine.v2.ops.catalog import transaction

from tests.ops_support import catalog


def _pred(row_id, *, as_of="2026-08-25", ticker="AAA", strategy="STR-THRU",
          event_date="2026-09-30", gate_pass=True, entry_cost=6.0, **overrides):
    row = {"row_id": row_id, "as_of": as_of, "ticker": ticker, "strategy": strategy,
           "event_date": event_date,
           "score": {"gate_pass": gate_pass, "win_model": 0.45, "exp_pnl_model": 0.1},
           "intended_prices": {"alpha": 0.5, "entry_cost": entry_cost, "spot": 50.0}}
    row.update(overrides)
    return json.dumps(row).encode()


def _outcome(row_id, *, resolved_at="2026-10-01T00:00:00+00:00", realized_pnl=0.12,
            realized_entry_cost=6.1, realized_exit_value=6.83, **overrides):
    row = {"row_id": row_id, "status": "resolved", "resolved_at": resolved_at,
           "realized_pnl": realized_pnl, "realized_entry_cost": realized_entry_cost,
           "realized_exit_value": realized_exit_value}
    row.update(overrides)
    return json.dumps(row).encode()


@pytest.fixture
def conn(tmp_path):
    c, _clock, _supervisor = catalog(tmp_path)
    with transaction(c):
        set_authority(c, None, "catalog", "2026-08-25T00:00:00.000000Z")
    return c


def _import(conn, pred_lines, outcome_lines=()):
    with transaction(conn):
        import_lines(conn, "sha256:preds", pred_lines, kind="prediction",
                    created_at="2026-08-25T00:00:00.000000Z")
        if outcome_lines:
            import_lines(conn, "sha256:outs", outcome_lines, kind="outcome",
                        created_at="2026-10-01T00:00:00.000000Z")


class TestBookAgreesWithLegacyAccounting:
    def test_book_and_summary_totals_match_legacy_build_book(self, conn):
        preds = [_pred("p1", ticker="AAA", event_date="2026-09-30"),
                 _pred("p2", ticker="BBB", event_date="2026-09-30", gate_pass=False, entry_cost=8.0)]
        outs = [_outcome("p1")]
        _import(conn, preds, outs)

        book = v2_portfolio.build_book(conn)
        summary = v2_portfolio.summarize(book)

        legacy_book = legacy_portfolio.build_book(
            predictions=[json.loads(p) for p in preds],
            outcomes=[json.loads(o) for o in outs])
        legacy_summary = legacy_portfolio.summarize(legacy_book)

        assert summary == legacy_summary
        assert list(book["ticker"]) == list(legacy_book["ticker"])
        pd.testing.assert_series_equal(book["pnl"], legacy_book["pnl"], check_dtype=False)
        pd.testing.assert_series_equal(book["capital"], legacy_book["capital"], check_dtype=False)

    def test_include_declined_matches_legacy_contrarian_book(self, conn):
        preds = [_pred("p1", ticker="AAA", event_date="2026-09-30"),
                 _pred("p2", ticker="BBB", event_date="2026-09-30", gate_pass=False, entry_cost=8.0)]
        outs = [_outcome("p1")]
        _import(conn, preds, outs)

        book = v2_portfolio.build_book(conn, include_declined=True)
        summary = v2_portfolio.summarize(book)
        legacy_book = legacy_portfolio.build_book(
            predictions=[json.loads(p) for p in preds],
            outcomes=[json.loads(o) for o in outs], include_declined=True)
        legacy_summary = legacy_portfolio.summarize(legacy_book)

        assert summary == legacy_summary
        assert sorted(book["ticker"]) == sorted(legacy_book["ticker"])

    def test_contract_sizing_matches_legacy(self, conn):
        preds = [_pred("p1", ticker="AAA", event_date="2026-09-30", entry_cost=69.0)]
        outs = [_outcome("p1")]
        _import(conn, preds, outs)

        book = v2_portfolio.build_book(conn, contracts=1)
        legacy_book = legacy_portfolio.build_book(
            contracts=1, predictions=[json.loads(p) for p in preds],
            outcomes=[json.loads(o) for o in outs])
        assert book["contracts"].iloc[0] == legacy_book["contracts"].iloc[0] == 1.0


class TestSameSessionSupersessionAndImmutability:
    def test_a_restated_entry_cost_is_what_the_book_sizes_from(self, conn):
        """The original prediction's entry_cost stays on file untouched
        (append-only); a supersede row corrects it, and the book -- like
        legacy's -- prices from the corrected view."""
        original = _pred("p1", ticker="AAA", event_date="2026-09-30", entry_cost=5.0)
        restated = _pred("p2", ticker="AAA", event_date="2026-09-30", entry_cost=8.0,
                         supersedes="p1", supersede_reason="restated fill")
        _import(conn, [original, restated])

        book = v2_portfolio.build_book(conn)
        assert len(book) == 1
        assert book["entry_cost"].iloc[0] == pytest.approx(8.0)

    def test_original_decisions_stay_immutable_on_a_conflicting_retry(self, conn):
        _import(conn, [_pred("p1", ticker="AAA", event_date="2026-09-30", entry_cost=5.0)])
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                import_lines(conn, "sha256:different-source",
                            [_pred("p1", ticker="AAA", event_date="2026-09-30", entry_cost=99.0)],
                            kind="prediction", created_at="2026-08-26T00:00:00.000000Z")
        book = v2_portfolio.build_book(conn)
        assert book["entry_cost"].iloc[0] == pytest.approx(5.0)


class TestMissingFinalityLeavesAnOpenPosition:
    def test_no_committed_outcome_is_open_or_awaiting_exit_never_settled(self, conn):
        far_future = str((pd.Timestamp.today() + pd.Timedelta(days=30)).date())
        _import(conn, [_pred("p1", ticker="AAA", event_date=far_future)])
        book = v2_portfolio.build_book(conn)
        assert book["state"].iloc[0] != "settled"
        assert v2_portfolio.summarize(book)["n_settled"] == 0