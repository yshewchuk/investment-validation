"""engine.v2.ledger.status -- P6-3 decisions-status.

Proves the v2 catalog-backed status summary agrees with legacy's own pure
accounting (engine.ledger.scored_pairs/settlement_summary) over the SAME
synthetic decisions, and respects the catalog's append-only/supersession/
missing-finality rules.
"""
from __future__ import annotations

import json

import pytest

from engine import ledger
from engine.v2.ledger import status
from engine.v2.ledger.decisions import DecisionConflict, import_lines, set_authority
from engine.v2.ops.catalog import transaction

from tests.ops_support import catalog


def _pred(row_id, *, as_of="2026-09-01", ticker="AAPL", strategy="STR-THRU",
          event_date="2026-09-05", gate_pass=True, entry_cost=5.0, **overrides):
    row = {"row_id": row_id, "as_of": as_of, "ticker": ticker, "strategy": strategy,
           "event_date": event_date,
           "score": {"gate_pass": gate_pass, "win_model": 0.6, "exp_pnl_model": 0.05},
           "intended_prices": {"alpha": 0.5, "entry_cost": entry_cost}}
    row.update(overrides)
    return json.dumps(row).encode()


def _outcome(row_id, *, status_="resolved", resolved_at="2026-09-06T00:00:00+00:00",
             realized_pnl=0.10, realized_win=True, realized_entry_cost=5.1,
             realized_exit_value=5.6, **overrides):
    row = {"row_id": row_id, "status": status_, "resolved_at": resolved_at,
           "realized_pnl": realized_pnl, "realized_win": realized_win,
           "realized_entry_cost": realized_entry_cost, "realized_exit_value": realized_exit_value}
    row.update(overrides)
    return json.dumps(row).encode()


def _import(conn, pred_lines, outcome_lines=(), *, pred_hash="sha256:preds", out_hash="sha256:outs",
           pred_created="2026-09-01T00:00:00.000000Z", out_created="2026-09-06T00:00:00.000000Z"):
    with transaction(conn):
        import_lines(conn, pred_hash, pred_lines, kind="prediction", created_at=pred_created)
        if outcome_lines:
            import_lines(conn, out_hash, outcome_lines, kind="outcome", created_at=out_created)


@pytest.fixture
def conn(tmp_path):
    c, _clock, _supervisor = catalog(tmp_path)
    with transaction(c):
        set_authority(c, None, "catalog", "2026-09-01T00:00:00.000000Z")
    return c


class TestStatusAgreesWithLegacyAccounting:
    def test_settlement_diagnostics_match_legacy_scored_pairs(self, conn):
        preds = [_pred("p1"), _pred("p2", ticker="MSFT", gate_pass=False, entry_cost=7.0)]
        outs = [_outcome("p1")]
        _import(conn, preds, outs)

        result = status.status(conn)

        legacy_pairs = ledger.scored_pairs(predictions=[json.loads(p) for p in preds],
                                           outcomes=[json.loads(o) for o in outs])
        assert result["settlement_diagnostics"] == ledger.settlement_summary(legacy_pairs)
        assert result["predictions"] == 2
        assert result["outcomes"] == 1
        assert result["resolved"] == 1
        assert result["unresolvable"] == 0

    def test_an_unresolvable_outcome_is_counted_not_dropped(self, conn):
        preds = [_pred("p1")]
        outs = [_outcome("p1", status_="unresolvable", realized_pnl=None, realized_win=None,
                         realized_entry_cost=None, realized_exit_value=None,
                         reason="no priced replay")]
        _import(conn, preds, outs)
        result = status.status(conn)
        assert result["outcomes"] == 1
        assert result["resolved"] == 0
        assert result["unresolvable"] == 1

    def test_calibration_due_flips_true_at_the_trigger(self, conn):
        preds = [_pred(f"p{i}", event_date=f"2026-09-{5 + i:02d}") for i in range(3)]
        outs = [_outcome(f"p{i}") for i in range(3)]
        _import(conn, preds, outs)
        assert status.status(conn, trigger=3)["calibration_due"] is True
        assert status.status(conn, trigger=4)["calibration_due"] is False


class TestMissingFinalityLeavesRowsPending:
    def test_a_past_event_with_no_committed_outcome_is_pending_not_resolved(self, conn):
        """No outcome was ever committed for p1 -- e.g. decision_commit.
        import_settlement_candidates_in_transaction refused it for missing
        finality (tests/test_v2_ops_critical_review.py covers that gate).
        Status must show it pending, never silently resolved."""
        _import(conn, [_pred("p1", event_date="2020-01-01")])
        result = status.status(conn)
        assert result["resolved"] == 0
        assert result["pending_settlement"] == 1

    def test_a_future_event_is_not_pending(self, conn):
        _import(conn, [_pred("p1", event_date="2099-01-01")])
        assert status.status(conn)["pending_settlement"] == 0


class TestDuplicateRetryAndImmutability:
    def test_a_byte_identical_reimport_is_a_noop(self, conn):
        preds = [_pred("p1")]
        _import(conn, preds)
        first = status.status(conn)
        _import(conn, preds)  # same source_hash + bytes: import_lines no-ops
        assert status.status(conn) == first

    def test_a_conflicting_reimport_of_the_same_decision_is_refused(self, conn):
        _import(conn, [_pred("p1", entry_cost=5.0)])
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                import_lines(conn, "sha256:different-source", [_pred("p1", entry_cost=99.0)],
                            kind="prediction", created_at="2026-09-02T00:00:00.000000Z")
        # the original decision is untouched
        assert status.status(conn)["predictions"] == 1


class TestSameSessionSupersession:
    def test_a_supersede_row_replaces_the_original_in_the_resolved_view(self, conn):
        original = _pred("p1", entry_cost=5.0)
        restated = _pred("p2", entry_cost=6.0, supersedes="p1", supersede_reason="restated")
        _import(conn, [original, restated])
        result = status.status(conn)
        # both rows are on file (append-only)...
        assert result["prediction_rows_total"] == 2
        # ...but only the restatement counts in the resolved view
        assert result["predictions"] == 1