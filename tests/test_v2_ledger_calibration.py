"""engine.v2.ledger.calibration -- P6-3 decisions-calibration.

Proves the v2 catalog-backed calibrate() matches legacy's own pure
calibration math (engine.ledger.scored_pairs/_strategy_calibration/
settlement_summary) over the SAME synthetic decisions, publishes a real,
durably readable artifact, and honours the >=trigger recompute rule.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

from engine import ledger
from engine.v2.foundation import ArtifactStore
from engine.v2.ledger import calibration
from engine.v2.ledger.decisions import import_lines, set_authority
from engine.v2.ops.catalog import transaction

from tests.ops_support import catalog


class FakeClock:
    def now(self):
        return datetime(2026, 9, 19, tzinfo=timezone.utc)


def _pred(row_id, *, as_of="2026-09-01", ticker="AAPL", strategy="STR-THRU",
          event_date="2026-09-05", win=0.6, pnl=0.05, entry_cost=5.0, **overrides):
    row = {"row_id": row_id, "as_of": as_of, "ticker": ticker, "strategy": strategy,
           "event_date": event_date,
           "score": {"gate_pass": True, "win_model": win, "exp_pnl_model": pnl},
           "intended_prices": {"alpha": 0.5, "entry_cost": entry_cost}}
    row.update(overrides)
    return json.dumps(row).encode()


def _outcome(row_id, *, resolved_at="2026-09-06T00:00:00+00:00", realized_pnl=0.10,
             realized_win=True, realized_entry_cost=5.1, realized_exit_value=5.6, **overrides):
    row = {"row_id": row_id, "status": "resolved", "resolved_at": resolved_at,
           "realized_pnl": realized_pnl, "realized_win": realized_win,
           "realized_entry_cost": realized_entry_cost, "realized_exit_value": realized_exit_value}
    row.update(overrides)
    return json.dumps(row).encode()


def _nan_equal(left, right):
    """Structural equality with NaN equal to NaN.

    ``reliability_monotonicity`` is NaN whenever the reliability table has too
    few buckets (``engine.calibrate.monotonicity``), and BOTH sides of these
    parity assertions are the same legacy pure function's answer. Plain ``==``
    cannot express that (NaN != NaN), so compare recursively.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nan_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nan_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return left == right


@pytest.fixture
def conn(tmp_path):
    c, _clock, _supervisor = catalog(tmp_path)
    with transaction(c):
        set_authority(c, None, "catalog", "2026-09-01T00:00:00.000000Z")
    return c


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


def _import(conn, pred_lines, outcome_lines=()):
    with transaction(conn):
        import_lines(conn, "sha256:preds", pred_lines, kind="prediction",
                    created_at="2026-09-01T00:00:00.000000Z")
        if outcome_lines:
            import_lines(conn, "sha256:outs", outcome_lines, kind="outcome",
                        created_at="2026-09-06T00:00:00.000000Z")


def _three_trades():
    preds = [_pred(f"p{i}", event_date=f"2026-09-{5 + i:02d}", win=0.5 + i * 0.1)
            for i in range(3)]
    outs = [_outcome(f"p{i}", realized_win=(i != 1)) for i in range(3)]
    return preds, outs


class TestCalibrationMatchesLegacyMath:
    def test_overall_and_per_strategy_agree_with_the_legacy_pure_functions(self, conn, store):
        preds, outs = _three_trades()
        _import(conn, preds, outs)

        result = calibration.calibrate(conn, store, clock=FakeClock(), force=True)

        legacy_pairs = ledger.scored_pairs(predictions=[json.loads(p) for p in preds],
                                           outcomes=[json.loads(o) for o in outs])
        expected_overall = ledger._strategy_calibration(legacy_pairs)
        expected_per_strategy = {str(s): ledger._strategy_calibration(g)
                                 for s, g in legacy_pairs.groupby("strategy")}
        assert result["regenerated"] is True
        assert result["n_scored"] == 3
        assert _nan_equal(result["calibration"], expected_overall)
        assert _nan_equal(result["per_strategy"], expected_per_strategy)

    def test_the_published_health_artifact_is_real_and_readable(self, conn, store):
        preds, outs = _three_trades()
        _import(conn, preds, outs)
        result = calibration.calibrate(conn, store, clock=FakeClock(), force=True)

        health_bytes = store.read_verified(result["health"])
        health = json.loads(health_bytes)
        assert health["n_scored"] == 3
        assert _nan_equal(health["per_strategy"], result["per_strategy"])
        assert health["settlement_diagnostics"] == ledger.settlement_summary(
            ledger.scored_pairs(predictions=[json.loads(p) for p in preds],
                                outcomes=[json.loads(o) for o in outs]))

    def test_status_surfaces_the_same_artifact_calibrate_published(self, conn, store):
        from engine.v2.ledger import status

        preds, outs = _three_trades()
        _import(conn, preds, outs)
        result = calibration.calibrate(conn, store, clock=FakeClock(), force=True)
        assert status.status(conn)["health"] == result["health"]


class TestTheTriggerRule:
    def test_below_trigger_is_a_noop_unless_forced(self, conn, store):
        preds, outs = _three_trades()
        _import(conn, preds, outs)

        not_due = calibration.calibrate(conn, store, clock=FakeClock(), trigger=50)
        assert not_due == {"regenerated": False, "n_scored": 3, "n_at_last_report": 0,
                           "note": "3 new scored row(s); trigger is 50"}

        forced = calibration.calibrate(conn, store, clock=FakeClock(), trigger=50, force=True)
        assert forced["regenerated"] is True

    def test_a_second_call_after_a_forced_report_is_not_due_again(self, conn, store):
        preds, outs = _three_trades()
        _import(conn, preds, outs)
        calibration.calibrate(conn, store, clock=FakeClock(), force=True)

        again = calibration.calibrate(conn, store, clock=FakeClock())
        assert again["regenerated"] is False
        assert again["n_at_last_report"] == 3

    def test_new_scored_rows_reaching_the_trigger_recompute_without_force(self, conn, store):
        preds, outs = _three_trades()
        _import(conn, preds, outs)
        calibration.calibrate(conn, store, clock=FakeClock(), trigger=3, force=True)

        more_preds = [_pred("p3", event_date="2026-09-08")]
        more_outs = [_outcome("p3")]
        with transaction(conn):
            import_lines(conn, "sha256:more-preds", more_preds, kind="prediction",
                        created_at="2026-09-08T00:00:00.000000Z")
            import_lines(conn, "sha256:more-outs", more_outs, kind="outcome",
                        created_at="2026-09-09T00:00:00.000000Z")

        due = calibration.calibrate(conn, store, clock=FakeClock(), trigger=1)
        assert due["regenerated"] is True
        assert due["n_scored"] == 4


class TestExportHealthFile:
    def test_export_health_file_writes_published_artifact_bytes(self, conn, store, tmp_path):
        preds, outs = _three_trades()
        _import(conn, preds, outs)
        calibration.calibrate(conn, store, clock=FakeClock(), force=True)

        dest = tmp_path / "export" / "calibration_health.json"
        dest.parent.mkdir()
        calibration.export_health_file(conn, store, dest)

        ref = calibration.health_ref(conn)
        assert ref is not None
        assert dest.read_bytes() == store.read_verified(ref)

    def test_export_health_file_refuses_before_first_calibrate(self, conn, store, tmp_path):
        dest = tmp_path / "export" / "calibration_health.json"
        dest.parent.mkdir()

        with pytest.raises(calibration.CalibrationNotYetRun):
            calibration.export_health_file(conn, store, dest)

        assert not dest.exists()  # never an empty or placeholder file
        assert not dest.with_suffix(".tmp").exists()
