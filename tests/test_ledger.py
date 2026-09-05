"""engine.ledger — the append-only prediction ledger and its outcome scorer.

The ledger records predictions before outcomes and shares replay pricing, so
the tests pin the properties that make it evidence rather than notes: nothing
can be rewritten, a correction leaves the original readable, the file date
follows the decision date rather than the clock, and an outcome that cannot be
resolved is RECORDED as unresolvable instead of quietly dropped.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import ledger
from engine.ledger_settlement import POLICY
from engine.structures import STRUCTURES


@pytest.fixture()
def ledger_root(tmp_path, monkeypatch):
    """Point every ledger path at a temp tree."""
    from engine import paths

    monkeypatch.setattr(paths, "LEDGER", tmp_path / "ledger")
    monkeypatch.setattr(ledger, "_settlement_calendar", lambda: pd.DataFrame(
        [{k: r[k] for k in ("event_id", "ticker", "event_date", "session")}
         for r in ledger.read_predictions()],
        columns=["event_id", "ticker", "event_date", "session"]))
    return tmp_path / "ledger"


def _prediction(as_of="2026-10-15", ticker="AAPL", strategy="STR-THRU",
                win=0.55, pnl=0.04, **overrides) -> dict:
    rid = ledger.row_id(as_of, ticker, strategy, None, "2026-10-23")
    row = {
        "schema_version": ledger.SCHEMA_VERSION,
        "row_id": rid,
        "written_at": "2026-10-15T21:05:03+00:00",
        "as_of": as_of,
        "decision_ts": f"{as_of}T20:00:00+00:00",
        "ticker": ticker,
        "event_id": f"{ticker}-2026-10-16",
        "event_date": "2026-10-16",
        "session": "AMC",
        "strategy": strategy,
        "settlement": {"policy": POLICY, "spec_version": 1,
                       "structure_spec": asdict(STRUCTURES[strategy]())},
        "structure": {"strike": None, "expiry": "2026-10-23"},
        "intended_prices": {"alpha": 0.5, "entry_cost": 8.42},
        "score": {"win_model": win, "exp_pnl_model": pnl, "gate_pass": True},
        "model_versions": {"gate": "gate_midfill_str_thru@1"},
        "snapshot_hash": "dce985",
        "audit_receipt": None,
        "supersedes": None,
        "supersede_reason": None,
    }
    row.update(overrides)
    return row


class TestAppendOnly:
    def test_write_then_read(self, ledger_root):
        path = ledger.write_predictions([_prediction()])
        assert path.name == "2026-10-15.jsonl"
        rows = ledger.read_predictions()
        assert len(rows) == 1 and rows[0]["ticker"] == "AAPL"

    def test_rewriting_a_row_id_is_refused(self, ledger_root):
        ledger.write_predictions([_prediction()])
        with pytest.raises(ledger.LedgerError, match="append-only"):
            ledger.write_predictions([_prediction(win=0.99)])

    def test_second_batch_appends(self, ledger_root):
        ledger.write_predictions([_prediction(ticker="AAPL")])
        ledger.write_predictions([_prediction(ticker="MSFT")])
        assert len(ledger.read_predictions()) == 2

    def test_batch_may_not_span_two_as_of_dates(self, ledger_root):
        rows = [_prediction(as_of="2026-10-15"), _prediction(as_of="2026-10-16")]
        with pytest.raises(ledger.LedgerError, match="file date follows as_of"):
            ledger.write_predictions(rows)

    def test_file_date_follows_as_of_not_the_clock(self, ledger_root):
        # The nightly job that finishes after midnight must not split a board.
        ledger.write_predictions([_prediction(as_of="2026-10-15")],
                                 as_of="2026-10-15")
        assert (ledger.predictions_dir() / "2026-10-15.jsonl").exists()

    def test_missing_fields_refused(self, ledger_root):
        row = _prediction()
        row.pop("snapshot_hash")
        with pytest.raises(ledger.LedgerError, match="missing fields"):
            ledger.write_predictions([row])

    def test_no_delete_path_exists(self):
        assert not any(name.startswith("delete") or name.startswith("remove")
                       for name in dir(ledger))


class TestSupersede:
    def test_supersede_hides_the_old_row_but_keeps_it_on_file(self, ledger_root):
        original = _prediction()
        ledger.write_predictions([original])
        replacement = _prediction(as_of="2026-10-16", win=0.61)
        ledger.supersede(original["row_id"], replacement, reason="chain refreshed")

        visible = ledger.read_predictions()
        assert len(visible) == 1
        assert visible[0]["score"]["win_model"] == 0.61

        raw = ledger.read_predictions(resolve_supersedes=False)
        assert len(raw) == 2, "the original must remain on file"
        assert original["row_id"] in {r["row_id"] for r in raw}

    def test_supersede_needs_a_reason_and_a_known_target(self, ledger_root):
        ledger.write_predictions([_prediction()])
        with pytest.raises(ledger.LedgerError, match="reason"):
            ledger.supersede(_prediction()["row_id"], _prediction(as_of="2026-10-16"), "")
        with pytest.raises(ledger.LedgerError, match="unknown row_id"):
            ledger.supersede("nope", _prediction(as_of="2026-10-16"), "r")


class TestRowId:
    def test_stable_and_decision_dated(self):
        a = ledger.row_id("2026-10-15", "AAPL", "STR-THRU", None, "2026-10-23")
        b = ledger.row_id("2026-10-15", "AAPL", "STR-THRU", None, "2026-10-23")
        c = ledger.row_id("2026-10-16", "AAPL", "STR-THRU", None, "2026-10-23")
        assert a == b and a != c, "re-scoring on a later day is a NEW prediction"

    def test_strike_is_part_of_the_identity(self):
        assert (ledger.row_id("2026-10-15", "AAPL", "STR-THRU", 200.0, None)
                != ledger.row_id("2026-10-15", "AAPL", "STR-THRU", 210.0, None))


def _fake_replay(trades: pd.DataFrame):
    class _Result:
        def __init__(self, frame):
            self.trades = frame

    def _replay(strategy, events, **kwargs):
        return _Result(trades[trades["strategy"] == strategy]
                       if "strategy" in trades.columns else trades)

    return _replay


class TestOnlyTodaysEntriesAreRecorded:
    """The board looks days ahead; the ledger records a position.

    Recording every forward row every night wrote 3,716 predictions for about
    600 events — the same trade once per night, each on a quote a session
    staler than the last, none of which anyone would have acted on.
    """

    def _board(self):
        return pd.DataFrame([
            {"ticker": "TODAY", "strategy": "STR-THRU", "event_date": "2026-09-02",
             "entry_date": "2026-09-02", "strike": None, "expiry": "2026-09-04"},
            {"ticker": "BMOTODAY", "strategy": "STR-THRU", "event_date": "2026-09-03",
             "entry_date": "2026-09-02", "strike": None, "expiry": "2026-09-04"},
            {"ticker": "NEXTWEEK", "strategy": "STR-THRU", "event_date": "2026-09-09",
             "entry_date": "2026-09-09", "strike": None, "expiry": "2026-09-11"},
        ])

    def test_only_rows_entering_today_are_recorded(self, ledger_root):
        rows = ledger.build_prediction_rows(self._board(), as_of="2026-09-02")
        assert sorted(r["ticker"] for r in rows) == ["BMOTODAY", "TODAY"]

    def test_a_bmo_print_tomorrow_entering_today_is_recorded(self, ledger_root):
        """The entry date decides, not the event date: a BMO print on the 3rd
        is opened at the 2nd's close, so tonight is its night."""
        rows = ledger.build_prediction_rows(self._board(), as_of="2026-09-02")
        assert "BMOTODAY" in {r["ticker"] for r in rows}

    def test_the_same_trade_is_not_recorded_again_the_next_night(self, ledger_root):
        board = self._board()
        first = ledger.build_prediction_rows(board, as_of="2026-09-02")
        later = ledger.build_prediction_rows(board, as_of="2026-09-03")
        assert {r["ticker"] for r in first} & {r["ticker"] for r in later} == set()

    def test_a_row_with_no_entry_date_is_skipped_not_guessed(self, ledger_root):
        board = pd.DataFrame([{"ticker": "NOENTRY", "strategy": "STR-THRU",
                               "event_date": "2026-09-02", "entry_date": None,
                               "strike": None, "expiry": "2026-09-04"}])
        assert ledger.build_prediction_rows(board, as_of="2026-09-02") == []

    def test_the_old_behaviour_is_still_reachable_for_backfill(self, ledger_root):
        rows = ledger.build_prediction_rows(self._board(), as_of="2026-09-02",
                                            entry_dated_only=False)
        assert len(rows) == 3


class TestOutcomeScoring:
    def _priced(self, event_id="AAPL-2026-10-16", ret=0.12):
        return pd.DataFrame([{
            "event_id": event_id, "ticker": "AAPL", "strategy": "STR-THRU",
            "event_date": pd.Timestamp("2026-10-16"), "fill_alpha": 0.5,
            "entry_cost": 8.42, "exit_value": 9.43, "ret": ret,
            "exit_mode": "chain",
        }])

    def test_resolves_and_is_idempotent(self, ledger_root, monkeypatch):
        from engine import replay as replay_mod

        ledger.write_predictions([_prediction()])
        monkeypatch.setattr(replay_mod, "replay", _fake_replay(self._priced()))

        first = ledger.score_outcomes(through="2026-10-20")
        assert first["resolved"] == 1
        second = ledger.score_outcomes(through="2026-10-20")
        assert second["resolved"] == 0, "re-running must not duplicate outcomes"
        assert len(ledger.read_outcomes()) == 1

        outcome = ledger.read_outcomes()[0]
        assert outcome["realized_pnl"] == pytest.approx(0.12)
        assert outcome["realized_win"] is True
        assert outcome["predicted_win"] == 0.55

    def test_each_strategy_gets_its_own_realized_return(self, ledger_root, monkeypatch):
        """Three structures on the same event are three different trades.

        The lookup was keyed on (event_id, alpha) with no strategy, inside a
        loop over strategies — so each replay overwrote the last, and whichever
        `groupby` visited last (STR-THRU, alphabetically) was handed to every
        other strategy as its own realized return. STR-RUNUP reported outcomes
        for events its replay had just said it could not price.
        """
        from engine import replay as replay_mod

        for strategy in ("CAL-P", "STR-RUNUP", "STR-THRU"):
            ledger.write_predictions([_prediction(strategy=strategy)])

        rets = {"CAL-P": -0.30, "STR-RUNUP": 0.05, "STR-THRU": 0.12}

        def fake_replay(strategy, events, **kwargs):
            frame = self._priced(ret=rets[strategy])
            frame["strategy"] = strategy
            return _fake_replay(frame)(strategy, events, **kwargs)

        monkeypatch.setattr(replay_mod, "replay", fake_replay)
        ledger.score_outcomes(through="2026-10-20")
        got = {o["strategy"]: o["realized_pnl"] for o in ledger.read_outcomes()}
        assert got == pytest.approx(rets), got
        assert len(set(got.values())) == 3, "three structures must not share one number"

    def test_a_strategy_the_replay_cannot_price_stays_unresolved(
        self, ledger_root, monkeypatch
    ):
        """The failure that hid the bug: STR-RUNUP priced 0 events and still
        reported resolved outcomes, borrowed from STR-THRU."""
        from engine import replay as replay_mod

        for strategy in ("STR-RUNUP", "STR-THRU"):
            ledger.write_predictions([_prediction(strategy=strategy)])

        def fake_replay(strategy, events, **kwargs):
            frame = self._priced() if strategy == "STR-THRU" else pd.DataFrame(
                columns=["event_id", "strategy", "fill_alpha", "ret",
                         "event_date", "entry_cost", "exit_value"]
            )
            if len(frame):
                frame["strategy"] = strategy
            return _fake_replay(frame)(strategy, events, **kwargs)

        monkeypatch.setattr(replay_mod, "replay", fake_replay)
        ledger.score_outcomes(through="2026-10-20")
        by = {o["strategy"]: o["status"] for o in ledger.read_outcomes()}
        assert by["STR-THRU"] == "resolved"
        assert by["STR-RUNUP"] == "unresolvable"

    def test_an_unresolvable_row_is_retried_when_the_chain_arrives(
        self, ledger_root, monkeypatch
    ):
        """`unresolvable` is transient, not a verdict.

        ORATS publishes a session around midnight, so a print cannot settle on
        the night it happens. Treating the first failed attempt as final wrote
        off 861 real predictions whose chains arrived hours later.
        """
        from engine import replay as replay_mod

        ledger.write_predictions([_prediction()])
        empty = pd.DataFrame(columns=["event_id", "strategy", "fill_alpha", "ret",
                                      "event_date", "entry_cost", "exit_value"])
        monkeypatch.setattr(replay_mod, "replay", _fake_replay(empty))
        assert ledger.score_outcomes(through="2026-10-20")["unresolvable"] == 1

        # the chain lands, and the same row settles on a later pass
        monkeypatch.setattr(replay_mod, "replay", _fake_replay(self._priced(ret=0.4)))
        again = ledger.score_outcomes(through="2026-10-20")
        assert again["resolved"] == 1, "the retry must pick it up"

        pairs = ledger.scored_pairs()
        assert len(pairs) == 1, "one prediction, one verdict — the last one"
        assert pairs["realized_pnl"].iloc[0] == pytest.approx(0.4)

    def test_a_resolved_row_is_never_re_scored(self, ledger_root, monkeypatch):
        """Resolved IS terminal. Re-pricing a settled trade later would let the
        record drift with the data, which is the opposite of a frozen ledger."""
        from engine import replay as replay_mod

        ledger.write_predictions([_prediction()])
        monkeypatch.setattr(replay_mod, "replay", _fake_replay(self._priced(ret=0.12)))
        assert ledger.score_outcomes(through="2026-10-20")["resolved"] == 1

        monkeypatch.setattr(replay_mod, "replay", _fake_replay(self._priced(ret=0.99)))
        assert ledger.score_outcomes(through="2026-10-20")["resolved"] == 0
        assert ledger.scored_pairs()["realized_pnl"].iloc[0] == pytest.approx(0.12)

    def test_unresolvable_is_recorded_not_dropped(self, ledger_root, monkeypatch):
        from engine import replay as replay_mod

        ledger.write_predictions([_prediction()])
        empty = pd.DataFrame(columns=["event_id", "strategy", "fill_alpha", "ret",
                                      "event_date", "entry_cost", "exit_value"])
        monkeypatch.setattr(replay_mod, "replay", _fake_replay(empty))

        out = ledger.score_outcomes(through="2026-10-20")
        assert out["resolved"] == 0 and out["unresolvable"] == 1
        row = ledger.read_outcomes()[0]
        assert row["status"] == "unresolvable" and row["reason"]

    def test_future_events_are_not_scored(self, ledger_root, monkeypatch):
        from engine import replay as replay_mod

        ledger.write_predictions([_prediction()])
        monkeypatch.setattr(replay_mod, "replay", _fake_replay(self._priced()))
        out = ledger.score_outcomes(through="2026-10-15")
        assert out["resolved"] == 0 and out["unresolvable"] == 0

    def test_event_date_change_is_flagged(self, ledger_root, monkeypatch):
        from engine import replay as replay_mod

        ledger.write_predictions([_prediction()])
        moved = ledger._settlement_calendar()
        moved["event_date"] = "2026-10-19"
        monkeypatch.setattr(ledger, "_settlement_calendar", lambda: moved)
        def forbidden(*args, **kwargs):
            pytest.fail("a changed calendar must not replay a different schedule")
        monkeypatch.setattr(replay_mod, "replay", forbidden)
        ledger.score_outcomes(through="2026-10-20")
        assert ledger.read_outcomes()[0]["event_date_changed"] is True
        assert ledger.read_outcomes()[0]["status"] == "unresolvable"


class TestRecordedSettlement:
    """Exercise actual planning and quote pricing, with only data access replaced."""

    def _row(self, *, strike=95.0, expiry="2024-05-03", spec=None, alpha=0.5):
        spec = spec or asdict(STRUCTURES["STR-THRU"]())
        row = _prediction(
            as_of="2024-05-02", ticker="TEST", event_id="TEST_2024-05-02",
            event_date="2024-05-02",
            structure={"strike": strike, "expiry": expiry},
            intended_prices={"alpha": alpha, "entry_cost": 2.0, "quote_date": "2024-05-01"},
            settlement={"policy": POLICY, "spec_version": 1, "structure_spec": spec},
        )
        row["row_id"] += "|" + ledger.selection_key(row["settlement"], alpha)
        return row

    def _real_replay(self, monkeypatch):
        from engine import replay
        from engine.calendar import TradingCalendar
        from tests.test_replay import chain
        entry = chain("TEST", "2024-05-02")
        exit_ = chain("TEST", "2024-05-03")
        exit_.loc[exit_["strike"] == 105, ["bid", "ask"]] *= 2
        index = replay.ChainIndex({
            ("TEST", pd.Timestamp("2024-05-02")): entry,
            ("TEST", pd.Timestamp("2024-05-03")): exit_,
        })
        calendar = TradingCalendar(pd.bdate_range("2024-03-01", "2024-06-01"))
        original = replay.replay
        def with_data(*args, **kwargs):
            return original(*args, **kwargs, index=index, calendar=calendar)
        monkeypatch.setattr(replay, "replay", with_data)

    def test_stale_atm_estimate_reports_contract_and_cost_drift(self, ledger_root, monkeypatch):
        self._real_replay(monkeypatch)
        ledger.write_predictions([self._row(expiry="2024-05-24")])
        ledger.score_outcomes(through="2024-05-03")
        out = ledger.read_outcomes()[0]
        assert out["status"] == "resolved"
        assert out["contract_matched"] is False
        assert out["expiry_matched"] is False
        assert out["realized_structure"]["strike"] == 100
        assert len(out["realized_structure"]["legs"]) == 2
        assert out["intended_entry_cost"] == 2
        assert out["realized_entry_cost"] == pytest.approx(3.4)
        assert out["entry_cost_drift"] == pytest.approx(1.4)
        assert out["entry_cost_drift_fraction"] == pytest.approx(0.7)
        assert out["realized_pnl"] == pytest.approx(0)
        assert out["settlement_source"] == "orats_quote_simulation"

    def test_explicit_strikes_and_expiries_do_not_share_an_outcome(self, ledger_root, monkeypatch):
        from engine.score import Scorer, ScoreRequest
        self._real_replay(monkeypatch)
        for strike, expiry in ((100, "2024-05-03"), (105, "2024-05-03"),
                               (100, "2024-05-24")):
            spec = asdict(Scorer._structure(None, ScoreRequest(
                "TEST", "STR-THRU", strike=strike, expiry=pd.Timestamp(expiry))))
            ledger.write_predictions([self._row(strike=strike, expiry=expiry, spec=spec)])
        ledger.score_outcomes(through="2024-05-03")
        outcomes = ledger.read_outcomes()
        by_contract = {(o["realized_structure"]["strike"],
                        o["realized_structure"]["expiry"]): o for o in outcomes}
        assert by_contract[100, "2024-05-03"]["realized_pnl"] == pytest.approx(0)
        assert by_contract[105, "2024-05-03"]["realized_pnl"] == pytest.approx(1)
        assert by_contract[100, "2024-05-24"]["realized_entry_cost"] == pytest.approx(6.8)
        assert all(o["contract_matched"] for o in outcomes)
        assert len(ledger.scored_pairs()) == 3

    @pytest.mark.parametrize("alpha, cost", [(0, 3.8), (0.37, 3.504), (1, 3.0)])
    def test_exact_alpha_including_zero_is_used(self, ledger_root, monkeypatch, alpha, cost):
        self._real_replay(monkeypatch)
        ledger.write_predictions([self._row(alpha=alpha)])
        ledger.score_outcomes(through="2024-05-03")
        out = ledger.read_outcomes()[0]
        assert out["fill_alpha_used"] == alpha
        assert out["realized_entry_cost"] == pytest.approx(cost)

    def test_later_factory_defaults_cannot_change_recorded_rule(self, ledger_root, monkeypatch):
        self._real_replay(monkeypatch)
        ledger.write_predictions([self._row()])
        def forbidden():
            pytest.fail("settlement must not consult the current factory")
        monkeypatch.setitem(STRUCTURES, "STR-THRU", forbidden)
        assert ledger.score_outcomes(through="2024-05-03")["resolved"] == 1

    @pytest.mark.parametrize("corruption", ["missing", "partial", "version"])
    def test_missing_spec_is_not_guessed_from_defaults(self, ledger_root, monkeypatch, corruption):
        from engine import replay
        row = self._row()
        if corruption == "missing":
            row.pop("settlement")
            row["schema_version"] = 1
        elif corruption == "partial":
            row["settlement"]["structure_spec"]["legs"][0].pop("qty")
        else:
            row["settlement"]["spec_version"] = 999
        ledger.write_predictions([row])
        def forbidden(*args, **kwargs):
            pytest.fail("a missing specification must not be guessed")
        monkeypatch.setattr(replay, "replay", forbidden)
        assert ledger.score_outcomes(through="2024-05-03")["unresolvable"] == 1
        assert "unreproducible" in ledger.read_outcomes()[0]["reason"]

    def test_date_derived_event_id_disappearing_is_not_a_false_match(self, ledger_root, monkeypatch):
        ledger.write_predictions([self._row()])
        moved = ledger._settlement_calendar()
        moved["event_date"] = "2024-05-06"
        moved["event_id"] = "TEST_2024-05-06"
        monkeypatch.setattr(ledger, "_settlement_calendar", lambda: moved)
        assert ledger.score_outcomes(through="2024-05-07")["unresolvable"] == 1
        out = ledger.read_outcomes()[0]
        assert out["calendar_status"] == "missing"
        assert out["event_date_changed"] is None

    def test_exit_after_cutoff_is_not_settled(self, ledger_root, monkeypatch):
        self._real_replay(monkeypatch)
        ledger.write_predictions([self._row()])
        assert ledger.score_outcomes(through="2024-05-02")["unresolvable"] == 1
        assert "cutoff" in ledger.read_outcomes()[0]["reason"]

    def test_recorded_schedule_cannot_silently_move(self, ledger_root, monkeypatch):
        self._real_replay(monkeypatch)
        row = self._row()
        row["structure"]["entry_date"] = "2024-05-01"
        ledger.write_predictions([row])
        assert ledger.score_outcomes(through="2024-05-03")["unresolvable"] == 1
        assert "schedule differs" in ledger.read_outcomes()[0]["reason"]

    def test_ambiguous_replay_is_not_last_row_wins(self, ledger_root, monkeypatch):
        from engine import replay
        ledger.write_predictions([self._row()])
        trades = pd.DataFrame([
            {"event_id": "TEST_2024-05-02", "ticker": "TEST", "fill_alpha": 0.5,
             "strike": strike, "entry_cost": 3.4, "exit_value": 3.4, "ret": 0}
            for strike in (100, 105)
        ])
        monkeypatch.setattr(replay, "replay", _fake_replay(trades))
        assert ledger.score_outcomes(through="2024-05-03")["unresolvable"] == 1
        assert "unique trade" in ledger.read_outcomes()[0]["reason"]

    def test_comparison_checks_back_legs_and_quantities(self):
        from engine.ledger_settlement import comparison
        leg = {"name": "front", "right": "P", "side": "buy", "qty": 1,
               "strike": 100, "expiry": "2024-05-03"}
        back = leg | {"name": "back", "expiry": "2024-05-24"}
        row = self._row(strike=100)
        row["structure"]["legs"] = [leg, back]
        trade = {"strike": 100, "expiry": "2024-05-03", "entry_cost": 2,
                 "entry_legs": [leg, back | {"qty": 2}]}
        result = comparison(row, trade)
        assert result["strike_matched"] and result["expiry_matched"]
        assert result["contract_matched"] is False
        assert result["contract_comparison_basis"] == "all_legs"

    def test_snapshot_identity_preserves_variants_with_same_preview_contract(self, ledger_root):
        from engine.structures import put_calendar
        board = pd.DataFrame([
            {"ticker": "TEST", "strategy": "CAL-P", "event_date": "2024-05-02",
             "entry_date": "2024-05-02", "session": "AMC", "strike": 100,
             "expiry": "2024-05-03", "fill": alpha, "variant": f"back-{dte}",
             "structure_params": {"back_dte": dte},
             "structure_spec": asdict(put_calendar(back_dte=dte))}
            for dte, alpha in ((20, 0.5), (45, 0.5), (45, 0))
        ])
        rows = ledger.build_prediction_rows(board, as_of="2024-05-02")
        assert len({r["row_id"] for r in rows}) == 3
        assert len({ledger.trade_key(r) for r in rows}) == 3
        assert rows[1]["settlement"]["structure_params"] == {"back_dte": 45}
        assert rows[2]["intended_prices"]["alpha"] == 0

    def test_legacy_summary_uses_original_cost_but_does_not_invent_contracts(self, ledger_root):
        row = self._row()
        row.pop("settlement")
        ledger.write_predictions([row])
        ledger._write_outcomes([{
            "row_id": row["row_id"], "resolved_at": "2024-05-03",
            "status": "resolved", "realized_entry_cost": 3.0,
            "realized_pnl": 0.2, "realized_win": True,
        }])
        before = ledger.read_outcomes()
        summary = ledger.settlement_summary()
        assert summary["legacy_unverified"] == 1
        assert summary["contracts_compared"] == 0
        assert summary["median_abs_entry_cost_drift_fraction"] == pytest.approx(0.5)
        assert ledger.read_outcomes() == before


class TestCalibrationTrigger:
    def _fifty(self, ledger_root, monkeypatch):
        from engine import replay as replay_mod

        rows, priced = [], []
        rng = np.random.default_rng(0)
        for i in range(50):
            ticker = f"T{i:03d}"
            rows.append(_prediction(ticker=ticker, win=float(rng.uniform(0.3, 0.7)),
                                    pnl=0.03, event_id=f"{ticker}-e"))
            priced.append({"event_id": f"{ticker}-e", "ticker": ticker,
                           "strategy": "STR-THRU",
                           "event_date": pd.Timestamp("2026-10-16"),
                           "fill_alpha": 0.5, "entry_cost": 1.0, "exit_value": 1.0,
                           "ret": float(rng.normal(0.02, 0.1)), "exit_mode": "chain"})
        ledger.write_predictions(rows)
        monkeypatch.setattr(replay_mod, "replay", _fake_replay(pd.DataFrame(priced)))
        ledger.score_outcomes(through="2026-10-20")

    def test_fifty_scored_rows_regenerate_the_report_and_health(self, ledger_root, monkeypatch):
        due_before, _n, _last = ledger.calibration_due()
        assert not due_before

        self._fifty(ledger_root, monkeypatch)
        due, n_now, _ = ledger.calibration_due()
        assert due and n_now == 50

        out = ledger.calibrate()
        assert out["regenerated"] is True
        report = Path(out["report"])
        assert report.exists()
        body = report.read_text()
        assert "## 6. Calibration" in body and "Brier skill" in body

        health = json.loads(Path(out["health"]).read_text())
        for key in ("generated_at", "n_scored", "per_strategy", "champion_versions",
                    "snapshot_hash", "data_freshness", "quota_state"):
            assert key in health, f"health.json missing the frozen key {key}"
        assert health["n_scored"] == 50
        assert "STR-THRU" in health["per_strategy"]

    def test_trigger_does_not_refire_without_new_rows(self, ledger_root, monkeypatch):
        self._fifty(ledger_root, monkeypatch)
        ledger.calibrate()
        again = ledger.calibrate()
        assert again["regenerated"] is False


class TestStatus:
    def test_status_counts(self, ledger_root):
        ledger.write_predictions([_prediction()])
        st = ledger.status()
        assert st["predictions"] == 1 and st["outcomes"] == 0
        assert st["calibration_due"] is False


class TestSnapshotAuditReceipt:
    """The ledger's own leak boundary: evidence vs the row's own decision."""

    def test_forward_board_is_accepted(self):
        from engine.audit import audit_receipt_for_snapshot

        # Every nightly run looks like this: decisions days ahead of the clock.
        board = pd.DataFrame([
            {"ticker": "AAPL", "as_of": "2026-09-10", "evidence_cutoff": "2026-09-10"},
            {"ticker": "MSFT", "as_of": "2026-09-12", "evidence_cutoff": "2026-09-11"},
        ])
        receipt = audit_receipt_for_snapshot(board, decision_ts=pd.Timestamp("2026-08-30"))
        assert receipt.n_rows_checked == 2
        assert receipt.margin_seconds < 0, "a forward board's margin is negative"

    def test_evidence_after_its_own_decision_raises(self):
        from engine.audit import LeakError, audit_receipt_for_snapshot

        board = pd.DataFrame([{"ticker": "AAPL", "as_of": "2026-09-10",
                               "evidence_cutoff": "2026-09-11"}])
        with pytest.raises(LeakError, match="after their own decision"):
            audit_receipt_for_snapshot(board, decision_ts=pd.Timestamp("2026-08-30"))
