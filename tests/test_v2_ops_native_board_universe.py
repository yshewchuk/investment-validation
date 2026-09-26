"""Board-universe enumeration: parity with legacy score_calendar's ATM pass
for the native-covered strategies, DYN-SV shape, and refusal semantics.

Fully synthetic: the legacy comparison monkeypatches engine.score.store and
uses a fake calendar/scorer, so no real chain, panel, or store read happens
anywhere in this file.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_board_universe import BoardRequest, board_requests
from engine.v2.scoring.source_inputs import SUPPORTED_STRATEGIES


def _events(rows):
    return pd.DataFrame(rows)


class _FakeCalendar:
    """Every event is out of calendar range: plan_events skips all of them,
    so `score_calendar`'s pre-scoring chain-key pass needs no real chain
    index (`load_chain_index` is never called because the resulting key set
    is empty) — the comparison below stays fully synthetic."""

    def resolve_offsets(self, *args, **kwargs):
        raise KeyError("synthetic: no calendar range in this test")


class _FakeScorer:
    """Stands in for `engine.score.Scorer`: no chain, no panel, no model.
    `.score()` returns a real `ScoreResult` (a plain dataclass) built purely
    from the request's own identity fields plus fixed dummy numbers — never
    read for correctness by this test, only present so
    `dynamic_short_vol` (called at the very end of `score_calendar`) has the
    columns it needs and does not raise."""

    def __init__(self):
        self.calendar = _FakeCalendar()
        self._live_features_cache = {}

    def score(self, request, chain_index=None):
        from engine.score import ScoreResult

        return ScoreResult(
            ticker=request.ticker,
            strategy=request.strategy,
            as_of=pd.Timestamp("2026-01-01"),
            event_date=request.event_date,
            session=request.session,
            spot=100.0,
            exp_pnl_sim=0.01,
        )


def _legacy_atm_pairs(monkeypatch, events_df, as_of, horizon_days, tickers, strategies):
    """The `(ticker, strategy)` pairs legacy `score_calendar` would enumerate
    for `strategies`, on the ATM pass, against `events_df` — fully mocked,
    read-only comparison, never a second implementation of the filter."""
    import engine.score as score_mod

    monkeypatch.setattr(
        score_mod.store, "read_table",
        lambda *a, **k: events_df.copy(),
    )
    frame = score_mod.score_calendar(
        as_of=as_of,
        horizon_days=horizon_days,
        strategies=strategies,
        scorer=_FakeScorer(),
        tickers=tickers,
        progress_every=0,
        quote_max_age_sessions=None,
    )
    atm = frame[frame["strategy"].isin(strategies)]
    if "strike_offset" in atm.columns:
        atm = atm[atm["strike_offset"].isna()]
    return set(zip(atm["ticker"], atm["strategy"]))


class TestBoardUniverseParityWithLegacy:
    def test_native_covered_pairs_match_score_calendar_atm_pass(self, monkeypatch):
        events_df = _events([
            {"event_id": "e1", "ticker": "AAA", "event_date": pd.Timestamp("2026-02-01"), "session": "BMO"},
            {"event_id": "e2", "ticker": "BBB", "event_date": pd.Timestamp("2026-02-03"), "session": "AMC"},
        ])
        as_of = pd.Timestamp("2026-01-25")
        horizon_days = 21
        covered = sorted(SUPPORTED_STRATEGIES)

        native_pairs = {
            (r.ticker, r.strategy)
            for r in board_requests(as_of, horizon_days, None, events_df)
            if r.strategy != "DYN-SV"
        }
        legacy_pairs = _legacy_atm_pairs(
            monkeypatch, events_df, as_of, horizon_days, None, covered,
        )
        assert native_pairs == legacy_pairs
        assert native_pairs  # the fixture must actually exercise something


class TestDynSvIsOncePerEvent:
    def test_exactly_one_dyn_sv_request_per_event(self):
        events_df = _events([
            {"event_id": "e1", "ticker": "AAA", "event_date": pd.Timestamp("2026-02-01"), "session": "BMO"},
        ])
        requests = board_requests(
            pd.Timestamp("2026-01-25"), 21, None, events_df,
        )
        dyn_sv = [r for r in requests if r.strategy == "DYN-SV"]
        assert len(dyn_sv) == 1
        assert dyn_sv[0].ticker == "AAA"


class TestMalformedEventsTableRefuses:
    @pytest.mark.parametrize("drop_column", ["ticker", "event_date", "session"])
    def test_missing_required_column_raises_invalid_request(self, drop_column):
        events_df = _events([
            {"ticker": "AAA", "event_date": pd.Timestamp("2026-02-01"), "session": "BMO"},
        ]).drop(columns=[drop_column])
        with pytest.raises(OpsError) as excinfo:
            board_requests(pd.Timestamp("2026-01-25"), 21, None, events_df)
        assert excinfo.value.code == "INVALID_REQUEST"


class TestDeterministicOrder:
    def test_output_order_is_sorted_events_then_strategies_then_dyn_sv_last(self):
        events_df = _events([
            {"ticker": "BBB", "event_date": pd.Timestamp("2026-02-03"), "session": "AMC"},
            {"ticker": "AAA", "event_date": pd.Timestamp("2026-02-01"), "session": "BMO"},
        ])
        covered = sorted(SUPPORTED_STRATEGIES)
        requests = board_requests(
            pd.Timestamp("2026-01-25"), 21, None, events_df,
        )
        expected = tuple(
            BoardRequest(ticker, strategy, event_date, session)
            for ticker, event_date, session in [
                ("AAA", pd.Timestamp("2026-02-01"), "BMO"),
                ("BBB", pd.Timestamp("2026-02-03"), "AMC"),
            ]
            for strategy in (*covered, "DYN-SV")
        )
        assert requests == expected

    def test_two_calls_agree_and_are_not_just_agreeing_by_accident(self):
        events_df = _events([
            {"ticker": "AAA", "event_date": pd.Timestamp("2026-02-01"), "session": "BMO"},
        ])
        first = board_requests(pd.Timestamp("2026-01-25"), 21, None, events_df)
        second = board_requests(pd.Timestamp("2026-01-25"), 21, None, events_df)
        assert first == second
