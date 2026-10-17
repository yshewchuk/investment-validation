"""The batched ``daily_market`` read equals the one-shot read, exactly.

When ``daily=None``, :func:`add_orats_features` and :func:`add_pre_print_vol`
stream ``daily_market`` in ticker batches (:func:`_iter_daily_batches`) to bound
peak memory. Batching must be a pure memory optimization: every value, dtype and
NaN the panel carries has to match the explicit-``daily`` path, or the two read
paths would silently produce different panels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data import store
from engine.data.features import panel
from engine.data.features.panel import (
    ORATS_FEATURES,
    PRE_PRINT_FEATURES,
    add_orats_features,
    add_pre_print_vol,
)

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
ABSENT_TICKER = "ZZZ"
BATCH_SIZE = 2

#: Every source column either reader can ask for, de-duplicated: ``iv30`` and
#: ``exern_iv30`` are shared by both feature maps.
_SOURCES = list(dict.fromkeys([*ORATS_FEATURES, *PRE_PRINT_FEATURES]))

_ORATS_NEEDED = [
    "ticker", "date", "mcap_usd", "mcap_log", "src_iv", *ORATS_FEATURES.keys()
]
_PRE_PRINT_NEEDED = ["ticker", "date", *PRE_PRINT_FEATURES.keys()]


def _daily_frame() -> pd.DataFrame:
    """5 tickers x ~700 business days over 2022-2024, deterministic.

    Ragged in the ways the readers must tolerate: NaN in the IV columns and in
    ``mcap_usd``, one ticker whose ``src_iv`` is blank early, and one ticker
    whose coverage starts late so an event can precede its first row.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=700)
    frames = []
    for ticker in TICKERS:
        n = len(dates)
        mcap = rng.uniform(1e9, 5e9, n)
        frame = pd.DataFrame(
            {
                "ticker": ticker,
                "date": dates,
                "mcap_usd": mcap,
                "mcap_log": np.log(mcap),
                "src_iv": "orats.summaries",
            }
        )
        for source in _SOURCES:
            frame[source] = rng.uniform(5.0, 60.0, n)
        frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)

    # One ticker with no ORATS-summaries rows early: an event inside that gap
    # must find no prior IV row rather than reach forward to the first one.
    aaa_early = daily.index[daily["ticker"] == "AAA"][:400]
    daily.loc[aaa_early, "src_iv"] = np.nan

    # One ticker whose coverage starts late, so an event can precede its first
    # daily_market row.
    keep = ~(
        (daily["ticker"] == "EEE") & (daily["date"] < pd.Timestamp("2023-06-01"))
    )
    daily = daily.loc[keep].reset_index(drop=True)

    # Scattered gaps the reader must carry as NaN, not fill.
    n = len(daily)
    iv_gap = rng.random(n) < 0.03
    for source in _SOURCES:
        daily.loc[iv_gap, source] = np.nan
    daily.loc[rng.random(n) < 0.03, "mcap_usd"] = np.nan

    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
    return daily


def _events() -> pd.DataFrame:
    rows = [
        ("AAA", "2023-03-15"),
        ("AAA", "2024-05-20"),
        ("BBB", "2023-06-10"),
        ("BBB", "2024-08-01"),
        ("CCC", "2023-09-05"),
        ("CCC", "2024-02-14"),
        ("DDD", "2023-11-20"),
        ("DDD", "2024-08-15"),
        # Before EEE's first daily_market row (its coverage starts 2023-06-01).
        ("EEE", "2023-02-01"),
        ("EEE", "2024-04-10"),
        # Absent from daily_market entirely.
        (ABSENT_TICKER, "2023-07-01"),
        (ABSENT_TICKER, "2024-07-01"),
    ]
    return pd.DataFrame(
        {
            "ticker": [ticker for ticker, _ in rows],
            "date": pd.to_datetime([date for _, date in rows]).astype("datetime64[ns]"),
        }
    )


def _sessions(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": events["ticker"].to_numpy(),
            "event_date": events["date"].to_numpy(),
            "session": [
                "BMO", "AMC", "AMC", "BMO", "BMO", "AMC",
                "AMC", "BMO", "BMO", "AMC", "AMC", "BMO",
            ],
        }
    )


def _patch_iter_table(monkeypatch, daily: pd.DataFrame, calls: list) -> None:
    """Fake ``store.iter_table`` yielding one ``(year, frame[columns])`` per year."""

    def fake_iter_table(name, *, years=None, columns=None):
        calls.append(name)
        frame = daily if columns is None else daily[list(columns)]
        for year, group in frame.groupby(frame["date"].dt.year, sort=True):
            yield int(year), group.reset_index(drop=True)

    monkeypatch.setattr(panel.store, "iter_table", fake_iter_table)


def _one_shot(daily: pd.DataFrame, columns) -> pd.DataFrame:
    return store.coerce(daily, "daily_market", only=list(columns))


class TestBatchedOratsRead:
    def test_batched_equals_one_shot(self, monkeypatch):
        daily = _daily_frame()
        events = _events()

        reference = add_orats_features(events, daily=_one_shot(daily, _ORATS_NEEDED))

        calls: list = []
        _patch_iter_table(monkeypatch, daily, calls)
        monkeypatch.setattr(panel, "_DAILY_BATCH_TICKERS", BATCH_SIZE)
        batched = add_orats_features(events, daily=None)

        pd.testing.assert_frame_equal(batched, reference, check_exact=True)
        # Several batches, so the streamed path really was exercised.
        assert len(calls) > 1
        assert set(batched.columns) > set(events.columns)

        # Independent oracle: plain pandas over the raw fixture, no panel code.
        # (BBB, 2024-08-01) anchors on the last ORATS-summary row strictly
        # before the event date — that row's implied move, stamped in
        # ``orats_asof``.
        event_date = pd.Timestamp("2024-08-01")
        prior = daily[
            (daily["ticker"] == "BBB")
            & daily["src_iv"].notna()
            & (daily["date"] < event_date)
        ].sort_values("date")
        assert not prior.empty
        last = prior.iloc[-1]
        mask = (batched["ticker"] == "BBB") & (batched["date"] == event_date)
        assert int(mask.sum()) == 1
        np.testing.assert_array_equal(
            batched.loc[mask, ORATS_FEATURES["implied_move"]].to_numpy(),
            np.array([last["implied_move"]]),
        )
        np.testing.assert_array_equal(
            batched.loc[mask, "orats_asof"].to_numpy(dtype="datetime64[ns]"),
            np.array([last["date"].to_datetime64()], dtype="datetime64[ns]"),
        )

        # No prior ORATS row at all: the absent ticker (both events) and EEE's
        # event before its own coverage starts — NaN feature, NaT as-of.
        for ticker, date in [
            (ABSENT_TICKER, "2023-07-01"),
            (ABSENT_TICKER, "2024-07-01"),
            ("EEE", "2023-02-01"),
        ]:
            row = (batched["ticker"] == ticker) & (
                batched["date"] == pd.Timestamp(date)
            )
            assert int(row.sum()) == 1
            assert batched.loc[row, ORATS_FEATURES["implied_move"]].isna().all()
            assert batched.loc[row, "orats_asof"].isna().all()


class TestBatchedPrePrintVolRead:
    def test_batched_equals_one_shot(self, monkeypatch):
        daily = _daily_frame()
        events = _events()
        sessions = _sessions(events)

        reference = add_pre_print_vol(
            events, daily=_one_shot(daily, _PRE_PRINT_NEEDED), events=sessions
        )

        calls: list = []
        _patch_iter_table(monkeypatch, daily, calls)
        monkeypatch.setattr(panel, "_DAILY_BATCH_TICKERS", BATCH_SIZE)
        batched = add_pre_print_vol(events, daily=None, events=sessions)

        pd.testing.assert_frame_equal(batched, reference, check_exact=True)
        assert len(calls) > 1
        assert set(batched.columns) > set(events.columns)

        # Independent oracle: plain pandas over the raw fixture, no panel code.
        # The last session before the print — strictly before the event date
        # for BMO, on or before it for AMC — recomputed for all 12 events.
        for ticker, event_date, session in zip(
            sessions["ticker"], sessions["event_date"], sessions["session"]
        ):
            mask = (batched["ticker"] == ticker) & (batched["date"] == event_date)
            assert int(mask.sum()) == 1
            rows = daily[daily["ticker"] == ticker]
            if session == "AMC":
                prior = rows[rows["date"] <= event_date]
            else:
                prior = rows[rows["date"] < event_date]
            if prior.empty:
                # No daily row to anchor on, so every feature stays NaN.
                assert (
                    batched.loc[mask, list(PRE_PRINT_FEATURES.values())]
                    .isna()
                    .all()
                    .all()
                )
                continue
            expected = prior.sort_values("date").iloc[-1]
            for source, column in PRE_PRINT_FEATURES.items():
                np.testing.assert_array_equal(
                    batched.loc[mask, column].to_numpy(),
                    np.array([expected[source]]),
                )


class TestIterDailyBatches:
    def test_each_yielded_frame_holds_only_its_batch(self, monkeypatch):
        daily = _daily_frame()
        calls: list = []
        _patch_iter_table(monkeypatch, daily, calls)

        tickers = [*TICKERS, "ZZY", ABSENT_TICKER]
        ordered = sorted(set(tickers))
        expected_batches = [
            ordered[start : start + BATCH_SIZE]
            for start in range(0, len(ordered), BATCH_SIZE)
        ]

        frames = list(
            panel._iter_daily_batches(_ORATS_NEEDED, tickers, batch_size=BATCH_SIZE)
        )

        assert len(frames) == len(expected_batches)
        for frame, batch in zip(frames, expected_batches):
            assert set(frame["ticker"].unique()) <= set(batch)
        assert len(calls) == len(expected_batches)

        # One batch holds no rows at all (its tickers are absent from
        # daily_market) and must come back as the schema's empty frame with the
        # requested columns, not a bare one.
        empty = [
            (frame, batch)
            for frame, batch in zip(frames, expected_batches)
            if frame.empty
        ]
        assert len(empty) == 1
        empty_frame, empty_batch = empty[0]
        assert set(empty_batch) <= {ABSENT_TICKER, "ZZY"}
        assert list(empty_frame.columns) == _ORATS_NEEDED

        # Completeness: the batches together are exactly the one-shot read.
        streamed = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )
        one_shot = (
            _one_shot(daily, _ORATS_NEEDED)
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(streamed, one_shot, check_exact=True)

    def test_no_batch_or_partition_is_retained_across_yield(self, monkeypatch):
        import gc
        import weakref

        daily = _daily_frame()
        partition_refs: list = []

        def fake_iter_table(name, *, years=None, columns=None):
            frame = daily if columns is None else daily[list(columns)]
            for year, group in frame.groupby(frame["date"].dt.year, sort=True):
                part = group.reset_index(drop=True)
                partition_refs.append(weakref.ref(part))
                yield int(year), part

        monkeypatch.setattr(panel.store, "iter_table", fake_iter_table)

        gen = panel._iter_daily_batches(
            _ORATS_NEEDED, TICKERS, batch_size=BATCH_SIZE
        )
        assert not partition_refs
        first = next(gen)
        assert partition_refs, "no partitions were yielded"
        gc.collect()
        assert all(ref() is None for ref in partition_refs)

        # The yielded batch itself is the caller's alone: dropping it frees it
        # even though the generator is still suspended on that yield.
        ref_first = weakref.ref(first)
        del first
        gc.collect()
        assert ref_first() is None
        assert next(gen) is not None
