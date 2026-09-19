"""The analog matcher.

Guide acceptance test 5: a synthetic trade table with known bucket means must
come back with those means, and the widening ladder must fire at n < 30 in the
specified order. Both are here, plus the causality rule that keeps a 2019 score
from being informed by 2024 trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.analogs import (
    WIDENING_ORDER,
    AnalogMatcher,
    bucket_frame,
    _bucket,
    _dte_band,
)


def trades(
    n: int,
    *,
    ret: float,
    mcap: float,
    dte: float,
    moneyness_pct: float = 0.0,
    implied_ratio: float = 1.0,
    strategy: str = "STR-THRU",
    alpha: float = 0.5,
    year: int = 2020,
) -> pd.DataFrame:
    """A block of identical trades landing in one bucket, with a known mean."""
    spot = 100.0
    return pd.DataFrame(
        {
            "strategy": strategy,
            "fill_alpha": alpha,
            "ret": ret,
            "mcap_usd": mcap,
            "dte_entry": dte,
            "spot_entry": spot,
            "strike": spot * (1 + moneyness_pct / 100.0),
            "or_implied": 5.0 * implied_ratio,
            "mean_prior_or_implied": 5.0,
            "event_date": pd.Timestamp(f"{year}-06-01"),
            "exit_date": pd.Timestamp(f"{year}-06-02"),
        },
        index=range(n),
    )


class TestBucketing:
    def test_mcap_edges(self):
        from engine.analogs import MCAP_EDGES, MCAP_LABELS

        assert list(_bucket([5e8, 2e9, 5e10], MCAP_EDGES, MCAP_LABELS)) == [
            "<1B", "1-10B", ">=10B"
        ]

    def test_mcap_edge_belongs_to_the_bucket_above(self):
        from engine.analogs import MCAP_EDGES, MCAP_LABELS

        assert _bucket([1e9], MCAP_EDGES, MCAP_LABELS)[0] == "1-10B"

    def test_non_finite_is_unmatchable_not_a_bucket(self):
        from engine.analogs import MCAP_EDGES, MCAP_LABELS

        assert _bucket([np.nan], MCAP_EDGES, MCAP_LABELS)[0] is None

    def test_dte_bands_and_the_gap_above_45(self):
        assert list(_dte_band([1, 5, 20, 30, 60])) == ["1-3", "4-10", "11-25", "26-45", None]

    def test_bucket_frame_computes_moneyness_from_strike_and_spot(self):
        frame = bucket_frame(trades(1, ret=0.1, mcap=5e9, dte=5, moneyness_pct=3.0))
        assert frame["moneyness_pct"].iloc[0] == pytest.approx(3.0)
        assert frame["moneyness_band"].iloc[0] == "2-5%"

    def test_the_entry_date_implied_move_wins_where_supplied(self):
        """STR-RUNUP enters two weeks early; its entry-date reading is the one
        that must be matched against, not the event-date one."""
        pool = trades(40, ret=0.0, mcap=5e9, dte=5, implied_ratio=2.0)
        pool["implied_at_entry"] = 2.5  # ratio 0.5 against a prior mean of 5.0
        frame = bucket_frame(pool, implied_edges=(0.8, 1.5))
        assert frame["implied_ratio"].iloc[0] == pytest.approx(0.5)
        assert frame["implied_tercile"].iloc[0] == "low"

    def test_falls_back_to_the_event_level_implied_move(self):
        frame = bucket_frame(
            trades(40, ret=0.0, mcap=5e9, dte=5, implied_ratio=2.0),
            implied_edges=(0.8, 1.5),
        )
        assert frame["implied_ratio"].iloc[0] == pytest.approx(2.0)
        assert frame["implied_tercile"].iloc[0] == "high"

    def test_an_empty_frame_buckets_without_raising(self):
        """The fresh-install state: no replay has run yet."""
        from engine.data.schemas import empty_frame

        frame = bucket_frame(empty_frame("trades"))
        assert frame.empty
        for column in ("mcap_bucket", "dte_band", "moneyness_band", "implied_tercile"):
            assert column in frame.columns

    def test_missing_columns_yield_unmatchable_dimensions(self):
        """A dimension we cannot measure must be None, not an exception."""
        frame = bucket_frame(pd.DataFrame({"strategy": ["STR-THRU"], "ret": [0.1]}))
        assert frame["mcap_bucket"].iloc[0] is None
        assert frame["dte_band"].iloc[0] is None
        assert frame["moneyness_band"].iloc[0] is None
        assert frame["implied_tercile"].iloc[0] is None

    def test_implied_terciles_are_cut_on_the_pool(self):
        pool = pd.concat(
            [trades(20, ret=0.0, mcap=5e9, dte=5, implied_ratio=r) for r in (0.5, 1.0, 2.0)],
            ignore_index=True,
        )
        frame = bucket_frame(pool)
        assert set(frame["implied_tercile"]) == {"low", "mid", "high"}


class TestMatching:
    def test_no_dimension_has_a_value_is_an_empty_match_not_a_crash(self):
        """No chain, no size, no implied quote: every bucket is None.

        The widening loop has zero active dimensions and used to fall through
        to an 'unreachable' assert — hit live by board tickers with no ORATS
        daily rows and no chain (CANG, PDEX, ...). Matching on nothing must
        return the empty set, not the base rate wearing an empty label.
        """
        pool = trades(30, ret=0.10, mcap=5e9, dte=5)
        matcher = AnalogMatcher(pool, snapshot="test")
        result = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=None, dte=None, moneyness_pct=None,
                                implied_ratio=None),
            alpha=0.5,
        )
        assert result.n == 0
        assert result.mean is None
        assert result.thin
        assert len(result.unavailable) == 4

    def test_returns_the_known_bucket_mean(self):
        """Guide test 5: hand-built buckets → the matcher returns their means."""
        pool = pd.concat(
            [
                trades(60, ret=0.20, mcap=5e9, dte=5),
                trades(60, ret=-0.40, mcap=5e10, dte=5),
            ],
            ignore_index=True,
        )
        matcher = AnalogMatcher(pool, snapshot="test")
        small = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        assert small.n == 60
        assert small.mean == pytest.approx(0.20)
        assert small.win_rate == 1.0
        assert small.widened == 0

        large = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=5e10, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        assert large.mean == pytest.approx(-0.40)

    def test_alpha_selects_the_right_slice(self):
        pool = pd.concat(
            [
                trades(50, ret=-0.30, mcap=5e9, dte=5, alpha=0.0),
                trades(50, ret=0.10, mcap=5e9, dte=5, alpha=0.5),
            ],
            ignore_index=True,
        )
        matcher = AnalogMatcher(pool, snapshot="test")
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0
        )
        assert matcher.match("STR-THRU", buckets, alpha=0.0).mean == pytest.approx(-0.30)
        assert matcher.match("STR-THRU", buckets, alpha=0.5).mean == pytest.approx(0.10)

    def test_strategy_selects_the_right_slice(self):
        pool = pd.concat(
            [
                trades(50, ret=0.10, mcap=5e9, dte=5, strategy="STR-THRU"),
                trades(50, ret=-0.20, mcap=5e9, dte=5, strategy="STR-RUNUP"),
            ],
            ignore_index=True,
        )
        matcher = AnalogMatcher(pool, snapshot="test")
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0
        )
        assert matcher.match("STR-RUNUP", buckets, alpha=0.5).mean == pytest.approx(-0.20)


class TestWidening:
    def build(self):
        """Too few in the exact bucket; more once moneyness is dropped."""
        exact = trades(10, ret=1.0, mcap=5e9, dte=5, moneyness_pct=0.0)
        wider = trades(50, ret=0.0, mcap=5e9, dte=5, moneyness_pct=4.0)
        return AnalogMatcher(pd.concat([exact, wider], ignore_index=True), snapshot="t")

    def test_drops_moneyness_first(self):
        matcher = self.build()
        result = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        assert result.widened == 1
        assert result.dropped == ("moneyness_band",)
        assert result.n == 60

    def test_ladder_order_is_fixed(self):
        """Not "whichever dimension yields the most matches"."""
        assert WIDENING_ORDER == ("moneyness_band", "dte_band", "implied_tercile")

    def test_widens_further_when_still_thin(self):
        exact = trades(5, ret=1.0, mcap=5e9, dte=5, moneyness_pct=0.0)
        other_dte = trades(80, ret=0.0, mcap=5e9, dte=30, moneyness_pct=0.0)
        matcher = AnalogMatcher(pd.concat([exact, other_dte], ignore_index=True), snapshot="t")
        result = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        assert result.dropped[:2] == ("moneyness_band", "dte_band")
        assert result.n == 85

    def test_mcap_is_never_dropped(self):
        """Size is the dimension the evidence is most stratified on."""
        assert "mcap_bucket" not in WIDENING_ORDER
        matcher = AnalogMatcher(trades(80, ret=0.5, mcap=5e10, dte=5), snapshot="t")
        result = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=5e8, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        assert result.n == 0

    def test_thin_after_full_widening_is_reported_not_invented(self):
        matcher = AnalogMatcher(trades(4, ret=0.5, mcap=5e9, dte=5), snapshot="t")
        result = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        assert result.n == 4
        assert result.thin is True
        assert result.ci_low is None and result.ci_high is None
        assert result.widened == len(WIDENING_ORDER)


class TestCausality:
    def test_only_trades_closed_before_the_decision_are_eligible(self):
        past = trades(40, ret=0.10, mcap=5e9, dte=5, year=2018)
        future = trades(40, ret=9.99, mcap=5e9, dte=5, year=2024)
        matcher = AnalogMatcher(pd.concat([past, future], ignore_index=True), snapshot="t")
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0
        )
        result = matcher.match(
            "STR-THRU", buckets, alpha=0.5, as_of=pd.Timestamp("2020-01-01")
        )
        assert result.n == 40
        assert result.mean == pytest.approx(0.10)
        assert result.years == (2018,)

    def test_a_trade_still_open_on_the_decision_date_is_excluded(self):
        pool = trades(40, ret=0.10, mcap=5e9, dte=5, year=2020)
        matcher = AnalogMatcher(pool, snapshot="t")
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0
        )
        # Exits 2020-06-02; deciding on 2020-06-02 must not see it.
        assert matcher.match(
            "STR-THRU", buckets, alpha=0.5, as_of=pd.Timestamp("2020-06-02")
        ).n == 0
        assert matcher.match(
            "STR-THRU", buckets, alpha=0.5, as_of=pd.Timestamp("2020-06-03")
        ).n == 40


class TestBootstrap:
    def pool(self):
        rng = np.random.default_rng(1)
        frame = trades(200, ret=0.0, mcap=5e9, dte=5)
        frame["ret"] = rng.normal(0.05, 0.5, 200)
        return frame

    def test_ci_brackets_the_mean(self):
        matcher = AnalogMatcher(self.pool(), snapshot="t")
        result = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        assert result.ci_low < result.mean < result.ci_high

    def test_deterministic_for_the_same_request_and_snapshot(self):
        pool = self.pool()
        buckets = dict(
            mcap_bucket="1-10B", dte_band="4-10", moneyness_band="ATM", implied_tercile="mid"
        )
        first = AnalogMatcher(pool, snapshot="snap-a").match(
            "STR-THRU", buckets, alpha=0.5, request_key="k"
        )
        second = AnalogMatcher(pool, snapshot="snap-a").match(
            "STR-THRU", buckets, alpha=0.5, request_key="k"
        )
        assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)

    def test_the_interval_does_not_depend_on_the_pool_row_order(self):
        """The same analogs shuffled must give the same interval.

        `rng.choice` samples by INDEX, so a fixed seed picks the same
        positions and a reordered pool resamples different values from an
        identical population. Every other statistic here is order-invariant,
        which is how this stayed hidden: on 2026-09-11 a board and its own
        re-score agreed on n_analogs, exp_pnl_analog and win_analog and
        disagreed on ci_low/ci_high for the same set.

        The existing determinism test passes the SAME frame twice, so it
        could never have caught it.
        """
        pool = self.pool()
        shuffled = pool.sample(frac=1.0, random_state=99).reset_index(drop=True)
        buckets = dict(
            mcap_bucket="1-10B", dte_band="4-10", moneyness_band="ATM", implied_tercile="mid"
        )
        first = AnalogMatcher(pool, snapshot="snap-a").match(
            "STR-THRU", buckets, alpha=0.5, request_key="k"
        )
        second = AnalogMatcher(shuffled, snapshot="snap-a").match(
            "STR-THRU", buckets, alpha=0.5, request_key="k"
        )
        assert first.n == second.n
        assert first.mean == pytest.approx(second.mean)
        assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)

    def test_a_different_snapshot_is_a_different_draw(self):
        pool = self.pool()
        buckets = dict(
            mcap_bucket="1-10B", dte_band="4-10", moneyness_band="ATM", implied_tercile="mid"
        )
        a = AnalogMatcher(pool, snapshot="snap-a").match("STR-THRU", buckets, alpha=0.5)
        b = AnalogMatcher(pool, snapshot="snap-b").match("STR-THRU", buckets, alpha=0.5)
        assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)
        assert a.mean == b.mean  # the sample is the same; only the resampling differs

    def test_percentiles_describe_the_trade_distribution(self):
        matcher = AnalogMatcher(self.pool(), snapshot="t")
        result = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        # p10/p90 span single trades and must be far wider than the CI on the mean.
        assert result.p10 < result.ci_low
        assert result.p90 > result.ci_high


class TestAnalogSet:
    def test_serializes_for_a_report(self):
        matcher = AnalogMatcher(trades(50, ret=0.1, mcap=5e9, dte=5), snapshot="t")
        result = matcher.match(
            "STR-THRU",
            matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        doc = result.as_dict()
        assert doc["n_analogs"] == 50
        assert doc["buckets"]["mcap_bucket"] == "1-10B"

    def test_empty_pool_is_reported_as_thin(self):
        matcher = AnalogMatcher(trades(10, ret=0.1, mcap=5e9, dte=5), snapshot="t")
        result = matcher.match(
            "STR-RUNUP",
            matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0),
            alpha=0.5,
        )
        assert result.n == 0 and result.thin and result.mean is None


class TestCausalPoolCache:
    def _matcher(self):
        frame = pd.concat([
            trades(40, ret=0.05, mcap=5e9, dte=5, year=2020),
            trades(40, ret=-0.05, mcap=5e9, dte=5, year=2021),
        ], ignore_index=True)
        return AnalogMatcher(bucket_frame(frame))

    def test_same_as_of_reuses_the_cached_pool(self):
        matcher = self._matcher()
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
        first = matcher.match("STR-THRU", buckets, alpha=0.5,
                              as_of="2030-01-01")
        assert len(matcher._causal_pools) == 1
        again = matcher.match("STR-THRU", buckets, alpha=0.5,
                              as_of="2030-01-01")
        assert len(matcher._causal_pools) == 1
        assert again.mean == first.mean and again.n == first.n

    def test_cache_bounded_under_many_as_of(self):
        matcher = self._matcher()
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
        matcher.MAX_CAUSAL_CACHE = 2
        for as_of in ("2030-01-01", "2030-01-02", "2030-01-03"):
            matcher.match("STR-THRU", buckets, alpha=0.5, as_of=as_of)
        assert len(matcher._causal_pools) == 2

    def test_cache_cap_is_sized_to_the_board_not_to_a_round_number(self):
        """`MAX_CAUSAL_CACHE` is now a loose backstop, not the memory bound.

        2026-09-18, revised: a real 40-forward-event strict capture measured
        per-key cost varying roughly 10x key to key (run 3: +5 keys -> +770
        MB; elsewhere in the same run, +30 keys -> +350 MB), so a fixed
        key-count ceiling cannot bound this cache's memory — a workload that
        happens to visit large-population keys blows through any count cap
        sized for the small end of that range, and one sized for the large
        end wastes slots everywhere else. `CAUSAL_CACHE_BUDGET_BYTES`
        (module level) is the actual memory ceiling now (see
        `test_byte_budget_evicts_before_the_key_count_cap_would` below);
        `MAX_CAUSAL_CACHE` only guards against unbounded DICT overhead from
        a degenerate many-tiny-keys workload, so it can stay generous.

        LOWER bound unchanged: a full three-week board (3,120 rows)
        generates 34 distinct (strategy, alpha, as_of) keys — 31 entry dates
        x 2 scoreable strategies. A cap below that makes the cache evict
        entries the board is still using, on the one path where the cache
        actually pays.
        """
        cap = self._matcher().MAX_CAUSAL_CACHE
        assert cap >= 40, "cap sits below the measured 34-key board working set"
        # The byte budget is the real ceiling; the count backstop just needs
        # to stay well above the board's own working set (34) without being
        # so low it competes with the byte budget for which one binds first
        # in the normal (small-to-moderate per-key cost) case.
        assert cap <= 4096, "key-count backstop should not itself need scaling"

    def test_byte_budget_evicts_before_the_key_count_cap_would(self):
        """A handful of LARGE per-key pools must evict on bytes alone, well
        under the key-count backstop — the exact gap a count-only cap left
        open (measured: 5 keys costing 770 MB in one real run).
        """
        from engine import analogs as analogs_mod

        wide = pd.concat([
            trades(4000, ret=0.05, mcap=5e9, dte=5, year=2020 + i)
            for i in range(6)
        ], ignore_index=True)
        matcher = AnalogMatcher(bucket_frame(wide))
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
        original_budget = analogs_mod.CAUSAL_CACHE_BUDGET_BYTES
        analogs_mod.CAUSAL_CACHE_BUDGET_BYTES = 2_000_000  # force eviction fast
        try:
            for i in range(6):
                matcher.match("STR-THRU", buckets, alpha=0.5,
                              as_of=f"{2030 + i}-01-01")
            # Well under the count backstop (which never fired here) and
            # under 6 -- the budget evicted keys long before the count cap
            # would have needed to.
            assert len(matcher._causal_pools) < 6
            assert matcher._causal_cache_total_bytes <= (
                analogs_mod.CAUSAL_CACHE_BUDGET_BYTES
                + max(matcher._causal_cache_bytes.values(), default=0)
            )
        finally:
            analogs_mod.CAUSAL_CACHE_BUDGET_BYTES = original_budget


class TestPopulationPoolCache:
    """2026-09-18: an instrumented strict-capture run measured the forward
    pass's analog step costing net +1.48 GB of real RSS while
    `CAUSAL_CACHE_BUDGET_BYTES` (600 MB) stayed the ceiling — because
    `_pools`/`_population_row_caches` (keyed only on (strategy, alpha),
    documenting the WHOLE population the first time any candidate touches a
    key) were entirely invisible to that budget's accounting. These tests
    check that they are now priced into the SAME total and evicted, causal
    keys first, by the SAME LRU discipline the causal family already had.
    """

    def _matcher(self, *, alphas=(0.5,), n=4000):
        frame = pd.concat([
            trades(n, ret=0.05, mcap=5e9, dte=5, year=2020, alpha=alpha)
            for alpha in alphas
        ], ignore_index=True)
        return AnalogMatcher(bucket_frame(frame))

    def test_population_pool_is_priced_at_first_touch(self):
        matcher = self._matcher()
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=None)

        key = ("STR-THRU", 0.5)
        assert key in matcher._pools
        assert key in matcher._causal_cache_bytes
        assert matcher._causal_cache_bytes[key] > 0
        assert matcher._causal_cache_total_bytes == matcher._causal_cache_bytes[key]

    def test_population_pool_is_priced_once_not_once_per_call(self):
        """A cache HIT on an already-seeded key must not re-charge its cost
        — `_evidence_rows`' own row cache already makes the CPU side of a
        repeat touch cheap; this checks the byte ledger agrees.
        """
        matcher = self._matcher()
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=None)
        first_total = matcher._causal_cache_total_bytes

        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=None)

        assert matcher._causal_cache_total_bytes == first_total

    def test_population_touch_refreshes_recency(self):
        matcher = self._matcher()
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
        key = ("STR-THRU", 0.5)

        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=None)
        assert list(matcher._population_pool_order) == [key]
        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=None)
        # A dict-order re-insertion, not a duplicate or a reordering bug:
        # exactly one entry, still present, order preserved (nothing else
        # to reorder against with only one key touched so far).
        assert list(matcher._population_pool_order) == [key]

    def test_population_entries_evict_only_after_causal_is_exhausted(self):
        """Causal keys, being numerous and cheap to rebuild, must absorb
        eviction pressure before a population key's expensive
        full-population documentation is thrown away.
        """
        from engine import analogs as analogs_mod

        matcher = self._matcher()
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=None)  # seeds _pools
        population_cost = matcher._causal_cache_bytes[("STR-THRU", 0.5)]

        original_budget = analogs_mod.CAUSAL_CACHE_BUDGET_BYTES
        # Room for the population entry plus a sliver for causal ones —
        # every later causal key must therefore evict an OLDER causal key,
        # never the population entry, as long as any causal key remains.
        analogs_mod.CAUSAL_CACHE_BUDGET_BYTES = population_cost + 1
        try:
            for i in range(5):
                matcher.match("STR-THRU", buckets, alpha=0.5,
                              as_of=f"{2031 + i}-01-01")
            assert ("STR-THRU", 0.5) in matcher._pools
            assert len(matcher._causal_pools) <= 1
        finally:
            analogs_mod.CAUSAL_CACHE_BUDGET_BYTES = original_budget

    def test_population_lru_eviction_once_causal_is_empty(self):
        """With no causal entries to absorb the pressure, a new population
        key's arrival evicts the LEAST-recently-touched population key —
        never an arbitrary or insertion-order-only choice.
        """
        from engine import analogs as analogs_mod

        matcher = self._matcher(alphas=(0.5, 0.75, 0.9), n=2000)
        buckets = matcher.buckets_for(
            mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)

        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=None)
        matcher.match("STR-THRU", buckets, alpha=0.75, as_of=None)
        key_a, key_b = ("STR-THRU", 0.5), ("STR-THRU", 0.75)
        cost_each = matcher._causal_cache_bytes[key_a]

        # Touch key_a again -- key_b is now the least-recently-used.
        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=None)

        original_budget = analogs_mod.CAUSAL_CACHE_BUDGET_BYTES
        # Room for exactly two entries; a third's arrival must evict one.
        analogs_mod.CAUSAL_CACHE_BUDGET_BYTES = cost_each * 2 + 1
        try:
            matcher.match("STR-THRU", buckets, alpha=0.9, as_of=None)
            assert key_a in matcher._pools, "refreshed key must survive"
            assert key_b not in matcher._pools, "least-recently-used key must evict"
            assert key_b not in matcher._population_row_caches
            assert ("STR-THRU", 0.9) in matcher._pools
        finally:
            analogs_mod.CAUSAL_CACHE_BUDGET_BYTES = original_budget


def test_causal_cache_budget_is_400mb_not_600mb():
    """2026-09-19: lowered from 600 MB to 400 MB after a synthetic
    benchmark (scratch/bench_analog_budget.py, untracked) measured the
    cost -- ~2.25 MB/causal-key, so 400 MB still holds ~177 concurrent
    keys, comfortably above a real capture's causal-key cardinality (a
    handful of strategies x ALPHA_GRID(5) x the boundary events' own as_of
    dates). See CAUSAL_CACHE_BUDGET_BYTES's own docstring for the full
    numbers. A regression back to 600 MB would silently give up the ~200
    MB of headroom this round was measured and committed to free.
    """
    from engine import analogs as analogs_mod

    assert analogs_mod.CAUSAL_CACHE_BUDGET_BYTES == 400 * 1024 * 1024
