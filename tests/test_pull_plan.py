"""The Sep-1 pull plan.

This module spends money: each planned call is one of 20,000 monthly ORATS
credits. Budget arithmetic, cache-skipping, and the ordering of priorities are
therefore worth testing directly rather than discovering during a live run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data.pulls import sep2026_plan
from engine.data.pulls.sep2026_plan import (
    BATCH,
    BATCH_FOCUS,
    BATCH_T2,
    BUDGET_CALLS,
    DTE_RANGE,
    DTE_T2,
    PRIORITIES,
    PullJob,
    build_focus_plan,
    build_plan,
    build_t2_plan,
)


class FakeFetcher:
    """Reports a configurable set of requests as already cached."""

    def __init__(self, cached_keys=()):
        self.cached = set(cached_keys)
        self.asked = 0

    def has(self, source, endpoint, params=None, *, live=False):
        self.asked += 1
        return (params or {}).get("tradeDate") in self.cached


def make_events(n_per_point=4):
    """Events needing every observation point, in the target mcap buckets."""
    rows = []
    for i in range(n_per_point):
        rows.append(
            {
                "ticker": f"T{i:03d}",
                "event_date": pd.Timestamp("2024-05-08"),
                "year": 2024,
                "session": "AMC",
                "mcap_bucket": "1-10B" if i % 2 else ">10B",
                "entry_date": pd.Timestamp("2024-05-08"),
                "exit_date": pd.Timestamp("2024-05-09"),
                "runup_date": pd.Timestamp("2024-04-18"),
                "entry_both": False,
                "exit_both": False,
                "t14_both": False,
                "through_print_ready": False,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def patched(monkeypatch):
    def install(events):
        monkeypatch.setattr(
            sep2026_plan.coverage, "event_chain_coverage", lambda **kw: events
        )
        monkeypatch.setattr(sep2026_plan.coverage, "attach_mcap", lambda e, **kw: e)

    return install


class TestJobShape:
    def test_a_job_batches_tickers_into_one_call(self):
        job = PullJob("2024-05-09", ("AAA", "BBB", "CCC"), "exit")
        assert job.params["ticker"] == "AAA,BBB,CCC"
        assert job.params["tradeDate"] == "2024-05-09"

    def test_every_job_requests_the_full_dte_range(self):
        # 1-45 matches the cached chains, so new pulls normalize through the
        # same path as the existing ones.
        assert PullJob("2024-05-09", ("AAA",), "exit").params["dte"] == DTE_RANGE

    def test_both_option_sides_come_back_in_one_call(self):
        # The plan's "pull put-side chains specifically" premise does not hold:
        # /hist/strikes returns the call AND the put at every strike in one row,
        # so no budget needs to go to closing a put-side gap.
        fields = PullJob("2024-05-09", ("AAA",), "exit").params["fields"]
        for field in ("callBidPrice", "callAskPrice", "putBidPrice", "putAskPrice"):
            assert field in fields


class TestBatching:
    def test_tickers_are_grouped_into_batches_of_five(self, patched):
        # Batch size 5 is the verified-safe figure; 10 tickers on /hist/cores
        # returned 502 (payload too large for the gateway).
        events = make_events(n_per_point=12)
        patched(events)
        plan = build_plan(fetcher=FakeFetcher())
        exit_jobs = [j for j in plan.jobs if j.purpose == "exit"]
        assert len(exit_jobs) == 3  # ceil(12 / 5)
        assert all(len(j.tickers) <= BATCH for j in plan.jobs)

    def test_one_call_covers_one_trade_date(self, patched):
        patched(make_events(n_per_point=6))
        plan = build_plan(fetcher=FakeFetcher())
        for job in plan.jobs:
            assert isinstance(job.trade_date, str)
            assert len(job.trade_date) == 10


class TestPriorityOrder:
    def test_exit_chains_are_planned_first(self, patched):
        # Exit chains are the binding constraint: entry coverage is 80% in the
        # target slices and exit only 30%, and a through-print structure needs
        # both ends.
        patched(make_events(n_per_point=6))
        plan = build_plan(fetcher=FakeFetcher())
        purposes = [j.purpose for j in plan.jobs]
        assert purposes[0] == "exit"
        assert purposes.index("exit") < purposes.index("entry")

    def test_priorities_are_declared_in_the_intended_order(self):
        assert PRIORITIES == ("exit", "entry", "t14")

    def test_a_covered_point_is_not_planned(self, patched):
        events = make_events(n_per_point=4)
        events["exit_both"] = True  # nothing left to buy at the exit date
        patched(events)
        plan = build_plan(fetcher=FakeFetcher())
        assert plan.by_purpose.get("exit", 0) == 0
        assert all(j.purpose != "exit" for j in plan.jobs)


class TestBudget:
    def test_the_plan_never_exceeds_the_budget(self, patched):
        patched(make_events(n_per_point=400))
        plan = build_plan(budget=10, fetcher=FakeFetcher())
        assert plan.n_calls == 10
        assert plan.truncated_at_budget

    def test_truncation_is_reported_not_silent(self, patched):
        patched(make_events(n_per_point=400))
        plan = build_plan(budget=5, fetcher=FakeFetcher())
        assert plan.truncated_at_budget
        assert "truncated" in sep2026_plan.render_dry_run(plan)

    def test_a_plan_inside_budget_is_not_marked_truncated(self, patched):
        patched(make_events(n_per_point=4))
        plan = build_plan(budget=1000, fetcher=FakeFetcher())
        assert not plan.truncated_at_budget

    def test_the_default_budget_leaves_the_live_ops_reserve(self):
        assert BUDGET_CALLS == 16_000
        assert 20_000 - BUDGET_CALLS >= 3_000


class TestResumability:
    def test_already_cached_requests_cost_nothing(self, patched):
        # This is what makes an interrupted run resumable by re-running it.
        patched(make_events(n_per_point=4))
        plan = build_plan(fetcher=FakeFetcher(cached_keys={"2024-05-09"}))
        assert plan.skipped_cached > 0
        assert all(j.trade_date != "2024-05-09" for j in plan.jobs if j.purpose == "exit")

    def test_cached_calls_do_not_consume_budget(self, patched):
        patched(make_events(n_per_point=4))
        uncached = build_plan(budget=1000, fetcher=FakeFetcher()).n_calls
        cached = build_plan(
            budget=1000, fetcher=FakeFetcher(cached_keys={"2024-05-09"})
        ).n_calls
        assert cached < uncached

    def test_a_fully_cached_plan_is_empty(self, patched):
        patched(make_events(n_per_point=4))
        plan = build_plan(
            fetcher=FakeFetcher(cached_keys={"2024-05-08", "2024-05-09", "2024-04-18"})
        )
        assert plan.n_calls == 0


class TestScoping:
    def test_only_the_target_mcap_buckets_are_planned(self, patched):
        events = make_events(n_per_point=4)
        events["mcap_bucket"] = "unknown"  # outside the plan's slices
        patched(events)
        plan = build_plan(fetcher=FakeFetcher())
        assert plan.n_calls == 0
        assert plan.events_targeted == 0

    def test_smallcap_events_are_in_scope(self, patched):
        # The analog layer cannot score a request whose mcap bucket holds no
        # replayed trades; "<1B" held none, so the slice joined the target set.
        events = make_events(n_per_point=4)
        events["mcap_bucket"] = "<1B"
        patched(events)
        plan = build_plan(fetcher=FakeFetcher())
        assert plan.events_targeted == 4
        assert plan.n_calls > 0

    def test_events_with_an_unresolvable_date_are_skipped(self, patched):
        events = make_events(n_per_point=4)
        events["exit_date"] = pd.NaT
        patched(events)
        plan = build_plan(fetcher=FakeFetcher())
        assert all(j.purpose != "exit" for j in plan.jobs)

    def test_the_dry_run_reports_before_coverage(self, patched):
        patched(make_events(n_per_point=4))
        plan = build_plan(fetcher=FakeFetcher())
        text = sep2026_plan.render_dry_run(plan)
        assert "DRY RUN (nothing has been spent)" in text
        assert "--confirm" in text
        for point in PRIORITIES:
            assert point in text


def ev(ticker, bucket="1-10B", entry=None, exit_=None, runup=None,
       entry_both=True, exit_both=True, t14_both=True, event_date="2024-05-08"):
    """One coverage row; a None date becomes NaT and adds no needed pair."""
    return {
        "ticker": ticker,
        "event_id": f"{ticker}_{event_date}",
        "event_date": pd.Timestamp(event_date),
        "year": 2024,
        "session": "AMC",
        "mcap_bucket": bucket,
        "entry_date": pd.Timestamp(entry) if entry else pd.NaT,
        "exit_date": pd.Timestamp(exit_) if exit_ else pd.NaT,
        "runup_date": pd.Timestamp(runup) if runup else pd.NaT,
        "entry_both": entry_both,
        "exit_both": exit_both,
        "t14_both": t14_both,
        "through_print_ready": False,
    }


class TestFocusPlan:
    def test_focus_dates_carry_hitchhikers(self, patched):
        # FOC defines the date; OTH needs the same date and rides; FAR's date
        # is not a focus date, so FAR is not pulled.
        events = pd.DataFrame([
            ev("FOC", exit_="2024-05-09", exit_both=False),
            ev("OTH", exit_="2024-05-09", exit_both=False),
            ev("FAR", exit_="2024-06-01", exit_both=False),
        ])
        patched(events)
        plan = build_focus_plan(["FOC"], fetcher=FakeFetcher())
        assert plan.n_calls == 1
        assert plan.jobs[0].trade_date == "2024-05-09"
        assert set(plan.jobs[0].tickers) == {"FOC", "OTH"}

    def test_a_rider_on_a_different_purpose_still_rides(self, patched):
        # The ride set is date-based: FOC needs the date for exit, OTH for
        # entry — one call covers both.
        events = pd.DataFrame([
            ev("FOC", exit_="2024-05-09", exit_both=False),
            ev("OTH", entry="2024-05-09", entry_both=False),
        ])
        patched(events)
        plan = build_focus_plan(["FOC"], fetcher=FakeFetcher())
        assert plan.n_calls == 1
        assert set(plan.jobs[0].tickers) == {"FOC", "OTH"}

    def test_cross_purpose_pairs_dedupe_to_one_call(self, patched):
        # FOC needs the SAME date for both entry and exit; that is one pair and
        # one call, not two.
        events = pd.DataFrame([
            ev("FOC", entry="2024-05-09", exit_="2024-05-09",
               entry_both=False, exit_both=False),
        ])
        patched(events)
        plan = build_focus_plan(["FOC"], fetcher=FakeFetcher())
        assert plan.n_calls == 1
        assert plan.jobs[0].tickers == ("FOC",)

    def test_riders_batch_at_the_focus_cap(self, patched):
        # 12 tickers on one focus date batch into ceil(12/10) = 2 calls.
        rows = [ev(f"T{i:02d}", exit_="2024-05-09", exit_both=False)
                for i in range(12)]
        patched(pd.DataFrame(rows))
        plan = build_focus_plan(["T00"], fetcher=FakeFetcher())
        assert plan.n_calls == 2
        assert all(len(j.tickers) <= BATCH_FOCUS for j in plan.jobs)
        assert sum(len(j.tickers) for j in plan.jobs) == 12

    def test_partial_events_do_not_ride(self, patched):
        # PART needs the focus date for exit but ALSO 2024-06-01 for entry; the
        # event cannot complete, so not even its focus-date pair is fetched.
        events = pd.DataFrame([
            ev("FOC", exit_="2024-05-09", exit_both=False),
            ev("PART", exit_="2024-05-09", entry="2024-06-01",
               exit_both=False, entry_both=False),
        ])
        patched(events)
        plan = build_focus_plan(["FOC"], fetcher=FakeFetcher())
        assert plan.n_calls == 1
        assert plan.jobs[0].tickers == ("FOC",)

    def test_focus_budget_truncates(self, patched):
        # Three focus events on three dates -> three calls, truncated to two.
        rows = [ev("FOC", exit_=d, exit_both=False, event_date=d)
                for d in ("2024-05-09", "2024-05-10", "2024-05-11")]
        patched(pd.DataFrame(rows))
        plan = build_focus_plan(["FOC"], budget=2, fetcher=FakeFetcher())
        assert plan.n_calls == 2
        assert plan.truncated_at_budget


class TestDecisionPlan:
    """The T−2 pull: buy the close the trade would be DECIDED on.

    Costs real money — 3,628 calls at the measured 2026-09-02 universe — so the
    universe rule, the DTE widening and the batch arithmetic are pinned here
    rather than discovered against the live quota.
    """

    def _events(self):
        """Four replayable events plus one that is not, all needing a d1 chain."""
        rows = []
        for i in range(4):
            rows.append(
                {
                    "event_id": f"T{i:03d}_2024-05-08",
                    "ticker": f"T{i:03d}",
                    "event_date": pd.Timestamp("2024-05-08"),
                    "year": 2024,
                    "session": "AMC",
                    "mcap_bucket": "1-10B" if i % 2 else ">10B",
                    "entry_date": pd.Timestamp("2024-05-08"),
                    "exit_date": pd.Timestamp("2024-05-09"),
                    "runup_date": pd.Timestamp("2024-04-18"),
                    "d1_date": pd.Timestamp("2024-05-07"),
                    "d2_date": pd.Timestamp("2024-05-06"),
                    "entry_both": True,
                    "exit_both": True,
                    "t14_both": False,
                    "d1_both": False,
                    "d2_both": False,
                    "through_print_ready": True,
                }
            )
        # Not replayable: no exit chain, so a decision chain buys nothing.
        rows.append(
            {
                **rows[0],
                "event_id": "NOPE_2024-05-08",
                "ticker": "NOPE",
                "exit_both": False,
                "through_print_ready": False,
            }
        )
        return pd.DataFrame(rows)

    def test_only_replayable_events_are_bought_for(self, patched):
        patched(self._events())
        plan = build_t2_plan(-1, fetcher=FakeFetcher())
        bought = {t for job in plan.jobs for t in job.tickers}
        assert "NOPE" not in bought
        assert plan.events_targeted == 4

    def test_the_decision_pull_asks_for_one_more_day_of_dte(self, patched):
        # The traded expiry is a day further out seen from a day earlier, so a
        # 1,45 window would drop every event whose expiry sat at the ceiling.
        patched(self._events())
        plan = build_t2_plan(-1, fetcher=FakeFetcher())
        assert all(job.params["dte"] == DTE_T2 for job in plan.jobs)
        assert DTE_T2 != DTE_RANGE

    def test_it_buys_the_decision_date_not_the_entry_date(self, patched):
        patched(self._events())
        plan = build_t2_plan(-1, fetcher=FakeFetcher())
        assert {job.trade_date for job in plan.jobs} == {"2024-05-07"}

    def test_four_tickers_on_one_date_are_one_call(self, patched):
        patched(self._events())
        plan = build_t2_plan(-1, fetcher=FakeFetcher())
        assert plan.n_calls == 1  # 4 tickers, batch of 10
        assert plan.jobs[0].tickers == ("T000", "T001", "T002", "T003")

    def test_batches_are_composed_in_a_stable_order(self, patched):
        """The Fetcher caches on exact request params, so a batch built in a
        different order is a fresh call rather than a cache hit."""
        patched(self._events())
        first = build_t2_plan(-1, fetcher=FakeFetcher())
        patched(self._events())
        second = build_t2_plan(-1, fetcher=FakeFetcher())
        assert [j.params for j in first.jobs] == [j.params for j in second.jobs]

    def test_an_event_that_already_has_the_chain_is_not_re_bought(self, patched):
        events = self._events()
        events.loc[events["ticker"] == "T000", "d1_both"] = True
        patched(events)
        plan = build_t2_plan(-1, fetcher=FakeFetcher())
        assert "T000" not in {t for job in plan.jobs for t in job.tickers}
        assert plan.context["per_offset"]["d1"]["already_have_chain"] == 1

    def test_a_cached_request_costs_nothing(self, patched):
        patched(self._events())
        plan = build_t2_plan(-1, fetcher=FakeFetcher(cached_keys={"2024-05-07"}))
        assert plan.n_calls == 0
        assert plan.skipped_cached == 1

    def test_the_budget_truncates_rather_than_overspends(self, patched):
        patched(self._events())
        plan = build_t2_plan(-1, fetcher=FakeFetcher(), budget=0, batch=1)
        assert plan.n_calls == 0
        assert plan.truncated_at_budget

    def test_two_arms_share_batches_instead_of_paying_twice(self, patched):
        """The reason to plan both arms together: one call buys one date for up
        to ten tickers, and the arms' dates only partly overlap."""
        patched(self._events())
        both = build_t2_plan((-1, -2), fetcher=FakeFetcher())
        assert {job.trade_date for job in both.jobs} == {"2024-05-07", "2024-05-06"}
        assert both.n_calls == 2
        assert both.context["per_offset"]["d1"]["calls_alone"] == 1
        assert both.context["per_offset"]["d2"]["calls_alone"] == 1

    def test_the_plan_reports_what_staging_would_cost(self, patched):
        patched(self._events())
        both = build_t2_plan((-1, -2), fetcher=FakeFetcher())
        assert both.context["calls_if_staged"] >= both.n_calls

    def test_no_offsets_is_refused_rather_than_silently_planning_nothing(self):
        with pytest.raises(ValueError, match="at least one decision offset"):
            build_t2_plan((), fetcher=FakeFetcher())

    def test_call_counting_batches_per_date_not_across_dates(self):
        # Eleven pairs spread one-per-date is eleven calls, not two.
        spread = {(f"T{i}", f"2024-05-{i+1:02d}") for i in range(11)}
        assert sep2026_plan._batched_call_count(spread, 10) == 11
        stacked = {(f"T{i}", "2024-05-07") for i in range(11)}
        assert sep2026_plan._batched_call_count(stacked, 10) == 2

    def test_the_dry_run_names_the_coverage_gate_and_the_quota(self, patched):
        patched(self._events())
        plan = build_t2_plan(-1, fetcher=FakeFetcher())
        text = sep2026_plan.render_t2_dry_run(plan)
        assert "nothing has been spent" in text
        assert "80%" in text
        assert "--confirm" in text
