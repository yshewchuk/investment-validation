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
    BUDGET_CALLS,
    DTE_RANGE,
    PRIORITIES,
    PullJob,
    build_plan,
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
        events["mcap_bucket"] = "<1B"  # outside the plan's slices
        patched(events)
        plan = build_plan(fetcher=FakeFetcher())
        assert plan.n_calls == 0
        assert plan.events_targeted == 0

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
