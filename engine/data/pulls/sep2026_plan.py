#!/usr/bin/env python3
"""The Sep-1 ORATS pull: plan it, cost it, then spend deliberately.

    python3 -m engine.data.pulls.sep2026_plan --dry-run          # always first
    python3 -m engine.data.pulls.sep2026_plan --confirm          # spends quota

    # the T-2 decision chains (guides/str_thru_t2_decision.md step 3/4):
    python3 -m engine.data.pulls.sep2026_plan --t2 --dry-run              # arm A
    python3 -m engine.data.pulls.sep2026_plan --t2 --dry-run \
        --decision-offset -1 -2                                           # both arms
    python3 -m engine.data.pulls.sep2026_plan --t2 --confirm

**What the audit says to buy.** Coverage is not uniformly thin — it is thin in
one specific place. Entry-date chains already cover 48.7% of 2017+ events
(79–95% inside the target mcap slices), but *exit*-date chains cover only 18.0%.
Since a through-the-print structure needs both ends, exit chains are the binding
constraint and get first claim on the budget. (The one slice thinner than exit
coverage is `<1B`, near zero everywhere; it joined the target set 2026-09-01
because the analog layer cannot score a request whose bucket holds no trades.)

**One assumption in the plan needed correcting.** The plan calls for pulling
"put-side chains specifically", on the grounds that the cache was pulled
straddle-centric. It was not, in the way that matters: ORATS ``/hist/strikes``
returns the call *and* the put at every strike in the same row, so call and put
coverage are identical (0.487 / 0.487 at entry) and always will be. There is no
put-side gap to close, and no call needs to be spent closing one.

**Budgeting.** One call fetches one trade date for up to ``BATCH`` tickers, so
cost is driven by (dates × ticker-batches), not by event count. The budget is
16,000 calls with a 3,000-call live-operations reserve untouched; the quota
guard in :mod:`engine.data.throttle` enforces the floor independently.

Resumability comes free from Tier 1: a request already in the raw cache is a
cache hit that costs nothing, so an interrupted run resumes by re-running.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.data import coverage, store
from engine.data.fetch import Fetcher, iter_cached
from engine.data.throttle import QuotaExhausted, latest_quota

__all__ = ["PullJob", "PullPlan", "build_plan", "build_focus_plan", "build_t2_plan",
           "execute", "render_dry_run", "render_t2_dry_run",
           "BATCH", "BATCH_FOCUS", "BATCH_T2", "BUDGET_CALLS", "DTE_RANGE", "DTE_T2",
           "FIELDS", "LIQUIDITY_FIELDS"]

#: Tickers per ``/hist/strikes`` call. Five is the verified-safe batch: ten
#: tickers on ``/hist/cores`` returned 502 (payload too large for the gateway),
#: and the existing strikes puller settled on five.
BATCH = 5

#: Focus pulls batch at ten — the documented ticker cap of ``/hist/strikes``,
#: verified in production by the nightly forward-chain pull, which runs at ten
#: and diffs requested against returned. A focus plan is mostly small-cap dates
#: (~300 rows per call median), far under any row cap, so ten is safe and halves
#: the call count. :func:`execute` still verifies every response.
BATCH_FOCUS = 10

#: Total calls this plan may spend, against the 20,000/month allowance.
BUDGET_CALLS = 16_000

#: Fields requested per strike.
#:
#: The first fifteen are the slim set every cached chain already carries, so new
#: pulls normalize through the same path as the old ones. The eight after them
#: are **liquidity**, and they are the reason this constant is worth reading
#: twice: ORATS bills per CALL, not per field, so open interest, volume and the
#: size resting at the touch cost exactly nothing to add — while the existing
#: 19,061-file cache has none of them, because this list did not ask.
#:
#: That matters more than it sounds. Every headline in this program assumes a
#: mid fill, the ungated STR-THRU edge lives entirely in the widest-quoted
#: names (reports/exp-105_log_diagnostics.md), and nothing in the store can say
#: whether a single contract had size at the touch. These eight fields are the
#: first real evidence for or against the assumption the whole program rests
#: on, and a pull that omits them cannot be cheaply repeated: back-filling them
#: costs the same 16,000 calls again.
FIELDS = (
    "ticker,tradeDate,expirDate,dte,strike,stockPrice,callBidPrice,callAskPrice,"
    "putBidPrice,putAskPrice,callMidIv,putMidIv,smvVol,delta,spotPrice,"
    "callVolume,callOpenInterest,callBidSize,callAskSize,"
    "putVolume,putOpenInterest,putBidSize,putAskSize"
)

#: The liquidity fields, split out so the normalizer and the tests agree on one
#: list. Confirmed against the ORATS datav2 field reference, not assumed.
LIQUIDITY_FIELDS = (
    "callVolume", "callOpenInterest", "callBidSize", "callAskSize",
    "putVolume", "putOpenInterest", "putBidSize", "putAskSize",
)

DTE_RANGE = "1,45"

#: The decision pull asks for one more day of DTE than the entry pull does.
#: The expiry a through-the-print structure trades is the first one that
#: survives the print; seen from one session earlier that same expiry is one
#: day further out, so a `1,45` window silently drops every event whose traded
#: expiry sat at the ceiling. Widening by one costs no extra calls — ORATS
#: bills per call, not per row.
DTE_T2 = "1,46"

#: Tickers per call for the decision pull. Ten is the documented `/hist/strikes`
#: cap, run in production every night by the forward-chain pull, and
#: :func:`execute` verifies every response and single-ticker retries a
#: suspect-looking shortfall. Halving the call count halves the cost.
BATCH_T2 = 10

#: Priority order. Exit chains first: they are the binding constraint on every
#: through-the-print structure, and entry coverage is already good.
PRIORITIES = ("exit", "entry", "t14")

#: Every mcap slice, not just the liquid two: the board's analog layer buckets
#: trades on the SAME edges (engine.analogs.MCAP_EDGES), and a slice with zero
#: replayed trades returns an empty match for every request in it. That was the
#: state of "<1B" — 20k panel events, one chain on an event date — which left
#: every sub-$1B name on the forward board unscored. The slice is pulled last
#: because the plan walks events in bucket order and the budget may run out.
TARGET_BUCKETS = ("1-10B", ">10B", "<1B")


@dataclass
class PullJob:
    trade_date: str
    tickers: tuple[str, ...]
    purpose: str  # exit | entry | t14 | focus | decision
    events_unlocked: int = 0
    #: Per-job, because the decision pull needs a day more (:data:`DTE_T2`).
    #: The cache key is the exact request params, so this is also what makes a
    #: `1,46` job a different call from a `1,45` one rather than a cache hit.
    dte: str = DTE_RANGE

    @property
    def params(self) -> dict:
        return {
            "ticker": ",".join(self.tickers),
            "tradeDate": self.trade_date,
            "dte": self.dte,
            "fields": FIELDS,
        }


@dataclass
class PullPlan:
    jobs: list[PullJob] = field(default_factory=list)
    skipped_cached: int = 0
    by_purpose: dict = field(default_factory=dict)
    coverage_before: dict = field(default_factory=dict)
    events_targeted: int = 0
    truncated_at_budget: bool = False
    #: Free-form, plan-specific numbers a renderer wants. Kept out of the
    #: fields above so the two existing plans serialize exactly as they did.
    context: dict = field(default_factory=dict)

    @property
    def n_calls(self) -> int:
        return len(self.jobs)

    def summary(self) -> dict:
        out = {
            "calls": self.n_calls,
            "budget": BUDGET_CALLS,
            "skipped_already_cached": self.skipped_cached,
            "by_purpose": self.by_purpose,
            "events_targeted": self.events_targeted,
            "coverage_before": self.coverage_before,
            "truncated_at_budget": self.truncated_at_budget,
        }
        if self.context:
            out["context"] = self.context
        return out


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def build_plan(
    min_year: int = 2017,
    buckets: tuple[str, ...] = TARGET_BUCKETS,
    budget: int = BUDGET_CALLS,
    fetcher: Fetcher | None = None,
) -> PullPlan:
    """Enumerate the calls that would close the coverage gaps, in priority order."""
    print("planning: computing current coverage …", flush=True)
    events = coverage.attach_mcap(coverage.event_chain_coverage(min_year=min_year))
    target = events[events["mcap_bucket"].isin(buckets)].copy()

    plan = PullPlan()
    plan.events_targeted = int(len(target))
    plan.coverage_before = {
        point: round(float(target[f"{point}_both"].mean()), 4) if len(target) else 0.0
        for point in PRIORITIES
    }
    print(
        f"  {len(target):,} events in {buckets}; coverage "
        + ", ".join(f"{k} {v:.1%}" for k, v in plan.coverage_before.items()),
        flush=True,
    )

    fetcher = fetcher or Fetcher()
    date_column = {"exit": "exit_date", "entry": "entry_date", "t14": "runup_date"}

    for purpose in PRIORITIES:
        missing = target[~target[f"{purpose}_both"]]
        column = date_column[purpose]
        missing = missing[missing[column].notna()]
        if missing.empty:
            plan.by_purpose[purpose] = 0
            continue

        needed = (
            missing.groupby(missing[column].dt.strftime("%Y-%m-%d"))["ticker"]
            .apply(lambda s: sorted(set(s)))
            .to_dict()
        )
        added = 0
        for trade_date in sorted(needed):
            tickers = needed[trade_date]
            for start in range(0, len(tickers), BATCH):
                batch = tuple(tickers[start : start + BATCH])
                job = PullJob(trade_date=trade_date, tickers=batch, purpose=purpose,
                              events_unlocked=len(batch))
                # A request already in Tier 1 costs nothing and must not be
                # counted against the budget — this is also what makes an
                # interrupted run resumable by simply re-running.
                if fetcher.has("orats", "hist/strikes", job.params):
                    plan.skipped_cached += 1
                    continue
                if len(plan.jobs) >= budget:
                    plan.truncated_at_budget = True
                    break
                plan.jobs.append(job)
                added += 1
            if plan.truncated_at_budget:
                break
        plan.by_purpose[purpose] = added
        print(f"  {purpose}: {added:,} calls planned", flush=True)
        if plan.truncated_at_budget:
            print(f"  budget of {budget:,} reached — plan truncated", flush=True)
            break
    return plan


def build_focus_plan(
    focus: Sequence[str],
    *,
    min_year: int = 2017,
    buckets: tuple[str, ...] = TARGET_BUCKETS,
    budget: int = BUDGET_CALLS,
    fetcher: Fetcher | None = None,
    batch: int = BATCH_FOCUS,
) -> PullPlan:
    """Focus tickers define the dates; every ticker that needs those dates rides.

    The economy this exploits: one call buys one trade DATE for up to ``batch``
    tickers, so once a date is being fetched for a focus name, any other name
    that needs the same date costs nothing until the batch fills. Focus events
    are few and their dates scatter across years; the events of the wider
    universe that happen to share those dates are many. Measured 2026-09-01 for
    the five-name focus: 180 focus dates / 192 focus pairs, and 2,058 extra
    events across 1,341 tickers hitchhike on them — 517 calls all-in against
    14,882 for the full ``<1B`` slice.

    Pairs are deduplicated ACROSS purposes — a (ticker, date) needed for both an
    exit and a t14 is one call, not two. The ride set is only the events that
    COMPLETE on the focus dates — every one of their missing pairs falls on a
    date already being bought. Partial events are refused: a replay needs the
    entry AND the exit chain, so a fetched pair whose event stays incomplete is
    quota spent on a trade that still cannot be priced. That distinction is the
    difference between 517 calls and 1,390 for the five-name focus.
    """
    print(f"planning (focus): {len(focus)} names, computing coverage …", flush=True)
    events = coverage.attach_mcap(coverage.event_chain_coverage(min_year=min_year))
    target = events[events["mcap_bucket"].isin(buckets)].copy()
    focus_set = {str(t).upper() for t in focus}
    foc = target[target["ticker"].isin(focus_set)]

    plan = PullPlan()
    plan.events_targeted = int(len(foc))
    plan.coverage_before = {
        point: round(float(foc[f"{point}_both"].mean()), 4) if len(foc) else 0.0
        for point in PRIORITIES
    }

    def needed_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for purpose in PRIORITIES:
            column = {"exit": "exit_date", "entry": "entry_date", "t14": "runup_date"}[purpose]
            missing = frame[~frame[f"{purpose}_both"] & frame[column].notna()]
            pairs |= {
                (str(t), str(d)[:10])
                for t, d in zip(missing["ticker"], missing[column])
            }
        return pairs

    focus_pairs = needed_pairs(foc)
    focus_dates = {d for _, d in focus_pairs}
    ride_pairs: set[tuple[str, str]] = set()
    complete_events = 0
    for _, event_rows in target.groupby("event_id"):
        pairs = needed_pairs(event_rows)
        if pairs and all(d in focus_dates for _, d in pairs):
            ride_pairs |= pairs
            complete_events += 1
    ride_pairs -= focus_pairs
    print(
        f"  focus: {len(foc)} events, {len(focus_pairs)} pairs on {len(focus_dates)} dates; "
        f"hitchhikers: {len(ride_pairs)} pairs from {complete_events:,} events "
        f"that complete on those dates",
        flush=True,
    )

    by_date: dict[str, set[str]] = {}
    for t, d in focus_pairs | ride_pairs:
        by_date.setdefault(d, set()).add(t)

    fetcher = fetcher or Fetcher()
    added = 0
    for trade_date in sorted(by_date):
        tickers = sorted(by_date[trade_date])
        for start in range(0, len(tickers), batch):
            chunk = tuple(tickers[start : start + batch])
            job = PullJob(trade_date=trade_date, tickers=chunk, purpose="focus",
                          events_unlocked=len(chunk))
            if fetcher.has("orats", "hist/strikes", job.params):
                plan.skipped_cached += 1
                continue
            if len(plan.jobs) >= budget:
                plan.truncated_at_budget = True
                break
            plan.jobs.append(job)
            added += 1
        if plan.truncated_at_budget:
            print(f"  budget of {budget:,} reached — plan truncated", flush=True)
            break
    plan.by_purpose["focus"] = added
    print(f"  focus: {added:,} calls planned", flush=True)
    return plan


def build_t2_plan(
    decision_offsets: Sequence[int] | int = -1,
    *,
    min_year: int = 2018,
    buckets: tuple[str, ...] | None = None,
    budget: int = BUDGET_CALLS,
    fetcher: Fetcher | None = None,
    batch: int = BATCH_T2,
    already_requested: set[tuple[str, str]] | None = None,
) -> PullPlan:
    """The decision-date chains a T−2 STR-THRU book cannot be backtested without.

    Different question from :func:`build_plan`, which buys the chains that make
    an event *replayable at all*. This one starts from the events that already
    are — entry and exit chains both present — and buys the one chain that says
    what the board would have quoted a session earlier. An event without a
    decision chain is not a thinner backtest; it is an event the T−2 variant
    cannot be measured on, so nothing is gained by buying a decision chain for
    an event that is not replayable in the first place.

    Two deviations from the entry pull, both deliberate:

    - ``dte=1,46`` (:data:`DTE_T2`) — the traded expiry is a day further out
      seen from a day earlier.
    - ``batch=10`` (:data:`BATCH_T2`) — the documented cap, halving the cost.

    Several offsets can be planned **together**, and doing so is cheaper than
    planning them one after another: a call buys one trade date for up to
    ``batch`` tickers, and the two arms' dates overlap heavily, so their pairs
    share batches instead of each paying for a half-empty one. Staging is still
    a legitimate choice — it just is not free, and the dry run prices both.

    ``buckets=None`` means every market-cap slice, ``unknown`` included. That is
    the right default here and the wrong one for :func:`build_plan`: the entry
    pull is closing a coverage gap that is measured per slice, while this one is
    buying a decision date for a fixed, already-replayable event set, and
    dropping the events whose panel row has no market cap would restrict the
    retrain universe for a reason that has nothing to do with the decision date.

    **Nothing already held is re-bought.** Three guards, in order of how much
    they can see:

    1. *Coverage.* A pair whose chain is in the store is not in the missing set
       at all. This is content-based, so it holds however the request that
       fetched it was composed.
    2. *Request history* (:func:`requested_pairs`, overridable via
       ``already_requested``). A pair asked for before and returned empty is
       never re-asked. Per-pair, so it survives batch regrouping.
    3. *``Fetcher.has``.* An identical request is a free cache hit, which is
       what makes an interrupted run resumable by re-running it.

    The dry run prints what each one skipped, so "we already have this" is a
    number you can read rather than a claim you have to trust.
    """
    if isinstance(decision_offsets, int):
        decision_offsets = (decision_offsets,)
    offsets = tuple(int(o) for o in decision_offsets)
    if not offsets:
        raise ValueError("build_t2_plan needs at least one decision offset")
    labels = [coverage.decision_label(o) for o in offsets]
    shown = ", ".join(f"T{o:+d}" for o in offsets)
    print(f"planning ({shown} decision chains): computing coverage …", flush=True)

    events = coverage.attach_mcap(
        coverage.event_chain_coverage(min_year=min_year, decision_offsets=offsets)
    )
    if buckets is not None:
        events = events[events["mcap_bucket"].isin(buckets)].copy()

    # The universe is what can actually be replayed end to end today. Anything
    # else is a decision chain bought for a trade that still cannot be priced.
    replayable = events[events["through_print_ready"]].copy()

    plan = PullPlan()
    plan.events_targeted = int(len(replayable))
    plan.coverage_before = {
        point: round(float(replayable[f"{point}_both"].mean()), 4) if len(replayable) else 0.0
        for point in ("entry", "exit", *labels)
    }

    if already_requested is None:
        already_requested = requested_pairs()
        print(
            f"  {len(already_requested):,} (ticker, date) pairs have ever been "
            "requested on hist/strikes",
            flush=True,
        )

    pairs: set[tuple[str, str]] = set()
    per_offset: dict[str, dict] = {}
    for offset, label in zip(offsets, labels):
        column = f"{label}_date"
        have = replayable[f"{label}_both"]
        resolvable = replayable[column].notna()
        missing = replayable[~have & resolvable]
        arm_pairs = {
            (str(t), str(d)[:10]) for t, d in zip(missing["ticker"], missing[column])
        }
        # One-sided data would mean the pair IS in the store while `_both` says
        # it is not. ORATS returns the call and the put in the same row so this
        # should always be zero; it is reported rather than assumed, because a
        # non-zero here is quota about to be spent on rows we hold.
        partial = int((~have & resolvable & replayable[f"{label}_any"]).sum())
        # Asked for before and still not in the store: ORATS has nothing there,
        # and asking again buys nothing.
        empty_before = arm_pairs & already_requested
        arm_pairs -= empty_before
        pairs |= arm_pairs
        per_offset[label] = {
            "offset": int(offset),
            "already_have_chain": int(have.sum()),
            "partial_side_in_store": partial,
            "skipped_requested_but_empty": len(empty_before),
            "unresolvable_date": int((~resolvable).sum()),
            "events_needing_a_pull": int(len(missing)),
            "new_pairs": len(arm_pairs),
            "distinct_dates": len({d for _, d in arm_pairs}),
            "calls_alone": _batched_call_count(arm_pairs, batch),
            "by_year": {
                int(y): int(n)
                for y, n in missing["event_date"].dt.year.value_counts().sort_index().items()
            },
            "by_bucket": {
                str(k): int(v) for k, v in missing["mcap_bucket"].value_counts().items()
            },
        }
        print(
            f"  {label}: {len(missing):,} events to buy, {len(arm_pairs):,} pairs, "
            f"{per_offset[label]['calls_alone']:,} calls if pulled alone",
            flush=True,
        )

    by_date: dict[str, set[str]] = {}
    for ticker, trade_date in pairs:
        by_date.setdefault(trade_date, set()).add(ticker)

    plan.context = {
        "decision_offsets": list(offsets),
        "labels": labels,
        "min_year": int(min_year),
        "dte": DTE_T2,
        "batch": int(batch),
        "replayable_events": int(len(replayable)),
        "new_pairs": len(pairs),
        "distinct_dates": len(by_date),
        "calls_if_staged": sum(v["calls_alone"] for v in per_offset.values()),
        "pairs_ever_requested": len(already_requested),
        "skipped_requested_but_empty": sum(
            v["skipped_requested_but_empty"] for v in per_offset.values()
        ),
        "partial_side_in_store": sum(v["partial_side_in_store"] for v in per_offset.values()),
        "per_offset": per_offset,
    }
    print(
        f"  {len(replayable):,} replayable events; {len(pairs):,} (ticker, date) "
        f"pairs to buy on {len(by_date):,} dates",
        flush=True,
    )

    fetcher = fetcher or Fetcher()
    added = 0
    for trade_date in sorted(by_date):
        # Sorted, so a re-run composes the same batches. The Fetcher caches on
        # exact request params, and a batch built in a different order is a
        # fresh call rather than a cache hit — batch order is money here.
        tickers = sorted(by_date[trade_date])
        for start in range(0, len(tickers), batch):
            chunk = tuple(tickers[start : start + batch])
            job = PullJob(
                trade_date=trade_date, tickers=chunk, purpose="decision",
                events_unlocked=len(chunk), dte=DTE_T2,
            )
            if fetcher.has("orats", "hist/strikes", job.params):
                plan.skipped_cached += 1
                continue
            if len(plan.jobs) >= budget:
                plan.truncated_at_budget = True
                break
            plan.jobs.append(job)
            added += 1
        if plan.truncated_at_budget:
            print(f"  budget of {budget:,} reached — plan truncated", flush=True)
            break
    plan.by_purpose["decision"] = added
    print(f"  decision: {added:,} calls planned", flush=True)
    return plan


def requested_pairs() -> set[tuple[str, str]]:
    """Every ``(ticker, tradeDate)`` ever ASKED FOR on ``hist/strikes``.

    Not the same set as what the chain store holds, and the difference is the
    point: a pair that was requested and came back empty is absent from the
    store forever, so a coverage-driven plan will keep proposing to buy it on
    every future run. Measured 2026-09-02 that is 52 pairs of 104,050 (0.05%) —
    small, but it is the one re-pull the other two guards cannot see.

    ``Fetcher.has`` cannot stand in for this. Its cache key is a hash of the
    exact request params, so it only recognises a call whose ticker batch was
    composed identically. After a partial pull is ingested the missing set
    shrinks, the batches recompose, and those hits are lost — while this set,
    being per-pair, survives any regrouping.

    Cheap: reads the Tier-1 sidecars, not the bodies (~1s for 21,706 of them).
    """
    pairs: set[tuple[str, str]] = set()
    for entry in iter_cached("orats", "hist/strikes"):
        params = entry.params
        trade_date = str(params.get("tradeDate", ""))[:10]
        if not trade_date:
            continue
        for ticker in str(params.get("ticker", "")).split(","):
            ticker = ticker.strip()
            if ticker:
                pairs.add((ticker, trade_date))
    return pairs


def _batched_call_count(pairs: set[tuple[str, str]], batch: int) -> int:
    """Calls one set of (ticker, date) pairs would cost on its own.

    Not `len(pairs) / batch`: a call buys ONE date, so the batching is per date
    and every date pays for its own remainder.
    """
    by_date: dict[str, int] = {}
    for _, trade_date in pairs:
        by_date[trade_date] = by_date.get(trade_date, 0) + 1
    return sum(-(-n // batch) for n in by_date.values())


def render_t2_dry_run(plan: PullPlan) -> str:
    """The decision-pull dry run, with the numbers the go/no-go actually needs."""
    ctx = plan.context
    labels = ctx.get("labels", [])
    per_offset = ctx.get("per_offset", {})
    quota = latest_quota()
    remaining = quota.get("remaining")
    floor = quota.get("floor", 0) or 0
    spendable = (remaining - floor) if remaining is not None else None
    replayable = ctx.get("replayable_events", 0)
    arms = ", ".join(f"T{o:+d}" for o in ctx.get("decision_offsets", []))

    lines = [
        "",
        "=" * 68,
        f"DECISION-CHAIN PULL ({arms}) — DRY RUN (nothing has been spent)",
        "=" * 68,
        f"  decision closes      : {arms} sessions relative to the entry close",
        f"  dte window           : {ctx.get('dte')}  (one wider than the entry pull)",
        f"  tickers per call     : {ctx.get('batch')}",
        f"  events from          : {ctx.get('min_year')}",
        "",
        "  universe (events replayable end-to-end TODAY — entry and exit both present):",
        f"    replayable events  : {replayable:,}",
    ]
    for label in labels:
        arm = per_offset.get(label, {})
        lines += [
            f"    {label}: have {arm.get('already_have_chain', 0):,}, "
            f"unresolvable {arm.get('unresolvable_date', 0):,}, "
            f"to buy {arm.get('events_needing_a_pull', 0):,} events "
            f"({arm.get('new_pairs', 0):,} pairs on {arm.get('distinct_dates', 0):,} dates)",
        ]
    lines += [
        "",
        "  coverage now, over those replayable events:",
        *[f"    {k:<6s} {v:.1%}" for k, v in plan.coverage_before.items()],
        "",
        "  nothing already held is re-bought — what each guard skipped:",
        f"    1. chain already in the store   : "
        f"{sum(v.get('already_have_chain', 0) for v in per_offset.values()):,} events",
        f"       (one-sided, would slip guard 1: "
        f"{ctx.get('partial_side_in_store', 0):,} — expect 0)",
        f"    2. asked before, came back empty: "
        f"{ctx.get('skipped_requested_but_empty', 0):,} pairs "
        f"(of {ctx.get('pairs_ever_requested', 0):,} ever requested)",
        f"    3. identical request cached     : {plan.skipped_cached:,} calls",
        "",
        "  cost:",
        f"    NEW (ticker,date)  : {ctx.get('new_pairs', 0):,} pairs "
        "— none of these are in the store",
        f"    distinct dates     : {ctx.get('distinct_dates', 0):,}",
        f"    planned calls      : {plan.n_calls:,}",
    ]
    staged = ctx.get("calls_if_staged")
    if staged is not None and len(labels) > 1:
        lines += [
            f"    if pulled one arm at a time: {staged:,} "
            f"(+{staged - plan.n_calls:,} — the arms share dates, "
            "so staging pays for half-empty batches twice)",
        ]
    lines += [
        "",
        "  quota:",
        f"    remaining          : "
        + (f"{remaining:,}" if remaining is not None else "unknown")
        + (f"   (as of {quota.get('ts')})" if quota.get("ts") else ""),
        f"    live-ops floor     : {floor:,}",
        "    spendable          : "
        + (f"{spendable:,}" if spendable is not None else "unknown"),
    ]
    if spendable is not None and plan.n_calls:
        share = 100.0 * plan.n_calls / spendable if spendable > 0 else float("inf")
        verdict = "FITS" if plan.n_calls <= spendable else "DOES NOT FIT"
        lines.append(f"    this plan          : {share:.0f}% of spendable — {verdict}")

    for label in labels:
        by_year = (per_offset.get(label) or {}).get("by_year") or {}
        if not by_year:
            continue
        total = sum(by_year.values())
        lines += ["", f"  {label}: events still to buy, by year (staging guide):"]
        running = 0
        for year in sorted(by_year, reverse=True):
            running += by_year[year]
            lines.append(
                f"    {year}  {by_year[year]:>6,}   "
                f"({running:>6,} cumulative, {100*running/total:.0f}% from {year} on)"
            )
        by_bucket = (per_offset.get(label) or {}).get("by_bucket") or {}
        if by_bucket:
            lines += [f"  {label}: by mcap bucket:"]
            lines += [f"    {k:<8s} {v:>6,}" for k, v in sorted(by_bucket.items())]

    if plan.jobs:
        lines += [
            "",
            "  first 5 calls:",
            *[
                f"    {j.trade_date}  {j.purpose:<8s} dte={j.dte}  {','.join(j.tickers)}"
                for j in plan.jobs[:5]
            ],
        ]
    if plan.truncated_at_budget:
        lines += ["", "  NOTE: plan truncated at the budget; re-run for the rest."]
    lines += [
        "",
        f"  Coverage gate after ingest: at least 80% of the {replayable:,} replayable",
        "  events must come back with a usable decision chain. Below that, report the",
        "  shortfall by year and mcap bucket and STOP — do not retrain on whatever",
        "  survived (guides/str_thru_t2_decision.md §3).",
        "",
        "  To spend: re-run with --confirm",
        "=" * 68,
        "",
    ]
    return "\n".join(lines)


def render_dry_run(plan: PullPlan) -> str:
    lines = [
        "",
        "=" * 68,
        "SEP-2026 ORATS PULL — DRY RUN (nothing has been spent)",
        "=" * 68,
        f"  planned calls        : {plan.n_calls:,}",
        f"  budget               : {BUDGET_CALLS:,}",
        f"  live-ops reserve     : 3,000 (untouched; guard enforces the floor)",
        f"  already cached, free : {plan.skipped_cached:,}",
        f"  events in scope      : {plan.events_targeted:,}",
        "",
        "  coverage now (target mcap slices, both sides):",
        *[f"    {k:<6s} {v:.1%}" for k, v in plan.coverage_before.items()],
        "",
        "  calls by purpose (priority order — exit is the binding gap):",
        *[f"    {k:<6s} {v:,}" for k, v in plan.by_purpose.items()],
        "",
    ]
    if plan.jobs:
        by_date: dict[str, int] = {}
        for job in plan.jobs:
            by_date[job.trade_date] = by_date.get(job.trade_date, 0) + 1
        dates = sorted(by_date)
        lines += [
            f"  distinct trade dates : {len(dates):,} "
            f"({dates[0]} → {dates[-1]})",
            f"  calls per date       : min {min(by_date.values())}, "
            f"median {int(np.median(list(by_date.values())))}, max {max(by_date.values())}",
            "",
            "  first 5 calls:",
            *[
                f"    {j.trade_date}  {j.purpose:<5s}  {','.join(j.tickers)}"
                for j in plan.jobs[:5]
            ],
        ]
    if plan.truncated_at_budget:
        lines += ["", "  NOTE: plan truncated at the budget; re-run next cycle for the rest."]
    lines += [
        "",
        "  To spend: re-run with --confirm",
        "=" * 68,
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def execute(plan: PullPlan, fetcher: Fetcher | None = None, log_every: float = 30.0) -> dict:
    """Spend the plan. One process, cache-first, quota-guarded, resumable.

    Every fresh response is verified against what was asked for. A shortfall is
    only EVER retried single-ticker when truncation is actually suspect — a full
    ``batch`` of ten (the documented cap) or a payload near a row cap. Below
    that, a missing ticker is a genuine absence (a name with no chain that day)
    and re-asking would spend calls the way the 404-retry bug used to.
    """
    fetcher = fetcher or Fetcher()
    started = time.time()
    last_log = 0.0
    done = cached = failed = absent = truncation_retries = 0
    errors: list[str] = []

    for i, job in enumerate(plan.jobs, 1):
        try:
            record = fetcher.fetch(
                "orats", "hist/strikes", job.params, note=f"sep2026:{job.purpose}"
            )
            if record.from_cache:
                cached += 1
            else:
                done += 1
                try:
                    rows = record.json().get("data") or []
                except (ValueError, AttributeError):
                    rows = []
                got = {r.get("ticker") for r in rows}
                missing = [t for t in job.tickers if t not in got]
                suspect = len(job.tickers) >= 10 or len(rows) >= 4_500
                if missing and suspect:
                    # Distinguish truncation from absence: one single each.
                    for t in missing:
                        single = dict(job.params)
                        single["ticker"] = t
                        if fetcher.has("orats", "hist/strikes", single):
                            continue
                        truncation_retries += 1
                        rec2 = fetcher.fetch(
                            "orats", "hist/strikes", single,
                            note=f"sep2026:{job.purpose}:verify",
                        )
                        rows2 = rec2.json().get("data") or []
                        if not any(r.get("ticker") == t for r in rows2):
                            absent += 1
                elif missing:
                    absent += len(missing)
        except QuotaExhausted as exc:
            # A clean stop, not a crash: the plan resumes next cycle, and every
            # call already made is in Tier 1.
            print(f"\nQUOTA GUARD: {exc}", flush=True)
            errors.append(str(exc))
            break
        except Exception as exc:  # noqa: BLE001 - one bad date must not end the run
            failed += 1
            errors.append(f"{job.trade_date} {job.purpose}: {type(exc).__name__}: {exc}")
            if failed > 50:
                print("\nABORT: more than 50 failures; stopping.", flush=True)
                break

        now = time.time()
        if now - last_log >= log_every or i == len(plan.jobs):
            last_log = now
            elapsed = now - started
            rate = i / elapsed if elapsed else 0
            eta = (len(plan.jobs) - i) / rate if rate else 0
            print(
                f"  [pull] {i}/{len(plan.jobs)} ({100*i/len(plan.jobs):.1f}%) "
                f"fetched={done} cached={cached} failed={failed} absent={absent} "
                f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s",
                flush=True,
            )

    return {
        "planned": len(plan.jobs),
        "fetched": done,
        "cache_hits": cached,
        "failed": failed,
        "absent_tickers": absent,
        "truncation_retries": truncation_retries,
        "elapsed_s": round(time.time() - started, 1),
        "errors": errors[:50],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="plan and cost it; spend nothing")
    ap.add_argument("--confirm", action="store_true", help="actually spend quota")
    ap.add_argument("--budget", type=int, default=BUDGET_CALLS)
    ap.add_argument("--min-year", type=int, default=2017)
    ap.add_argument(
        "--buckets", nargs="*", default=None,
        help="mcap slices to target (default: all of them); labels as in "
             "engine.data.coverage.MCAP_BUCKETS, e.g. '<1B' '1-10B' '>10B'",
    )
    ap.add_argument(
        "--focus", nargs="*", default=None, metavar="TICKER",
        help="focus+hitchhike mode: fetch these names' dates, and let every "
             "ticker that needs the same dates ride along (see build_focus_plan)",
    )
    ap.add_argument(
        "--t2", action="store_true",
        help="decision-chain mode: buy the close a T-2 trade would be DECIDED "
             "on, for the events that are already replayable end-to-end "
             "(see build_t2_plan)",
    )
    ap.add_argument(
        "--decision-offset", type=int, nargs="+", default=[-1], metavar="N",
        help="sessions before the entry close (default -1). Space-separated, "
             "NOT '=' — `--decision-offset -1 -2` plans both arms; "
             "`--decision-offset=-1 -2` leaves the -2 dangling and argparse "
             "rejects it. Arms planned together share date batches, which is "
             "cheaper than pulling them one at a time.",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if not (args.dry_run or args.confirm):
        ap.error("pass --dry-run to plan, or --confirm to spend. Never both implicitly.")

    buckets = tuple(args.buckets) if args.buckets else None
    if buckets is not None:
        from engine.data.coverage import MCAP_BUCKETS

        known = {label for label, _, _ in MCAP_BUCKETS}
        bad = sorted(set(buckets) - known)
        if bad:
            ap.error(f"unknown bucket labels {bad}; known: {sorted(known)}")

    if args.t2 and args.focus:
        ap.error("--t2 and --focus are different plans; run one at a time")

    if args.t2:
        # min_year defaults to 2017 for the entry pull; the replay universe the
        # T-2 book is measured on starts in 2018.
        min_year = 2018 if args.min_year == 2017 else args.min_year
        plan = build_t2_plan(
            args.decision_offset, min_year=min_year, budget=args.budget,
            buckets=buckets,  # None = every slice, which is what this pull wants
        )
        print(render_t2_dry_run(plan))
    elif args.focus:
        plan = build_focus_plan(
            args.focus, min_year=args.min_year, budget=args.budget,
            buckets=buckets if buckets is not None else TARGET_BUCKETS,
        )
        print(render_dry_run(plan))
    else:
        plan = build_plan(min_year=args.min_year, budget=args.budget,
                          buckets=buckets if buckets is not None else TARGET_BUCKETS)
        print(render_dry_run(plan))

    report = {"plan": plan.summary(), "generated_at": datetime.now(timezone.utc).isoformat()}
    if args.confirm:
        print("--confirm given: spending quota now.\n", flush=True)
        report["execution"] = execute(plan)
        print(f"\n{json.dumps(report['execution'], indent=1)}", flush=True)
        print(
            "\nNext: re-run `python3 -m engine.data.rebuild --table chains` to "
            "normalize the new raw files into Tier 2, then "
            "`python3 checks/phase0_audit.py` to re-measure coverage.",
            flush=True,
        )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
