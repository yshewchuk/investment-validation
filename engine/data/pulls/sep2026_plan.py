#!/usr/bin/env python3
"""The Sep-1 ORATS pull: plan it, cost it, then spend deliberately.

    python3 -m engine.data.pulls.sep2026_plan --dry-run          # always first
    python3 -m engine.data.pulls.sep2026_plan --confirm          # spends quota

**What the audit says to buy.** Coverage is not uniformly thin — it is thin in
one specific place. Entry-date chains already cover 48.7% of 2017+ events
(79–95% inside the target mcap slices), but *exit*-date chains cover only 18.0%.
Since a through-the-print structure needs both ends, exit chains are the binding
constraint and get first claim on the budget.

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

import numpy as np
import pandas as pd

from engine import paths
from engine.data import coverage, store
from engine.data.fetch import Fetcher
from engine.data.throttle import QuotaExhausted

__all__ = ["PullJob", "PullPlan", "build_plan", "execute", "BATCH", "BUDGET_CALLS"]

#: Tickers per ``/hist/strikes`` call. Five is the verified-safe batch: ten
#: tickers on ``/hist/cores`` returned 502 (payload too large for the gateway),
#: and the existing strikes puller settled on five.
BATCH = 5

#: Total calls this plan may spend, against the 20,000/month allowance.
BUDGET_CALLS = 16_000

#: Fields requested per strike — the same slim set the cached chains carry, so
#: new pulls normalize through exactly the same path as the existing ones.
FIELDS = (
    "ticker,tradeDate,expirDate,dte,strike,stockPrice,callBidPrice,callAskPrice,"
    "putBidPrice,putAskPrice,callMidIv,putMidIv,smvVol,delta,spotPrice"
)

DTE_RANGE = "1,45"

#: Priority order. Exit chains first: they are the binding constraint on every
#: through-the-print structure, and entry coverage is already good.
PRIORITIES = ("exit", "entry", "t14")

TARGET_BUCKETS = ("1-10B", ">10B")


@dataclass
class PullJob:
    trade_date: str
    tickers: tuple[str, ...]
    purpose: str  # exit | entry | t14
    events_unlocked: int = 0

    @property
    def params(self) -> dict:
        return {
            "ticker": ",".join(self.tickers),
            "tradeDate": self.trade_date,
            "dte": DTE_RANGE,
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

    @property
    def n_calls(self) -> int:
        return len(self.jobs)

    def summary(self) -> dict:
        return {
            "calls": self.n_calls,
            "budget": BUDGET_CALLS,
            "skipped_already_cached": self.skipped_cached,
            "by_purpose": self.by_purpose,
            "events_targeted": self.events_targeted,
            "coverage_before": self.coverage_before,
            "truncated_at_budget": self.truncated_at_budget,
        }


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
    """Spend the plan. One process, cache-first, quota-guarded, resumable."""
    fetcher = fetcher or Fetcher()
    started = time.time()
    last_log = 0.0
    done = cached = failed = 0
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
                f"fetched={done} cached={cached} failed={failed} "
                f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s",
                flush=True,
            )

    return {
        "planned": len(plan.jobs),
        "fetched": done,
        "cache_hits": cached,
        "failed": failed,
        "elapsed_s": round(time.time() - started, 1),
        "errors": errors[:50],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="plan and cost it; spend nothing")
    ap.add_argument("--confirm", action="store_true", help="actually spend quota")
    ap.add_argument("--budget", type=int, default=BUDGET_CALLS)
    ap.add_argument("--min-year", type=int, default=2017)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if not (args.dry_run or args.confirm):
        ap.error("pass --dry-run to plan, or --confirm to spend. Never both implicitly.")

    plan = build_plan(min_year=args.min_year, budget=args.budget)
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
