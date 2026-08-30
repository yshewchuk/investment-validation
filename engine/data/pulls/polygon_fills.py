#!/usr/bin/env python3
"""The Polygon real-trade pull: daily traded bars for every contract the
simulated trades touch.

    python3 -m engine.data.pulls.polygon_fills --dry-run        # always first
    python3 -m engine.data.pulls.polygon_fills --confirm        # spends rate

**Why this pull exists.** Every headline in this program is reported at
worst / mid / best fills because the fill assumption flips verdicts, and until
now the only prices in the store were ORATS EOD *quotes*. This pull brings in
what actually *traded* — Polygon daily aggregates: close, VWAP, volume and the
number of fills — for exactly the contracts in the Tier-2 ``trades`` table, so
the mid-fill assumption can be measured instead of assumed wherever the two
sources overlap (2024-08-19 onward on this plan).

**Entitlement, probed 2026-08-30.** This plan's Polygon options cover daily
aggregates, reference and live snapshots. ``/v3/trades``, ``/v3/quotes`` and
intraday bars all return NOT_AUTHORIZED, so daily bars are the finest real-fill
evidence available — do not retry the tick endpoints without a plan change.

**Cost model.** One ``/v2/aggs`` call fetches one contract's whole daily life,
so the budget is one call per contract, not per contract-day. The trade-driven
universe is ~9.3k contracts ≈ 19k contract-days; at the 6.5s pacing that is
roughly 100 minutes of wall clock and no per-call quota (Polygon bills by plan
tier). Jobs run most-observed-first, so an interrupted run already holds the
highest-value bars.

Resumability comes free from Tier 1: a request already in the raw cache is a
cache hit that costs nothing, so an interrupted run resumes by re-running.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from engine.data import store
from engine.data.fetch import Fetcher
from engine.data.sources.polygon import option_ticker

__all__ = [
    "POLYGON_OPTIONS_START",
    "ENTRY_BUFFER_DAYS",
    "RECENT_DAYS",
    "AGGS_LIMIT",
    "ContractJob",
    "FillPullPlan",
    "collect_contracts",
    "build_plan",
    "execute",
    "render_dry_run",
]

#: First day of Polygon options data on this plan (established by the
#: bt/straddle pulls; daily aggs before this date do not exist here).
POLYGON_OPTIONS_START = "2024-08-19"

#: Bars fetched before the earliest observed trade date of a contract. The
#: entry-day sweeps and ±1-day slippage stress tests price days the resolved
#: trades never touched, and they cost nothing extra inside one range call.
ENTRY_BUFFER_DAYS = 21

#: Contracts observed within this many days of the newest observation go to the
#: FRONT of the queue. The recent month is what live decisions and the first
#: fill-quality readings need; an interrupted run should always hold the newest
#: evidence first. Measured from the newest observation date in the trades
#: table (not wall-clock), so a plan is a function of the data, not of when it
#: runs.
RECENT_DAYS = 31

#: Rows one range call may return. A contract's daily life inside the window
#: is at most ~500 bars; the headroom makes a longer window safe, not slower.
AGGS_LIMIT = 10_000


@dataclass(frozen=True)
class ContractJob:
    """One API call: the full daily-aggs life of one contract."""

    contract_ticker: str
    ticker: str
    start: str  # inclusive, YYYY-MM-DD
    end: str  # inclusive, YYYY-MM-DD (the contract expiry)
    n_obs_dates: int = 0

    @property
    def endpoint(self) -> str:
        # Polygon's aggs take the date range as PATH segments; the same values
        # as query params return 404 (verified live 2026-08-30).
        return f"v2/aggs/ticker/{self.contract_ticker}/range/1/day/{self.start}/{self.end}"

    @property
    def params(self) -> dict:
        return {"adjusted": "false", "limit": AGGS_LIMIT}


@dataclass
class FillPullPlan:
    jobs: list[ContractJob] = field(default_factory=list)
    skipped_cached: int = 0
    contracts_in_trades: int = 0
    contract_days: int = 0
    recent_jobs: int = 0
    truncated_at_limit: bool = False

    @property
    def n_calls(self) -> int:
        return len(self.jobs)

    def summary(self) -> dict:
        return {
            "calls": self.n_calls,
            "skipped_already_cached": self.skipped_cached,
            "contracts_in_trades": self.contracts_in_trades,
            "contract_days": self.contract_days,
            "recent_jobs": self.recent_jobs,
            "truncated_at_limit": self.truncated_at_limit,
        }


# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------


def _read_trades() -> pd.DataFrame:
    frames = [
        chunk
        for _, chunk in store.iter_table(
            "trades", columns=["ticker", "legs", "entry_date", "exit_date"]
        )
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_contracts(
    trades: pd.DataFrame | None = None,
    min_date: str = POLYGON_OPTIONS_START,
) -> dict[str, dict]:
    """Every contract the simulated trades hold, with its observed date range.

    Returns ``{contract_ticker: {ticker, first_obs, last_obs, expiry,
    n_obs_dates}}``. Only observation dates on/after ``min_date`` count — that
    is where Polygon has data to answer with — but a contract observed only
    before it can still be relevant to no one, so it simply does not appear.
    """
    if trades is None:
        trades = _read_trades()
    info: dict[str, dict] = {}
    for row in trades.itertuples(index=False):
        legs = getattr(row, "legs", None)
        if not isinstance(legs, str):
            continue
        try:
            doc = json.loads(legs)
        except ValueError:
            continue
        for phase, obs in (("entry", row.entry_date), ("exit", row.exit_date)):
            date = pd.Timestamp(obs)
            if pd.isna(date) or date.strftime("%Y-%m-%d") < min_date:
                continue
            for leg in doc.get(phase) or []:
                try:
                    expiry = pd.Timestamp(leg["expiry"])
                    contract = option_ticker(
                        str(row.ticker),
                        expiry.strftime("%Y-%m-%d"),
                        str(leg["right"]),
                        float(leg["strike"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue  # a leg shape we do not understand adds no job
                rec = info.setdefault(
                    contract,
                    {
                        "ticker": str(row.ticker),
                        "first_obs": date,
                        "last_obs": date,
                        "expiry": expiry,
                        "dates": set(),
                    },
                )
                rec["first_obs"] = min(rec["first_obs"], date)
                rec["last_obs"] = max(rec["last_obs"], date)
                rec["expiry"] = max(rec["expiry"], expiry)
                rec["dates"].add(date.strftime("%Y-%m-%d"))
    return info


def build_plan(
    trades: pd.DataFrame | None = None,
    *,
    fetcher: Fetcher | None = None,
    min_date: str = POLYGON_OPTIONS_START,
    buffer_days: int = ENTRY_BUFFER_DAYS,
    recent_days: int | None = RECENT_DAYS,
    max_calls: int | None = None,
) -> FillPullPlan:
    """One job per contract, recent-month first, cached requests skipped.

    Ordering is (recent, most-observed, contract): contracts observed within
    ``recent_days`` of the newest observation lead the queue because the newest
    evidence is what live decisions read, and an interrupted run should hold
    that first; within a tier the most-observed contracts land earliest. A
    request already in the raw cache costs nothing and is not counted against
    ``max_calls`` — which is also what makes a re-run a resume.
    """
    info = collect_contracts(trades, min_date=min_date)
    plan = FillPullPlan(
        contracts_in_trades=len(info),
        contract_days=sum(len(rec["dates"]) for rec in info.values()),
    )

    fetcher = fetcher or Fetcher()
    buffer = timedelta(days=buffer_days)
    reference = max((rec["last_obs"] for rec in info.values()), default=None)
    recent_cut = reference - timedelta(days=recent_days) if (reference is not None and recent_days) else None

    def order_key(item):
        contract, rec = item
        is_recent = 0 if (recent_cut is not None and rec["last_obs"] >= recent_cut) else 1
        return (is_recent, -len(rec["dates"]), contract)

    ordered = sorted(info.items(), key=order_key)
    for contract, rec in ordered:
        start = (rec["first_obs"] - buffer).strftime("%Y-%m-%d")
        end = rec["expiry"].strftime("%Y-%m-%d")
        job = ContractJob(
            contract_ticker=contract,
            ticker=rec["ticker"],
            start=start,
            end=end,
            n_obs_dates=len(rec["dates"]),
        )
        if fetcher.has("polygon", job.endpoint, job.params):
            plan.skipped_cached += 1
            continue
        if max_calls is not None and len(plan.jobs) >= max_calls:
            plan.truncated_at_limit = True
            break
        plan.jobs.append(job)
        if is_recent := (recent_cut is not None and rec["last_obs"] >= recent_cut):
            plan.recent_jobs += 1
    return plan


def render_dry_run(plan: FillPullPlan) -> str:
    lines = [
        "",
        "=" * 68,
        "POLYGON REAL-TRADE PULL — DRY RUN (nothing has been fetched)",
        "=" * 68,
        f"  planned calls        : {plan.n_calls:,} (one per contract)",
        f"  already cached, free : {plan.skipped_cached:,}",
        f"  contracts in trades  : {plan.contracts_in_trades:,}",
        f"  contract-days needed : {plan.contract_days:,}",
        f"  data window          : {POLYGON_OPTIONS_START} onward (plan entitlement)",
        "",
        "  NOTE: tick trades, NBBO quotes and intraday bars are NOT entitled",
        "  on this plan (probed 2026-08-30). Daily bars are the real-fill",
        "  evidence: close, VWAP, volume, trade count.",
        "",
    ]
    if plan.jobs:
        obs = [j.n_obs_dates for j in plan.jobs]
        lines += [
            f"  obs dates per contract: min {min(obs)}, max {max(obs)}",
            f"  recent-month jobs      : {plan.recent_jobs:,} (fetched first)",
            "",
            "  first 5 calls (recent-month first, then most-observed):",
            *[
                f"    {j.contract_ticker}  {j.start} → {j.end}  ({j.n_obs_dates} obs)"
                for j in plan.jobs[:5]
            ],
        ]
    if plan.truncated_at_limit:
        lines += ["", "  NOTE: truncated at --max-calls; re-run to continue."]
    lines += ["", "  To fetch: re-run with --confirm", "=" * 68, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def execute(plan: FillPullPlan, fetcher: Fetcher | None = None, log_every: float = 30.0) -> dict:
    """Spend the plan. One process (Polygon lock), cache-first, resumable."""
    fetcher = fetcher or Fetcher()
    started = time.time()
    last_log = 0.0
    done = cached = failed = 0
    errors: list[str] = []

    for i, job in enumerate(plan.jobs, 1):
        try:
            record = fetcher.fetch(
                "polygon", job.endpoint, job.params, note="polygon-fills"
            )
            if record.from_cache:
                cached += 1
            else:
                done += 1
        except Exception as exc:  # noqa: BLE001 - one bad contract must not end the run
            failed += 1
            errors.append(f"{job.contract_ticker}: {type(exc).__name__}: {exc}")
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
    ap.add_argument("--dry-run", action="store_true", help="plan and cost it; fetch nothing")
    ap.add_argument("--confirm", action="store_true", help="actually fetch")
    ap.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="cap the number of network calls this run may make (staged pulls)",
    )
    ap.add_argument(
        "--recent-days",
        type=int,
        default=RECENT_DAYS,
        help="contracts observed within this window lead the queue; 0 disables",
    )
    ap.add_argument("--min-date", default=POLYGON_OPTIONS_START)
    ap.add_argument("--json", default=None, help="write the run report to this path")
    args = ap.parse_args(argv)

    if not (args.dry_run or args.confirm):
        ap.error("pass --dry-run to plan, or --confirm to fetch. Never both implicitly.")

    print("planning: reading the trades table …", flush=True)
    plan = build_plan(
        fetcher=Fetcher(),
        min_date=args.min_date,
        recent_days=args.recent_days,
        max_calls=args.max_calls,
    )
    print(render_dry_run(plan))

    report = {"plan": plan.summary(), "generated_at": datetime.now(timezone.utc).isoformat()}
    if args.confirm:
        print("--confirm given: fetching now.\n", flush=True)
        report["execution"] = execute(plan)
        print(f"\n{json.dumps(report['execution'], indent=1)}", flush=True)
        print(
            "\nNext: re-run `python3 -m engine.data.rebuild --table option_daily` "
            "to normalize the new raw payloads into Tier 2.",
            flush=True,
        )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
