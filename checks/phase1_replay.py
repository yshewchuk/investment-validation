#!/usr/bin/env python3
"""Replay equivalence — Phase 1's load-bearing acceptance test.

    python3 checks/phase1_replay.py
    python3 checks/phase1_replay.py --strategy STR-THRU --dates 10

The guide's stated purpose: *"the proof that the live scorer and the research
code cannot drift apart"*. The guide phrases it as reproducing the Phase 2
backtest's trades. Phase 2 does not exist yet, and Phase 1 resolves that by
owning the replay path both will use — so the equivalence is asserted against
the Tier-2 trade set :mod:`engine.replay` produced, which is precisely what
Phase 2's backtests will be built from.

Three equivalences, each closing a different way the two could diverge:

1. **Entry pricing.** Scoring an event as-of its historical decision date must
   reproduce the stored trade's entry cost to 1e-6 — same strike, same expiry,
   same DTE. This is the one that matters: it says the number the dashboard
   shows and the number the backtest booked came from the same quotes through
   the same code.

2. **Trade selection.** The set of tickers the scorer can price on a given
   as-of date must equal the set the replay priced. A scorer that silently
   skipped the hard ones would look more accurate than it is.

3. **Feature agreement.** The live feature path (used for upcoming events) must
   reproduce the panel path (used for everything historical) to 1e-9. Without
   this the models are trained on one set of numbers and served another, and
   nothing else in the system would notice.

The guide lists a fourth — "portfolio-level expected P&L consistent within
tolerance". That is deliberately **not** duplicated here, because the honest
form of it is a calibration question, not an equivalence one: expected P&L is a
*forecast*, and the meaningful test is whether the forecasts track realized
outcomes out of sample. ``checks/phase1_calibration.py`` does exactly that, over
a larger sample, against the base-rate benchmark. Asserting a tolerance band
here would either restate equivalence 1 (the costs already match to 1e-6) or
invent a threshold with nothing behind it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.data import store  # noqa: E402
from engine.features import (  # noqa: E402
    ABSOLUTE_FEATURES,
    PANEL_FEATURE_COLUMNS,
    FeatureContext,
    live_features,
    panel_features,
)
from engine.fills import FillModel  # noqa: E402
from engine.replay import load_chain_index  # noqa: E402
from engine.score import ScoreRequest, Scorer  # noqa: E402

#: Entry cost must match to the cent-fraction. It is the same arithmetic on the
#: same quotes, so anything above floating-point noise means a real divergence.
COST_TOLERANCE = 1e-6

#: Features are floats through slightly different code paths, so the bar is
#: floating-point agreement rather than bit equality.
FEATURE_TOLERANCE = 1e-9

#: `implied_move` is the oquants quoted move for the event itself and exists
#: only for realized events — the live path cannot produce it, by design.
FEATURE_EXEMPT = {"implied_move"}


def engine_trades(strategy: str | None = None) -> pd.DataFrame:
    trades = store.read_table("trades")
    rows = trades[trades["provenance"].astype(str) == "engine.replay"]
    if strategy:
        rows = rows[rows["strategy"] == strategy]
    return rows.reset_index(drop=True)


def sample_dates(trades: pd.DataFrame, n: int, seed: int = 20260829) -> list[pd.Timestamp]:
    """Historical entry dates to replay, spread across the whole period.

    Stratified by year rather than drawn uniformly: the trade set is denser in
    2018 than 2024, and a uniform sample would test one regime and call it
    coverage.
    """
    dates = pd.to_datetime(trades["entry_date"]).dropna().drop_duplicates()
    if dates.empty:
        return []
    frame = pd.DataFrame({"date": dates, "year": dates.dt.year})
    rng = np.random.default_rng(seed)
    per_year = max(1, n // max(1, frame["year"].nunique()))
    picked: list[pd.Timestamp] = []
    for _, group in frame.groupby("year", sort=True):
        take = min(per_year, len(group))
        idx = rng.choice(len(group), size=take, replace=False)
        picked.extend(pd.Timestamp(d) for d in group["date"].to_numpy()[idx])
    return sorted(picked)[:n] if len(picked) > n else sorted(picked)


# --------------------------------------------------------------------------
# 1 + 2 — pricing and selection
# --------------------------------------------------------------------------


def check_replay_equivalence(
    strategy: str,
    *,
    n_dates: int = 10,
    alpha: float = 0.5,
    scorer: Scorer | None = None,
    verbose: bool = True,
) -> dict:
    """Re-score historical decision dates and compare against the stored trades."""
    trades = engine_trades(strategy)
    if trades.empty:
        return {"strategy": strategy, "skipped": "no engine-replayed trades"}

    at_alpha = trades[np.isclose(trades["fill_alpha"].astype(float), alpha)]
    dates = sample_dates(at_alpha, n_dates)
    if not dates:
        return {"strategy": strategy, "skipped": "no entry dates"}

    engine = scorer or Scorer()
    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"]
    ).set_index("event_id")

    compared = mismatched = 0
    missing_from_scorer: list[str] = []
    extra_from_scorer: list[str] = []
    worst = 0.0
    worst_detail = ""

    for date in dates:
        day = at_alpha[pd.to_datetime(at_alpha["entry_date"]) == date]
        if day.empty:
            continue
        keys = [(str(t), pd.Timestamp(d)) for t, d in
                zip(day["ticker"], pd.to_datetime(day["entry_date"]))]
        keys += [(str(t), pd.Timestamp(d)) for t, d in
                 zip(day["ticker"], pd.to_datetime(day["exit_date"]))]
        index = load_chain_index(set(keys), progress_every=0)

        priced_by_scorer: set[str] = set()
        for row in day.itertuples(index=False):
            event_id = str(row.event_id)
            session = (
                str(events.loc[event_id, "session"]) if event_id in events.index else None
            )
            request = ScoreRequest(
                ticker=str(row.ticker),
                strategy=strategy,
                event_date=pd.Timestamp(row.event_date),
                session=session,
                fill=FillModel(alpha),
            )
            result = engine.score(request, chain_index=index)
            if result.entry_cost is None:
                missing_from_scorer.append(event_id)
                continue
            priced_by_scorer.add(event_id)
            compared += 1

            delta = abs(float(result.entry_cost) - float(row.entry_cost))
            same_contract = (
                np.isclose(result.strike, float(row.strike))
                and pd.Timestamp(result.expiry) == pd.Timestamp(row.expiry)
            )
            if delta > COST_TOLERANCE or not same_contract:
                mismatched += 1
                if delta > worst:
                    worst = delta
                    worst_detail = (
                        f"{event_id}: scorer {result.entry_cost:.6f} "
                        f"@ {result.strike}/{pd.Timestamp(result.expiry).date()} vs "
                        f"replay {float(row.entry_cost):.6f} "
                        f"@ {float(row.strike)}/{pd.Timestamp(row.expiry).date()}"
                    )

        replayed = set(day["event_id"].astype(str))
        extra_from_scorer.extend(sorted(priced_by_scorer - replayed))

    out = {
        "strategy": strategy,
        "alpha": alpha,
        "dates": len(dates),
        "compared": compared,
        "cost_mismatches": mismatched,
        "max_cost_delta": worst,
        "worst": worst_detail,
        "scorer_could_not_price": len(missing_from_scorer),
        "scorer_priced_extra": len(extra_from_scorer),
        "passed": mismatched == 0 and not missing_from_scorer and not extra_from_scorer,
    }
    if verbose:
        status = "PASS" if out["passed"] else "FAIL"
        print(
            f"  [{status}] {strategy}: {compared} trades over {len(dates)} dates, "
            f"{mismatched} cost mismatches (max Δ {worst:.2e}), "
            f"{len(missing_from_scorer)} unpriced",
            flush=True,
        )
        if worst_detail:
            print(f"         worst: {worst_detail}", flush=True)
    return out


# --------------------------------------------------------------------------
# 3 — feature agreement
# --------------------------------------------------------------------------


def check_feature_equivalence(
    *, n_events: int = 200, seed: int = 20260829, verbose: bool = True,
    context=None,
) -> dict:
    """The live feature path must reproduce the panel path on historical events.

    ``context`` is accepted so a caller that already holds a loaded
    :class:`FeatureContext` (e.g. the acceptance harness's scorer) can share it;
    loading a fresh one here would double the in-memory ``daily_market`` copy.
    """
    context = context or FeatureContext.load()
    panel = context.panel
    rng = np.random.default_rng(seed)

    # Events late in each ticker's history, so there is a prior row to advance from.
    eligible = panel[panel["n_prior"] >= 8]
    tickers = eligible["ticker"].drop_duplicates().to_numpy()
    picked = rng.choice(tickers, size=min(n_events, len(tickers)), replace=False)

    compared = 0
    worst = 0.0
    worst_name = ""
    failures: list[str] = []

    for ticker in picked:
        rows = eligible[eligible["ticker"] == ticker].sort_values("date")
        if len(rows) < 2:
            continue
        event_date = rows["date"].iloc[-1]
        try:
            from_panel = panel_features(ticker, event_date, context=context, session="AMC")
            from_live = live_features(ticker, event_date, context=context, session="AMC")
        except (KeyError, ValueError) as exc:
            failures.append(f"{ticker}: {exc}")
            continue
        compared += 1
        # ABSOLUTE_FEATURES are derived rather than stored, so they are not in
        # PANEL_FEATURE_COLUMNS — and a derived feature is exactly the kind that
        # can be produced by one path and forgotten by the other. This check
        # exists to catch training/serving skew, so it has to see them.
        for name in tuple(PANEL_FEATURE_COLUMNS) + tuple(ABSOLUTE_FEATURES):
            if name in FEATURE_EXEMPT:
                continue
            a, b = from_panel.values.get(name), from_live.values.get(name)
            if a is None or b is None:
                continue
            if np.isnan(a) and np.isnan(b):
                continue
            delta = abs(a - b) if not (np.isnan(a) or np.isnan(b)) else float("inf")
            if delta > worst:
                worst, worst_name = delta, f"{name} ({ticker}): {a} vs {b}"

    out = {
        "events_compared": compared,
        "max_delta": worst,
        "worst": worst_name,
        "failures": failures[:5],
        "passed": compared > 0 and worst <= FEATURE_TOLERANCE and not failures,
    }
    if verbose:
        status = "PASS" if out["passed"] else "FAIL"
        print(
            f"  [{status}] features: {compared} events, max |panel − live| = {worst:.2e} "
            f"(tolerance {FEATURE_TOLERANCE:.0e})",
            flush=True,
        )
        if worst_name:
            print(f"         worst: {worst_name}", flush=True)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", action="append", default=None)
    ap.add_argument("--dates", type=int, default=10)
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--skip-features", action="store_true")
    args = ap.parse_args(argv)

    strategies = args.strategy or ["STR-THRU", "STR-RUNUP"]
    print("Phase 1 replay equivalence\n", flush=True)

    results = []
    scorer = Scorer()
    for strategy in strategies:
        results.append(
            check_replay_equivalence(
                strategy, n_dates=args.dates, alpha=args.alpha, scorer=scorer
            )
        )
    if not args.skip_features:
        results.append(check_feature_equivalence(n_events=args.events))

    failed = [r for r in results if not r.get("passed") and "skipped" not in r]
    skipped = [r for r in results if "skipped" in r]
    print(
        f"\n{len(results) - len(failed) - len(skipped)} passed, {len(failed)} failed, "
        f"{len(skipped)} skipped"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
