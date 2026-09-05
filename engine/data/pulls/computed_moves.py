#!/usr/bin/env python3
"""Synthesize oquants-format moves files for tickers oquants does not carry.

    python3 -m engine.data.pulls.computed_moves --dry-run
    python3 -m engine.data.pulls.computed_moves --confirm

The panel's event universe is bounded by the oquants moves cache (2,936
tickers). EXP-117 Stage 3 measured 34 further tickers reaching the >=12-print
scoreability bar on the ORATS calendar alone; this pull gives those names the
event-history block the panel and the live scorer need, so they stop rendering
as MISSING_FEATURES rows.

Provenance is the point. The target values here are COMPUTED, not
vendor-supplied:

* dates + BMO/AMC sessions: Tier-2 ``earnings_events``, ORATS rows only;
* realized move: session-aware close-to-close on yfinance ``Close``
  (auto_adjust=False — split-adjusted, not dividend-adjusted), the series
  EXP-117 validated exact against Polygon truth (99.5% within 0.5pp, median
  diff 0.000);
* implied move: NO LONGER WRITTEN. The panel takes it from ORATS
  ``daily_market`` directly (panel.add_implied_history), so emitting it here
  would have duplicated the same series under a second name — which is exactly
  what the oquants column had become once its vendor was dropped;
* quarters: ordinal within the calendar year (a label; no model consumes it).

Events with no close on either side of the window, or a window wider than
five calendar days (halt/delisting), are excluded and counted, never guessed
(EXP-117 DEFINITION.md R3).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine import paths
from engine.data import store
from engine.data.fetch import Fetcher

#: The scoreability bar the board's champion models impose (span-12 EMAs).
MIN_SCOREABLE = 12

#: P→Q windows wider than this are halts/gaps, excluded not guessed.
MAX_GAP_CALENDAR_DAYS = 5


def target_tickers(*, all_scoreable: bool = False) -> tuple[list[str], dict]:
    """Scoreable on the ORATS calendar, with daily rows.

    The daily-market requirement is load-bearing: the champion size model
    needs or_implied / or_rvol30 / mcap_log, and a ticker with no
    daily_market rows would stay MISSING_FEATURES even with history rows.

    ``all_scoreable`` drops the "absent from oquants" condition, which is what
    turns this from a universe EXTENSION into a realized-move SOURCE. The panel
    merges per field (see ``build_events``): the computed realized move wins
    wherever it exists, oquants keeps ``implied_move``, and an event only this
    pull has is added outright.

    Two reasons the wider mode is the right default going forward. The oquants
    cache has no fetcher in this repository and lags — on 2026-09-05 it ended
    2026-08-31 while Tier 2 held prints through 09-04, so 103 events could not
    reach the panel at all. And the realized move computed here is the better
    measurement: EXP-117 validated it at 99.5% within 0.5pp against Polygon,
    and the 2026-09-05 arbitration found it matching oquants to the cent on
    92.5% of the events where oquants and ORATS spot disagreed.
    """
    ev = store.read_table(
        "earnings_events",
        columns=["ticker", "event_date", "session", "src_orats"],
    )
    today = pd.Timestamp.today().normalize()
    hist = ev[ev["src_orats"] & ev["session"].notna()
              & (pd.to_datetime(ev["event_date"]) < today)]
    counts = hist.groupby("ticker")["event_date"].size()
    scoreable = set(counts[counts >= MIN_SCOREABLE].index)

    oq_tickers = {p.name[len("moves_"):-len(".json")]
                  for p in paths.RAW_OQUANTS_MOVES.glob("moves_*.json")}
    dm = store.read_table("daily_market", columns=["ticker"])
    dm_tickers = set(dm["ticker"].astype(str))

    pool = scoreable if all_scoreable else (scoreable - oq_tickers)
    targets = sorted(pool & dm_tickers)
    report = {
        "mode": "all_scoreable" if all_scoreable else "extension_only",
        "scoreable_on_orats_calendar": len(scoreable),
        "also_in_oquants": len(scoreable & oq_tickers),
        "no_daily_market_rows": len(pool - dm_tickers),
        "targets": len(targets),
    }
    return targets, report


def fetch_history(f: Fetcher, ticker: str) -> tuple[np.ndarray, np.ndarray] | None:
    """yfinance Close series (split-adjusted, not dividend-adjusted)."""
    # The Fetcher RAISES on a non-200 rather than returning one, so the status
    # guard below never fired and a single delisted ticker took the whole run
    # down — BF_B, at 352 of 2,857, after 351 successful fetches. A universe
    # pull must survive its worst member: one name with no price history is a
    # fact about that name, not a reason to abandon the other 2,505.
    from engine.data.sources.base import FetchError

    try:
        rec = f.fetch("yfinance", "history", {"ticker": ticker, "period": "max"},
                      note="computed-moves")
    except (FetchError, OSError, ValueError):
        return None
    if rec is None or rec.status != 200:
        return None
    try:
        frame = pd.read_csv(io.BytesIO(rec.body))
    except (ValueError, OSError):
        return None
    if frame.empty or "Close" not in frame.columns:
        return None
    date_col = frame.columns[0]
    # yfinance writes the index with per-row UTC offsets that flip at DST;
    # parse through UTC, then drop the tz — the trade date survives intact.
    # Parse through UTC to survive the per-row offsets yfinance writes (they
    # flip at DST), then drop the tz AND NORMALIZE TO MIDNIGHT.
    #
    # The normalize is load-bearing and its absence was a silent one-session
    # error. Dropping the tz on a -05:00 midnight leaves 05:00, so a caller
    # searching for `np.datetime64("2012-01-24")` — midnight — lands BEFORE
    # that row and anchors on the previous session. Every computed move was
    # then the day before the print: AAPL 2012-01-24 came out at -1.64%, which
    # is the 23rd's move, against a true +6.24%.
    #
    # Found 2026-09-05 by rebuilding the panel from this pull at scale and
    # checking the result against the values it replaced: 49% of shared events
    # differed by more than 1pp and the SIGNS disagreed 27% of the time, which
    # is what a one-day shift looks like rather than a measurement difference.
    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    dates = dates.dt.tz_localize(None).dt.normalize()
    closes = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=float)
    ok = dates.notna() & np.isfinite(closes) & (closes > 0)
    dates = dates[ok].to_numpy(dtype="datetime64[ns]")
    closes = closes[ok]
    order = np.argsort(dates, kind="stable")
    return dates[order], closes[order]


def session_move(sd, sc, t, session) -> float | None:
    if session == "BMO":
        j_pre = int(np.searchsorted(sd, t, side="left")) - 1
        j_post = int(np.searchsorted(sd, t, side="left"))
    else:
        j_pre = int(np.searchsorted(sd, t, side="right")) - 1
        j_post = int(np.searchsorted(sd, t, side="right"))
    if j_pre < 0 or j_post >= len(sd):
        return None
    if (sd[j_post] - sd[j_pre]) / np.timedelta64(1, "D") > MAX_GAP_CALENDAR_DAYS:
        return None
    p, q = sc[j_pre], sc[j_post]
    if not np.isfinite(p) or not np.isfinite(q) or p <= 0:
        return None
    return float((q / p - 1.0) * 100.0)


def build_ticker(ticker: str, events: pd.DataFrame, sd, sc,
                 daily: pd.DataFrame) -> dict | None:
    dates: list[str] = []
    moves: list[float] = []
    implied: list = []
    quarters: list[int] = []
    skipped = 0

    dm_dates = daily["date"].to_numpy()
    dm_im = daily["implied_move"].to_numpy(dtype=float)

    year_seen: dict[int, int] = {}
    for r in events.itertuples():
        t = r.event_date.to_datetime64()
        m = session_move(sd, sc, t, r.session)
        if m is None:
            skipped += 1
            continue
        # panel as-of convention: the last EOD row strictly before the print
        j = int(np.searchsorted(dm_dates, t, side="left")) - 1
        im = float(dm_im[j]) if j >= 0 and np.isfinite(dm_im[j]) else None
        year = int(str(r.event_date)[:4])
        year_seen[year] = year_seen.get(year, 0) + 1
        dates.append(str(r.event_date)[:10])
        moves.append(m)
        implied.append(im)
        quarters.append(year_seen[year])

    if len(dates) < 5:  # panel admission is k>=4, so fewer cannot score
        return None
    return {
        "ok": True,
        "ticker": ticker,
        "n_events": len(dates),
        "skipped_events": skipped,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "provenance": "computed: yfinance closes + ORATS calendar (EXP-117)",
        "data": {
            "dates": dates,
            "realized_moves": moves,
            "abs_realized_moves": [abs(m) for m in moves],
            "implied_moves": implied,
            "quarters": quarters,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--tickers", default=None, help="comma-separated override")
    ap.add_argument("--all-scoreable", action="store_true",
                    help="every scoreable ticker, not only those oquants lacks — "
                         "the realized-move SOURCE mode")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.confirm:
        print("pass --dry-run or --confirm", file=sys.stderr)
        return 2

    targets, selection = target_tickers(all_scoreable=args.all_scoreable)
    if args.tickers:
        keep = {t.strip().upper() for t in args.tickers.split(",")}
        targets = [t for t in targets if t in keep]
    print(json.dumps(selection, indent=1), flush=True)
    print(f"building moves for {len(targets)} tickers", flush=True)
    if args.dry_run:
        print("dry run: nothing written", flush=True)
        return 0

    ev = store.read_table(
        "earnings_events",
        columns=["ticker", "event_date", "session", "src_orats"],
    )
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    ev = ev[ev["src_orats"] & ev["session"].notna()]
    dm = store.read_table("daily_market",
                          columns=["ticker", "date", "implied_move"])

    out_dir = paths.COMPUTED_MOVES
    out_dir.mkdir(parents=True, exist_ok=True)
    f = Fetcher()
    started = time.time()
    written = no_history = too_few = 0
    for i, tk in enumerate(targets):
        # A ticker already written this run is skipped, so an interrupted pull
        # resumes instead of refetching 2,857 names from the top.
        if (out_dir / f"moves_{tk}.json").exists():
            written += 1
            continue
        try:
            series = fetch_history(f, tk)
        except Exception as exc:  # nothing about one name may stop the universe
            print(f"  [{i+1}/{len(targets)}] {tk}: FAILED {type(exc).__name__}", flush=True)
            no_history += 1
            continue
        if series is None:
            no_history += 1
            print(f"  [{i+1}/{len(targets)}] {tk}: no yfinance history", flush=True)
            continue
        sd, sc = series
        tk_events = ev[(ev["ticker"] == tk) & (ev["event_date"] >= pd.Timestamp(sd[0]))]
        tk_daily = dm[dm["ticker"] == tk].sort_values("date")
        doc = build_ticker(tk, tk_events, sd, sc, tk_daily)
        if doc is None:
            too_few += 1
            print(f"  [{i+1}/{len(targets)}] {tk}: too few computable events", flush=True)
            continue
        (out_dir / f"moves_{tk}.json").write_text(json.dumps(doc))
        written += 1
        print(f"  [{i+1}/{len(targets)}] {tk}: {doc['n_events']} events "
              f"({doc['skipped_events']} skipped), {time.time()-started:.0f}s", flush=True)
    print(f"FINISHED written={written} no_history={no_history} too_few={too_few} "
          f"-> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
