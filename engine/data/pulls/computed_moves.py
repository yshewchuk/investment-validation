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
import hashlib
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


def target_tickers(*, all_scoreable: bool = False, since=None) -> tuple[list[str], dict]:
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
    }
    if since is not None:
        # Ongoing mode: only names that have PRINTED since the watermark need
        # their moves recomputed. Every other ticker's realized history is
        # unchanged by definition, so rebuilding it is 2,800 needless network
        # fetches — the difference between a nightly step and an afternoon.
        since = pd.Timestamp(since).normalize()
        recent = ev[ev["src_orats"] & ev["session"].notna()
                    & (pd.to_datetime(ev["event_date"]) >= since)]
        printed = set(recent["ticker"].astype(str))
        targets = [t for t in targets if t in printed]
        report["since"] = str(since.date())
        report["printed_since"] = len(printed)
    report["targets"] = len(targets)
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


#: Append-only log of tickers finished in the CURRENT build, so an interrupted
#: pull resumes instead of restarting. Dot-prefixed so it cannot match the
#: ``moves_*.json`` glob the panel and EXP-119 read this directory with.
CHECKPOINT_NAME = ".checkpoint.jsonl"

#: How far the realized moves have been built. The nightly reads it to decide
#: which names printed since, and advances it when the pull succeeds.
STATE_NAME = ".state.json"

#: Sessions of deliberate overlap when advancing the watermark. An AMC print on
#: the last final session is not measurable until the NEXT close exists, so a
#: watermark that ran to the edge would step over those events and never revisit
#: them. Re-examining a few names costs seconds; a permanently skipped event is
#: a hole in the panel that nothing later repairs.
WATERMARK_OVERLAP_SESSIONS = 3


def read_state(out_dir: Path | None = None) -> dict:
    path = (out_dir or paths.COMPUTED_MOVES) / STATE_NAME
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write_state(through, out_dir: Path | None = None) -> Path:
    """Record how far the moves are built. tmp+replace so a kill cannot leave a
    truncated watermark, which would read as 'never built' and trigger a full
    rebuild, or worse parse as a date far in the future and skip everything."""
    directory = out_dir or paths.COMPUTED_MOVES
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STATE_NAME
    tmp = directory / f".{STATE_NAME}.tmp"
    tmp.write_text(json.dumps({
        "moves_through": str(pd.Timestamp(through).date()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    tmp.replace(path)
    return path


def _build_fingerprint(all_scoreable: bool, events: pd.DataFrame) -> str:
    """Identity of one build, so a checkpoint is only resumed into its own.

    Mode plus the event table's own extent: a pull asked for a wider universe,
    or run after new prints landed, is a DIFFERENT build and must not inherit
    the previous one's completed set — which is precisely the mistake the
    ``.exists()`` check below used to make.
    """
    payload = json.dumps(
        {
            "mode": "all_scoreable" if all_scoreable else "extension_only",
            "n_events": int(len(events)),
            "max_event_date": str(pd.to_datetime(events["event_date"]).max().date()),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_checkpoint(path: Path, fingerprint: str) -> dict[str, str]:
    """``{ticker: outcome}`` already finished under ``fingerprint``.

    A checkpoint from a different build, or one that cannot be parsed, is
    ignored rather than trusted — a corrupt resume that silently skips work is
    worse than repeating it.
    """
    if not path.exists():
        return {}
    done: dict[str, str] = {}
    try:
        with path.open() as fh:
            for n, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if n == 0:
                    if record.get("fingerprint") != fingerprint:
                        return {}
                    continue
                done[str(record["ticker"])] = str(record.get("outcome", "written"))
    except (OSError, ValueError, KeyError):
        return {}
    return done


def _start_checkpoint(path: Path, fingerprint: str, total: int) -> None:
    path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "targets": int(total),
    }) + "\n")


def _record(path: Path, ticker: str, outcome: str) -> None:
    """Append one finished ticker. Every TERMINAL outcome is recorded, not just
    a successful write: a name with no history is settled business, and a
    resume that retried it would pay the same network call again for the same
    answer."""
    with path.open("a") as fh:
        fh.write(json.dumps({"ticker": ticker, "outcome": outcome}) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--tickers", default=None, help="comma-separated override")
    ap.add_argument("--all-scoreable", action="store_true",
                    help="every scoreable ticker, not only those oquants lacks — "
                         "the realized-move SOURCE mode")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and rebuild every target")
    ap.add_argument("--since", default=None,
                    help="only names that printed on or after this date — the "
                         "ongoing mode; pass 'state' to use the recorded watermark")
    ap.add_argument("--advance-watermark", default=None,
                    help="on success, record the moves as built through this date")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.confirm:
        print("pass --dry-run or --confirm", file=sys.stderr)
        return 2

    since = args.since
    if since == "state":
        since = read_state().get("moves_through")
        if since is None:
            print("no watermark recorded yet; building the full universe", flush=True)
    targets, selection = target_tickers(all_scoreable=args.all_scoreable, since=since)
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

    # Resume against THIS build, not against whatever files happen to be on
    # disk. The previous rule skipped any ticker with an existing
    # `moves_<TK>.json`, which reads as resume-safety but means the second run
    # of this pull writes nothing at all: the directory already holds a file
    # per ticker from the last build, so every target is skipped and the
    # realized moves stay frozen at the date they were last built. That is how
    # the panel sat at 2026-09-03 while Tier 2 held prints through 09-10.
    checkpoint = out_dir / CHECKPOINT_NAME
    fingerprint = _build_fingerprint(args.all_scoreable, ev)
    done = {} if args.fresh else _load_checkpoint(checkpoint, fingerprint)
    if done:
        print(f"resuming build {fingerprint}: {len(done):,} of {len(targets):,} "
              f"already finished", flush=True)
    else:
        _start_checkpoint(checkpoint, fingerprint, len(targets))
        print(f"starting build {fingerprint} over {len(targets):,} tickers", flush=True)

    f = Fetcher()
    started = time.time()
    written = sum(1 for v in done.values() if v == "written")
    no_history = sum(1 for v in done.values() if v == "no_history")
    too_few = sum(1 for v in done.values() if v == "too_few")
    for i, tk in enumerate(targets):
        if tk in done:
            continue
        try:
            series = fetch_history(f, tk)
        except Exception as exc:  # nothing about one name may stop the universe
            print(f"  [{i+1}/{len(targets)}] {tk}: FAILED {type(exc).__name__}", flush=True)
            no_history += 1
            _record(checkpoint, tk, "no_history")
            continue
        if series is None:
            no_history += 1
            print(f"  [{i+1}/{len(targets)}] {tk}: no yfinance history", flush=True)
            _record(checkpoint, tk, "no_history")
            continue
        sd, sc = series
        tk_events = ev[(ev["ticker"] == tk) & (ev["event_date"] >= pd.Timestamp(sd[0]))]
        tk_daily = dm[dm["ticker"] == tk].sort_values("date")
        doc = build_ticker(tk, tk_events, sd, sc, tk_daily)
        if doc is None:
            too_few += 1
            print(f"  [{i+1}/{len(targets)}] {tk}: too few computable events", flush=True)
            _record(checkpoint, tk, "too_few")
            continue
        # tmp+replace: a kill between truncate and write would otherwise leave
        # a half-written moves file that the panel would read as this ticker's
        # whole history.
        tmp = out_dir / f".moves_{tk}.json.tmp"
        tmp.write_text(json.dumps(doc))
        tmp.replace(out_dir / f"moves_{tk}.json")
        written += 1
        _record(checkpoint, tk, "written")
        print(f"  [{i+1}/{len(targets)}] {tk}: {doc['n_events']} events "
              f"({doc['skipped_events']} skipped), {time.time()-started:.0f}s", flush=True)
    print(f"FINISHED written={written} no_history={no_history} too_few={too_few} "
          f"-> {out_dir}", flush=True)
    if args.advance_watermark:
        path = write_state(args.advance_watermark, out_dir)
        print(f"watermark: moves built through {args.advance_watermark} -> {path}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
