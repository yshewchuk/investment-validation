"""The nightly orchestrator: refresh → validate → score → ledger → render →
selfcheck → publish → flags → backup, each step gating the next.

Order is load-bearing, per the guide:

1. **Refresh** Tier 1/2 for calendar names through the fetch wrapper, on the
   quota guard. A refresh failure degrades to cached data (staleness stays
   visible in meta.json) — except a rotated credential, which stops the run:
   retry loops against a dead key only burn goodwill.
2. **Validate** the fresh data. Red stops the pipeline; yesterday's snapshot
   stays published and a flag is raised.
3. **Score** the calendar (one shared Scorer, one shared chain index).
4. **Ledger** — predictions are frozen BEFORE rendering; the frozen record is
   the point. Missed nights are backfilled honestly: their rows carry the true
   (late) ``decision_ts``, and the flags name them — never fabricated on-time.
   Only the ATM board is frozen; the strike ladder rendered for the explorer is
   an EXTRAPOLATED view of the same decision, not a second prediction.
5. **Render** the bundle; **selfcheck** re-scores board rows directly through
   the engine. Any mismatch stops the publish.
6. **Publish** atomically. A down target never blocks: the local bundle still
   rendered, and the retry is next night's.
7. **Flags**: new gate triggers, earnings-date changes, calibration drift,
   quota below reserve.
8. **Backup** sync (public code + private mirror). A failure raises a flag but
   does not block the publish — the snapshot and the backup are independent.

Idempotent by construction: re-running re-reads, re-renders, re-publishes, and
the ledger refuses duplicate ``row_id`` writes.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.data.throttle import QuotaExhausted
from engine.score import LADDER_STEP

__all__ = [
    "NightlyStop", "NightlyReport", "run_nightly", "refresh_calendar_data",
    "single_run_lock",
    "strike_ladder", "validate_refresh",
]

#: How stale the newest daily row may be before the validation battery goes red.
#: Weekends + one grace day: data from Friday still serves a Monday run.
MAX_STALENESS_DAYS = 4

#: Fraction of calendar tickers that must carry a fresh daily row.
MIN_FRESH_TICKER_SHARE = 0.80

#: Missed-night backfill is capped so a long outage cannot trigger an unbounded
#: re-spend of scoring work; the gap beyond this is flagged for a human.
MAX_BACKFILL_DAYS = 7

STATE_FILENAME = ".earnings_state.json"


class NightlyStop(RuntimeError):
    """A gating step failed; the run stops and the previous snapshot stays up."""

    def __init__(self, step: str, detail: str):
        super().__init__(f"nightly stopped at {step}: {detail}")
        self.step = step
        self.detail = detail


@dataclass
class NightlyReport:
    as_of: str
    requested_as_of: str | None = None
    resolved_as_of: str | None = None
    finality: dict | None = None
    steps: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    stopped: str | None = None
    elapsed_s: float = 0.0
    #: Wall-clock cost of each step, in the order they ran. Separate from
    #: ``steps`` because that dict holds what a step PRODUCED, and the two
    #: were being conflated: ``model_evidence`` reported the ``elapsed_s``
    #: stored in its own cached artifact, so a night that skipped the rebuild
    #: entirely still claimed 1,636s — inside a run whose total was 721s.
    #: Timing that can exceed the run containing it is not timing.
    timeline: list = field(default_factory=list)
    _marked_at: float = 0.0

    def start(self, started: float) -> None:
        self._marked_at = started

    def mark(self, step: str, *, started: float) -> float:
        """Record what the step just finished actually cost, and return it.

        Called after each step rather than derived from `steps`, because a
        step that degrades or skips still costs time and still has to appear:
        the run grew from 117s to 3,636s with only four of fourteen steps
        reporting any duration at all, which is why nobody could say where it
        went.
        """
        now = time.time()
        elapsed = now - (self._marked_at or started)
        self._marked_at = now
        self.timeline.append({
            "step": step,
            "elapsed_s": round(elapsed, 1),
            "at_s": round(now - started, 1),
        })
        return elapsed

    def as_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "requested_as_of": self.requested_as_of,
            "resolved_as_of": self.resolved_as_of,
            "finality": self.finality,
            "steps": self.steps,
            "timeline": self.timeline,
            "flags": self.flags,
            "stopped": self.stopped,
            "elapsed_s": round(self.elapsed_s, 1),
        }


@contextlib.contextmanager
def single_run_lock(path: Path | None = None):
    """Hold an exclusive lock for the duration of one nightly, or refuse.

    Two of these overlapping is the box's worst case, not a merely untidy one:
    each holds a ``Scorer`` (measured 5.4GB peak on this 7GB machine), so the
    second one does not queue — it OOMs whichever is unluckier, and a kernel
    OOM kill leaves no traceback anywhere, which is precisely the failure that
    reads as "the nightly silently did nothing". The run takes up to an hour,
    the cron fires nightly and the guide documents running it by hand, so the
    overlap needs no unusual bad luck.

    Non-blocking on purpose. A second run that WAITS an hour and then starts
    scoring against a store the first one has already refreshed is not a
    recovery; the honest answer is to decline and say who holds it. The lock
    is released by the OS if the holder dies, so a killed run does not wedge
    the next one.
    """
    path = Path(path) if path is not None else paths.REPORTS / ".nightly.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            holder = handle.read().strip() or "an unrecorded process"
            raise NightlyStop(
                "lock", f"another nightly is already running ({holder}); "
                        "not starting a second one"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()} started {datetime.now(timezone.utc).isoformat()}")
        handle.flush()
        yield path
    finally:
        # flock is released with the descriptor; the file is left in place so
        # its contents name the last holder.
        handle.close()


def _nights_to_backfill(last, as_of, *, calendar):
    """Which missed nights are worth re-scoring, and which are free to skip.

    Returns ``(nights, skipped_closed, next_unreached)``: the sessions to score,
    the closed dates passed over, and the first date not reached because the cap
    ran out (equal to ``as_of`` when the gap was fully covered).

    **A non-session cannot produce a ledger row.** ``ledger.snapshot`` records
    only rows whose ENTRY is that day, and an entry date is a trading day by
    construction — so backfilling a Saturday scores the entire calendar and then
    discards every row of it. Measured on 2026-09-06: the 09-05 pass cost a full
    201-event, 8-structure re-score, about 45 minutes, for a guaranteed zero.
    The loop walked calendar days while the thing it calls only ever writes on
    sessions, and after a long weekend that is three such passes.

    Skipping does not count against :data:`MAX_BACKFILL_DAYS`. That cap bounds
    SCORING work, and a skipped day does none.
    """
    day = pd.Timestamp(last).normalize() + pd.Timedelta(days=1)
    as_of = pd.Timestamp(as_of).normalize()
    nights: list[pd.Timestamp] = []
    skipped: list[str] = []
    while day < as_of and len(nights) < MAX_BACKFILL_DAYS:
        if calendar.is_trading_day(day):
            nights.append(day)
        else:
            skipped.append(str(day.date()))
        day += pd.Timedelta(days=1)
    return nights, skipped, day


def _default_target():
    """Where a night publishes when the caller did not say.

    ``DASHBOARD_PUBLISH_CMD`` (a wrangler / rclone / aws command template
    containing ``{bundle}``) is the remote channel once the user has created the
    Cloudflare project and put Access in front of it — see
    ``dashboard/README.md``. Until then the local directory publisher is the
    target, so the atomicity and secret-scan guarantees are exercised nightly
    rather than first tried on the day the remote appears.
    """
    command = os.environ.get("DASHBOARD_PUBLISH_CMD")
    if command and "{bundle}" in command:
        return command
    return paths.ROOT / "dashboard" / "published"


def _state_path(bundle_dir: Path | None = None) -> Path:
    base = Path(bundle_dir).parent if bundle_dir is not None else paths.ROOT / "dashboard"
    return base / STATE_FILENAME


def _read_state(bundle_dir: Path | None) -> dict:
    path = _state_path(bundle_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write_state(bundle_dir: Path | None, state: dict) -> None:
    path = _state_path(bundle_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


# --------------------------------------------------------------------------
# step 1 — refresh
# --------------------------------------------------------------------------


#: How far back a print counts as "just happened" and worth asking ORATS to
#: confirm. ORATS backfills an announcement within a few days, and that backfill
#: is what upgrades a row's ``session_src`` from a forward source to the
#: authority — so the confirmation pass only needs to look at the recent past.
CONFIRM_LOOKBACK_DAYS = 10

#: Endpoints that serve a ticker's whole history from one call.
HISTORY_ENDPOINTS = ("hist/summaries", "hist/cores")


#: Symbols ORATS returned 404 for. A negative cache: the fetch store keeps only
#: 2xx, so without this the nightly re-asks for symbols that do not exist there
#: every night. Reviewable by hand — a symbology fix (BF.A vs BFA) is a delete
#: away from being retried.
UNKNOWN_SYMBOLS_PATH = paths.REPORTS / "orats_unknown_symbols.json"


def _unknown_symbols() -> set[str]:
    try:
        return set(json.loads(UNKNOWN_SYMBOLS_PATH.read_text()))
    except (OSError, ValueError):
        return set()


def _remember_unknown_symbol(ticker: str) -> None:
    known = _unknown_symbols()
    known.add(str(ticker))
    try:
        paths.assert_writable(UNKNOWN_SYMBOLS_PATH).parent.mkdir(parents=True, exist_ok=True)
        UNKNOWN_SYMBOLS_PATH.write_text(json.dumps(sorted(known), indent=1))
    except OSError:
        pass


def backfill_ticker_history(tickers, *, fetcher=None) -> dict:
    """Full per-ticker history for names we have never fetched, once each.

    ORATS serves a ticker's ENTIRE history — 2007 to today — from a single
    ``hist/summaries?ticker=X`` call with no ``tradeDate``. So a ticker needs
    this exactly once; from then on the market-wide daily pull keeps it current.
    That is what makes the board's coverage gap cheap to close: the tickers with
    no prediction are overwhelmingly ones with no prior prints and no price
    history, because they only became visible when the nightly started ingesting
    market-wide summaries (~6,000 tickers against a historical ~2,900).

    **One ticker per call, deliberately.** Batching is possible — the endpoint
    accepts a comma list — but it TRUNCATES SILENTLY: a request for 50 tickers
    returns 5 with HTTP 200 and no indication the rest were dropped. Batching
    would therefore need its own record of which tickers really arrived, whereas
    one-per-call makes the Tier-1 cache that record: ``has()`` answers "have we
    ever fetched this ticker" exactly, because the cache key IS the ticker.
    ``live=False`` for the same reason — this is immutable history, and a
    per-day cache key would re-buy it every night.
    """
    from engine.data.fetch import Fetcher

    fetcher = fetcher or Fetcher()
    out = {"considered": len(tickers), "already_cached": 0, "fetched": 0,
           "no_data": [], "failed": [], "unknown_symbol": [], "skipped_unknown": 0,
           "calls": 0}
    unknown = _unknown_symbols()

    for ticker in sorted(set(tickers)):
        params = {"ticker": str(ticker)}
        if all(fetcher.has("orats", ep, params) for ep in HISTORY_ENDPOINTS):
            out["already_cached"] += 1
            continue
        if str(ticker) in unknown:
            out["skipped_unknown"] += 1
            continue
        rows = 0
        try:
            for endpoint in HISTORY_ENDPOINTS:
                record = fetcher.fetch("orats", endpoint, params, note="history backfill")
                out["calls"] += 0 if record.from_cache else 1
                rows += len(record.json().get("data") or [])
        except QuotaExhausted:
            raise
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed
            # 404 is PERMANENT: ORATS does not carry this symbol, and asking
            # again tomorrow will not change that. The Tier-1 cache only stores
            # 2xx, so without recording it here `has()` stays False and the
            # nightly re-buys the same 404 every night — 25 wasted calls a night
            # on the first board this ran against, forever. Anything else (429,
            # 502, transport) IS transient and must stay retryable.
            #
            # The status is kept, not just the exception type: collapsing them
            # to `FetchError` is what hid this distinction in the first place.
            detail = str(exc)
            if "HTTP 404" in detail:
                out["unknown_symbol"].append(str(ticker))
                _remember_unknown_symbol(ticker)
            else:
                out["failed"].append(f"{ticker}: {detail[-40:]}")
            continue
        if rows:
            out["fetched"] += 1
        else:
            # Cached as an empty answer, so a delisted or brand-new name is not
            # re-asked every night for data that does not exist.
            out["no_data"].append(str(ticker))
    return out


#: Tickers per `hist/strikes` call. MEASURED, not guessed: the endpoint caps at
#: 10 and TRUNCATES SILENTLY — a request for 30 returns byte-identical content
#: to a request for the first 10, with HTTP 200 and no indication the other 20
#: were dropped. `pull()` therefore diffs requested against returned and refuses
#: to continue on a short response. (The legacy pullers use 5, which is safe but
#: half the throughput; nothing about 5 was ever measured.)
CHAIN_BATCH = 10

#: The chain fields the replay/scoring path needs, matching what
#: `_shared/strike_pull.py` requests so both produce the same cache key shape.
CHAIN_FIELDS = (
    "ticker,tradeDate,expirDate,dte,strike,stockPrice,callBidPrice,callAskPrice,"
    "putBidPrice,putAskPrice,callMidIv,putMidIv,smvVol,delta,spotPrice"
)
#: DTE window for the forward chain pull.
#:
#: 75, not 45, because 45 days is less than TWO monthly cycles. STR-RUNUP
#: selects `first_dte_at_least(30)`, and for a name with no weeklies the first
#: monthly at or past 30 DTE swings between ~30 and ~63 days out depending on
#: where in the cycle you are — so a 45-day window loses it for about half of
#: every month. Measured on the 1,105 pairs held at 2026-09-02: the October
#: monthly (2026-10-16) appeared in ZERO of them, having been 45–50 DTE on
#: every date that pulled successfully, and only 269 pairs — the names that
#: have weeklies — held any expiry at all past the September monthly. STR-RUNUP
#: could resolve on 3 of 245 rows as a result.
#:
#: This costs no quota: ORATS bills per (tradeDate, ticker-batch), not per row.
#: It costs store size and rebuild time, which is the trade being made.
#:
#: Widening does NOT heal chains already held — `skip_held` is a content check
#: on the (ticker, date) pair and cannot see that what we hold is too narrow.
#: A pair pulled at the old window stays at the old window until something
#: re-pulls it deliberately.
CHAIN_DTE = "1,75"

#: Trading sessions the nightly refreshes, counting back from ``as_of``.
#: Two, because ORATS publishes a session around midnight: a run in the evening
#: cannot get today, so it must also ask for yesterday or it acquires nothing.
#: It also heals a night that did not run — a session is re-asked until it is
#: actually held, and `skip_held` makes an already-held one free.
CHAIN_SESSIONS = 2


def refresh_forward_chains(tickers, as_of, *, fetcher=None, batch: int = CHAIN_BATCH,
                           sessions: int = CHAIN_SESSIONS, skip_held: bool = True) -> dict:
    """EOD option chains for the names the board is about to score.

    This is what the board has been missing. Everything chain-dependent —
    expected P&L, the gate, the premium, the win rate — is blank without it,
    which is why a board of 603 rows showed numbers in three columns. Nothing in
    the nightly fetched chains: `hist/strikes` appeared nowhere in `engine/`, and
    the only pullers that touched it were the strategy backtest scripts pulling
    HISTORY.

    Cost is ~18 calls a night for a 176-ticker board, against a 20,000/month
    budget — the endpoint is per (tradeDate, ticker-batch), so the whole board's
    chains for one session cost less than a rounding error.

    Truncation is checked, not assumed: see :data:`CHAIN_BATCH`.

    **It refreshes the last ``sessions`` trading days, not just ``as_of``.**
    ORATS publishes a session's chains around midnight, so a run in the evening
    that asks only for today gets nothing back — silently, because an
    unpublished date and a name with no chain look identical. Asking for the
    previous session too means the run always acquires something, and it heals
    a gap left by a night that did not run: each session is asked for again
    until it is actually held.

    **A pair already in the store is never re-bought** (``skip_held``). Without
    it, covering several sessions would multiply the nightly cost; with it the
    second session is usually free, because the previous run already got it.
    This is a CONTENT check, not a cache-key check — `Fetcher.has` only
    recognises an identical ticker batch, and the board's ticker set changes
    nightly, so every key would miss.
    """
    from engine.data.fetch import Fetcher

    fetcher = fetcher or Fetcher()
    as_of = pd.Timestamp(as_of).normalize()
    stamp = str(as_of.date())
    unknown = _unknown_symbols()
    wanted_all = [str(t) for t in sorted(set(tickers)) if str(t) not in unknown]

    from engine.calendar import trading_calendar

    cal = trading_calendar()
    stamps: list[str] = []
    try:
        pos = cal.index_of(as_of, side="prev")
        for k in range(max(1, int(sessions))):
            if pos - k >= 0:
                stamps.append(str(cal.days[pos - k].date()))
    except (KeyError, ValueError):
        stamps = [stamp]

    held: set = set()
    if skip_held:
        from engine.replay import available_chain_keys

        try:
            held = available_chain_keys()
        except Exception:  # a cost optimisation must never take the refresh down
            held = set()

    out = {"as_of": stamp, "sessions": stamps, "requested": len(wanted_all),
           "returned": 0, "rows": 0, "calls": 0, "cache_hits": 0,
           "skipped_held": 0, "missing": [], "failed": []}
    for stamp in stamps:
        _refresh_one_session(
            fetcher, wanted_all, stamp, held, batch=batch, out=out, skip_held=skip_held
        )
    return out


def _refresh_one_session(fetcher, wanted_all, stamp, held, *, batch, out, skip_held) -> None:
    """Fetch one trade date for whatever part of the universe still needs it."""
    session_day = pd.Timestamp(stamp).normalize()
    if skip_held:
        wanted = [t for t in wanted_all if (t, session_day) not in held]
        out["skipped_held"] += len(wanted_all) - len(wanted)
    else:
        wanted = list(wanted_all)
    for start in range(0, len(wanted), batch):
        chunk = wanted[start : start + batch]
        params = {"ticker": ",".join(chunk), "tradeDate": stamp,
                  "dte": CHAIN_DTE, "fields": CHAIN_FIELDS}
        try:
            record = fetcher.fetch("orats", "hist/strikes", params,
                                   note="phase3 forward chains")
        except QuotaExhausted:
            raise
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed
            out["failed"].append(f"{chunk[0]}..: {str(exc)[-40:]}")
            continue
        out["calls"] += 0 if record.from_cache else 1
        out["cache_hits"] += 1 if record.from_cache else 0
        rows = record.json().get("data") or []
        out["rows"] += len(rows)
        got = {r.get("ticker") for r in rows}
        out["returned"] += len(got & set(chunk))
        # A short response has TWO causes and they look identical: the endpoint
        # truncating (it caps at 10 and returns HTTP 200 regardless), or the
        # ticker genuinely having no chain on this date. Both are recorded as
        # `missing` rather than asserting one — a single-ticker retry is what
        # distinguishes them, and on the first real board all 7 were 404s, i.e.
        # genuinely absent. Calling that "truncated" would have blamed the
        # batching for the data's own gap.
        out["missing"].extend(t for t in chunk if t not in got)


def _market_wide_days(as_of: pd.Timestamp, lookback: int = 6) -> list:
    """as_of's session first, then recent sessions, weekends skipped.

    ORATS builds the market-wide EOD file later than the per-ticker series,
    so ``tradeDate=<tonight>`` 404s for a while every evening (AGENTS.md).
    The ladder walks back to the newest session ORATS has published; a
    holiday only costs one extra 404 attempt.
    """
    from datetime import timedelta

    days: list = []
    day = pd.Timestamp(as_of).date()
    while len(days) < lookback:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return days


def _newest_published(refresh_out: dict, *, default):
    """The newest tradeDate the market-wide pass actually got data for.

    Pass 2 discovers this by walking back one call per session until ORATS
    answers. Returning it here is free; re-discovering it in the chain pass
    costs one call per ten tickers per session.
    """
    dates = [
        block.get("tradeDate")
        for block in (refresh_out.get("endpoints") or {}).values()
        if isinstance(block, dict) and block.get("tradeDate")
    ]
    if not dates:
        return default
    newest = max(pd.Timestamp(d) for d in dates)
    return min(newest, pd.Timestamp(default))


def refresh_calendar_data(
    tickers: Sequence[str],
    as_of,
    *,
    fetcher=None,
    batch: int = 10,
    horizon_days: int = 21,
    sessions: int = CHAIN_SESSIONS,
    forward: bool = True,
    max_confirmations: int = 400,
) -> dict:
    """Refresh what the board scores from, cheapest and most load-bearing first.

    Three passes, and the order reflects which resource each one spends:

    1. **The forward calendar** (unmetered: Nasdaq + yfinance). This is the one
       that decides whether the board has any rows at all, and it costs no
       ORATS quota — see :mod:`engine.data.pulls.forward_calendar`.
    2. **Summaries and cores**: ONE ORATS call each, for the whole market, via
       ``tradeDate``. Spot, IVs and implied moves. ORATS builds the market-wide
       EOD file later than the per-ticker series, so today's ``tradeDate``
       404s for a while every evening — the pass walks back to the newest
       published session rather than failing the whole refresh.
    3. **ORATS earnings confirmation**, batched, and ONLY for names that printed
       in the last :data:`CONFIRM_LOOKBACK_DAYS` days. This pass used to run
       over every ticker on the calendar frontier — ~290 calls a night against
       a 3,000-call monthly reserve, which is ten nights before live operation
       is out of budget. It also could not do what it appeared to: ORATS
       ``/hist/earnings`` carries no forward dates, so scanning the whole
       universe with it never produced an upcoming event. Narrowed to the
       recent past it does the job it can actually do — turning a forward
       guess into the ORATS-confirmed article.

    Everything is ``live=True``, one cache entry per source per day, so a re-run
    the same day is a cache hit rather than a re-spend.
    """
    from engine.data.fetch import Fetcher

    as_of = pd.Timestamp(as_of).normalize()
    fetcher = fetcher or Fetcher()
    tickers = sorted(set(tickers))
    out = {"calls": 0, "cache_hits": 0, "endpoints": {}, "tickers": len(tickers)}

    # -- 1. the forward calendar (unmetered) --------------------------------
    if forward:
        from engine.data.pulls.forward_calendar import refresh_forward_calendar

        result = refresh_forward_calendar(
            as_of, horizon_days=horizon_days, fetcher=fetcher,
            max_confirmations=max_confirmations, rebuild_events=False,
        )
        out["forward_calendar"] = result.as_dict()

    # -- 1b. per-ticker history, once per ticker ----------------------------
    # Costs nothing after the first time a ticker is seen, and it is what turns
    # a row with no prior prints into a scoreable one: the prior-move features
    # need ~12 past prints, which no amount of today's data supplies.
    out["history"] = backfill_ticker_history(tickers, fetcher=fetcher)
    out["calls"] += out["history"]["calls"]

    # -- 2. market-wide summaries and cores: one ORATS call each ------------
    # ORATS publishes the market-wide EOD file later than the per-ticker
    # series, so tradeDate=as_of 404s for hours after every close (see
    # AGENTS.md). Walk back over recent sessions until the newest published
    # one: a daily-state column one session stale beats a failed refresh.
    from engine.data.fetch import FetchError

    for endpoint in ("hist/summaries", "hist/cores"):
        record = None
        resolved = None
        missed: list[str] = []
        for day in _market_wide_days(as_of):
            try:
                record = fetcher.fetch(
                    "orats", endpoint, {"tradeDate": str(day)},
                    live=True, note="phase3 nightly refresh",
                )
                resolved = str(day)
                break
            except FetchError as exc:
                if "HTTP 404" not in str(exc):
                    raise
                missed.append(str(day))
        if record is None:
            raise FetchError(
                f"orats {endpoint} returned HTTP 404 for {', '.join(missed)}"
            )
        out["calls"] += 0 if record.from_cache else 1
        out["cache_hits"] += 1 if record.from_cache else 0
        try:
            rows = record.json().get("data") or []
        except (ValueError, AttributeError):
            rows = []
        out["endpoints"][endpoint] = {"rows": len(rows), "tradeDate": resolved}

    # -- 3. ORATS confirmation of prints that just happened -----------------
    confirm = _recently_printed(as_of, tickers)
    for start in range(0, len(confirm), batch):
        chunk = confirm[start : start + batch]
        record = fetcher.fetch(
            "orats", "hist/earnings", {"ticker": ",".join(chunk)},
            live=True, note="phase3 earnings confirmation",
        )
        out["calls"] += 0 if record.from_cache else 1
        out["cache_hits"] += 1 if record.from_cache else 0
    out["endpoints"]["hist/earnings"] = {
        "tickers": len(confirm),
        "batches": (len(confirm) + batch - 1) // batch,
        "lookback_days": CONFIRM_LOOKBACK_DAYS,
    }

    # -- 4. option chains for the board's own names --------------------------
    # Last, because it is the only pass whose cost scales with the board, and
    # because everything above must succeed for the board to exist at all.
    #
    # Anchored on the newest session ORATS has actually PUBLISHED, not on the
    # wall clock. Pass 2 just walked back to find that date at one call per
    # session; the chain pass costs one call per TEN TICKERS per session, so
    # asking it for an unpublished date is the expensive way to learn what pass
    # 2 already knows. Measured 2026-09-03: an as_of the market had not yet
    # closed produced 30 of 30 `hist/strikes` 404s before the pass moved on to
    # the session that existed. The prescribed 21:30 cron would have paid that
    # every weeknight - ORATS publishes around 00:12 - which is ~630 calls a
    # month against a 3,000-call live reserve.
    #
    # A board entry that outruns the newest chain is not a problem this needs to
    # solve: `quote_max_age_sessions` already prices it off the newest chain
    # held and flags STALE_QUOTE.
    chain_anchor = _newest_published(out, default=as_of)
    if chain_anchor != as_of:
        out["chain_anchor"] = {
            "as_of": str(pd.Timestamp(as_of).date()),
            "used": str(pd.Timestamp(chain_anchor).date()),
            "why": "newest session ORATS has published; as_of has not closed yet",
        }
    out["chains"] = refresh_forward_chains(tickers, chain_anchor, fetcher=fetcher,
                                           sessions=sessions)
    out["calls"] += out["chains"]["calls"]

    from engine.data.rebuild import rebuild

    rebuild_result = rebuild(tables=("events", "daily", "chains"))
    out["rebuild"] = {"snapshot": rebuild_result.snapshot, "elapsed_s": round(rebuild_result.elapsed_s, 1)}
    return out


def _recently_printed(as_of, tickers: Sequence[str]) -> list[str]:
    """Names whose print landed in the confirmation window, newest first."""
    from engine.data import store

    events = store.read_table("earnings_events", columns=["ticker", "event_date", "src_orats"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    window = events[
        (events["event_date"] >= as_of - pd.Timedelta(days=CONFIRM_LOOKBACK_DAYS))
        & (events["event_date"] <= as_of)
    ]
    if tickers:
        window = window[window["ticker"].isin(set(tickers))]
    # A row ORATS already carries needs no confirming.
    if "src_orats" in window.columns:
        window = window[~window["src_orats"].astype(bool)]
    return sorted(window["ticker"].unique().tolist())


# --------------------------------------------------------------------------
# step 2 — validation battery
# --------------------------------------------------------------------------


def validate_refresh(
    tickers: Sequence[str],
    as_of,
    *,
    max_staleness_days: int = MAX_STALENESS_DAYS,
    min_fresh_share: float = MIN_FRESH_TICKER_SHARE,
) -> list[dict]:
    """Sanity-check the store the run is about to score from.

    Three checks, each returned as ``{"name", "passed", "detail"}``:

    * the newest daily row for the calendar names is not stale beyond
      ``max_staleness_days``;
    * at least ``min_fresh_share`` of the calendar names have ANY recent row
      (a pull that silently lost half the universe must not score);
    * the daily slice passes the structural battery in
      :func:`engine.data.validate.validate_daily`.

    Any failure means red; the caller stops the pipeline.
    """
    from engine.data import store, validate

    as_of = pd.Timestamp(as_of).normalize()
    checks: list[dict] = []
    wanted = sorted(set(tickers))
    years = sorted({as_of.year - 1, as_of.year})

    daily = store.read_table("daily_market", years=years, columns=["ticker", "date", "spot"])
    daily = daily[daily["ticker"].isin(set(wanted))] if wanted else daily

    if daily.empty:
        checks.append({"name": "daily_rows_present", "passed": False,
                       "detail": "no daily_market rows for the calendar names"})
        return checks

    newest = pd.to_datetime(daily["date"]).max().normalize()
    age_days = int((as_of - newest).days)
    checks.append({
        "name": "daily_freshness",
        "passed": age_days <= max_staleness_days,
        "detail": f"newest daily row {newest.date()}, {age_days}d old (limit {max_staleness_days}d)",
    })

    per_ticker_newest = daily.groupby("ticker")["date"].max()
    fresh = per_ticker_newest[
        (as_of - pd.to_datetime(per_ticker_newest).dt.normalize()).dt.days <= max_staleness_days
    ]
    share = len(fresh) / len(per_ticker_newest) if len(per_ticker_newest) else 0.0
    checks.append({
        "name": "ticker_coverage",
        "passed": share >= min_fresh_share,
        "detail": f"{len(fresh)}/{len(per_ticker_newest)} tickers fresh "
                  f"({share:.0%}, floor {min_fresh_share:.0%})",
    })

    recent = daily[pd.to_datetime(daily["date"]) >= newest - pd.Timedelta(days=10)].copy()
    recent["iv30"] = np.nan
    recent["implied_move"] = np.nan
    recent["rvol30"] = np.nan
    _, report = validate.validate_daily(recent)
    structural = [c for c in report.checks if not c.passed]
    checks.append({
        "name": "structural",
        "passed": not structural,
        "detail": "; ".join(f"{c.name}: {c.n_failed}/{c.n_checked}" for c in structural) or "clean",
    })
    return checks


# --------------------------------------------------------------------------
# step 3b — the strike ladder
# --------------------------------------------------------------------------


#: Alternative strikes are offered as fractions of spot either side of ATM.
#: Re-exported rather than restated: two copies of the same step is how the
#: explorer's grid and the board's ladder drift apart.
STRIKE_STEP = LADDER_STEP


def strike_ladder(board: pd.DataFrame, *, scorer, alt_strikes: int, as_of) -> list[dict]:
    """Score ±``alt_strikes`` strikes around ATM — but only where the gate passed.

    The published bundle is static, so the explorer's strike grid has to be
    rendered ahead of time; the desk server can call ``score()`` on demand, a
    phone cannot. But scoring the ladder for the whole board triples a run that
    already takes ~40 minutes over ~2,200 events, and every non-ATM score is
    labelled EXTRAPOLATED until the moneyness experiment is promoted (Phase 2
    backlog 4) — so the expensive rows would be labelled guesses on trades the
    gate has already rejected.

    Scoring the ladder only for gate passers puts the grid exactly where someone
    might act on it, at a few percent of the cost. Rows carry ``strike_offset``
    so the self-check reconstructs the same request that produced them.
    """
    if alt_strikes <= 0 or not len(board):
        return []

    from engine.fills import FillModel
    from engine.score import UNSCORABLE, ScoreRequest, ladder_strike, unscorable_result

    # A forecast-sized structure is placed BY the forecast — TWIN-P's plateau
    # sits at 1.5w from the anchor, chosen so the predicted move lands on it.
    # Re-anchoring it at +/-2.5% keeps the arithmetic coherent (the dependent
    # legs follow) but breaks the only claim the row makes: that this tent is
    # placed on this forecast. A ladder of tents placed nowhere in particular
    # is not the "what if I picked another strike" view the explorer is for.
    from engine.forecast_sizing import FORECAST_SIZED

    from engine.structures import STRUCTURES

    live = board[board["gate_pass"].fillna(False).astype(bool)]
    # A ladder row is "the same structure at another strike", so the strategy
    # has to BE a structure. DYN-SV is a chooser over structures and has no
    # legs of its own: laddering it asked the engine to re-score a name it
    # cannot resolve, and produced three DYN-SV rows for one event, two of them
    # sharing a row_id.
    live = live[live["strategy"].isin(set(STRUCTURES))]
    live = live[~live["strategy"].isin(set(FORECAST_SIZED))]
    offsets = [
        step * sign
        for step in (STRIKE_STEP * k for k in range(1, alt_strikes + 1))
        for sign in (-1.0, 1.0)
    ]
    rows: list[dict] = []
    for record in live.to_dict(orient="records"):
        spot = record.get("spot")
        if spot is None or not np.isfinite(float(spot)):
            continue  # no ATM anchor: nothing to step off
        session = record.get("session")
        for offset in offsets:
            request = ScoreRequest(
                ticker=str(record["ticker"]),
                strategy=str(record["strategy"]),
                as_of=None,
                event_date=pd.Timestamp(record["event_date"]),
                session=None if pd.isna(session) else str(session),
                # Quantized: an unrounded `spot * (1 + offset)` does not
                # survive the bundle's float rounding, and the self-check then
                # re-scores it under a different seed. See `ladder_strike`.
                strike=ladder_strike(spot, offset),
                fill=FillModel(float(record.get("fill", 0.5))),
                variant=record.get("variant"),
                decision_offset=(
                    (record.get("structure_spec") or {}).get("decision_offset")
                ),
                structure_params=record.get("structure_params"),
                # Same quote bound as the board row this ladder steps off. A
                # ladder priced under a stricter rule than its own ATM anchor is
                # not a view of the same decision, and the selfcheck cannot
                # reconstruct it: a ladder row that fails to price has a null
                # strike, so its row_id collapses to "atm" and it becomes
                # indistinguishable from the base row.
                quote_max_age_sessions=(
                    int(record["quote_max_age_sessions"])
                    if record.get("quote_max_age_sessions") is not None else None
                ),
                # And the same ceiling on that bound, for the same reason.
                chain_as_of=as_of,
            )
            try:
                result = scorer.score(request)
            except UNSCORABLE as exc:
                result = unscorable_result(
                    request, as_of=as_of, snapshot=scorer.snapshot, exc=exc
                )
            rows.append(result.as_dict() | {"strike_offset": offset})
    return rows


# --------------------------------------------------------------------------
# flag builders
# --------------------------------------------------------------------------


def _quota_flag() -> dict | None:
    """Raise when ORATS quota has fallen under the reserve kept for live operation.

    Reads every quota ledger rather than one. This flag was silent through all
    of August — it read `paths.QUOTA_LOG`, which no process writes, while the
    strike pulls logged elsewhere and ran the budget down to 875 of 20,000.
    """
    from engine.data.throttle import latest_quota

    state = latest_quota()
    if state["remaining"] is None or not state["below_reserve"]:
        return None
    return {"kind": "quota_below_reserve", "remaining": state["remaining"],
            "floor": state["floor"], "as_of": state["ts"]}


def _calibration_flag() -> dict | None:
    from engine import ledger

    path = ledger.health_path()
    if not path.exists():
        return None
    try:
        health = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    problems = []
    for strategy, block in (health.get("per_strategy") or {}).items():
        if not block.get("available"):
            continue
        skill = block.get("brier_skill")
        if skill is not None and np.isfinite(skill) and skill < -0.05:
            problems.append(f"{strategy} Brier skill {skill:.2f}")
        pred, real = block.get("predicted_mean_pnl"), block.get("realized_mean_pnl")
        if pred is not None and real is not None and abs(pred - real) > 0.02:
            problems.append(f"{strategy} mean-PnL gap {pred:+.1%} vs {real:+.1%}")
    if not problems:
        return None
    return {"kind": "calibration_drift", "problems": problems}


def _date_change_flag(
    previous_events: list[dict], current_events: pd.DataFrame, *, as_of=None
) -> dict | None:
    from engine.calendar import detect_date_changes

    previous = pd.DataFrame(previous_events or [])
    if previous.empty and current_events.empty:
        return None
    # The run's own as-of anchors the comparison window: a replayed night must
    # see the drift that was visible THEN, not the empty window that "today"
    # would give it.
    changes = detect_date_changes(previous, current_events, as_of=as_of)
    if not changes:
        return None
    return {
        "kind": "earnings_date_changed",
        "changes": [
            {"ticker": c.ticker, "change": c.kind, "old": c.old, "new": c.new}
            for c in changes
        ],
    }


def _date_conflict_flag(events: pd.DataFrame) -> dict | None:
    """Name the events whose forward sources disagree about the date.

    A phantom print puts the entry on the wrong day, which the plan lists as a
    known loss source. Both candidate dates stay on the board — the calendar
    never resolves a disagreement silently — so the flag is what tells a reader
    that one of the two rows in front of them is wrong.
    """
    if "date_conflict" not in events.columns or events.empty:
        return None
    rows = events[events["date_conflict"].fillna(False).astype(bool)]
    if rows.empty:
        return None
    by_ticker: dict[str, list[str]] = {}
    for row in rows.itertuples(index=False):
        by_ticker.setdefault(str(row.ticker), []).append(str(pd.Timestamp(row.event_date).date()))
    return {
        "kind": "calendar_date_conflict",
        "detail": "forward sources disagree on the print date; both rows are on "
                  "the board and one of them is wrong",
        "tickers": {k: sorted(v) for k, v in sorted(by_ticker.items())},
    }


def _upcoming_event_rows(events: pd.DataFrame) -> list[dict]:
    return [
        {"ticker": r.ticker, "event_date": str(r.event_date.date()), "session": r.session}
        for r in events.itertuples(index=False)
    ]


def _write_flag_report(as_of, flags: list[dict], steps: dict) -> Path:
    out = paths.assert_writable(paths.REPORTS / "phase3_flags")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{pd.Timestamp(as_of).date()}.json"
    path.write_text(json.dumps(
        {"as_of": str(pd.Timestamp(as_of).date()),
         "generated_at": datetime.now(timezone.utc).isoformat(),
         "flags": flags, "steps": list(steps)},
        indent=1, default=str,
    ))
    return path


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


#: How far ahead the board looks, in calendar days.
#:
#: Set by STR-RUNUP, not by STR-THRU. STR-THRU is entered at the last pre-print
#: close, so a 21-day horizon showed it three weeks of warning. STR-RUNUP is
#: entered **14 trading days before** that close — roughly 20 calendar days —
#: so on a 21-day horizon its entry was already in the past on every row but
#: the last day's, and the board carried 195 STR-RUNUP rows with one gate
#: verdict between them. Nothing was broken; the trades had simply expired
#: before they were shown.
#:
#: 35 days puts STR-RUNUP entries from today out to ~11 trading days ahead.
#: It costs ~10 extra (free) Nasdaq calendar calls and ~2 extra ORATS chain
#: batches a night; the ORATS reserve does not notice.
#:
#: Far-out STR-THRU rows do NOT become guesses as a result: the quote fallback
#: is bounded in SESSIONS from the entry, so an entry a month out still reports
#: NO_CHAIN rather than pricing itself off today's close.
HORIZON_DAYS = 35


def run_nightly(
    as_of=None,
    *,
    horizon_days: int = HORIZON_DAYS,
    alt_strikes: int = 1,
    tickers: Iterable[str] | None = None,
    bundle_dir: Path | str | None = None,
    target: Path | str | None = None,
    refresh: bool = True,
    tiers: bool = True,
    publish: bool = True,
    backup: bool = False,
    backfill: bool = True,
    chain_sessions: int = CHAIN_SESSIONS,
    fetcher=None,
    scorer=None,
    max_staleness_days: int = MAX_STALENESS_DAYS,
    require_as_of: bool = False,
) -> NightlyReport:
    """One nightly pass. Raises :class:`NightlyStop` when a gating step fails."""
    from engine import ledger
    from engine.dashboard.publish import PublishError, publish_bundle
    from engine.dashboard.render import (
        build_health,
        build_meta,
        freshness_summary,
        quota_state,
        render_bundle,
        size_model_mae_from_ledger,
    )
    from engine.dashboard.selfcheck import selfcheck
    from engine.data import store
    from engine.score import Scorer, score_calendar

    started = time.time()
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    requested_as_of = as_of
    bundle_dir = Path(bundle_dir) if bundle_dir is not None else paths.ROOT / "dashboard" / "earnings"
    target = target if target is not None else _default_target()
    probe_url = os.environ.get("DASHBOARD_PROBE_URL") or None

    report = NightlyReport(
        as_of=str(as_of.date()),
        requested_as_of=str(as_of.date()),
    )
    report.start(started)
    state = _read_state(bundle_dir)

    # -- the universe: confirmed events in the horizon ----------------------
    events = store.read_table(
        "earnings_events",
        columns=["event_id", "ticker", "event_date", "session", "date_conflict"],
    )
    events["event_date"] = pd.to_datetime(events["event_date"])
    horizon = as_of + pd.Timedelta(days=horizon_days)
    upcoming = events[
        (events["event_date"] >= as_of)
        & (events["event_date"] <= horizon)
        & events["session"].notna()
    ]
    if tickers is not None:
        upcoming = upcoming[upcoming["ticker"].isin(set(tickers))]

    # The REFRESH universe is wider than the scoring universe, and it must not
    # depend on forward rows already being in the store: a store whose calendar
    # ends yesterday has zero upcoming events, and a refresh keyed on upcoming
    # events would then fetch nothing and stay blind forever. Instead refresh
    # the calendar FRONTIER — every name with an event near the store's newest
    # date — which is exactly the set whose next print dates arrive now.
    frontier_lo = as_of - pd.Timedelta(days=7)
    frontier_hi = events["event_date"].max() + pd.Timedelta(days=45) if len(events) else horizon
    frontier = events[
        (events["event_date"] >= frontier_lo) & (events["event_date"] <= frontier_hi)
    ]
    if tickers is not None:
        frontier = frontier[frontier["ticker"].isin(set(tickers))]
    calendar_tickers = sorted(frontier["ticker"].unique().tolist())
    report.steps["universe"] = {
        "events": int(len(upcoming)),
        "scoring_tickers": int(upcoming["ticker"].nunique()),
        "refresh_tickers": len(calendar_tickers),
    }
    if not len(upcoming):
        report.flags.append({"kind": "no_upcoming_events",
                             "detail": f"nothing confirmed in {as_of.date()} → {horizon.date()} "
                                       "(the calendar may need a refresh)"})

    report.mark("universe", started=started)
    # -- 1. refresh -----------------------------------------------------------
    if refresh and calendar_tickers:
        from engine.data.fetch import CredentialRotated, FetchError
        from engine.data.throttle import QuotaExhausted

        try:
            report.steps["refresh"] = refresh_calendar_data(
                calendar_tickers, as_of, fetcher=fetcher, horizon_days=horizon_days,
                sessions=chain_sessions,
            )
            # The calendar may have moved under the refresh — re-read it.
            events = store.read_table(
                "earnings_events",
                columns=["event_id", "ticker", "event_date", "session", "date_conflict"],
            )
            events["event_date"] = pd.to_datetime(events["event_date"])
            upcoming = events[
                (events["event_date"] >= as_of)
                & (events["event_date"] <= horizon)
                & events["session"].notna()
            ]
            if tickers is not None:
                upcoming = upcoming[upcoming["ticker"].isin(set(tickers))]
        except CredentialRotated as exc:
            raise NightlyStop("refresh", f"credential rotated — {exc}. Update .env; do not retry.") from exc
        except (FetchError, QuotaExhausted, ValueError) as exc:
            report.steps["refresh"] = {"degraded": True, "error": f"{type(exc).__name__}: {exc}"}
            report.flags.append({"kind": "refresh_degraded",
                                 "detail": f"scoring from cached data — {type(exc).__name__}: {exc}"})
    else:
        report.steps["refresh"] = {"skipped": not refresh or not calendar_tickers}

    report.mark("refresh", started=started)
    # -- 1b. resolve the board clock from positive close-finality evidence ---
    # Refresh may intentionally walk back while ORATS is still publishing the
    # current market-wide file. Do not leave the board stamped with the
    # requested wall-clock date in that case: a stamp is a claim about data.
    if calendar_tickers:
        from engine.calendar import trading_calendar
        from engine.data.finality import resolve_final_session

        try:
            finality = resolve_final_session(
                requested_as_of, calendar_tickers, calendar=trading_calendar()
            )
        except RuntimeError as exc:
            report.stopped = "finality"
            report.flags.append({"kind": "session_not_final", "detail": str(exc)})
            _write_flag_report(requested_as_of, report.flags, report.steps)
            raise NightlyStop("finality", str(exc)) from exc
        as_of = pd.Timestamp(finality.date).normalize()
        report.as_of = str(as_of.date())
        report.resolved_as_of = str(as_of.date())
        report.finality = finality.as_dict()
        report.steps["finality"] = report.finality
        if as_of != requested_as_of:
            detail = (
                f"requested {requested_as_of.date()} resolved to final "
                f"{as_of.date()}: {finality.detail}"
            )
            report.flags.append({"kind": "as_of_resolved", "detail": detail})
            if require_as_of:
                report.stopped = "finality"
                _write_flag_report(as_of, report.flags, report.steps)
                raise NightlyStop("finality", detail)
    else:
        report.resolved_as_of = str(as_of.date())

    # -- 2. validation battery — red stops the pipeline ----------------------
    # Validate what will actually be scored (falling back to the frontier set
    # when nothing is upcoming yet, so a blind store still gets checked).
    scoring_tickers = sorted(upcoming["ticker"].unique().tolist())
    validate_tickers = scoring_tickers or calendar_tickers
    checks = validate_refresh(validate_tickers, as_of, max_staleness_days=max_staleness_days) \
        if validate_tickers else [{"name": "no_calendar", "passed": True, "detail": "no tickers to validate"}]
    report.steps["validate"] = checks
    red = [c for c in checks if not c["passed"]]
    if red:
        detail = "; ".join(f"{c['name']}: {c['detail']}" for c in red)
        report.stopped = "validate"
        report.flags.append({"kind": "validation_red", "detail": detail})
        _write_flag_report(as_of, report.flags, report.steps)
        raise NightlyStop("validate", detail)

    report.mark("validate", started=started)
    # -- 2b. Tier 3 and Tier 4 ------------------------------------------------
    # The panel is a deterministic function of Tier 2 and Tier 4 is a
    # deterministic function of the panel, so a refresh that moves Tier 2 and
    # stops leaves BOTH stale — and staleness there is not cosmetic. Every
    # forecast-sized structure reads Tier 4, and a model that has to be SERVED
    # for a forward event needs panel columns that only a rebuild produces.
    # Found 2026-09-05 with the panel three days behind Tier 2: 161 events had
    # printed and were in neither tier.
    #
    # Degrades rather than stops. A stale panel scores yesterday's universe,
    # which is wrong but visible; a nightly that refuses to render leaves the
    # board dark, which is worse and less visible.
    if refresh and tiers:
        from engine.data.rebuild import rebuild as rebuild_tables

        try:
            tier_result = rebuild_tables(("panel", "tier4"))
            report.steps["tiers"] = {
                "rebuilt": ["panel", "tier4"],
                "elapsed_s": getattr(tier_result, "elapsed_s", None),
            }
        except Exception as exc:
            report.steps["tiers"] = {"degraded": True,
                                     "error": f"{type(exc).__name__}: {exc}"[:300]}
            report.flags.append({
                "kind": "tiers_degraded",
                "detail": ("Tier 3/Tier 4 not rebuilt — scoring from the stored panel and "
                           f"forecasts, which may not cover recent prints. {type(exc).__name__}: {exc}")[:300],
            })

    # -- earnings-date changes (needs the refreshed calendar) ----------------
    # Only against a PREVIOUS run's calendar: with no prior state every event
    # is trivially "added", and a first-night flag listing the whole board as
    # new is noise that teaches the reader to ignore the flag.
    if state.get("calendar") is not None:
        change_flag = _date_change_flag(state["calendar"], upcoming, as_of=as_of)
        if change_flag:
            report.flags.append(change_flag)

    report.mark("tiers", started=started)
    # -- 3. score -------------------------------------------------------------
    # The board is scored ATM-only; the strike ladder comes after, for the rows
    # the gate passed (see :func:`strike_ladder`).
    if scorer is None:
        # "Scorer()" defaults to a full daily_market load. That is roughly
        # 9m rows, while the board needs live state only for its current
        # calendar names, so keeping this context narrow is what makes the
        # nightly scorer fit alongside the Tier-3 and Tier-4 rebuild outputs.
        #
        # This used to say the analog table "can fall back to its event-level
        # implied move" for entry dates outside the slice. It no longer can:
        # that fallback was reached by 98.48% of analog trades and silently
        # made the board's analog block a function of how wide this context
        # happened to be. `Scorer` now reads those quotes from the trades'
        # own span, so narrowing here costs live state only — which is the
        # only thing it was ever supposed to bound.
        from engine.features import FeatureContext

        context_tickers = set(scoring_tickers or calendar_tickers)
        # Backfill is a historical re-score, not a reason to construct a
        # second default Scorer. Include its possible events in this one
        # bounded daily slice so the existing engine can score those nights.
        # The old loop called ledger.snapshot without scores and loaded the
        # complete daily_market table while the current board was still held.
        if backfill and state.get("last_successful_as_of"):
            backfill_start = pd.Timestamp(state["last_successful_as_of"]).normalize()
            backfill_events = events[
                (events["event_date"] >= backfill_start)
                & (events["event_date"] <= horizon)
                & events["session"].notna()
            ]
            if tickers is not None:
                backfill_events = backfill_events[
                    backfill_events["ticker"].isin(set(tickers))
                ]
            context_tickers.update(backfill_events["ticker"].dropna().astype(str))
        context_years = range(as_of.year - 1, horizon.year + 1)
        context = FeatureContext.load(sorted(context_tickers), years=context_years)
        engine = Scorer(context=context)
    else:
        engine = scorer
    scores = score_calendar(
        as_of, horizon_days=horizon_days, alt_strikes=0,
        scorer=engine, tickers=tickers,
        # Finer than score_calendar's own default (50): the nightly board
        # spans hundreds of events x every live strategy, and 50-event
        # granularity gave too few checkpoints to tell "slow" from "stuck"
        # over a run that can now take much longer than it used to as
        # strategies were added.
        progress_every=10,
    )
    board_scores = scores

    report.mark("score", started=started)
    # -- 4. ledger, BEFORE rendering — the frozen record is the point --------
    # The ledger records the ATM board only: the ladder rows are EXTRAPOLATED
    # views of the same decision, and freezing them would inflate the
    # calibration sample with rows nobody would trade.
    report.steps["ledger"] = ledger.snapshot(
        as_of=as_of, scores=board_scores, finality=report.finality
    )

    report.mark("ledger", started=started)
    # -- 4a. SETTLE what has already happened --------------------------------
    # The nightly wrote predictions for six nights and settled none of them,
    # because nothing ever called this. A frozen prediction with no outcome is
    # a record of an opinion, not of a result: the calibration report, the
    # health page and the hypothetical book all read `scored_pairs()`, and all
    # three were empty for the same reason.
    #
    # It runs AFTER the refresh, so a chain acquired tonight settles tonight
    # rather than waiting a further day.
    try:
        report.steps["settle"] = ledger.score_outcomes()
    except Exception as exc:  # noqa: BLE001 — a settlement failure must not
        # take the board down; the predictions are already frozen and the
        # outcomes can be scored again tomorrow.
        report.steps["settle"] = {"failed": f"{type(exc).__name__}: {exc}"}
        report.flags.append({
            "kind": "settle_failed",
            "detail": f"outcomes not scored tonight — {type(exc).__name__}: {exc}",
        })

    report.mark("settle", started=started)
    # -- 3b. the strike ladder, for the explorer -------------------------------
    ladder = strike_ladder(
        board_scores, scorer=engine, alt_strikes=alt_strikes, as_of=as_of
    )
    if ladder:
        scores = pd.concat([scores, pd.DataFrame(ladder)], ignore_index=True)
    report.steps["score"] = {
        "board_rows": int(len(board_scores)),
        "ladder_rows": len(ladder),
        "alt_strikes": alt_strikes,
        # Share of the analog population bucketed on a real entry-date implied
        # move rather than the event-level fallback. Recorded because its
        # collapse — to 1.5%, when the scoring context was narrowed — moved
        # published analog numbers for weeks with every guard still green.
        # A run well below ~0.95 means the trades' surface rows went missing,
        # not that the board changed.
        "analog_entry_coverage": engine.analog_entry_coverage,
    }

    report.mark("ladder", started=started)
    # -- 4b. honest backfill of missed nights --------------------------------
    late_as_ofs: list[str] = []
    skipped_closed: list[str] = []
    if backfill:
        last = state.get("last_successful_as_of")
        if last:
            nights, skipped_closed, day = _nights_to_backfill(
                last, as_of, calendar=engine.calendar
            )
            for night in nights:
                backfill_scores = score_calendar(
                    night,
                    horizon_days=horizon_days,
                    alt_strikes=0,
                    scorer=engine,
                    tickers=tickers,
                    progress_every=10,
                )
                result = ledger.snapshot(
                    as_of=night,
                    horizon_days=horizon_days,
                    scores=backfill_scores,
                )
                late_as_ofs.append(
                    {"as_of": str(night.date()), "rows": result.get("rows", 0)}
                )
            if skipped_closed:
                report.flags.append({
                    "kind": "backfill_skipped_closed",
                    "detail": "market closed — no row can enter on these dates",
                    "dates": skipped_closed,
                })
            if day < as_of:
                report.flags.append({
                    "kind": "backfill_gap",
                    "detail": f"missed nights remain before {as_of.date()} — run nightly per missed date",
                })
            if late_as_ofs:
                report.flags.append({
                    "kind": "late_backfill",
                    "detail": "rows written after their decision date (decision_ts is honest)",
                    "as_ofs": late_as_ofs,
                })
        report.steps["backfill"] = {
            "scored": late_as_ofs, "skipped_closed": skipped_closed,
        }

    # -- flags that need scores ----------------------------------------------
    triggered = sorted(
        f"{r.ticker}|{r.strategy}|{r.event_date}"
        for r in board_scores[board_scores["gate_pass"].fillna(False).astype(bool)].itertuples(index=False)
    ) if len(board_scores) else []
    previously = set(state.get("gate_triggers") or [])
    new_triggers = sorted(set(triggered) - previously)
    if new_triggers:
        report.flags.append({"kind": "new_gate_triggers", "rows": new_triggers})

    conflict_flag = _date_conflict_flag(upcoming)
    if conflict_flag:
        report.flags.append(conflict_flag)

    quota_flag = _quota_flag()
    if quota_flag:
        report.flags.append(quota_flag)
    calib_flag = _calibration_flag()
    if calib_flag:
        report.flags.append(calib_flag)

    report.mark("backfill", started=started)
    # -- 4c. model evidence — rebuilt only when a champion changed -----------
    # render copies data/features/model_evidence.json into the bundle verbatim,
    # and the file is keyed by the champions' artifact fingerprint. Nothing
    # else in the pipeline rebuilt it: after the 2026-09-06 gate promotion the
    # dashboard kept serving the Sep-5 table — the new champion was absent and
    # the STR-THRU gate showed the incumbent's inputs instead of the registered
    # set. The fingerprint check makes this free on nights nothing changed; a
    # failed rebuild degrades to the cached table and raises a flag rather than
    # taking the board down.
    try:
        from engine.dashboard.model_evidence import build_model_evidence

        evidence = build_model_evidence(registry=engine.registry)
        report.steps["model_evidence"] = {
            "generated_at": evidence.get("generated_at"),
            "models": sorted((evidence.get("models") or {}).keys()),
            # The artifact's OWN elapsed_s, which is how long whichever run
            # last rebuilt it took — NOT this run. On a night the fingerprint
            # matches, nothing is rebuilt and this number is someone else's:
            # it read 1,636s in a 721s run on 2026-09-09. Named for what it is;
            # tonight's cost is in `timeline`.
            "cached_build_elapsed_s": evidence.get("elapsed_s"),
        }
    except Exception as exc:  # noqa: BLE001 — stale evidence beats a dark board
        report.steps["model_evidence"] = {
            "degraded": True, "error": f"{type(exc).__name__}: {exc}"[:300],
        }
        report.flags.append({
            "kind": "model_evidence_stale",
            "detail": ("evidence rebuild failed; the bundle carries the cached "
                       f"table, which may predate the current champions — "
                       f"{type(exc).__name__}: {exc}")[:300],
        })

    report.mark("model_evidence", started=started)
    # -- 5. render -------------------------------------------------------------
    meta = build_meta(
        scores,
        as_of=as_of,
        horizon_days=horizon_days,
        fill_alpha=float(board_scores["fill"].iloc[0]) if len(board_scores) else 0.5,
        alt_strikes=alt_strikes,
        freshness=freshness_summary(as_of),
        quota=quota_state(),
        late_as_ofs=[x["as_of"] for x in late_as_ofs],
        registry=engine.registry,
    )
    meta["execution_clock"] = {
        "requested_as_of": str(requested_as_of.date()),
        "resolved_as_of": str(as_of.date()),
        "finality": report.finality,
    }
    meta["cron"] = {
        "entry": f"30 21 * * 1-5  cd {paths.ROOT.name} && python3 -m engine.dashboard.nightly "
                 ">> dashboard/nightly.log 2>&1",
        "note": "verify the daemon with `service cron status` — on WSL2 it does not "
                "start by default, and a cron that never ran looks exactly like a job "
                "with nothing to say. The job is idempotent and backfills missed "
                "nights on the next run. Full entry in dashboard/README.md.",
    }
    health = build_health(
        as_of=as_of,
        selfcheck_report=state.get("last_selfcheck"),
        size_mae=size_model_mae_from_ledger(panel=engine.context.panel),
    )
    render_summary = render_bundle(
        scores,
        bundle_dir,
        as_of=as_of,
        horizon_days=horizon_days,
        fill_alpha=meta["fill_alpha"],
        alt_strikes=alt_strikes,
        panel=engine.context.panel,
        trades=engine.trades,
        meta=meta,
        health=health,
        flags=report.flags,
        registry=engine.registry,
    )
    report.steps["render"] = render_summary

    report.mark("render", started=started)
    # -- 5b. selfcheck — any mismatch stops the publish -----------------------
    check = selfcheck(bundle_dir, scorer=engine)
    report.steps["selfcheck"] = check.as_dict()
    if not check.ok:
        report.stopped = "selfcheck"
        report.flags.append({"kind": "selfcheck_red", "detail": check.detail,
                             "mismatches": check.mismatches[:5]})
        _write_flag_report(as_of, report.flags, report.steps)
        raise NightlyStop("selfcheck", check.detail)

    report.mark("selfcheck", started=started)
    # -- 6. publish atomically -------------------------------------------------
    if publish:
        try:
            result = publish_bundle(bundle_dir, target, probe_url=probe_url)
            report.steps["publish"] = result.as_dict()
        except PublishError as exc:
            report.steps["publish"] = {"failed": str(exc)}
            report.flags.append({"kind": "publish_failed", "detail": str(exc)})
    else:
        report.steps["publish"] = {"skipped": True}

    report.mark("publish", started=started)
    # -- 7. persist flags + state ----------------------------------------------
    flag_path = _write_flag_report(as_of, report.flags, report.steps)
    report.steps["flags"] = {"path": str(flag_path), "count": len(report.flags)}
    _write_state(bundle_dir, {
        "last_successful_as_of": str(as_of.date()),
        "calendar": _upcoming_event_rows(upcoming),
        "gate_triggers": triggered,
        "last_selfcheck": check.as_dict(),
    })

    report.mark("flags", started=started)
    # -- 8. backup sync — failures flag, never block ----------------------------
    if backup:
        report.steps["backup"] = _backup_sync(report.flags)
    else:
        report.steps["backup"] = {"skipped": True}

    report.mark("backup", started=started)
    report.elapsed_s = time.time() - started
    return report


def _backup_sync(flags: list) -> dict:
    """Push code to the public repo and mirror irreplaceables to the private one.

    Never raises: a dead remote must cost a flag, not a snapshot. The hygiene
    hook runs inside the git push path (pre-commit), exactly as the plan pins.
    """
    out: dict = {}
    try:
        push = subprocess.run(
            ["git", "push", "origin", "HEAD"],
            cwd=str(paths.ROOT), capture_output=True, text=True, timeout=300,
        )
        out["git_push"] = {"ok": push.returncode == 0,
                           "detail": (push.stderr or push.stdout)[-200:].strip()}
        if push.returncode != 0:
            flags.append({"kind": "backup_failed", "detail": out["git_push"]["detail"]})
    except Exception as exc:
        out["git_push"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        flags.append({"kind": "backup_failed", "detail": out["git_push"]["detail"]})

    mirror = paths.TOOLS / "private_mirror.py"
    if mirror.exists():
        try:
            proc = subprocess.run(
                [sys.executable, str(mirror), "--push"],
                cwd=str(paths.ROOT), capture_output=True, text=True, timeout=600,
            )
            out["private_mirror"] = {"ok": proc.returncode == 0,
                                     "detail": (proc.stderr or proc.stdout)[-200:].strip()}
            if proc.returncode != 0:
                flags.append({"kind": "backup_failed", "detail": out["private_mirror"]["detail"]})
        except Exception as exc:
            out["private_mirror"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
            flags.append({"kind": "backup_failed", "detail": out["private_mirror"]["detail"]})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--require-as-of", action="store_true",
                        help="fail instead of resolving --as-of back to the newest final session")
    parser.add_argument("--horizon", type=int, default=HORIZON_DAYS,
                        help="calendar days of prints to score; the default is "
                             "set by STR-RUNUP's 14-trading-day entry")
    parser.add_argument("--alt-strikes", type=int, default=1,
                        help="strikes either side of ATM, scored for gate passers only")
    parser.add_argument("--tickers", default=None, help="comma-separated restriction")
    parser.add_argument("--bundle", default=None, help="bundle dir (default dashboard/earnings)")
    parser.add_argument("--target", default=None, help="publish target dir")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--no-tiers", action="store_true",
                        help="skip the Tier-3 / Tier-4 rebuild (they follow a refresh by default)")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--no-backfill", action="store_true")
    parser.add_argument("--backup", action="store_true", help="run the git + private-mirror sync")
    parser.add_argument(
        "--chain-sessions", type=int, default=CHAIN_SESSIONS,
        help="trading sessions of chains to refresh, counting back from --as-of. "
             "2 is the steady state (ORATS publishes around midnight, so tonight "
             "needs yesterday too); widen it once to heal a gap left by nights "
             "that did not run. Already-held pairs are skipped, so a wider "
             "window costs only what is genuinely missing.",
    )
    parser.add_argument("--max-staleness", type=int, default=MAX_STALENESS_DAYS)
    parser.add_argument("--json", default=None, help="write the run report to this path")
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    try:
        # Around the CLI, not around `run_nightly`: this is where the two
        # overlapping callers actually are — the cron and a hand-run, plus the
        # desk app's `POST /api/refresh`, which shells out to exactly this
        # entry point. Tests and other in-process callers use `run_nightly`
        # directly and are deliberately unaffected.
        with single_run_lock():
            report = run_nightly(
                args.as_of,
                horizon_days=args.horizon,
                alt_strikes=args.alt_strikes,
                tickers=tickers,
                bundle_dir=args.bundle,
                target=args.target,
                refresh=not args.no_refresh,
                tiers=not args.no_tiers,
                publish=not args.no_publish,
                backup=args.backup,
                backfill=not args.no_backfill,
                chain_sessions=args.chain_sessions,
                max_staleness_days=args.max_staleness,
                require_as_of=args.require_as_of,
            )
    except NightlyStop as exc:
        print(f"NIGHTLY STOPPED — {exc}", file=sys.stderr)
        return 1

    text = json.dumps(report.as_dict(), indent=1, default=str)
    print(text)
    if args.json:
        Path(args.json).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
