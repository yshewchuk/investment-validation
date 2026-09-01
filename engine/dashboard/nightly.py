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

__all__ = [
    "NightlyStop", "NightlyReport", "run_nightly", "refresh_calendar_data",
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
    steps: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    stopped: str | None = None
    elapsed_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "steps": self.steps,
            "flags": self.flags,
            "stopped": self.stopped,
            "elapsed_s": round(self.elapsed_s, 1),
        }


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
CHAIN_DTE = "1,45"


def refresh_forward_chains(tickers, as_of, *, fetcher=None, batch: int = CHAIN_BATCH) -> dict:
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
    """
    from engine.data.fetch import Fetcher

    fetcher = fetcher or Fetcher()
    stamp = str(pd.Timestamp(as_of).normalize().date())
    unknown = _unknown_symbols()
    wanted = [str(t) for t in sorted(set(tickers)) if str(t) not in unknown]

    out = {"as_of": stamp, "requested": len(wanted), "returned": 0, "rows": 0,
           "calls": 0, "cache_hits": 0, "missing": [], "failed": []}
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
    return out


def refresh_calendar_data(
    tickers: Sequence[str],
    as_of,
    *,
    fetcher=None,
    batch: int = 10,
    horizon_days: int = 21,
    forward: bool = True,
    max_confirmations: int = 400,
) -> dict:
    """Refresh what the board scores from, cheapest and most load-bearing first.

    Three passes, and the order reflects which resource each one spends:

    1. **The forward calendar** (unmetered: Nasdaq + yfinance). This is the one
       that decides whether the board has any rows at all, and it costs no
       ORATS quota — see :mod:`engine.data.pulls.forward_calendar`.
    2. **Summaries and cores** for the as-of date: ONE ORATS call each, for the
       whole market, via ``tradeDate``. Spot, IVs and implied moves.
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
    for endpoint in ("hist/summaries", "hist/cores"):
        record = fetcher.fetch(
            "orats", endpoint, {"tradeDate": str(as_of.date())},
            live=True, note="phase3 nightly refresh",
        )
        out["calls"] += 0 if record.from_cache else 1
        out["cache_hits"] += 1 if record.from_cache else 0
        try:
            rows = record.json().get("data") or []
        except (ValueError, AttributeError):
            rows = []
        out["endpoints"][endpoint] = {"rows": len(rows)}

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
    out["chains"] = refresh_forward_chains(tickers, as_of, fetcher=fetcher)
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


#: Alternative strikes are offered as fractions of spot either side of ATM —
#: the same ±2.5% steps :func:`engine.score.score_calendar` uses.
STRIKE_STEP = 0.025


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
    from engine.score import UNSCORABLE, ScoreRequest, unscorable_result

    live = board[board["gate_pass"].fillna(False).astype(bool)]
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
                strike=float(spot) * (1 + offset),
                fill=FillModel(float(record.get("fill", 0.5))),
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


def run_nightly(
    as_of=None,
    *,
    horizon_days: int = 21,
    alt_strikes: int = 1,
    tickers: Iterable[str] | None = None,
    bundle_dir: Path | str | None = None,
    target: Path | str | None = None,
    refresh: bool = True,
    publish: bool = True,
    backup: bool = False,
    backfill: bool = True,
    fetcher=None,
    scorer=None,
    max_staleness_days: int = MAX_STALENESS_DAYS,
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
    bundle_dir = Path(bundle_dir) if bundle_dir is not None else paths.ROOT / "dashboard" / "earnings"
    target = target if target is not None else _default_target()
    probe_url = os.environ.get("DASHBOARD_PROBE_URL") or None

    report = NightlyReport(as_of=str(as_of.date()))
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

    # -- 1. refresh -----------------------------------------------------------
    if refresh and calendar_tickers:
        from engine.data.fetch import CredentialRotated, FetchError
        from engine.data.throttle import QuotaExhausted

        try:
            report.steps["refresh"] = refresh_calendar_data(
                calendar_tickers, as_of, fetcher=fetcher, horizon_days=horizon_days
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

    # -- earnings-date changes (needs the refreshed calendar) ----------------
    # Only against a PREVIOUS run's calendar: with no prior state every event
    # is trivially "added", and a first-night flag listing the whole board as
    # new is noise that teaches the reader to ignore the flag.
    if state.get("calendar") is not None:
        change_flag = _date_change_flag(state["calendar"], upcoming, as_of=as_of)
        if change_flag:
            report.flags.append(change_flag)

    # -- 3. score -------------------------------------------------------------
    # The board is scored ATM-only; the strike ladder comes after, for the rows
    # the gate passed (see :func:`strike_ladder`).
    engine = scorer or Scorer()
    scores = score_calendar(
        as_of, horizon_days=horizon_days, alt_strikes=0,
        scorer=engine, tickers=tickers,
    )
    board_scores = scores

    # -- 4. ledger, BEFORE rendering — the frozen record is the point --------
    # The ledger records the ATM board only: the ladder rows are EXTRAPOLATED
    # views of the same decision, and freezing them would inflate the
    # calibration sample with rows nobody would trade.
    report.steps["ledger"] = ledger.snapshot(as_of=as_of, scores=board_scores)

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
    }

    # -- 4b. honest backfill of missed nights --------------------------------
    late_as_ofs: list[str] = []
    if backfill:
        last = state.get("last_successful_as_of")
        if last:
            day = pd.Timestamp(last).normalize() + pd.Timedelta(days=1)
            while day < as_of and len(late_as_ofs) < MAX_BACKFILL_DAYS:
                result = ledger.snapshot(as_of=day, horizon_days=horizon_days)
                late_as_ofs.append({"as_of": str(day.date()), "rows": result.get("rows", 0)})
                day += pd.Timedelta(days=1)
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
        report.steps["backfill"] = late_as_ofs

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

    # -- 5b. selfcheck — any mismatch stops the publish -----------------------
    check = selfcheck(bundle_dir, scorer=engine)
    report.steps["selfcheck"] = check.as_dict()
    if not check.ok:
        report.stopped = "selfcheck"
        report.flags.append({"kind": "selfcheck_red", "detail": check.detail,
                             "mismatches": check.mismatches[:5]})
        _write_flag_report(as_of, report.flags, report.steps)
        raise NightlyStop("selfcheck", check.detail)

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

    # -- 7. persist flags + state ----------------------------------------------
    flag_path = _write_flag_report(as_of, report.flags, report.steps)
    report.steps["flags"] = {"path": str(flag_path), "count": len(report.flags)}
    _write_state(bundle_dir, {
        "last_successful_as_of": str(as_of.date()),
        "calendar": _upcoming_event_rows(upcoming),
        "gate_triggers": triggered,
        "last_selfcheck": check.as_dict(),
    })

    # -- 8. backup sync — failures flag, never block ----------------------------
    if backup:
        report.steps["backup"] = _backup_sync(report.flags)
    else:
        report.steps["backup"] = {"skipped": True}

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
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--alt-strikes", type=int, default=1,
                        help="strikes either side of ATM, scored for gate passers only")
    parser.add_argument("--tickers", default=None, help="comma-separated restriction")
    parser.add_argument("--bundle", default=None, help="bundle dir (default dashboard/earnings)")
    parser.add_argument("--target", default=None, help="publish target dir")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--no-backfill", action="store_true")
    parser.add_argument("--backup", action="store_true", help="run the git + private-mirror sync")
    parser.add_argument("--max-staleness", type=int, default=MAX_STALENESS_DAYS)
    parser.add_argument("--json", default=None, help="write the run report to this path")
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    try:
        report = run_nightly(
            args.as_of,
            horizon_days=args.horizon,
            alt_strikes=args.alt_strikes,
            tickers=tickers,
            bundle_dir=args.bundle,
            target=args.target,
            refresh=not args.no_refresh,
            publish=not args.no_publish,
            backup=args.backup,
            backfill=not args.no_backfill,
            max_staleness_days=args.max_staleness,
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
