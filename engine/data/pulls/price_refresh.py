#!/usr/bin/env python3
"""Scheduled full-history yfinance price re-downloader (price data only).

Feeds the Tier-2 ``price_history`` table another agent is building. Two
refresh cadences, per the 2026-09-14 decision:

* **Daily** — every ticker on the board whose earnings event falls inside a
  window running from the board's forward horizon before the print to 5
  trading sessions after it. "The board" is never a stored list: both
  ``score_calendar`` (``engine/score.py:2988-3033``, the dashboard's own
  input) and the real nightly runner (``engine/dashboard/nightly.py:1144-
  1172``, ``upcoming``/``scoring_tickers``) compute it fresh every run from
  Tier-2 ``earnings_events`` — confirmed rows (``session`` not null) with
  ``event_date`` in ``[as_of, as_of + horizon_days]``. ``horizon_days``
  defaults to 35 at the one call site that actually drives the live board
  (``engine/v2/ops/legacy_adapter.py:115-119`` ``invoke_score_calendar`` →
  ``score_calendar``), which is where :data:`HORIZON_DAYS` below comes from.
  A ticker with an event outside that pre-print window, or none at all, is
  simply outside the "daily" bucket for that one session — it is not
  special-cased, it falls straight through to monthly below.
* **Monthly** — every other ticker in the price universe (the px tree plus
  every ticker the Tier-1 store has ever fetched yfinance history for; see
  :func:`load_price_universe`). Refreshed once no successful dated fetch
  exists for it in the current calendar month, which is what makes a run
  resumable without a separate state file.

Daily resumability is keyed on **coverage**, not on fetch arrival: a daily
ticker is skipped only when some cached fetch's BODY already holds a VALID bar
for the session's (or, for a weekend/holiday session, the last trading day's)
date — see :func:`load_fetch_coverage`, which borrows capture's own row
validity instead of running a second rule that could drift from it. The
``fetched_at`` date alone is not coverage: a pre-market fetch lands today but
its last bar is yesterday's close, and counting it as covering today would
skip the very session this pull exists to capture (the 2026-10-12 defect).
And when the plan *does* ask
for the session, :func:`run_refresh` gives each daily call a fresh
``refetch_token`` — today's cached body is a known-stale answer, so the token
guarantees a real network call rather than a same-day cache hit. Monthly
resumability keeps using ``fetched_at`` dates, which is the right key for "did
we refresh this month at all".

Delisted or otherwise failing tickers are **never pruned** from either group
— they stay in the universe and every run's report lists them, so a human
decides, not this pull.

**Dated-key pattern**, copied from ``engine.data.pulls.computed_moves``
(``computed_moves.py:140-152``, ``fetch_history``): a plain
``Fetcher().fetch("yfinance", "history", {"ticker": T, "period": "max"})``
call would be cached forever on the FIRST response ever seen for ``T``,
because the cache key does not vary. Passing ``live=True`` puts local
``date.today()`` into the key (``engine/data/fetch.py:104-113``), so a
same-ticker fetch on a later day is a genuinely new Tier-1 body instead of a
cache hit. This pull's whole reason to exist is to notice new closes, so
``live=True`` is load-bearing here exactly as it is there. The existing
UNDATED entry some callers still read (``panel.py:444-474``) is never
touched — this module only ever writes dated keys.

:func:`plan_refresh` is pure — no I/O, no network, no clock reads beyond the
``session`` argument — and takes already-loaded ``events``/``price_universe``/
``fetch_history``/``fetch_coverage`` so it can be tested without a filesystem.
The CLI (``ops price-refresh``) is what actually reads those from disk, via
the ``load_*`` helpers below, and calls :func:`run_refresh` against a real
:class:`~engine.data.fetch.Fetcher`.
"""
from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping

import pandas as pd

from engine import calendar as engine_calendar
from engine import paths
from engine.data.fetch import Fetcher, iter_cached

__all__ = [
    "HORIZON_DAYS",
    "POST_EVENT_TRADING_SESSIONS",
    "plan_refresh",
    "run_refresh",
    "fetch_one",
    "load_events",
    "load_price_universe",
    "load_fetch_history",
    "load_fetch_coverage",
]

#: Calendar days from the board's session forward that still count as "about
#: to print" — the scorer's own horizon (``engine/v2/ops/legacy_adapter.py``
#: ``invoke_score_calendar``'s ``horizon_days=35`` default, which is what
#: ``score_calendar``/the live board actually run with).
HORIZON_DAYS = 35

#: Trading sessions after a print a ticker stays in the daily group, so the
#: newly-realized close and the first couple of post-earnings sessions are
#: captured before falling back to monthly cadence.
POST_EVENT_TRADING_SESSIONS = 5


# --------------------------------------------------------------------------
# window arithmetic
# --------------------------------------------------------------------------


def _nth_trading_session_after(event_date: pd.Timestamp, n: int) -> pd.Timestamp:
    """The n-th trading session strictly after ``event_date``, holiday-aware.

    ``engine.calendar.projected_trading_days(start, end)`` returns weekdays in
    ``(start, end]`` with scheduled NYSE holidays removed (``engine/calendar.py
    :176-190``). The end is padded generously — worst case a run of six
    holidays/weekend days in a row, which never happens, so ``n*3 + 14``
    calendar days of padding always yields at least ``n`` trading days.
    """
    end = event_date + pd.Timedelta(days=n * 3 + 14)
    days = engine_calendar.projected_trading_days(event_date, end)
    return days[n - 1]


def in_daily_window(
    session: pd.Timestamp,
    event_date: pd.Timestamp,
    *,
    horizon_days: int = HORIZON_DAYS,
    post_event_sessions: int = POST_EVENT_TRADING_SESSIONS,
) -> bool:
    """True if ``session`` is inside the daily-refresh window for one event.

    Two disjoint legs, both inclusive at their far edge:

    * pre-print: ``session <= event_date <= session + horizon_days`` calendar
      days — the ticker is "about to print" on this session.
    * post-print: ``event_date < session``, and ``session`` is on or before
      the ``post_event_sessions``-th trading session after ``event_date``.
    """
    if session <= event_date <= session + pd.Timedelta(days=horizon_days):
        return True
    if event_date < session:
        cutoff = _nth_trading_session_after(event_date, post_event_sessions)
        if session <= cutoff:
            return True
    return False


# --------------------------------------------------------------------------
# plan_refresh — pure
# --------------------------------------------------------------------------


#: Calendar days scanned back for "the latest trading day on or before the
#: session". The longest realistic market closure is a three-day weekend, so
#: ten days always contains at least one candidate unless the calendar itself
#: is broken — and then the session is the requirement rather than every
#: on-board ticker silently skipping.
_DAILY_REQUIRED_LOOKBACK_DAYS = 10


def _required_daily_bar_date(session) -> pd.Timestamp:
    """The DATA date a daily fetch must already contain to skip this session.

    The latest trading day on or before ``session``: itself when it is a
    trading day, the Friday before a Monday holiday, the Friday before a
    weekend. Discovered with ``engine_calendar.projected_trading_days`` over a
    short window ending at ``session`` (which returns weekdays in
    ``(start, end]``, so the session itself is included and every candidate is
    ``<= session``); with no trading day in that window — which cannot happen
    for a real market calendar — the session is the requirement rather than
    silently skipping everything.
    """
    session_ts = pd.Timestamp(session).normalize()
    days = engine_calendar.projected_trading_days(
        session_ts - pd.Timedelta(days=_DAILY_REQUIRED_LOOKBACK_DAYS), session_ts)
    if len(days) == 0:
        return session_ts
    return days[-1]


def plan_refresh(
    session,
    *,
    events: pd.DataFrame,
    price_universe: Iterable[str],
    fetch_history: Mapping[str, Iterable] | None = None,
    fetch_coverage: Mapping[str, Iterable] | None = None,
    horizon_days: int = HORIZON_DAYS,
    post_event_sessions: int = POST_EVENT_TRADING_SESSIONS,
    always_daily: Iterable[str] = (),
) -> dict:
    """Pure planning: which tickers get fetched today, which wait for their
    monthly turn, and which are skipped because a run already covered them.

    Parameters
    ----------
    session:
        The trading day this plan is for (``date``/``str``/``Timestamp``).
    events:
        Confirmed earnings events, columns ``ticker``/``event_date`` (Tier-2
        ``earnings_events``, typically pre-filtered to ``session.notna()`` by
        the caller — see :func:`load_events`). Not restricted to any date
        range: both future prints (pre-print leg) and past ones (post-print
        trailing leg) are needed for the window test above.
    price_universe:
        Every OTHER ticker price history should ever cover — tickers with no
        row in ``events`` still land in the monthly group via this. A ticker
        that appears only in ``events`` (e.g. brand new, never yet fetched)
        is still covered: the daily/monthly universe is
        ``set(price_universe) | set(events["ticker"])``.
    fetch_history:
        ``{ticker: [date, ...]}`` — every date a dated Tier-1 fetch already
        landed for that ticker, keyed on the fetch's own ``fetched_at`` (see
        :func:`load_fetch_history`). Governs MONTHLY resumability only: a
        monthly ticker already fetched somewhere in ``session``'s calendar
        month is skipped. The daily group reads :func:`load_fetch_coverage`
        instead, because ``fetched_at`` says when bytes arrived, not which
        bars they contain. Omit (or ``{}``) for a from-scratch plan.
    fetch_coverage:
        ``{ticker: [date, ...]}`` — per dated Tier-1 fetch, the newest VALID
        BAR date its body actually holds (see :func:`load_fetch_coverage`;
        valid means valid to capture's own reader, and these are DATA dates,
        never ``fetched_at``). Governs DAILY resumability: a
        ticker on the board is skipped only if some fetch covers
        ``_required_daily_bar_date`` (the latest trading day on or before
        ``session``) or is newer; anything else — an old body, a corrupt one,
        or no fetch at all — goes on today's daily list. A pre-market fetch
        whose last bar is the prior session does NOT excuse the session.
    always_daily:
        Tickers that are always on the daily schedule regardless of events,
        e.g. the native calendar series. A forced ticker whose cached bars
        already reach ``_required_daily_bar_date`` is skipped like any other
        daily fetch.

    Returns
    -------
    ``{"session": "YYYY-MM-DD", "daily": [...], "monthly": [...],
    "skipped_already_fetched": [...]}`` — each ticker list sorted, and every
    ticker in the combined universe appears in exactly one of the three.
    """
    session_ts = pd.Timestamp(session).normalize()
    history = fetch_history or {}
    coverage = fetch_coverage or {}
    required = _required_daily_bar_date(session_ts)

    events = events.copy()
    if len(events):
        events["event_date"] = pd.to_datetime(events["event_date"]).dt.normalize()
        events["ticker"] = events["ticker"].astype(str)

    events_by_ticker: dict[str, list[pd.Timestamp]] = {}
    for row in events.itertuples():
        events_by_ticker.setdefault(str(row.ticker), []).append(row.event_date)

    forced = {str(t) for t in always_daily}
    universe = {str(t) for t in price_universe} | set(events_by_ticker) | forced

    daily: list[str] = []
    monthly: list[str] = []
    skipped: list[str] = []

    for ticker in sorted(universe):
        on_board = ticker in forced or any(
            in_daily_window(session_ts, event_date, horizon_days=horizon_days,
                            post_event_sessions=post_event_sessions)
            for event_date in events_by_ticker.get(ticker, ())
        )
        if on_board:
            covered = any(
                pd.Timestamp(d).normalize() >= required for d in coverage.get(ticker, ())
            )
            if covered:
                skipped.append(ticker)
            else:
                daily.append(ticker)
        else:
            already = {pd.Timestamp(d).normalize() for d in history.get(ticker, ())}
            fetched_this_month = any(
                d.year == session_ts.year and d.month == session_ts.month for d in already
            )
            if fetched_this_month:
                skipped.append(ticker)
            else:
                monthly.append(ticker)

    return {
        "session": str(session_ts.date()),
        "daily": daily,
        "monthly": monthly,
        "skipped_already_fetched": skipped,
    }


# --------------------------------------------------------------------------
# run_refresh — the network side
# --------------------------------------------------------------------------


def fetch_one(fetcher: Fetcher, ticker: str, *, refetch_token: str | None = None) -> dict:
    """One ticker's full-history dated fetch. Never raises for an ordinary
    per-ticker failure — a delisted or renamed ticker is a fact about that
    name, not a reason to abort a 200-ticker run (the lesson
    ``computed_moves.py``'s BF_B comment, lines ~150-153, already paid for).

    ``refetch_token`` is passed through to ``fetcher.fetch``: a token makes the
    call a guaranteed cache miss (a distinct Tier-1 entry) when the cached
    answer — e.g. today's pre-market body — is known stale. ``None`` is the
    ordinary cache-first path.

    ``CredentialRotated`` is the one exception and is left to propagate: it
    means every subsequent call will fail identically until a human updates
    ``.env`` (``engine/data/sources/base.py``'s own docstring), so it is a
    whole-run condition, not a per-ticker one.
    """
    from engine.data.sources.base import FetchError

    try:
        rec = fetcher.fetch(
            "yfinance", "history", {"ticker": ticker, "period": "max"},
            live=True, note="price-refresh", refetch_token=refetch_token,
        )
    except (FetchError, OSError, ValueError) as exc:
        return {"ticker": ticker, "ok": False, "error_class": type(exc).__name__}
    if rec is None or rec.status != 200:
        status = getattr(rec, "status", None)
        return {"ticker": ticker, "ok": False, "error_class": f"http_{status}"}
    return {"ticker": ticker, "ok": True}


def run_refresh(plan: Mapping, fetcher: Fetcher) -> dict:
    """Fetch every ticker in ``plan``'s ``daily``/``monthly`` groups.

    Never aborts on a per-ticker failure (see :func:`fetch_one`). Returns
    per-group counts plus a flat list of failures (ticker, group, error
    class — never a payload; a failure never carries fetched bytes).

    The daily group gets a fresh ``refetch_token`` per ticker call: a daily
    ticker is on the list precisely because no body on disk holds the required
    bar, so a same-day cached (typically pre-market) entry must not satisfy the
    request out of cache — the token forces a real network call and a second,
    append-only Tier-1 entry. The monthly group keeps the plain cache-first
    call: its resumability is arrival-based, so an already-answered same-day
    request should be a cache hit, not a re-fetch.
    """
    fetched: dict[str, list[str]] = {"daily": [], "monthly": []}
    failed: list[dict] = []

    for group in ("daily", "monthly"):
        for ticker in plan.get(group, ()):
            token = (
                datetime.now(timezone.utc).isoformat() if group == "daily" else None
            )
            outcome = fetch_one(fetcher, ticker, refetch_token=token)
            if outcome["ok"]:
                fetched[group].append(ticker)
            else:
                failed.append({
                    "ticker": ticker, "group": group,
                    "error_class": outcome["error_class"],
                })

    return {
        "session": plan.get("session"),
        "counts": {
            "daily": len(plan.get("daily", ())),
            "monthly": len(plan.get("monthly", ())),
            "skipped_already_fetched": len(plan.get("skipped_already_fetched", ())),
            "fetched": len(fetched["daily"]) + len(fetched["monthly"]),
            "failed": len(failed),
        },
        "fetched": fetched,
        "failed": failed,
    }


# --------------------------------------------------------------------------
# loaders — real I/O, kept separate so plan_refresh stays pure/injectable
# --------------------------------------------------------------------------


def load_events() -> pd.DataFrame:
    """Every confirmed Tier-2 event: ``ticker``/``event_date``, ``session``
    not null — the same "confirmed" filter ``score_calendar``
    (``engine/score.py:3022-3027``) and the real nightly (``engine/dashboard/
    nightly.py:1144-1150``) both apply. Deliberately unrestricted by date:
    the post-print trailing leg of the window needs past events too.
    """
    from engine.data import store

    events = store.read_table("earnings_events", columns=["ticker", "event_date", "session"])
    return events[events["session"].notna()][["ticker", "event_date"]].reset_index(drop=True)


def load_price_universe(*, yf_dir=None, fetch_root=None) -> set[str]:
    """The px tree (grandfathered, read-only, ``px_{TICKER}.csv`` under
    ``engine.paths.RAW_YF`` — ``engine/data/features/panel.py:495``,
    ``engine/data/validate.py:359``) unioned with every ticker the Tier-1
    store has ever fetched yfinance ``history`` for
    (``engine.data.fetch.iter_cached``, ``engine/data/fetch.py:163-188``).
    Read-only on both sides; nothing here ever writes to ``RAW_YF``.

    ``yf_dir``/``fetch_root`` default to ``engine.paths.RAW_YF``/``RAW_FETCH``
    and exist only so tests can point this at a throwaway tree.
    """
    yf_dir = yf_dir if yf_dir is not None else paths.RAW_YF
    px = set()
    if yf_dir.exists():
        px = {p.name[len("px_"):-len(".csv")] for p in yf_dir.glob("px_*.csv")}
    cached = set()
    for entry in iter_cached("yfinance", "history", root=fetch_root):
        ticker = entry.params.get("ticker")
        if ticker:
            cached.add(str(ticker))
    return px | cached


def load_fetch_history(*, fetch_root=None) -> dict[str, list[date_cls]]:
    """``{ticker: [date, ...]}`` from every dated ``period=max`` Tier-1
    yfinance ``history`` fetch already on disk, keyed by the fetch's own
    ``fetched_at`` (UTC — ``engine/data/fetch.py``'s ``_persist``, the field
    downstream already orders by, per the task's own verified design facts).

    ``fetch_root`` defaults to ``engine.paths.RAW_FETCH`` and exists only so
    tests can point this at a throwaway tree.
    """
    out: dict[str, list[date_cls]] = {}
    for entry in iter_cached("yfinance", "history", root=fetch_root):
        if entry.params.get("period") != "max":
            continue
        ticker = entry.params.get("ticker")
        fetched_at = entry.meta.get("fetched_at")
        if not ticker or not fetched_at:
            continue
        when = pd.Timestamp(fetched_at).normalize().date()
        out.setdefault(str(ticker), []).append(when)
    return out


def load_fetch_coverage(session, *, read_bars: Callable[[bytes], Iterable],
                        fetch_root=None) -> dict[str, list[date_cls]]:
    """``{ticker: [newest_valid_bar_date, ...]}`` — the newest valid bar date
    each dated Tier-1 ``period=max`` yfinance ``history`` fetch on disk
    actually holds.

    The dates returned are DATA dates read out of the body, never the fetch's
    ``fetched_at``: what a fetch *covers* is the bars it carries. A fetch
    stamped today but ending at yesterday's close — the pre-market fetch that
    caused the 2026-10-12 skip defect — contributes yesterday, which
    :func:`plan_refresh` refuses to accept for today's session. The one
    entry-per-fetch shape mirrors :func:`load_fetch_history`, so a ticker with
    several fetches gets several candidate dates and the plan asks only "does
    any of them reach far enough".

    WHICH rows of a body count as valid bars is not decided here: it is
    capture's rule, injected as ``read_bars`` — a callable handed the
    decompressed body bytes that returns the valid bar dates in it. The real
    caller passes ``engine.v2.data.price_download_sources.read_tier1_body``'s
    ``date`` column (see ``engine/v2/ops/legacy_adapter.py``
    ``invoke_price_refresh``), the same reader the ``price_history`` capture
    uses: a row with a null date or a null ``Adj Close`` is not a bar capture
    would store, and a body missing ``Close``/``Adj Close``/``High``, or with
    one of them parsing entirely NaN, is a typed contract refusal rather than a
    coverage answer. Legacy cannot import ``engine.v2``
    (``checks/import_layers.py``), so the rule arrives as a parameter instead of
    being reimplemented here as a second rule that could drift from capture's:
    a bar the capture would not write is a bar this loader never counts.

    ``fetched_at`` is read for one purpose only, a cheap prefilter: a fetch
    stamped before :func:`_required_daily_bar_date` (the latest trading day on
    or before ``session``) cannot possibly hold that required bar — the market
    had not opened yet on that day when the bytes arrived — so its body is
    never opened. The floor is the required BAR date, not ``session`` minus one
    day: a fetch stamped the Friday before a Saturday, Sunday or
    Monday-holiday session IS read, because Friday's close is exactly the bar
    those sessions require.

    The dates ``read_bars`` returns are normalized the way
    ``computed_moves.fetch_history`` normalizes them — ``pd.to_datetime(...,
    utc=True)`` through the per-row UTC offsets yfinance writes (they flip at
    DST), then ``tz_localize(None)`` and ``normalize()`` to midnight, so the
    trade date is what survives. No ``newest_bar`` meta key is read or written
    here: a shortcut stamped by the writer would be exactly the arrival-time
    assumption this loader exists to stop trusting.

    Any failure to read one fetch — an unreadable or corrupt body, ANY
    exception raised by ``entry.body()`` or ``read_bars`` (a broken gzip
    stream, a ``ValueError`` from the CSV parse, capture's own typed refusal),
    no valid dates in the result, an unusable ``fetched_at`` — makes that fetch
    contribute NOTHING. Unknown is treated as not covered, which can only ever
    lengthen today's daily list, never shorten it: a fetch nobody can read is
    never evidence that a session's close is already on disk.

    ``fetch_root`` defaults to ``engine.paths.RAW_FETCH`` and exists only so
    tests can point this at a throwaway tree.
    """
    session_ts = pd.Timestamp(session).normalize()
    required = _required_daily_bar_date(session_ts)
    out: dict[str, list[date_cls]] = {}
    for entry in iter_cached("yfinance", "history", root=fetch_root):
        if entry.params.get("period") != "max":
            continue
        ticker = entry.params.get("ticker")
        fetched_at = entry.meta.get("fetched_at")
        if not ticker or not fetched_at:
            continue
        try:
            when = pd.Timestamp(fetched_at)
            if when.tzinfo is not None:
                when = when.tz_convert("UTC").tz_localize(None)
            if when.normalize() < required:
                continue
        except (TypeError, ValueError):
            continue
        # Broad on purpose: a broken stream, a CSV that will not parse or
        # capture's own typed refusal all mean the same thing here — this
        # fetch is evidence of nothing, so it covers nothing.
        try:
            bars = pd.to_datetime(list(read_bars(entry.body())), errors="coerce", utc=True)
            bars = bars.tz_localize(None).normalize().dropna()
        except Exception:
            continue
        if bars.empty:
            continue
        out.setdefault(str(ticker), []).append(bars.max().date())
    return out
