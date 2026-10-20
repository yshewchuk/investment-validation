"""Pure computed-move math, moved from ``engine.data.pulls.computed_moves``
(spec s4b Change 2).

``session_move``/``MIN_SCOREABLE``/``MAX_GAP_CALENDAR_DAYS`` are copied
verbatim from the untouched legacy pull (the source of record for the legacy
path); the legacy module itself is never edited. :func:`build_rows` keeps
``build_ticker``'s realised-move computation byte-for-byte and returns one row
dict per event (``computed_moves.v4``: plus the ``event_history`` columns), matching
:mod:`engine.v2.data.computed_moves_table`'s columns, including one row per
SKIPPED event (``skipped=True``, ``realized_move_pct=None``) and, on every
computable row, ``available_as_of_date`` (the measured close date plus one
calendar day, YYYY-MM-DD -- null exactly when ``realized_move_pct`` is null:
a skipped event is not available to any decision). A skip must be visible,
not just counted (EXP-117 DEFINITION.md R3's materiality rule).

``build_event_rows`` (``computed_moves.v5``, issue #966) is the same math over EVERY
considered event: a CONFIRMED one (``src_orats`` and a session) gets its v4 row plus
``not_computed_reason``; the events v4 dropped and the two ticker-level refusals become
skipped, no-outcome rows carrying that reason rather than vanishing.
``classify_event_reasons`` holds the reason rule alone so the capture path can reuse it;
both entry points collapse same-date rows, then require strictly ascending dates
(``CONTRACT_MISMATCH``) before any population branch; an empty ``events`` is the ONE
way either returns ``[]``.

Layer 1 of ``system_rearchitecture.md`` §4.1: no ``engine.v2.ops`` and no
legacy ``engine.*`` imports -- this module is pure numpy/pandas math, the same
rule ``price_history_table.py`` states for its own layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import fail
from .event_history import AsofSeries, event_sequence, strict_prior_mean

__all__ = [
    "MAX_GAP_CALENDAR_DAYS",
    "MIN_SCOREABLE",
    "NativeTradingCalendar",
    "build_event_rows",
    "build_rows",
    "build_ticker",
    "classify_event_reasons",
    "native_trading_calendar",
    "projected_trading_days",
    "session_move",
    "us_market_holidays",
]

#: The scoreability bar the board's champion models impose (span-12 EMAs).
MIN_SCOREABLE = 12

#: P→Q windows wider than this are halts/gaps, excluded not guessed.
MAX_GAP_CALENDAR_DAYS = 5

#: Panel admission is k>=4, so fewer than this many computable events cannot score.
ADMISSION_EVENTS = 5

#: Legacy panel MIN_HISTORY: an event is admitted only when 4 or more events precede it.
WARMUP = 4


def _below_admission(computable: int) -> bool:
    """Fewer than ``ADMISSION_EVENTS`` computable events can never score.

    The ONE admission comparison ``build_rows`` and ``classify_event_reasons``
    share, so their verdicts cannot drift apart.
    """
    return computable < ADMISSION_EVENTS


def _session_move_indexed(sd, sc, t, session) -> tuple[float, np.datetime64] | None:
    """Same computation as session_move, plus the measured close date (sd[j_post])."""
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
    return float((q / p - 1.0) * 100.0), sd[j_post]


def session_move(sd, sc, t, session) -> float | None:
    result = _session_move_indexed(sd, sc, t, session)
    return result[0] if result is not None else None


#: Absent ``not_computed_reason``: a ``computed_moves.v4`` row carries no such key.
_ABSENT = object()


def _row(ticker: str, event_date: str, *, realized, implied, available_as_of_date,
         ordinal: int, admitted: bool, mean_prior, n_prior: int, skipped: bool,
         computed_at: str, source_hash: str, capture_id: str,
         not_computed_reason=_ABSENT) -> dict:
    row = {
        "ticker": ticker,
        "event_date": event_date,
        "realized_move_pct": realized,
        "available_as_of_date": available_as_of_date,
        "implied_move_pct": implied,
        "event_ordinal": ordinal,
        "admitted": admitted,
        "mean_prior_or_implied": mean_prior,
        "n_prior_implied": n_prior,
        "skipped": skipped,
        "computed_at": computed_at,
        "source_hash": source_hash,
        "capture_id": capture_id,
    }
    if not_computed_reason is not _ABSENT:
        row["not_computed_reason"] = not_computed_reason
    return row


def _validate_daily(daily: pd.DataFrame) -> None:
    """The ``daily`` frame contract: ``date``, ``implied_move`` and ``src_iv`` present.

    The ONE place these refusals and their wording live; ``build_rows`` reaches it
    through :func:`_implied_by_event`, ``build_event_rows`` calls it BEFORE any
    branch, so a ticker-level refusal cannot mask a malformed frame, empty or not.
    ``date`` must parse as datetimes with no NaT, ``implied_move`` must be numeric
    (all-null None fine); a bad cell refuses here, never as a raw parse error later.
    """
    if not {"date", "implied_move", "src_iv"} <= set(daily.columns):
        raise fail("CONTRACT_MISMATCH", "daily_market frame lacks date, implied_move or src_iv")
    try:
        parsed = pd.to_datetime(daily["date"])
    except (ValueError, TypeError, OverflowError):
        raise fail("CONTRACT_MISMATCH", "daily_market dates are unparseable") from None
    if parsed.isna().any():
        raise fail("CONTRACT_MISMATCH", "daily_market dates are unparseable")
    if not pd.api.types.is_numeric_dtype(daily["implied_move"]) and any(
            isinstance(cell, (bool, np.bool_)) or not isinstance(cell, (int, float, np.number))
            for cell in daily["implied_move"].dropna()):
        raise fail("CONTRACT_MISMATCH", "daily_market implied_move must be numeric")


def _require_daily_order(daily: pd.DataFrame) -> None:
    """Strictly ascending, unique ``daily`` calendar days whenever the frame is non-empty.

    The ordering refusal :func:`build_rows` reaches through ``AsofSeries`` (v4's,
    unchanged), checked UP FRONT by :func:`build_event_rows` before any population
    branch, so a ticker-level reason cannot skip it. The comparison is the SAME
    ``%Y-%m-%d`` projection :func:`_implied_by_event` feeds ``AsofSeries``, so two
    ascending intraday timestamps on one calendar day refuse here exactly as there
    -- in every branch. An empty frame is still fine.
    """
    if not len(daily):
        return
    days = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    if not (days.is_monotonic_increasing and days.is_unique):
        raise fail("CONTRACT_MISMATCH", "rows must be strictly date-ascending")


def _implied_by_event(dates: list[str], daily: pd.DataFrame) -> list[float | None]:
    """Implied move per event date: the last row strictly before it whose ``src_iv`` is
    non-null (``AsofSeries``, ``on_missing="null"``); an anchor whose ``implied_move`` is
    null or non-finite yields None and never falls back to an older row."""
    rows = []
    _validate_daily(daily)
    if len(daily):
        days = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
        for day, im, src in zip(days, daily["implied_move"], daily["src_iv"]):
            rows.append({
                "date": day,
                "implied_move": None if pd.isna(im) or not np.isfinite(im) else float(im),
                "src_iv": None if pd.isna(src) else src})
    series = AsofSeries(rows, eligible=("src_iv",), value="implied_move", on_missing="null")
    return [series.at(day) for day in dates]


def build_rows(ticker: str, events: pd.DataFrame, sd, sc, daily: pd.DataFrame, *,
               computed_at: str, source_hash: str, capture_id: str) -> list[dict]:
    """One ``computed_moves.v4`` row dict per event for ``ticker`` -- skipped events included.

    The realised move is the legacy ``build_ticker`` computation verbatim
    (session-aware close-to-close). ``available_as_of_date`` is the measured close
    date (``session_move``'s ``sd[j_post]``) plus one calendar day, null exactly when
    ``realized_move_pct`` is null. The event-history columns come from the
    ``event_history`` primitives (design #651): ``event_ordinal``/``admitted`` count
    every event of ``events`` (``WARMUP`` = 4); ``implied_move_pct`` is the as-of
    ``src_iv`` anchor, filled for skipped events; ``mean_prior_or_implied`` and
    ``n_prior_implied`` are the strict-prior mean and count over earlier admitted
    events. Fewer than five computable events returns ``[]`` (no partial write).
    """
    dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in events["event_date"]]
    sequence = event_sequence(dates, warmup=WARMUP)
    implied = _implied_by_event(dates, daily)
    aggregates = strict_prior_mean(sequence, implied, population="admitted")

    rows: list[dict] = []
    computable = 0
    for r, ev, im, (mean_prior, n_prior) in zip(events.itertuples(), sequence, implied,
                                                aggregates):
        indexed = _session_move_indexed(sd, sc, r.event_date.to_datetime64(), r.session)
        if indexed is None:
            realized, available_as_of_date, skipped = None, None, True
        else:
            realized, close_date = indexed
            available_as_of_date = str((pd.Timestamp(close_date) + pd.Timedelta(days=1)).date())
            skipped = False
            computable += 1
        rows.append(_row(
            ticker, ev.event_date, realized=realized, implied=im,
            available_as_of_date=available_as_of_date, ordinal=ev.ordinal,
            admitted=ev.admitted, mean_prior=mean_prior, n_prior=n_prior, skipped=skipped,
            computed_at=computed_at, source_hash=source_hash, capture_id=capture_id))

    if _below_admission(computable):  # panel admission is k>=4, so fewer cannot score
        return []
    return rows


#: The legacy name of the same builder, kept so a reader of the moved code
#: finds the function it was moved from (spec s4b Change 2).
build_ticker = build_rows


# --------------------------------------------------------------------------
# computed_moves.v5: one row per considered event, with a reason (issue #966)
# --------------------------------------------------------------------------#: The two reasons that apply to EVERY event of a ticker, whatever the event itself.
_TICKER_LEVEL = ("no_price_history", "ticker_too_few_events")


def _event_reasons(events: pd.DataFrame) -> list[str | None]:
    """The per-event reason before any ticker-level override: ``None`` = confirmed.

    The ONE definition of confirmed (``src_orats`` and a session) and the two
    per-event refusals, so counting and row building can never disagree.
    """
    reasons: list[str | None] = []
    for src_orats, session in zip(events["src_orats"], events["session"]):
        if not bool(src_orats):
            reasons.append("not_orats")
        elif pd.isna(session):
            reasons.append("no_session")
        else:
            reasons.append(None)
    return reasons


#: Every considered-event frame must carry these (the v5 builder's inputs).
_EVENT_COLUMNS = ("event_date", "session", "src_orats")


def _validate_inputs(events: pd.DataFrame, sd, sc, min_scoreable: int) -> None:
    """Public-input guard: the typed ``CONTRACT_MISMATCH`` refusal, raised first.

    Both v5 entry points call it before any work, so a malformed frame refuses
    outright instead of half-building rows (every refusal one registered code,
    never a bare exception). ``event_date`` must be a datetime column -- the store
    converts with ``pd.to_datetime``, a string refusing here rather than escaping
    as an ``AttributeError`` from ``.to_datetime64()`` later -- with no NaT and
    only calendar days (a time component refuses too). ``src_orats`` must hold a
    real ``bool``/``np.bool_`` per row (``None``/``pd.NA``/``NaN``/text are not a
    verdict). Every non-null ``session`` must be exactly ``"BMO"`` or ``"AMC"``
    (the source schema's set), a null staying allowed (it classifies as
    ``no_session``), so a typo refuses here instead of reading as AMC later. The
    price-series stage lives in :func:`_require_price_series`.
    """
    if (isinstance(min_scoreable, (bool, np.bool_))
            or not isinstance(min_scoreable, (int, np.integer)) or min_scoreable < 0):
        raise fail("CONTRACT_MISMATCH", "min_scoreable must be a non-negative int")
    missing = [name for name in _EVENT_COLUMNS if name not in events.columns]
    if missing:
        raise fail("CONTRACT_MISMATCH",
                   "events lacks required columns: " + ", ".join(missing))
    if not pd.api.types.is_datetime64_any_dtype(events["event_date"]):
        raise fail("CONTRACT_MISMATCH", "events event_date must be a datetime column")
    if events["event_date"].isna().any():
        raise fail("CONTRACT_MISMATCH", "events event_date must not hold NaT")
    if (events["event_date"] != events["event_date"].dt.normalize()).any():
        raise fail("CONTRACT_MISMATCH", "events event_date must be a calendar day")
    if any(not isinstance(value, (bool, np.bool_)) for value in events["src_orats"]):
        raise fail("CONTRACT_MISMATCH", "events src_orats must hold a real bool per row")
    if any(not pd.isna(session) and session not in ("BMO", "AMC")
           for session in events["session"]):
        raise fail("CONTRACT_MISMATCH", "events session must be BMO or AMC")
    _require_price_series(sd, sc)


def _require_price_series(sd, sc) -> None:
    """The price-series stage of the public-input contract: equal lengths, ``sd`` ordered.

    Split out of :func:`_validate_inputs` (complexity budget): a length mismatch
    or a non-ascending ``sd`` refuses BEFORE any row is built, from both v5 entry
    points alike; a length below 2 is always ordered.
    """
    if len(sd) != len(sc):
        raise fail("CONTRACT_MISMATCH", "price series sd and sc differ in length")
    if len(sd) < 2:
        return
    dates = np.asarray(sd).astype("datetime64[ns]")
    if not np.all(np.diff(dates) >= np.timedelta64(0, "ns")):
        raise fail("CONTRACT_MISMATCH", "price dates sd must be ascending")


def _collapse_same_date(events: pd.DataFrame) -> pd.DataFrame:
    """One row per event date: the CONFIRMED one, else the first in input order.

    Same-date rows (a phantom plus its confirmation) would reach
    ``event_sequence`` as a duplicate date, so both v5 entry points collapse
    BEFORE counting or building -- same frame, same rule, so they can never
    disagree. A survivor keeps its first-occurrence position: input order is
    preserved and the first row wins while every same-date candidate is
    unconfirmed.
    """
    if not len(events):
        return events
    reasons = _event_reasons(events)
    kept: dict[str, int] = {}
    confirmed: dict[str, bool] = {}
    for position, (day, reason) in enumerate(zip(events["event_date"], reasons)):
        key = pd.Timestamp(day).strftime("%Y-%m-%d")
        wins = reason is None
        if key not in kept or (wins and not confirmed[key]):
            kept[key], confirmed[key] = position, wins
    return events.iloc[[kept[key] for key in kept]]


def _collapse_and_order(events: pd.DataFrame) -> pd.DataFrame:
    """Same-date collapse, then the ONE ordering check both v5 entry points run.

    AFTER the collapse and BEFORE any population branch, so a reversed ``events``
    frame refuses from :func:`classify_event_reasons` exactly as from
    :func:`build_event_rows` -- same ``CONTRACT_MISMATCH``, same message (the one
    ``build_rows``' ``event_sequence`` gives), whatever the ticker-level outcome.
    """
    events = _collapse_same_date(events)
    days = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in events["event_date"]]
    for previous, current in zip(days, days[1:]):
        if current <= previous:
            raise fail("CONTRACT_MISMATCH", "event dates must be strictly ascending")
    return events


def _classify(events: pd.DataFrame, sd, sc, *, min_scoreable: int) -> list[str | None]:
    """The reason list of an ALREADY validated, ALREADY collapsed frame.

    A confirmed event no window brackets earns ``no_price_window`` HERE when the
    ticker still clears both bars, so the classification alone is that reason's
    source and no caller re-derives it from a built row.
    """
    reasons = _event_reasons(events)
    if not len(reasons):
        return []
    if not len(sd):
        return ["no_price_history"] * len(reasons)
    confirmed = computable = 0
    classified = list(reasons)
    for position, event in enumerate(events.itertuples()):
        if reasons[position] is not None:
            continue
        confirmed += 1
        if _session_move_indexed(sd, sc, event.event_date.to_datetime64(),
                                 event.session) is None:
            classified[position] = "no_price_window"
        else:
            computable += 1
    if confirmed < min_scoreable or _below_admission(computable):
        return ["ticker_too_few_events"] * len(reasons)
    return classified


def classify_event_reasons(events: pd.DataFrame, sd, sc, *,
                           min_scoreable: int = MIN_SCOREABLE) -> list[str | None]:
    """The ``not_computed_reason`` of every considered event of one ticker (issue #966).

    ``None`` means CONFIRMED (``src_orats`` and a session), the ticker clears both
    bars, and a price window brackets the event. A CONFIRMED event with no window
    is ``no_price_window`` even on a clearing ticker: this list, not a built row,
    is that reason's source, so :func:`build_event_rows` writes exactly what comes
    back. Per-event refusals: ``not_orats`` (no ORATS confirmation, e.g. a
    ``forward_claim`` phantom), ``no_session`` (confirmed, session null).
    Ticker-level reasons override every event: ``no_price_history`` when ``sd``
    holds no closes (checked first), else ``ticker_too_few_events`` when fewer
    than ``min_scoreable`` events are confirmed or fewer than ``ADMISSION_EVENTS``
    compute. Same-date collapse first, then strictly ascending dates or
    ``CONTRACT_MISMATCH``, exactly as :func:`build_event_rows`; ``[]`` only for an
    empty ``events``. Pure, no I/O and no clock, for the capture path to reuse.
    """
    _validate_inputs(events, sd, sc, min_scoreable)
    return _classify(_collapse_and_order(events), sd, sc, min_scoreable=min_scoreable)


def build_event_rows(ticker: str, events: pd.DataFrame, sd, sc, daily: pd.DataFrame, *,
                     computed_at: str, source_hash: str, capture_id: str,
                     min_scoreable: int = MIN_SCOREABLE) -> list[dict]:
    """One ``computed_moves.v5`` row dict per considered event of ``events``.

    ``[]`` only for an empty ``events``; every non-empty list yields one row per DISTINCT
    event date, so ``(ticker, event_date)`` is unique. ``events`` is the ticker's ordered
    ``earnings_events`` slice, and what v4 dropped is now a row. The ``daily`` frame is
    validated up front -- parseable, no-NaT dates, numeric ``implied_move``, calendar days
    strictly ascending and unique -- before any branch, so a ticker-level refusal cannot
    hide a malformed frame. Same-date collapse, then strictly ascending dates (the check
    :func:`classify_event_reasons` runs), so ``build_rows`` never sees a duplicate or
    descending date. A CONFIRMED event gets exactly the :func:`build_rows` row over the
    confirmed-only frame -- ``event_ordinal`` counts earlier confirmed events -- plus
    ``not_computed_reason`` taken STRAIGHT from the classification: null on a computed
    outcome (``skipped`` false), ``no_price_window`` where the classification found no
    window (that row is ``skipped`` true). An unconfirmed event, and every event under a
    ticker-level reason, gets the no-outcome shape (null moves, ``admitted`` false,
    ``skipped`` true, ``n_prior_implied`` 0) with that reason and the count of earlier
    confirmed events as its ordinal; a phantom never shifts a neighbour's ordinal or
    prior mean.
    """
    _validate_inputs(events, sd, sc, min_scoreable)
    _validate_daily(daily)
    _require_daily_order(daily)
    events = _collapse_and_order(events)
    reasons = _classify(events, sd, sc, min_scoreable=min_scoreable)
    per_event = _event_reasons(events)
    ticker_reason = reasons[0] if reasons and reasons[0] in _TICKER_LEVEL else None
    computed: list[dict] = []
    if ticker_reason is None:
        frame = events.iloc[[i for i, reason in enumerate(per_event) if reason is None]]
        computed = build_rows(ticker, frame, sd, sc, daily, computed_at=computed_at,
                              source_hash=source_hash, capture_id=capture_id)

    rows: list[dict] = []
    ordinal = taken = 0
    for position, event in enumerate(events.itertuples()):
        is_confirmed = per_event[position] is None
        row_ordinal = ordinal
        if is_confirmed:
            ordinal += 1
        if ticker_reason is None and is_confirmed:
            row = computed[taken]
            taken += 1
            row["not_computed_reason"] = reasons[position]
        else:
            row = _row(ticker, pd.Timestamp(event.event_date).strftime("%Y-%m-%d"),
                       realized=None, implied=None, available_as_of_date=None,
                       ordinal=row_ordinal, admitted=False, mean_prior=None, n_prior=0,
                       skipped=True, computed_at=computed_at, source_hash=source_hash,
                       capture_id=capture_id, not_computed_reason=reasons[position])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# the native forward trading calendar (spec s4c Rewrite 1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeTradingCalendar:
    """A trading-day list whose observed prefix came from a pinned snapshot.

    ``observed_through`` is the last real session the snapshot carries;
    anything after it is rule-projected weekdays (holidays removed), exactly
    the split ``engine.calendar.TradingCalendar`` makes. Only ``days`` is
    consumed by ``horizon_dates``.
    """

    days: tuple[pd.Timestamp, ...]
    observed_through: pd.Timestamp


def _easter(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (anonymous computus) — Good Friday is two days earlier."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """``n``-th ``weekday`` (Mon=0) of a month; ``n=-1`` means the last one."""
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = days[(days.month == month) & (days.weekday == weekday)]
    return days[n if n < 0 else n - 1]


def _observed(date: pd.Timestamp) -> pd.Timestamp:
    """NYSE observation rule: Saturday -> the Friday before, Sunday -> the Monday after."""
    if date.weekday() == 5:
        return date - pd.Timedelta(days=1)
    if date.weekday() == 6:
        return date + pd.Timedelta(days=1)
    return date


def _observed_new_year(year: int) -> pd.Timestamp | None:
    """New Year's Day, which does *not* follow the Saturday->Friday rule."""
    day = pd.Timestamp(year=year, month=1, day=1)
    if day.weekday() == 5:
        return None
    return _observed(day)


def us_market_holidays(year: int) -> set[pd.Timestamp]:
    """The scheduled NYSE holidays for ``year``, with observation rules applied.

    Moved verbatim (spec s4c Rewrite 1) from ``engine.calendar`` — pure
    arithmetic, no I/O — so the v2 layer never imports the legacy module.
    """
    out = {
        _nth_weekday(year, 1, 0, 3),  # MLK Day (from 1998)
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        _easter(year) - pd.Timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(pd.Timestamp(year=year, month=7, day=4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(pd.Timestamp(year=year, month=12, day=25)),
    }
    new_year = _observed_new_year(year)
    if new_year is not None:
        out.add(new_year)
    if year >= 2022:  # Juneteenth became a market holiday in 2022
        out.add(_observed(pd.Timestamp(year=year, month=6, day=19)))
    return out


def projected_trading_days(start, end) -> pd.DatetimeIndex:
    """Weekdays in ``(start, end]`` that are not scheduled market holidays.

    Moved verbatim (spec s4c Rewrite 1) from ``engine.calendar``; used only to
    extend the calendar *past* the last session the pinned snapshot observed.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end <= start:
        return pd.DatetimeIndex([])
    days = pd.date_range(start + pd.Timedelta(days=1), end, freq="B")
    holidays: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        holidays |= us_market_holidays(year)
    return days[~days.isin(pd.DatetimeIndex(sorted(holidays)))]


def native_trading_calendar(daily_by_ticker, extend_days: int = 400, *,
                            horizon_end=None) -> NativeTradingCalendar:
    """The forward calendar sourced from the pinned snapshot's own ``daily_market``.

    ``daily_by_ticker`` is one pass of the snapshot's ``daily_market`` rows
    grouped by ticker (``engine/v2/ops/*_store``'s single scan). The observed
    sessions are the DISTINCT ``date`` values across every ticker -- the same
    "a day the index did not trade is a day no chain was observed" rule the
    legacy calendar derives from its S&P series, but read from the pinned
    snapshot instead of a mutable legacy CSV. Future dates are rule-projected
    (``projected_trading_days``), never observed. ``horizon_end``, when given,
    is the caller's actual requested horizon end (as_of + horizon_days): the
    projection always extends through at least ``last + extend_days`` AND at
    least ``horizon_end``, so a stale panel (an old last observed session) or
    a wide horizon can never silently truncate the requested date range.
    """
    observed = sorted({
        pd.Timestamp(value).normalize()
        for frame in daily_by_ticker.values()
        for value in frame["date"]
    })
    if not observed:
        raise ValueError("native trading calendar needs at least one daily_market session")
    last = observed[-1]
    end = last + pd.Timedelta(days=extend_days)
    if horizon_end is not None:
        target = pd.Timestamp(horizon_end).normalize()
        if target > end:
            end = target
    future = projected_trading_days(last, end)
    return NativeTradingCalendar(tuple(observed) + tuple(future), last)
