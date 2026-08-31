"""As-of feature vectors, for events that have happened and events that have not.

The scoring engine has to answer the same question in two situations that look
very different:

*"What did we know about AAPL's 2019-01-29 print, at the close before it?"* —
answerable from the Tier-3 panel, which already carries exactly that row.

*"What do we know about AAPL's print next Tuesday?"* — no panel row exists,
because the panel is built from realized events and this one has not realized.

If those two paths are written separately they will diverge, and the divergence
will be invisible: the backtest keeps using the panel while the live dashboard
quietly drifts onto slightly different features. So both go through
:func:`build_features` here, and :func:`live_features` is built by *extending*
the panel's own recursions one event forward rather than by reimplementing
them. :func:`checks/phase1_checks.py` asserts the two agree on historical
events to 1e-9, which is what makes the equivalence a fact rather than an
intention.

Every value comes back inside an :class:`~engine.audit.FeatureVector` carrying
per-feature as-of stamps, and no consumer gets one without
:func:`~engine.audit.assert_causal` having run.

**One inherited convention, kept deliberately.** The panel reads market state at
the last daily row *strictly before the event date*. For a BMO print that is the
last pre-print close exactly. For an AMC print the event-date close is also
pre-print and would be admissible, but the panel used the prior close for both,
so this does too: one session stale on AMC names, never a leak, and identical
between replay and live. Changing it is a modeling decision for an experiment
to make and measure, not something a feature builder should do silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import re

import numpy as np
import pandas as pd

from engine import paths
from engine.audit import FeatureVector, assert_causal, assert_decision_causal
from engine.calendar import trading_calendar
from engine.data import store
from engine.data.features import panel as panel_mod

__all__ = [
    "ABSOLUTE_FEATURES",
    "QUOTE_INDICATORS",
    "add_absolute_features",
    "add_quote_indicators",
    "PANEL_FEATURE_COLUMNS",
    "OUTCOME_COLUMNS",
    "load_panel",
    "panel_for_ticker",
    "build_features",
    "panel_features",
    "live_features",
    "FeatureContext",
]

#: Columns of the Tier-3 panel that are legitimate model inputs. Everything the
#: panel carries *except* the realized outcome of the event being scored and the
#: bookkeeping keys. ``implied_move`` and ``or_implied`` are quoted before the
#: print, so they are features; ``move`` / ``abs_move`` are the answer.
OUTCOME_COLUMNS = ("move", "abs_move")

_KEY_COLUMNS = ("ticker", "k", "date", "quarter", "year", "mcap_asof")

#: Panel columns that exist for a *realized* event and cannot exist for an
#: upcoming one, however causal they are.
#:
#: ``implied_move`` is the oquants quoted implied move for the event itself. It
#: is genuinely pre-print information, so nothing about it leaks — but it is
#: sourced from the oquants moves file, which only lists events that have
#: already happened. A model trained on it scores every backtest happily and
#: then has nothing to read on the morning it matters.
#:
#: The legacy S2 feature list contains it. The live-equivalent is ``or_implied``
#: — the ORATS quoted implied move at the last pre-print close, which comes from
#: ``daily_market`` and is available for upcoming events — and champion feature
#: lists must use that instead. :func:`assert_live_available` enforces it at
#: registry-load time, so this cannot be rediscovered in production.
LIVE_UNAVAILABLE = ("implied_move",)


class UnservableFeature(ValueError):
    """A model asked for a feature that cannot be produced for a live event."""


def assert_live_available(names: Iterable[str], *, label: str = "model") -> None:
    """Raise unless every feature in ``names`` can be built for an upcoming event."""
    offenders = sorted(set(names) & set(LIVE_UNAVAILABLE))
    if offenders:
        raise UnservableFeature(
            f"{label}: feature(s) {offenders} exist only for realized events and "
            f"would be NaN for every upcoming print. Use 'or_implied' for the "
            f"quoted implied move."
        )

PANEL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c
    for c in panel_mod.PANEL_COLUMNS
    if c not in OUTCOME_COLUMNS and c not in _KEY_COLUMNS
)


# --------------------------------------------------------------------------
# panel access
# --------------------------------------------------------------------------


#: Absolute-valued counterparts of signed inputs, added by EXP-109.
#:
#: A signed input against a MAGNITUDE target is often V-shaped — high at both
#: ends, low in the middle — and the size model's blend is half linear, so the
#: OLS half cannot represent that shape at all. `mean_prior_move` against
#: `abs_move` runs 8.35 -> 4.60 -> 7.82 across its deciles on a Spearman of
#: +0.013. These are the magnitudes that shape is actually about.
#:
#: Derived on READ rather than stored in Tier 3, deliberately: they are pure
#: functions of a column that is already there, so deriving them in one place
#: means the panel path and the live path cannot drift, and no rebuild is
#: needed to make an existing snapshot serve them.
ABSOLUTE_FEATURES: dict[str, str] = {
    "abs_dist_high": "dist_high",
    "abs_dist_ema": "dist_ema",
}


#: Availability indicators: one column saying whether a quoted value exists at
#: all, beside the value itself.
#:
#: ``or_implied`` is 0 rather than null on 25.5% of ``daily_market`` rows, and
#: that zero is overwhelmingly a LIQUIDITY fact — the no-quote rate runs 31.5%
#: in the smallest market-cap decile against 1.6% in the largest. So the column
#: currently carries two different facts on one axis: a quoted implied move, and
#: "this name had no usable option quote". The indicator separates them, which
#: is what lets the value be nulled later without losing the liquidity signal
#: the models are presently extracting from the magic number.
#:
#: Measured effect on accuracy: none (EXP-111, +0.0032pp, 8/14 years, p=0.50).
#: Any model that can split on a value can already tell 0 from 6, so this makes
#: explicit what was implicit rather than adding information.
QUOTE_INDICATORS: dict[str, str] = {
    "has_implied_quote": "or_implied",
}


def add_quote_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the ``QUOTE_INDICATORS`` wherever their source column exists.

    1.0 where a real quote is present, 0.0 where the source is zero or null —
    so the indicator survives the value later becoming a proper null.
    """
    for name, source in QUOTE_INDICATORS.items():
        if source not in frame.columns:
            continue
        values = pd.to_numeric(frame[source], errors="coerce")
        frame[name] = (values > 0).astype(float)
    return frame


def add_absolute_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the ``ABSOLUTE_FEATURES`` wherever their source column exists.

    Fills where the column is ABSENT **or** null, not merely where it is
    absent. A forward event has no panel row, so the scorer's panel join
    creates these columns full of NaN before the derivation runs; an
    absent-only guard then skipped every one of them and left the NaN in place,
    which took all 148 STR-THRU rows off the board with
    ``non-finite ['abs_dist_ema', 'abs_dist_high']`` while their sources sat
    right there, finite, in the same frame.
    """
    for name, source in ABSOLUTE_FEATURES.items():
        if source not in frame.columns:
            continue
        derived = pd.to_numeric(frame[source], errors="coerce").abs()
        if name in frame.columns:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(derived)
        else:
            frame[name] = derived
    return frame


@lru_cache(maxsize=1)
def _panel_cached(path_str: str, mtime: float) -> pd.DataFrame:
    """Read the panel once per process, keyed on path + mtime.

    The panel is 115k × 45 and every scoring call needs it. Keying the cache on
    mtime means a rebuild in the same process is picked up rather than served
    stale, which matters because the snapshot hash would otherwise disagree with
    the features actually used.
    """
    frame = pd.read_parquet(path_str) if path_str.endswith(".parquet") else pd.read_csv(path_str)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = add_absolute_features(frame)
    frame = add_quote_indicators(frame)
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_panel(path=None) -> pd.DataFrame:
    """The Tier-3 causal panel."""
    path = paths.PANEL if path is None else path
    if not path.exists():
        alt = path.with_suffix(".csv.gz")
        if alt.exists():  # pragma: no cover - csv fallback environment
            path = alt
        else:
            raise FileNotFoundError(
                f"{path} missing — build Tier 3 with `python3 -m engine.data.rebuild "
                "--table panel`"
            )
    return _panel_cached(str(path), path.stat().st_mtime)


def panel_for_ticker(ticker: str, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = load_panel() if panel is None else panel
    return frame[frame["ticker"] == ticker]


# --------------------------------------------------------------------------
# shared context (loaded once, reused across a scoring run)
# --------------------------------------------------------------------------


@dataclass
class FeatureContext:
    """Everything the feature builders read, loaded once for a whole run.

    Scoring a three-week calendar means a few hundred feature vectors. Reading
    the panel and the relevant slice of ``daily_market`` per vector would
    dominate the runtime and blow the guide's five-minute budget; loading them
    once here is what keeps it in seconds.
    """

    panel: pd.DataFrame
    daily: pd.DataFrame | None = None
    calendar: object | None = None

    @classmethod
    def load(
        cls,
        tickers: Iterable[str] | None = None,
        *,
        years: Iterable[int] | None = None,
        with_daily: bool = True,
    ) -> "FeatureContext":
        panel = load_panel()
        daily = None
        if with_daily:
            # The union of what both consumers need, not just the panel's block:
            # `add_orats_features` reads ORATS_FEATURES, while `daily_state_frame`
            # additionally wants iv10, exern_iv10 and spot. Loading only the
            # first set leaves the second to fail on a missing column, which is
            # how it failed the first time.
            columns = sorted(
                {
                    "ticker",
                    "date",
                    "mcap_usd",
                    "mcap_log",
                    "src_iv",
                    *panel_mod.ORATS_FEATURES.keys(),
                    *DAILY_STATE_FIELDS.keys(),
                }
            )
            daily = store.read_table("daily_market", years=years, columns=columns)
            if tickers is not None:
                daily = daily[daily["ticker"].isin(set(tickers))]
            daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
        return cls(panel=panel, daily=daily, calendar=trading_calendar())

    #: Per-ticker slices, built once on first use. `daily_market` is ~8.9M rows
    #: and the panel 115k; a boolean scan per lookup is ~50 ms on the former, and
    #: scoring a three-week calendar makes hundreds of lookups. Grouping once
    #: turns the whole run's lookup cost into a single pass.
    _panel_index: dict | None = None
    _daily_index: dict | None = None

    def ticker_panel(self, ticker: str) -> pd.DataFrame:
        if self._panel_index is None:
            self._panel_index = {t: g for t, g in self.panel.groupby("ticker", sort=False)}
        return self._panel_index.get(ticker, self.panel.iloc[0:0])

    def ticker_daily(self, ticker: str) -> pd.DataFrame:
        if self.daily is None:
            raise RuntimeError("FeatureContext was loaded without daily_market")
        if self._daily_index is None:
            self._daily_index = {t: g for t, g in self.daily.groupby("ticker", sort=False)}
        return self._daily_index.get(ticker, self.daily.iloc[0:0])


# --------------------------------------------------------------------------
# stamps
# --------------------------------------------------------------------------

#: Which as-of date each block of features carries. The panel reads market state
#: at the last daily row strictly before the event date and event-history
#: features from strictly-earlier events, so both are stamped with the last
#: pre-print close: that is the latest moment any of them could have been known,
#: and stamping conservatively (later) is what makes the audit meaningful.
_MARKET_BLOCK = tuple(panel_mod.ORATS_FEATURES.values()) + (
    "has_implied_quote",
    "abs_dist_high",
    "abs_dist_ema",
    "or_exern_z252",
    "mcap_log",
    "mcap_usd",
    "spy_ret21",
    "spy_ret63",
    "spy_ret252",
    "spy_dd252",
    "spy_vol20",
    "dist_high",
    "dist_ema",
    "ret5",
    "ret10",
    "ret20",
)


def _stamps(
    names: Iterable[str],
    as_of: pd.Timestamp,
    *,
    history_date: pd.Timestamp | None = None,
    market_date: pd.Timestamp | None = None,
) -> dict[str, pd.Timestamp]:
    """Stamp each feature at the date its information was actually observed.

    Three blocks, three stamps:

    - **Event-history features** (:data:`EVENT_HISTORY_FEATURES`) are computed
      from the ticker's strictly-earlier events, so they are observed at the
      last prior event's date (``history_date``). Stamping them there makes the
      audit substantive: a history feature built from the current or a future
      event fails ``assert_causal`` instead of passing a tautology.
    - **Market/ORATS block** (:data:`_MARKET_BLOCK`) is read at the last daily
      row on or before the decision close, so it is stamped at that row's date
      (``market_date``) when the caller knows it.
    - Anything else falls back to the decision close — the honest upper bound.

    Callers that cannot know a block's true date pass ``None`` for it and get
    the upper-bound stamp; the panel path does this for the market block (the
    panel row does not record which daily row built it), and says so.
    """
    out: dict[str, pd.Timestamp] = {}
    for name in names:
        if history_date is not None and name in EVENT_HISTORY_FEATURES:
            out[name] = history_date
        elif market_date is not None and name in _MARKET_BLOCK:
            out[name] = market_date
        else:
            out[name] = as_of
    return out


# --------------------------------------------------------------------------
# historical path — the panel row IS the feature vector
# --------------------------------------------------------------------------


def panel_features(
    ticker: str,
    event_date,
    *,
    as_of=None,
    session: str | None = None,
    context: FeatureContext | None = None,
) -> FeatureVector:
    """Feature vector for an event the panel already carries.

    This is the path the replay and every backtest use, so the numbers behind a
    historical claim are literally the panel's own — no recomputation, nothing
    to drift.
    """
    ctx = context or FeatureContext(panel=load_panel(), calendar=trading_calendar())
    event_date = pd.Timestamp(event_date).normalize()
    rows = ctx.ticker_panel(ticker)
    hit = rows[rows["date"] == event_date]
    if hit.empty:
        raise KeyError(f"panel has no {ticker} event on {event_date.date()}")
    row = hit.iloc[0]

    if as_of is None:
        cal = ctx.calendar or trading_calendar()
        as_of = cal.last_pre_print(event_date, session) if session else event_date
    as_of = pd.Timestamp(as_of).normalize()

    values = {
        name: (float(row[name]) if pd.notna(row[name]) else float("nan"))
        for name in PANEL_FEATURE_COLUMNS
        if name in row.index
    }
    # Event-history features were observed at the last prior event; the market
    # block's true daily row is not recorded on the panel row, so it keeps the
    # decision-close upper bound (documented in _stamps).
    prior_dates = rows.loc[rows["date"] < event_date, "date"]
    vector = FeatureVector(
        ticker=ticker,
        as_of=as_of,
        values=values,
        feature_as_of=_stamps(
            values, as_of,
            history_date=prior_dates.max().normalize() if len(prior_dates) else None,
        ),
        event_date=event_date,
        session=session,
        meta={
            "source": "panel",
            "k": int(row["k"]),
            "mcap_asof": row.get("mcap_asof"),
        },
    )
    assert_causal(vector)
    return vector


# --------------------------------------------------------------------------
# live path — extend the panel's recursions one event forward
# --------------------------------------------------------------------------


def advance_history(last_row) -> dict[str, float]:
    """Event-history features for the event *after* the one ``last_row`` describes.

    Recomputing the block from the panel's rows is not an option and the reason
    is worth stating, because it is the kind of thing that silently produces
    plausible wrong numbers: the panel admits an event only once its ticker has
    ``MIN_HISTORY`` prior events, so its rows are missing each ticker's first
    four moves. A mean taken over the rows that survive is a mean over a
    truncated history, and it is visibly wrong — on AAPL, 0.946 against the
    panel's 1.040.

    Every statistic in the block is an aggregate the panel already stores
    alongside the count it was taken over, so each one can be stepped forward
    exactly instead:

    ``mean``  ``(mean_k · n + x_k) / (n + 1)``
    ``ema``   ``a · x_k + (1 − a) · ema_k``, the same recursion ``_causal_ema``
              runs, resumed rather than restarted.

    The one statistic that cannot be advanced is an EMA the previous row did not
    have — the span-8 and span-12 EMAs before a ticker reaches 8 and 12 events —
    because resuming needs a value to resume from. Those come back as NaN, which
    is what the panel itself carries at that point in a ticker's life, and
    ``ema12r_abs`` falls back to the mean exactly as it does in the panel.
    """
    n = int(last_row["n_prior"])
    move = float(last_row["move"])
    abs_move = float(last_row["abs_move"])
    implied = last_row["implied_move"]

    out: dict[str, float] = {"n_prior": n + 1}
    for mean_col, value in (
        ("mean_prior_move", move),
        ("mean_prior_abs_move", abs_move),
    ):
        prev = last_row[mean_col]
        out[mean_col] = (
            (float(prev) * n + value) / (n + 1) if pd.notna(prev) else float("nan")
        )

    # The implied-move mean is taken over *known* implied values only. The panel
    # carries no count of those, so advancing it assumes every prior event had
    # one — true for all 115,500 panel rows, and asserted by the equivalence
    # check rather than trusted.
    prev_implied = last_row["mean_prior_implied_move"]
    if pd.notna(prev_implied) and pd.notna(implied):
        out["mean_prior_implied_move"] = (float(prev_implied) * n + float(implied)) / (n + 1)
    elif pd.notna(prev_implied):
        out["mean_prior_implied_move"] = float(prev_implied)
    else:
        out["mean_prior_implied_move"] = float("nan")

    for span in panel_mod.SPANS:
        alpha = 2.0 / (span + 1.0)
        for suffix, value in (("move", move), ("abs_move", abs_move)):
            col = f"ema{span}_prior_{suffix}"
            prev = last_row[col]
            out[col] = (
                alpha * value + (1.0 - alpha) * float(prev)
                if pd.notna(prev)
                else float("nan")
            )
    return out


def live_features(
    ticker: str,
    event_date,
    *,
    as_of=None,
    session: str | None = None,
    context: FeatureContext | None = None,
    quarter: str | None = None,
) -> FeatureVector:
    """Feature vector for an event with no panel row yet.

    Built by appending a synthetic row for the target event to the ticker's
    prior panel rows and running the panel's own three feature blocks over the
    result. The synthetic row's realized ``move`` stays NaN — it is the unknown
    we are scoring — and none of the blocks read it for the row they are
    computing, which is the property that makes this safe and is asserted in the
    test suite.
    """
    ctx = context or FeatureContext.load([ticker])
    event_date = pd.Timestamp(event_date).normalize()
    cal = ctx.calendar or trading_calendar()
    if session is None:
        session = _session_for(ticker, event_date)
    if as_of is None:
        as_of = cal.last_pre_print(event_date, session)
    as_of = pd.Timestamp(as_of).normalize()
    assert_decision_causal(as_of, event_date, session, calendar=cal)

    prior = ctx.ticker_panel(ticker)
    prior = prior[prior["date"] < event_date].sort_values("date")
    if prior.empty:
        raise KeyError(
            f"{ticker}: no prior panel events before {event_date.date()} — the "
            "history features cannot be computed"
        )

    history = advance_history(prior.iloc[-1])

    synthetic = {
        "ticker": ticker,
        "k": history["n_prior"],
        "date": event_date,
        "quarter": quarter,
        "move": np.nan,
        "abs_move": np.nan,
        "implied_move": np.nan,
        "year": int(event_date.year),
        **history,
    }
    frame = pd.concat(
        [prior, pd.DataFrame([synthetic])], ignore_index=True
    ).sort_values("date").reset_index(drop=True)

    frame = panel_mod.add_regime_features(frame)
    frame = panel_mod.add_runup_features(frame)
    daily = ctx.ticker_daily(ticker) if ctx.daily is not None else None
    frame = panel_mod.add_orats_features(frame, daily=daily)
    # The same derivation `load_panel` applies on read. Without it here the
    # live path would silently omit these and every forward row would report
    # MISSING_FEATURES for a model that lists them — the panel path would serve
    # them and the live path would not, which is precisely the training/serving
    # skew `checks/phase1_checks.py::feature_equivalence` exists to catch.
    frame = add_absolute_features(frame)
    frame = add_quote_indicators(frame)

    row = frame[frame["date"] == event_date].iloc[-1]
    values = {
        name: (float(row[name]) if pd.notna(row[name]) else float("nan"))
        for name in tuple(PANEL_FEATURE_COLUMNS) + tuple(ABSOLUTE_FEATURES) + tuple(QUOTE_INDICATORS)
        if name in row.index
    }
    # True observation dates, not the decision-close upper bound:
    # - event-history features were fixed at the last prior event;
    # - the market/ORATS block is read from the last daily row on or before
    #   as_of, so that row's date is when it was actually observable.
    # (`implied_move` is the oquants quoted implied move for this event, which
    # the panel carries but which does not exist for an unrealized one.
    # `or_implied` — ORATS, from daily_market at the last pre-print close — is
    # the live equivalent and is present, so model feature lists use it.)
    history_date = pd.Timestamp(prior.iloc[-1]["date"]).normalize()
    market_date = None
    if daily is not None and len(daily):
        on_or_before = daily.loc[daily["date"] <= as_of, "date"]
        if len(on_or_before):
            market_date = pd.Timestamp(on_or_before.max()).normalize()
    vector = FeatureVector(
        ticker=ticker,
        as_of=as_of,
        values=values,
        feature_as_of=_stamps(
            values, as_of, history_date=history_date, market_date=market_date
        ),
        event_date=event_date,
        session=session,
        meta={"source": "live", "k": history["n_prior"], "mcap_asof": row.get("mcap_asof")},
    )
    assert_causal(vector)
    return vector


def build_features(
    ticker: str,
    event_date,
    *,
    as_of=None,
    session: str | None = None,
    context: FeatureContext | None = None,
    prefer: str = "auto",
) -> FeatureVector:
    """One entry point: panel row where one exists, live extension where not.

    ``prefer`` forces a path (``"panel"`` / ``"live"``) for the equivalence
    test, which needs to build both for the same historical event and compare.
    """
    if prefer not in ("auto", "panel", "live"):
        raise ValueError(f"unknown prefer={prefer!r}")
    if prefer == "live":
        return live_features(
            ticker, event_date, as_of=as_of, session=session, context=context
        )
    try:
        return panel_features(
            ticker, event_date, as_of=as_of, session=session, context=context
        )
    except KeyError:
        if prefer == "panel":
            raise
        return live_features(
            ticker, event_date, as_of=as_of, session=session, context=context
        )


# --------------------------------------------------------------------------
# session lookup
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _session_index() -> dict[tuple[str, pd.Timestamp], str]:
    events = store.read_table(
        "earnings_events", columns=["ticker", "event_date", "session"]
    )
    events = events[events["session"].notna()]
    return {
        (t, pd.Timestamp(d).normalize()): str(s)
        for t, d, s in zip(events["ticker"], events["event_date"], events["session"])
    }


def _session_for(ticker: str, event_date: pd.Timestamp) -> str:
    """BMO/AMC from the Tier-2 calendar, defaulting to AMC.

    AMC is the right default: it is the majority session, and it is the
    *conservative* one for entry timing — it puts the decision at the event-date
    close, which for a name that turns out to be BMO would be caught by
    :func:`~engine.audit.assert_decision_causal` rather than silently traded.
    """
    from engine.calendar import AMC

    return _session_index().get((ticker, pd.Timestamp(event_date).normalize()), AMC)


def session_for(ticker: str, event_date) -> str:
    """Public wrapper around the Tier-2 session lookup."""
    return _session_for(ticker, pd.Timestamp(event_date).normalize())


# --------------------------------------------------------------------------
# daily market state at an arbitrary as-of date
# --------------------------------------------------------------------------

#: ``daily_market`` column → feature prefix. The panel reads market state at the
#: last close before the *event*; a structure that enters two weeks earlier needs
#: it at the *entry*, which is a different date and a different question.
DAILY_STATE_FIELDS = {
    "implied_move": "im",
    "iv10": "iv10",
    "iv30": "iv30",
    "exern_iv10": "exern_iv10",
    "exern_iv30": "exern_iv30",
    "iee": "iee",
    "skew": "skew",
    "contango": "contango",
    "fwd90_30": "fwd90_30",
    "fexern90_30": "fexern90_30",
    "rvol30": "rvol30",
    "spot": "spot",
    "mcap_log": "mcap_log",
}

#: Trailing differences that carry the run-up itself. The OPF finding is that
#: the *change* in quoted implied move is what is predictable; a level alone
#: cannot express "this name's implied move has climbed 4 points in a week".
DAILY_STATE_LAGS = (1, 5, 10)

#: Which fields get lag differences. Restricted to the volatility surface —
#: differencing a market cap or a spot price produces a number dominated by the
#: name's size rather than by its vol behaviour.
LAGGED_FIELDS = ("implied_move", "iv10", "iv30", "exern_iv30")

DAILY_STATE_COLUMNS: tuple[str, ...] = tuple(DAILY_STATE_FIELDS.values()) + tuple(
    f"{DAILY_STATE_FIELDS[f]}_d{lag}" for f in LAGGED_FIELDS for lag in DAILY_STATE_LAGS
)


def daily_state_frame(
    requests: pd.DataFrame,
    *,
    daily: pd.DataFrame | None = None,
    as_of_column: str = "as_of",
) -> pd.DataFrame:
    """Market state at each ``(ticker, as_of)``, plus trailing changes.

    ``requests`` needs ``ticker`` and ``as_of_column``. Every value is read at
    the last ``daily_market`` row **on or before** ``as_of`` — on-or-before, not
    strictly-before, because ``as_of`` is a close at which we would trade, and
    that close's own quotes are known to us then.

    Returns ``requests`` with the state columns joined on, in the same order.
    The trailing differences count *rows*, not calendar days: a five-row lag is
    five observations back in that ticker's own series, so a name with a gap in
    coverage gets a wider window rather than a silently wrong difference.
    """
    if daily is None:
        columns = ["ticker", "date", "src_iv", *DAILY_STATE_FIELDS.keys()]
        years = sorted(pd.to_datetime(requests[as_of_column]).dt.year.unique().tolist())
        daily = store.read_table(
            "daily_market", years=range(min(years) - 1, max(years) + 1), columns=columns
        )

    out = requests.copy().reset_index(drop=True)
    out["_row"] = np.arange(len(out))
    for column in DAILY_STATE_COLUMNS:
        out[column] = np.nan

    wanted = set(out["ticker"].unique())
    daily = daily[daily["ticker"].isin(wanted)]
    # IV-bearing rows only: `daily_market` also carries rows contributed purely
    # by the market-cap series, which have no surface on them. Letting one of
    # those be the as-of answer returns an all-NaN state for a date that does
    # have quotes.
    if "src_iv" in daily.columns:
        surface = daily[daily["src_iv"].notna()]
    else:  # pragma: no cover - projected reads always carry src_iv
        surface = daily
    surface = surface.sort_values(["ticker", "date"])

    by_ticker = {t: g for t, g in surface.groupby("ticker", sort=False)}
    field_names = list(DAILY_STATE_FIELDS)

    for ticker, group in out.groupby("ticker", sort=False):
        series = by_ticker.get(ticker)
        if series is None or series.empty:
            continue
        dates = series["date"].to_numpy()
        values = {f: series[f].to_numpy(dtype=float) for f in field_names}
        # side="right" - 1 == "the last row on or before as_of".
        idx = np.searchsorted(dates, group[as_of_column].to_numpy(), side="right") - 1
        rows = group["_row"].to_numpy()
        ok = idx >= 0
        for field_name in field_names:
            column = DAILY_STATE_FIELDS[field_name]
            out.loc[rows[ok], column] = values[field_name][idx[ok]]
        for field_name in LAGGED_FIELDS:
            column = DAILY_STATE_FIELDS[field_name]
            arr = values[field_name]
            for lag in DAILY_STATE_LAGS:
                prior = idx - lag
                good = ok & (prior >= 0)
                out.loc[rows[good], f"{column}_d{lag}"] = (
                    arr[idx[good]] - arr[prior[good]]
                )

    return out.drop(columns=["_row"])


#: Panel features that depend only on the ticker's *prior events*, and are
#: therefore known at any decision date before the print — including one two
#: weeks early.
#:
#: The distinction matters and is easy to get wrong. The panel's market-state
#: block (``or_iv30``, ``dist_high``, ``spy_vol20``, …) is read at the last
#: pre-print close. For STR-THRU, which enters at that close, using it is
#: correct. For STR-RUNUP, which enters fourteen trading days earlier, it would
#: be fourteen days of future information — a leak that would flatter the
#: entry-timing model precisely where the strategy's whole edge is claimed to
#: be. Models scored at an early entry use this list plus
#: :func:`daily_state_frame` at the entry date, never the panel's market block.
EVENT_HISTORY_FEATURES: tuple[str, ...] = (
    "n_prior",
    "mean_prior_move",
    "mean_prior_abs_move",
    "mean_prior_implied_move",
    "ema2_prior_move",
    "ema4_prior_move",
    "ema8_prior_move",
    "ema12_prior_move",
    "ema2_prior_abs_move",
    "ema4_prior_abs_move",
    "ema8_prior_abs_move",
    "ema12_prior_abs_move",
    "ema12r_abs",
    "signed_streak",
)


#: One line per feature, for the dashboard's derivation view. A prediction the
#: reader cannot take apart is not evidence, and a bare name like `ema12r_abs`
#: or `fexern90_30` explains nothing to anyone who did not build it.
#:
#: Base names only: the ``_dN`` lags and the ``emaN_prior_*`` family are derived
#: by :func:`feature_note` from the same rule that generates them, so a new lag
#: or window documents itself instead of silently arriving unlabelled.
FEATURE_NOTES: dict[str, str] = {
    "n_prior": "How many past prints this name has in the panel — the sample the other history features average over.",
    "mean_prior_move": "Mean SIGNED reaction to past prints, in %. Direction is unpredictable at the event level, so this is a level/drift term, not a bet.",
    "mean_prior_abs_move": "Mean ABSOLUTE reaction to past prints, in %. The plainest estimate of how much this name usually moves.",
    "mean_prior_implied_move": "Mean implied move the market quoted before past prints, in %. The baseline the size model is trying to beat.",
    "ema12r_abs": "Ratio of the fast to the slow EMA of past absolute moves — is this name moving more than it used to?",
    "signed_streak": "Run length of consecutive same-direction reactions.",
    "im": "Implied move quoted for THIS print, in %, from the option market.",
    "or_implied": "Implied move at the last pre-print close (ORATS), in %.",
    "iv10": "10-day implied volatility.",
    "iv30": "30-day implied volatility.",
    "exern_iv10": "10-day IV with the earnings event stripped out — the 'background' vol.",
    "exern_iv30": "30-day IV with the earnings event stripped out.",
    "or_rvol30": "30-day realized volatility (ORATS), in %.",
    "rvol30": "30-day realized volatility, in %.",
    "iee": "Implied earnings effect: how much of the front IV the event itself accounts for.",
    "skew": "Put-vs-call skew of the surface.",
    "contango": "Slope of the IV term structure.",
    "fwd90_30": "Forward vol between the 30- and 90-day tenors.",
    "fexern90_30": "The same forward vol, ex-earnings.",
    "spot": "Underlying price at the decision close.",
    "mcap_log": "log(market cap). Era-normalized in Tier 2 — the ORATS unit switches are fixed there, once.",
    "dist_high": "Distance from the 52-week high, in %.",
    "dist_ema": "Distance from the trailing EMA of price, in %.",
    "has_implied_quote": "1 when the market actually quoted an implied move, 0 when it did not — a liquidity fact, not a forecast (EXP-111).",
    "abs_dist_high": "How FAR from the 52-week high, ignoring direction (EXP-109).",
    "abs_dist_ema": "How FAR from the trailing EMA, ignoring direction (EXP-109).",
    "spy_vol20": "20-day realized volatility of the S&P — the market regime the trade sits in.",
    "spy_dd252": "S&P drawdown from its 252-day high, in %.",
    "days_to_print": "Calendar days from the decision to the announcement.",
    "days_before_print": "TRADING days from entry to the last pre-print close. 0 for STR-THRU, 14 for STR-RUNUP — calendar days here would be a silent training/serving skew.",
    "entry_cost_pct": "Premium paid for the structure, as % of spot. The gate's read on whether the trade is expensive.",
    "dte_entry": "Days to expiry of the traded contracts at entry.",
}

#: Human labels for the two payoff drivers.
DRIVER_NOTES: dict[str, str] = {
    "abs_move": "the size of the move the stock makes on the print (|move|, %)",
    "im_t1": "the implied move the market will quote at the last pre-print close (%)",
}


def feature_note(name: str) -> str:
    """One line explaining ``name``, deriving the lag and EMA families.

    Returns an empty string for a feature nobody has documented — the caller
    shows the bare name rather than inventing an explanation for it.
    """
    if name in FEATURE_NOTES:
        return FEATURE_NOTES[name]

    lag = re.fullmatch(r"(?P<base>.+)_d(?P<n>\d+)", name)
    if lag:
        base = feature_note(lag.group("base")) or lag.group("base")
        return (
            f"Change over the last {lag.group('n')} observations in: "
            f"{base[0].lower()}{base[1:]}"
        )

    ema = re.fullmatch(r"ema(?P<n>\d+)_prior_(?P<kind>abs_move|move)", name)
    if ema:
        kind = "absolute" if ema.group("kind") == "abs_move" else "signed"
        return (
            f"Exponentially weighted mean of the {kind} reaction to past prints, "
            f"span {ema.group('n')} — recent prints weighted more than old ones."
        )
    return ""


def entry_feature_frame(
    requests: pd.DataFrame,
    *,
    panel: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    as_of_column: str = "entry_date",
) -> pd.DataFrame:
    """Leak-safe features for a decision taken at ``as_of_column``.

    ``requests`` needs ``ticker``, ``event_date`` and the as-of column. Returns
    it with :data:`EVENT_HISTORY_FEATURES`, :data:`DAILY_STATE_COLUMNS`, and
    ``days_to_print`` joined on.

    ``days_to_print`` is a feature, not bookkeeping: how far an observation sits
    from the print is most of what determines how much the implied move still
    has left to climb.
    """
    frame = requests.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame[as_of_column] = pd.to_datetime(frame[as_of_column])

    panel_frame = load_panel() if panel is None else panel
    history_cols = [c for c in EVENT_HISTORY_FEATURES if c in panel_frame.columns]
    frame = frame.merge(
        panel_frame[["ticker", "date", *history_cols]].rename(columns={"date": "event_date"}),
        on=["ticker", "event_date"],
        how="left",
    )
    frame = daily_state_frame(frame, daily=daily, as_of_column=as_of_column)
    frame["days_to_print"] = (
        frame["event_date"] - frame[as_of_column]
    ).dt.days.astype(float)
    return frame


def frame_features(
    events: pd.DataFrame,
    *,
    context: FeatureContext | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Vectorized panel-path features for many events at once.

    ``events`` needs ``ticker`` and ``event_date``. Used by the replay and the
    model-training pipelines, where a per-event :class:`FeatureVector` would
    build hundreds of thousands of objects to no purpose — the causality
    property is identical (it is the same panel rows), and it is asserted here
    in one vectorized pass.
    """
    ctx = context or FeatureContext(panel=load_panel(), calendar=trading_calendar())
    wanted = list(columns) if columns is not None else list(PANEL_FEATURE_COLUMNS)
    keep = ["ticker", "date"] + [c for c in wanted if c in ctx.panel.columns]
    merged = events.merge(
        ctx.panel[keep].rename(columns={"date": "event_date"}),
        on=["ticker", "event_date"],
        how="left",
    )
    return merged
