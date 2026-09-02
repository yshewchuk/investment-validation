"""Deterministic as-of replay: events + a structure → priced trades.

This module is the single place where a structure meets real quotes and becomes
a trade. The scoring engine calls it to price a candidate at a decision date;
the backtest calls it to price every candidate in a year. **They call the same
function.** That is not a tidiness preference — the guide's load-bearing
acceptance test is that the live scorer and the research code cannot drift
apart, and the cheapest way to guarantee that is to leave nowhere for them to
drift to.

The replay is deliberately dumb. It does not decide *whether* to trade — no
gate, no model, no filter beyond "the chain exists and the structure resolves".
Selection lives in :mod:`engine.score`. Keeping the two apart is what lets the
analog layer draw on an unselected trade population, which the HANDOFF's first
recorded trap says is the difference between a real measurement and a statistic
conditioned on future information.

**Every trade is priced at a grid of fill alphas, not one.** The legacy trade
sets recorded a single worst-case fill, which is why their means read −16% to
−22% and why they cannot answer a question asked at mid. A row per (trade,
alpha) keeps the Tier-2 schema unchanged and makes the fill-degradation curve a
`groupby` rather than a re-run.
"""
from __future__ import annotations

import json
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.calendar import TradingCalendar, trading_calendar
from engine.data import store
from engine.fills import FillModel
from engine.structures import (
    STRUCTURES,
    ChainSnapshot,
    Structure,
    StructureError,
    price_structure,
    structure_return,
)

__all__ = [
    "ALPHA_GRID",
    "SKIP_REASONS",
    "ReplayPlan",
    "plan_events",
    "ChainIndex",
    "load_chain_index",
    "latest_chain_date",
    "replay_one",
    "replay",
    "ReplayResult",
    "to_trades_table",
]

#: Fill alphas every replayed trade is priced at. Worst / mid / best are the
#: three the program reports side by side; the quarter points make the
#: degradation curve a lookup instead of an interpolation, and they are cheap —
#: pricing is a handful of arithmetic on rows already in memory.
ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Why a planned trade produced no row. Counted rather than dropped: a replay
#: that silently loses 40% of its candidates is a replay whose headline number
#: is about the surviving 60%, and nobody can see which 60% that was.
SKIP_REASONS = (
    "no_entry_chain",
    "no_exit_chain",
    # Only ever non-zero for a structure decided before it enters: when the
    # decision close is the entry close, an event without a decision chain has
    # already been counted as `no_entry_chain`.
    "no_decision_chain",
    "structure_unresolved",
    "expiry_gone_at_exit",
    "bad_quote",
    "no_session",
    "calendar_out_of_range",
    "zero_cost",
)


def _log(message: str) -> None:
    print(f"  [replay] {message}", flush=True)


# --------------------------------------------------------------------------
# planning — pure calendar arithmetic, no quotes
# --------------------------------------------------------------------------


@dataclass
class ReplayPlan:
    """Which dates each event would be traded on, before any chain is touched.

    Separated from pricing because it is cheap, deterministic, and the thing the
    chain loader needs in order to know what to load. It is also independently
    checkable: a plan is wrong in ways (off-by-one sessions, entry after exit)
    that are much easier to see here than inside a P&L number.
    """

    frame: pd.DataFrame
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def chain_keys(self) -> set[tuple[str, pd.Timestamp]]:
        """Every (ticker, date) chain this plan needs loaded.

        Three dates, not two: a structure decided early needs the decision
        chain to quote the premium the board shows as well as the entry chain
        it actually fills on. It is a set, so for the usual case — decide and
        enter at the same close — the decision keys collapse into the entry
        keys and nothing extra is loaded.
        """
        keys = set(zip(self.frame["ticker"], self.frame["entry_date"]))
        keys |= set(zip(self.frame["ticker"], self.frame["exit_date"]))
        if "decision_date" in self.frame.columns:
            keys |= set(zip(self.frame["ticker"], self.frame["decision_date"]))
        return keys

    @property
    def years(self) -> list[int]:
        dates = pd.concat([self.frame["entry_date"], self.frame["exit_date"]])
        return sorted(pd.to_datetime(dates).dt.year.unique().tolist())


def plan_events(
    structure: Structure,
    events: pd.DataFrame,
    calendar: TradingCalendar | None = None,
) -> ReplayPlan:
    """Resolve every event's entry and exit dates for ``structure``.

    ``events`` needs ``ticker``, ``event_date`` and ``session``. Events whose
    session is unknown are skipped rather than defaulted: the session decides
    which close is information-free, and guessing it is the one-day error that
    makes a backtest look brilliant.
    """
    cal = calendar or trading_calendar()
    rows: list[dict] = []
    skipped = {reason: 0 for reason in SKIP_REASONS}

    for event in events.itertuples(index=False):
        session = getattr(event, "session", None)
        if session is None or (isinstance(session, float) and np.isnan(session)) or pd.isna(session):
            skipped["no_session"] += 1
            continue
        event_date = pd.Timestamp(event.event_date).normalize()
        try:
            window = cal.resolve_offsets(
                event_date, str(session), structure.entry_offset, structure.exit_offset,
                decision_offset=structure.decision_offset,
            )
        except KeyError:
            skipped["calendar_out_of_range"] += 1
            continue
        rows.append(
            {
                "event_id": getattr(event, "event_id", f"{event.ticker}_{event_date.date()}"),
                "ticker": str(event.ticker),
                "event_date": event_date,
                "session": str(session),
                "decision_date": window.decision_date,
                "entry_date": window.entry_date,
                "exit_date": window.exit_date,
                "last_pre_print": window.last_pre_print,
                "first_post_print": window.first_post_print,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "event_id", "ticker", "event_date", "session", "decision_date",
            "entry_date", "exit_date", "last_pre_print", "first_post_print",
        ],
    )
    if len(frame):
        frame = frame.sort_values(["ticker", "event_date"]).reset_index(drop=True)
    return ReplayPlan(frame=frame, skipped=skipped)


# --------------------------------------------------------------------------
# chain access
# --------------------------------------------------------------------------

_CHAIN_COLUMNS = (
    "ticker", "obs_date", "expiry", "dte", "strike", "right",
    "bid", "ask", "spot", "quote_repaired",
)


class ChainIndex:
    """``(ticker, obs_date)`` → chain rows, loaded once for a whole replay.

    The Tier-2 chain table is 15.3M rows. A replay touching 40k (ticker, date)
    pairs cannot afford a filtered read per pair, and cannot afford to hold the
    whole table either. So the plan is computed first, the needed keys are known
    before any I/O, and each year partition is filtered down to those keys as it
    is read — peak memory is the kept subset, not the table.
    """

    def __init__(self, groups: Mapping[tuple[str, pd.Timestamp], pd.DataFrame]):
        self._groups = dict(groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __contains__(self, key) -> bool:
        return (str(key[0]), pd.Timestamp(key[1]).normalize()) in self._groups

    def get(self, ticker: str, obs_date) -> pd.DataFrame | None:
        return self._groups.get((str(ticker), pd.Timestamp(obs_date).normalize()))

    @property
    def keys(self):
        return self._groups.keys()


_AVAILABLE_KEYS: set[tuple[str, pd.Timestamp]] | None = None


def available_chain_keys(refresh: bool = False) -> set[tuple[str, pd.Timestamp]]:
    """Every ``(ticker, obs_date)`` the Tier-2 chain table holds.

    Two columns over 15.3M rows, so it costs ~17s and ~10 MB — cheap enough to
    do once and worth doing, because it lets a replay discard the events it
    could never price *before* loading a single quote. Without it a STR-THRU
    replay loads 48k entry chains to use 17.8k of them.
    """
    global _AVAILABLE_KEYS
    if _AVAILABLE_KEYS is None or refresh:
        keys: set[tuple[str, pd.Timestamp]] = set()
        for _, frame in store.iter_table("option_chains", columns=["ticker", "obs_date"]):
            keys |= set(zip(frame["ticker"].astype(str), pd.to_datetime(frame["obs_date"])))
        _AVAILABLE_KEYS = keys
    return _AVAILABLE_KEYS


def filter_plan_by_availability(
    plan: ReplayPlan, available: set[tuple[str, pd.Timestamp]] | None = None
) -> ReplayPlan:
    """Drop planned events whose decision, entry or exit chain is not in the store.

    The dropped events are counted under the same skip reasons pricing would
    have used, so the accounting is identical either way — this only moves the
    discovery earlier, where it is free.

    The decision chain is checked because a structure decided early cannot be
    replayed without it: the premium the board would have quoted comes from
    that close. Where the decision close *is* the entry close the two checks
    are the same check, and `no_decision_chain` stays zero.
    """
    if plan.frame.empty:
        return plan
    keys = available if available is not None else available_chain_keys()
    frame = plan.frame
    has_entry = np.array(
        [(t, d) in keys for t, d in zip(frame["ticker"], frame["entry_date"])]
    )
    has_exit = np.array(
        [(t, d) in keys for t, d in zip(frame["ticker"], frame["exit_date"])]
    )
    if "decision_date" in frame.columns:
        has_decision = np.array(
            [(t, d) in keys for t, d in zip(frame["ticker"], frame["decision_date"])]
        )
    else:
        has_decision = np.ones(len(frame), dtype=bool)
    skipped = dict(plan.skipped)
    skipped["no_entry_chain"] = skipped.get("no_entry_chain", 0) + int((~has_entry).sum())
    skipped["no_exit_chain"] = skipped.get("no_exit_chain", 0) + int(
        (has_entry & ~has_exit).sum()
    )
    skipped["no_decision_chain"] = skipped.get("no_decision_chain", 0) + int(
        (has_entry & has_exit & ~has_decision).sum()
    )
    keep = has_entry & has_exit & has_decision
    return ReplayPlan(frame=frame[keep].reset_index(drop=True), skipped=skipped)


_CHAIN_DATES_BY_TICKER: dict[str, list[pd.Timestamp]] | None = None


def latest_chain_date(ticker: str, on_or_before) -> pd.Timestamp | None:
    """The newest chain we hold for ``ticker`` at or before a date.

    For scoring an UPCOMING event: its entry date has not happened, so no chain
    for it can exist, and requiring one leaves the whole forward board unpriced.
    The newest chain we do hold is the honest substitute — it is strictly OLDER
    information, so it cannot leak, and the caller labels what it used.

    Built once from :func:`available_chain_keys` and cached, because the caller
    is a board loop asking a few hundred times.
    """
    global _CHAIN_DATES_BY_TICKER
    if _CHAIN_DATES_BY_TICKER is None:
        by_ticker: dict[str, list[pd.Timestamp]] = {}
        for t, d in available_chain_keys():
            by_ticker.setdefault(str(t), []).append(pd.Timestamp(d))
        for dates in by_ticker.values():
            dates.sort()
        _CHAIN_DATES_BY_TICKER = by_ticker

    dates = _CHAIN_DATES_BY_TICKER.get(str(ticker))
    if not dates:
        return None
    cutoff = pd.Timestamp(on_or_before).normalize()
    position = bisect_right(dates, cutoff)
    return dates[position - 1] if position else None


def load_chain_index(
    keys: Iterable[tuple[str, pd.Timestamp]],
    *,
    years: Iterable[int] | None = None,
    progress_every: int = 1,
) -> ChainIndex:
    """Load exactly the chains a plan needs, one year partition at a time."""
    wanted = {(str(t), pd.Timestamp(d).normalize()) for t, d in keys}
    if not wanted:
        return ChainIndex({})
    if years is None:
        years = sorted({d.year for _, d in wanted})

    tickers = {t for t, _ in wanted}
    groups: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    started = time.time()
    for i, (year, frame) in enumerate(
        store.iter_table("option_chains", years=years, columns=list(_CHAIN_COLUMNS))
    ):
        frame = frame[frame["ticker"].isin(tickers)]
        if frame.empty:
            continue
        frame["obs_date"] = pd.to_datetime(frame["obs_date"])
        key_index = pd.MultiIndex.from_arrays([frame["ticker"], frame["obs_date"]])
        frame = frame[key_index.isin(wanted)]
        for key, chunk in frame.groupby(["ticker", "obs_date"], sort=False):
            groups[(str(key[0]), pd.Timestamp(key[1]))] = chunk.reset_index(drop=True)
        if progress_every and i % progress_every == 0:
            _log(
                f"chains {year}: {len(groups):,}/{len(wanted):,} keys resolved, "
                f"{time.time() - started:.0f}s"
            )
    _log(f"chain index: {len(groups):,} of {len(wanted):,} requested keys present")
    return ChainIndex(groups)


# --------------------------------------------------------------------------
# pricing one event
# --------------------------------------------------------------------------


def _clean(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop rows a pricing path must never see.

    ``FillModel`` raises on a NaN or crossed quote, by design — those are
    ingestion bugs, and turning them into a plausible number is worse than
    stopping. But a *chain* legitimately contains rows with no quote at all
    (strikes that did not trade), and the right response to those is to leave
    them out of strike selection rather than to abort the event.
    """
    ok = rows["bid"].notna() & rows["ask"].notna()
    ok &= rows["bid"] >= 0
    ok &= rows["ask"] >= 0
    ok &= rows["bid"] <= rows["ask"]
    ok &= rows["ask"] > 0
    return rows[ok]


def replay_one(
    structure: Structure,
    plan_row: Mapping,
    index: ChainIndex,
    *,
    alphas: Sequence[float] = ALPHA_GRID,
) -> tuple[list[dict], str | None]:
    """Price one planned event at every alpha. Returns ``(rows, skip_reason)``.

    The entry legs are resolved once and *pinned* for the exit, so the position
    is closed on the contracts it was opened on rather than on whatever is ATM
    after the print — the mistake that turns a straddle backtest into a
    measurement of nothing.
    """
    ticker = plan_row["ticker"]
    entry_rows = index.get(ticker, plan_row["entry_date"])
    if entry_rows is None or entry_rows.empty:
        return [], "no_entry_chain"
    exit_rows = index.get(ticker, plan_row["exit_date"])
    if exit_rows is None or exit_rows.empty:
        return [], "no_exit_chain"

    entry_rows = _clean(entry_rows)
    exit_rows = _clean(exit_rows)
    if entry_rows.empty or exit_rows.empty:
        return [], "bad_quote"

    entry_snap = ChainSnapshot(
        ticker=ticker,
        obs_date=plan_row["entry_date"],
        event_date=plan_row["event_date"],
        rows=entry_rows,
        session=plan_row["session"],
    )
    exit_snap = ChainSnapshot(
        ticker=ticker,
        obs_date=plan_row["exit_date"],
        event_date=plan_row["event_date"],
        rows=exit_rows,
        session=plan_row["session"],
    )

    rows: list[dict] = []
    pinned = None
    for alpha in alphas:
        fill = FillModel(float(alpha))
        try:
            entry = price_structure(structure, entry_snap, fill)
            # Pin from the first alpha's resolution: the contracts a structure
            # selects must not depend on the fill assumption, or the alpha sweep
            # would be comparing different trades.
            if pinned is None:
                pinned = entry.legs
            else:
                entry = price_structure(structure, entry_snap, fill, pin=pinned)
            exit_ = price_structure(
                structure, exit_snap, fill, pin=pinned, closing=True
            )
        except StructureError as exc:
            reason = (
                "expiry_gone_at_exit"
                if "pinned expiry" in str(exc)
                else "structure_unresolved"
            )
            return [], reason
        except ValueError:
            return [], "bad_quote"

        result = structure_return(entry, exit_)
        if result["cost"] <= 0:
            # A structure opened for a credit has no return-on-debit, and every
            # metric downstream is quoted on the debit. CAL-P can legitimately
            # price at a credit; it is skipped here and counted, not booked with
            # a meaningless denominator.
            return [], "zero_cost"

        rows.append(
            {
                "event_id": plan_row["event_id"],
                "ticker": ticker,
                "event_date": plan_row["event_date"],
                "session": plan_row["session"],
                "entry_date": plan_row["entry_date"],
                "exit_date": plan_row["exit_date"],
                "fill_alpha": float(alpha),
                "entry_cost": result["cost"],
                "exit_value": result["exit_value"],
                "pnl": result["pnl"],
                "ret": result["ret"],
                "spot_entry": entry.spot,
                "spot_exit": exit_.spot,
                "strike": entry.legs[0].strike,
                "expiry": entry.legs[0].expiry,
                "dte_entry": int(entry.legs[0].dte),
                "n_legs": len(entry.legs),
                "wide_market": entry.any_wide_market or exit_.any_wide_market,
                "quote_repaired": bool(
                    entry_rows.get("quote_repaired", pd.Series(dtype=bool)).any()
                    or exit_rows.get("quote_repaired", pd.Series(dtype=bool)).any()
                ),
                # The Tier-2 schema has no column for spot, and every consumer
                # that quotes a value per unit of spot (the payoff fit, the
                # moneyness bucket) needs the one the trade was actually priced
                # against — not a spot re-read later from a different table.
                "legs": json.dumps(
                    {
                        "spot_entry": entry.spot,
                        "spot_exit": exit_.spot,
                        "dte_entry": int(entry.legs[0].dte),
                        "entry": entry.to_dict()["legs"],
                        "exit": exit_.to_dict()["legs"],
                    },
                    default=str,
                ),
            }
        )
    return rows, None


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


@dataclass
class ReplayResult:
    strategy: str
    variant: str
    trades: pd.DataFrame
    skipped: dict[str, int]
    planned: int
    replayable: int = 0
    elapsed_s: float = 0.0

    @property
    def n_trades(self) -> int:
        """Distinct events priced (not rows — there is one row per alpha)."""
        return int(self.trades["event_id"].nunique()) if len(self.trades) else 0

    @property
    def coverage(self) -> float:
        """Priced share of every event the calendar resolved.

        Quoted against the full planned universe, not the chain-covered subset,
        because the gap between them is a real limit on what the evidence covers
        — 18% for STR-THRU — and a ratio that hides it would read as 100%.
        """
        return self.n_trades / self.planned if self.planned else 0.0

    @property
    def fill_rate(self) -> float:
        """Priced share of the events that *had* both chains."""
        return self.n_trades / self.replayable if self.replayable else 0.0

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "variant": self.variant,
            "planned": self.planned,
            "replayable": self.replayable,
            "priced": self.n_trades,
            "rows": int(len(self.trades)),
            "coverage": round(self.coverage, 4),
            "fill_rate": round(self.fill_rate, 4),
            "skipped": {k: v for k, v in sorted(self.skipped.items()) if v},
            "elapsed_s": round(self.elapsed_s, 1),
        }


def replay(
    strategy: str,
    events: pd.DataFrame,
    *,
    structure: Structure | None = None,
    variant: str | None = None,
    alphas: Sequence[float] = ALPHA_GRID,
    calendar: TradingCalendar | None = None,
    index: ChainIndex | None = None,
    progress_every: int = 2000,
) -> ReplayResult:
    """Plan, load, and price every event for one strategy.

    ``structure`` defaults to the registered factory for ``strategy``; pass one
    explicitly to replay a parameter variant (a different back DTE, an earlier
    entry day) without touching the registry.
    """
    started = time.time()
    if structure is None:
        if strategy not in STRUCTURES:
            raise KeyError(f"unknown strategy {strategy!r}; known: {sorted(STRUCTURES)}")
        structure = STRUCTURES[strategy]()
    variant = variant or _variant_label(structure)

    plan = plan_events(structure, events, calendar=calendar)
    _log(
        f"{strategy}/{variant}: planned {len(plan.frame):,} of {len(events):,} events "
        f"({plan.skipped.get('no_session', 0):,} without a session)"
    )
    planned_total = int(len(plan.frame))
    if index is None and len(plan.frame):
        plan = filter_plan_by_availability(plan)
        _log(f"{strategy}/{variant}: {len(plan.frame):,} events have both chains")
    if plan.frame.empty:
        return ReplayResult(strategy, variant, _empty_trades(), plan.skipped,
                            planned_total, 0, time.time() - started)

    if index is None:
        index = load_chain_index(plan.chain_keys)

    rows: list[dict] = []
    skipped = dict(plan.skipped)
    for i, plan_row in enumerate(plan.frame.to_dict("records")):
        priced, reason = replay_one(structure, plan_row, index, alphas=alphas)
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
        rows.extend(priced)
        if progress_every and i and i % progress_every == 0:
            _log(
                f"{strategy}: {i:,}/{len(plan.frame):,} events, "
                f"{len(rows):,} rows, {time.time() - started:.0f}s"
            )

    trades = pd.DataFrame(rows) if rows else _empty_trades()
    if len(trades):
        trades.insert(0, "strategy", strategy)
        trades.insert(1, "variant", variant)
        trades = trades.sort_values(["ticker", "event_date", "fill_alpha"]).reset_index(drop=True)

    result = ReplayResult(
        strategy=strategy,
        variant=variant,
        trades=trades,
        skipped=skipped,
        planned=planned_total,
        replayable=int(len(plan.frame)),
        elapsed_s=time.time() - started,
    )
    _log(
        f"{strategy}/{variant}: priced {result.n_trades:,} events — "
        f"{result.fill_rate:.1%} of the {result.replayable:,} with chains, "
        f"{result.coverage:.1%} of the {result.planned:,} planned, "
        f"in {result.elapsed_s:.0f}s"
    )
    return result


def _variant_label(structure: Structure) -> str:
    """Stable, human-readable parameterization key for a structure.

    The decision offset appears only when it differs from the entry offset, so
    every label written before decision offsets existed still means exactly what
    it meant then. Once it does differ the label must carry it: a T−2 book and a
    T−1 book are different trade sets, and `e+0x+1` cannot be allowed to name
    both of them in the `trades` table.
    """
    parts = [f"e{structure.entry_offset:+d}", f"x{structure.exit_offset:+d}"]
    if structure.decided_early:
        parts.append(f"d{structure.decided_at:+d}")
    for key in sorted(structure.params):
        value = structure.params[key]
        if value is not None:
            parts.append(f"{key}={value}")
    return "_".join(parts)


def _empty_trades() -> pd.DataFrame:
    columns = [
        "strategy", "variant", "event_id", "ticker", "event_date", "session",
        "entry_date", "exit_date", "fill_alpha", "entry_cost", "exit_value",
        "pnl", "ret", "spot_entry", "spot_exit", "strike", "expiry",
        "dte_entry", "n_legs", "wide_market", "quote_repaired", "legs",
    ]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


# --------------------------------------------------------------------------
# Tier-2 handoff
# --------------------------------------------------------------------------


def legs_spot_dte(trades: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Recover entry spot and DTE from the stored ``legs`` blob.

    The Tier-2 ``trades`` schema has no column for either, so :func:`replay_one`
    records both in the JSON when it prices the trade. It lives here because
    this module wrote the blob and is therefore the module that should know how
    to read it — every consumer that needs a value per unit of spot (the payoff
    fit, the analog moneyness bucket, the gate's premium feature) calls this
    rather than re-deriving spot from ``daily_market`` and getting a subtly
    different number wherever a name's daily coverage has a gap.
    """
    spots = np.full(len(trades), np.nan)
    dtes = np.full(len(trades), np.nan)
    if "legs" not in trades.columns:
        return pd.Series(spots, index=trades.index), pd.Series(dtes, index=trades.index)

    for i, blob in enumerate(trades["legs"].to_numpy()):
        if not isinstance(blob, str):
            continue
        try:
            doc = json.loads(blob)
        except ValueError:
            continue
        if not isinstance(doc, dict):
            continue
        spots[i] = _as_float(doc.get("spot_entry"))
        dte = doc.get("dte_entry")
        if dte is None:
            legs = doc.get("entry") or []
            dte = legs[0].get("dte") if legs else None
        dtes[i] = _as_float(dte)
    return pd.Series(spots, index=trades.index), pd.Series(dtes, index=trades.index)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_trades_table(results: Sequence[ReplayResult]) -> pd.DataFrame:
    """Shape replay output into the Tier-2 ``trades`` schema.

    ``trade_id`` carries the alpha, because the schema's primary key is the
    trade id and one event priced at five alphas is five rows. Making the alpha
    part of the identity — rather than, say, adding it to the key — keeps the
    Phase 0 schema untouched.
    """
    frames = []
    for result in results:
        if not len(result.trades):
            continue
        trades = result.trades
        out = pd.DataFrame(
            {
                "trade_id": (
                    trades["strategy"] + ":" + trades["variant"] + ":"
                    + trades["ticker"] + ":"
                    + pd.to_datetime(trades["event_date"]).dt.strftime("%Y%m%d") + ":a"
                    + (trades["fill_alpha"].astype(float) * 100).round().astype(int).astype(str)
                ),
                "kind": "sim",
                "strategy": trades["strategy"],
                "variant": trades["variant"],
                "ticker": trades["ticker"],
                "event_id": trades["event_id"],
                "event_date": pd.to_datetime(trades["event_date"]),
                "year": pd.to_datetime(trades["event_date"]).dt.year,
                "legs": trades["legs"],
                "entry_date": pd.to_datetime(trades["entry_date"]),
                "exit_date": pd.to_datetime(trades["exit_date"]),
                "strike": trades["strike"].astype(float),
                "expiry": pd.to_datetime(trades["expiry"]),
                "fill_alpha": trades["fill_alpha"].astype(float),
                "entry_cost": trades["entry_cost"].astype(float),
                "exit_value": trades["exit_value"].astype(float),
                "ret": trades["ret"].astype(float),
                "provenance": "engine.replay",
            }
        )
        frames.append(out)
    if not frames:
        from engine.data.schemas import empty_frame

        return empty_frame("trades")
    return pd.concat(frames, ignore_index=True)
