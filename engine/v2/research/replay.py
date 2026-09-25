"""Deterministic as-of replay over one pinned snapshot — native rewrite.

This is the slice-7 move of ``engine/replay.py`` onto
``engine.v2.data.Repository``. The pure cores (planning, chain indexing,
single-event pricing, result shaping) are byte-identical to the legacy module
so the v2 path and the legacy board cannot drift; only the store-reaching
edges are rewritten: ``store.iter_table("option_chains", ...)`` becomes
``engine.v2.research._snapshot.read_table`` against one resolved
``SnapshotRef``, and the module-level availability caches are gone (a pinned
snapshot never changes under a run, so there is nothing to invalidate and no
reason to hold global state).

The pricing primitives the legacy replay imported from ``engine.structures`` /
``engine.fills`` / ``engine.calendar`` live in
``engine.v2.research._pricing`` (supervisor decision: no legacy adapter).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from engine.v2.research import _snapshot
from engine.v2.research._pricing import (
    MIN_MEANINGFUL_COST,
    STRUCTURES,
    ChainSnapshot,
    FillModel,
    Structure,
    StructureError,
    TradingCalendar,
    execution_variant_label,
    price_structure,
    structure_return,
    trading_calendar,
)
from engine.v2.research._snapshot import read_table

__all__ = [
    "ALPHA_GRID",
    "SKIP_REASONS",
    "ReplayPlan",
    "plan_events",
    "ChainIndex",
    "read_chain_keys",
    "read_chains_for_years",
    "load_chain_index",
    "filter_plan_by_availability",
    "replay_one",
    "replay",
    "ReplayResult",
    "legs_spot_dte",
    "legs_exit_spot",
    "to_trades_table",
    "run",
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
    """Which dates each event would be traded on, before any chain is touched."""

    frame: pd.DataFrame
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def chain_keys(self) -> set[tuple[str, pd.Timestamp]]:
        """Every (ticker, date) chain this plan needs loaded."""
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
    """Resolve every event's entry and exit dates for ``structure``."""
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
# chain access — rewritten over Repository/read_table
# --------------------------------------------------------------------------

_CHAIN_COLUMNS = (
    "ticker", "obs_date", "expiry", "dte", "strike", "right",
    "bid", "ask", "delta", "spot", "quote_repaired",
)


class ChainIndex:
    """``(ticker, obs_date)`` → chain rows, loaded once for a whole replay."""

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


def read_chain_keys(repository, snapshot_ref) -> set[tuple[str, pd.Timestamp]]:
    """Every (ticker, obs_date) the snapshot's option_chains table holds.

    Rewrite of ``engine/replay.py``'s ``available_chain_keys`` — no
    module-level cache (that cache is a legacy hot-reload guard tied to a
    mutable store; a pinned snapshot never changes under a run, so there is
    nothing to invalidate and no reason to hold global state).
    """
    frame = read_table(repository, snapshot_ref, "option_chains", ("ticker", "obs_date"))
    return set(zip(frame["ticker"].astype(str), pd.to_datetime(frame["obs_date"])))


def read_chains_for_years(repository, snapshot_ref, years) -> pd.DataFrame:
    """option_chains rows for ``years``, projected to ``_CHAIN_COLUMNS``."""
    return read_table(repository, snapshot_ref, "option_chains", _CHAIN_COLUMNS,
                      partition_keys=[str(y) for y in years])


def load_chain_index(repository, snapshot_ref, keys) -> ChainIndex:
    """Load exactly the chains a plan needs, one year partition at a time.

    Rewrite of ``engine/replay.py``'s ``load_chain_index``. ``keys`` is
    REQUIRED (no store-wide default); years are derived from ``keys``,
    matching the legacy function's own
    ``years = sorted({d.year for _, d in wanted})``.
    """
    wanted = {(str(t), pd.Timestamp(d).normalize()) for t, d in keys}
    if not wanted:
        return ChainIndex({})
    years = sorted({d.year for _, d in wanted})
    tickers = {t for t, _ in wanted}
    frame = read_chains_for_years(repository, snapshot_ref, years)
    frame = frame[frame["ticker"].isin(tickers)]
    frame["obs_date"] = pd.to_datetime(frame["obs_date"])
    key_index = pd.MultiIndex.from_arrays([frame["ticker"], frame["obs_date"]])
    frame = frame[key_index.isin(wanted)]
    groups = {(str(k[0]), pd.Timestamp(k[1])): g.reset_index(drop=True)
              for k, g in frame.groupby(["ticker", "obs_date"], sort=False)}
    _log(f"chain index: {len(groups):,} of {len(wanted):,} requested keys present")
    return ChainIndex(groups)


def filter_plan_by_availability(plan: ReplayPlan, available: set) -> ReplayPlan:
    """Drop planned events whose decision, entry or exit chain is not present.

    Rewrite of ``engine/replay.py``'s ``filter_plan_by_availability`` with
    ``available`` required: the legacy default was the store-reaching
    ``available_chain_keys()`` call this slice removes. A v2 caller resolves
    it once via :func:`read_chain_keys` and passes it explicitly.
    """
    if plan.frame.empty:
        return plan
    keys = available
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


# The legacy module's `latest_chain_date` / `_CHAIN_DATES_BY_TICKER` are not
# moved: they exist only for scoring an UPCOMING event against the newest held
# chain, which is legacy-score board behavior, not replay over a pinned
# snapshot. Nothing in this slice reads them.


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


def _chain_rows(
    structure: Structure, plan_row: Mapping, index: ChainIndex
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None,
           str | None]:
    """Stage 1: the chain rows one planned event prices against, cleaned.

    Returns ``(decision_rows, entry_rows, exit_rows, skip_reason)``; the
    decision rows are ``None`` unless the structure is decided early.
    """
    ticker = plan_row["ticker"]
    decision_date = plan_row.get("decision_date", plan_row["entry_date"])
    decision_rows = None
    if structure.decided_early:
        decision_rows = index.get(ticker, decision_date)
        if decision_rows is None or decision_rows.empty:
            return None, None, None, "no_decision_chain"
    entry_rows = index.get(ticker, plan_row["entry_date"])
    if entry_rows is None or entry_rows.empty:
        return None, None, None, "no_entry_chain"
    exit_rows = index.get(ticker, plan_row["exit_date"])
    if exit_rows is None or exit_rows.empty:
        return None, None, None, "no_exit_chain"

    entry_rows = _clean(entry_rows)
    exit_rows = _clean(exit_rows)
    if entry_rows.empty or exit_rows.empty:
        return None, None, None, "bad_quote"
    return decision_rows, entry_rows, exit_rows, None


def _chain_snapshot(
    ticker: str, obs_date, plan_row: Mapping, rows: pd.DataFrame
) -> ChainSnapshot:
    """The pricing view of one cleaned chain at one as-of date."""
    return ChainSnapshot(
        ticker=ticker,
        obs_date=obs_date,
        event_date=plan_row["event_date"],
        rows=rows,
        session=plan_row["session"],
    )


def _decision_price(
    structure: Structure,
    ticker: str,
    plan_row: Mapping,
    decision_date,
    decision_rows: pd.DataFrame,
) -> tuple[list | None, float, float, int | None, str | None]:
    """Stage 2: name the contracts on the DECISION chain, priced at mid.

    Returns ``(legs, cost, spot, dte, skip_reason)``.
    """
    decision_rows = _clean(decision_rows)
    if decision_rows.empty:
        return None, float("nan"), float("nan"), None, "bad_quote"
    decision_snap = _chain_snapshot(ticker, decision_date, plan_row, decision_rows)
    try:
        quoted = price_structure(structure, decision_snap, FillModel(0.5))
    except (StructureError, ValueError):
        return None, float("nan"), float("nan"), None, "structure_unresolved"
    # DTE seen from the decision close. `dte_entry` is the gate's dominant
    # feature (EXP-114, -0.353), and at the decision it is `dte_entry + 1`
    # by construction — leaving the gate reading the entry value would be a
    # one-day leak in the feature it leans on hardest.
    return quoted.legs, quoted.cost, quoted.spot, int(quoted.legs[0].dte), None


def replay_one(
    structure: Structure,
    plan_row: Mapping,
    index: ChainIndex,
    *,
    alphas: Sequence[float] = ALPHA_GRID,
    include_legs: bool = False,
) -> tuple[list[dict], str | None]:
    """Price one planned event at every alpha. Returns ``(rows, skip_reason)``."""
    ticker = plan_row["ticker"]
    decision_date = plan_row.get("decision_date", plan_row["entry_date"])
    decision_rows, entry_rows, exit_rows, reason = _chain_rows(
        structure, plan_row, index
    )
    if reason is not None:
        return [], reason

    entry_snap = _chain_snapshot(
        ticker, plan_row["entry_date"], plan_row, entry_rows
    )
    exit_snap = _chain_snapshot(
        ticker, plan_row["exit_date"], plan_row, exit_rows
    )

    # A structure decided early names its contract on the DECISION chain and
    # then buys that contract at the entry. Re-resolving ATM at the entry would
    # book a trade the decision never selected — and would quietly make the T−2
    # book a different strategy rather than the same one decided sooner.
    pinned = None
    quoted_cost = float("nan")
    spot_decision = float("nan")
    dte_decision = None
    if structure.decided_early:
        pinned, quoted_cost, spot_decision, dte_decision, reason = _decision_price(
            structure, ticker, plan_row, decision_date, decision_rows
        )
        if reason is not None:
            return [], reason

    return _price_alphas(
        structure, plan_row, entry_snap, exit_snap, entry_rows, exit_rows,
        pinned, quoted_cost, spot_decision, dte_decision, ticker,
        decision_date, alphas, include_legs,
    )


def _price_alphas(
    structure: Structure,
    plan_row: Mapping,
    entry_snap: ChainSnapshot,
    exit_snap: ChainSnapshot,
    entry_rows: pd.DataFrame,
    exit_rows: pd.DataFrame,
    pinned: list | None,
    quoted_cost: float,
    spot_decision: float,
    dte_decision: int | None,
    ticker: str,
    decision_date,
    alphas: Sequence[float],
    include_legs: bool,
) -> tuple[list[dict], str | None]:
    """Stage 3: price the pinned structure at every alpha, one row each."""
    rows: list[dict] = []
    for alpha in alphas:
        fill = FillModel(float(alpha))
        try:
            entry = price_structure(structure, entry_snap, fill, pin=pinned)
            # Pin from the first alpha's resolution: the contracts a structure
            # selects must not depend on the fill assumption, or the alpha sweep
            # would be comparing different trades.
            if pinned is None:
                pinned = entry.legs
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
        if result["cost"] <= MIN_MEANINGFUL_COST:
            # A structure opened for a credit has no return-on-debit, and every
            # metric downstream is quoted on the debit. CAL-P can legitimately
            # price at a credit; it is skipped here and counted, not booked with
            # a meaningless denominator. The floor above zero catches the same
            # thing by another route: a multi-leg structure with offsetting
            # long/short legs (CND-P) can sum to a cost floating-point cannot
            # tell from zero — EXP-121 found trades costing 1e-17 to 2e-15,
            # nowhere near the next real price ($0.01) — and dividing pnl by
            # that noise produced returns in the quadrillions of percent.
            return [], "zero_cost"

        rows.append(
            _trade_record(
                plan_row, ticker, decision_date, alpha, quoted_cost,
                spot_decision, dte_decision, result, entry, exit_, entry_rows,
                exit_rows, include_legs,
            )
        )
    return rows, None


def _trade_record(
    plan_row: Mapping,
    ticker: str,
    decision_date,
    alpha: float,
    quoted_cost: float,
    spot_decision: float,
    dte_decision: int | None,
    result: Mapping,
    entry,
    exit_,
    entry_rows: pd.DataFrame,
    exit_rows: pd.DataFrame,
    include_legs: bool,
) -> dict:
    """Stage 4: the result record for one priced (event, alpha)."""
    return {
        "event_id": plan_row["event_id"],
        "ticker": ticker,
        "event_date": plan_row["event_date"],
        "session": plan_row["session"],
        "decision_date": decision_date,
        "entry_date": plan_row["entry_date"],
        "exit_date": plan_row["exit_date"],
        "fill_alpha": float(alpha),
        #: What the board would have QUOTED at the decision close, mid.
        #: NaN when the decision is the entry, where the two are the
        #: same number by construction.
        "quoted_cost": quoted_cost,
        "spot_decision": spot_decision,
        "dte_decision": dte_decision,
        "entry_cost": result["cost"],
        "exit_value": result["exit_value"],
        "pnl": result["pnl"],
        "ret": result["ret"],
        "spot_entry": entry.spot,
        "spot_exit": exit_.spot,
        **({"entry_legs": [
            {"name": leg.name, "right": leg.right, "side": leg.side,
             "qty": leg.qty, "strike": leg.strike, "expiry": leg.expiry,
             "bid": leg.bid, "ask": leg.ask, "price": leg.price}
            for leg in entry.legs
        ]} if include_legs else {}),
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
        """Priced share of every event the calendar resolved."""
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
    repository,
    snapshot_ref,
    strategy: str,
    events: pd.DataFrame,
    *,
    structure: Structure | None = None,
    variant: str | None = None,
    alphas: Sequence[float] = ALPHA_GRID,
    calendar: TradingCalendar | None = None,
    index: ChainIndex | None = None,
    progress_every: int = 2000,
    include_legs: bool = False,
) -> ReplayResult:
    """Plan, load, and price every event for one strategy.

    Rewrite of ``engine/replay.py``'s ``replay``: same orchestration body,
    with ``load_chain_index``/``filter_plan_by_availability``/
    ``read_chain_keys`` bound to ``(repository, snapshot_ref)`` in place of the
    legacy store calls.
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
        plan = filter_plan_by_availability(plan, read_chain_keys(repository, snapshot_ref))
        _log(f"{strategy}/{variant}: {len(plan.frame):,} events have both chains")
    if plan.frame.empty:
        return ReplayResult(strategy, variant, _empty_trades(), plan.skipped,
                            planned_total, 0, time.time() - started)

    if index is None:
        index = load_chain_index(repository, snapshot_ref, plan.chain_keys)

    rows: list[dict] = []
    skipped = dict(plan.skipped)
    for i, plan_row in enumerate(plan.frame.to_dict("records")):
        priced, reason = replay_one(structure, plan_row, index, alphas=alphas,
                                   include_legs=include_legs)
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
    """Stable, human-readable parameterization key for a structure."""
    return execution_variant_label(structure)


def _empty_trades() -> pd.DataFrame:
    columns = [
        "strategy", "variant", "event_id", "ticker", "event_date", "session",
        "decision_date", "entry_date", "exit_date", "fill_alpha",
        "quoted_cost", "spot_decision", "dte_decision", "entry_cost", "exit_value",
        "pnl", "ret", "spot_entry", "spot_exit", "strike", "expiry",
        "dte_entry", "n_legs", "wide_market", "quote_repaired", "legs",
    ]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


# --------------------------------------------------------------------------
# Tier-2 handoff
# --------------------------------------------------------------------------


def legs_spot_dte(trades: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Recover entry spot and DTE from the stored ``legs`` blob."""
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


def legs_exit_spot(trades: pd.DataFrame) -> pd.Series:
    """Recover the exit spot stored beside the pinned exit legs."""
    spots = np.full(len(trades), np.nan)
    if "legs" not in trades.columns:
        return pd.Series(spots, index=trades.index)
    for index, blob in enumerate(trades["legs"].to_numpy()):
        if not isinstance(blob, str):
            continue
        try:
            document = json.loads(blob)
        except ValueError:
            continue
        if isinstance(document, dict):
            spots[index] = _as_float(document.get("spot_exit"))
    return pd.Series(spots, index=trades.index)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_trades_table(results: Sequence[ReplayResult]) -> pd.DataFrame:
    """Shape replay output into the Tier-2 ``trades`` schema.

    ``trade_id`` carries the alpha, because the schema's primary key is the
    trade id and one event priced at five alphas is five rows.
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
        columns = [
            "trade_id", "kind", "strategy", "variant", "ticker", "event_id",
            "event_date", "year", "legs", "entry_date", "exit_date", "strike",
            "expiry", "fill_alpha", "entry_cost", "exit_value", "ret", "provenance",
        ]
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# the tool entrypoint
# --------------------------------------------------------------------------


def _events_frame(repository, snapshot_ref, years=None) -> pd.DataFrame:
    """The event universe for a replay: earnings_events with a known session.

    This is ``engine/build_trades.py``'s ``event_universe`` body split at its
    one store read (the same technique ``fill_quality.join_from_snapshot``
    used), with the v2 table read in place of the legacy one.
    """
    events = _snapshot.read_table(
        repository, snapshot_ref, "earnings_events",
        ["event_id", "ticker", "event_date", "session"],
    )
    events = events[events["session"].notna()].copy()
    events["event_date"] = pd.to_datetime(events["event_date"])
    if years is not None:
        wanted = {int(y) for y in years}
        events = events[events["event_date"].dt.year.isin(wanted)]
    return events.sort_values(["ticker", "event_date"]).reset_index(drop=True)


def run(repository, *, strategies: Sequence[str], events: pd.DataFrame,
        reports_dir: Path = Path("reports"), scope: str = _snapshot.DEFAULT_SCOPE,
        snapshot_id: str | None = None, stamp: str | None = None) -> dict:
    """Replay every strategy against one pinned snapshot and write a report.

    The returned ``trades`` frame is stamped with ``provenance =
    "engine.v2.research.replay"`` (overwriting the legacy ``engine.replay``
    marker ``to_trades_table`` writes) and with the snapshot id that produced
    it, so a v2 row can never be mistaken for a legacy-replay row.
    """
    snapshot = _snapshot.resolve_snapshot(repository, scope=scope, snapshot_id=snapshot_id)
    results = [replay(repository, snapshot, s, events) for s in strategies]
    trades = to_trades_table(results)
    if len(trades):
        trades["provenance"] = "engine.v2.research.replay"
        trades["snapshot_id"] = snapshot.snapshot_id

    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"replay_{stamp}.json"
    report = {
        "snapshot_id": snapshot.snapshot_id,
        "generated_at": stamp,
        "results": [result.as_dict() for result in results],
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return {
        "snapshot_id": snapshot.snapshot_id,
        "results": [result.as_dict() for result in results],
        "trades": trades,
        "path": str(path),
    }
