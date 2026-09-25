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

The pieces this module used to carry live beside it, split out to keep every
module inside the §4.3 fan-out budget with no behaviour change:

* event planning in :mod:`engine.v2.research._plan`;
* chain reads and the availability filter in :mod:`engine.v2.research._chains`;
* the per-trade record and the Tier-2 table in
  :mod:`engine.v2.research._trades_table`;
* the tool entrypoint (``run``) in :mod:`engine.v2.research._replay_run`.

The pricing primitives the legacy replay imported from ``engine.structures`` /
``engine.fills`` / ``engine.calendar`` live in
``engine.v2.research._pricing`` (supervisor decision: no legacy adapter).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from engine.v2.research._chains import (
    ChainIndex,
    filter_plan_by_availability,
    load_chain_index,
    read_chain_keys,
)
from engine.v2.research._plan import SKIP_REASONS, plan_events
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
)
from engine.v2.research._trades_table import _trade_record

__all__ = [
    "ALPHA_GRID",
    "SKIP_REASONS",
    "ReplayResult",
    "replay_one",
    "replay",
]

#: Fill alphas every replayed trade is priced at. Worst / mid / best are the
#: three the program reports side by side; the quarter points make the
#: degradation curve a lookup instead of an interpolation, and they are cheap —
#: pricing is a handful of arithmetic on rows already in memory.
ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def _log(message: str) -> None:
    print(f"  [replay] {message}", flush=True)


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
