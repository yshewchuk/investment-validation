"""Option A of the 2026-09-29 decision (EXP-182-style parallel path): a new v2
trades loader for experiments, migrating nothing yet. ``snapshot_id`` is always
explicit -- never a scope-head resolve. ``experiments/common.py::load_engine_trades``
keeps serving every existing caller unchanged; this module has none yet."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.research import _pricing, experiment_trades
from engine.v2.research._chains import ChainIndex, load_chain_index
from engine.v2.research.replay import replay_one

__all__ = ["load_v2_trades", "make_v2_repricer", "load_v2_chain_quotes"]

#: The shift magnitudes engine/evaluate.py's stress stages request from a
#: repricer built here (stress_slippage days=(-1, 1); stress_stale_dates
#: shift=1). A repricer's first call eagerly folds in chain keys for the
#: OTHER shifts in this tuple, over that call's own trade population, so a
#: later call at a different shift finds its keys already cached instead of
#: triggering its own chain scan. Purely a cache-priming hint: a shift
#: outside this tuple is still priced correctly, it just pays for its own
#: scan on first use.
_KNOWN_STRESS_SHIFT_DAYS: tuple[int, ...] = (-1, 1)


def load_v2_trades(strategy: str, *, catalog: Path | str, store_root: Path | str,
                   snapshot_id: str, as_of_month=None, purpose="training",
                   event_ids=None) -> pd.DataFrame:
    """The committed v2-replay trades for ``strategy``, one pinned snapshot.
    SNAPSHOT_NOT_FOUND -- unknown ``snapshot_id`` (``Repository.resolve``);
    CONTRACT_MISMATCH -- that snapshot carries no ``trades`` table;
    POPULATION_COLLAPSED -- no row for this strategy/provenance pair."""
    conn = open_catalog(Path(catalog), clock=SystemClock())
    try:
        repository = Repository(conn, ArtifactStore(Path(store_root)))
        return experiment_trades.load_trades(
            repository, repository.resolve(snapshot_id), strategy,
            as_of_month=as_of_month, purpose=purpose, event_ids=event_ids)
    finally:
        conn.close()


def make_v2_repricer(
    strategy: str,
    *,
    catalog,
    store_root,
    snapshot_id: str,
    structure=None,
    alpha: float = 0.5,
):
    """The v2 analog of :func:`experiments.common.make_repricer`.

    It reads the pinned v2 snapshot's ``option_chains``/``daily_market`` tables
    instead of the legacy mutable store, so a v2-trades run's T±1 slippage and
    stale-date stress checks never mix a pinned snapshot with the legacy live
    store. ``snapshot_id`` is required with no default and is never resolved as
    "latest".
    """
    conn = open_catalog(Path(catalog), clock=SystemClock())
    try:
        repository = Repository(conn, ArtifactStore(Path(store_root)))
        snapshot = repository.resolve(snapshot_id)
        cal = _pricing.trading_calendar_from_snapshot(repository, snapshot)
    finally:
        conn.close()

    struct = structure or _pricing.STRUCTURES[strategy]()
    #: (ticker, obs_date) -> matched chain rows, or None once that key has
    #: been scanned for and found absent. Persists across every call this
    #: closure makes (one call per stress shift), so the snapshot's chains
    #: are scanned at most once per key ever needed, never once per call.
    chain_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame | None] = {}

    def _plan_and_keys(t: pd.DataFrame, shift: int):
        plan_rows: list[dict | None] = []
        keys: set[tuple[str, pd.Timestamp]] = set()
        for row in t.itertuples(index=False):
            try:
                entry = cal.shift(pd.Timestamp(row.entry_date), int(shift))
                exit_ = cal.shift(pd.Timestamp(row.exit_date), int(shift))
            except KeyError:
                plan_rows.append(None)
                continue
            plan_rows.append({
                "event_id": row.event_id,
                "ticker": row.ticker,
                "event_date": pd.Timestamp(row.event_date),
                "session": row.session,
                "entry_date": entry,
                "exit_date": exit_,
            })
            keys.add((row.ticker, entry))
            keys.add((row.ticker, exit_))
        return plan_rows, keys

    def repricer(trades: pd.DataFrame, shift_days: int) -> pd.DataFrame:
        t = trades.reset_index(drop=True)
        plan_rows, keys = _plan_and_keys(t, int(shift_days))

        union_keys = set(keys)
        if not chain_cache:
            # First call ever on this repricer: also fold in this trade
            # population's keys at the OTHER known stress shifts, so a
            # later call at a different shift reuses this one scan.
            for other_shift in _KNOWN_STRESS_SHIFT_DAYS:
                if other_shift == int(shift_days):
                    continue
                _, extra_keys = _plan_and_keys(t, other_shift)
                union_keys |= extra_keys

        missing = union_keys - chain_cache.keys()
        started = time.time()
        if missing:
            chain_conn = open_catalog(Path(catalog), clock=SystemClock())
            try:
                chain_repository = Repository(chain_conn, ArtifactStore(Path(store_root)))
                fetched = load_chain_index(chain_repository, snapshot, missing)
            finally:
                chain_conn.close()
            for key in missing:
                chain_cache[key] = fetched.get(*key)
        index = ChainIndex({
            key: chain_cache[key] for key in keys if chain_cache.get(key) is not None
        })
        print(
            f"  [repricer] {strategy} shift {int(shift_days):+d}d: "
            f"{len(index):,}/{len(keys):,} shifted chains loaded in {time.time() - started:.0f}s "
            f"({len(missing):,} new keys scanned, {len(chain_cache):,} cached total)",
            flush=True,
        )

        out_rows: list[dict] = []
        for plan_row in plan_rows:
            if plan_row is None:
                continue
            priced, _reason = replay_one(struct, plan_row, index, alphas=(alpha,))
            out_rows.extend(priced)

        out = pd.DataFrame(out_rows) if out_rows else t.iloc[0:0].copy()
        coverage = float(len(out) / len(t)) if len(t) else float("nan")
        out.attrs["coverage"] = coverage
        return out

    return repricer


def load_v2_chain_quotes(
    keys: Iterable[tuple[str, pd.Timestamp]], *,
    catalog: Path | str, store_root: Path | str, snapshot_id: str,
) -> ChainIndex:
    """Raw ``option_chains`` quotes for ``(ticker, obs_date)`` ``keys``, read
    from one pinned v2 snapshot. Read-only infrastructure for a planned
    future experiment; it has no caller yet. A pure pass-through with NO new
    error handling: SNAPSHOT_NOT_FOUND -- unknown ``snapshot_id``
    (``Repository.resolve``, as in :func:`load_v2_trades`); missing or
    unavailable keys, an empty ``keys``, or any other read failure raise
    (or return) whatever ``load_chain_index`` already does. The catalog
    connection is closed in ``finally`` even when ``load_chain_index``
    raises."""
    conn = open_catalog(Path(catalog), clock=SystemClock())
    try:
        repository = Repository(conn, ArtifactStore(Path(store_root)))
        snapshot = repository.resolve(snapshot_id)
        return load_chain_index(repository, snapshot, set(keys))
    finally:
        conn.close()
