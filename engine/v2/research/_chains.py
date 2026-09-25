"""Chain access — rewritten over ``Repository``/``read_table``.

Split out of ``engine.v2.research.replay`` (review blocker: module fan-out).
The bodies are unchanged from what replay.py carried; the store-reaching edge
is ``engine.v2.research._snapshot.read_table`` against one resolved
``SnapshotRef``, and the module-level availability caches are gone (a pinned
snapshot never changes under a run, so there is nothing to invalidate and no
reason to hold global state).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from engine.v2.research._plan import ReplayPlan
from engine.v2.research._snapshot import read_table

__all__ = [
    "ChainIndex",
    "filter_plan_by_availability",
    "load_chain_index",
    "read_chain_keys",
    "read_chains_for_years",
]

#: The ``option_chains`` projection every chain read uses.
_CHAIN_COLUMNS = (
    "ticker", "obs_date", "expiry", "dte", "strike", "right",
    "bid", "ask", "delta", "spot", "quote_repaired",
)


def _log(message: str) -> None:
    print(f"  [replay] {message}", flush=True)


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
