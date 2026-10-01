"""One-family v2 candidate-grid pricer — the PRICING step of the DYN-SV v2
refresh (GitHub issue #266), as its own module.

A swept generalisation of ``engine.v2.research.replay.replay()`` across several
grid-position ``steps`` values instead of one hardcoded default, against one
pinned v2 snapshot. Every (event, step, fill_alpha) priced row is emitted;
selection, simulation (``exp_pnl_sim``), and the forecast-dependent geometry
columns (``width_over_forecast``, ``anchor_over_spot``) are explicitly a later
slice and do not appear here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.research._chains import filter_plan_by_availability, read_chain_keys
from engine.v2.research._plan import plan_events
from engine.v2.research._pricing import (
    STRUCTURES,
    execution_variant_label,
    trading_calendar_from_snapshot,
)
from engine.v2.research.replay import ALPHA_GRID, replay_one
from experiments.common_v2 import load_v2_chain_quotes

__all__ = ["price_candidate_grid"]


def _empty_grid() -> pd.DataFrame:
    """The documented empty result: the two identity columns, zero rows."""
    return pd.DataFrame(
        {"strategy": pd.Series(dtype="object"), "steps": pd.Series(dtype="int64")}
    )


def price_candidate_grid(
    strategy: str, events: pd.DataFrame, *,
    catalog: Path | str, store_root: Path | str, snapshot_id: str,
    steps: Sequence[int] = (1, 2, 3, 4, 5, 6), alphas=ALPHA_GRID,
) -> pd.DataFrame:
    """Price ONE strategy family across grid-position ``steps`` on one snapshot.

    One ``plan_events`` call covers every step (the plan depends on the
    strategy's entry/exit/decision offsets, which are the same for every step
    of one strategy), and ONE batched
    :func:`experiments.common_v2.load_v2_chain_quotes` read covers the whole
    grid — never once per step. Empty plans and plans with no available chains
    return an empty DataFrame with columns ``["strategy", "steps"]``.

    A pure pass-through with NO new error handling:

    - unknown ``snapshot_id`` -- ``Repository.resolve``'s own error, as in
      :func:`experiments.common_v2.load_v2_chain_quotes`;
    - unknown ``strategy`` (not a ``STRUCTURES`` key) -- a plain ``KeyError``
      (never caught), the same behavior ``replay()`` already has;
    - an event a step cannot price -- whatever ``replay_one``'s existing skip
      behavior returns for that (step, plan_row) pair: the grid adds no new
      skip or error path, it just calls ``replay_one`` once per pair the way
      ``replay()`` calls it once per plan row.
    """
    conn = open_catalog(Path(catalog), clock=SystemClock())
    try:
        repository = Repository(conn, ArtifactStore(Path(store_root)))
        snapshot = repository.resolve(snapshot_id)
        calendar = trading_calendar_from_snapshot(repository, snapshot)
        probe = STRUCTURES[strategy](steps=int(steps[0]))
        plan = plan_events(probe, events, calendar=calendar)
        if not plan.frame.empty:
            plan = filter_plan_by_availability(
                plan, read_chain_keys(repository, snapshot)
            )
    finally:
        conn.close()

    if plan.frame.empty:
        return _empty_grid()

    index = load_v2_chain_quotes(
        plan.chain_keys, catalog=catalog, store_root=store_root,
        snapshot_id=snapshot_id,
    )
    plan_rows = plan.frame.to_dict("records")
    rows: list[dict] = []
    for step in steps:
        structure = STRUCTURES[strategy](steps=int(step))
        variant = execution_variant_label(structure)
        for plan_row in plan_rows:
            priced, _reason = replay_one(structure, plan_row, index, alphas=alphas)
            for row in priced:
                row["steps"] = int(step)
                row["variant"] = variant
                rows.append(row)

    if not rows:
        return _empty_grid()

    frame = pd.DataFrame(rows)
    frame.insert(0, "strategy", strategy)
    return frame
