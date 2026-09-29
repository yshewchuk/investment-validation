"""Trades read for experiment consumers, over one pinned snapshot.

This is the research package's entrypoint for engine/v2 consumers outside
``tools/v2_*.py``: it lets ``experiments/common_v2.py`` read the committed
``trades`` dataset version ``tools/v2_build_trades.py`` publishes, without
the caller touching this package's internal read helpers directly. No CLI
of its own; no write path.
"""
from __future__ import annotations

import pandas as pd

from engine.v2.data import errors
from engine.v2.research._trades_publish import read_event_rows, read_existing_trades
from engine.v2.research._trades_revisions import PROVENANCE

__all__ = ["PROVENANCE", "load_trades"]


def load_trades(repository, snapshot, strategy: str) -> pd.DataFrame:
    """The committed v2-replay trade rows for ``strategy``, one pinned snapshot.

    Same frame contract as ``experiments.common.load_engine_trades``: every
    ``trades`` column plus a ``session`` column joined from
    ``earnings_events`` on ``event_id``. Filters to this package's own
    ``PROVENANCE`` tag (never a legacy ``engine.replay`` row the same
    snapshot might also carry). ``event_date``, ``entry_date`` and
    ``exit_date`` are returned as ``pd.Timestamp`` (via ``pd.to_datetime``),
    matching the legacy loader.

    Raises ``engine.v2.data.errors.DataError``:
    - ``CONTRACT_MISMATCH`` — the snapshot has no ``trades`` table at all
      (raised by ``read_existing_trades`` itself).
    - ``POPULATION_COLLAPSED`` — the snapshot's ``trades`` table has no row
      for this ``strategy``/``PROVENANCE`` pair.

    Does not itself resolve a snapshot id and does not itself decide "missing
    snapshot" — that is ``Repository.resolve``'s ``SNAPSHOT_NOT_FOUND``,
    raised by the caller's own resolve call before this function is reached
    (see ``experiments/common_v2.py``).
    """
    trades = read_existing_trades(repository, snapshot)
    rows = trades[
        (trades["strategy"] == strategy) & (trades["provenance"] == PROVENANCE)
    ].reset_index(drop=True)
    if rows.empty:
        raise errors.fail(
            "POPULATION_COLLAPSED",
            "no trades rows for this strategy/provenance in the resolved snapshot",
            details={"snapshot_id": snapshot.snapshot_id, "strategy": strategy},
        )
    for column in ("event_date", "entry_date", "exit_date"):
        rows[column] = pd.to_datetime(rows[column])

    events = read_event_rows(repository, snapshot)
    events["event_date"] = pd.to_datetime(events["event_date"])
    rows = rows.merge(
        events[["event_id", "session"]].drop_duplicates("event_id"),
        on="event_id", how="left",
    )
    return rows
