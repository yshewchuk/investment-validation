"""Option A of the 2026-09-29 decision (EXP-182-style parallel path): a new v2
trades loader for experiments, migrating nothing yet. ``snapshot_id`` is always
explicit -- never a scope-head resolve. ``experiments/common.py::load_engine_trades``
keeps serving every existing caller unchanged; this module has none yet."""
from __future__ import annotations

from pathlib import Path

from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.research import experiment_trades

__all__ = ["load_v2_trades"]


def load_v2_trades(strategy: str, *, catalog: Path | str, store_root: Path | str,
                   snapshot_id: str) -> pd.DataFrame:
    """The committed v2-replay trades for ``strategy``, one pinned snapshot.
    SNAPSHOT_NOT_FOUND -- unknown ``snapshot_id`` (``Repository.resolve``);
    CONTRACT_MISMATCH -- that snapshot carries no ``trades`` table;
    POPULATION_COLLAPSED -- no row for this strategy/provenance pair."""
    conn = open_catalog(Path(catalog), clock=SystemClock())
    try:
        repository = Repository(conn, ArtifactStore(Path(store_root)))
        return experiment_trades.load_trades(repository, repository.resolve(snapshot_id), strategy)
    finally:
        conn.close()
