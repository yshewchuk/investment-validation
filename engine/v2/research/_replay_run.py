"""The replay tool entrypoint — resolve one snapshot, replay, write a report.

Split out of ``engine.v2.research.replay`` (review blocker: module fan-out)
with the bodies unchanged; ``run`` is the same function the CLI and the tests
called there, now imported from here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from engine.v2.research._snapshot import DEFAULT_SCOPE, read_table, resolve_snapshot
from engine.v2.research._trades_table import to_trades_table
from engine.v2.research.replay import replay

__all__ = ["events_frame", "run"]


def events_frame(repository, snapshot_ref, years=None) -> pd.DataFrame:
    """The event universe for a replay: earnings_events with a known session.

    This is ``engine/build_trades.py``'s ``event_universe`` body split at its
    one store read (the same technique ``fill_quality.join_from_snapshot``
    used), with the v2 table read in place of the legacy one.
    """
    events = read_table(
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
        reports_dir: Path = Path("reports"), scope: str = DEFAULT_SCOPE,
        snapshot_id: str | None = None, stamp: str | None = None) -> dict:
    """Replay every strategy against one pinned snapshot and write a report.

    The returned ``trades`` frame is stamped with ``provenance =
    "engine.v2.research.replay"`` (overwriting the legacy ``engine.replay``
    marker ``to_trades_table`` writes) and with the snapshot id that produced
    it, so a v2 row can never be mistaken for a legacy-replay row.
    """
    snapshot = resolve_snapshot(repository, scope=scope, snapshot_id=snapshot_id)
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
