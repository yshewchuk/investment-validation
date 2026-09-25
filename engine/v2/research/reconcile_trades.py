"""Prune the v2 ``trades`` table to the canonical earnings-event universe.

The v2 counterpart of ``tools/reconcile_trades.py``: instead of READ-MODIFY-
WRITING the legacy mutable Tier-2 table, it computes the same canonical-event
filter and publishes the removals as a new ``trades`` dataset version through
``engine.v2.data.generic_incremental``, tombstoning only the non-canonical
rows. Canonical rows, other strategies' rows and live/paper records are not
rewritten; the legacy table is never touched.

``filter_to_canonical_events`` lives in
``engine.v2.research._trades_revisions`` (the same definition the rebuild path
uses), and the commit goes through the same
``engine.v2.research._trades_publish`` path, so reconcile is not a second
mechanism.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from engine.v2.research._trades_publish import (
    DEFAULT_SCOPE,
    publish,
    read_event_rows,
    read_existing_trades,
)
from engine.v2.research._trades_publish import resolve as resolve_parent
from engine.v2.research._trades_revisions import (
    filter_to_canonical_events,
    revisions_for_removal,
)

__all__ = ["filter_to_canonical_events", "run"]


def run(repository, *, scope: str = DEFAULT_SCOPE, snapshot_id: str | None = None,
        reports_dir: Path | None = None, stamp: str | None = None,
        dry_run: bool = False) -> dict:
    """Tombstone the non-canonical simulated rows of the pinned ``trades`` table.

    The parent head is pinned before the read and fenced at commit: an explicit
    ``snapshot_id`` reproduces that snapshot's reconciliation (``dry_run=True``
    when the head has moved), and a commit against a stale parent refuses
    (``SNAPSHOT_CONFLICT``) rather than applying a diff to a table it did not
    read.
    """
    snapshot = resolve_parent(repository, scope=scope, snapshot_id=snapshot_id)
    events = read_event_rows(repository, snapshot)
    trades = read_existing_trades(repository, snapshot)
    clean, filter_report = filter_to_canonical_events(trades, events)
    kept_ids = set(clean["trade_id"].astype(str)) if len(clean) else set()
    removed = trades[~trades["trade_id"].astype(str).isin(kept_ids)] if len(trades) else trades
    revisions = revisions_for_removal(removed)
    published = publish(repository, snapshot, revisions=revisions, rows=removed,
                        scope=scope, dry_run=dry_run)
    report = {
        "snapshot_id": snapshot.snapshot_id,
        "committed": published["committed"],
        "committed_snapshot_id": published["committed_snapshot_id"],
        "outcome": published["outcome"],
        "rows_in": int(len(trades)),
        "rows_out": int(len(trades) - len(removed)),
        "rows_removed": int(len(removed)),
        "removed_trade_ids": sorted(removed["trade_id"].astype(str)),
        "filter": filter_report,
    }
    if reports_dir is not None:
        stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"reconcile_trades_{stamp}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        report["path"] = str(path)
    return report
