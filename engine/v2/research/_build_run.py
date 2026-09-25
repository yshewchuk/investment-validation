"""The build-trades tool entrypoint — replay and publish as a new version.

Split out of ``engine.v2.research.build_trades`` (review blocker: module
fan-out) with the body unchanged; ``run`` is the same function the CLI and the
tests called there, now imported from here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from engine.v2.research._trades_publish import (
    DEFAULT_SCOPE,
    publish,
    read_event_rows,
    read_existing_trades,
    resolve,
)
from engine.v2.research._trades_revisions import (
    PROVENANCE,
    filter_events,
    revisions_for_rebuild,
)
from engine.v2.research._trades_table import to_trades_table
from engine.v2.research.replay import replay

__all__ = ["run"]


def run(repository, *, strategies: Sequence[str], years=None,
        scope: str = DEFAULT_SCOPE, snapshot_id: str | None = None,
        reports_dir: Path = Path("reports"), stamp: str | None = None,
        dry_run: bool = False) -> dict:
    """Replay ``strategies`` and publish the result as a new ``trades`` version.

    The parent head is pinned before the read and fenced at commit: if another
    writer advanced the scope's head in between, the commit refuses
    (``SNAPSHOT_CONFLICT``) rather than silently retrying or overwriting.
    """
    started = time.time()
    snapshot = resolve(repository, scope=scope, snapshot_id=snapshot_id)
    events = filter_events(read_event_rows(repository, snapshot), years=years)
    results = [replay(repository, snapshot, strategy, events)
               for strategy in strategies]
    engine_rows = to_trades_table(results)
    if len(engine_rows):
        engine_rows["provenance"] = PROVENANCE
        engine_rows["snapshot_id"] = snapshot.snapshot_id
    existing = read_existing_trades(repository, snapshot)
    revisions = revisions_for_rebuild(existing, engine_rows, set(strategies))
    published = publish(
        repository, snapshot, revisions=revisions, rows=engine_rows,
        scope=scope, dry_run=dry_run,
    )
    report = {
        "snapshot_id": snapshot.snapshot_id,
        "committed": published["committed"],
        "rebuilt_strategies": sorted({str(s) for s in strategies}),
        "rows": int(len(engine_rows)),
        "emitted_revisions": published["emitted_revisions"],
        "outcome": published["outcome"],
        "elapsed_s": round(time.time() - started, 1),
        "trades": engine_rows,
    }
    if reports_dir is not None:
        stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"build_trades_{stamp}.json"
        path.write_text(json.dumps(
            {key: value for key, value in report.items() if key != "trades"},
            indent=2, sort_keys=True))
        report["path"] = str(path)
    return report
