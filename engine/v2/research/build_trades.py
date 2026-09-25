"""Publish replay output as the v2 ``trades`` table, pinned to its snapshot.

This is the slice-7 write path. It does not touch the legacy mutable Tier-2
store the way ``engine/build_trades.py`` does: it builds a candidate over the
resolved parent snapshot and commits it through
``engine.v2.data.generic_incremental``, so ``trades`` gains a version chain
like every other non-daily table. A rebuild tombstones only the rows it
replaces (its own strategies' engine rows whose id the rebuild no longer
produces) and appends the new ones; legacy and other-strategy rows are
untouched, because a partial rebuild that silently deletes the strategies it
did not name is a table that looks complete afterwards.

Legacy ``engine/build_trades.py`` and ``tools/reconcile_trades.py`` are
unchanged; this writes a NEW v2 dataset version under the same snapshot scope
alongside them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from engine.v2.contracts import (
    CompletedCoverage,
    CoverageKey,
    CoverageOutcome,
    RevisionCandidate,
    TimeInterval,
)
from engine.v2.data import incremental_tables
from engine.v2.data.generic_incremental import (
    build_generic_table_candidate,
    commit_generic_table_candidate,
)
from engine.v2.foundation import SystemClock, canonical_json, content_hash, format_timestamp
from engine.v2.research import _snapshot
from engine.v2.research.replay import to_trades_table

__all__ = [
    "FIRST_YEAR",
    "LEGACY_PROVENANCE",
    "PROVENANCE",
    "filter_events",
    "revisions_for_rebuild",
    "run",
]

#: Chains start in 2017 and the first full year of usable pairs is 2018.
FIRST_YEAR = 2017

#: The provenance a row this tool writes carries.
PROVENANCE = "engine.v2.research.replay"

#: The legacy replay's marker, replaced on a first migration run.
LEGACY_PROVENANCE = "engine.replay"

_SOURCE = "engine.v2.research.build_trades"
_TRADES_COLUMNS = (
    "trade_id", "kind", "strategy", "variant", "ticker", "event_id",
    "event_date", "year", "legs", "entry_date", "exit_date", "strike",
    "expiry", "fill_alpha", "entry_cost", "exit_value", "ret", "provenance",
)


def filter_events(events: pd.DataFrame, years=None) -> pd.DataFrame:
    """Every calendar event with a known session, optionally one year set.

    ``engine/build_trades.py``'s ``event_universe`` body split at its one
    store read: the caller supplies the frame, this only filters it.
    """
    events = events[events["session"].notna()].copy()
    events["event_date"] = pd.to_datetime(events["event_date"])
    if years is not None:
        wanted = {int(y) for y in years}
        events = events[events["event_date"].dt.year.isin(wanted)]
    return events.sort_values(["ticker", "event_date"]).reset_index(drop=True)


def _revision(row: dict, *, deleted: bool) -> incremental_tables.GenericRevision:
    """One append/correction (or tombstone) revision keyed by ``trade_id``."""
    logical_key = canonical_json([row.get("trade_id")])
    payload = None if deleted else {name: row.get(name) for name in _TRADES_COLUMNS}
    content_hash = incremental_tables.revision_hash(
        logical_key=logical_key, row=payload, deleted=deleted
    )
    candidate = RevisionCandidate(
        revision_id=f"{_SOURCE}:{logical_key}:{content_hash.removeprefix('sha256:')[:16]}",
        logical_key=logical_key,
        source=_SOURCE,
        source_priority=0,
        finality="final",
        revision_ordinal=1,
        received_at=format_timestamp(SystemClock().now()),
        content_hash=content_hash,
    )
    return incremental_tables.GenericRevision(
        candidate=candidate, row=payload, deleted=deleted,
        partition_key=str(int(row["year"])),
    )


def revisions_for_rebuild(existing: pd.DataFrame, engine_rows: pd.DataFrame,
                          rebuilt_strategies) -> list[incremental_tables.GenericRevision]:
    """The tombstones and appends one rebuild of ``rebuilt_strategies`` needs.

    A row is tombstoned only when it is a replay-produced row for a strategy
    being rebuilt AND the rebuild no longer produces its ``trade_id``. Every
    engine row is appended (or corrected in place, when its id already exists).
    """
    rebuilt = {str(s) for s in rebuilt_strategies}
    new_ids = set(engine_rows["trade_id"].astype(str)) if len(engine_rows) else set()
    revisions: list[incremental_tables.GenericRevision] = []
    if len(existing):
        provenance = existing["provenance"].astype(str)
        is_replay = provenance.str.startswith(PROVENANCE) | provenance.str.startswith(
            LEGACY_PROVENANCE
        )
        doomed = existing[
            is_replay
            & existing["strategy"].astype(str).isin(rebuilt)
            & ~existing["trade_id"].astype(str).isin(new_ids)
        ]
        for row in doomed.to_dict("records"):
            revisions.append(_revision(row, deleted=True))
    for row in engine_rows.to_dict("records"):
        revisions.append(_revision(row, deleted=False))
    return revisions


def _events_for_read(repository, snapshot, years):
    frame = _snapshot.read_table(
        repository, snapshot, "earnings_events",
        ["event_id", "ticker", "event_date", "session"],
    )
    return filter_events(frame, years=years)


def _existing_trades(repository, snapshot) -> pd.DataFrame:
    if "trades" not in snapshot.table_versions:
        raise _contract_refusal(snapshot)
    return _snapshot.read_table(repository, snapshot, "trades", _TRADES_COLUMNS)


def _contract_refusal(snapshot):
    from engine.v2.data import errors

    return errors.fail(
        "CONTRACT_MISMATCH",
        "the parent snapshot has no trades table; commit one before rebuilding",
        details={"snapshot_id": snapshot.snapshot_id},
    )


def _coverage(parent, engine_rows, revisions):
    """A complete coverage statement over exactly the rebuilt trade ids."""
    trades_ref = parent.snapshot.table_versions["trades"].table_contract_ref
    live = [item for item in revisions if not item.deleted]
    by_key = {item.candidate.logical_key: item.candidate.revision_id for item in live}
    expected, outcomes = [], []
    for row in engine_rows.to_dict("records"):
        key = canonical_json([row["trade_id"]])
        item_key = str(row["trade_id"])
        expected.append(CoverageKey(
            item_key=item_key, session_date=str(row["entry_date"])[:10],
            ticker=str(row["ticker"]),
        ))
        outcomes.append(CoverageOutcome(
            key=expected[-1], status="present", receipt_id=f"{_SOURCE}:{item_key}",
            revision_id=by_key.get(key), finality="final",
        ))
    entries = pd.to_datetime(engine_rows["entry_date"]) if len(engine_rows) else None
    start = str(entries.min().date()) if entries is not None and len(entries) else "1970-01-01"
    end = str((entries.max() + pd.Timedelta(days=1)).date()) if entries is not None and len(entries) else "2100-01-01"
    coverage_id = "coverage_" + content_hash(
        {"table": "trades", "snapshot": parent.snapshot.snapshot_id,
         "rows": sorted(str(item) for item in engine_rows["trade_id"])}
    ).removeprefix("sha256:")[:32]
    return CompletedCoverage(
        coverage_id=coverage_id,
        table_contract_ref=trades_ref,
        source=_SOURCE,
        endpoint="replay",
        interval=TimeInterval(column="entry_date", start_inclusive=start, end_exclusive=end),
        expected=tuple(expected),
        outcomes=tuple(outcomes),
        covered_tickers=tuple(sorted(set(engine_rows["ticker"].astype(str))))
        if len(engine_rows) else (),
        acquisition_receipt_refs=(f"{_SOURCE}:{parent.snapshot.snapshot_id}",),
        state="complete",
        completed_at=format_timestamp(SystemClock().now()),
    )


def run(repository, *, strategies: Sequence[str], years=None,
        scope: str = _snapshot.DEFAULT_SCOPE, snapshot_id: str | None = None,
        reports_dir: Path = Path("reports"), stamp: str | None = None,
        dry_run: bool = False) -> dict:
    """Replay ``strategies`` and publish the result as a new ``trades`` version.

    The parent head is pinned before the read and fenced at commit: if another
    writer advanced the scope's head in between, the commit refuses
    (``SNAPSHOT_CONFLICT``) rather than silently retrying or overwriting.
    """
    from engine.v2.research import replay as replay_tool

    started = time.time()
    snapshot = _snapshot.resolve_snapshot(repository, scope=scope, snapshot_id=snapshot_id)
    events = _events_for_read(repository, snapshot, years)
    results = [replay_tool.replay(repository, snapshot, strategy, events)
               for strategy in strategies]
    engine_rows = to_trades_table(results)
    if len(engine_rows):
        engine_rows["provenance"] = PROVENANCE
        engine_rows["snapshot_id"] = snapshot.snapshot_id
    existing = _existing_trades(repository, snapshot)
    parent = repository.resolve_full(snapshot.snapshot_id)
    revisions = revisions_for_rebuild(existing, engine_rows, set(strategies))
    coverage = _coverage(parent, engine_rows, revisions)
    store = _snapshot.artifact_store(repository)
    candidate = build_generic_table_candidate(
        parent, store, "trades", revisions, coverage=coverage,
        parent_snapshot_id=snapshot.snapshot_id,
    )
    if not dry_run:
        commit_generic_table_candidate(
            _snapshot.catalog_connection(repository), store, candidate,
            scope=scope,
            expected_head_snapshot_id=snapshot.snapshot_id,
            expected_head_generation=_snapshot.head_generation(repository, scope),
        )
    report = {
        "snapshot_id": snapshot.snapshot_id,
        "committed": not dry_run,
        "rebuilt_strategies": sorted({str(s) for s in strategies}),
        "rows": int(len(engine_rows)),
        "emitted_revisions": len(revisions),
        "outcome": candidate.changeset.outcome,
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
