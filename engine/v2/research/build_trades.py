"""Pure halves of the v2 ``trades`` write path: filters, revisions, coverage.

Split out of the slice-7 original (review blocker: module fan-out). This module
holds the definitions both trades writers share — the canonical-event filter,
the rebuild/tombstone revision builders (re-exported from
``engine.v2.research._trades_revisions``) and the coverage statement that
describes a change. The read and commit helpers live in
``engine.v2.research._trades_publish`` and the tool entrypoint in
``engine.v2.research._build_run``.

Legacy ``engine/build_trades.py`` and ``tools/reconcile_trades.py`` are
unchanged; the v2 path writes a NEW ``trades`` dataset version under the same
snapshot scope alongside them.
"""
from __future__ import annotations

import pandas as pd

from engine.v2.contracts import (
    CompletedCoverage,
    CoverageKey,
    CoverageOutcome,
    TimeInterval,
)
from engine.v2.foundation import SystemClock, canonical_json, content_hash, format_timestamp
from engine.v2.research._trades_revisions import (
    _SOURCE,
    FIRST_YEAR,
    LEGACY_PROVENANCE,
    PROVENANCE,
    filter_events,
    filter_to_canonical_events,
    revisions_for_rebuild,
    revisions_for_removal,
)

__all__ = [
    "FIRST_YEAR",
    "LEGACY_PROVENANCE",
    "PROVENANCE",
    "coverage",
    "filter_events",
    "filter_to_canonical_events",
    "revisions_for_rebuild",
    "revisions_for_removal",
]


def coverage(parent, rows: pd.DataFrame, revisions) -> CompletedCoverage:
    """A complete coverage statement over exactly the touched trade ids."""
    trades_ref = parent.snapshot.table_versions["trades"].table_contract_ref
    by_key = {item.candidate.logical_key: item.candidate.revision_id
              for item in revisions}
    expected, outcomes = [], []
    for row in rows.to_dict("records"):
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
    entries = pd.to_datetime(rows["entry_date"]) if len(rows) else None
    start = str(entries.min().date()) if entries is not None and len(entries) else "1970-01-01"
    end = str((entries.max() + pd.Timedelta(days=1)).date()) if entries is not None and len(entries) else "2100-01-01"
    coverage_id = "coverage_" + content_hash(
        {"table": "trades", "snapshot": parent.snapshot.snapshot_id,
         "rows": sorted(str(item) for item in rows["trade_id"])}
    ).removeprefix("sha256:")[:32]
    return CompletedCoverage(
        coverage_id=coverage_id,
        table_contract_ref=trades_ref,
        source=_SOURCE,
        endpoint="replay",
        interval=TimeInterval(column="entry_date", start_inclusive=start, end_exclusive=end),
        expected=tuple(expected),
        outcomes=tuple(outcomes),
        covered_tickers=tuple(sorted(set(rows["ticker"].astype(str))))
        if len(rows) else (),
        acquisition_receipt_refs=(f"{_SOURCE}:{parent.snapshot.snapshot_id}",),
        state="complete",
        completed_at=format_timestamp(SystemClock().now()),
    )
