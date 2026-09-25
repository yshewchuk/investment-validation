"""Pure revision builders for the v2 ``trades`` write path.

Split out of ``engine.v2.research.build_trades`` (review blocker: module
fan-out). Everything here is pure — frames in, revision objects out — so both
the rebuild tool and the reconcile tool can share one definition of "what a
trades change is" without a second mechanism. The coverage statement that
describes a change lives in ``engine.v2.research.build_trades``, the read and
commit helpers in ``engine.v2.research._trades_publish``.
"""
from __future__ import annotations

import pandas as pd

from engine.v2.contracts import RevisionCandidate
from engine.v2.data import incremental_tables
from engine.v2.foundation import SystemClock, canonical_json, format_timestamp

__all__ = [
    "FIRST_YEAR",
    "LEGACY_PROVENANCE",
    "PROVENANCE",
    "filter_events",
    "filter_to_canonical_events",
    "revisions_for_rebuild",
    "revisions_for_removal",
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


def filter_to_canonical_events(
    trades: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Remove simulated trades whose event claim was not canonicalized.

    Moved verbatim from ``engine/data/normalize/n_trades.py``: a v2 module may
    not import legacy code without a declared adapter, and this is a pure
    filter over two frames. Reconciliation only removes event claims, so every
    retained priced trade is still valid and does not need an expensive replay.
    Live and paper records are preserved even if a later calendar correction
    changes their event key; they are ledger facts rather than reproducible
    simulations.
    """
    if events.empty:
        return trades.copy().reset_index(drop=True), {
            "rows_in": int(len(trades)),
            "rows_out": int(len(trades)),
            "rows_removed": 0,
            "event_ids_removed": 0,
            "reason": "empty canonical event universe; refusing destructive filter",
        }

    valid = set(events["event_id"].dropna().astype(str))
    event_ids = trades["event_id"].astype("string")
    if "kind" in trades:
        simulated = trades["kind"].astype(str).eq("sim")
    else:
        provenance = trades.get(
            "provenance", pd.Series("", index=trades.index)
        ).astype(str)
        simulated = provenance.str.startswith(("engine.replay", "legacy:"))
    simulated &= event_ids.notna()
    invalid = simulated & ~event_ids.astype(str).isin(valid)
    out = trades.loc[~invalid].copy().reset_index(drop=True)
    report = {
        "rows_in": int(len(trades)),
        "rows_out": int(len(out)),
        "rows_removed": int(invalid.sum()),
        "event_ids_removed": int(event_ids[invalid].nunique()),
    }
    return out, report


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


def revisions_for_removal(rows: pd.DataFrame) -> list[incremental_tables.GenericRevision]:
    """Tombstones for exactly the rows a canonical-event reconciliation removes."""
    return [_revision(row, deleted=True) for row in rows.to_dict("records")]
