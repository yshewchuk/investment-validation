"""The shared trades write path: resolve, read, build, commit.

Split out of the slice-7 original (review blocker: module fan-out) so the
rebuild tool and the reconcile tool commit through one mechanism instead of
two. A candidate is built over the resolved parent snapshot and committed
through ``engine.v2.data.generic_incremental``; the scope head is fenced at
commit, so a writer that lost its parent refuses (``SNAPSHOT_CONFLICT``)
rather than silently overwriting.
"""
from __future__ import annotations

from engine.v2.data import errors
from engine.v2.data.generic_incremental import (
    build_generic_table_candidate,
    commit_generic_table_candidate,
)
from engine.v2.research._snapshot import (
    DEFAULT_SCOPE,
    artifact_store,
    catalog_connection,
    head_generation,
    read_table,
    resolve_snapshot,
)
from engine.v2.research._trades_revisions import _TRADES_COLUMNS
from engine.v2.research.build_trades import coverage

__all__ = [
    "DEFAULT_SCOPE",
    "publish",
    "read_event_rows",
    "read_existing_trades",
    "resolve",
]


def resolve(repository, *, scope, snapshot_id=None):
    """The pinned parent snapshot: an explicit id, else the scope head."""
    return resolve_snapshot(repository, scope=scope, snapshot_id=snapshot_id)


def read_event_rows(repository, snapshot):
    """The canonical-event universe columns, read through the pinned snapshot."""
    return read_table(
        repository, snapshot, "earnings_events",
        ["event_id", "ticker", "event_date", "session"],
    )


def read_existing_trades(repository, snapshot):
    """The current version of the ``trades`` table under the pinned snapshot."""
    if "trades" not in snapshot.table_versions:
        raise errors.fail(
            "CONTRACT_MISMATCH",
            "the parent snapshot has no trades table",
            details={"snapshot_id": snapshot.snapshot_id},
        )
    return read_table(repository, snapshot, "trades", _TRADES_COLUMNS)


def publish(repository, snapshot, *, revisions, rows, scope, dry_run: bool = False) -> dict:
    """Commit ``revisions`` as a new ``trades`` version, fenced to ``snapshot``.

    ``rows`` is the frame the coverage statement is built over — the appended
    rows on a rebuild, the removed rows on a reconcile. With no revisions
    nothing is built and nothing is committed.
    """
    outcome = {
        "committed": False,
        "outcome": "noop",
        "committed_snapshot_id": None,
        "emitted_revisions": len(revisions),
    }
    if not revisions:
        return outcome

    parent = repository.resolve_full(snapshot.snapshot_id)
    statement = coverage(parent, rows, revisions)
    store = artifact_store(repository)
    candidate = build_generic_table_candidate(
        parent, store, "trades", revisions, coverage=statement,
        parent_snapshot_id=snapshot.snapshot_id,
    )
    outcome["outcome"] = candidate.changeset.outcome
    if not dry_run:
        commit_generic_table_candidate(
            catalog_connection(repository), store, candidate,
            scope=scope,
            expected_head_snapshot_id=snapshot.snapshot_id,
            expected_head_generation=head_generation(repository, scope),
        )
        outcome["committed"] = True
        outcome["committed_snapshot_id"] = candidate.snapshot.snapshot_id
    return outcome
