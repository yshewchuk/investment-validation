"""Canonical experiment populations, admitted before any outcome read."""
from __future__ import annotations

from engine.v2.contracts.data import SnapshotRef
from engine.v2.data import errors
from engine.v2.research import progress
from engine.v2.research._scan import read_table
from engine.v2.research.experiment_trades import _exclude_holdouts, _holdout_context
from engine.v2.research.snapshot_inputs import admit_snapshot


def canonical_events(repository, snapshot):
    """Read the six canonical metadata columns from a native snapshot reference."""
    admit_snapshot(repository, snapshot)
    return _canonical_events(repository, snapshot)


def _canonical_events(repository, snapshot, *, max_object_bytes=None):
    """Canonical event metadata with same-ticker/date duplicates flagged; shared with the final reader."""
    events = read_table(repository, snapshot, "earnings_events", (
        "event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"),
        max_object_bytes=max_object_bytes)
    events.loc[events.duplicated(["ticker", "event_date"], keep=False), "date_conflict"] = True
    return events


def load_population(repository, snapshot, *, as_of_month, purpose="selection",
                    event_ids=None, max_object_bytes=None):
    """Read canonical metadata only; explicit excluded IDs refuse the whole read.

    Canonical scans log stage boundaries and bounded row progress for both modes.
    Full-universe reads also log outer completion when they take at least 5 s.
    ``max_object_bytes`` bounds the scan's decoded row-group bytes; ``None``
    keeps the decoder default unchanged."""
    args = (repository, snapshot)
    kwargs = dict(as_of_month=as_of_month, purpose=purpose, event_ids=event_ids,
                  max_object_bytes=max_object_bytes)
    if event_ids is not None:
        return _load_population(*args, **kwargs)
    with progress.stage("load_population", quiet_under=5.0):
        return _load_population(*args, **kwargs)


def _load_population(repository, snapshot, *, as_of_month, purpose="selection",
                     event_ids=None, max_object_bytes=None):
    """Read canonical metadata only; explicit excluded IDs refuse the whole read."""
    if (not isinstance(snapshot, SnapshotRef) or not isinstance(snapshot.snapshot_id, str)
            or not snapshot.snapshot_id.strip()):
        raise errors.fail("SNAPSHOT_UNRESOLVED", "an exact committed snapshot is required")
    try:
        resolved = repository.resolve(snapshot.snapshot_id)
    except (errors.DataError, ValueError, TypeError, KeyError):
        raise errors.fail("SNAPSHOT_UNRESOLVED", "the requested snapshot cannot be resolved") from None
    if resolved != snapshot:
        raise errors.fail("SNAPSHOT_UNRESOLVED", "snapshot differs from its committed identity")
    policy, context = _holdout_context(snapshot, as_of_month, purpose, event_ids)
    events = _canonical_events(repository, snapshot, max_object_bytes=max_object_bytes)
    if event_ids is not None:
        if not event_ids or set(event_ids) - set(events["event_id"]):
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "requested event membership cannot be resolved")
        selected = events[events["event_id"].isin(event_ids)]
    else:
        selected = events
    if selected.empty:
        raise errors.fail("POPULATION_COLLAPSED", "the canonical experiment population is empty")
    return _exclude_holdouts(selected[["event_id", "ticker", "event_date"]].copy(),
                             events.copy(), policy, context, purpose, event_ids)
