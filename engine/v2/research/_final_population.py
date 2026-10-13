"""Private metadata-only final-holdout population (#490); no grant, outcome read or write."""
from __future__ import annotations

from dataclasses import dataclass

from engine.v2.contracts.final_holdout import FinalHoldoutRequest
from engine.v2.data import errors
from engine.v2.foundation.final_holdout import request_from_document
from engine.v2.foundation.typed import to_document
from engine.v2.research.experiment_population import _canonical_events
from engine.v2.research.experiment_trades import _membership_labels, _policy


@dataclass(frozen=True)
class FinalPopulation:
    """The two held-out sets, kept separate; never averaged or merged."""

    snapshot_id: str
    holdout_as_of_month: str
    random_membership_version: str
    rolling_membership_version: str
    random_event_ids: tuple[str, ...]
    rolling_event_ids: tuple[str, ...]
    overlap_count: int
    ambiguous_count: int


def load_final_population(repository, request) -> FinalPopulation:
    """Classify the request's pinned canonical events with the shared membership code.

    Reads ``earnings_events`` metadata only. Ambiguous events enter neither set.
    """
    try:
        if not isinstance(request, FinalHoldoutRequest):
            raise ValueError("not a final-read request")
        request = request_from_document(to_document(request))
        policy = _policy(request.holdout_as_of_month)
    except (ValueError, TypeError, KeyError):
        raise errors.fail("HOLDOUT_ACCESS_DENIED", "a valid final-read request is required") from None
    try:
        snapshot = repository.resolve(request.snapshot_id)
    except (errors.DataError, ValueError, TypeError, KeyError) as error:
        if isinstance(error, errors.DataError) and error.problem.category == "integrity":
            raise
        raise errors.fail("SNAPSHOT_UNRESOLVED", "the requested snapshot cannot be resolved") from None
    if getattr(snapshot, "snapshot_id", None) != request.snapshot_id:
        raise errors.fail("SNAPSHOT_UNRESOLVED", "snapshot differs from its committed identity")
    events = _canonical_events(repository, snapshot)
    _, labels = _membership_labels(events[["event_id", "ticker", "event_date"]].copy(), events.copy(), policy)
    clean = {key: value for key, value in labels.items() if key is not None and "ambiguous" not in value}
    random = tuple(sorted(key for key, value in clean.items() if "random" in value))
    rolling = tuple(sorted(key for key, value in clean.items() if "rolling" in value))
    if not random and not rolling:
        raise errors.fail("POPULATION_COLLAPSED", "the final holdout population is empty")
    return FinalPopulation(
        snapshot_id=snapshot.snapshot_id, holdout_as_of_month=request.holdout_as_of_month,
        random_membership_version=request.random_membership_version,
        rolling_membership_version=request.rolling_membership_version,
        random_event_ids=random, rolling_event_ids=rolling,
        overlap_count=len(set(random) & set(rolling)),
        ambiguous_count=sum(1 for value in labels.values() if "ambiguous" in value))
