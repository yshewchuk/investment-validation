"""Bind supported stored-history production to native experiment admission."""
from engine.v2.data.errors import DataError
from engine.v2.foundation import to_document
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_history_inputs import admit_stored
from experiments.magnitude_history_store import context


def history_binding(repository, snapshot, temporal, request, *, as_of_month, event_ids):
    """Derive and admit the original feature pin before any attempt or payload read."""
    try:
        derived = context(repository, snapshot, as_of_month=as_of_month, event_ids=event_ids,
            evaluation_years=temporal["evaluation_years"], training_start=temporal["training_start"],
            history_start=request["history_start"])
        if derived["temporal_plan"] != temporal or derived["source_snapshot"] != to_document(snapshot):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "history context differs from the registered temporal plan")
        feature = repository.resolve(request["feature_snapshot_id"])
        admit_stored(repository, feature, derived)
    except DataError as error:
        if error.code in {"IDENTITY_CONFLICT", "SNAPSHOT_NOT_FOUND", "SNAPSHOT_UNRESOLVED", "CONTRACT_MISMATCH"}:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "stored history identity could not be admitted") from None
        raise
    return {"schema_version": "magnitude_history_binding.v1", "feature_snapshot": to_document(feature),
            "context": derived}
