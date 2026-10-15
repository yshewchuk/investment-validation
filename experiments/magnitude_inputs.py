"""Read-only native magnitude admission and whole-run physical label preflight."""
from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import from_document
from engine.v2.ops.native_registration import magnitude_execution_document
from engine.v2.research.magnitude_inputs import preflight_magnitude_targets
from experiments.native_registration import require_native_registration


def admit_magnitude_inputs(conn, store, repository, spec, *, code_root, as_of_month,
                           magnitude_training_start, magnitude_labels_before,
                           scope="shadow", event_ids=None, magnitude_feature_snapshot_id=None, magnitude_history_start=None,
                           magnitude_fit_model=None):
    """Bind the original execution plan and prove every actual participant before labels."""
    magnitude_execution_document(magnitude_labels_before)
    registration = require_native_registration(conn, store, repository, spec,
        code_root=code_root, as_of_month=as_of_month, scope=scope, event_ids=event_ids,
        magnitude_training_start=magnitude_training_start, magnitude_labels_before=magnitude_labels_before,
        magnitude_feature_snapshot_id=magnitude_feature_snapshot_id, magnitude_history_start=magnitude_history_start,
        magnitude_fit_model=magnitude_fit_model)
    binding = registration.document
    participants = sorted({event_id for fold in binding["magnitude_plan"]["folds"]
        for field in ("training_candidate_event_ids", "evaluation_event_ids") for event_id in fold[field]})
    preflight = preflight_magnitude_targets(repository, from_document(SnapshotRef, binding["snapshot"]),
        as_of_month=binding["holdouts"]["holdout_as_of_month"], event_ids=participants)
    return registration, preflight
