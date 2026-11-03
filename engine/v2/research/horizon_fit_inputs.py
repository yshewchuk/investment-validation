"""Ordered source-free fit matrices and target-column-bound input evidence."""
import numpy as np

from engine.v2.foundation import content_hash
from engine.v2.foundation.iso_dates import validate_train_event_dates


def _matrix(indexed, event_ids, horizon, columns):
    rows = [indexed[event, horizon] for event in event_ids]
    row_keys = [[row[key] for key in ("event_id", "context_id", "horizon_sessions")]
                for row in rows]
    values = [[row[column] for column in columns] for row in rows]
    return row_keys, values, np.asarray(values, dtype=float)


def _require_target(target_column):
    if (type(target_column) is not str or not target_column.strip()
            or target_column in ("columns", "row_keys", "values", "event_dates")):
        raise ValueError("horizon inputs require a nonempty nonreserved target column")


def fit_evidence(columns, train_features, prediction_features, horizon_sessions,
                 training_event_ids, evaluation_event_ids, training_row_keys,
                 evaluation_row_keys, *, target_column, labels=None,
                 train_event_dates=None):
    """Hash ordered fit values and row keys into the shared evidence shape."""
    _require_target(target_column)
    columns = list(columns)
    training = {"columns": columns, "row_keys": training_row_keys,
                "values": train_features}
    prediction = {"columns": columns, "row_keys": evaluation_row_keys,
                  "values": prediction_features}
    if labels is not None:
        training[target_column] = labels
    if train_event_dates is not None:
        validate_train_event_dates(train_event_dates, train_features)
        training["event_dates"] = list(train_event_dates)
    return {"horizon_sessions": horizon_sessions,
            "training_event_ids": list(training_event_ids),
            "evaluation_event_ids": list(evaluation_event_ids),
            "training_matrix_id": content_hash(training),
            "prediction_matrix_id": content_hash(prediction)}


def prepare_horizon_inputs(indexed, training_ids, evaluation_ids, horizon, columns, labels, *, target_column, train_event_dates=None):
    """Hash original ordered values before fitting can mutate detached matrices.

    Caller owns membership, target/domain policy and source admission. Only supplied
    training labels are consumed; evaluation labels are neither needed nor read.
    Event-ID and column iterables are materialized once after target validation.
    This performs no fitting, prediction validation, I/O or experiment admission.
    """
    _require_target(target_column)
    training_ids, evaluation_ids, columns = list(training_ids), list(evaluation_ids), list(columns)
    training_keys, training_values, matrix = _matrix(indexed, training_ids, horizon, columns)
    prediction_keys, prediction_values, test_matrix = _matrix(indexed, evaluation_ids, horizon, columns)
    evidence = fit_evidence(columns, training_values, prediction_values, horizon,
        training_ids, evaluation_ids, training_keys, prediction_keys,
        target_column=target_column, labels=np.asarray(labels, dtype=float).tolist(),
        train_event_dates=train_event_dates)
    return matrix, test_matrix, evidence
