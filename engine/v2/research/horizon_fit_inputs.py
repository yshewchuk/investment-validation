"""Ordered source-free fit matrices and target-column-bound input evidence."""
import numpy as np

from engine.v2.foundation import content_hash


def _matrix(indexed, event_ids, horizon, columns):
    rows = [indexed[event, horizon] for event in event_ids]
    document = {"columns": columns, "row_keys": [
        [row[key] for key in ("event_id", "context_id", "horizon_sessions")] for row in rows],
        "values": [[row[column] for column in columns] for row in rows]}
    return document, np.asarray(document["values"], dtype=float)


def prepare_horizon_inputs(indexed, training_ids, evaluation_ids, horizon, columns, labels, *, target_column):
    """Hash original ordered values before fitting can mutate detached matrices.

    Caller owns membership, target/domain policy and source admission. Only supplied
    training labels are consumed; evaluation labels are neither needed nor read.
    Event-ID and column iterables are materialized once after target validation.
    This performs no fitting, prediction validation, I/O or experiment admission.
    """
    if (type(target_column) is not str or not target_column.strip()
            or target_column in ("columns", "row_keys", "values")):
        raise ValueError("horizon inputs require a nonempty nonreserved target column")
    training_ids, evaluation_ids, columns = list(training_ids), list(evaluation_ids), list(columns)
    training, matrix = _matrix(indexed, training_ids, horizon, columns)
    prediction, test_matrix = _matrix(indexed, evaluation_ids, horizon, columns)
    training[target_column] = np.asarray(labels, dtype=float).tolist()
    evidence = {"horizon_sessions": horizon, "training_event_ids": list(training_ids),
        "evaluation_event_ids": list(evaluation_ids), "training_matrix_id": content_hash(training),
        "prediction_matrix_id": content_hash(prediction)}
    return matrix, test_matrix, evidence
