"""Source-free regression target identity; not label-read or execution authority."""
import re

from engine.v2.foundation import content_hash


def target_descriptor(*, column, prediction_column, domain, definition_id):
    """Bind explicit target semantics by hash; never infer them from column names."""
    columns = (column, prediction_column)
    if (any(type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
            or name in {"event_id", "context_id", "horizon_sessions", "columns", "row_keys", "values"}
            for name in columns) or column == prediction_column
            or type(domain) is not str or domain not in ("signed", "nonnegative")
            or type(definition_id) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", definition_id) is None):
        raise ValueError("invalid regression target descriptor")
    identity = {"schema_version": "regression_target.v1", "column": column,
        "prediction_column": prediction_column, "domain": domain, "definition_id": definition_id}
    return {**identity, "target_id": content_hash(identity)}


def validate_target(document):
    """Return a detached closed descriptor; reject unknown fields and stale hashes."""
    if type(document) is not dict or set(document) != {
            "schema_version", "column", "prediction_column", "domain", "definition_id", "target_id"}:
        raise ValueError("invalid regression target descriptor")
    expected = target_descriptor(**{name: document[name] for name in (
        "column", "prediction_column", "domain", "definition_id")})
    if document != expected:
        raise ValueError("regression target identity differs")
    return expected
