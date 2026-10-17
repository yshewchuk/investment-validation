"""Shared artifact-error classification with caller-owned sanitized messages."""
from engine.v2.ops.errors import OpsError, fail


class SearchArtifactError(OpsError):
    """Tagged artifact failure carrying shared, sanitized identity/resource policy."""


def artifact_failure(error, *, integrity_message, resource_message, unsafe_is_conflict=False):
    """Classify integrity and optional unsafe-reference failures without leaking paths."""
    conflict = error.code == "INTEGRITY_FAILED" or (unsafe_is_conflict and error.code == "UNSAFE_PATH")
    code = "EXPERIMENT_IDENTITY_CONFLICT" if conflict else "RESOURCE_UNAVAILABLE"
    return SearchArtifactError(fail(code, integrity_message if conflict else resource_message).problem)
