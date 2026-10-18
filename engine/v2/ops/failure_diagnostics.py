"""Preserve oversized coordinator failures before committing bounded receipts."""
from __future__ import annotations

from dataclasses import replace

from engine.v2.contracts import FAILURE_CODES, PROBLEM_V1, ArtifactRef, Problem
from engine.v2.foundation import ArtifactError, ArtifactStore
from engine.v2.ops.catalog import MAX_JOB_FAILURE_BYTES, dumps, dumps_failure
from engine.v2.ops.errors import OpsError, fail, make_problem


class DiagnosticPublicationError(OpsError):
    """The original Problem could not be published; completion must remain uncommitted."""


def prepare_failure(problem: Problem, store: ArtifactStore) -> tuple[Problem, ArtifactRef | None]:
    """Publish the entire original envelope; no catalog write or completion happens here."""
    if (not isinstance(problem.code, str) or problem.code not in FAILURE_CODES
            or problem.category != FAILURE_CODES[problem.code][0]
            or type(problem.retryable) is not bool or problem.schema_version != PROBLEM_V1):
        raise fail("INTEGRITY_FAILED", "failure diagnostic requires a valid Problem classification") from OpsError(problem)
    payload = dumps(problem).encode("utf-8")
    if len(payload) <= MAX_JOB_FAILURE_BYTES:
        return problem, None
    try:
        ref = store.publish_bytes(payload, schema_ref=PROBLEM_V1)
    except OSError:
        raise DiagnosticPublicationError(make_problem(
            "RESOURCE_UNAVAILABLE", "failure diagnostic could not be published")) from OpsError(problem)
    except ArtifactError:
        raise DiagnosticPublicationError(make_problem(
            "INTEGRITY_FAILED", "failure diagnostic could not be published")) from OpsError(problem)
    summary = replace(make_problem(problem.code, "failure details preserved in diagnostic artifact"),
                      retryable=problem.retryable, diagnostic_ref=ref.artifact_id)
    dumps_failure(summary)
    return summary, ref
