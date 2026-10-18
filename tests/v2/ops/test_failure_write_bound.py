import dataclasses

import pytest

from engine.v2.ops import catalog, submission
from engine.v2.ops.errors import OpsError, make_problem


def test_oversized_failure_requires_a_diagnostic_without_losing_details():
    problem = make_problem("WORKER_FAILED", "x", details={"blob": "y" * (2 * 1024 * 1024)})
    original = catalog.dumps(problem)
    with pytest.raises(OpsError) as captured:
        catalog.dumps_failure(problem)
    assert captured.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert captured.value.problem.message == "failure requires a diagnostic artifact before persistence"
    assert captured.value.__cause__.problem is problem
    assert catalog.dumps(problem) == original


def test_small_problem_round_trips_identical():
    problem = make_problem("WORKER_FAILED", "x", details={"blob": "small"})
    assert catalog.dumps_failure(problem) == catalog.dumps(problem)


def test_submission_shares_catalog_bound():
    assert submission.MAX_JOB_FAILURE_BYTES == catalog.MAX_JOB_FAILURE_BYTES


def test_oversized_non_detail_fields_refuse_without_clearing_metadata():
    problem = make_problem("WORKER_FAILED", "x", details={})
    p = dataclasses.replace(
        problem,
        stage="s" * (2 * 1024 * 1024),
        dependency_refs=tuple("r" * 1000 for _ in range(2000)),
    )
    original = catalog.dumps(p)
    with pytest.raises(OpsError) as captured:
        catalog.dumps_failure(p)
    assert captured.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert captured.value.problem.message == "failure requires a diagnostic artifact before persistence"
    assert captured.value.__cause__.problem is p
    assert catalog.dumps(p) == original
