from engine.v2.ops import catalog, submission
from engine.v2.ops.errors import make_problem


def test_failure_json_stays_within_byte_bound():
    problem = make_problem("WORKER_FAILED", "x", details={"blob": "y" * (2 * 1024 * 1024)})
    assert len(catalog.dumps_failure(problem).encode("utf-8")) <= catalog.MAX_JOB_FAILURE_BYTES


def test_small_problem_round_trips_identical():
    problem = make_problem("WORKER_FAILED", "x", details={"blob": "small"})
    assert catalog.dumps_failure(problem) == catalog.dumps(problem)


def test_submission_shares_catalog_bound():
    assert submission.MAX_JOB_FAILURE_BYTES == catalog.MAX_JOB_FAILURE_BYTES