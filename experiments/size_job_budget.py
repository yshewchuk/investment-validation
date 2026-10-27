"""One immutable execution deadline in the existing journal, never a retry grant."""
from datetime import timedelta

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.foundation import SystemClock, format_timestamp, parse_timestamp
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import fail
from experiments.magnitude_search_execution import execution_limits
from experiments.magnitude_search_records import append_record, read_record


def remaining(conn, store, registration, plan, *, clock=None):
    """Internal: caller has admitted the sealed driver/plan; completed reports need no budget."""
    clock = clock or SystemClock()
    now, budget = clock.now(), plan["request"]["limits"]["job"]["wall_seconds"]
    saved = read_record(conn, store, registration.run_id, "size-job-deadline")
    if saved is None:
        forbid_fitting("size_job_budget.start")
        forbid_legacy_fitting("size_job_budget.start")
        saved = append_record(conn, store, registration.run_id, "size-job-deadline", {
            "plan_id": plan["plan_id"], "started_at": format_timestamp(now),
            "deadline": format_timestamp(now + timedelta(seconds=budget))})
    try:
        start, deadline = parse_timestamp(saved["started_at"]), parse_timestamp(saved["deadline"])
        if (set(saved) != {"plan_id", "started_at", "deadline"} or saved["plan_id"] != plan["plan_id"]
                or deadline - start != timedelta(seconds=budget)):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "execution job deadline differs") from None
    now = clock.now()
    seconds = int((deadline - now).total_seconds())
    if now < start or seconds < 1:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "execution job deadline exhausted or clock moved backward")
    return seconds


def resolved_limits(conn, store, registration, plan, limits):
    """Clamp each child to the remaining durable budget; never extend sealed limits."""
    if limits != plan["request"]["limits"]["child"]:
        raise fail("INVALID_EXPERIMENT_SPEC", "child limits differ from sealed request")
    seconds = remaining(conn, store, registration, plan)
    resolved = execution_limits(limits)
    return {**resolved, "wall_seconds": min(resolved["wall_seconds"], seconds)}
