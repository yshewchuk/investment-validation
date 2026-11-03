"""Dedicated-process CPU/RSS/wall bounds without economic experiment imports."""
import math
import resource
import signal
from contextlib import contextmanager

from engine.v2.ops.errors import fail
from experiments.magnitude_search_execution import validate_execution_limits
from experiments.memory_guard import rss_watchdog

__all__ = ["contained"]


def require(condition):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution run contract")


@contextmanager
def contained(limits, seconds):
    """Dedicated main-thread process only; restore soft limits, never raise hard limits."""
    validate_execution_limits(limits)
    require(type(seconds) in (int, float) and 0 < seconds < math.inf)
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))
    old = resource.getrlimit(resource.RLIMIT_CPU)
    handler = signal.getsignal(signal.SIGALRM)
    def expired(*args):
        raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution whole-job deadline exceeded")
    try:
        used = resource.getrusage(resource.RUSAGE_SELF)
        requested = math.ceil(used.ru_utime + used.ru_stime) + limits["cpu_seconds"]
        soft, hard = old
        ceilings = [requested] + [value for value in old if value != resource.RLIM_INFINITY]
        resource.setrlimit(resource.RLIMIT_CPU, (min(ceilings), hard))
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(limits["wall_seconds"], seconds))
        with rss_watchdog(limits["memory_bytes"]):
            yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        resource.setrlimit(resource.RLIMIT_CPU, old)

