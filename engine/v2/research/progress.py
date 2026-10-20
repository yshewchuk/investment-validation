"""Bounded-cadence stderr progress for long research passes.

Output goes to stderr only (stdout carries the JSON result callers parse), in the
nightly scorer's ``[score] N/M events, ... ETA`` style. Ticks are rate-limited by
time, never per item, so a pass shorter than ``min_interval`` prints nothing.
Logging failures are swallowed: progress never changes results or errors.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager

from engine.v2.data.errors import DataError

__all__ = ["Progress", "decision", "stage"]

DEFAULT_INTERVAL_S = 30.0


def _emit(line: str) -> None:
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        return


def _rss() -> str:
    """Peak RSS suffix for stage/progress lines; empty when unavailable."""
    try:
        import resource
        return f", peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB"
    except Exception:
        return ""


def decision(name: str, message: str) -> None:
    """Emit one aggregate control decision, never call once per item."""
    _emit(f"  [{name}] {message}")


@contextmanager
def stage(name, *, quiet_under=0.0, clock=time.monotonic, emit=_emit, report_refusals=False):
    """Log ``start`` and ``done in Ns`` (or ``failed after Ns``) with peak RSS around a stage.

    With ``quiet_under`` > 0 the start line is omitted and the end line is printed only
    when the stage took at least that many seconds, so short calls stay silent.
    ``report_refusals`` adds native DataError code/redacted message, never row/details data."""
    started = clock()
    if not quiet_under:
        emit(f"  [{name}] start{_rss()}")
    try:
        yield
    except BaseException as exc:
        elapsed = clock() - started
        if not quiet_under or elapsed >= quiet_under:
            reason = f"; refused {exc.code}: {exc.problem.message}" if report_refusals and isinstance(exc, DataError) else ""
            emit(f"  [{name}] failed after {elapsed:.0f}s{_rss()}{reason}")
        raise
    elapsed = clock() - started
    if not quiet_under or elapsed >= quiet_under:
        emit(f"  [{name}] done in {elapsed:.0f}s{_rss()}")


class Progress:
    """``tick()`` after each item; emits at most one line per ``min_interval`` seconds
    and not before ``min_items`` items have completed since the last line."""

    def __init__(self, name, total, *, min_interval=DEFAULT_INTERVAL_S, min_items=1,
                 clock=time.monotonic, emit=_emit):
        self.name, self.total = name, total
        self.min_interval, self.min_items = min_interval, min_items
        self._clock, self._emit = clock, emit
        self._started = self._last = clock()
        self.done = self._last_done = 0
        self.lines = 0

    def tick(self, count=1):
        self.done += count
        now = self._clock()
        if now - self._last < self.min_interval or self.done - self._last_done < self.min_items:
            return
        elapsed = now - self._started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        self._last, self._last_done = now, self.done
        self.lines += 1
        self._emit(f"  [{self.name}] {self.done:,}/{self.total:,}, {elapsed:.0f}s elapsed, "
                   f"{rate:.2f}/s, ETA {eta:.0f}s{_rss()}")
