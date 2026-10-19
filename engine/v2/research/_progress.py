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

__all__ = ["Progress", "stage"]

DEFAULT_INTERVAL_S = 30.0


def _emit(line: str) -> None:
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        return


@contextmanager
def stage(name, *, clock=time.monotonic, emit=_emit):
    """Log ``start`` and ``done in Ns`` (or ``failed after Ns``) around a stage."""
    started = clock()
    emit(f"  [{name}] start")
    try:
        yield
    except BaseException:
        emit(f"  [{name}] failed after {clock() - started:.0f}s")
        raise
    emit(f"  [{name}] done in {clock() - started:.0f}s")


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
                   f"{rate:.2f}/s, ETA {eta:.0f}s")
