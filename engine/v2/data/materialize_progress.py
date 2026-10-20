"""Bounded stderr progress for legacy materialization phases (#1018).

Never stdout, never per item; cadence is ``_PROGRESS_INTERVAL_S``.
"""
from __future__ import annotations

import resource
import sys
import time

_PROGRESS_INTERVAL_S = 30.0


class TableProgress:
    """Bounded ``[materialize]`` progress on stderr for one phase (#1018).

    Never prints to stdout, never raises (an OSError from a closed or
    full stderr is swallowed), and never prints more than one line per
    :data:`_PROGRESS_INTERVAL_S` (every tick at ``done == total``, plus the
    one ``start``/``done in`` bracket line each phase).
    """

    def __init__(self, total: int, label: str = "tables") -> None:
        self.total = total
        self.label = label
        self.done = 0
        self.started = time.monotonic()
        self.last = self.started
        self._print(f"[materialize] start, {total} {label}")

    def tick(self, done=None) -> None:
        """Advance one table (or jump to ``done``) and maybe print a line."""
        self.done = self.done + 1 if done is None else done
        now = time.monotonic()
        if self.done != self.total and now - self.last < _PROGRESS_INTERVAL_S:
            return
        self.last = now
        elapsed = now - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.done
        eta = remaining / rate if rate > 0 else 0.0
        self._print(f"[materialize] {self.done}/{self.total} {self.label}, "
                    f"{rate:.2f}/s, ETA {eta:.0f}s")

    def finish(self) -> None:
        """Print the phase's elapsed-seconds bracket line with peak RSS."""
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        self._print(f"[materialize] done in {time.monotonic() - self.started:.1f}s, "
                    f"peak RSS {peak_mb:.0f} MB")

    @staticmethod
    def _print(message: str) -> None:
        if sys.stderr is None:
            return
        try:
            print(message, file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass
