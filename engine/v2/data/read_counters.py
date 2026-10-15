"""Per-``Repository`` read counters (#754): observational only.

One :class:`ReadCounters` lives on each ``Repository`` instance, so counts are
scoped to the job attempt that owns the repository; nothing is global. A
counting error is swallowed: instrumentation never changes scan results or
fails a scan.
"""
from __future__ import annotations

import time

__all__ = ["ReadCounters", "clock"]


def clock() -> float:
    """The monotonic clock the counters' ``seconds`` use."""
    return time.perf_counter()


_FIELDS = ("scans", "fragments_opened", "rows_decoded", "rows_returned",
           "bytes_read", "seconds")


class ReadCounters:
    """Per-table totals of scans, fragments opened, rows decoded and returned,
    bytes read (fragment file size) and seconds (open plus read-and-filter)."""

    def __init__(self) -> None:
        self._tables: dict[str, dict[str, float]] = {}

    def add(self, table: str, **amounts: float) -> None:
        """Add ``amounts`` (keys from the field list) to ``table``'s totals."""
        try:
            row = self._tables.setdefault(table, dict.fromkeys(_FIELDS, 0))
            for name, amount in amounts.items():
                if name in row:
                    row[name] += amount
        except Exception:
            return

    def snapshot(self) -> dict[str, dict[str, float]]:
        """A plain copy: ``{table: {field: total}}``."""
        return {table: dict(row) for table, row in self._tables.items()}
