"""Job-end read summary and per-step read deltas (#754)."""
from __future__ import annotations

import time
from typing import Any

__all__ = ["ReadTracker"]


class ReadTracker:
    """Per-step read deltas plus a job-end summary for ``nightly_raw_row_producer``."""

    def __init__(self, counters: Any) -> None:
        self._counters = counters
        try:
            self._t0 = time.monotonic()
        except Exception:
            self._t0 = None
        try:
            self._cpu0 = time.process_time()
        except Exception:
            self._cpu0 = None
        try:
            self._last = counters.snapshot()
        except Exception:
            self._last = {}
        self._steps: list[dict[str, Any]] = []

    def delta(self) -> dict[str, dict[str, float]]:
        """``{table: {"rows_decoded": d, "bytes_read": d}}`` since the previous call,
        top 3 tables by rows decoded (ties by name), omitting tables with both zero;
        updates the last-seen snapshot. Degrades to ``{}`` on any error."""
        try:
            current = self._counters.snapshot()
            previous, self._last = self._last, current
            delta: dict[str, dict[str, float]] = {}
            for table, row in current.items():
                prior = previous.get(table, {})
                rows_decoded = row.get("rows_decoded", 0) - prior.get("rows_decoded", 0)
                bytes_read = row.get("bytes_read", 0) - prior.get("bytes_read", 0)
                if rows_decoded or bytes_read:
                    delta[table] = {"rows_decoded": rows_decoded, "bytes_read": bytes_read}
            ranked = sorted(delta.items(), key=lambda item: (-item[1]["rows_decoded"], item[0]))
            return dict(ranked[:3])
        except Exception:
            return {}

    def record(self, step_name: str, seconds: float, reads: dict[str, Any]) -> None:
        """Keep the 10 slowest steps, sorted slowest first. Degrades silently."""
        try:
            self._steps.append({"step": step_name, "seconds": seconds, "reads": reads})
            self._steps.sort(key=lambda item: item["seconds"], reverse=True)
            del self._steps[10:]
        except Exception:
            return

    def summary(self) -> dict[str, Any]:
        """Wall/cpu/peak RSS/process read bytes, per-table totals with amplification,
        the slowest-10 steps, and ``unavailable`` field names. Each field degrades
        independently (failed fields are ``None`` and named in ``unavailable``);
        only a wholly unexpected error yields ``{"unavailable": ["summary"]}``."""
        unavailable: list[str] = []
        try:
            try:
                wall = time.monotonic() - self._t0 if self._t0 is not None else None
            except Exception:
                wall = None
            if wall is None:
                unavailable.append("wall_seconds")
            try:
                cpu = time.process_time() - self._cpu0 if self._cpu0 is not None else None
            except Exception:
                cpu = None
            if cpu is None:
                unavailable.append("cpu_seconds")
            try:
                import resource
                peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            except Exception:
                peak_rss = None
                unavailable.append("peak_rss_bytes")
            try:
                with open("/proc/self/io", "r", encoding="utf-8") as stream:
                    process_read = next(
                        int(line.split()[1]) for line in stream if line.startswith("rchar:"))
            except Exception:
                process_read = None
                unavailable.append("process_read_bytes")
            try:
                snapshot = self._counters.snapshot()
                tables: dict[str, dict[str, Any]] = {}
                for table, row in snapshot.items():
                    returned = row.get("rows_returned", 0)
                    tables[table] = {
                        "scans": row.get("scans"), "fragments_opened": row.get("fragments_opened"),
                        "rows_decoded": row.get("rows_decoded"), "rows_returned": returned,
                        "bytes_read": row.get("bytes_read"), "seconds": row.get("seconds"),
                        "amplification": (row.get("rows_decoded") / returned) if returned else None,
                    }
            except Exception:
                tables = None
                unavailable.append("tables")
            slowest_steps = list(self._steps)
            return {"wall_seconds": wall, "cpu_seconds": cpu, "peak_rss_bytes": peak_rss,
                    "process_read_bytes": process_read, "tables": tables,
                    "slowest_steps": slowest_steps, "unavailable": unavailable}
        except Exception:
            return {"unavailable": ["summary"]}