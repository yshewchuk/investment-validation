"""Worker-side step API, and the reader that lets the supervisor follow it.

The worker subprocess (``engine/v2/ops/worker.py``) already talks to the
supervisor over exactly one channel: a one-shot pipe write at exit. There is
no existing live channel a step could ride on, so this module adds one --
a private, per-attempt, append-only newline-delimited-JSON file under the
attempt's own staging directory (``diagnostics/steps.ndjson``, beside the
existing ``worker.stderr``/``failure_details.json`` private files). The
supervisor tails it every poll tick (~1s) the same way ``executor_watchdog``
already samples memory at that resolution.

No-op by default (§ observation-only rule): a caller that never calls
:func:`configure` -- any legacy code imported outside a real ops worker
process, a unit test that exercises ``engine.score``/``engine.features``
directly -- gets an inert recorder. ``engine/v2/ops/worker.py`` is the only
caller that configures a real sink, once, before ``dispatch``.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["configure", "current_step", "read_new_records", "reset", "step",
          "step_end", "step_start", "STEPS_FILENAME"]

STEPS_FILENAME = "steps.ndjson"

#: The active recorder, or ``None`` (no-op) until ``configure`` runs.
_active: "_Recorder | None" = None


class _Recorder:
    def __init__(self, path: Path) -> None:
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._t0 = time.monotonic()
        #: name -> True while a step is open, so a mismatched end is a no-op
        #: rather than a crash inside instrumentation.
        self._open: dict[str, bool] = {}

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def write(self, record: dict) -> None:
        try:
            os.write(self._fd, json.dumps(record).encode() + b"\n")
        except OSError:
            pass

    def close(self) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass


def _rss_bytes() -> int:
    """Current resident set size of this process, best-effort.

    ``/proc/self/status`` VmRSS: a boundary reading, not a substitute for the
    supervisor's own ≤1s watchdog sampling (which also sees a fork/thread this
    process could not see about itself), but cheap and always available on
    this box's Linux hosts.
    """
    try:
        with open("/proc/self/status", "rb") as stream:
            for line in stream:
                if line.startswith(b"VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def configure(staging_root: Path | str) -> None:
    """Open the sink for this worker process. Real workers call this once,
    before ``dispatch`` -- see ``engine/v2/ops/worker.py::main``."""
    global _active
    reset()
    directory = Path(staging_root) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    _active = _Recorder(directory / STEPS_FILENAME)


def reset() -> None:
    """Close and clear the active sink. Idempotent; safe when never configured."""
    global _active
    if _active is not None:
        _active.close()
    _active = None


def current_step() -> str | None:
    """The most recently started, not-yet-ended step name, or ``None``."""
    if _active is None or not _active._open:
        return None
    return next(reversed(_active._open))


def step_start(name: str) -> None:
    if _active is None:
        return
    _active._open[name] = True
    _active.write({"event": "start", "step": name, "elapsed_seconds": _active.elapsed(),
                  "rss_bytes": _rss_bytes()})


def step_end(name: str, *, units: int | None = None) -> None:
    if _active is None:
        return
    _active._open.pop(name, None)
    record = {"event": "end", "step": name, "elapsed_seconds": _active.elapsed(),
             "rss_bytes": _rss_bytes()}
    if units is not None:
        record["units"] = units
    _active.write(record)


@contextmanager
def step(name: str, *, units: int | None = None) -> Iterator[None]:
    """``with step("scorer_build"):`` -- start/end even if the body raises."""
    step_start(name)
    try:
        yield
    finally:
        step_end(name, units=units)


def read_new_records(path: Path, offset: int) -> tuple[list[dict], int]:
    """Records appended to ``path`` since ``offset``, and the new offset.

    Reads only complete lines: a writer mid-``os.write`` on the next line
    leaves a partial trailing line, which stays unread until it completes on
    a later call. Missing file (nothing written yet, or an attempt that
    predates this feature) is an empty, unmoved read, never an error.
    """
    try:
        with open(path, "rb") as stream:
            stream.seek(offset)
            data = stream.read()
    except OSError:
        return [], offset
    complete, separator, _partial = data.rpartition(b"\n")
    if not separator:
        return [], offset
    records = []
    for line in complete.decode("utf-8", errors="replace").splitlines():
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records, offset + len(complete) + len(separator)
