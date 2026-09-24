#!/usr/bin/env python3
"""The lifecycle-managed capture heartbeat: a progress/ETA line, always.

``tools/capture_tier0_corpus.py`` (and the offline
``tools/capture_attach_probe.py`` recapture path) are slow in exactly the
places a progress line cannot be printed from the work loop: loading a
selected dump, building the Scorer (the panel plus half a million replayed
trades), streaming one ``ChainIndex``, scoring one long candidate,
attaching one strict trace, writing one multi-GB pair, and finalizing /
renaming the published corpus. Milestones printed BETWEEN units go silent
while a single unit blocks, so this heartbeat runs on its own daemon
thread: whatever the main thread is stuck in, it states where the capture
is (``phase``), how long it has been running (``elapsed``), how many units
are done (``completed/total`` where known), and an estimated remaining
duration for the WHOLE operation -- labeled by what it is based on.

The ETA has two bases, never mixed silently:

* **prior-run** -- before any unit of the current stage has completed
  there is no throughput to measure, so the remaining time falls back to
  the last real full strict capture, version ``20260924T014500Z``: it
  began ~2026-09-23 21:45 Toronto time, its ``CAPTURE_DUMP_SELECTED`` dump
  landed ~22:04 (gathering ~19 min) and its ``INDEX.json`` ~23:06 (attach
  + write ~62 min; ~82 min wall clock total). The stage weights are exactly
  those measured sums. They are an estimate, not a schedule: every ETA
  prints as "~" and names its basis on every line.
* **observed** -- once units of a stage with a known total have completed,
  that stage's own measured rate projects the rest of it. Stages that have
  not started yet still carry their prior-run weights, and the label says
  so whenever the two are combined. Earlier gather sub-phases deliberately
  keep the prior-run basis: per-candidate scoring time is the most uneven
  signal in the run (one long candidate holds the loop for minutes), so
  extrapolating it would claim a reliability the measurement does not
  have. That is the "where reliable" rule, applied honestly.

**The actual guarantee (and its limit).** A thread cannot preempt a native
C call that holds the GIL. The once-a-minute beat is delivered whenever the
interpreter can schedule this daemon thread -- true for the pandas / I/O /
pure-Python work that dominates these tools (they release the GIL during
their long stretches), NOT for a single long-running C call that never
releases it. The interval is set strictly under a minute so that ordinary
scheduling jitter cannot drop a beat; a GIL-holding native call can. This
module does not pretend to beat such a call.

Stdlib only (``threading``/``time``), imports nothing from the engine,
writes nothing to disk, adds no persistent artifact: the heartbeat can
never change a capture's fixtures, and a test can drive it with a fake
clock. An un-started heartbeat is INERT -- ``begin_stage``/``phase``/
``advance`` do nothing at all -- so every capture function behaves exactly
as before outside the ``with heartbeat():`` lifecycle. Supporting two
independent captures in ONE process is out of scope: each owns its own
instance and installs it as the module singleton for its own run.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Iterable, Mapping

#: The last real full strict capture the prior-run fallback is measured
#: from (see the module docstring for the wall-clock evidence).
PRIOR_STRICT_CAPTURE = "20260924T014500Z"

#: Minutes, measured from that run: start->dump ~19, dump->INDEX ~62. The
#: ~62 of attach+write is a single measured total; the attach/write split
#: below is an even-ish assumption refined by observation as soon as the
#: first unit of either completes (the whole-capture ETA shown at t=0 is
#: their sum + gather = ~81m, the measured figure, never the split).
DEFAULT_STAGE_PRIORS_MINUTES: Mapping[str, float] = {
    "gather": 19.0,
    "attach": 41.0,
    "write": 21.0,
}
DEFAULT_STAGE_ORDER: tuple[str, ...] = ("gather", "attach", "write")

#: Strictly under a minute so ordinary scheduling jitter can never drop a
#: beat (a GIL-holding native call still can -- see the module docstring).
DEFAULT_INTERVAL_SECONDS = 45.0


def format_elapsed(seconds: float) -> str:
    """``5m03s`` for 303 seconds; never negative, never sub-second."""
    total = max(0, int(seconds))
    return f"{total // 60}m{total % 60:02d}s"


def format_eta(seconds: float) -> str:
    """A "~"-prefixed whole-minute estimate -- never presented as exact.

    One shared rounding function for both bases (prior-run and observed),
    so the two can never differ in precision theatre, only in the honest
    basis label beside the number. A still-pending stage floors at "~<1m",
    never "~0m", so an unfinished finalize/publication unit can never read
    as "done".
    """
    if seconds <= 0:
        return "~0m"
    if seconds < 60:
        return "~<1m"
    return f"~{math.ceil(seconds / 60)}m"


class CaptureHeartbeat:
    """A daemon-thread progress/ETA line for one long-running capture.

    Lifecycle state is guarded by one lock. ``beat``/``line`` take a single
    consistent snapshot of phase, counts AND the ETA under that lock, so a
    line can never mix a phase from one instant with a count from another.
    ``stop`` sets inert state and joins the thread with a bounded timeout,
    and ``__exit__`` guarantees that join even when the wrapped work raises.
    ``is_alive`` reports the thread's actual state: if a blocked emitter
    outlives the join timeout it is still reported alive (and the daemon
    thread dies with the interpreter), never silently dropped to look dead.
    """

    def __init__(
        self,
        *,
        prefix: str = "[corpus]",
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        emit: Callable[[str], None] | None = None,
        stage_priors_minutes: Mapping[str, float] = DEFAULT_STAGE_PRIORS_MINUTES,
        stage_order: Iterable[str] = DEFAULT_STAGE_ORDER,
        join_grace_seconds: float = 10.0,
        thread_name: str = "capture-heartbeat",
    ) -> None:
        self.prefix = prefix
        self.interval_seconds = float(interval_seconds)
        self.join_grace_seconds = float(join_grace_seconds)
        self._clock = clock
        self._emit = emit if emit is not None else self._print_line
        self._priors = dict(stage_priors_minutes)
        self._order = tuple(stage_order)
        self._thread_name = thread_name
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.running = False
        self._reset_state(0.0)

    def _reset_state(self, now: float) -> None:
        """Every per-run field, so a restart begins with no stale
        stage/phase/count and no leftover observed-rate projection."""
        self._run_started = now
        self._stage: str | None = None
        self._stage_started = now
        self._stage_units_total: int | None = None
        self._stage_units_done = 0
        self._phase: str | None = None
        self._phase_started = now
        self._phase_total: int | None = None
        self._phase_done = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Become live (fresh state), launch the daemon, emit the t=0 line."""
        if self.running:
            return
        self._wake.clear()
        with self._lock:
            self._reset_state(self._clock())
            self.running = True
            thread = threading.Thread(
                target=self._loop, name=self._thread_name, daemon=True,
            )
            self._thread = thread
        thread.start()
        self.beat()

    def stop(self) -> None:
        """Go inert and join the daemon. Safe to call twice or before start.

        The thread reference is kept while the thread is alive, so
        :attr:`is_alive` cannot lie even if a blocked emitter outlives the
        join timeout; a genuinely finished thread is joined and cleared.
        """
        self._wake.set()
        with self._lock:
            self.running = False
            thread = self._thread
        if thread is not None:
            thread.join(timeout=self.interval_seconds + self.join_grace_seconds)
            if not thread.is_alive():
                with self._lock:
                    if self._thread is thread:
                        self._thread = None

    def __enter__(self) -> "CaptureHeartbeat":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()  # cleanup on failure included
        return False

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def _loop(self) -> None:
        while not self._wake.wait(self.interval_seconds):
            try:
                self.beat()
            except Exception:  # noqa: BLE001 - a dead log must never kill a capture
                pass

    @staticmethod
    def _print_line(line: str) -> None:
        print(line, flush=True)

    # ------------------------------------------------------------------
    # state updates (inert until started, so capture functions untouched)
    # ------------------------------------------------------------------

    def begin_stage(self, name: str, *, units_total: int | None = None) -> None:
        """Enter a whole-operation stage: prior-ETA bookkeeping resets here."""
        with self._lock:
            if not self.running:
                return
            self._stage = name
            self._stage_started = self._clock()
            self._stage_units_total = units_total
            self._stage_units_done = 0
            self._phase = None
            self._phase_started = self._clock()
            self._phase_total = None
            self._phase_done = 0

    def phase(self, name: str, *, total: int | None = None) -> None:
        """Label the current unit of work; ``total`` when it is enumerable."""
        with self._lock:
            if not self.running:
                return
            self._phase = name
            self._phase_started = self._clock()
            self._phase_total = total
            self._phase_done = 0

    def advance(self, units: int = 1) -> None:
        """Mark ``units`` of the current stage done (feeds the observed ETA).

        Call this only once a unit's ENTIRE work -- the pair file, its
        checkpoint case, any finalization it owns -- has completed, so the
        count never runs ahead of durable progress.
        """
        with self._lock:
            if not self.running:
                return
            self._stage_units_done += units
            if self._phase is not None:
                self._phase_done += units

    # ------------------------------------------------------------------
    # ETA (the public entry snapshots time once; the core is lock-held)
    # ------------------------------------------------------------------

    def estimate_remaining(self, now: float | None = None) -> tuple[float, str]:
        """``(seconds_remaining_for_whole_operation, basis)``."""
        with self._lock:
            return self._estimate_locked(self._clock() if now is None else now)

    def _estimate_locked(self, now: float) -> tuple[float, str]:
        """Compute the remaining estimate; the caller must hold ``_lock``."""
        stage = self._stage
        units_total = self._stage_units_total
        units_done = self._stage_units_done
        future = self._future_prior_seconds(stage)
        if units_total and units_done > 0:
            rate = (now - self._stage_started) / units_done
            remaining = rate * (units_total - units_done) + future
            label = (
                f"observed: {units_done}/{units_total} {stage} unit(s) "
                f"measured this run"
            )
            if future > 0:
                label += " + prior-run for later stages"
            return remaining, label
        label = (
            f"prior-run estimate ({PRIOR_STRICT_CAPTURE}, whole operation "
            f"~{int(self._total_prior_seconds() // 60)}m)"
        )
        if stage is None:
            return (max(self._total_prior_seconds() - (now - self._run_started), 0.0),
                    label)
        current = self._order.index(stage) if stage in self._order else -1
        spent = max(self._priors.get(stage, 0.0) * 60.0 - (now - self._stage_started), 0.0)
        held = spent if current >= 0 else self._total_prior_seconds()
        return held + future, label

    def _total_prior_seconds(self) -> float:
        return sum(self._priors[name] for name in self._order) * 60.0

    def _future_prior_seconds(self, stage: str | None) -> float:
        current = self._order.index(stage) if stage in self._order else -1
        return sum(self._priors[name]
                   for index, name in enumerate(self._order) if index > current) * 60.0

    # ------------------------------------------------------------------
    # the line (one consistent snapshot of phase + counts + ETA)
    # ------------------------------------------------------------------

    def line(self, now: float | None = None) -> str:
        """Build one heartbeat line; phase, count and ETA share a snapshot."""
        with self._lock:
            stamp = self._clock() if now is None else now
            phase = self._phase or "starting"
            phase_total = self._phase_total
            phase_done = self._phase_done
            elapsed = stamp - self._run_started
            remaining, basis = self._estimate_locked(stamp)
        done = f"{phase_done}/{phase_total}" if phase_total is not None else "n/a"
        return (
            f"{self.prefix} heartbeat phase={phase} "
            f"elapsed={format_elapsed(elapsed)} "
            f"done={done} eta={format_eta(remaining)} remaining ({basis})"
        )

    def beat(self, now: float | None = None) -> str | None:
        """Emit one line (flushed) while live; a no-op returning ``None`` after.

        Returns the line it emitted (or would emit) so a test can assert on
        it; ``None`` once stopped, proving no beat survives ``stop``.
        """
        if not self.running:
            return None
        built = self.line(now)
        self._emit(built)
        return built
