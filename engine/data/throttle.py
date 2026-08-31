"""Per-source pacing, backoff, and the quota guard — centralized.

The AGENTS.md throttling playbook exists because multi-hour pulls died without
it. Rather than sprinkle sleeps at call sites, every source has one config here
and :meth:`Throttle.acquire` is called once, at the top of the fetch path.

Three mechanisms:

**Pacing.** A minimum interval between consecutive network calls per source.
Polygon's price/aggs endpoints are ~10 req/min on this plan, so the 6.5s gate
is the difference between a pull that finishes and one that stalls. ORATS is
quota-limited rather than rate-limited (1,000 req/min against a 20k/month
budget), so its pacing is nominal.

**Backoff.** On 429 or 5xx, sleep hard and retry — Polygon's 65s backoff, six
attempts. Hammering a throttled endpoint extends the stall rather than ending it.

**Quota guard.** ORATS bills per call against 20,000/month. Below the
``ORATS_RESERVE_FLOOR`` the guard refuses new calls so daily live operation
still has budget, unless ``ORATS_ALLOW_RESERVE=1`` says otherwise.

Cross-process safety: Polygon splits one 10/min budget across every process
touching it, so two concurrent pulls both stall. A lockfile enforces one.
"""
from __future__ import annotations

import errno
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from engine import paths

__all__ = [
    "SourceConfig",
    "SOURCES",
    "Throttle",
    "QuotaExhausted",
    "PolygonBusy",
    "polygon_lock",
    "ORATS_RESERVE_FLOOR",
]

#: Calls held back for daily live operation (Phase 3), out of 20,000/month.
ORATS_RESERVE_FLOOR = 3_000


class QuotaExhausted(RuntimeError):
    """A source's remaining quota is below its reserve floor."""


class PolygonBusy(RuntimeError):
    """Another process already holds the Polygon lock."""


@dataclass(frozen=True)
class SourceConfig:
    name: str
    min_interval: float  # seconds between consecutive network calls
    backoff_base: float  # seconds to sleep on the first 429/5xx
    max_retries: int
    timeout: float
    monthly_quota: int | None = None
    reserve_floor: int = 0
    single_process: bool = False
    notes: str = ""


SOURCES: dict[str, SourceConfig] = {
    "orats": SourceConfig(
        name="orats",
        min_interval=0.3,
        backoff_base=30.0,
        max_retries=6,
        timeout=180.0,
        monthly_quota=20_000,
        reserve_floor=ORATS_RESERVE_FLOOR,
        notes=(
            "Auth is the ?token= query param, not a header. 1,000 req/min, so "
            "quota is the binding constraint, not rate. Batch pulls need "
            "timeout >= 120s (10 big tickers is ~43k rows / ~2.8MB)."
        ),
    ),
    "polygon": SourceConfig(
        name="polygon",
        min_interval=6.5,
        backoff_base=65.0,
        max_retries=6,
        timeout=120.0,
        single_process=True,
        notes=(
            "price/aggs ~10 req/min. Fresh-process urllib 401s with a valid "
            "key, so the adapter shells out to curl. One process at a time."
        ),
    ),
    "oquants": SourceConfig(
        name="oquants",
        min_interval=0.5,
        backoff_base=15.0,
        max_retries=4,
        timeout=120.0,
        notes="Playwright cookie→token dance; reuse the bearer within a process.",
    ),
    "nasdaq": SourceConfig(
        name="nasdaq",
        min_interval=0.7,
        backoff_base=15.0,
        max_retries=3,
        timeout=60.0,
        notes=(
            "Keyless public calendar endpoint, one call per DATE for the whole "
            "market. Refuses the default python user-agent with a 403, which is "
            "a code fix and not a credential failure. Paced politely: a nightly "
            "needs ~21 calls."
        ),
    ),
    "yfinance": SourceConfig(
        name="yfinance",
        min_interval=0.2,
        backoff_base=10.0,
        max_retries=3,
        timeout=60.0,
        notes="Unmetered but rate-sensitive; disk-cached like everything else.",
    ),
}


class Throttle:
    """Pacing and backoff state for all sources in one process.

    ``sleep_fn`` and ``time_fn`` are injectable so the test-suite can assert on
    the pacing arithmetic without spending wall-clock seconds on it.
    """

    def __init__(
        self,
        configs: dict[str, SourceConfig] | None = None,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
    ):
        self.configs = dict(configs or SOURCES)
        self._sleep = sleep_fn
        self._time = time_fn
        self._last: dict[str, float] = {}
        self.slept_total: float = 0.0

    def config(self, source: str) -> SourceConfig:
        try:
            return self.configs[source]
        except KeyError:
            raise KeyError(
                f"unknown source {source!r}; known: {sorted(self.configs)}"
            ) from None

    def acquire(self, source: str) -> float:
        """Block until it is polite to make a network call. Returns seconds slept."""
        cfg = self.config(source)
        now = self._time()
        last = self._last.get(source)
        wait = 0.0
        if last is not None:
            wait = max(0.0, cfg.min_interval - (now - last))
            if wait > 0:
                self._sleep(wait)
                self.slept_total += wait
        self._last[source] = self._time()
        return wait

    def backoff(self, source: str, attempt: int) -> float:
        """Sleep after a throttled or failed attempt. Returns seconds slept."""
        cfg = self.config(source)
        delay = cfg.backoff_base * (1 + attempt)
        self._sleep(delay)
        self.slept_total += delay
        self._last[source] = self._time()
        return delay

    # -- quota ------------------------------------------------------------

    def check_quota(self, source: str, remaining: int | None) -> None:
        """Raise :class:`QuotaExhausted` when a call would eat into the reserve.

        ``remaining is None`` is not treated as permission to proceed blindly —
        ORATS omits quota headers when a response is served from a CDN cache —
        but it cannot block either, so the call is allowed and the caller is
        expected to count rows in the quota log instead.
        """
        cfg = self.config(source)
        if cfg.reserve_floor <= 0 or remaining is None:
            return
        if os.environ.get(f"{source.upper()}_ALLOW_RESERVE") == "1":
            return
        if remaining < cfg.reserve_floor:
            raise QuotaExhausted(
                f"{source}: {remaining} calls remaining is below the "
                f"{cfg.reserve_floor}-call live-operations reserve. "
                f"Set {source.upper()}_ALLOW_RESERVE=1 to spend it deliberately."
            )


# --------------------------------------------------------------------------
# cross-process lock
# --------------------------------------------------------------------------


class _FileLock:
    """Minimal exclusive lock via ``O_EXCL``, with a stale-PID takeover.

    Deliberately not ``fcntl.flock``: a WSL2 host that sleeps mid-job can leave
    a lock held by a process that no longer exists, and a lock nobody can break
    is worse than no lock at all. The PID recorded in the file lets a later run
    tell a live holder from a dead one.
    """

    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def _holder_alive(self) -> bool:
        try:
            pid = int(self.path.read_text().strip() or 0)
        except (OSError, ValueError):
            return False
        if pid <= 0 or pid == os.getpid():
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError as exc:
            return exc.errno == errno.EPERM  # exists but owned by someone else

    def acquire(self) -> None:
        paths.assert_writable(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._holder_alive():
                raise PolygonBusy(
                    f"another process holds {self.path}; Polygon allows one at a time"
                ) from None
            self.path.unlink(missing_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as fh:
            fh.write(str(os.getpid()))
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "_FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def polygon_lock(path: Path | None = None) -> _FileLock:
    """Context manager asserting this is the only Polygon-touching process."""
    return _FileLock(path or paths.POLYGON_LOCK)
