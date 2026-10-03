"""Read effective Linux limits without starting jobs or changing host policy."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from engine.v2.contracts import CapacitySample
from engine.v2.foundation import Clock, format_timestamp


def cgroup_directory(proc: Path = Path("/proc"), mount: Path = Path("/sys/fs/cgroup")) -> Path:
    for line in (proc / "self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            return mount / line[3:].lstrip("/")
    return mount


def _inactive_file_bytes(directory: Path) -> int | None:
    """The exact ``inactive_file`` value from this directory's ``memory.stat``.

    ``None`` when the file is unreadable (including undecodable content), the
    key is absent, or the value is malformed/non-numeric/negative — callers
    then keep raw ``memory.current``.
    """
    try:
        lines = (directory / "memory.stat").read_text().splitlines()
    except (OSError, UnicodeError):
        return None
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] == "inactive_file":
            try:
                value = int(fields[1])
            except ValueError:
                return None
            return value if value >= 0 else None
    return None


def memory_limits(directory: Path) -> tuple[int | None, int | None]:
    limits = []
    for parent in (directory, *directory.parents):
        maximum = parent / "memory.max"
        if maximum.exists():
            value = maximum.read_text().strip()
            if value != "max":
                current = parent / "memory.current"
                used = int(current.read_text()) if current.exists() else None
                if used is not None:
                    inactive = _inactive_file_bytes(parent)
                    if inactive is not None:
                        used = max(0, used - inactive)
                limits.append((int(value), used))
    if not limits:
        return None, None
    ceiling = min(limit for limit, _ in limits)
    remaining = min(0 if used is None else limit - used for limit, used in limits)
    return ceiling, ceiling - remaining


def sample_capacity(root: Path, *, clock: Clock, proc: Path = Path("/proc")) -> CapacitySample:
    mem = {}
    for line in (proc / "meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.split()[0]) * 1024
    limit, current = memory_limits(cgroup_directory(proc))
    return CapacitySample(
        sampled_at=format_timestamp(clock.now()), allowed_cpu_ids=tuple(sorted(os.sched_getaffinity(0))),
        host_total_bytes=mem["MemTotal"], host_available_bytes=mem["MemAvailable"],
        container_limit_bytes=limit, container_current_bytes=current,
        swap_total_bytes=mem["SwapTotal"], swap_free_bytes=mem["SwapFree"],
        disk_free_bytes=shutil.disk_usage(root).free,
        executor_mode="watchdog", containment="best_effort",
        notes=("Cgroup containment requires an explicitly probed delegated root.",))
