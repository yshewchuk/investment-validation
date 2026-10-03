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


def _ascii_decimal(token: str) -> int | None:
    """One strictly non-negative ASCII-decimal stat value, else ``None``.

    Rejects signs, underscores, non-ASCII/Unicode digits and non-numeric
    text; a digit string too long for ``int`` (CPython's conversion limit
    raises ``ValueError``) is unusable too.
    """
    if not token.isascii() or not token.isdigit():
        return None
    try:
        value = int(token)
    except ValueError:
        return None
    return value if value >= 0 else None


def _reclaimable_cache_bytes(directory: Path) -> int | None:
    """Reclaimable page cache from this directory's ``memory.stat`` (#347).

    ``file - shmem`` when the exact ``file`` and ``shmem`` fields are both
    valid ASCII decimals and ``shmem <= file`` — active and inactive file
    cache are reclaimable while shmem/tmpfs stays counted, and a valid
    ``0``/``0`` pair takes precedence over the fallback. Otherwise the exact
    ``inactive_file`` value when it is valid (the #343 fallback). ``None``
    when the stat file is unreadable/undecodable or no usable field is
    present, so callers keep raw ``memory.current``. One read per directory;
    unrelated, similarly named keys are ignored.
    """
    try:
        lines = (directory / "memory.stat").read_text().splitlines()
    except (OSError, UnicodeError):
        return None
    values = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] in ("file", "shmem", "inactive_file"):
            values[fields[0]] = fields[1]
    file_bytes = _ascii_decimal(values["file"]) if "file" in values else None
    shmem_bytes = _ascii_decimal(values["shmem"]) if "shmem" in values else None
    if file_bytes is not None and shmem_bytes is not None and shmem_bytes <= file_bytes:
        return file_bytes - shmem_bytes
    if "inactive_file" in values:
        return _ascii_decimal(values["inactive_file"])
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
                    reclaimable = _reclaimable_cache_bytes(parent)
                    if reclaimable is not None:
                        used = max(0, used - reclaimable)
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
