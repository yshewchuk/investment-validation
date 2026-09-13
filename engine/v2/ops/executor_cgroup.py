"""Optional delegated cgroup v2 containment, never inferred from mount presence."""
from __future__ import annotations

import os
import uuid
from pathlib import Path


def probe(root: Path) -> dict:
    path = root / ("ops-probe-" + uuid.uuid4().hex)
    try:
        path.mkdir()
        required = ("memory.max", "memory.high", "cgroup.procs", "cgroup.kill", "cpuset.cpus")
        available = all((path / name).exists() and os.access(path / name, os.W_OK) for name in required)
        return {"available": available, "mode": "cgroup" if available else "watchdog"}
    except OSError:
        return {"available": False, "mode": "watchdog", "reason": "delegation_unavailable"}
    finally:
        if path.exists():
            path.rmdir()


def configure(root: Path, attempt_id: str, resources) -> Path:
    if not attempt_id.startswith("att_") or not attempt_id[4:].isalnum():
        raise ValueError("invalid attempt identity")
    path = root / attempt_id
    path.mkdir()
    (path / "memory.max").write_text(str(resources.reserved_memory_bytes))
    (path / "memory.high").write_text(str(resources.reserved_memory_bytes * 9 // 10))
    (path / "cpuset.cpus").write_text(",".join(map(str, resources.assigned_cpu_ids)))
    return path


def members(path: Path) -> list[int]:
    return [int(value) for value in (path / "cgroup.procs").read_text().split()]


def kill(path: Path) -> None:
    (path / "cgroup.kill").write_text("1")


def evidence(path: Path) -> dict:
    events = dict(line.split() for line in (path / "memory.events").read_text().splitlines())
    return {"current_bytes": int((path / "memory.current").read_text()),
            "oom_kill": int(events.get("oom_kill", 0)), "members": members(path)}
