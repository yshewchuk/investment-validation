"""Read-only operator diagnostics with no catalog mutation or secret reads."""
from __future__ import annotations

import configparser
import hashlib
import json
import os
from pathlib import Path

from engine.v2.foundation import content_hash
from engine.v2.ops.executor_cgroup import probe
from engine.v2.ops.executor_watchdog import process_table
from engine.v2.ops.fingerprints import environment_identity
from engine.v2.ops.profiles import DEFAULT_POLICY


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _hook_path(root: Path) -> Path:
    git = root / ".git"
    if git.is_file():
        text = git.read_text().strip()
        if text.startswith("gitdir:"):
            git = (root / text.split(":", 1)[1].strip()).resolve()
    config = configparser.ConfigParser()
    config_path = git / "config"
    if config_path.is_file():
        config.read(config_path)
        value = config.get("core", "hooksPath", fallback="")
        if value:
            return (root / value).resolve() if not os.path.isabs(value) else Path(value)
    return git / "hooks"


def hook_status(root: Path) -> dict:
    path = _hook_path(root) / "pre-commit"
    if not path.is_file():
        return {"path": str(path), "state": "missing"}
    actual = _hash(path)
    source = root / "checks" / "hooks" / "pre-commit"
    expected = _hash(source) if source.is_file() else None
    return {"path": str(path), "state": "present", "hash": actual,
            "expected_hash": expected,
            "drift": expected is not None and actual != expected,
            "executable": os.access(path, os.X_OK)}


def owned_identities(conn) -> set[tuple[str, int, int]]:
    rows = conn.execute("SELECT pid, start_ticks, identity_json FROM process_members").fetchall()
    return {(json.loads(row[2])["boot_id"], row[0], row[1]) for row in rows}


def unmanaged_processes(conn, *, boot_id: str) -> list[dict]:
    owned = owned_identities(conn)
    rows = []
    for identity, _, state, rss, _ in process_table(boot_id).values():
        if (identity.boot_id, identity.pid, identity.start_ticks) in owned:
            continue
        if identity.pid in {0, 1, os.getpid()}:
            continue
        try:
            comm = (Path("/proc") / str(identity.pid) / "comm").read_text().strip()
        except OSError:
            comm = "unknown"
        rows.append({"pid": identity.pid, "start_ticks": identity.start_ticks,
                     "comm": comm, "state": state, "rss_bytes": rss})
    return sorted(rows, key=lambda row: row["pid"])


def environment_status(conn) -> dict:
    current = environment_identity()
    expected = [row[0] for row in conn.execute(
        "SELECT DISTINCT json_extract(spec_json, ?) FROM jobs WHERE state IN (?,?,?)",
        ("$.environment_ref", "queued", "running", "retry_wait")).fetchall() if row[0]]
    current_hash = content_hash(current)
    return {"current": current, "planned_refs": expected,
            "current_hash": current_hash,
            "drift": bool(expected and current_hash not in expected)}


def report(conn, root: Path, *, boot_id: str) -> dict:
    return {
        "hook": hook_status(root),
        "unmanaged_processes": unmanaged_processes(conn, boot_id=boot_id),
        "environment": environment_status(conn),
        "executor": probe(Path("/sys/fs/cgroup")),
        "profiles": [{"name": item.name, "memory_bytes": item.memory_bytes,
                      "cpu_count": item.cpu_count, "scratch_bytes": item.scratch_bytes}
                     for item in DEFAULT_POLICY.profiles],
    }


#: The production subtrees a shadow worker has no business touching (§9.2, O16).
SENSITIVE_ROOTS = ("ledger", "data/features", "engine/models", "reports", "dashboard")


def snapshot_sensitive(root: Path, domains=SENSITIVE_ROOTS) -> dict:
    """The before-side of a write audit: identity stamps of sensitive files."""
    state = {}
    base = Path(root)
    for domain in domains:
        top = base / domain
        if top.is_symlink() or not top.is_dir():
            continue
        for path in sorted(top.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            info = path.lstat()
            state[str(path.relative_to(base))] = (info.st_mtime_ns, info.st_size)
    return state


def audit_writes(root: Path, before: dict, disclosed=()) -> list[dict]:
    """New or modified sensitive files outside the disclosed roots (O16)."""
    base = Path(root)
    after = snapshot_sensitive(base)
    disclosed_paths = tuple(str(Path(item)) for item in disclosed)
    findings = []
    for name in sorted(set(after) | set(before)):
        if name in before and before[name] == after.get(name):
            continue
        absolute = str((base / name).resolve()) if name in after else str(base / name)
        if any(absolute == item or absolute.startswith(item + "/")
               for item in disclosed_paths):
            continue
        findings.append({"path": name, "kind": "new" if name not in before else "modified"})
    return findings
