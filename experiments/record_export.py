"""Private run-record export for experiments (issue #1078).

Writes a stable EXP-###_<slug>/ run folder with canonical JSON results and a
REPORT.md stub. Never overwrites existing bytes; oversized docs are trimmed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from engine.v2.ops.errors import fail

MAX_JSON_BYTES = 1_000_000

_EXP_DIR_RE = re.compile(r"^EXP-(\d+)")
_EXP_NUM_DIR_RE = re.compile(r"^EXP-(\d+)_")
_SLUG_BAD_RE = re.compile(r"[^a-z0-9_]+")
_SLUG_MULTI_UNDER_RE = re.compile(r"_+")


def _dump(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sanitize_slug(slug) -> str:
    s = _SLUG_BAD_RE.sub("_", str(slug or "").lower())
    s = _SLUG_MULTI_UNDER_RE.sub("_", s).strip("_")
    if not s:
        s = "run"
    return s[:40]


def _read_manifest(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _manifest_for(folder: Path):
    return _read_manifest(folder / "results" / "export_manifest.json")


def _iter_exp_dirs(root: Path):
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and _EXP_DIR_RE.match(child.name):
            yield child


def _linked_number(root: Path, experiment_id):
    for folder in _iter_exp_dirs(root):
        man = _manifest_for(folder)
        if man is None:
            continue
        if man.get("experiment_id") == experiment_id:
            m = _EXP_DIR_RE.match(folder.name)
            if m:
                return int(m.group(1))
            num = man.get("exp_number")
            if isinstance(num, int):
                return num
    return None


def _highest_exp_number(root: Path) -> int:
    highest = 0
    for folder in _iter_exp_dirs(root):
        m = _EXP_DIR_RE.match(folder.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest


def _folders_with_number(root: Path, n: int):
    out = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = _EXP_NUM_DIR_RE.match(child.name)
        if m and int(m.group(1)) == n:
            out.append(child)
    return out


def resolve_exp_number(experiments_root, experiment_id, requested):
    """Resolve the EXP number for this run.

    ``requested`` is None, an int, or \"auto\" (digit strings are accepted).
    """
    root = Path(experiments_root)
    linked = _linked_number(root, experiment_id)

    req_int = None
    if isinstance(requested, bool):
        raise fail(
            "INVALID_EXPERIMENT_SPEC",
            f"invalid --exp-number {requested!r} for {experiment_id}",
        )
    if isinstance(requested, int):
        req_int = requested
    elif isinstance(requested, str):
        text = requested.strip()
        if text.lower() == "auto":
            req_int = None
        elif text.isdigit():
            req_int = int(text)
        else:
            raise fail(
                "INVALID_EXPERIMENT_SPEC",
                f"invalid --exp-number {requested!r} for {experiment_id}",
            )

    if linked is not None:
        if req_int is not None and req_int != linked:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"{experiment_id} is linked to EXP-{linked:03d} "
                f"but EXP-{req_int:03d} was requested",
            )
        return linked

    if req_int is None:
        if requested is None:
            raise fail(
                "INVALID_EXPERIMENT_SPEC",
                f"run has no EXP number: {experiment_id} is not linked under "
                f"{root} and --exp-number was not given",
            )
        return 1 + _highest_exp_number(root)

    existing = _folders_with_number(root, req_int)
    for folder in existing:
        man = _manifest_for(folder)
        if man is None:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"EXP-{req_int:03d} in {folder.name} has no export manifest "
                f"(legacy/unlinked folder), not linked to {experiment_id!r}",
            )
        if man.get("experiment_id") != experiment_id:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"EXP-{req_int:03d} in {folder.name} already belongs to "
                f"{man.get('experiment_id')!r}, not {experiment_id!r}",
            )
    return req_int


def _encode_doc(doc) -> tuple[bytes, bool]:
    raw = _dump(doc)
    if len(raw) <= MAX_JSON_BYTES:
        return raw, False
    stub = {
        "trimmed": True,
        "original_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if isinstance(doc, list):
        lo, hi = 0, len(doc)
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = dict(stub)
            cand["kept"] = doc[:mid]
            encoded = _dump(cand)
            if len(encoded) <= MAX_JSON_BYTES:
                best = encoded
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            return best, True
    return _dump(stub), True


def _write_bytes(path: Path, data: bytes) -> tuple[bool, bool]:
    """Atomically write ``data``; returns (wrote_new, conflicted)."""
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError:
            existing = None
        if existing == data:
            return False, False
        print(
            f"WARNING: record_export keeping existing {path} "
            f"({len(existing) if existing is not None else '?'} bytes) "
            f"would-be {len(data)} bytes",
            file=sys.stderr,
        )
        return False, True
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".record_export.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return True, False


def _resolve_folder(root: Path, exp_number: int, slug: str) -> Path:
    existing = _folders_with_number(root, exp_number)
    if existing:
        return existing[0]
    return root / f"EXP-{exp_number:03d}_{slug}"


def _status_line(outcome) -> str:
    if isinstance(outcome, dict):
        code = outcome.get("failure_code")
        if code:
            return str(code)
    return "ran"


def _render_report(exp_number: int, experiment_id, outcome, report_text) -> str:
    lines = [
        f"# EXP-{exp_number:03d} {experiment_id}",
        f"Status: {_status_line(outcome)}",
    ]
    if report_text is not None:
        lines.append("")
        lines.append(str(report_text))
    lines.append("")
    lines.append("Machine-readable run record: see `results/` in this directory.")
    return "\n".join(lines) + "\n"


def export_run_record(
    experiments_root,
    *,
    exp_number,
    experiment_id,
    slug,
    spec,
    request,
    event_ids,
    outcome,
    report_text,
    recording_mode,
):
    root = Path(experiments_root)
    exp_number = int(exp_number)
    for folder in _folders_with_number(root, exp_number):
        man = _manifest_for(folder)
        if man is None:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"EXP-{exp_number:03d} in {folder.name} has no export manifest "
                f"(legacy/unlinked folder), not linked to {experiment_id!r}",
            )
        if man.get("experiment_id") != experiment_id:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"EXP-{exp_number:03d} in {folder.name} already belongs to "
                f"{man.get('experiment_id')!r}, not {experiment_id!r}",
            )
    folder = _resolve_folder(root, exp_number, _sanitize_slug(slug))
    results = folder / "results"
    results.mkdir(parents=True, exist_ok=True)

    trimmed: list[str] = []
    file_hashes: dict[str, str] = {}
    wrote_any = False
    conflicted = False

    docs = {
        "spec.json": spec,
        "request.json": request,
        "event_ids.json": event_ids,
        "outcome.json": outcome,
    }
    for name, doc in docs.items():
        raw, was_trimmed = _encode_doc(doc)
        if was_trimmed and name not in trimmed:
            trimmed.append(name)
        wrote, conflict = _write_bytes(results / name, raw)
        wrote_any = wrote_any or wrote
        conflicted = conflicted or conflict
        file_hashes[name] = hashlib.sha256(raw).hexdigest()

    manifest = {
        "experiment_id": experiment_id,
        "exp_number": exp_number,
        "recording_mode": recording_mode,
        "trimmed": trimmed,
        "files": file_hashes,
    }
    man_raw = _dump(manifest)
    wrote, conflict = _write_bytes(results / "export_manifest.json", man_raw)
    wrote_any = wrote_any or wrote
    conflicted = conflicted or conflict

    report_path = folder / "REPORT.md"
    if not report_path.exists():
        report_raw = _render_report(
            exp_number, experiment_id, outcome, report_text
        ).encode("utf-8")
        wrote, conflict = _write_bytes(report_path, report_raw)
        wrote_any = wrote_any or wrote
        conflicted = conflicted or conflict

    if conflicted:
        status = "unchanged"
    else:
        status = "written" if wrote_any else "unchanged"
    return {"dir": str(folder), "status": status, "trimmed": trimmed}


def _git_commit_id() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def export_native_run(
    experiments_root,
    registration,
    spec_document,
    request,
    saved,
    exp_number,
    replayed,
    started,
    report_path=None,
):
    """Export one native run (gate or prediction) under its EXP folder."""
    document = registration.document
    experiment_id = document["execution_plan"]["experiment_id"]
    snapshot = document.get("snapshot")
    snapshot_id = snapshot.get("snapshot_id") if isinstance(snapshot, dict) else None
    report_text = None
    if saved["outcome"]["failure_code"] is None and report_path is not None:
        report_text = Path(report_path).read_text()
    closure = document.get("source_closure")
    source_closure_hash = hashlib.sha256(
        json.dumps(closure, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return export_run_record(
        experiments_root,
        exp_number=exp_number,
        experiment_id=experiment_id,
        slug=experiment_id,
        spec=spec_document,
        request=request,
        event_ids=document.get("event_ids", []),
        outcome={
            "outcome": saved["outcome"],
            "recording_mode": saved["receipt"].get("recording_mode"),
            "failure_code": saved["outcome"]["failure_code"],
            "replayed": replayed,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "snapshot_id": snapshot_id,
            "commit_id": _git_commit_id(),
            "source_closure_hash": source_closure_hash,
            "environment": document.get("environment"),
        },
        report_text=report_text,
        recording_mode=saved["receipt"].get("recording_mode"),
    )
