"""Private run-record export for experiments (issue #1078).

Writes a stable EXP-###_<slug>/ run folder with canonical JSON results and a
REPORT.md stub. The operator assigns the number; no existing bytes are ever
overwritten, oversized docs are trimmed, and every conflict is preflighted
before the first write.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import stat as stat_mod
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from engine.v2.ops.errors import fail
from experiments.lib import FIRST_NUMBER, experiment_dirs, parse_experiment_id, slugify

MAX_JSON_BYTES = 1_000_000
_NUM_DIR_RE = re.compile(r"^EXP-(\d+)_")


def _dump(obj) -> bytes:
    return (
        json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_capped(path: Path) -> bytes | None:
    """Read at most MAX_JSON_BYTES+1 bytes; None if unreadable/non-regular."""
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat_mod.S_ISREG(os.fstat(fd).st_mode):
            return None
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_JSON_BYTES:
            chunk = os.read(fd, MAX_JSON_BYTES + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _read_manifest(path: Path):
    """Parse a manifest JSON object; None if missing/oversized/invalid."""
    raw = _read_capped(path)
    if raw is None or len(raw) > MAX_JSON_BYTES:
        return None
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _read_manifest_following(path: Path):
    """Like _read_manifest but follows symlinks (plain capped open)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(MAX_JSON_BYTES + 1)
    except OSError:
        return None
    if len(raw) > MAX_JSON_BYTES:
        return None
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _manifest_for(folder: Path):
    """Export manifest for folder; None for a symlinked folder/results dir."""
    if folder.is_symlink() or (folder / "results").is_symlink():
        return None
    return _read_manifest(folder / "results" / "export_manifest.json")


def _linked_number(root: Path, experiment_id):
    """Find an EXP folder whose manifest links this experiment_id."""
    for num, folder in experiment_dirs(root).items():
        man = _manifest_for(folder)
        if man is None:
            if not (folder.is_symlink() or (folder / "results").is_symlink()):
                continue
            man = _read_manifest_following(
                folder / "results" / "export_manifest.json"
            )
            if (
                man is None
                or man.get("experiment_id") != experiment_id
            ):
                continue
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"{experiment_id} in {folder.name}: run folder is a symlink",
            )
        if man.get("experiment_id") != experiment_id:
            continue
        if man.get("exp_number") != num:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"export manifest in {folder.name} claims EXP-{man.get('exp_number')!r} "
                f"but the folder name is EXP-{num:03d}",
            )
        return num
    return None


def _folders_with_number(root: Path, n: int) -> list[Path]:
    """All EXP-###_ dirs in root with number n."""
    if not root.is_dir():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if child.is_symlink() or child.is_dir():
            m = _NUM_DIR_RE.match(child.name)
            if m and int(m.group(1)) == n:
                out.append(child)
    return out


def _check_number_folder(folder: Path, exp_number: int, experiment_id):
    """Reject a symlinked folder/results dir, manifestless folder, or one
    linked to a different experiment_id."""
    if folder.is_symlink():
        raise fail(
            "EXPERIMENT_IDENTITY_CONFLICT",
            f"folder {folder.name} is a symlink; refusing to treat it as EXP-{exp_number:03d}",
        )
    if (folder / "results").is_symlink():
        raise fail(
            "EXPERIMENT_IDENTITY_CONFLICT",
            f"results folder in {folder.name} is a symlink; refusing to "
            f"treat it as EXP-{exp_number:03d}",
        )
    man = _manifest_for(folder)
    if man is None:
        raise fail(
            "EXPERIMENT_IDENTITY_CONFLICT",
            f"EXP-{exp_number:03d} in {folder.name} has no export manifest "
            f"(unlinked folder), not linked to {experiment_id!r}",
        )
    if man.get("experiment_id") != experiment_id:
        raise fail(
            "EXPERIMENT_IDENTITY_CONFLICT",
            f"EXP-{exp_number:03d} in {folder.name} already belongs to "
            f"{man.get('experiment_id')!r}, not {experiment_id!r}",
        )
    if man.get("exp_number") != exp_number:
        raise fail(
            "EXPERIMENT_IDENTITY_CONFLICT",
            f"export manifest in {folder.name} claims EXP-{man.get('exp_number')!r} "
            f"but the folder name is EXP-{exp_number:03d}",
        )


def _bad(requested, experiment_id, why: str = ""):
    """INVALID_EXPERIMENT_SPEC for a bad --exp-number value."""
    msg = f"invalid --exp-number {requested!r} for {experiment_id}"
    if why:
        msg += f": {why}"
    return fail("INVALID_EXPERIMENT_SPEC", msg)


def _resolve_folder(root: Path, exp_number: int, slug: str) -> Path:
    """Reuse the existing folder for exp_number, else propose EXP-###_slug."""
    existing = _folders_with_number(root, exp_number)
    return existing[0] if existing else root / f"EXP-{exp_number:03d}_{slug}"


def _check_number(value, experiment_id):
    """Validate an EXP number boundary; raise INVALID_EXPERIMENT_SPEC."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _bad(value, experiment_id)
    if value < FIRST_NUMBER:
        raise _bad(value, experiment_id, f"new experiments start at EXP-{FIRST_NUMBER}")
    return value


def resolve_exp_number(experiments_root, experiment_id, requested):
    """Resolve the operator-supplied EXP number: None/int/digit-string only.

    Any other string (e.g. ``"auto"``) is ``INVALID_EXPERIMENT_SPEC``; there
    is no auto-numbering.
    """
    root = Path(experiments_root)
    req_int = None
    if isinstance(requested, bool) or (
        requested is not None and not isinstance(requested, (int, str))
    ):
        raise _bad(requested, experiment_id)
    if isinstance(requested, int):
        req_int = requested
    elif isinstance(requested, str):
        text = requested.strip()
        if text.isdigit() and len(text) <= 9:
            req_int = int(text)
        else:
            raise _bad(requested, experiment_id)
    if req_int is not None:
        _check_number(req_int, experiment_id)

    linked = _linked_number(root, experiment_id)
    if linked is not None:
        if linked < FIRST_NUMBER:
            raise _bad(
                requested,
                experiment_id,
                f"linked number is below EXP-{FIRST_NUMBER}",
            )
        if req_int is not None and req_int != linked:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"{experiment_id} is linked to EXP-{linked:03d} "
                f"but EXP-{req_int:03d} was requested",
            )
        return linked

    if req_int is None:
        raise fail(
            "INVALID_EXPERIMENT_SPEC",
            f"run has no EXP number: {experiment_id} is not linked under "
            f"{root} and --exp-number was not given",
        )
    try:
        parse_experiment_id(f"EXP-{req_int:03d}")
    except ValueError as exc:
        raise _bad(requested, experiment_id, str(exc))
    for folder in _folders_with_number(root, req_int):
        _check_number_folder(folder, req_int, experiment_id)
    return req_int


def _manifest_bytes(experiment_id, exp_number, recording_mode, trimmed,
                    doc_bytes: dict[str, bytes]) -> bytes:
    """Canonical export_manifest.json bytes over the four proposed doc bytes."""
    return _dump({
        "experiment_id": experiment_id,
        "exp_number": exp_number,
        "recording_mode": recording_mode,
        "trimmed": trimmed,
        "files": {name: hashlib.sha256(raw).hexdigest()
                  for name, raw in doc_bytes.items()},
    })


def _encode_doc(doc) -> tuple[bytes, bool]:
    """Encode doc as JSON bytes, trimming lists over MAX_JSON_BYTES (trim, did_trim)."""
    raw = _dump(doc)
    if len(raw) <= MAX_JSON_BYTES:
        return raw, False
    stub = {
        "trimmed": True,
        "original_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if isinstance(doc, list):
        lo, hi, best = 0, len(doc), None
        while lo <= hi:
            mid = (lo + hi) // 2
            encoded = _dump({**stub, "kept": doc[:mid]})
            if len(encoded) <= MAX_JSON_BYTES:
                best = encoded
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            return best, True
    return _dump(stub), True


def _write_new(path: Path, data: bytes) -> None:
    """Create-only write; parent dir must already exist."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".record_export.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def _cap_report(text):
    """Cap report text to MAX_JSON_BYTES-8192 bytes and mark truncation."""
    encoded = str(text).encode("utf-8")
    limit = MAX_JSON_BYTES - 8192
    capped = encoded[:limit].decode("utf-8", errors="ignore")
    if len(encoded) > limit:
        capped += "\n[report truncated to fit the 1 MB export cap]"
    return capped


def _report_md(exp_number: int, experiment_id, outcome, report_text) -> str:
    """Render the REPORT.md stub (ids, status, optional success report text)."""
    status = str(outcome.get("failure_code") or "ran") if isinstance(outcome, dict) else "ran"
    lines = [
        f"# EXP-{exp_number:03d} {experiment_id}",
        f"Status: {status}",
    ]
    if report_text is not None:
        lines += ["", str(report_text)]
    return "\n".join(lines + ["", "Machine-readable run record: see `results/` in this directory."]) + "\n"


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
    """Create-only export: preflight every conflict, then write only missing files.

    Folder/results symlinks and manifestless folders raise before any write.
    Unreadable/non-regular/oversized existing targets raise before any write.
    An existing manifest must replay: its files hashes must cover the four
    docs and match the persisted bytes, else EXPERIMENT_IDENTITY_CONFLICT.
    Existing targets with different bytes are kept (stderr warning, status
    unchanged); manifest hashes describe the proposed bytes, which equal the
    bytes on disk for every file this call writes.
    """
    root = Path(experiments_root)
    if not isinstance(slug, str):
        raise fail(
            "INVALID_EXPERIMENT_SPEC",
            f"slug must be a string, got {type(slug).__name__}",
        )
    if not isinstance(experiment_id, str):
        raise fail(
            "INVALID_EXPERIMENT_SPEC",
            f"experiment_id must be a string, got {type(experiment_id).__name__}",
        )
    _check_number(exp_number, experiment_id)
    linked = _linked_number(root, experiment_id)
    if linked is not None and linked != exp_number:
        raise fail(
            "EXPERIMENT_IDENTITY_CONFLICT",
            f"{experiment_id} is linked to EXP-{linked:03d} "
            f"but this export targets EXP-{exp_number:03d}",
        )
    if report_text is not None:
        report_text = _cap_report(report_text)
    for folder in _folders_with_number(root, exp_number):
        _check_number_folder(folder, exp_number, experiment_id)
    folder = _resolve_folder(root, exp_number, slugify(slug))
    if folder.is_symlink():
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", f"run folder {folder} is a symlink; refusing to export into it")
    results = folder / "results"
    if results.is_symlink():
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", f"results folder {results} is a symlink; refusing to export into it")

    trimmed: list[str] = []
    docs = {
        "spec.json": spec,
        "request.json": request,
        "event_ids.json": event_ids,
        "outcome.json": outcome,
    }
    proposed: dict[Path, bytes] = {}
    for name, doc in docs.items():
        try:
            raw, was_trimmed = _encode_doc(doc)
        except (TypeError, ValueError) as exc:
            raise fail(
                "INVALID_EXPERIMENT_SPEC",
                f"cannot JSON-encode {name}: {exc}",
            ) from exc
        if was_trimmed and name not in trimmed:
            trimmed.append(name)
        proposed[results / name] = raw
    proposed[folder / "REPORT.md"] = _report_md(
        exp_number, experiment_id, outcome, report_text
    ).encode("utf-8")

    # Preflight: read every existing target, conflict or keep before any write.
    wrote = False
    kept_differing = False
    persisted: dict[Path, bytes] = {}
    for path, data in proposed.items():
        if not os.path.lexists(path):
            persisted[path] = data
            continue
        existing = _read_capped(path)
        if existing is None or len(existing) > MAX_JSON_BYTES:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"existing {path} is unreadable, not a regular file, or oversized",
            )
        if existing != data:
            kept_differing = True
            print(
                f"WARNING: record_export keeping existing {path} (differs from proposed export)",
                file=sys.stderr,
            )
        persisted[path] = existing

    manifest_path = results / "export_manifest.json"
    try:
        manifest_data = _manifest_bytes(
            experiment_id,
            exp_number,
            recording_mode,
            trimmed,
            {name: persisted[results / name] for name in docs},
        )
    except (TypeError, ValueError) as exc:
        raise fail(
            "INVALID_EXPERIMENT_SPEC",
            f"cannot JSON-encode export_manifest.json: {exc}",
        ) from exc
    if not os.path.lexists(manifest_path):
        persisted[manifest_path] = manifest_data
    else:
        existing = _read_capped(manifest_path)
        if existing is None or len(existing) > MAX_JSON_BYTES:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"existing {manifest_path} is unreadable, not a regular file, or oversized",
            )
        try:
            manifest_doc = json.loads(existing)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"existing {manifest_path} is not valid JSON: {exc}",
            ) from exc
        if not isinstance(manifest_doc, dict):
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"existing {manifest_path} is not a JSON object",
            )
        manifest_files = manifest_doc.get("files")
        if not isinstance(manifest_files, dict):
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"existing {manifest_path} has no files object",
            )
        for name in docs:
            if name not in manifest_files:
                raise fail(
                    "EXPERIMENT_IDENTITY_CONFLICT",
                    f"existing {manifest_path} does not list {name}",
                )
            doc_path = results / name
            if not os.path.lexists(doc_path):
                raise fail(
                    "EXPERIMENT_IDENTITY_CONFLICT",
                    f"existing {manifest_path} lists {name} but {doc_path} is missing",
                )
            digest = hashlib.sha256(persisted[doc_path]).hexdigest()
            if digest != manifest_files[name]:
                raise fail(
                    "EXPERIMENT_IDENTITY_CONFLICT",
                    f"replay mismatch for {doc_path}: persisted sha256 {digest} "
                    f"!= manifest {manifest_files[name]!r}",
                )
        if existing != manifest_data:
            kept_differing = True
            print(
                f"WARNING: record_export keeping existing {manifest_path} (differs from proposed export)",
                file=sys.stderr,
            )
        persisted[manifest_path] = existing

    # Write only missing files (folder/results dirs created lazily after preflight).
    results.mkdir(parents=True, exist_ok=True)
    for path, data in persisted.items():
        if not os.path.lexists(path):
            _write_new(path, data)
            wrote = True
    return {
        "dir": str(folder),
        "status": "written" if wrote and not kept_differing else "unchanged",
        "trimmed": trimmed,
    }


def _git_commit_id() -> str | None:
    """Current git HEAD commit id, or None if unavailable."""
    repo_root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
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
    """Export one native run (gate or prediction) under its EXP folder.

    Reads the report text, computes provenance (snapshot, commit, closure hash,
    RSS/elapsed), and delegates to ``export_run_record``.
    """
    document = registration.document
    experiment_id = document["execution_plan"]["experiment_id"]
    snapshot = document.get("snapshot")
    snapshot_id = snapshot.get("snapshot_id") if isinstance(snapshot, dict) else None
    report_text = None
    if saved["outcome"]["failure_code"] is None and report_path is not None:
        report_raw = _read_capped(Path(report_path))
        if report_raw is not None:
            report_text = report_raw.decode("utf-8", errors="ignore")
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