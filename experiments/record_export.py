"""Private run-record export for experiments (issue #1078).

Writes a stable EXP-###_<slug>/ run folder with canonical JSON results and a
REPORT.md stub. The operator assigns the number; no existing bytes are ever
overwritten, oversized docs are trimmed, and every conflict is preflighted
before the first write.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import resource
import stat as stat_mod
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from engine.v2.ops.errors import OpsError, fail
from experiments.lib import FIRST_NUMBER, parse_experiment_id, slugify

MAX_JSON_BYTES = 1_000_000
_NUM_DIR_RE = re.compile(r"^EXP-(\d+)_")


def export_failure_cause(error) -> str:
    """Public cause code for a failed export attempt."""
    if isinstance(error, OpsError):
        return error.code
    if isinstance(error, OSError):
        return "RESOURCE_UNAVAILABLE"
    if isinstance(error, ValueError):
        return "INVALID_EXPERIMENT_SPEC"
    raise error


def public_result(result) -> dict:
    """CLI projection of an export result: only status and trimmed, never dir."""
    return {"status": result["status"], "trimmed": result["trimmed"]}


class _ReadRefused(Exception):
    """A bounded read failed; reason is unreadable or not_regular."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _read_bounded(path, max_bytes: int, *, follow: bool = True) -> bytes:
    """Read at most max_bytes+1 bytes of a regular file; raise _ReadRefused if unreadable/non-regular. Callers detect oversize by length > max_bytes."""
    flags = os.O_RDONLY | os.O_NONBLOCK
    if not follow:
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise _ReadRefused("unreadable") from exc
    try:
        if not stat_mod.S_ISREG(os.fstat(fd).st_mode):
            raise _ReadRefused("not_regular")
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(fd, max_bytes + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise _ReadRefused("unreadable") from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def read_spec_document(path, max_bytes=16 * 1024 * 1024):
    """Read and parse an experiment spec JSON file, capped at max_bytes.

    A missing/unreadable/non-regular file or a file over the cap raises
    INVALID_EXPERIMENT_SPEC; JSON decode errors propagate as today.
    """
    try:
        raw = _read_bounded(path, max_bytes)
    except _ReadRefused as refused:
        message = {
            "unreadable": "cannot read spec file",
            "not_regular": "spec file is not a regular file",
        }[refused.reason]
        # strerror carries the OS reason (ENOENT vs EACCES) and no path.
        detail = getattr(refused.__cause__, "strerror", None)
        if detail:
            message = f"{message}: {detail}"
        raise fail("INVALID_EXPERIMENT_SPEC", message) from refused
    if len(raw) > max_bytes:
        raise fail("INVALID_EXPERIMENT_SPEC", "spec file exceeds 16 MiB")
    return json.loads(raw)


@contextmanager
def _export_lock(root: Path):
    """Serialize export_run_record: one exporter preflights and publishes at a time."""
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(root / ".record_export.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _dump(obj) -> bytes:
    return (
        json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_capped(path: Path, *, follow: bool = False) -> bytes | None:
    """Read at most MAX_JSON_BYTES+1 bytes; None if unreadable/non-regular."""
    try:
        return _read_bounded(path, MAX_JSON_BYTES, follow=follow)
    except _ReadRefused:
        return None


def _read_manifest(path: Path, *, follow: bool = False):
    """Parse a manifest JSON object; None if unreadable/non-regular/oversized/invalid."""
    raw = _read_capped(path, follow=follow)
    if raw is None or len(raw) > MAX_JSON_BYTES:
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


def _exp_folders(root: Path) -> list[tuple[int, Path]]:
    """Every EXP-###_* dir/symlink entry in root as (number, folder), uncollapsed."""
    if not root.is_dir():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if child.is_symlink() or child.is_dir():
            m = _NUM_DIR_RE.match(child.name)
            if m:
                out.append((int(m.group(1)), child))
    return out


def _linked_number(root: Path, experiment_id):
    """Find an EXP folder whose manifest links this experiment_id.

    Iterates every EXP entry itself so duplicate-number folders are not
    collapsed: two folders naming this experiment_id under different numbers
    are EXPERIMENT_IDENTITY_CONFLICT.
    """
    linked = None
    for num, folder in _exp_folders(root):
        man = _manifest_for(folder)
        if man is None:
            if not (folder.is_symlink() or (folder / "results").is_symlink()):
                continue
            man = _read_manifest(
                folder / "results" / "export_manifest.json", follow=True
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
        if linked is not None and linked != num:
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"{experiment_id} is linked to EXP-{linked:03d} and EXP-{num:03d} "
                f"in {folder.name}",
            )
        linked = num
    return linked


def _folders_with_number(root: Path, n: int) -> list[Path]:
    """All EXP-###_ dirs in root with number n."""
    return [folder for num, folder in _exp_folders(root) if num == n]


_PARTIAL_RESULTS_NAMES = {
    "spec.json",
    "request.json",
    "event_ids.json",
    "outcome.json",
    "runs",
}


def _is_partial_export(folder: Path) -> bool:
    """True iff folder holds a recoverable partial export with no manifest.

    Only results/spec.json (or nothing) plus at most the other run-record
    names may be present; any unknown entry makes the folder unadoptable.
    """
    results = folder / "results"
    try:
        if results.is_symlink() or not results.is_dir():
            return False
        if (folder / "results" / "export_manifest.json").exists():
            return False
        for entry in os.scandir(results):
            if entry.name == "runs":
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    return False
            elif entry.name not in _PARTIAL_RESULTS_NAMES - {"runs"}:
                return False
            elif entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                return False
        return True
    except OSError:
        return False


def _check_number_folder(folder: Path, exp_number: int, experiment_id):
    """Reject a symlinked folder/results dir or manifestless folder that is not
    a recoverable partial export, or one linked to a different experiment_id."""
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
        if _is_partial_export(folder):
            return
        raise fail(
            "EXPERIMENT_IDENTITY_CONFLICT",
            f"EXP-{exp_number:03d} in {folder.name} has no export manifest (an unlinked folder, or a crashed or partial export that never wrote its manifest), not linked to {experiment_id!r}",
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
    if value > 999_999_999:
        raise _bad(value, experiment_id, "EXP number above 999999999")
    return value


def _check_experiment_id(experiment_id):
    """Validate an experiment id; raise INVALID_EXPERIMENT_SPEC."""
    if not isinstance(experiment_id, str) or not experiment_id:
        raise fail(
            "INVALID_EXPERIMENT_SPEC",
            f"experiment_id must be a non-empty string, got {experiment_id!r}",
        )


def resolve_exp_number(experiments_root, experiment_id, requested):
    """Resolve the operator-supplied EXP number: None/int/digit-string only.

    Any other string (e.g. ``"auto"``) is ``INVALID_EXPERIMENT_SPEC``; there
    is no auto-numbering.
    """
    _check_experiment_id(experiment_id)
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
        if requested.isascii() and text.isdigit() and len(text) <= 9:
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
        for folder in _folders_with_number(root, linked):
            _check_number_folder(folder, linked, experiment_id)
        return linked

    if req_int is None:
        raise fail(
            "INVALID_EXPERIMENT_SPEC",
            f"run has no EXP number: {experiment_id} is not linked and "
            f"--exp-number was not given",
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


def _is_trim_stub(raw: bytes) -> bool:
    """True iff raw is a JSON object declaring a trim (stub from _encode_doc)."""
    try:
        doc = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return False
    return (
        isinstance(doc, dict)
        and doc.get("trimmed") is True
        and "original_bytes" in doc
        and "sha256" in doc
    )


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

    Folder/results symlinks and manifestless folders that are not recoverable
    partial exports raise before any write; a partial export of the same spec
    bytes is completed (missing files and the manifest are written), while one
    of a different spec is EXPERIMENT_IDENTITY_CONFLICT.
    Unreadable/non-regular/oversized existing targets raise before any write.
    An existing manifest must replay: its files hashes must cover the four
    docs and match the persisted bytes, else EXPERIMENT_IDENTITY_CONFLICT.
    The first run of an experiment_id keeps the flat layout; a different run
    (another arm, or a recorded run after a smoke run) is written under
    results/runs/<key>/ where key is derived from the spec, request and
    recording mode bytes. A replay of the same run keeps the first persisted
    bytes silently (status unchanged); differing existing bytes of the same
    run are kept silently. Manifest hashes describe the persisted bytes on
    disk.
    """
    root = Path(experiments_root)
    with _export_lock(root):
        _check_experiment_id(experiment_id)
        if not isinstance(slug, str):
            raise fail(
                "INVALID_EXPERIMENT_SPEC",
                f"slug must be a string, got {type(slug).__name__}",
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
        top = folder / "results"
        if top.is_symlink():
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", f"results folder {top} is a symlink; refusing to export into it")
        results = top

        if (
            not os.path.lexists(top / "spec.json")
            and not os.path.lexists(top / "export_manifest.json")
            and any(
                os.path.lexists(top / name)
                for name in ("request.json", "event_ids.json", "outcome.json")
            )
        ):
            raise fail(
                "EXPERIMENT_IDENTITY_CONFLICT",
                f"{top} holds run files without spec.json and a manifest; "
                f"refusing to adopt them",
            )

        trimmed: list[str] = []
        docs = {
            "spec.json": spec,
            "request.json": request,
            "event_ids.json": event_ids,
            "outcome.json": outcome,
        }
        encoded: dict[str, bytes] = {}
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
            encoded[name] = raw

        spec_raw = encoded["spec.json"]
        request_raw = encoded["request.json"]
        if os.path.lexists(top / "spec.json"):
            existing_spec = _read_capped(top / "spec.json")
            if existing_spec is None or len(existing_spec) > MAX_JSON_BYTES:
                raise fail(
                    "EXPERIMENT_IDENTITY_CONFLICT",
                    f"existing {top / 'spec.json'} is "
                    f"unreadable, not a regular file, or oversized",
                )
            if os.path.lexists(top / "request.json"):
                existing_request = _read_capped(top / "request.json")
                if existing_request is None or len(existing_request) > MAX_JSON_BYTES:
                    raise fail(
                        "EXPERIMENT_IDENTITY_CONFLICT",
                        f"existing {top / 'request.json'} is "
                        f"unreadable, not a regular file, or oversized",
                    )
            else:
                existing_request = None
            existing_mode = None
            manifest_present = os.path.lexists(top / "export_manifest.json")
            if manifest_present:
                existing_manifest_doc = _read_manifest(top / "export_manifest.json")
                if (
                    existing_manifest_doc is None
                    or not isinstance(existing_manifest_doc.get("files"), dict)
                ):
                    raise fail(
                        "EXPERIMENT_IDENTITY_CONFLICT",
                        f"existing {top / 'export_manifest.json'} is missing, invalid, "
                        f"or has no files object",
                    )
                files = existing_manifest_doc["files"]
                if "request.json" in files and existing_request is None:
                    raise fail(
                        "EXPERIMENT_IDENTITY_CONFLICT",
                        f"{top / 'export_manifest.json'} lists request.json but "
                        f"{top / 'request.json'} is missing",
                    )
                for name, existing_bytes in (
                    ("spec.json", existing_spec),
                    ("request.json", existing_request),
                ):
                    if existing_bytes is None:
                        continue
                    digest = hashlib.sha256(existing_bytes).hexdigest()
                    if digest != files.get(name):
                        raise fail(
                            "EXPERIMENT_IDENTITY_CONFLICT",
                            f"replay mismatch for {top / name}: persisted sha256 "
                            f"{digest} != manifest {files.get(name)!r}",
                        )
                existing_mode = existing_manifest_doc.get("recording_mode")
            else:
                if existing_spec != spec_raw:
                    raise fail(
                        "EXPERIMENT_IDENTITY_CONFLICT",
                        f"{top} holds a partial export of a different spec; refusing to adopt it",
                    )
                for name, this_raw in (
                    ("request.json", request_raw),
                    ("event_ids.json", encoded["event_ids.json"]),
                ):
                    if not os.path.lexists(top / name):
                        continue
                    existing_raw = _read_capped(top / name)
                    if (
                        existing_raw is None
                        or len(existing_raw) > MAX_JSON_BYTES
                        or existing_raw != this_raw
                    ):
                        raise fail(
                            "EXPERIMENT_IDENTITY_CONFLICT",
                            f"{top} holds a partial export whose {name} differs "
                            f"from this run; refusing to adopt it",
                        )
                # Same run resumed: keep the persisted spec/request/event_ids
                # bytes. If an existing outcome.json carries a recording_mode,
                # keep that mode so a differing mode routes this run to
                # results/runs/<key>/ and leaves the partial folder untouched;
                # otherwise adopt this run's recording mode so the writes
                # below fill the missing files and the manifest.
                existing_mode = recording_mode
                outcome_path = top / "outcome.json"
                if os.path.lexists(outcome_path):
                    outcome_raw = _read_capped(outcome_path)
                    if outcome_raw is not None and len(outcome_raw) <= MAX_JSON_BYTES:
                        try:
                            outcome_doc = json.loads(outcome_raw)
                        except ValueError:
                            outcome_doc = None
                        if (
                            isinstance(outcome_doc, dict)
                            and "recording_mode" in outcome_doc
                        ):
                            existing_mode = outcome_doc["recording_mode"]
            if (
                existing_spec != spec_raw
                or (existing_request is not None and existing_request != request_raw)
                or existing_mode != recording_mode
            ):
                key = hashlib.sha256(
                    spec_raw + b"\0" + request_raw + b"\0"
                    + (recording_mode or "").encode("utf-8")
                ).hexdigest()[:12]
                runs_dir = top / "runs"
                if runs_dir.is_symlink():
                    raise fail(
                        "EXPERIMENT_IDENTITY_CONFLICT",
                        f"runs directory {runs_dir} is a symlink; refusing to "
                        f"export into it",
                    )
                results = runs_dir / key

        proposed: dict[Path, bytes] = {
            (results / name): raw for name, raw in encoded.items()
        }
        report_path = folder / "REPORT.md" if results == top else results / "REPORT.md"
        proposed[report_path] = _report_md(
            exp_number, experiment_id, outcome, report_text
        ).encode("utf-8")

        # Preflight: read every existing target, conflict or keep before any write.
        wrote = False
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
            persisted[path] = existing

        manifest_path = results / "export_manifest.json"
        trimmed = [name for name in docs
                   if _is_trim_stub(persisted[results / name])]
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
            persisted[manifest_path] = existing

        # The returned trimmed list describes the PERSISTED record: an existing
        # manifest read during preflight supplies it; otherwise use the newly
        # computed list for a manifest this call is about to write.
        returned_trimmed = trimmed
        if os.path.lexists(manifest_path):
            persisted_manifest = _read_manifest(manifest_path)
            if persisted_manifest is not None and isinstance(
                persisted_manifest.get("trimmed"), list
            ):
                returned_trimmed = persisted_manifest["trimmed"]

        # Write only missing files (folder/results dirs created lazily after preflight).
        results.mkdir(parents=True, exist_ok=True)
        for path, data in persisted.items():
            if not os.path.lexists(path):
                _write_new(path, data)
                wrote = True
        return {
            "dir": str(folder),
            "status": "written" if wrote else "unchanged",
            "trimmed": returned_trimmed,
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
        if report_raw is None:
            raise fail("RESOURCE_UNAVAILABLE", "supplied report could not be read")
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
            "magnitude_request": document.get("magnitude_request"),
            "magnitude_execution": document.get("magnitude_execution"),
        },
        report_text=report_text,
        recording_mode=saved["receipt"].get("recording_mode"),
    )


def export_or_skip(experiments_root, registration, spec_document, request_dict,
                   saved, exp_number, replayed, started, report_path):
    """CLI helper: export the run record unless no EXP number was given.

    Returns (record, refusal): record is "skipped" or the public result;
    refusal is None or the JSON-able refusal dict the CLI should print (exit 2).
    """
    if exp_number is None:
        return "skipped", None
    try:
        return public_result(export_native_run(
            experiments_root, registration, spec_document, request_dict,
            saved, exp_number, replayed, started, report_path=report_path)), None
    except (OpsError, OSError, ValueError) as error:
        return None, {"refused": "record_export_failed",
                      "cause": export_failure_cause(error)}