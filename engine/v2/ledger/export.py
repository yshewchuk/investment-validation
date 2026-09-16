"""Complete, sequence-ordered compatibility exports for legacy readers."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

from engine.v2.foundation import canonical_json, content_hash, fsync_directory


def _no_fault(_name):
    return None


def export_generation(conn, root: Path | str, *, generation: str | None = None, purposes=None,
                      fault=None) -> Path:
    """Write all prediction rows through a durable generation and switch CURRENT atomically.

    ``purposes``, when given, restricts the export to ``decisions`` rows whose
    ``purpose`` column is in that set (P2-5/D20: only ``legacy_import`` and
    ``shadow`` ever belong in a legacy-compatible export; a
    ``research_reconstruction`` row is never a legacy row and must never
    appear). Left ``None``, every row is exported — the pre-existing,
    unfiltered behaviour every caller before D20 still relies on.

    Omit ``generation`` to identify the exact exported file names and bytes.
    The identity and files come from the same captured catalog rows: catalog
    growth produces a new immutable directory even when decision receipts
    are unchanged. Explicit names retain their strict verification contract.

    Resumable (P2-C06): every file is written into a private sibling
    ``root/.<generation>.partial-<attempt>/`` first, fsynced file-by-file and
    directory-by-directory, then atomically ``os.rename``d onto
    ``root/<generation>`` -- only THEN does ``CURRENT`` ever move. A crash or
    injected fault partway through a partial directory never touches
    ``root/<generation>`` or ``CURRENT`` at all: the old ``CURRENT`` (if any)
    stays exactly as it was, still readable by legacy readers, and a retry
    starts a brand-new partial directory (a fresh ``uuid4`` attempt id) that
    the stale one never blocks. ``root/<generation>`` itself, once it exists,
    is a complete, previously-renamed generation and is never written into
    again -- only verified byte-for-byte against the catalog (the pre-D06
    behaviour, preserved for the already-durable case) or left as-is.
    """
    if generation is not None and ("/" in generation or generation in ("", ".", "..")):
        raise ValueError("unsafe export generation")
    grouped = _grouped_lines(conn, purposes)
    if generation is None:
        generation = content_hash([
            "ledger_export_files.v1",
            [[bucket + "/" + date + ".jsonl", hashlib.sha256(b"".join(lines)).hexdigest()]
             for (bucket, date), lines in sorted(grouped.items())],
        ])
    fault = fault or _no_fault
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / generation
    if destination.exists():
        _verify_complete(destination, grouped)
    else:
        _write_generation(root, destination, generation, grouped, fault)
    _switch_current(root, generation)
    return destination


def _grouped_lines(conn, purposes):
    if purposes is None:
        rows = conn.execute("SELECT kind,payload_json FROM decisions ORDER BY sequence").fetchall()
    else:
        placeholders = ",".join("?" for _ in purposes)
        rows = conn.execute(
            "SELECT kind,payload_json FROM decisions WHERE purpose IN (" + placeholders + ") "
            "ORDER BY sequence", tuple(purposes)).fetchall()
    grouped = defaultdict(list)
    for row in rows:
        payload = json.loads(row[1])
        date = _partition_date(payload)
        date = str(date)[:10] if date else "unknown"
        bucket = "predictions" if row[0] == "prediction" else "outcomes"
        grouped[(bucket, date)].append(canonical_json(payload).encode("utf-8") + b"\n")
    return grouped


def _verify_complete(destination, grouped):
    expected = {bucket for bucket, _ in grouped}
    expected.update(bucket + "/" + date + ".jsonl" for bucket, date in grouped)
    actual = set()
    for path in destination.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("export generation differs from catalog")
        actual.add(path.relative_to(destination).as_posix())
    if destination.is_symlink() or not destination.is_dir() or actual != expected:
        raise ValueError("export generation differs from catalog")
    for (bucket, date), lines in grouped.items():
        path = destination / bucket / (date + ".jsonl")
        if not path.is_file() or path.read_bytes() != b"".join(lines):
            raise ValueError("export generation differs from catalog")


def _write_generation(root, destination, generation, grouped, fault):
    partial = root / (".{}.partial-{}".format(generation, uuid.uuid4().hex))
    partial.mkdir()
    for (bucket, date), lines in sorted(grouped.items()):
        name = bucket + "/" + date + ".jsonl"
        path = partial / bucket / (date + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(lines))
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        fault(name)
    for directory in sorted({p.parent for p in partial.rglob("*") if p.is_file()}, reverse=True):
        fsync_directory(directory)
    fsync_directory(partial)
    # Never renamed onto ``destination`` until every file above is durable:
    # a fault raised mid-loop leaves this whole ``partial`` directory
    # orphaned beside ``root/<generation>`` (which was never created) and
    # ``CURRENT`` (never touched) -- exactly the resumable-crash state.
    os.rename(partial, destination)
    fsync_directory(root)
    _sweep_stale_partials(root, generation, destination)


def _sweep_stale_partials(root, generation, destination):
    prefix = "." + generation + ".partial-"
    for child in root.iterdir():
        if child.name.startswith(prefix) and child != destination:
            shutil.rmtree(child, ignore_errors=True)
    fsync_directory(root)


def _switch_current(root, generation):
    pointer = root / ("CURRENT." + generation)
    pointer.write_text(generation + "\n")
    with pointer.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pointer, root / "CURRENT")
    fsync_directory(root)


def _partition_date(payload):
    return (payload.get("as_of") or payload.get("resolved_at")
            or payload.get("settled_at") or payload.get("event_date"))
