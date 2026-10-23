"""Bootstrap a fresh shadow catalog with legacy prediction/outcome history.

A brand-new catalog's ``decisions``/``decision_imports`` tables start empty:
the nightly only ever commits *that night's* predictions
(``decision_commit.commit_decisions_in_transaction``), so every trade entered
before the catalog existed has nothing for ``legacy_settlement``
(``decision_commit._settlement_line``) to settle against — it looks up
``decisions.decision_id = "prediction:" + row_id`` and fails
``VALIDATION_FAILED`` when that row was never committed.

This module reads the legacy JSONL ledger (``ledger/predictions/*.jsonl``,
``ledger/outcomes/*.jsonl`` — gitignored, declared as ``ledger_glob`` families
in ``engine.v2.data.legacy_nightly_read_plan``) read-only and imports it
through :func:`engine.v2.ledger.decisions.import_lines`, the same exact-bytes,
idempotent-per-``(source_hash, line_number)`` importer the real nightly's
``legacy_settlement`` action already uses for captured settlement candidates.

``decision_id`` consistency (checked, not assumed): :func:`engine.v2.ledger.
decisions._import_decision_id` builds ``"prediction:" + row_id`` for
``kind="prediction"`` — byte-for-byte what ``_settlement_line`` looks up and
what ``decision_commit._commit_row_or_diverge`` computes for a live nightly
commit. No fix was needed; a probe in this module's tests pins the identity.

2026-09-15: every freshly-imported ``kind="outcome"`` row is now stamped
with ``generation_ref`` -- the UTC date of its own ``resolved_at``, via
:func:`engine.v2.ledger.decisions.import_lines`'s ``derive_generation_ref``
(``engine.v2.ledger.decisions.outcome_generation_ref``, the same rule
``engine.v2.ops.session_backfill`` uses to recover it for rows this importer
committed before this date). Without it, this importer would keep minting
new ``generation_ref IS NULL`` rows forever, which can never dedupe a
same-session settlement rerun.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import date
from pathlib import Path

from engine.v2.foundation import format_timestamp
from engine.v2.ledger.decisions import DecisionConflict, import_lines, set_authority
from engine.v2.ops.errors import fail
from engine.v2.ops.recovery import SupervisorLock

__all__ = ["import_history", "discover_history_files"]

#: (family name, directory under source_root, decisions.kind, row date field).
#: The date field is read from the row itself, never inferred from the
#: filename: predictions are dated by ``as_of`` (``engine/ledger.py``
#: ``build_prediction_rows`` -- the file itself is also named by ``as_of``,
#: ``_date_file``), outcomes by ``resolved_at`` (``engine/ledger.py``
#: ``score_outcomes`` -- the field ``_import_decision_id`` already keys its
#: own outcome observation identity on).
_FAMILIES = (
    ("predictions", "ledger/predictions", "prediction", "as_of"),
    ("outcomes", "ledger/outcomes", "outcome", "resolved_at"),
)


def _row_date(date_field, payload):
    value = payload.get(date_field)
    if not value:
        raise fail("VALIDATION_FAILED", "legacy row is missing its date field",
                   details={"field": date_field})
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        raise fail("VALIDATION_FAILED", "legacy row date field is not a valid date",
                   details={"field": date_field}) from None


def _read_pinned_bytes(source_root, path):
    path = Path(path)
    try:
        root = Path(source_root).resolve(strict=True)
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            opened = Path(os.readlink(f"/proc/self/fd/{stream.fileno()}")).resolve(strict=True)
            if not opened.is_relative_to(root):
                raise fail("INPUT_CHANGED", "pinned legacy ledger file escapes source root",
                           details={"file": path.name})
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise fail("INPUT_CHANGED", "pinned legacy ledger file is not regular",
                           details={"file": path.name})
            return stream.read()
    except OSError:
        raise fail("INPUT_CHANGED", "pinned legacy ledger file is unavailable",
                   details={"file": path.name}) from None


def _read_lines(path, *, expected_sha256=None, source_root=None):
    if expected_sha256 is None:
        data = path.read_bytes()
    else:
        data = _read_pinned_bytes(source_root, path)
    content_sha256 = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and content_sha256 != expected_sha256:
        raise fail("INPUT_CHANGED", "pinned legacy ledger file changed",
                   details={"file": path.name})
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise fail("VALIDATION_FAILED", "legacy ledger file is not valid UTF-8 text",
                   details={"file": path.name}) from None
    source_hash = f"sha256:{content_sha256}"
    return [line.strip() for line in text.splitlines() if line.strip()], source_hash


def _included_prefix(lines, date_field, through):
    """The longest leading run of ``lines`` whose own date field is on or
    before ``through``, plus the count of those rows dated exactly ``through``.
    Ledger files are append-only and written in chronological order
    (``engine/ledger.py`` ``write_predictions``/``_write_outcomes``), so a
    prefix is exact; stopping at the first later-dated row is the conservative
    failure mode if that ever stops holding -- it can only exclude more than a
    strict per-row filter would, never less, which keeps the "never imports
    rows the nightly under test is about to commit itself" guarantee. Keeping
    a true prefix (rather than filtering rows out of the middle) also keeps
    each surviving line's 1-based position equal to its real line number in
    the file, which is what ``decision_imports``' ``(source_hash,
    line_number)`` idempotency key depends on across repeated/differently-
    bounded runs. Exact-date coverage counts only included rows, so it never
    claims a later-dated row that the prefix deliberately excludes.
    """
    if through is None:
        return list(lines), 0
    included = []
    exact_rows = 0
    for raw in lines:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise fail("VALIDATION_FAILED", "legacy ledger line is not valid JSON") from None
        row_date = _row_date(date_field, payload)
        if row_date > through:
            break
        if row_date == through:
            exact_rows += 1
        included.append(raw)
    return included, exact_rows


def _file_transaction(conn, *, dry_run):
    """One immediate transaction per file; on ``dry_run`` it always rolls
    back (even on success), matching ``engine.v2.ops.catalog.transaction``'s
    shape otherwise -- reused instead of imported because that helper always
    commits on success and cannot be told to discard a clean run."""
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        if conn.in_transaction:
            raise RuntimeError("nested catalog transaction")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("ROLLBACK" if dry_run else "COMMIT")

    return _scope()


def import_history(conn, root, source_root, *, through=None, dry_run=False, clock,
                   pinned_files=None):
    """Import legacy ledger history into ``root``'s catalog.

    ``root`` is the operations root (holding ``supervisor.lock``, acquired
    here exactly as ``ops serve`` acquires it -- a running supervisor makes
    this refuse ``RESOURCE_UNAVAILABLE`` rather than race it for the catalog).
    ``source_root`` is the legacy checkout to read ``ledger/predictions`` and
    ``ledger/outcomes`` from, read-only. ``through`` (a ``datetime.date``,
    optional) excludes any row dated after it. ``dry_run`` writes nothing
    (every file's transaction rolls back) but reports the same counts.
    ``pinned_files=None`` keeps discovery behavior; a supplied sequence must contain
    every discovered relative JSONL path and its raw lowercase SHA-256 digest.
    An empty sequence explicitly pins an empty file set.

    Returns a JSON-safe summary dict; raises a typed :class:`OpsError`
    (``engine.v2.ops.errors.fail``) refusal on a bad ``source_root``, a lost
    catalog race, or a conflicting legacy duplicate.
    """
    source_root = Path(source_root)
    if not (source_root / "ledger").is_dir():
        raise fail("SOURCE_NOT_FOUND", "source root has no ledger directory",
                   details={"source_root": str(source_root)})
    lock = SupervisorLock(Path(root) / "supervisor.lock")
    if not lock.acquire():
        raise fail("RESOURCE_UNAVAILABLE", "a running supervisor holds this catalog")
    try:
        return _run(conn, source_root, through=through, dry_run=dry_run, clock=clock,
                    pinned_files=pinned_files)
    finally:
        lock.release()


def _date_span(included, date_field):
    """The (min, max) row date across an already-filtered line list, or
    ``(None, None)`` for an empty one."""
    dates = [_row_date(date_field, json.loads(raw)) for raw in included]
    return (min(dates), max(dates)) if dates else (None, None)


def _merge_bound(current, candidate, *, keep_lower):
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if (candidate < current) == keep_lower else current


def _set_authority_once(conn, committed_owner, stamp, *, dry_run):
    """Idempotent, matching ``tests/v2/ops/test_v2_ops_scope_separation.py``'s
    production pattern: read the durably committed owner once (by the
    caller) and only call ``set_authority`` when it is not already
    ``"catalog"`` -- calling it every file would otherwise bump
    ``decision_authority.generation`` on every commit for no reason. Under
    ``dry_run`` the caller's own transaction always rolls back, so the
    "committed" tracking variable must NOT advance -- the next file (or the
    next invocation) must see the same unset state again.
    """
    if committed_owner == "catalog":
        return committed_owner
    set_authority(conn, committed_owner, "catalog", stamp)
    return committed_owner if dry_run else "catalog"


def _new_divergences(conn, before_ids):
    """Divergence rows recorded by THIS call, still inside its (possibly
    about-to-roll-back) transaction -- read here, before the caller's
    ``with`` block exits, so a ``dry_run`` still sees them before they are
    discarded."""
    after = conn.execute(
        "SELECT divergence_id, occurrence FROM decision_divergences "
        "WHERE scope='legacy_import'").fetchall()
    return [row for row in after if row[0] not in before_ids]


def _import_one_file(conn, path, family, kind, date_field, *, through, dry_run, clock,
                     committed_owner, totals, divergent_ids, summary, source_root=None,
                     expected_sha256=None):
    """Import one legacy ledger file's leading in-scope prefix, mutating
    ``totals`` (this family's running counters, already installed in
    ``summary``) and ``divergent_ids`` (this family's distinct diverged
    row_ids, a set) in place. Returns ``(committed_owner, low, high,
    exact_rows)`` -- the (possibly advanced) committed decision-authority
    owner, this file's own date span (``(None, None)`` when nothing was in
    scope) and its count of in-scope rows dated exactly on ``through``.

    The user's standing 2026-09-14 decision applies here too: a row_id that
    repeats within the legacy ledger with a DIFFERING payload never blocks
    the import and never overwrites the first-imported occurrence --
    ``import_lines(..., on_conflict="diverge")`` keeps the first content
    authoritative and records every later, differing occurrence as a
    ``decision_divergences`` row (``engine.v2.ledger.decisions.
    _record_legacy_divergence``). A byte-identical duplicate (same content,
    any position) still stays a plain no-op, and a changed byte on a line
    already imported under this exact ``(source_hash, line_number)`` -- a
    provenance conflict, not a legacy duplicate -- still refuses typed.
    """
    totals["files"] += 1
    lines, source_hash = _read_lines(
        path, expected_sha256=expected_sha256, source_root=source_root)
    included, exact_rows = _included_prefix(lines, date_field, through)
    totals["lines"] += len(included)
    low, high = _date_span(included, date_field)
    if not included:
        return committed_owner, low, high, exact_rows
    stamp = format_timestamp(clock.now())
    existing_before = {row[0] for row in conn.execute(
        "SELECT line_number FROM decision_imports WHERE source_hash=?",
        (source_hash,)).fetchall()}
    divergences_before = {row[0] for row in conn.execute(
        "SELECT divergence_id FROM decision_divergences WHERE scope='legacy_import'").fetchall()}
    with _file_transaction(conn, dry_run=dry_run):
        committed_owner = _set_authority_once(conn, committed_owner, stamp, dry_run=dry_run)
        try:
            import_lines(conn, source_hash, included, kind=kind, created_at=stamp,
                        on_conflict="diverge", provenance_label=path.name,
                        derive_generation_ref=True)
        except DecisionConflict as exc:
            totals["conflicts"].append("IDEMPOTENCY_CONFLICT")
            raise fail("IDEMPOTENCY_CONFLICT",
                      "legacy history conflicts with already-committed catalog content",
                      details={"file": path.name, "family": family,
                               "partial_summary": summary}) from exc
        new_divergences = _new_divergences(conn, divergences_before)
        # Count actual provenance writes before a dry-run transaction rolls back.
        # Identical content committed under another purpose adds no provenance.
        new_line_count = conn.execute(
            "SELECT COUNT(*) FROM decision_imports WHERE source_hash=?",
            (source_hash,)).fetchone()[0] - len(existing_before)
    totals["divergences"] += len(new_divergences)
    divergent_ids.update(row[1] for row in new_divergences)
    totals["divergent_row_ids"] = len(divergent_ids)
    totals["imported"] += new_line_count - len(new_divergences)
    totals["already_present"] += len(included) - new_line_count
    return committed_owner, low, high, exact_rows


def discover_history_files(source_root):
    if ((source_root / "ledger").is_symlink()
            or any((source_root / subdir).is_symlink()
                   for _, subdir, _, _ in _FAMILIES)):
        raise fail("INPUT_CHANGED", "ledger history directory is a symlink")
    discovered = {path.relative_to(source_root).as_posix(): path
                  for _, subdir, _, _ in _FAMILIES
                  for path in (source_root / subdir).glob("*.jsonl")}
    if any(path.is_symlink() or not path.is_file() for path in discovered.values()):
        raise fail("INPUT_CHANGED", "ledger history file is not a regular file")
    return discovered


def _validate_history_pins(pinned_files, discovered):
    if not isinstance(pinned_files, (tuple, list)):
        raise fail("INVALID_REQUEST", "pinned_files must be a sequence of pairs")
    pins = {}
    for pair in pinned_files:
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2
                and isinstance(pair[0], str) and isinstance(pair[1], str)):
            raise fail("INVALID_REQUEST", "pinned_files must be pairs of (path, sha256)")
        rel, digest = pair
        if rel in pins:
            raise fail("INVALID_REQUEST", "duplicate pinned file path")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise fail("INVALID_REQUEST", "pinned file hash must be 64 lowercase hex")
        pins[rel] = digest
    if set(pins) != set(discovered):
        raise fail("INPUT_CHANGED", "pinned legacy ledger file set differs from source",
                   details={"missing": sorted(set(pins) - set(discovered)),
                            "added": sorted(set(discovered) - set(pins))})
    return pins


def _validate_pinned_history_files(source_root, pinned_files):
    discovered = discover_history_files(source_root)
    pins = _validate_history_pins(pinned_files, discovered)
    for relative, digest in pins.items():
        data = _read_pinned_bytes(source_root, discovered[relative])
        if hashlib.sha256(data).hexdigest() != digest:
            raise fail("INPUT_CHANGED", "pinned legacy ledger file changed",
                       details={"file": discovered[relative].name})


def _history_files_by_family(source_root, pinned_files):
    groups = {family: [] for family, _, _, _ in _FAMILIES}
    if pinned_files is None:
        for family, subdir, _, _ in _FAMILIES:
            directory = source_root / subdir
            groups[family] = [(path, None) for path in sorted(directory.glob("*.jsonl"))] \
                if directory.is_dir() else []
        return groups
    discovered = discover_history_files(source_root)
    for rel, digest in _validate_history_pins(pinned_files, discovered).items():
        path = discovered[rel]
        family = next(f for f, subdir, _, _ in _FAMILIES
                      if rel.startswith(subdir + "/"))
        groups[family].append((path, digest))
    for files in groups.values():
        files.sort(key=lambda item: item[0].name)
    return groups


def _run(conn, source_root, *, through, dry_run, clock, pinned_files=None):
    summary = {"schema_version": "ledger_history_import.v1.1", "dry_run": dry_run,
               "through": through.isoformat() if through else None, "families": {}}
    committed_owner_row = conn.execute(
        "SELECT owner FROM decision_authority WHERE singleton=1").fetchone()
    committed_owner = committed_owner_row[0] if committed_owner_row else None
    seen_min, seen_max = None, None
    coverage_families = {}
    by_family = _history_files_by_family(source_root, pinned_files)
    for family, subdir, kind, date_field in _FAMILIES:
        files = by_family[family]
        totals = {"files": 0, "lines": 0, "imported": 0, "already_present": 0, "conflicts": [],
                  "divergences": 0, "divergent_row_ids": 0}
        summary["families"][family] = totals
        divergent_ids = set()
        exact_rows = 0
        for path, expected_sha256 in files:
            committed_owner, low, high, file_exact_rows = _import_one_file(
                conn, path, family, kind, date_field, through=through, dry_run=dry_run,
                clock=clock, committed_owner=committed_owner, totals=totals,
                divergent_ids=divergent_ids, summary=summary, source_root=source_root,
                expected_sha256=expected_sha256)
            exact_rows += file_exact_rows
            seen_min = _merge_bound(seen_min, low, keep_lower=True)
            seen_max = _merge_bound(seen_max, high, keep_lower=False)
        coverage_families[family] = {
            "rows": exact_rows, "status": "present" if exact_rows else "no_rows"}
    if pinned_files is not None:
        _validate_pinned_history_files(source_root, pinned_files)
    summary["date_range"] = {"min": seen_min.isoformat() if seen_min else None,
                             "max": seen_max.isoformat() if seen_max else None}
    summary["coverage"] = (
        {"through": through.isoformat(), "families": coverage_families}
        if through is not None else None)
    return summary
