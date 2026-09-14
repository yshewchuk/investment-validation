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
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from engine.v2.foundation import format_timestamp
from engine.v2.ledger.decisions import DecisionConflict, import_lines, set_authority
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.recovery import SupervisorLock

__all__ = ["import_history"]

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


def _read_lines(path):
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        raise fail("VALIDATION_FAILED", "legacy ledger file is not valid UTF-8 text",
                   details={"file": path.name}) from None
    return [line.strip() for line in text.splitlines() if line.strip()]


def _included_prefix(lines, date_field, through):
    """The longest leading run of ``lines`` whose own date field is on or
    before ``through``. Ledger files are append-only and written in
    chronological order (``engine/ledger.py`` ``write_predictions``/
    ``_write_outcomes``), so a prefix is exact; stopping at the first
    later-dated row is the conservative failure mode if that ever stops
    holding -- it can only exclude more than a strict per-row filter would,
    never less, which keeps the "never imports rows the nightly under test
    is about to commit itself" guarantee. Keeping a true prefix (rather than
    filtering rows out of the middle) also keeps each surviving line's
    1-based position equal to its real line number in the file, which is
    what ``decision_imports``' ``(source_hash, line_number)`` idempotency
    key depends on across repeated/differently-bounded runs.
    """
    if through is None:
        return list(lines)
    included = []
    for raw in lines:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise fail("VALIDATION_FAILED", "legacy ledger line is not valid JSON") from None
        if _row_date(date_field, payload) > through:
            break
        included.append(raw)
    return included


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


def import_history(conn, root, source_root, *, through=None, dry_run=False, clock):
    """Import legacy ledger history into ``root``'s catalog.

    ``root`` is the operations root (holding ``supervisor.lock``, acquired
    here exactly as ``ops serve`` acquires it -- a running supervisor makes
    this refuse ``RESOURCE_UNAVAILABLE`` rather than race it for the catalog).
    ``source_root`` is the legacy checkout to read ``ledger/predictions`` and
    ``ledger/outcomes`` from, read-only. ``through`` (a ``datetime.date``,
    optional) excludes any row dated after it. ``dry_run`` writes nothing
    (every file's transaction rolls back) but reports the same counts.

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
        return _run(conn, source_root, through=through, dry_run=dry_run, clock=clock)
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
    """Idempotent, matching ``tests/test_v2_ops_scope_separation.py``'s
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
                     committed_owner, totals, divergent_ids, summary):
    """Import one legacy ledger file's leading in-scope prefix, mutating
    ``totals`` (this family's running counters, already installed in
    ``summary``) and ``divergent_ids`` (this family's distinct diverged
    row_ids, a set) in place. Returns ``(committed_owner, low, high)`` -- the
    (possibly advanced) committed decision-authority owner and this file's
    own date span (``(None, None)`` when nothing was in scope).

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
    included = _included_prefix(_read_lines(path), date_field, through)
    totals["lines"] += len(included)
    low, high = _date_span(included, date_field)
    if not included:
        return committed_owner, low, high
    source_hash = file_hash(path)
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
                        on_conflict="diverge", provenance_label=path.name)
        except DecisionConflict as exc:
            totals["conflicts"].append("IDEMPOTENCY_CONFLICT")
            raise fail("IDEMPOTENCY_CONFLICT",
                      "legacy history conflicts with already-committed catalog content",
                      details={"file": path.name, "family": family,
                               "partial_summary": summary}) from exc
        new_divergences = _new_divergences(conn, divergences_before)
    new_line_count = sum(1 for i in range(1, len(included) + 1) if i not in existing_before)
    totals["divergences"] += len(new_divergences)
    divergent_ids.update(row[1] for row in new_divergences)
    totals["divergent_row_ids"] = len(divergent_ids)
    totals["imported"] += new_line_count - len(new_divergences)
    totals["already_present"] += len(included) - new_line_count
    return committed_owner, low, high


def _run(conn, source_root, *, through, dry_run, clock):
    summary = {"schema_version": "ledger_history_import.v1.0", "dry_run": dry_run,
              "through": through.isoformat() if through else None, "families": {}}
    committed_owner_row = conn.execute(
        "SELECT owner FROM decision_authority WHERE singleton=1").fetchone()
    committed_owner = committed_owner_row[0] if committed_owner_row else None
    seen_min, seen_max = None, None
    for family, subdir, kind, date_field in _FAMILIES:
        directory = source_root / subdir
        files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
        totals = {"files": 0, "lines": 0, "imported": 0, "already_present": 0, "conflicts": [],
                  "divergences": 0, "divergent_row_ids": 0}
        summary["families"][family] = totals
        divergent_ids = set()
        for path in files:
            committed_owner, low, high = _import_one_file(
                conn, path, family, kind, date_field, through=through, dry_run=dry_run,
                clock=clock, committed_owner=committed_owner, totals=totals,
                divergent_ids=divergent_ids, summary=summary)
            seen_min = _merge_bound(seen_min, low, keep_lower=True)
            seen_max = _merge_bound(seen_max, high, keep_lower=False)
    summary["date_range"] = {"min": seen_min.isoformat() if seen_min else None,
                             "max": seen_max.isoformat() if seen_max else None}
    return summary
