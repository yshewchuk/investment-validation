"""Current operational health is independent of an immutable board release."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date, timedelta
from pathlib import Path

from engine.v2.foundation import format_timestamp, fsync_directory
from engine.v2.ops.catalog import dumps, transaction
from engine.v2.ops.errors import OpsError, fail


def _record_check(conn, occurrence, kind, ok, receipt):
    conn.execute(
        "INSERT INTO health_observations(occurrence,kind,ok,receipt_json,attempts) VALUES (?,?,?,?,0) "
        "ON CONFLICT(occurrence,kind) DO UPDATE SET ok=excluded.ok, receipt_json=excluded.receipt_json, "
        "attempts=health_observations.attempts+1",
        (occurrence, kind, None if ok is None else int(ok), dumps(receipt)))


def record_check(conn, occurrence, kind, ok, receipt):
    """Record one observation for ``(occurrence, kind)``; a second call for
    the SAME pair (a retry, or a later generation observing the same
    scheduled night -- guide §5.5 item 2) updates that occurrence's latest
    ``ok``/``receipt`` and bumps its own ``attempts`` counter, never adding a
    second row -- the PRIMARY KEY(occurrence, kind) already in place does the
    collapsing; this function only has to keep a retry count as it does so.

    Callable both standalone (as every pre-existing caller/test does -- opens
    its own transaction) and from inside an already-open effect transaction
    (``engineering_gate_effect``, new in guide §5.5 item 2): the one function
    in this module with both calling conventions, since it predates being
    called from inside an effect and existing callers must not have to
    change.
    """
    if conn.in_transaction:
        _record_check(conn, occurrence, kind, ok, receipt)
        return
    with transaction(conn):
        _record_check(conn, occurrence, kind, ok, receipt)


def trailing_occurrences(session, *, nights=14):
    """The last ``nights`` SCHEDULED TRADING SESSIONS ending at (and
    including) ``session``, oldest first -- the engineering history window
    guide §5.5 item 2 asks for.

    2026-09-14 review fix: an earlier version of this function used calendar
    days, so every weekend and every US market holiday showed up as an
    ``unknown`` night in the window -- a night nobody was ever scheduled to
    observe is not the same "unknown" as one that was scheduled and simply
    never got a check, and padding the window with them made
    ``consecutive_nights`` / the visible pass-fail ratio meaningless.
    ``engine.v2.ops.legacy_adapter.projected_trading_sessions`` (pure
    weekday/US-market-holiday rule, capture-inputs, no ``engine.paths``
    dependency) is the audited source of "what night was actually
    scheduled" this module can call without a repo root. The buffer below
    (``nights`` trading sessions need at most ~``nights * 7 / 5`` calendar
    days, plus a handful of holidays) is generous enough that a second,
    wider attempt is never needed in practice; if the calendar still cannot
    produce enough sessions (or resolving it raises at all -- a stale/broken
    ``engine.calendar`` import), this REFUSES rather than silently falling
    back to calendar days, per the guide's own instruction.
    """
    from engine.v2.ops.legacy_adapter import projected_trading_sessions

    end = date.fromisoformat(str(session)[:10])
    start = end - timedelta(days=nights * 2 + 15)
    try:
        sessions = [d for d in projected_trading_sessions(start, end) if d <= end.isoformat()]
    except Exception as exc:
        raise fail("VALIDATION_FAILED", "trading-session calendar could not be resolved for the "
                  "engineering history window", details={"session": str(session)}) from exc
    if len(sessions) < nights:
        raise fail("VALIDATION_FAILED", "not enough trading-session history to build the "
                  "engineering history window", details={"session": str(session), "found": len(sessions)})
    return tuple(sessions[-nights:])


def engineering_history(conn, occurrences):
    """Per-night engineering status over an explicit window of scheduled
    occurrences (guide §5.5 item 2): every occurrence in ``occurrences``
    appears exactly once, in the order given, each as
    ``{"occurrence", "status", "retry_count", "detail"}``.

    ``status`` is ``"unknown"`` for a night with no recorded observation at
    all (never silently dropped the way a bare query over
    ``health_observations`` would drop it), or for one explicitly recorded
    with ``ok=None``; otherwise ``"pass"``/``"fail"``. ``retry_count`` is
    ``record_check``'s own per-occurrence ``attempts`` counter: zero for a
    night observed exactly once, however many retries or later generations
    also observed the SAME night for every one after the first -- never a
    count of nights.
    """
    rows = {row["occurrence"]: row for row in conn.execute(
        "SELECT occurrence, ok, receipt_json, attempts FROM health_observations WHERE kind='engineering'")}
    history = []
    for occurrence in occurrences:
        row = rows.get(occurrence)
        if row is None:
            history.append({"occurrence": occurrence, "status": "unknown", "retry_count": 0, "detail": None})
            continue
        status = "unknown" if row["ok"] is None else ("pass" if row["ok"] else "fail")
        history.append({"occurrence": occurrence, "status": status,
                        "retry_count": row["attempts"], "detail": json.loads(row["receipt_json"])})
    return tuple(history)


def engineering_streak_from_history(history):
    """The guide's ``engineering_streak`` summary, derived DIRECTLY from the
    SAME windowed ``history`` an ``OperationsStatus`` document already
    carries -- 2026-09-14 review fix. ``history`` is a sequence of
    ``engine.v2.contracts.EngineeringNight`` (or anything with the same
    ``occurrence``/``status``/``detail`` attributes), oldest first, exactly
    what callers build from :func:`engineering_history`'s own dict rows
    before constructing an ``OperationsStatus``.

    ``budget_streak`` (below) answers a DIFFERENT question over a DIFFERENT
    population: it scans ``health_observations`` UNBOUNDED (every occurrence
    ever recorded, not the ``nights``-long trailing window), so its
    ``consecutive_nights`` can exceed, or simply disagree with, what the
    same document's own ``engineering_history`` visibly shows -- a status
    document that displays 14 nights while claiming a 20-night failure
    streak is not "consistent," it is two different reports glued together.
    This function counts the exact same thing (consecutive FAILURES walking
    back from the most recent night, stopping at the first PASS; an
    UNKNOWN night neither breaks the streak nor counts as a failure --
    "never observed" is not itself a failure) but ONLY over ``history``, so
    the two fields of one document can never contradict each other.
    ``budget_streak`` itself is UNCHANGED and still feeds ``/health.json``'s
    ``code_budgets`` (the legacy shell's own "degraded" banner), which has
    always wanted the unbounded, all-time streak, not a windowed one.
    """
    failures, unknown, first, latest = 0, [], None, None
    for night in reversed(history):
        if latest is None:
            latest = night
        if night.status == "pass":
            break
        if night.status == "unknown":
            unknown.append(night.occurrence)
            continue
        failures += 1
        first = night.occurrence
    return {"ok": latest is not None and latest.status == "pass",
            "first_failed_on": first, "consecutive_nights": failures,
            "unknown_occurrences": unknown,
            "latest": latest.detail if latest is not None else None, "override": None}


def budget_streak(conn):
    rows = conn.execute("SELECT occurrence,ok,receipt_json FROM health_observations "
                        "WHERE kind='engineering' ORDER BY occurrence DESC").fetchall()
    failures, unknown, first = 0, [], None
    for row in rows:
        if row["ok"] == 1:
            break
        if row["ok"] is None:
            unknown.append(row["occurrence"])
        else:
            failures += 1
            first = row["occurrence"]
    return {"ok": bool(rows) and rows[0]["ok"] == 1, "first_failed_on": first,
            "consecutive_nights": failures, "unknown_occurrences": unknown,
            "latest": json.loads(rows[0]["receipt_json"]) if rows else None,
            "override": None}


#: A canonical session is a plain ``YYYY-MM-DD`` calendar date string -- the
#: one form the durable receipts carry and the only form accepted as evidence.
_CANONICAL_SESSION = re.compile(r"\d{4}-\d{2}-\d{2}")


def _receipt_document(text, *, kind, logical_key):
    """Decode one delivered outbox receipt. Missing text, malformed JSON or a
    non-object payload are the same ``VALIDATION_FAILED`` class as a missing
    chain -- never a fallback."""
    details = {"kind": kind, "logical_key": logical_key}
    if not isinstance(text, str):
        raise fail("VALIDATION_FAILED", f"delivered {kind} receipt is missing or not durable JSON text",
                   details=details)
    try:
        receipt = json.loads(text)
    except ValueError:
        raise fail("VALIDATION_FAILED", f"delivered {kind} receipt is malformed JSON",
                   details=details) from None
    if not isinstance(receipt, dict):
        raise fail("VALIDATION_FAILED", f"delivered {kind} receipt is not a JSON object",
                   details=details)
    return receipt


def _canonical_session(value, *, field, details):
    if not isinstance(value, str) or not _CANONICAL_SESSION.fullmatch(value):
        raise fail("VALIDATION_FAILED", f"{field} is not a canonical YYYY-MM-DD session", details=details)
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise fail("VALIDATION_FAILED", f"{field} is not a real calendar date", details=details) from None


def _export_chain_sessions(conn, logical_key, *, release_id):
    """The chain's authoritative ``ledger_export_receipt.v1.0`` -- its
    ``requested_session`` and ``session`` (the resolved one) returned verbatim,
    never re-derived."""
    row = conn.execute("SELECT state,receipt_json FROM outbox WHERE kind='export' AND logical_key=?",
                       (logical_key,)).fetchone()
    if row is None or row["state"] != "delivered":
        raise fail("VALIDATION_FAILED", "the release_intent chain has no delivered export receipt",
                   details={"release_id": release_id, "logical_key": logical_key})
    receipt = _receipt_document(row["receipt_json"], kind="export", logical_key=logical_key)
    details = {"release_id": release_id, "logical_key": logical_key}
    if receipt.get("schema_version") != "ledger_export_receipt.v1.0":
        raise fail("VALIDATION_FAILED", "the export receipt is not an authoritative "
                   "ledger_export_receipt.v1.0", details=details)
    scope = receipt.get("scope")
    if not isinstance(scope, str) or not scope:
        raise fail("VALIDATION_FAILED", "the export receipt scope is not a non-empty string", details=details)
    requested = _canonical_session(receipt.get("requested_session"),
                                   field="export receipt requested_session", details=details)
    resolved = _canonical_session(receipt.get("session"), field="export receipt session", details=details)
    if resolved > requested:
        raise fail("VALIDATION_FAILED", "the export receipt resolved a session later than the session "
                   "it requested", details=details)
    return receipt["requested_session"], receipt["session"]


def _release_session_evidence(conn, current):
    """``(requested_session, resolved_session)`` for the delivered release
    ``current`` names, copied ONLY from the durable catalog receipt chain: the
    one delivered ``release_intent`` outbox row whose
    ``receipt_json.release_id`` equals ``current.release_id`` selects, by its
    ``logical_key``, the paired delivered ``export`` row carrying the
    authoritative ``ledger_export_receipt.v1.0``. Missing, malformed,
    ambiguous or mismatched evidence is ``VALIDATION_FAILED`` -- never
    ``clock.now()``, never the release occurrence alone, never an equality
    default, never a fabricated value. The ``health()`` contract ALWAYS
    requires one unique, complete chain for its current delivered release:
    with no delivered release selected the required evidence is absent and
    this refuses rather than emitting null/default session values. A
    ``release_intent`` row whose receipt cannot be decoded or is not an
    object cannot match the current release, so the scan skips it; if that
    leaves zero matching chains the refusal below reports the missing
    evidence. A malformed receipt inside the matched chain -- the export
    receipt -- still refuses. The ``release_intent`` scan is keyed and
    ordered deterministically so identical catalog contents always produce
    the same result."""
    if current is None:
        raise fail("VALIDATION_FAILED", "the required session evidence is absent: no delivered "
                   "release is selected for the health contract")
    release_id = current["release_id"]
    logical_keys = []
    for row in conn.execute("SELECT logical_key,receipt_json FROM outbox WHERE kind='release_intent' "
                             "AND state='delivered' ORDER BY logical_key"):
        try:
            receipt = _receipt_document(row["receipt_json"], kind="release_intent",
                                        logical_key=row["logical_key"])
        except OpsError:
            continue  # an undecodable UNRELATED receipt cannot match the current release
        release_ids = receipt.get("release_ids")
        if (receipt.get("release_id") == release_id
                or (isinstance(release_ids, list) and release_id in release_ids)):
            logical_keys.append(row["logical_key"])
    if len(logical_keys) != 1:
        raise fail("VALIDATION_FAILED", "the current delivered release does not resolve to exactly one "
                   "complete release_intent receipt chain",
                   details={"release_id": release_id, "matching_release_intent_rows": len(logical_keys)})
    requested, resolved = _export_chain_sessions(conn, logical_keys[0], release_id=release_id)
    if resolved != current["occurrence"]:
        raise fail("VALIDATION_FAILED", "the export receipt session does not equal the delivered "
                   "release occurrence",
                   details={"release_id": release_id, "occurrence": current["occurrence"],
                            "resolved_session": resolved})
    return requested, resolved


def _withheld_release(conn, current=None):
    """The withheld-release projection shared by :func:`health` and the
    operations-status sidecar, as the same dict-or-None value: the latest
    INELIGIBLE release, reported only while no later occurrence has been
    delivered since -- once delivery catches up to (or passes) it, the banner
    must clear rather than stick forever.

    ``current`` is the already-read current delivered-release row; when absent
    it is queried here. Internal only: it emits no health document and never
    invents a session field, so a caller can read the banner before any export
    receipt exists -- where the public ``health()`` producer must still refuse.
    """
    if current is None:
        current = conn.execute("SELECT release_id,occurrence,delivered_at FROM releases "
                               "WHERE delivered_at IS NOT NULL ORDER BY occurrence DESC, release_id DESC LIMIT 1").fetchone()
    withheld = conn.execute("SELECT release_id,occurrence FROM releases WHERE eligible=0 "
                            "ORDER BY occurrence DESC, release_id DESC LIMIT 1").fetchone()
    show = withheld is not None and (
        current is None or withheld["occurrence"] > current["occurrence"])
    return dict(withheld) if show else None


def health(conn, *, clock, executor_mode="watchdog"):
    jobs = [dict(row) for row in conn.execute(
        "SELECT job_id,kind,state,created_at,updated_at,queue_reason_json FROM jobs "
        "WHERE state NOT IN ('succeeded','cancelled') ORDER BY created_at, job_id")]
    current = conn.execute("SELECT release_id,occurrence,delivered_at FROM releases "
                           "WHERE delivered_at IS NOT NULL ORDER BY occurrence DESC, release_id DESC LIMIT 1").fetchone()
    withheld_release = _withheld_release(conn, current)
    # A nonzero count here (any scope) means some run recorded evidence that
    # a later generation/legacy line disagreed with an already-committed
    # decision instead of overwriting it (guide §5.5 item 1; includes the
    # ``legacy_settlement`` scope's contract-field divergences) -- surfaced
    # here so an operator sees it without reading the ledger directly.
    divergences = {row["scope"]: row["count"] for row in conn.execute(
        "SELECT scope, COUNT(*) AS count FROM decision_divergences GROUP BY scope ORDER BY scope")}
    # 2026-09-15: a generation_ref IS NULL outcome row can never dedupe a
    # same-session legacy_settlement rerun (decision_commit.
    # _match_same_session matches only a recorded generation_ref) --
    # engine.v2.ops.session_backfill fixes every row it can derive a
    # session for automatically (cli.main, on every write-opened catalog);
    # what is left here is either underivable by design, or this catalog
    # has not been opened for write since the fix -- surfaced either way.
    undetermined_outcome_sessions = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE kind='outcome' "
        "AND generation_ref IS NULL").fetchone()[0]
    requested_session, resolved_session = _release_session_evidence(conn, current)
    return {"schema_version": "operations_health.v1.1", "generated_at": format_timestamp(clock.now()),
            "executor_mode": executor_mode, "containment": "best_effort" if executor_mode == "watchdog" else "kernel",
            "jobs": jobs, "watermarks": [dict(row) for row in conn.execute("SELECT * FROM watermarks ORDER BY pipeline, scope, stage")],
            "current_release": dict(current) if current else None,
            "requested_session": requested_session, "resolved_session": resolved_session,
            "withheld_release": withheld_release,
            "code_budgets": budget_streak(conn), "activation": "shadow_only",
            "decision_divergences": divergences,
            "undetermined_outcome_sessions": undetermined_outcome_sessions}


def write_health(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    temporary.write_text(json.dumps(document))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
