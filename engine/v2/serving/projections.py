"""The minimal serving index — rearchitecture phase-3 guide §5.4 (P3-1b).

``build_candidate`` is the offline projection coordinator's one write path:
resolve every score row's ``(ticker, event_date)`` to an ``EventRef`` through
the pinned Phase 2 repository (§5.3 point 3), run :func:`engine.v2.serving.
bridge.build_bridges` over the result, publish every large engine/detail
payload as an immutable :class:`~engine.v2.foundation.ArtifactStore` object,
and — only when :class:`~engine.v2.contracts.ProjectionFindings` is ``ok`` —
insert one release plus its event/score index rows in a single SQLite
transaction. ``findings.ok is False`` still writes the findings receipt (an
ordinary content-addressed object, safe to leave unreferenced) but inserts no
row: "no selectable candidate" (§5.4). No "current" pointer is created here —
that is a later task's single published pointer.

Separate SQLite file, own ``schema_versions`` sequence (owner ``"serving"``).
**Not** ``engine.v2.ops.migrations``: serving and ops are both layer 7 peers
(this guide's §2, restated at the top of §5) and neither may import the
other, so the checksum-and-refuse-newer *pattern* is reimplemented here,
minimally, rather than imported — the judgement call the task brief asked
this module to record.

Layer 7 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts``, ``engine.v2.foundation``, ``engine.v2.data.
repository`` (layer 1) and this package's own ``bridge`` — never
``engine.v2.ops`` or ``engine.v2.diagnosis``.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from engine.v2.contracts import (
    LEGACY_SCORE_BRIDGE_V1,
    ArtifactRef,
    DataQuery,
    EventPage,
    EventPageItem,
    EventRef,
    EventScoreSummary,
    KeyPredicate,
    LegacyScoreBridge,
    PreviewInput,
    PreviewRelease,
    Problem,
    ProjectionFindings,
    SnapshotRef,
)
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    Clock,
    SystemClock,
    canonical_json,
    content_hash,
    format_timestamp,
    from_document,
    to_document,
)

from .bridge import build_bridges

__all__ = [
    "DEFAULT_PAGE_SIZE", "MAX_PAGE_SIZE",
    "ServingIndexError",
    "build_candidate", "connect", "ensure_schema",
    "event_scores", "get_release", "get_score_detail", "list_events",
    "resolve_event_refs",
]

_OWNER = "serving"
_EVENTS_TABLE = "earnings_events"
_LOOKUP_BATCH_CAP = 1000
_LOOKUP_RESULT_CAP = 1000
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

#: A static identity for this coordinator's own join/mapping logic — not a
#: hash of the file's bytes (out of scope here), but distinct from
#: ``preview_input.source_code_hash``, which stays the legacy scorer's own.
_PROJECTION_CODE_HASH = content_hash({"module": "engine.v2.serving.projections", "version": 1})


class ServingIndexError(Exception):
    """A refused serving-index operation, with its ``Problem`` envelope."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(f"{problem.code}: {problem.message}")
        self.problem = problem


# --------------------------------------------------------------------------
# connection, schema, migrations — the checksum-and-refuse-newer pattern,
# reimplemented locally (serving may not import engine.v2.ops.migrations)
# --------------------------------------------------------------------------

_SCHEMA_VERSIONS_DDL = """CREATE TABLE IF NOT EXISTS schema_versions (
    owner TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    PRIMARY KEY (owner, version)
) STRICT"""

_V1 = (
    """CREATE TABLE serving_release (
        release_id TEXT PRIMARY KEY,
        document_json TEXT NOT NULL,
        findings_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status = 'candidate'),
        written_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE serving_object (
        artifact_id TEXT PRIMARY KEY,
        ref_json TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE serving_event_summary (
        release_id TEXT NOT NULL REFERENCES serving_release(release_id),
        event_id TEXT NOT NULL,
        calendar_revision TEXT NOT NULL,
        ticker TEXT NOT NULL,
        event_date TEXT NOT NULL,
        session TEXT,
        clock_id TEXT NOT NULL,
        readiness TEXT NOT NULL,
        PRIMARY KEY (release_id, event_id)
    ) STRICT""",
    "CREATE INDEX ix_serving_event_summary_order "
    "ON serving_event_summary(release_id, event_date, ticker, event_id)",
    """CREATE TABLE serving_score_summary (
        release_id TEXT NOT NULL REFERENCES serving_release(release_id),
        score_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        strategy TEXT NOT NULL,
        verdict TEXT,
        refusal_reason TEXT,
        driver_forecast REAL,
        market_implied_move REAL,
        entry_premium REAL,
        expected_return REAL,
        expected_return_model REAL,
        expected_return_analog REAL,
        expected_return_sim REAL,
        chosen_strategy TEXT,
        chosen_margin REAL,
        menu_size INTEGER,
        detail_artifact_id TEXT NOT NULL REFERENCES serving_object(artifact_id),
        PRIMARY KEY (release_id, score_id)
    ) STRICT""",
    "CREATE INDEX ix_serving_score_summary_event "
    "ON serving_score_summary(release_id, event_id)",
)

#: ``(version, name, statements)`` — one transaction each, numbered 1..n.
_MIGRATIONS = ((1, "serving_projections", _V1),)


@contextmanager
def _transaction(conn: sqlite3.Connection):
    """One short immediate transaction — mirrors ``engine.v2.ops.catalog.
    transaction`` exactly, reimplemented rather than imported (a layer-7 peer)."""
    if conn.in_transaction:
        raise RuntimeError("nested serving transaction: effects commit in the caller's transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _checksum(version: int, name: str, statements: tuple[str, ...]) -> str:
    return content_hash({"version": version, "name": name, "statements": list(statements)})


def ensure_schema(conn: sqlite3.Connection, *, clock: Clock | None = None) -> None:
    """Apply every pending ``serving`` migration; refuse a newer or edited one."""
    clock = clock or SystemClock()
    known = {version: _checksum(version, name, statements)
             for version, name, statements in _MIGRATIONS}
    with _transaction(conn):
        conn.execute(_SCHEMA_VERSIONS_DDL)
        applied = {int(row[0]): str(row[1]) for row in conn.execute(
            "SELECT version, checksum FROM schema_versions WHERE owner = ?", (_OWNER,))}
        _check_applied(known, applied)
    for version, name, statements in _MIGRATIONS:
        with _transaction(conn):
            applied = {int(row[0]) for row in conn.execute(
                "SELECT version FROM schema_versions WHERE owner = ?", (_OWNER,))}
            if version in applied:
                continue
            for statement in statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_versions (owner, version, name, checksum, applied_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (_OWNER, version, name, known[version], format_timestamp(clock.now())))


def _check_applied(known: dict[int, str], applied: dict[int, str]) -> None:
    newer = sorted(v for v in applied if v not in known)
    if newer:
        raise ServingIndexError(Problem(
            code="INTEGRITY_FAILED", category="integrity", retryable=False,
            message="serving schema is newer than this code supports",
            details={"applied": newer[-1], "supported": len(known)}))
    for version, found in sorted(applied.items()):
        if known[version] != found:
            raise ServingIndexError(Problem(
                code="INTEGRITY_FAILED", category="integrity", retryable=False,
                message="a serving migration differs from the one applied",
                details={"version": version}))


def connect(path: str, *, clock: Clock | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the serving SQLite file, schema applied."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    ensure_schema(conn, clock=clock)
    return conn


# --------------------------------------------------------------------------
# §5.3 point 3: the (ticker, event_date) -> EventRef resolver this task adds
# --------------------------------------------------------------------------


def resolve_event_refs(repository: Repository, snapshot_ref: SnapshotRef,
                       pairs) -> dict[tuple[str, str], EventRef | None]:
    """``{(ticker, event_date): EventRef | None}`` for ``build_bridges``.

    Built on ``Repository.scan`` (bounded, never a raw parquet read): one
    query per distinct ticker, filtered client-side to the exact date, the
    same technique ``events._has_cluster_conflict`` already uses. A pair with
    zero matches is simply absent from the returned mapping (unmapped, per
    ``bridge._resolve_event``); two or more distinct ``event_id`` rows for
    one pair map to ``None`` (ambiguous). Neither ever invents an id.

    **Gap**: if the pinned snapshot carries no ``earnings_events`` table at
    all, every pair is unmapped (an empty mapping is returned) rather than a
    hard failure — the same "report the handoff gap, keep the preview usable"
    stance §5.3's closing paragraph asks for.
    """
    if _EVENTS_TABLE not in snapshot_ref.table_versions:
        return {}
    dvr = snapshot_ref.table_versions[_EVENTS_TABLE]
    rows_by_ticker: dict[str, list[dict]] = {}
    result: dict[tuple[str, str], EventRef | None] = {}
    for ticker, event_date in pairs:
        if ticker not in rows_by_ticker:
            rows_by_ticker[ticker] = _rows_by_ticker(repository, snapshot_ref, dvr.table_contract_ref, ticker)
        matches = [row for row in rows_by_ticker[ticker] if _event_date_str(row["event_date"]) == event_date]
        if not matches:
            continue
        ids = {row["event_id"] for row in matches}
        result[(ticker, event_date)] = (
            None if len(ids) > 1
            else EventRef(event_id=matches[0]["event_id"], calendar_revision=dvr.dataset_version_id))
    return result


def _rows_by_ticker(repository: Repository, snapshot_ref: SnapshotRef, contract_ref, ticker: str) -> list[dict]:
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id, table_contract_ref=contract_ref,
        columns=("event_id", "ticker", "event_date"),
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=(ticker,)),),
        order_by=("event_id",), max_batch_rows=_LOOKUP_BATCH_CAP, max_result_rows=_LOOKUP_RESULT_CAP)
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=_EVENTS_TABLE):
        rows.extend(batch.to_pylist())
    return rows


def _event_date_str(value) -> str:
    return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]


def _pairs(score_doc, bundle_rows_by_ticker) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in list(score_doc.get("rows") or ()) + list(score_doc.get("ladder") or ()):
        if row.get("ticker") is not None and row.get("event_date") is not None:
            pairs.add((str(row["ticker"]), str(row["event_date"])))
    for rows in bundle_rows_by_ticker.values():
        for row in rows:
            if row.get("ticker") is not None and row.get("event_date") is not None:
                pairs.add((str(row["ticker"]), str(row["event_date"])))
    return pairs


# --------------------------------------------------------------------------
# build_candidate — §5.3 points 5-8
# --------------------------------------------------------------------------


def _release_id(preview_input: PreviewInput, snapshot_ref: SnapshotRef, score_doc,
                bundle_rows_by_ticker, *, requested_as_of: str, resolved_as_of: str) -> str:
    """Pure function of content: identical inputs give an identical id; a
    changed score or bundle changes it (§5.3 point 8's idempotency test)."""
    return content_hash({
        "source_release_id": preview_input.source_release_id,
        "score_batch_ref": preview_input.score_batch_ref,
        "bundle_manifest_ref": preview_input.bundle_manifest_ref,
        "model_registry_artifact_refs": list(preview_input.model_registry_artifact_refs),
        "score_job_input_refs": list(preview_input.score_job_input_refs),
        "snapshot_id": snapshot_ref.snapshot_id,
        "snapshot_manifest_hash": snapshot_ref.manifest_hash,
        "requested_as_of": requested_as_of, "resolved_as_of": resolved_as_of,
        "score_doc": score_doc,
        "bundle_rows_by_ticker": dict(bundle_rows_by_ticker),
        "projection_code_hash": _PROJECTION_CODE_HASH,
    })


def _publish_document(store, document, schema_ref: str) -> ArtifactRef:
    return store.publish_bytes(canonical_json(document).encode("utf-8"), schema_ref=schema_ref)


def _coverage_summary(findings: ProjectionFindings) -> dict[str, float]:
    planned = findings.planned_population
    return {
        "planned_population": float(planned),
        "compared_population": float(findings.compared_population),
        "coverage": (findings.compared_population / planned) if planned else 0.0,
    }


def _score_summary_fields(bridge: LegacyScoreBridge) -> dict:
    """§6: the board-table fields, read straight off an already-built
    ``display_record`` through the field names ``LEGACY_DISPLAY_MAPPING_V1``
    already checks — never recomputed, never chosen between (review fix).

    ``verdict``: the row's own ``gate_pass`` field, kept as its raw JSON
    value (``"true"``/``"false"``), never translated into an invented
    ``TRADE``/``REFUSED`` vocabulary the legacy row does not carry.

    ``expected_return``: always ``None`` here — see
    ``EventScoreSummary.expected_return``'s docstring for why picking one of
    ``exp_pnl_model``/``_analog``/``_sim`` would itself be the "choose
    between rendered values" §5.2/§5.3 refuse. All three are copied to their
    own fields instead, exactly as the row carries them; a null
    ``exp_pnl_model`` with a present ``exp_pnl_analog`` therefore stays two
    separate values, never folded into one.
    """
    display = bridge.display_record
    gate_pass = display.get("gate_pass")
    verdict = None if gate_pass is None else ("true" if gate_pass else "false")
    refusal_reason = display.get("detail") if gate_pass is False else None
    return dict(
        strategy=display.get("strategy"), verdict=verdict, refusal_reason=refusal_reason,
        driver_forecast=display.get("driver_prediction"), market_implied_move=display.get("implied_move"),
        entry_premium=display.get("entry_cost"), expected_return=None,
        expected_return_model=display.get("exp_pnl_model"),
        expected_return_analog=display.get("exp_pnl_analog"),
        expected_return_sim=display.get("exp_pnl_sim"),
        chosen_strategy=display.get("chosen_strategy"), chosen_margin=display.get("chosen_margin"),
        menu_size=display.get("menu_size"))


def build_candidate(
    preview_input: PreviewInput, score_doc, bundle_rows_by_ticker, *,
    repository: Repository, snapshot_ref: SnapshotRef, store, conn: sqlite3.Connection,
    requested_as_of: str, resolved_as_of: str, clock: Clock | None = None,
    fault=None,
) -> PreviewRelease | Problem:
    """§5.3-§5.4: resolve, bridge, publish, and — only if ``findings.ok`` —
    index one release, transactionally and idempotently.

    ``fault``, when given, is called with a named point after every durable
    write (``"findings_written"``, ``"details_written"``, before the index
    transaction) so a test can raise mid-sequence and assert the rerun
    succeeds with no partial release — the O11-style fault hook this
    package's siblings already use.
    """
    clock = clock or SystemClock()
    ensure_schema(conn, clock=clock)
    event_refs = resolve_event_refs(repository, snapshot_ref, _pairs(score_doc, bundle_rows_by_ticker))
    bridges, findings = build_bridges(
        score_doc, bundle_rows_by_ticker, event_refs,
        score_batch_ref=preview_input.score_batch_ref, snapshot_ref=snapshot_ref.snapshot_id,
        model_registry_artifact_refs=preview_input.model_registry_artifact_refs,
        request_provenance_refs=preview_input.score_job_input_refs)
    release_id = _release_id(preview_input, snapshot_ref, score_doc, bundle_rows_by_ticker,
                             requested_as_of=requested_as_of, resolved_as_of=resolved_as_of)
    findings_ref = _publish_document(store, to_document(findings), "projection_findings.v1.0")
    if fault is not None:
        fault("findings_written")
    if not findings.ok:
        return Problem(
            code="PROJECTION_REFUSED", category="validation", retryable=False,
            message="projection findings were not ok; no candidate release was written",
            diagnostic_ref=findings_ref.content_hash,
            details={"release_id": release_id, "findings_artifact_id": findings_ref.artifact_id,
                     "findings": to_document(findings)})
    objects: dict[str, ArtifactRef] = {findings_ref.artifact_id: findings_ref}
    detail_refs: dict[str, ArtifactRef] = {}
    for bridge in bridges:
        ref = _publish_document(store, to_document(bridge), LEGACY_SCORE_BRIDGE_V1)
        detail_refs[bridge.score_id] = ref
        objects[ref.artifact_id] = ref
    if fault is not None:
        fault("details_written")
    manifest_ref = _publish_document(store, {
        "release_id": release_id, "source_release_id": preview_input.source_release_id,
        "snapshot_ref": snapshot_ref.snapshot_id, "score_batch_ref": preview_input.score_batch_ref,
        "bundle_manifest_ref": preview_input.bundle_manifest_ref,
        "event_ids": sorted({b.event_ref.event_id for b in bridges}),
        "score_ids": sorted(b.score_id for b in bridges),
    }, "serving_projection_manifest.v1.0")
    objects[manifest_ref.artifact_id] = manifest_ref
    release = PreviewRelease(
        release_id=release_id, source_release_id=preview_input.source_release_id,
        projection_manifest_ref=manifest_ref.artifact_id, snapshot_ref=snapshot_ref.snapshot_id,
        score_batch_ref=preview_input.score_batch_ref, bundle_manifest_ref=preview_input.bundle_manifest_ref,
        model_registry_artifact_refs=preview_input.model_registry_artifact_refs,
        model_evidence_ref=preview_input.model_evidence_ref,
        comparison_receipt_refs=(preview_input.score_comparison_receipt_ref,
                                 preview_input.render_comparison_receipt_ref, findings_ref.artifact_id),
        source_code_hash=preview_input.source_code_hash, projection_code_hash=_PROJECTION_CODE_HASH,
        requested_as_of=requested_as_of, resolved_as_of=resolved_as_of,
        clock_ids=tuple(sorted({b.clock_id for b in bridges})),
        coverage_summary=_coverage_summary(findings), stale_or_degraded_reasons=())
    if fault is not None:
        fault("release_built")
    _write_index(conn, release, findings, bridges, objects, detail_refs, clock=clock, fault=fault)
    return release


def _write_index(conn: sqlite3.Connection, release: PreviewRelease, findings: ProjectionFindings,
                 bridges, objects: dict, detail_refs: dict, *, clock: Clock, fault) -> None:
    """§5.4: one transaction. Every statement is ``INSERT OR IGNORE`` keyed
    by its primary key, so a rerun over the same content — release_id, event
    ids and score ids all unchanged — writes zero additional rows."""
    events: dict[str, dict] = {}
    for bridge in bridges:
        event_id = bridge.event_ref.event_id
        if event_id not in events:
            events[event_id] = dict(
                event_id=event_id, calendar_revision=bridge.event_ref.calendar_revision,
                ticker=str(bridge.engine_record.get("ticker")),
                event_date=str(bridge.engine_record.get("event_date")),
                session=bridge.display_record.get("session"), clock_id=bridge.clock_id)
    with _transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO serving_release "
            "(release_id, document_json, findings_json, status, written_at) VALUES (?, ?, ?, 'candidate', ?)",
            (release.release_id, json.dumps(to_document(release)), json.dumps(to_document(findings)),
             format_timestamp(clock.now())))
        for artifact_id, ref in objects.items():
            conn.execute("INSERT OR IGNORE INTO serving_object (artifact_id, ref_json) VALUES (?, ?)",
                        (artifact_id, json.dumps(to_document(ref))))
        for event in events.values():
            conn.execute(
                "INSERT OR IGNORE INTO serving_event_summary "
                "(release_id, event_id, calendar_revision, ticker, event_date, session, clock_id, readiness) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'ready')",
                (release.release_id, event["event_id"], event["calendar_revision"], event["ticker"],
                 event["event_date"], event["session"], event["clock_id"]))
        for bridge in bridges:
            fields = _score_summary_fields(bridge)
            conn.execute(
                "INSERT OR IGNORE INTO serving_score_summary "
                "(release_id, score_id, event_id, strategy, verdict, refusal_reason, driver_forecast, "
                "market_implied_move, entry_premium, expected_return, expected_return_model, "
                "expected_return_analog, expected_return_sim, chosen_strategy, chosen_margin, "
                "menu_size, detail_artifact_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (release.release_id, bridge.score_id, bridge.event_ref.event_id, fields["strategy"],
                 fields["verdict"], fields["refusal_reason"], fields["driver_forecast"],
                 fields["market_implied_move"], fields["entry_premium"], fields["expected_return"],
                 fields["expected_return_model"], fields["expected_return_analog"], fields["expected_return_sim"],
                 fields["chosen_strategy"], fields["chosen_margin"], fields["menu_size"],
                 detail_refs[bridge.score_id].artifact_id))
        if fault is not None:
            fault("index_rows_written")


# --------------------------------------------------------------------------
# read helpers — bounded, for later API use (P3-2)
# --------------------------------------------------------------------------


def get_release(conn: sqlite3.Connection, release_id: str) -> PreviewRelease | None:
    row = conn.execute("SELECT document_json FROM serving_release WHERE release_id = ?",
                       (release_id,)).fetchone()
    return None if row is None else from_document(PreviewRelease, json.loads(row["document_json"]))


def _load_ref(conn: sqlite3.Connection, artifact_id: str) -> ArtifactRef:
    row = conn.execute("SELECT ref_json FROM serving_object WHERE artifact_id = ?",
                       (artifact_id,)).fetchone()
    if row is None:
        raise ServingIndexError(Problem(
            code="INTEGRITY_FAILED", category="integrity", retryable=False,
            message="a serving index row names an object this index never registered",
            details={"artifact_id": artifact_id}))
    return from_document(ArtifactRef, json.loads(row["ref_json"]))


def get_score_detail(conn: sqlite3.Connection, store, release_id: str, score_id: str) -> LegacyScoreBridge | None:
    row = conn.execute(
        "SELECT detail_artifact_id FROM serving_score_summary WHERE release_id = ? AND score_id = ?",
        (release_id, score_id)).fetchone()
    if row is None:
        return None
    ref = _load_ref(conn, row["detail_artifact_id"])
    document = json.loads(store.read_verified(ref).decode("utf-8"))
    return from_document(LegacyScoreBridge, document)


def _score_summary_from_row(row: sqlite3.Row) -> EventScoreSummary:
    return EventScoreSummary(
        score_id=row["score_id"], strategy=row["strategy"], verdict=row["verdict"],
        refusal_reason=row["refusal_reason"], driver_forecast=row["driver_forecast"],
        market_implied_move=row["market_implied_move"], entry_premium=row["entry_premium"],
        expected_return=row["expected_return"], expected_return_model=row["expected_return_model"],
        expected_return_analog=row["expected_return_analog"], expected_return_sim=row["expected_return_sim"],
        chosen_strategy=row["chosen_strategy"],
        chosen_margin=row["chosen_margin"], menu_size=row["menu_size"])


def event_scores(conn: sqlite3.Connection, release_id: str, event_id: str) -> tuple[EventScoreSummary, ...]:
    rows = conn.execute(
        "SELECT * FROM serving_score_summary WHERE release_id = ? AND event_id = ? ORDER BY score_id",
        (release_id, event_id)).fetchall()
    return tuple(_score_summary_from_row(row) for row in rows)


def _event_page_item(conn: sqlite3.Connection, release_id: str, row: sqlite3.Row) -> EventPageItem:
    return EventPageItem(
        event_ref=EventRef(event_id=row["event_id"], calendar_revision=row["calendar_revision"]),
        ticker=row["ticker"], event_date=row["event_date"], session=row["session"],
        clock_id=row["clock_id"], readiness=row["readiness"],
        scores=event_scores(conn, release_id, row["event_id"]))


def _encode_cursor(row: sqlite3.Row) -> str:
    return "|".join((row["event_date"], row["ticker"], row["event_id"]))


def _decode_cursor(cursor: str | None) -> tuple[str, str, str] | None:
    if cursor is None:
        return None
    parts = cursor.split("|", 2)
    if len(parts) != 3:
        raise ServingIndexError(Problem(
            code="INVALID_REQUEST", category="validation", retryable=False,
            message="malformed event-page cursor", details={"cursor": cursor}))
    return parts[0], parts[1], parts[2]


def list_events(conn: sqlite3.Connection, release_id: str, *,
                limit: int = DEFAULT_PAGE_SIZE, cursor: str | None = None) -> EventPage:
    """§6: bounded, ordered ``event_date, ticker, event_id`` paging with a
    keyset cursor. Integrity-protecting the cursor with a server-held key is
    P3-2 scope (the read API); this is the bounded query it will wrap."""
    limit = max(1, min(limit, MAX_PAGE_SIZE))
    after = _decode_cursor(cursor)
    params: list = [release_id]
    where = "release_id = ?"
    if after is not None:
        where += " AND (event_date, ticker, event_id) > (?, ?, ?)"
        params.extend(after)
    rows = conn.execute(
        f"SELECT * FROM serving_event_summary WHERE {where} "
        "ORDER BY event_date, ticker, event_id LIMIT ?", (*params, limit + 1)).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM serving_event_summary WHERE release_id = ?",
                         (release_id,)).fetchone()[0]
    page, has_more = rows[:limit], len(rows) > limit
    items = tuple(_event_page_item(conn, release_id, row) for row in page)
    next_cursor = _encode_cursor(page[-1]) if has_more else None
    query_hash = content_hash({"release_id": release_id, "limit": limit, "cursor": cursor})
    return EventPage(release_id=release_id, query_hash=query_hash, items=items,
                     next_cursor=next_cursor, total_matching=total)
