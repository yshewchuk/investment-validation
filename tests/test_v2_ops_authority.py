"""O18-O26: decision authority, budget-only refusal, publication races, watermarks.

Every control plants the exact condition the guide names and asserts durable
catalog state — decisions written, effects enqueued, pointers moved — not exit
status.
"""
from __future__ import annotations

import json
import os

import pytest

from engine.v2.foundation import ArtifactStore, content_hash, to_document
from engine.v2.ledger.decisions import (
    DecisionConflict,
    import_lines,
    insert,
    rows,
    set_authority,
)
from engine.v2.ledger.export import export_generation
from engine.v2.ops.backup import prepare_backup, restore_backup, run_backup
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.decision_commit import commit_decisions, validate_candidates
from engine.v2.ops.errors import OpsError
from engine.v2.ops.health import health, record_check
from engine.v2.ops.lifecycle import heartbeat, request_cancel
from engine.v2.ops.outbox import claim as claim_effect
from engine.v2.ops.outbox import fail_effect, watermark
from engine.v2.ops.publication import current, publish_local, stage_release
from tests.ops_support import catalog, enqueue_claim

STAMP = "2026-09-12T00:00:00.000000Z"
REQUIRED = ("causality", "coverage", "finality", "replay", "selection")


def _authority(conn):
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)


def _context(input_hash, *, session="2026-09-12", purpose="shadow", scope="shadow"):
    return {"purpose": purpose, "session": session, "deployment": "deployment-1",
            "clock": STAMP, "scope": scope, "input_hash": input_hash,
            "validations": {kind: {"ok": True, "input_hash": input_hash}
                            for kind in REQUIRED}}


def _row(input_hash, *, row_id="r1", event_id="e1", strategy="TWIN-P",
         session="2026-09-12", ladder=False):
    return {"row_id": row_id, "event_id": event_id, "strategy": strategy,
            "as_of": session, "snapshot_hash": input_hash, "score": {"is_ladder": ladder}}


def _counts(conn):
    return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("decisions", "outbox", "watermarks", "releases")}


def test_o18_planted_defects_write_no_predictions_and_settlement_proceeds(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _authority(conn)
    input_hash = content_hash({"snapshot": "candidate-scores"})
    claim = enqueue_claim(conn, clock, supervisor, input_refs=(input_hash,))

    defects = (
        lambda ctx, rows_: ctx["validations"]["replay"].update(ok=False),
        lambda ctx, rows_: ctx["validations"].pop("causality"),
        lambda ctx, rows_: ctx["validations"]["coverage"].update(input_hash="sha256:" + "0" * 64),
        lambda ctx, rows_: ctx["validations"]["finality"].update(ok=False),
        lambda ctx, rows_: rows_[0].update(as_of="2026-09-13"),
        lambda ctx, rows_: rows_[0]["score"].update(is_ladder=True),
        lambda ctx, rows_: rows_[0].update(snapshot_hash="sha256:" + "1" * 64),
        lambda ctx, rows_: rows_[0].pop("event_id"),
    )
    for defect in defects:
        context = _context(input_hash)
        candidates = [_row(input_hash)]
        defect(context, candidates)
        with pytest.raises(OpsError) as excinfo:
            validated = validate_candidates(candidates, context)
            commit_decisions(conn, claim, candidates, context, validated, clock=clock)
        assert excinfo.value.code == "VALIDATION_FAILED"
        counts = _counts(conn)
        assert counts == {"decisions": 0, "outbox": 0, "watermarks": 0, "releases": 0}

    # An independent, valid settlement proceeds under the same authority while
    # the prediction path is blocked by its own validation.
    with transaction(conn):
        outcome = insert(conn, logical_key="outcome:e0", decision_id="outcome:o1",
                         payload={"row_id": "o1", "event_id": "e0",
                                  "settled_at": "2026-09-11"},
                         purpose="shadow", kind="outcome", validations={}, created_at=STAMP)
    assert outcome["decision_id"] == "outcome:o1"

    # The valid candidate commits decisions, release intent and the decisions
    # watermark in one transaction.
    context = _context(input_hash)
    candidates = [_row(input_hash), _row(input_hash, row_id="r2", event_id="e2")]
    validated = validate_candidates(candidates, context)
    receipts = commit_decisions(conn, claim, candidates, context, validated, clock=clock)
    assert len(receipts) == 2
    kinds = sorted(row[0] for row in conn.execute("SELECT kind FROM outbox"))
    assert kinds == ["export", "release_intent"]
    mark = conn.execute("SELECT pipeline,scope,stage,occurrence FROM watermarks").fetchone()
    assert tuple(mark) == ("nightly", "shadow", "decisions", "2026-09-12")


def test_o21_finality_deadline_and_backfill_rules(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _authority(conn)
    input_hash = content_hash({"snapshot": "candidate-scores"})
    claim = enqueue_claim(conn, clock, supervisor, input_refs=(input_hash,))
    candidates = [_row(input_hash)]

    # A non-final session cannot freeze an official decision.
    context = _context(input_hash)
    context["validations"]["finality"] = {"ok": False, "input_hash": input_hash}
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        validate_candidates(candidates, context)

    # Production authority is refused outright in Phase 1, so no backfill can
    # be laundered into a timely production decision through this path.
    context = _context(input_hash, purpose="production")
    validated = validate_candidates(candidates, context)
    with pytest.raises(OpsError, match="shadow-only"):
        commit_decisions(conn, claim, candidates, context, validated, clock=clock)

    # A backfilled night records its REAL creation time and stays labelled a
    # research reconstruction; the decision is never backdated to its session.
    # The clock travels in lease-sized steps with heartbeats, exactly as a
    # live supervisor would keep the attempt's lease renewed.
    for _ in range(2 * 24 * 36):
        clock.advance(100)
        assert heartbeat(conn, claim.attempt_id, claim.fence, clock=clock, lease_seconds=120)
    context = _context(input_hash, purpose="research_reconstruction")
    validated = validate_candidates(candidates, context)
    commit_decisions(conn, claim, candidates, context, validated, clock=clock)
    row = conn.execute("SELECT created_at, purpose FROM decisions").fetchone()
    assert row["created_at"].startswith("2026-09-14")
    assert row["purpose"] == "research_reconstruction"


def test_o22_budget_failure_withholds_publication_only(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _authority(conn)
    record_check(conn, "2026-09-12", "engineering", False, {"failed": ["fan_out"]})
    record_check(conn, "2026-09-12", "engineering", False,
                 {"failed": ["fan_out"], "attempt": 2})
    document = health(conn, clock=clock)
    budgets = document["code_budgets"]
    assert budgets["ok"] is False
    assert budgets["first_failed_on"] == "2026-09-12"
    assert budgets["consecutive_nights"] == 1  # nights, not retries

    # Decision work still advances under its own validations.
    input_hash = content_hash({"snapshot": "candidate-scores"})
    claim = enqueue_claim(conn, clock, supervisor, input_refs=(input_hash,))
    context = _context(input_hash)
    candidates = [_row(input_hash)]
    validated = validate_candidates(candidates, context)
    assert len(commit_decisions(conn, claim, candidates, context, validated, clock=clock)) == 1

    # Backup effects remain claimable while engineering is red.
    prepare_backup(conn, "night-1", {}, clock=clock)
    assert claim_effect(conn, "backup", owner="worker", clock=clock) is not None

    # Publication is refused, and the board pointer is never created.
    store = ArtifactStore(tmp_path)
    target = tmp_path / "public"
    release = _stage(conn, store, claim, "rel-1", "2026-09-12", clock,
                     engineering={"ok": False, "reason": "fan_out"})
    assert release["eligible"] is False
    with pytest.raises(OpsError, match="PUBLICATION_REFUSED"):
        publish_local(conn, claim, store, target, "rel-1", scope="global", clock=clock)
    assert not (target / "CURRENT").exists()

    # A crashed/unknown check is not a green night and not a failed night.
    record_check(conn, "2026-09-13", "engineering", None, {"reason": "check crashed"})
    budgets = health(conn, clock=clock)["code_budgets"]
    assert budgets["unknown_occurrences"] == ["2026-09-13"]
    assert budgets["consecutive_nights"] == 1
    assert budgets["ok"] is False


def test_o23_no_override_replaces_a_gate_receipt(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)

    # An "override" document in place of the engineering receipt is not a
    # passed gate: the release stays ineligible and cannot be published.
    files = _files(store, "one")
    gates = _all_gates(store, "rel-1", "2026-09-12", files)
    gates["engineering"] = {"ok": True, "override": {"approved": True, "expiry": None}}
    staged = stage_release(conn, store, "rel-1", "2026-09-12", files,
                           expected_current=None, gates=gates, clock=clock)
    assert staged["eligible"] is False
    with pytest.raises(OpsError, match="PUBLICATION_REFUSED"):
        publish_local(conn, claim, store, tmp_path / "public", "rel-1",
                      scope="global", clock=clock)

    # A receipt artifact that says "waived" instead of "passed" is refused too,
    # even with a well-formed reference and binding.
    files = _files(store, "two")
    binding = _binding("rel-2", "2026-09-12", files)
    waived = _gate_receipt(store, "engineering", "waived_by_override", binding)
    gates = _all_gates(store, "rel-2", "2026-09-12", files)
    gates["engineering"] = _gate(waived, binding)
    staged = stage_release(conn, store, "rel-2", "2026-09-12", files,
                           expected_current=None, gates=gates, clock=clock, claim=claim)
    assert staged["eligible"] is False

    # A passed engineering gate never substitutes for the causal or security
    # gates: each missing gate independently keeps the release ineligible.
    for missing in ("decision", "security", "projection"):
        release_id = "rel-3-" + missing
        files = _files(store, "three-" + missing)
        gates = _all_gates(store, release_id, "2026-09-12", files)
        gates.pop(missing)
        staged = stage_release(conn, store, release_id, "2026-09-12", files,
                               expected_current=None, gates=gates, clock=clock)
        assert staged["eligible"] is False

    # The health document exposes no override: the streak stands.
    record_check(conn, "2026-09-12", "engineering", False,
                 {"override": {"approved": True}})
    budgets = health(conn, clock=clock)["code_budgets"]
    assert budgets["override"] is None
    assert budgets["ok"] is False and budgets["consecutive_nights"] == 1


def test_o24_publication_crash_reconciles_and_stale_delivery_is_refused(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    target = tmp_path / "public"
    claim = enqueue_claim(conn, clock, supervisor)

    files = _files(store, "one")
    gates = _all_gates(store, "rel-1", "2026-09-11", files)
    staged = stage_release(conn, store, "rel-1", "2026-09-11", files,
                           expected_current=None, gates=gates, clock=clock, claim=claim)
    assert staged["eligible"] is True
    assert publish_local(conn, claim, store, target, "rel-1",
                         scope="global", clock=clock)["delivered"] is True
    assert current(target) == "rel-1"

    # Crash between pointer success and local acknowledgement: the catalog
    # transaction rolls back, so nothing is delivered as far as the catalog
    # is concerned, and the last acknowledged release stays authoritative.
    files2 = _files(store, "two")
    gates2 = _all_gates(store, "rel-2", "2026-09-12", files2)
    stage_release(conn, store, "rel-2", "2026-09-12", files2,
                  expected_current="rel-1", gates=gates2, clock=clock, claim=claim)

    def fault(point):
        if point == "pointer_before_ack":
            raise RuntimeError("host died before the local ack")

    with pytest.raises(RuntimeError):
        publish_local(conn, claim, store, target, "rel-2", scope="global",
                      clock=clock, fault=fault)
    row = conn.execute("SELECT published_at FROM releases WHERE release_id='rel-2'").fetchone()
    assert row["published_at"] is None
    marks = {tuple(item) for item in conn.execute(
        "SELECT pipeline,scope,stage,occurrence FROM watermarks")}
    # The crashed attempt moved nothing: the delivery watermark still names
    # the last acknowledged release occurrence, not the unacknowledged one.
    assert ("nightly", "global", "delivery", "2026-09-12") not in marks
    assert ("nightly", "global", "delivery", "2026-09-11") in marks

    # Retrying the SAME release probes the pointer and reconciles without
    # recopying or producing a new decision.
    assert publish_local(conn, claim, store, target, "rel-2",
                         scope="global", clock=clock)["delivered"] is True
    row = conn.execute("SELECT published_at FROM releases WHERE release_id='rel-2'").fetchone()
    assert row["published_at"] is not None
    marks = {tuple(item) for item in conn.execute(
        "SELECT pipeline,scope,stage,occurrence FROM watermarks")}
    assert ("nightly", "global", "publication", "2026-09-12") in marks
    assert ("nightly", "global", "delivery", "2026-09-12") in marks

    # An older release cannot replace a newer accepted one.
    with pytest.raises(OpsError, match="STALE_EXPECTATION"):
        publish_local(conn, claim, store, target, "rel-1", scope="global", clock=clock)
    assert current(target) == "rel-2"

    # A stale fence cannot move the pointer after a cancellation takeover.
    files3 = _files(store, "three")
    gates3 = _all_gates(store, "rel-3", "2026-09-13", files3)
    stage_release(conn, store, "rel-3", "2026-09-13", files3,
                  expected_current="rel-2", gates=gates3, clock=clock, claim=claim)
    request_cancel(conn, claim.job_id, claim.attempt_id, clock=clock)
    with pytest.raises(OpsError) as excinfo:
        publish_local(conn, claim, store, target, "rel-3", scope="global", clock=clock)
    assert excinfo.value.code in ("CANCELLED", "LEASE_LOST")
    assert current(target) == "rel-2"

    # A corrupted release object refuses to materialize; the last good
    # release stays current and the candidate stays unacknowledged.
    stale = dict(files3)
    ref = stale["index.html"]
    object_path = store.verify(ref)
    os.chmod(object_path, 0o644)
    object_path.write_bytes(b"corrupted after staging")
    with pytest.raises(Exception):
        publish_local(conn, claim, store, target, "rel-3", scope="global", clock=clock)
    assert current(target) == "rel-2"
    row = conn.execute("SELECT published_at FROM releases WHERE release_id='rel-3'").fetchone()
    assert row["published_at"] is None


def test_o26_watermarks_are_scoped_monotone_and_independent(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    with transaction(conn):
        watermark(conn, "nightly", "global", "scoring", "2026-09-12", "receipt-a", clock=clock)
        watermark(conn, "nightly", "smoke:FAKE", "scoring", "2026-09-12", "receipt-s",
                  clock=clock)
        watermark(conn, "nightly", "global", "decisions", "2026-09-12", "receipt-d",
                  clock=clock)

    def mark(scope, stage):
        return conn.execute("SELECT occurrence,receipt_ref FROM watermarks "
                            "WHERE pipeline='nightly' AND scope=? AND stage=?",
                            (scope, stage)).fetchone()

    # A subset run advances only its own scope, never the global watermark.
    with transaction(conn):
        watermark(conn, "nightly", "smoke:FAKE", "scoring", "2026-09-13", "receipt-s2",
                  clock=clock)
    assert tuple(mark("global", "scoring")) == ("2026-09-12", "receipt-a")
    assert tuple(mark("smoke:FAKE", "scoring")) == ("2026-09-13", "receipt-s2")

    # An older occurrence cannot regress a watermark.
    with transaction(conn):
        watermark(conn, "nightly", "global", "scoring", "2026-09-11", "receipt-old",
                  clock=clock)
    assert tuple(mark("global", "scoring")) == ("2026-09-12", "receipt-a")

    # A completed occurrence with a different receipt is an integrity error.
    with pytest.raises(OpsError, match="IDEMPOTENCY_CONFLICT"):
        with transaction(conn):
            watermark(conn, "nightly", "global", "scoring", "2026-09-12", "receipt-b",
                      clock=clock)

    # Stages are independent facts: a failed delivery leaves its stage absent
    # while the completed stages stand, and a failed backup retries alone.
    assert mark("global", "delivery") is None
    assert tuple(mark("global", "decisions")) == ("2026-09-12", "receipt-d")
    prepare_backup(conn, "night-2", {}, clock=clock)
    effect = claim_effect(conn, "backup", owner="worker", clock=clock)
    fail_effect(conn, effect["effect_id"], {"code": "BACKUP_FAILED"}, owner="worker",
                claim_token=effect["claim_token"], clock=clock)
    assert tuple(mark("global", "decisions")) == ("2026-09-12", "receipt-d")
    retry = claim_effect(conn, "backup", owner="worker-2", clock=clock)
    assert retry["effect_id"] == effect["effect_id"] and retry["attempts"] == 2


def test_o32_authority_rollback_rehearsal_preserves_decisions(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    legacy_lines = [
        b'{"row_id":"p1","event_id":"e1","as_of":"2026-09-10"}\n',
        b'{"row_id":"p2","event_id":"e2","as_of":"2026-09-11",'
        b'"supersedes":"p1","supersede_reason":"restated"}\n',
    ]
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
        imported = import_lines(conn, "sha256:legacy", legacy_lines, kind="prediction",
                                created_at=STAMP)
        insert(conn, logical_key="k-new", decision_id="prediction:p3",
               payload={"row_id": "p3", "event_id": "e3", "as_of": "2026-09-12"},
               purpose="shadow", kind="prediction", validations={}, created_at=STAMP)
    assert imported[1]["supersedes"] == "prediction:p1"

    # Consistent backup, then an isolated restore (§14.3: rehearsal on copies).
    prepare_backup(conn, "cutover-1", {}, clock=clock)
    run_backup(conn, key="cutover-1", owner="operator", target=tmp_path / "backup",
               clock=clock, store=store)
    restored_root = restore_backup(tmp_path / "backup", tmp_path / "restored")
    restored = open_catalog(restored_root / "ops.sqlite", clock=clock)
    try:
        assert len(rows(restored)) == 3

        # The exported legacy generation is a projection of the catalog, and
        # the restored catalog exports the identical bytes.
        original = export_generation(conn, tmp_path / "compat", generation="g1")
        recovered = export_generation(restored, tmp_path / "compat-restored", generation="g1")
        for path in sorted(original.rglob("*.jsonl")):
            twin = recovered / path.relative_to(original)
            assert twin.read_bytes() == path.read_bytes()

        # Rollback hands writer ownership back to legacy exactly once; the
        # committed history survives and the catalog writer is now refused.
        with transaction(restored):
            set_authority(restored, "catalog", "legacy", "2026-09-13T00:00:00.000000Z")
        with pytest.raises(DecisionConflict):
            with transaction(restored):
                insert(restored, logical_key="k-late", decision_id="prediction:p9",
                       payload={"row_id": "p9"}, purpose="shadow", kind="prediction",
                       validations={}, created_at="2026-09-13T00:00:00.000000Z")
        # A stale rollback under the wrong expected owner is refused, so two
        # writers can never both believe they hold the pen.
        with pytest.raises(DecisionConflict):
            with transaction(restored):
                set_authority(restored, "catalog", "legacy", "2026-09-14T00:00:00.000000Z")
        assert len(rows(restored)) == 3
    finally:
        restored.close()


# --------------------------------------------------------------------------
# release fixtures
# --------------------------------------------------------------------------


def _files(store, name):
    return {"index.html": store.publish_bytes(("<html>" + name + "</html>").encode(),
                                              schema_ref="release_file.v1.0")}


def _binding(release_id, occurrence, files):
    return content_hash({"release_id": release_id, "occurrence": occurrence, "files": files})


def _gate_receipt(store, kind, status, binding):
    return store.publish_bytes(json.dumps({"kind": kind, "status": status,
                                           "input_hash": binding}).encode(),
                               schema_ref="gate_receipt.v1.0")


def _gate(ref, binding):
    return {"ok": True, "receipt_ref": ref.content_hash, "input_hash": binding,
            "receipt_artifact": to_document(ref)}


def _all_gates(store, release_id, occurrence, files):
    binding = _binding(release_id, occurrence, files)
    return {kind: _gate(_gate_receipt(store, kind, "passed", binding), binding)
            for kind in ("decision", "projection", "security", "engineering")}


def _stage(conn, store, claim, release_id, occurrence, clock, *, files=None,
           engineering=None):
    files = files if files is not None else _files(store, release_id)
    gates = _all_gates(store, release_id, occurrence, files)
    if engineering is not None:
        gates["engineering"] = engineering
    return stage_release(conn, store, release_id, occurrence, files,
                         expected_current=None, gates=gates, clock=clock,
                         claim=claim if engineering is None or engineering.get("ok") else None)
