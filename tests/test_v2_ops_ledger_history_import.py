"""``engine.v2.ops.ledger_history_import``: bootstrapping a fresh shadow
catalog's decision history from the legacy JSONL ledger, so
``legacy_settlement`` has a committed prediction to settle against on a
catalog's first shadow night (real shadow nightly attempt 9 defect).
"""
from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore
from engine.v2.ledger.decisions import _import_decision_id
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.decision_commit import (
    _commit_row_or_diverge,
    import_settlement_candidates_in_transaction,
)
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.ledger_history_import import import_history
from engine.v2.ops.recovery import SupervisorLock
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit
from tests.ops_support import DEFAULT_POLICY, catalog, sample

POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _root(tmp_path):
    root = tmp_path / "cat"
    root.mkdir()
    return root


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _prediction(row_id, as_of, *, ticker="FAKE", strategy="TWIN-P", event_date="2026-08-15",
               event_id="evt1"):
    return {"row_id": row_id, "as_of": as_of, "ticker": ticker, "strategy": strategy,
            "event_date": event_date, "event_id": event_id,
            "decision_ts": as_of + "T20:00:00Z", "written_at": as_of + "T20:00:01Z",
            "snapshot_hash": "sha256:" + "a" * 64}


def _outcome(row_id, resolved_at, *, ticker="FAKE", strategy="TWIN-P", event_date="2026-08-15",
            status="unresolvable"):
    return {"row_id": row_id, "resolved_at": resolved_at, "ticker": ticker, "strategy": strategy,
            "event_date": event_date, "status": status, "reason": "test"}


def _publish(store, conn, clock, value, schema_ref):
    ref = store.publish_bytes(json.dumps(value, sort_keys=True).encode(), schema_ref=schema_ref)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


# --------------------------------------------------------------------------
# decision_id consistency (task brief's "check first")
# --------------------------------------------------------------------------


def test_import_decision_id_matches_the_settlement_lookup_identity():
    """``_settlement_line`` (decision_commit.py) looks up
    ``"prediction:" + row_id``; ``_commit_row_or_diverge`` computes the same
    string for a live nightly commit. ``_import_decision_id`` must produce
    byte-for-byte the same string for a bootstrap import, or settlement
    would never find imported history. No fix was needed -- this pins it."""
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    imported = _import_decision_id("prediction", row_id, {"row_id": row_id})
    assert imported == "prediction:" + row_id


# --------------------------------------------------------------------------
# core import behaviour
# --------------------------------------------------------------------------


def test_import_predictions_and_outcomes_then_reimport_is_idempotent(tmp_path):
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    source = tmp_path / "legacy"
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    _write(source / "ledger/predictions/2026-08-01.jsonl", [_prediction(row_id, "2026-08-01")])
    _write(source / "ledger/outcomes/2026-08-16.jsonl", [_outcome(row_id, "2026-08-16T22:00:00Z")])

    summary = import_history(conn, root, source, clock=clock)
    assert summary["families"]["predictions"] == {
        "files": 1, "lines": 1, "imported": 1, "already_present": 0, "conflicts": [],
        "divergences": 0, "divergent_row_ids": 0}
    assert summary["families"]["outcomes"] == {
        "files": 1, "lines": 1, "imported": 1, "already_present": 0, "conflicts": [],
        "divergences": 0, "divergent_row_ids": 0}
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 2

    again = import_history(conn, root, source, clock=clock)
    assert again["families"]["predictions"]["imported"] == 0
    assert again["families"]["predictions"]["already_present"] == 1
    assert again["families"]["outcomes"]["imported"] == 0
    assert again["families"]["outcomes"]["already_present"] == 1
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 2


def test_legacy_duplicate_row_id_diverges_first_stays_authoritative(tmp_path):
    """The user's standing 2026-09-14 decision applied to the ledger itself:
    a row_id that repeats WITHIN the legacy history with a differing payload
    never blocks the import -- the first occurrence (deterministic file/line
    order) is imported, and every later differing occurrence is recorded as
    a divergence, never a crash, never an overwrite."""
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    source = tmp_path / "legacy"
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    path = source / "ledger/predictions/2026-08-01.jsonl"
    _write(path, [_prediction(row_id, "2026-08-01", event_id="evt1"),
                 _prediction(row_id, "2026-08-01", event_id="evt1-CHANGED")])

    summary = import_history(conn, root, source, clock=clock)
    assert summary["families"]["predictions"]["lines"] == 2
    assert summary["families"]["predictions"]["imported"] == 1
    assert summary["families"]["predictions"]["divergences"] == 1
    assert summary["families"]["predictions"]["divergent_row_ids"] == 1
    assert summary["families"]["predictions"]["conflicts"] == []

    # exactly one decisions row, holding the FIRST occurrence's content.
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    row = conn.execute("SELECT payload_json FROM decisions WHERE decision_id=?",
                       ("prediction:" + row_id,)).fetchone()
    assert json.loads(row[0])["event_id"] == "evt1"

    divergences = conn.execute(
        "SELECT * FROM decision_divergences WHERE scope='legacy_import'").fetchall()
    assert len(divergences) == 1
    assert divergences[0]["decision_id"] == "prediction:" + row_id
    assert divergences[0]["occurrence"] == row_id
    assert divergences[0]["existing_payload_hash"] != divergences[0]["attempted_payload_hash"]
    assert "legacy_duplicate_row_id" in divergences[0]["reason"]
    assert "2026-08-01.jsonl" in divergences[0]["reason"]
    assert "line 2" in divergences[0]["reason"]

    # provenance for the SECOND (divergent) line still holds its own original
    # bytes, referencing the first occurrence's decision_id -- no uniqueness
    # violation (decision_imports has no per-decision_id constraint).
    source_hash = file_hash(path)
    prov = conn.execute(
        "SELECT decision_id, original_bytes FROM decision_imports "
        "WHERE source_hash=? AND line_number=2", (source_hash,)).fetchone()
    assert prov["decision_id"] == "prediction:" + row_id
    assert json.loads(bytes(prov["original_bytes"]))["event_id"] == "evt1-CHANGED"

    # re-run: idempotent -- no new decisions, no new divergences.
    again = import_history(conn, root, source, clock=clock)
    assert again["families"]["predictions"]["imported"] == 0
    assert again["families"]["predictions"]["already_present"] == 2
    assert again["families"]["predictions"]["divergences"] == 0
    assert again["families"]["predictions"]["divergent_row_ids"] == 0
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM decision_divergences WHERE scope='legacy_import'"
    ).fetchone()[0] == 1


def test_identical_duplicate_row_id_stays_a_plain_no_op(tmp_path):
    """A row_id repeated with BYTE-IDENTICAL content (any position) is a
    plain no-op -- one decision, no divergence -- distinct from the
    differing-payload case above."""
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    source = tmp_path / "legacy"
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    row = _prediction(row_id, "2026-08-01")
    _write(source / "ledger/predictions/2026-08-01.jsonl", [row, dict(row)])

    summary = import_history(conn, root, source, clock=clock)
    assert summary["families"]["predictions"]["lines"] == 2
    # both lines are NEW (source_hash, line_number) provenance entries, so
    # both count as "imported" -- but they collapse onto the SAME single
    # decisions row (insert()'s own byte-identical no-op), never a second
    # one, and never a divergence.
    assert summary["families"]["predictions"]["imported"] == 2
    assert summary["families"]["predictions"]["already_present"] == 0
    assert summary["families"]["predictions"]["divergences"] == 0
    assert summary["families"]["predictions"]["divergent_row_ids"] == 0
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM decision_divergences WHERE scope='legacy_import'"
    ).fetchone()[0] == 0


def test_provenance_conflict_on_tampered_source_same_line_still_refuses(tmp_path):
    """A changed byte at an ALREADY-recorded ``(source_hash, line_number)``
    -- provenance tampering, not a legacy duplicate -- must still refuse
    typed, in the divergence-tolerant mode too: the ``prior`` check runs
    before ``on_conflict`` is even consulted. Constructed by corrupting the
    recorded provenance directly, since a real changed byte on disk changes
    the whole-file ``source_hash`` too (making it a NEW source, which is the
    legacy-duplicate case covered above, not this one)."""
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    source = tmp_path / "legacy"
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    path = source / "ledger/predictions/2026-08-01.jsonl"
    _write(path, [_prediction(row_id, "2026-08-01")])
    import_history(conn, root, source, clock=clock)

    source_hash = file_hash(path)
    with transaction(conn):
        conn.execute("UPDATE decision_imports SET original_bytes=? "
                    "WHERE source_hash=? AND line_number=1", (b"tampered-bytes", source_hash))

    with pytest.raises(OpsError) as excinfo:
        import_history(conn, root, source, clock=clock)
    assert excinfo.value.code == "IDEMPOTENCY_CONFLICT"


def test_settlement_validates_against_the_first_diverged_prediction(tmp_path):
    root = _root(tmp_path)
    conn, clock, supervisor = catalog(root)
    store = ArtifactStore(root)
    source = tmp_path / "legacy"
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    _write(source / "ledger/predictions/2026-08-01.jsonl",
          [_prediction(row_id, "2026-08-01", event_id="evt1"),
           _prediction(row_id, "2026-08-01", event_id="evt1-CHANGED")])
    import_history(conn, root, source, clock=clock)

    settlement_job = JobSpec(kind="legacy_settlement", implementation_ref="x", spec_hash=None,
                             environment_ref="x",
                             parameters={"expected_ids": ("legacy_settlement",), "session": "2026-08-16"},
                             output_namespace="shadow", resource_class="legacy_rebuild",
                             retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace="shadow", idempotency_key="settle",
          principal="operator", job=settlement_job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock, registry=registry())
    assert claim is not None

    row = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-08-15",
          "status": "unresolvable", "resolved_at": "2026-08-16T22:00:00.000000Z"}
    captured = [{"row": row, "original_b64": base64.b64encode(json.dumps(row).encode()).decode("ascii")}]
    candidate_ref = _publish(store, conn, clock, {"rows": captured}, "legacy_action.v1.0")
    with transaction(conn):
        receipts = import_settlement_candidates_in_transaction(
            conn, claim, candidate_ref, captured, clock=clock)
    # settlement validated against the FIRST (authoritative) prediction --
    # it succeeded at all, which it could not if it had somehow been
    # checked against the never-committed second (divergent) occurrence.
    assert len(receipts) == 1
    assert json.loads(receipts[0]["payload_json"])["row_id"] == row_id


def test_through_excludes_rows_dated_after_it(tmp_path):
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    source = tmp_path / "legacy"
    early_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    late_id = "2026-08-20|FAKE|TWIN-P|100.0|2026-09-05"
    _write(source / "ledger/predictions/2026-08-01.jsonl", [_prediction(early_id, "2026-08-01")])
    _write(source / "ledger/predictions/2026-08-20.jsonl",
          [_prediction(late_id, "2026-08-20", event_date="2026-09-05")])
    _write(source / "ledger/outcomes/2026-08-16.jsonl",
          [_outcome(early_id, "2026-08-16T22:00:00Z")])

    summary = import_history(conn, root, source, through=date(2026, 8, 10), clock=clock)
    assert summary["families"]["predictions"]["files"] == 2
    assert summary["families"]["predictions"]["lines"] == 1
    assert summary["families"]["predictions"]["imported"] == 1
    assert summary["families"]["outcomes"]["files"] == 1
    assert summary["families"]["outcomes"]["lines"] == 0  # resolved 2026-08-16, after --through
    assert summary["date_range"] == {"min": "2026-08-01", "max": "2026-08-01"}
    assert conn.execute("SELECT COUNT(*) FROM decisions WHERE decision_id=?",
                        ("prediction:" + late_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM decisions WHERE decision_id=?",
                        ("prediction:" + early_id,)).fetchone()[0] == 1


def test_dry_run_writes_nothing(tmp_path):
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    source = tmp_path / "legacy"
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    _write(source / "ledger/predictions/2026-08-01.jsonl", [_prediction(row_id, "2026-08-01")])

    summary = import_history(conn, root, source, dry_run=True, clock=clock)
    assert summary["families"]["predictions"]["imported"] == 1
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM decision_authority").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM decision_imports").fetchone()[0] == 0

    # a real run afterwards reports the SAME counts -- nothing leaked through.
    real = import_history(conn, root, source, clock=clock)
    assert real["families"]["predictions"]["imported"] == 1
    assert real["families"]["predictions"]["already_present"] == 0


def test_refuses_when_source_root_has_no_ledger_dirs(tmp_path):
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    empty_source = tmp_path / "not_a_checkout"
    empty_source.mkdir()
    with pytest.raises(OpsError) as excinfo:
        import_history(conn, root, empty_source, clock=clock)
    assert excinfo.value.code == "SOURCE_NOT_FOUND"


def test_refuses_when_another_supervisor_holds_the_catalog(tmp_path):
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    source = tmp_path / "legacy"
    _write(source / "ledger/predictions/2026-08-01.jsonl",
          [_prediction("2026-08-01|FAKE|TWIN-P|100.0|2026-08-15", "2026-08-01")])

    lock = SupervisorLock(root / "supervisor.lock")
    assert lock.acquire()
    try:
        with pytest.raises(OpsError) as excinfo:
            import_history(conn, root, source, clock=clock)
        assert excinfo.value.code == "RESOURCE_UNAVAILABLE"
    finally:
        lock.release()
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


# --------------------------------------------------------------------------
# settlement interplay (deliverable 2)
# --------------------------------------------------------------------------


def test_settlement_finds_a_bootstrap_imported_prediction(tmp_path):
    root = _root(tmp_path)
    conn, clock, supervisor = catalog(root)
    store = ArtifactStore(root)
    source = tmp_path / "legacy"
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    _write(source / "ledger/predictions/2026-08-01.jsonl", [_prediction(row_id, "2026-08-01")])
    import_history(conn, root, source, clock=clock)

    settlement_job = JobSpec(kind="legacy_settlement", implementation_ref="x", spec_hash=None,
                             environment_ref="x",
                             parameters={"expected_ids": ("legacy_settlement",), "session": "2026-08-16"},
                             output_namespace="shadow", resource_class="legacy_rebuild",
                             retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace="shadow", idempotency_key="settle",
          principal="operator", job=settlement_job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock, registry=registry())
    assert claim is not None

    row = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-08-15",
          "status": "unresolvable", "resolved_at": "2026-08-16T22:00:00.000000Z"}
    captured = [{"row": row, "original_b64": base64.b64encode(json.dumps(row).encode()).decode("ascii")}]
    candidate_ref = _publish(store, conn, clock, {"rows": captured}, "legacy_action.v1.0")
    with transaction(conn):
        receipts = import_settlement_candidates_in_transaction(
            conn, claim, candidate_ref, captured, clock=clock)
    assert len(receipts) == 1
    assert receipts[0]["decision_id"].startswith("outcome:" + row_id + ":")
    assert json.loads(receipts[0]["payload_json"])["row_id"] == row_id


def _context(row_id):
    return {"purpose": "shadow", "deployment": "shadow:impl-1",
            "clock": "2026-09-10T21:00:00.000000Z", "session": "2026-08-01",
            "scope": "shadow", "validations": []}


def test_bootstrap_prediction_then_identical_nightly_recommit_is_safe(tmp_path):
    """Not a real-operation scenario (bootstrap's ``--through`` and the
    ``row_id``'s own embedded ``as_of`` keep the two namespaces disjoint in
    practice), but the ops-level safety net must not crash on it.

    ``insert()``'s exact-match no-op path can never fire here: the
    bootstrap import's ``purpose``/``logical_key`` (``import_lines``:
    ``purpose="legacy_import"``, ``logical_key=decision_id``) are
    structurally different from a live shadow commit's (``purpose="shadow"``,
    ``logical_key=content_hash([...])``) by construction, for ANY payload.
    So even byte-identical content goes through the divergence path -- this
    test measures that, rather than asserting a bare no-op, and pins the
    one place it DOES stay a true no-op: no second decision row, and the
    two recorded hashes come out equal."""
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    source = tmp_path / "legacy"
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    payload = _prediction(row_id, "2026-08-01")
    _write(source / "ledger/predictions/2026-08-01.jsonl", [payload])
    import_history(conn, root, source, clock=clock)
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1

    with transaction(conn):
        result = _commit_row_or_diverge(conn, _context(row_id), dict(payload), "genref-1", clock=clock)

    assert result is None  # nothing new committed
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    divergences = conn.execute("SELECT * FROM decision_divergences").fetchall()
    assert len(divergences) == 1
    assert divergences[0]["existing_payload_hash"] == divergences[0]["attempted_payload_hash"]
    committed = conn.execute("SELECT payload_json FROM decisions WHERE decision_id=?",
                             ("prediction:" + row_id,)).fetchone()
    assert json.loads(committed[0])["event_id"] == "evt1"


def test_bootstrap_prediction_then_different_nightly_recommit_diverges_not_crashes(tmp_path):
    root = _root(tmp_path)
    conn, clock, _ = catalog(root)
    source = tmp_path / "legacy"
    row_id = "2026-08-01|FAKE|TWIN-P|100.0|2026-08-15"
    payload = _prediction(row_id, "2026-08-01")
    _write(source / "ledger/predictions/2026-08-01.jsonl", [payload])
    import_history(conn, root, source, clock=clock)

    changed = dict(payload, event_id="evt1-changed")
    with transaction(conn):
        result = _commit_row_or_diverge(conn, _context(row_id), changed, "genref-1", clock=clock)

    assert result is None
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    divergences = conn.execute("SELECT * FROM decision_divergences").fetchall()
    assert len(divergences) == 1
    assert divergences[0]["existing_payload_hash"] != divergences[0]["attempted_payload_hash"]
    # the first-committed (bootstrap-imported) content stays authoritative.
    committed = conn.execute("SELECT payload_json FROM decisions WHERE decision_id=?",
                             ("prediction:" + row_id,)).fetchone()
    assert json.loads(committed[0])["event_id"] == "evt1"
