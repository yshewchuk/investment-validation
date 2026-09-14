"""Coordinator-side work for the export, engineering-gate, publication and
backup outbox effects (P2-5/Task5, guide §9.4 item 3).

Each of these four job kinds has a trivial pure worker (``worker.py``'s
``_dispatch_effect_receipt``) that never touches the catalog. All real
catalog, outbox, watermark and filesystem work happens here, called from
``supervisor.Service._coordinator_effect`` — the same fenced finish path
``decision_commit``/``decision_evidence`` already use.

``checks/*`` is verification/dev-tooling, never importable from
``engine/v2/**`` (``checks/import_layers.py``'s "production packages cannot
depend on verification or test code"), and only ``engine.v2.ops.executor``
and ``engine.v2.ops.legacy_adapter`` may call ``subprocess`` at all
(``check_runtime_edges``'s "undeclared-process-edge"). The engineering gate,
the security scan and the export read-back all cross a process boundary, so
this module never calls ``subprocess`` itself — it delegates to the three
audited functions ``legacy_adapter`` exposes for exactly this.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from engine.v2.foundation import content_hash, to_document
from engine.v2.ledger.export import export_generation
from engine.v2.ops.backup import prepare_backup, run_backup
from engine.v2.ops.catalog import dumps, transaction
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import fail
from engine.v2.ops.input_bindings import recorded_bindings
from engine.v2.ops.legacy_adapter import run_engineering_gate as _run_engineering_gate_subprocess
from engine.v2.ops.legacy_adapter import run_security_scan
from engine.v2.ops.legacy_adapter import verify_export_generation as _verify_generation
from engine.v2.ops.outbox import fail_effect, watermark
from engine.v2.ops.publication import current as release_current
from engine.v2.ops.publication import publish_local, stage_release
from engine.v2.ops.session_resolution import resolve_effective_session

__all__ = ["EXPORT_PURPOSES", "backup_effect", "effect_scope", "engineering_gate_effect",
           "ledger_export_effect", "publication_effect"]

#: P2-5/D20: only these two purposes are legacy-compatible rows; a
#: ``research_reconstruction`` row must never reach a legacy-compatible export.
EXPORT_PURPOSES = ("legacy_import", "shadow")


def effect_scope(claim):
    """The watermark/export/release scope this attempt was planned under."""
    return claim.spec.parameters.get("effect_scope") or claim.spec.output_namespace


def _watermark_row(conn, scope, stage):
    return conn.execute(
        "SELECT occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' "
        "AND scope=? AND stage=?", (scope, stage)).fetchone()


# --------------------------------------------------------------------------
# ledger_export
# --------------------------------------------------------------------------


def _catalog_counts(conn, purposes):
    placeholders = ",".join("?" for _ in purposes)
    rows = conn.execute(
        "SELECT kind, COUNT(*) FROM decisions WHERE purpose IN (" + placeholders + ") "
        "GROUP BY kind", tuple(purposes)).fetchall()
    counts = {"predictions": 0, "outcomes": 0}
    for kind, count in rows:
        counts["predictions" if kind == "prediction" else "outcomes"] = count
    return counts


def _tar_bytes(generation_dir):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for child in sorted(Path(generation_dir).iterdir()):
            archive.add(child, arcname=child.name)
    return buffer.getvalue()


def _mark_export_delivered(conn, logical_key, receipt):
    conn.execute("UPDATE outbox SET state='delivered', attempts=attempts+1, receipt_json=? "
                 "WHERE kind='export' AND logical_key=? AND state IN ('pending','running')",
                 (dumps(receipt), logical_key))


def _no_keepalive():
    return None


def ledger_export_effect(conn, store, claim, ops_root, repo_root, *, clock,
                         keepalive=_no_keepalive):
    """Build, verify and publish one export generation; deliver its outbox effects.

    Settlement is read from its own watermark, never as a scheduler
    dependency (nightly.py's ``parent_map``): a failed or not-yet-run
    settlement leaves that watermark absent, and export still proceeds,
    recording settlement as absent in its receipt.
    """
    scope = effect_scope(claim)
    # P2-C03: the job's own ``session`` param is always the REQUESTED date;
    # the decisions watermark's occurrence is decision_commit's own
    # finality-RESOLVED date (context["session"], written verbatim by
    # commit_decisions_in_transaction) — export refers to that resolved
    # session, never the requested one, which legitimately differs on a
    # walk-back night.
    requested_session = claim.spec.parameters["session"]
    decisions_wm = _watermark_row(conn, scope, "decisions")
    if decisions_wm is None:
        raise fail("VALIDATION_FAILED", "no committed decisions for this session",
                   details={"scope": scope, "session": requested_session})
    session = decisions_wm["occurrence"]
    release_key = decisions_wm["receipt_ref"]
    settlement_wm = _watermark_row(conn, scope, "settlement")
    settlement_present = settlement_wm is not None and settlement_wm["occurrence"] == session

    root = Path(ops_root) / "exports" / scope
    generation_dir = export_generation(conn, root, generation=release_key, purposes=EXPORT_PURPOSES)
    keepalive()
    catalog_counts = _catalog_counts(conn, EXPORT_PURPOSES)
    verified_counts = _verify_generation(generation_dir, repo_root)
    keepalive()
    if verified_counts != catalog_counts:
        raise fail("INTEGRITY_FAILED", "exported generation disagrees with the catalog",
                   details={"catalog": catalog_counts, "verified": verified_counts})

    tar_ref = store.publish_bytes(_tar_bytes(generation_dir), schema_ref="ledger_generation.v1.0")
    keepalive()
    receipt = {"schema_version": "ledger_export_receipt.v1.0", "scope": scope, "session": session,
               "requested_session": requested_session,
               "generation": release_key, "counts": verified_counts,
               "settlement": {"present": settlement_present,
                              "release_key": settlement_wm["receipt_ref"] if settlement_present
                              else None}}

    def effect(inner_conn):
        _mark_export_delivered(inner_conn, release_key, receipt)
        if settlement_present:
            _mark_export_delivered(inner_conn, settlement_wm["receipt_ref"], receipt)
        watermark(inner_conn, "nightly", scope, "export", session, tar_ref.content_hash, clock=clock)

    return effect, (("ledger_export", tar_ref),)


# --------------------------------------------------------------------------
# engineering_gate
# --------------------------------------------------------------------------


def _run_engineering_gate(repo_root):
    raw = _run_engineering_gate_subprocess(repo_root)
    # Coverage is a commit-time ratchet, not a nightly check (decision #3):
    # excluded from both the recorded rows and this stage's own "ok".
    rows = {**raw.get("structural", {}),
            **{k: v for k, v in raw.get("engineering", {}).items() if k != "coverage"}}
    ok = bool(rows) and all(isinstance(row, dict) and row.get("ok") for row in rows.values())
    return {"schema_version": "engineering_gate.v1.0", "ok": ok, "rows": rows,
            "code_hash": raw.get("code_hash")}


def engineering_gate_effect(conn, store, claim, repo_root, *, clock):
    scope = effect_scope(claim)
    session = claim.spec.parameters["session"]
    document = _run_engineering_gate(repo_root)
    ref = store.publish_bytes(json.dumps(document, sort_keys=True, default=str).encode(),
                              schema_ref="engineering_gate.v1.0")

    def effect(inner_conn):
        watermark(inner_conn, "nightly", scope, "engineering_gate", session, ref.content_hash,
                 clock=clock)

    return effect, (("engineering_gate", ref),)


# --------------------------------------------------------------------------
# publication
# --------------------------------------------------------------------------


def _gate_dict(store, document):
    ref = store.publish_bytes(json.dumps(document, sort_keys=True, default=str).encode(),
                              schema_ref=document["kind"] + "_gate.v1.0")
    return {"ok": document["status"] == "passed", "receipt_ref": ref.content_hash,
            "input_hash": document["input_hash"], "receipt_artifact": to_document(ref)}


def _decision_gate(conn, store, scope, session, binding_hash):
    # Passed only when the ("nightly", scope, "decisions") watermark shows
    # occurrence == this release's own session — the exact session decision
    # commit's watermark(...) call wrote, whether or not it carried any
    # candidates: a no-entry night still commits (zero decisions, export and
    # release_intent still enqueued) and still advances this watermark, so it
    # still passes. An earlier (or absent) watermark means this release does
    # not speak for its own session's decisions and must be refused.
    row = _watermark_row(conn, scope, "decisions")
    passed = row is not None and row["occurrence"] == session
    document = {"schema_version": "decision_gate.v1.0", "kind": "decision",
                "status": "passed" if passed else "failed", "input_hash": binding_hash,
                "release_key": row["receipt_ref"] if row else None}
    return _gate_dict(store, document)


def _projection_gate(conn, store, bindings, binding_hash):
    row = bindings.get("selfcheck.json")
    passed, detail = False, "selfcheck receipt missing"
    if row is not None:
        ref = artifact(conn, store, row.artifact_id)
        payload = json.loads(store.read_verified(ref))
        passed = payload.get("ok") is True
        detail = None if passed else "selfcheck ok is not true"
    document = {"schema_version": "projection_gate.v1.0", "kind": "projection",
                "status": "passed" if passed else "failed", "input_hash": binding_hash,
                "detail": detail}
    return _gate_dict(store, document)


def _engineering_receipt_gate(conn, store, bindings, binding_hash):
    row = bindings.get("engineering_gate.json")
    passed = False
    if row is not None:
        ref = artifact(conn, store, row.artifact_id)
        payload = json.loads(store.read_verified(ref))
        passed = payload.get("ok") is True
    document = {"schema_version": "engineering_receipt_gate.v1.0", "kind": "engineering",
                "status": "passed" if passed else "failed", "input_hash": binding_hash}
    return _gate_dict(store, document)


def _security_gate(store, files, binding_hash, repo_root):
    bundle_ref = files.get("bundle.tar")
    scan = run_security_scan(store.verify(bundle_ref), repo_root) if bundle_ref else {"ok": False}
    document = {"schema_version": "security_gate.v1.0", "kind": "security",
                "status": "passed" if scan.get("ok") else "failed", "input_hash": binding_hash,
                "violations": scan.get("violations", [])}
    return _gate_dict(store, document)


def publication_effect(conn, store, claim, ops_root, repo_root, *, clock,
                       keepalive=_no_keepalive):
    """Build the four gate receipts, stage the release, then publish it.

    ``stage_release``/``publish_local`` manage their own short transactions
    (§catalog.py forbids nesting), so this whole function runs outside the
    supervisor's own commit_attempt transaction — everything it commits is
    already durable by the time it returns, so the effect closure returned
    to the caller is a no-op.
    """
    scope = effect_scope(claim)
    bindings = recorded_bindings(conn, claim.attempt_id)
    # P2-C03: the bound finality document is the independent anchor for
    # "which session does this release speak for" — never the decisions
    # watermark's own occurrence alone, so a stale watermark for the wrong
    # session still refuses ``_decision_gate`` below, exactly as before,
    # while a genuine walk-back (resolved != requested) still passes.
    finality_row = bindings.get("finality.json")
    if finality_row is None:
        raise fail("VALIDATION_FAILED", "publication has no finality bound")
    finality_doc = json.loads(store.read_verified(artifact(conn, store, finality_row.artifact_id)))
    session = resolve_effective_session(finality_doc, claim.spec.parameters["session"])
    bundle_row = bindings.get("bundle.tar")
    if bundle_row is None:
        raise fail("VALIDATION_FAILED", "publication has no projection bundle bound")
    files = {"bundle.tar": artifact(conn, store, bundle_row.artifact_id)}
    target = Path(ops_root) / "releases" / scope
    release_id = "rel" + content_hash([scope, session]).split(":")[1][:24]
    expected_current = release_current(target)
    binding_hash = content_hash({"release_id": release_id, "occurrence": session, "files": files})
    builders = (
        ("decision", lambda: _decision_gate(conn, store, scope, session, binding_hash)),
        ("projection", lambda: _projection_gate(conn, store, bindings, binding_hash)),
        ("security", lambda: _security_gate(store, files, binding_hash, repo_root)),
        ("engineering", lambda: _engineering_receipt_gate(conn, store, bindings, binding_hash)),
    )
    gates = {}
    for name, build in builders:
        keepalive()
        gates[name] = build()
    keepalive()
    stage_release(conn, store, release_id, session, files, expected_current=expected_current,
                 gates=gates, clock=clock, claim=claim)
    keepalive()
    publish_local(conn, claim, store, target, release_id, scope=scope, clock=clock)
    return None, ()


# --------------------------------------------------------------------------
# backup
# --------------------------------------------------------------------------


def backup_effect(conn, store, claim, ops_root, *, clock, fault=None, keepalive=_no_keepalive):
    """Prepare then run one backup; a failure releases its outbox effect back
    to pending (rather than waiting out its lease) so an immediate retry can
    claim it — the job's own retry policy is what makes this "retryable"."""
    scope = effect_scope(claim)
    # P2-C03: key/watermark the backup by the same resolved session export
    # and decisions use, falling back to the requested one before any
    # decisions watermark exists.
    decisions_wm = _watermark_row(conn, scope, "decisions")
    session = decisions_wm["occurrence"] if decisions_wm is not None else claim.spec.parameters["session"]
    key = "bkp" + content_hash([scope, session]).split(":")[1][:24]
    owner = claim.attempt_id
    prepare_backup(conn, key, {}, clock=clock)
    target = Path(ops_root) / "backups" / scope
    try:
        manifest = run_backup(conn, key=key, owner=owner, target=target, clock=clock, store=store,
                              fault=fault, keepalive=keepalive)
    except Exception:
        row = conn.execute(
            "SELECT effect_id, claim_token FROM outbox WHERE kind='backup' AND logical_key=? "
            "AND claimed_by=?", (key, owner)).fetchone()
        if row is not None:
            fail_effect(conn, row["effect_id"], {"error": "backup_failed"}, owner=owner,
                       claim_token=row["claim_token"], clock=clock)
        raise
    # "its own" watermark (D20): a retry after failure never advances any
    # OTHER stage's watermark, only this one, and only on success.
    with transaction(conn):
        watermark(conn, "nightly", scope, "backup", session, content_hash(manifest), clock=clock)
    return None, ()
