"""D15: ``engine.v2.ops.parity.score_parity_receipt`` — legacy-vs-snapshot
score parity, over two real committed ``legacy_score`` jobs (no subprocess,
no legacy tree, no provider calls: every job is driven straight through the
real submit/claim/``commit_attempt`` machinery with a synthetic ``score.json``
artifact registered as its output, mirroring
``tests/test_v2_ops_nightly_completion.py``'s ``_succeed_parent`` pattern).

``score_parity_receipt`` may not import ``engine.v2.diagnosis`` (a sink layer
— see the module's docstring), so it returns a plain dict shaped as
``comparison_receipt.v1.1``. Every test here round-trips that dict through
``engine.v2.data.documents.decode_document(ComparisonReceipt, ...)`` (this
test module is unconstrained by the layer rule) to prove the shape is really
strict-decodable, not merely dict-shaped.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.data.documents import decode_document
from engine.v2.diagnosis.receipt import AGREE, DIFFER, ComparisonReceipt
from engine.v2.foundation import ArtifactStore, to_document
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.input_bindings import ResolvedBinding, record_resolved_bindings
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.parity import score_parity_receipt
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit
from tests.ops_support import catalog as ops_catalog
from tests.ops_support import sample
from tests.test_checks_phase2_gate import _snapshot_ref

POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
CODE_HASH = "sha256:" + "1" * 64
ENV_HASH = "sha256:" + "2" * 64


def _row(row_id, **fields):
    base = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P",
           "event_date": "2026-09-10", "strike": 100.0, "expiry": "2026-10-16",
           "iv30": 0.42, "structure_params": {"width": 0.25}}
    base.update(fields)
    return base


def _score_doc(rows, *, session="2026-09-10", tickers=("FAKE",)):
    keys = [f'{r["ticker"]}|{r["strategy"]}|{r["event_date"]}' for r in rows]
    return {"rows": rows, "expected_population": keys, "observed_population": sorted(keys),
           "ladder": {}, "tickers": list(tickers), "analog_entry_coverage": {},
           "session": session, "requested_session": session}


def _submit_score_job(conn, clock, supervisor, store, *, key, rows, session="2026-09-10",
                      tickers=("FAKE",), horizon_days=35, snapshot_ref=None):
    """Submit, claim and synthetically succeed one ``legacy_score`` job: a
    real row in ``jobs``/``attempts``, a real ``score.json`` artifact
    registered under ``attempt_outputs`` name ``legacy_score`` — exactly what
    the real worker (``legacy_adapter._action_score``) produces — and,
    when ``snapshot_ref`` is given, a real ``attempt_input_bindings`` row for
    ``snapshot_ref.json``."""
    doc = _score_doc(rows, session=session, tickers=tickers)
    job = JobSpec(kind="legacy_score", implementation_ref="x", spec_hash=None, environment_ref="x",
                 parameters={"expected_ids": ("legacy_score",), "session": session,
                             "tickers": tuple(tickers), "year_start": 2020, "year_end": 2026,
                             "horizon_days": horizon_days, "expected_population": tuple(doc["expected_population"])},
                 output_namespace="shadow", resource_class="legacy_score",
                 retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                       clock=clock, registry=registry())
    score_ref = store.publish_bytes(json.dumps(doc, sort_keys=True).encode(),
                                    schema_ref="legacy_action.v1.0")
    snap_ref_artifact = None
    if snapshot_ref is not None:
        snap_ref_artifact = store.publish_bytes(
            json.dumps(to_document(snapshot_ref), sort_keys=True).encode(),
            schema_ref="snapshot_ref.v1.0")

    def effects(inner_conn):
        register_artifact(inner_conn, score_ref, claim.attempt_id, clock)
        inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                           (claim.attempt_id, "legacy_score", score_ref.artifact_id))
        if snap_ref_artifact is not None:
            register_artifact(inner_conn, snap_ref_artifact, claim.attempt_id, clock)
            record_resolved_bindings(inner_conn, claim.attempt_id, {
                "snapshot_ref.json": ResolvedBinding(
                    name="snapshot_ref.json", binding=snap_ref_artifact.artifact_id,
                    artifact_id=snap_ref_artifact.artifact_id,
                    content_hash=snap_ref_artifact.content_hash)})

    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)
    return job_id_for("shadow", key)


def _world(tmp_path, *, legacy_rows, snapshot_rows, legacy_session="2026-09-10",
          snapshot_session="2026-09-10", bind_snapshot=False):
    conn, clock, supervisor = ops_catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    snap_ref = _snapshot_ref("snap_current") if bind_snapshot else None
    legacy_job = _submit_score_job(conn, clock, supervisor, store, key="legacy",
                                   rows=legacy_rows, session=legacy_session)
    snapshot_job = _submit_score_job(conn, clock, supervisor, store, key="snapshot",
                                     rows=snapshot_rows, session=snapshot_session,
                                     snapshot_ref=snap_ref)
    return conn, store, legacy_job, snapshot_job, snap_ref


def _receipt(conn, store, legacy_job, snapshot_job):
    return score_parity_receipt(conn, store, legacy_job_id=legacy_job, snapshot_job_id=snapshot_job,
                                code_hash=CODE_HASH, environment_hash=ENV_HASH)


# --------------------------------------------------------------------------


def test_identical_rows_agree_with_correct_populations(tmp_path):
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    conn, store, legacy_job, snapshot_job, _ = _world(tmp_path, legacy_rows=rows, snapshot_rows=rows)
    doc = _receipt(conn, store, legacy_job, snapshot_job)
    decoded = decode_document(ComparisonReceipt, doc)
    assert decoded.verdict == AGREE
    assert decoded.population.expected == decoded.population.supported == decoded.population.compared == 1
    assert decoded.findings == ()


def test_float_beyond_tolerance_disagrees_naming_row_and_field(tmp_path):
    row_id = "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16"
    legacy = [_row(row_id, iv30=0.42)]
    snapshot = [_row(row_id, iv30=0.43)]
    conn, store, legacy_job, snapshot_job, _ = _world(tmp_path, legacy_rows=legacy, snapshot_rows=snapshot)
    doc = _receipt(conn, store, legacy_job, snapshot_job)
    decoded = decode_document(ComparisonReceipt, doc)
    assert decoded.verdict == DIFFER
    assert len(decoded.findings) == 1
    finding = decoded.findings[0]
    assert finding.field_path == f"{row_id}::iv30"
    assert finding.kind == "value"
    assert finding.left_value is None and finding.right_value is None  # never values


def test_within_tolerance_agrees(tmp_path):
    """SCORE_RECORD_V1 declares no per-field tolerance yet (exact everywhere),
    so an equal float pair — the only pair that is genuinely "within
    tolerance" today — agrees, on a field distinct from the identical-rows
    test above."""
    row_id = "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16"
    rows = [_row(row_id, iv30=0.4200001)]
    conn, store, legacy_job, snapshot_job, _ = _world(tmp_path, legacy_rows=rows, snapshot_rows=rows)
    doc = _receipt(conn, store, legacy_job, snapshot_job)
    assert doc["verdict"] == AGREE


def test_null_mask_difference_disagrees(tmp_path):
    row_id = "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16"
    legacy = [_row(row_id, iv30=None)]
    snapshot = [_row(row_id, iv30=0.42)]
    conn, store, legacy_job, snapshot_job, _ = _world(tmp_path, legacy_rows=legacy, snapshot_rows=snapshot)
    doc = _receipt(conn, store, legacy_job, snapshot_job)
    decoded = decode_document(ComparisonReceipt, doc)
    assert decoded.verdict == DIFFER
    assert [f.kind for f in decoded.findings] == ["null_mask"]
    assert decoded.findings[0].field_path == f"{row_id}::iv30"


def test_missing_row_on_either_side_disagrees(tmp_path):
    id_x = "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16"
    id_y = "FAKE|TWIN-C|2026-09-10|100.0|2026-10-16"
    legacy = [_row(id_x), _row(id_y)]
    snapshot = [_row(id_x)]
    conn, store, legacy_job, snapshot_job, _ = _world(tmp_path, legacy_rows=legacy, snapshot_rows=snapshot)
    doc = _receipt(conn, store, legacy_job, snapshot_job)
    decoded = decode_document(ComparisonReceipt, doc)
    assert decoded.verdict == DIFFER
    assert decoded.population.expected == 2
    assert decoded.population.supported == 1
    assert decoded.population.compared == 1
    missing = [f for f in decoded.findings if f.kind == "missing_field"]
    assert len(missing) == 1
    assert missing[0].field_path == id_y
    assert missing[0].null_mask_right is True  # absent from the snapshot side


def test_session_mismatch_between_jobs_is_refused(tmp_path):
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    conn, store, legacy_job, snapshot_job, _ = _world(
        tmp_path, legacy_rows=rows, snapshot_rows=rows,
        legacy_session="2026-09-10", snapshot_session="2026-09-11")
    with pytest.raises(OpsError) as err:
        _receipt(conn, store, legacy_job, snapshot_job)
    assert err.value.code == "INPUT_CHANGED"


def test_receipt_binds_snapshot_and_passes_evidence_binding_checks(tmp_path):
    """The envelope binds the snapshot job's bound SnapshotRef id/manifest_hash,
    and the resulting document passes the strict evidence validator's own
    binding checks when placed in an otherwise-minimal valid evidence set."""
    from checks.rearchitecture_phase2_evidence import validate_evidence

    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    conn, store, legacy_job, snapshot_job, snap_ref = _world(
        tmp_path, legacy_rows=rows, snapshot_rows=rows, bind_snapshot=True)
    doc = _receipt(conn, store, legacy_job, snapshot_job)
    assert doc["envelope"]["snapshot_id"] == snap_ref.snapshot_id
    assert doc["envelope"]["snapshot_manifest_hash"] == snap_ref.manifest_hash

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    receipt_bytes = json.dumps(doc).encode()
    (artifacts_dir / "score.json").write_bytes(receipt_bytes)
    import hashlib
    ref = {"path": "score.json", "content_hash": "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()}
    snapshot_bytes = json.dumps(to_document(snap_ref)).encode()
    (artifacts_dir / "snapshot.json").write_bytes(snapshot_bytes)
    snapshot_ref_doc = {"path": "snapshot.json",
                        "content_hash": "sha256:" + hashlib.sha256(snapshot_bytes).hexdigest()}
    evidence = {"schema_version": "phase2_evidence.v1.0", "code_hash": CODE_HASH,
               "environment_hash": ENV_HASH, "authority_mode": "shadow",
               "snapshot_ref": snapshot_ref_doc, "comparison_receipt_ref": ref}
    findings, field_ok, document_ok = validate_evidence(
        evidence, artifact_root=artifacts_dir, code_hash=CODE_HASH, environment_hash=ENV_HASH)
    assert document_ok is True
    assert field_ok.get("comparison_receipt_ref") is True
    bad_codes = {f["code"] for f in findings} & {
        "CODE_HASH_MISMATCH", "ENVIRONMENT_MISMATCH", "SNAPSHOT_BINDING_MISMATCH",
        "ARTIFACT_SHAPE_INVALID", "RECEIPT_KIND_MISMATCH"}
    assert bad_codes == set(), findings
