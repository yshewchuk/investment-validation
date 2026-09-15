"""D15 comparison and receipt construction: ``checks/rearchitecture_phase2_parity.py``.

``build_receipt`` is the real comparator (``compare_records`` under
``SCORE_RECORD_V1``, a real ``ComparisonReceipt``) — tested here directly
against hand-built ``ScoreParityInputs``, with no catalog/job machinery
needed for the comparison scenarios themselves. The CLI tests at the bottom
still drive the full ``load_score_parity_inputs`` -> ``build_receipt`` ->
``publish`` pipeline end to end.
"""
from __future__ import annotations

import hashlib
import json

from checks import rearchitecture_phase2_parity as cli
from checks.rearchitecture_phase1_gate import source_files, source_hash
from checks.rearchitecture_phase2_evidence import validate_evidence
from checks.rearchitecture_phase2_gate import environment_hash
from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.diagnosis import AGREE, DIFFER
from engine.v2.foundation import ArtifactStore, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.parity import ScoreParityInputs
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit
from tests.ops_support import FakeClock, sample
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


def _inputs(legacy_rows, snapshot_rows, *, snapshot_ref=None):
    return ScoreParityInputs(
        legacy_job_id="legacy_job", snapshot_job_id="snapshot_job",
        legacy_rows={r["row_id"]: r for r in legacy_rows},
        snapshot_rows={r["row_id"]: r for r in snapshot_rows}, snapshot_ref=snapshot_ref)


def _build(legacy_rows, snapshot_rows, **kw):
    return cli.build_receipt(_inputs(legacy_rows, snapshot_rows, **kw),
                             code_hash=CODE_HASH, environment_hash=ENV_HASH)


# --------------------------------------------------------------------------
# build_receipt: the real comparator, on hand-built inputs
# --------------------------------------------------------------------------


def test_identical_rows_agree_with_correct_populations():
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    receipt = _build(rows, rows)
    assert receipt.verdict == AGREE
    assert receipt.population.expected == receipt.population.supported == receipt.population.compared == 1
    assert receipt.findings == ()


def test_float_beyond_tolerance_disagrees_naming_row_and_field():
    row_id = "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16"
    receipt = _build([_row(row_id, iv30=0.42)], [_row(row_id, iv30=0.43)])
    assert receipt.verdict == DIFFER
    assert len(receipt.findings) == 1
    finding = receipt.findings[0]
    assert finding.field_path == f"{row_id}::iv30"
    assert finding.kind == "value"
    assert finding.left_value is None and finding.right_value is None  # never values


def test_within_tolerance_agrees():
    """SCORE_RECORD_V1 declares no per-field tolerance yet (exact everywhere),
    so an equal float pair -- the only pair that is genuinely "within
    tolerance" today -- agrees, on a field distinct from the identical-rows
    test above."""
    row_id = "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16"
    rows = [_row(row_id, iv30=0.4200001)]
    assert _build(rows, rows).verdict == AGREE


def test_null_mask_difference_disagrees():
    row_id = "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16"
    receipt = _build([_row(row_id, iv30=None)], [_row(row_id, iv30=0.42)])
    assert receipt.verdict == DIFFER
    assert [f.kind for f in receipt.findings] == ["null_mask"]
    assert receipt.findings[0].field_path == f"{row_id}::iv30"


def test_missing_row_on_either_side_disagrees():
    id_x = "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16"
    id_y = "FAKE|TWIN-C|2026-09-10|100.0|2026-10-16"
    receipt = _build([_row(id_x), _row(id_y)], [_row(id_x)])
    assert receipt.verdict == DIFFER
    assert receipt.population.expected == 2
    assert receipt.population.supported == 1
    assert receipt.population.compared == 1
    missing = [f for f in receipt.findings if f.kind == "missing_field"]
    assert len(missing) == 1
    assert missing[0].field_path == id_y
    assert missing[0].null_mask_right is True  # absent from the snapshot side


def test_receipt_binds_snapshot_and_passes_evidence_binding_checks(tmp_path):
    """The envelope binds the snapshot job's bound SnapshotRef id/manifest_hash,
    and the resulting document passes the strict evidence validator's own
    binding checks when placed in an otherwise-minimal valid evidence set."""
    snap_ref = _snapshot_ref("snap_current")
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    receipt = _build(rows, rows, snapshot_ref=snap_ref)
    assert receipt.envelope.snapshot_id == snap_ref.snapshot_id
    assert receipt.envelope.snapshot_manifest_hash == snap_ref.manifest_hash

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    receipt_bytes = json.dumps(to_document(receipt)).encode()
    (artifacts_dir / "score.json").write_bytes(receipt_bytes)
    score_ref = {"path": "score.json",
                "content_hash": "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()}
    snapshot_bytes = json.dumps(to_document(snap_ref)).encode()
    (artifacts_dir / "snapshot.json").write_bytes(snapshot_bytes)
    snapshot_ref_doc = {"path": "snapshot.json",
                        "content_hash": "sha256:" + hashlib.sha256(snapshot_bytes).hexdigest()}
    evidence = {"schema_version": "phase2_evidence.v1.0", "code_hash": CODE_HASH,
               "environment_hash": ENV_HASH, "authority_mode": "shadow",
               "snapshot_ref": snapshot_ref_doc, "comparison_receipt_ref": score_ref}
    findings, field_ok, document_ok = validate_evidence(
        evidence, artifact_root=artifacts_dir, code_hash=CODE_HASH, environment_hash=ENV_HASH)
    assert document_ok is True
    assert field_ok.get("comparison_receipt_ref") is True
    bad_codes = {f["code"] for f in findings} & {
        "CODE_HASH_MISMATCH", "ENVIRONMENT_MISMATCH", "SNAPSHOT_BINDING_MISMATCH",
        "ARTIFACT_SHAPE_INVALID", "RECEIPT_KIND_MISMATCH"}
    assert bad_codes == set(), findings


# --------------------------------------------------------------------------
# CLI: the full load -> compare -> publish pipeline
# --------------------------------------------------------------------------


def _ops_catalog(root):
    """Like ``tests.ops_support.catalog``, but at the ``catalog.sqlite`` path
    ``checks/rearchitecture_phase2_parity.py`` (and ``engine/v2/ops/cli.py``)
    actually open under ``--root``."""
    clock = FakeClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    return conn, clock, Supervisor(epoch, "boot")


def _submit(conn, clock, supervisor, store, key, session, iv30, input_mode="legacy"):
    row = {"row_id": "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16", "ticker": "FAKE",
          "strategy": "TWIN-P", "event_date": "2026-09-10", "strike": 100.0,
          "expiry": "2026-10-16", "iv30": iv30}
    doc = {"rows": [row], "expected_population": [row["row_id"]],
          "observed_population": [row["row_id"]], "ladder": {}, "tickers": ["FAKE"],
          "analog_entry_coverage": {}, "session": session, "requested_session": session}
    parameters = {"expected_ids": ("legacy_score",), "session": session,
                 "tickers": ("FAKE",), "year_start": 2020, "year_end": 2026,
                 "expected_population": (row["row_id"],), "input_mode": input_mode}
    if input_mode == "snapshot":
        # input_mode_problems (stages.py) requires all three snapshot
        # bindings before admitting a snapshot-mode legacy_score job.
        parameters["input_bindings"] = {"snapshot_ref.json": "snap#ref",
                                        "materialization_request.json": "snap#request",
                                        "materialization_manifest.json": "snap#manifest"}
    job = JobSpec(kind="legacy_score", implementation_ref="x", spec_hash=None, environment_ref="x",
                 parameters=parameters, output_namespace="shadow", resource_class="legacy_score",
                 retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                       clock=clock, registry=registry())
    ref = store.publish_bytes(json.dumps(doc, sort_keys=True).encode(), schema_ref="legacy_action.v1.0")

    def effects(inner_conn):
        register_artifact(inner_conn, ref, claim.attempt_id, clock)
        inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                           (claim.attempt_id, "legacy_score", ref.artifact_id))

    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)
    return job_id_for("shadow", key)


def _root(tmp_path):
    conn, clock, supervisor = _ops_catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    legacy = _submit(conn, clock, supervisor, store, "legacy", "2026-09-10", 0.42,
                     input_mode="legacy")
    snapshot = _submit(conn, clock, supervisor, store, "snapshot", "2026-09-10", 0.42,
                       input_mode="snapshot")
    conn.close()
    return tmp_path, legacy, snapshot


def test_cli_agree_exits_zero_and_prints_verdict(tmp_path, capsys):
    root, legacy_job, snapshot_job = _root(tmp_path)
    artifact_root = tmp_path / "artifacts"
    code = cli.main(["--root", str(root), "--legacy-job", legacy_job,
                     "--snapshot-job", snapshot_job, "--artifact-root", str(artifact_root)])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "agree"
    published = artifact_root / out["path"]
    assert published.is_file()
    assert out["content_hash"].startswith("sha256:")
    code_hash = source_hash(source_files(cli.ROOT))
    env_hash, _ = environment_hash(cli.ROOT)
    payload = json.loads(published.read_text())
    assert payload["envelope"]["code_hash"] == code_hash
    assert payload["envelope"]["environment_hash"] == env_hash


def test_cli_refused_session_mismatch_exits_nonzero(tmp_path, capsys):
    conn, clock, supervisor = _ops_catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    legacy = _submit(conn, clock, supervisor, store, "legacy", "2026-09-10", 0.42)
    snapshot = _submit(conn, clock, supervisor, store, "snapshot", "2026-09-11", 0.42)
    conn.close()
    code = cli.main(["--root", str(tmp_path), "--legacy-job", legacy, "--snapshot-job", snapshot,
                     "--artifact-root", str(tmp_path / "artifacts")])
    assert code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["refused"] == "INPUT_CHANGED"
