"""D15 CLI wrapper: ``checks/rearchitecture_phase2_parity.py``."""
from __future__ import annotations

import json

from checks import rearchitecture_phase2_parity as cli
from checks.rearchitecture_phase1_gate import source_files, source_hash
from checks.rearchitecture_phase2_gate import environment_hash
from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit
from tests.ops_support import FakeClock, sample

POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _ops_catalog(root):
    """Like ``tests.ops_support.catalog``, but at the ``catalog.sqlite`` path
    ``checks/rearchitecture_phase2_parity.py`` (and ``engine/v2/ops/cli.py``)
    actually open under ``--root``."""
    clock = FakeClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    return conn, clock, Supervisor(epoch, "boot")


def _submit(conn, clock, supervisor, store, key, session, iv30):
    row = {"row_id": "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16", "ticker": "FAKE",
          "strategy": "TWIN-P", "event_date": "2026-09-10", "strike": 100.0,
          "expiry": "2026-10-16", "iv30": iv30}
    doc = {"rows": [row], "expected_population": [row["row_id"]],
          "observed_population": [row["row_id"]], "ladder": {}, "tickers": ["FAKE"],
          "analog_entry_coverage": {}, "session": session, "requested_session": session}
    job = JobSpec(kind="legacy_score", implementation_ref="x", spec_hash=None, environment_ref="x",
                 parameters={"expected_ids": ("legacy_score",), "session": session,
                             "tickers": ("FAKE",), "year_start": 2020, "year_end": 2026,
                             "expected_population": (row["row_id"],)},
                 output_namespace="shadow", resource_class="legacy_score",
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
    legacy = _submit(conn, clock, supervisor, store, "legacy", "2026-09-10", 0.42)
    snapshot = _submit(conn, clock, supervisor, store, "snapshot", "2026-09-10", 0.42)
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
