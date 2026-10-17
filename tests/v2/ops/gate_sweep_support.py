"""Public setup helpers for isolated native sweep collection tests."""
import json
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactStore, SystemClock, canonical_json
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.fingerprints import snapshot_code, worker_source_manifest
from experiments import v2_gate_sweep as sweep
from experiments.native_registration import NativeRegistration
from tests.v2.research.test_gate_report import MONTH, REPO, build_gate_spec


def sweep_request():
    return {"schema_version": "native_gate_sweep_request.v1.0", "primary_comparison": ["candidate", "control"],
            "variants": [json.loads(canonical_json(asdict(build_gate_spec(alpha, arm))))
                         for alpha, arm in ((0.0, "candidate"), (1.0, "control"))]}


def call_sweep(source, action=sweep.register_sweep, document=None, *, code_root=REPO, **kwargs):
    return action(source.conn, source.store, source.repository, document or sweep_request(),
                  code_root=code_root, as_of_month=MONTH, **kwargs)


def sealed_sweep(source):
    return json.loads(source.conn.execute("SELECT document_json FROM native_gate_sweeps").fetchone()[0])


def scientific_rows(source):
    return [tuple(row) for row in source.conn.execute("SELECT * FROM experiment_runs ORDER BY run_id")]


def forbid_effect(*args, **kwargs):
    pytest.fail("refusal crossed the protected read/effect boundary")


def sweep_checkout(tmp_path):
    path = tmp_path / "checkout"
    snapshot_code(REPO, path, worker_source_manifest(REPO))
    for name in ("v2_gate_sweep.py", "native_sweep_collection.py", "v2_gate_collect.py"):
        path.joinpath("experiments", name).write_bytes(REPO.joinpath("experiments", name).read_bytes())
    return path


def replace_checkout_source(path, text):
    """Replace only a disposable test checkout's read-only snapshot copy."""
    path.unlink()
    path.write_text(text)


def terminal_case(tmp_path, *, encoding="UTF-8"):
    """Minimal committed wire data for branch-level terminal-reader tests."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA encoding='{encoding}'")
    conn.executescript("""
        CREATE TABLE attempts (attempt_id TEXT, job_id TEXT, attempt_number INTEGER,
            fence INTEGER, state TEXT, ended_at TEXT, failure_json TEXT);
        CREATE TABLE attempt_outputs (attempt_id TEXT, name TEXT, artifact_id TEXT);
        CREATE TABLE artifacts (artifact_id TEXT PRIMARY KEY, ref_json TEXT,
            producer_attempt_id TEXT, created_at TEXT);
        INSERT INTO attempts VALUES ('attempt-1', 'job-1', 1, 2, 'succeeded', 'ended', NULL);
    """)
    registration = NativeRegistration("native-fixture", canonical_json({"fixture": "registered"}).encode())
    result = {"schema_version": "native_gate_result.v1.0", "variant_id": registration.variant_id,
              "run_id": registration.run_id, "attempted_variants": 1, "provenance": registration.document}
    case = SimpleNamespace(conn=conn, store=ArtifactStore(tmp_path / "objects"), registration=registration,
        job=SimpleNamespace(job_id="job-1", attempt_count=1, fence=2, state="succeeded", failure=None),
        result=result)
    set_terminal_result(case)
    return case


def set_terminal_result(case, *, payload=None, schema="native_gate_result.v1.0"):
    """Replace only a synthetic attempt's output with explicit test bytes."""
    data = canonical_json(case.result).encode() if payload is None else payload
    case.ref = case.store.publish_bytes(data, schema_ref=schema)
    register_artifact(case.conn, case.ref, "attempt-1", SystemClock())
    case.conn.execute("DELETE FROM attempt_outputs")
    case.conn.execute("INSERT INTO attempt_outputs VALUES ('attempt-1', 'native_gate_result', ?)",
                      (case.ref.artifact_id,))
