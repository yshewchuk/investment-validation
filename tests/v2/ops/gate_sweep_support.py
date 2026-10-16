"""Public setup helpers for isolated native sweep collection tests."""
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactStore, SystemClock, canonical_json
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.fingerprints import snapshot_code, source_closure, worker_source_manifest
from engine.v2.ops.migrations import migrate
from engine.v2.ops.schema import MIGRATIONS, OWNER
from engine.v2.ops.submission import submit
from experiments import native_registration, native_sweep_collection
from experiments import v2_gate_sweep as sweep
from experiments.native_registration import NativeRegistration
from tests.ops_support import POLICY, REGISTRY, request
from tests.v2.research.test_gate_report import MONTH, REPO, build_gate_spec


@pytest.fixture(scope="session")
def fixed_sweep_code_maps():
    """Real source maps, once per worker, for this unchanged test checkout."""
    entries = (("experiments/v2_gate.py",), ("experiments/v2_gate_sweep.py",),
               ("experiments/native_sweep_collection.py", "experiments/v2_gate_collect.py"))
    return {entry: source_closure(REPO, entry) for entry in entries}, worker_source_manifest(REPO)


@pytest.fixture
def fixed_sweep_code(monkeypatch, fixed_sweep_code_maps):
    """Opt-in source inputs for synthetic receipt tests, not source-discovery tests.

    Only the real, unchanged repository and declared entries are supported. The
    real-child and source-change/read-error tests never request this fixture.
    Admission, submissions, catalog/artifact reads and publication remain real.
    """
    closures, manifest = fixed_sweep_code_maps

    def closure(root, entries):
        assert Path(root) == REPO, "fixed source fixture received another checkout"
        entries = tuple(entries)
        assert entries in closures, "fixed source fixture received unknown entrypoints"
        return dict(closures[entries])

    def worker_manifest(root):
        assert Path(root) == REPO, "fixed source fixture received another checkout"
        return dict(manifest)

    with monkeypatch.context() as patch:
        for module in (sweep, native_registration, native_sweep_collection):
            patch.setattr(module, "source_closure", closure)
        patch.setattr(sweep, "worker_source_manifest", worker_manifest)
        yield


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


def submitted_job(tmp_path, *, encoding="UTF-8", strict=True):
    """A real submitted ops job, optionally with a damaged legacy jobs table."""
    conn = sqlite3.connect(tmp_path / "job.sqlite", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA encoding='{encoding}'")
    clock = SystemClock()
    migrate(conn, OWNER, MIGRATIONS, clock=clock)
    if not strict:
        # Current STRICT TEXT rejects BLOBs; model a malformed older catalog.
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name='jobs'").fetchone()[0]
        conn.execute("DROP TABLE jobs")
        conn.execute(ddl.removesuffix("STRICT"))
    document = request()
    receipt = submit(conn, REGISTRY, POLICY, document, clock=clock)
    return SimpleNamespace(conn=conn, clock=clock, request=document, job_id=receipt.job_id)
