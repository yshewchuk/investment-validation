"""P2-6 §9.3: snapshot-backed legacy stages in the Phase 1 supervisor (D13/D14).

Real SQLite catalog, real ArtifactStore, real ``Service`` under TEST_POLICY,
and a real committed synthetic snapshot (the six read-plan tables from
``tests/test_v2_data_legacy_materialization.py``). The ``legacy_materialize``
kind runs its real worker subprocess end to end.

Worker-side seam for the scoring stage: the real legacy scorer cannot run on
this synthetic tree (no registered champions, no model artifacts), so the
snapshot-backed ``legacy_score`` launch swaps ONLY the argv of the executor's
``subprocess.Popen`` for a tiny recording stub (``_STUB``) — same env, same
stdin envelope, same result pipe, same Service finish path. The seam lives in
this file; the production registry and worker are untouched.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from engine.v2.contracts import JobSpec, LegacyFileRef, LegacyInputManifest, SubmitRequest
from engine.v2.data import catalog as data_catalog
from engine.v2.data import manifests
from engine.v2.data.objects import partition_logical_hash
from engine.v2.data.reference_catalog import (
    ReferenceInput,
    insert_reference_inputs,
    reference_inputs_for_snapshot,
)
from engine.v2.data.reference_inputs import LEGACY_REFERENCE_INPUTS_V1
from engine.v2.data.repository import Repository
from engine.v2.foundation import SystemClock, content_hash, format_timestamp, to_document
from engine.v2.ops import executor
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import cache_identity, register_artifact
from engine.v2.ops.cli import dispatch, parser
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.input_bindings import recorded_bindings, resolved_inputs_hash
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.snapshot_roots import default_materialization_base, hash_tree, materialization_root
from engine.v2.ops.snapshot_stages import snapshot_cache_inputs
from engine.v2.ops.snapshots import resolve_snapshot_head
from engine.v2.ops.stages import BARRIER_ONLY_REASONS, SNAPSHOT_BACKED_KINDS, registry
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.data_scan_support import RECEIPT, contract_for, fake_hash, publish_and_inspect
from tests.ops_support import TEST_POLICY
from tests.test_v2_data_legacy_materialization import (
    _EE_COMMON,
    PANEL_ROWS,
    TABLES,
    _build_request,
    _build_snapshot,
    _ref,
    _snapshot_object_ref,
)

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
SESSION = "2026-09-12"
MANIFEST = "materialization_manifest"

_STUB = r"""
import json, os, sys
envelope = json.loads(sys.stdin.buffer.readline())
staging = envelope["staging"]
record = {"envelope": envelope, "env_root": os.environ.get("INVESTING_PLAN_ROOT"),
          "staging_legacy_exists": os.path.lexists(os.path.join(staging, "legacy"))}
with open(os.path.join(RECORD_DIR, envelope["attempt_id"] + ".json"), "w") as fh:
    json.dump(record, fh)
with open(os.path.join(staging, "score.json"), "w") as fh:
    json.dump({"rows": [], "stub": True}, fh)
result = {"schema_version": "worker_result.v1.0", "job_id": envelope["job_id"],
          "attempt_id": envelope["attempt_id"], "fence": envelope["fence"],
          "outputs": [{"name": envelope["worker"], "path": "score.json",
                       "schema": "legacy_action.v1.0"}],
          "completed_ids": list(envelope["parameters"]["expected_ids"])}
os.write(int(envelope["result_fd"]), json.dumps(result).encode() + b"\n")
"""


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


class Case:
    """One synthetic snapshot committed into the Service's own catalog/store."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.conn, self.store, self.snap = _build_snapshot(tmp_path)
        self.root = Path(self.store.root)
        self.live_store = tmp_path / "prod"
        self.live_store.mkdir()
        self.clock = SystemClock()
        self.repository = Repository(self.conn, self.store)
        self.snapshot_object = _snapshot_object_ref(self.store)
        self.request = _build_request(self.repository, self.snap, self.snapshot_object, self.store)
        record_reference_inputs(self, self.snap.snapshot_id, "r1-references")
        self.snapshot_ref = resolve_snapshot_head(self.conn, self.store, "shadow", clock=self.clock)
        self.request_ref = self.publish(to_document(self.request), "legacy_materialization_request.v1.0")
        self.base = default_materialization_base(self.root)
        self.records = tmp_path / "records"
        self.records.mkdir()

    def publish(self, document, schema_ref):
        ref = self.store.publish_bytes(json.dumps(document, sort_keys=True).encode(),
                                       schema_ref=schema_ref)
        with transaction(self.conn):
            register_artifact(self.conn, ref, None, self.clock)
        return ref

    def service(self):
        return Service(self.conn, self.root, registry(), TEST_POLICY, clock=self.clock,
                       code_source=REPO, store_root=self.live_store)

    def run(self, job_id, timeout=90):
        service = self.service()
        service.start()
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                service.tick()
                state = self.state(job_id)
                if state in ("succeeded", "failed", "blocked", "cancelled"):
                    return state
                time.sleep(0.05)
            return self.state(job_id)
        finally:
            service.close()

    def state(self, job_id):
        return self.conn.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]

    def failure(self, job_id):
        return self.conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                                 (job_id,)).fetchone()[0] or ""

    def attempts(self, job_id):
        return [row[0] for row in self.conn.execute(
            "SELECT attempt_id FROM attempts WHERE job_id=? ORDER BY attempt_number", (job_id,))]

    def output(self, job_id, name):
        attempt = self.attempts(job_id)[-1]
        return self.conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=? "
                                 "AND name=?", (attempt, name)).fetchone()[0]

    def submit(self, kind, parameters, input_refs, key, resource, contract, deps=()):
        profile = profile_named(DEFAULT_POLICY, resource)
        job = JobSpec(
            kind=kind, implementation_ref=content_hash(worker_source_manifest(REPO)), spec_hash=None,
            environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
            parameters=parameters, input_refs=tuple(input_refs), dependency_job_ids=tuple(deps),
            output_namespace="shadow", resource_class=resource, retry_policy_ref="bounded",
            checkpoint_contract_ref=contract)
        return submit(self.conn, registry(), POLICY, SubmitRequest(
            namespace="shadow", idempotency_key=key, principal="operator", job=job),
            clock=self.clock).job_id

    def submit_materialize(self, key, request_ref=None):
        request_ref = request_ref or self.request_ref
        bindings = {"snapshot_ref.json": self.snapshot_ref.artifact_id,
                    "materialization_request.json": request_ref.artifact_id}
        return self.submit("legacy_materialize",
                           {"expected_ids": ["legacy_materialize"], "input_bindings": bindings,
                            "scratch_estimate_bytes": 1 << 20},
                           (self.snapshot_ref.artifact_id, request_ref.artifact_id), key,
                           "materialize", "legacy_materialization_manifest.v1.0")

    def submit_score(self, key, manifest_binding, deps=(), snapshot_ref=None, request_ref=None,
                     extra_refs=()):
        snapshot_ref = snapshot_ref or self.snapshot_ref
        request_ref = request_ref or self.request_ref
        bindings = {"snapshot_ref.json": snapshot_ref.artifact_id,
                    "materialization_request.json": request_ref.artifact_id,
                    "materialization_manifest.json": manifest_binding}
        parameters = {"expected_ids": ["legacy_score"], "session": SESSION,
                      "tickers": ["AAA", "BBB"], "year_start": 2020, "year_end": 2021,
                      "expected_population": ["AAA|S1|2020-01-15"], "input_mode": "snapshot",
                      "input_bindings": bindings}
        refs = (snapshot_ref.artifact_id, request_ref.artifact_id, *extra_refs)
        return self.submit("legacy_score", parameters, refs, key, "legacy_score",
                           "legacy_action.v1.0", deps)

    def install_stub(self, monkeypatch):
        real = subprocess.Popen
        source = "RECORD_DIR = " + repr(str(self.records)) + "\n" + _STUB

        def popen(args, **kwargs):
            if list(args[-2:]) == ["-m", "engine.v2.ops.worker"]:
                args = [sys.executable, "-u", "-c", source]
            return real(args, **kwargs)

        monkeypatch.setattr(executor.subprocess, "Popen", popen)

    def recorded(self):
        return [json.loads(path.read_text()) for path in sorted(self.records.glob("*.json"))]


def record_reference_inputs(case, snapshot_id, receipt_id, *, registry=b'{"models": []}'):
    """A committed import receipt for ``snapshot_id`` plus its reference rows, in one
    transaction, as the snapshot import coordinator records them. The default bytes
    equal ``_pinned_refs``'s, so planning rebuilds exactly ``case.request``."""
    inputs = LEGACY_REFERENCE_INPUTS_V1["inputs"]
    published = (("calendar", b"date\n2020-01-02\n2021-01-04\n"), ("model_registry", registry))
    rows = []
    for kind, data in published:
        ref = case.store.publish_bytes(data, schema_ref="legacy_pinned_ref.v1")
        rows.append(ReferenceInput(kind=kind, legacy_path=inputs[kind]["path"], object_id=ref.artifact_id,
                                   content_hash=ref.content_hash, byte_size=ref.byte_size))
    obj = case.snapshot_object
    rows.append(ReferenceInput(kind="legacy_snapshot", legacy_path=inputs["legacy_snapshot"]["path"],
                               object_id=obj.object_id, content_hash=obj.content_hash,
                               byte_size=obj.byte_size))
    with transaction(case.conn):
        case.conn.execute(
            "INSERT INTO data_import_receipts (receipt_id, attempt_id, fence, source_manifest_hash, "
            "result_snapshot_id, status, registered_at, scope) VALUES (?, ?, 1, ?, ?, 'committed', ?, "
            "'shadow')", (receipt_id, "att-" + receipt_id, fake_hash(receipt_id), snapshot_id,
                          format_timestamp(case.clock.now())))
        insert_reference_inputs(case.conn, receipt_id, rows)


@pytest.fixture
def case(tmp_path):
    value = Case(tmp_path)
    try:
        yield value
    finally:
        value.conn.close()


def _modes(root):
    for path in [root, *root.rglob("*")]:
        mode = stat.S_IMODE(path.lstat().st_mode)
        assert not path.is_symlink()
        assert mode == (0o555 if path.is_dir() else 0o444), path


def _stats(root):
    return {str(p.relative_to(root)): (p.stat().st_ino, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def _tamper(root):
    target = next(p for p in sorted(root.rglob("*.parquet")))
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b"tampered")
    target.chmod(0o444)


# --------------------------------------------------------------------------
# legacy_materialize
# --------------------------------------------------------------------------


def test_materialize_job_writes_read_only_root_and_reuse_never_rewrites(case):
    first = case.submit_materialize("mat-1")
    assert case.run(first) == "succeeded", case.failure(first)

    root = materialization_root(case.base, case.request.request_hash)
    assert not str(root).startswith(str(case.root) + os.sep)
    _modes(root)
    manifest_id = case.output(first, MANIFEST)
    ref_row = case.conn.execute("SELECT ref_json FROM artifacts WHERE artifact_id=?",
                                (manifest_id,)).fetchone()[0]
    manifest = json.loads(case.store.read_verified(
        __import__("engine.v2.ops.checkpoints", fromlist=["artifact"]).artifact(
            case.conn, case.store, manifest_id)))
    assert json.loads(ref_row)["schema_ref"] == "legacy_materialization_manifest.v1.0"
    assert manifest["request_hash"] == case.request.request_hash
    assert manifest["files"] == hash_tree(root)
    before = _stats(root)

    second = case.submit_materialize("mat-2")
    assert case.run(second) == "succeeded", case.failure(second)
    assert _stats(root) == before
    assert case.output(second, MANIFEST) == manifest_id
    attempts = case.attempts(first) + case.attempts(second)
    marks = ",".join("?" * len(attempts))
    assert case.conn.execute(f"SELECT COUNT(*) FROM store_read_pins WHERE attempt_id IN ({marks})",
                             attempts).fetchone()[0] == 0
    assert case.conn.execute(f"SELECT COUNT(*) FROM store_leases WHERE attempt_id IN ({marks})",
                             attempts).fetchone()[0] == 0
    assert not list(case.base.glob(".*partial*"))


def test_tampered_root_refuses_next_stage_launch_and_rematerialization(case, monkeypatch):
    first = case.submit_materialize("mat-1")
    assert case.run(first) == "succeeded", case.failure(first)
    _tamper(materialization_root(case.base, case.request.request_hash))

    case.install_stub(monkeypatch)
    stage = case.submit_score("score-1", first + "#" + MANIFEST, deps=(first,))
    assert case.run(stage) == "failed"
    assert "INPUT_CHANGED" in case.failure(stage)
    assert case.recorded() == []

    monkeypatch.undo()  # the real legacy_materialize worker runs again
    again = case.submit_materialize("mat-2")
    assert case.run(again) == "failed"
    assert "INPUT_CHANGED" in case.failure(again)


# --------------------------------------------------------------------------
# snapshot-backed stage launch
# --------------------------------------------------------------------------


def test_snapshot_backed_score_launches_on_the_materialization_root_only(case, monkeypatch):
    first = case.submit_materialize("mat-1")
    assert case.run(first) == "succeeded", case.failure(first)
    root = materialization_root(case.base, case.request.request_hash)

    case.install_stub(monkeypatch)
    stage = case.submit_score("score-1", first + "#" + MANIFEST, deps=(first,))
    assert case.run(stage) == "succeeded", case.failure(stage)
    (record,) = case.recorded()
    attempt = case.attempts(stage)[0]
    staging = case.store.staging_dir(attempt)
    assert record["envelope"]["legacy_root"] == str(root)
    assert record["env_root"] == str(root)
    assert record["envelope"]["legacy_root"] not in (str(case.live_store), str(staging / "legacy"))
    assert "materialization" not in record["envelope"]
    assert record["staging_legacy_exists"] is False
    assert case.conn.execute("SELECT COUNT(*) FROM store_read_pins WHERE attempt_id=?",
                             (attempt,)).fetchone()[0] == 0
    assert case.conn.execute("SELECT COUNT(*) FROM store_leases WHERE attempt_id=?",
                             (attempt,)).fetchone()[0] == 0

    recorded = recorded_bindings(case.conn, attempt)
    manifest_id = case.output(first, MANIFEST)
    assert recorded["snapshot_ref.json"].artifact_id == case.snapshot_ref.artifact_id
    assert recorded["materialization_request.json"].artifact_id == case.request_ref.artifact_id
    assert recorded["materialization_manifest.json"].artifact_id == manifest_id
    spec = case.conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (stage,)).fetchone()[0]
    spec = __import__("engine.v2.ops.catalog", fromlist=["load_json"]).load_json(JobSpec, spec)
    expected_key = cache_identity(
        kind="legacy_score",
        inputs=snapshot_cache_inputs(
            resolved_inputs_hash(spec, recorded), snapshot_manifest_hash=case.snap.manifest_hash,
            request_hash=case.request.request_hash,
            manifest_content_hash=recorded["materialization_manifest.json"].content_hash),
        implementation=spec.implementation_ref, parameters=content_hash(spec.parameters),
        environment=spec.environment_ref, schema="legacy_action.v1.0", shard="default")
    assert case.conn.execute("SELECT COUNT(*) FROM checkpoints WHERE cache_key=?",
                             (expected_key,)).fetchone()[0] == 1

    # Retry with identical inputs: same key, checkpoint reused, no second worker.
    retry = case.submit_score("score-2", first + "#" + MANIFEST, deps=(first,))
    assert case.run(retry) == "succeeded", case.failure(retry)
    assert len(case.recorded()) == 1
    assert case.output(retry, "0") == case.output(stage, "0")


def _other_snapshot_ref(case):
    contract = contract_for("feature_panel")
    record = publish_and_inspect(case.store, contract, _ref(contract), PANEL_ROWS[:1],
                                 partition_key="all")
    from tests.data_scan_support import commit_tables
    commit_tables(case.conn, case.clock, {"feature_panel": [record]}, {"feature_panel": contract},
                  scope="other", receipt_id="r-other", attempt_id="att-other", store=case.store)
    return resolve_snapshot_head(case.conn, case.store, "other", clock=case.clock)


def test_request_built_for_another_snapshot_is_refused_before_launch(case, monkeypatch):
    other = _other_snapshot_ref(case)
    dummy = case.publish({"schema_version": "legacy_materialization_manifest.v1.0"},
                         "legacy_materialization_manifest.v1.0")
    case.install_stub(monkeypatch)
    stage = case.submit_score("score-mismatch", dummy.artifact_id, snapshot_ref=other,
                              extra_refs=(dummy.artifact_id,))
    assert case.run(stage) == "failed"
    assert "INPUT_CHANGED" in case.failure(stage)
    assert "another snapshot" in case.failure(stage)
    assert case.recorded() == []
    attempt = case.attempts(stage)[0]
    assert case.conn.execute("SELECT COUNT(*) FROM store_read_pins WHERE attempt_id=?",
                             (attempt,)).fetchone()[0] == 0


def test_incomplete_read_plan_is_refused_before_launch(case, monkeypatch):
    narrow = _build_request(case.repository, case.snap, case.snapshot_object, case.store,
                            direct_scope={"tickers": ["AAA", "CCC"], "years": [2020]})
    narrow_ref = case.publish(to_document(narrow), "legacy_materialization_request.v1.0")
    dummy = case.publish({"schema_version": "legacy_materialization_manifest.v1.0"},
                         "legacy_materialization_manifest.v1.0")
    case.install_stub(monkeypatch)
    stage = case.submit_score("score-incomplete", dummy.artifact_id, request_ref=narrow_ref,
                              extra_refs=(dummy.artifact_id,))
    assert case.run(stage) == "failed"
    assert "not complete" in case.failure(stage)
    assert case.recorded() == []


def test_uncommitted_manifest_is_refused_even_when_the_root_matches(case, monkeypatch):
    first = case.submit_materialize("mat-1")
    assert case.run(first) == "succeeded", case.failure(first)
    root = materialization_root(case.base, case.request.request_hash)
    # Same claims, different bytes: a manifest no legacy_materialize attempt committed.
    document = {"schema_version": "legacy_materialization_manifest.v1.0",
                "request_hash": case.request.request_hash, "snapshot_id": case.snap.snapshot_id,
                "snapshot_manifest_hash": case.snap.manifest_hash, "files": hash_tree(root)}
    forged = case.store.publish_bytes(json.dumps(document, indent=1).encode(),
                                      schema_ref="legacy_materialization_manifest.v1.0")
    with transaction(case.conn):
        register_artifact(case.conn, forged, None, case.clock)
    assert forged.artifact_id != case.output(first, MANIFEST)
    case.install_stub(monkeypatch)
    stage = case.submit_score("score-forged", forged.artifact_id, extra_refs=(forged.artifact_id,))
    assert case.run(stage) == "failed"
    assert "not committed" in case.failure(stage)
    assert case.recorded() == []


def test_snapshot_input_mode_is_refused_for_kinds_without_a_read_plan(case):
    assert SNAPSHOT_BACKED_KINDS == {"legacy_score", "legacy_score_requests",
                                     "legacy_decision_replay"}
    assert set(BARRIER_ONLY_REASONS) == {"legacy_finality", "legacy_model_evidence",
                                         "legacy_selfcheck"}
    bindings = {"snapshot_ref.json": case.snapshot_ref.artifact_id,
                "materialization_request.json": case.request_ref.artifact_id,
                "materialization_manifest.json": case.request_ref.artifact_id}
    with pytest.raises(Exception) as err:
        case.submit("legacy_finality", {"expected_ids": ["legacy_finality"], "session": SESSION,
                                        "input_mode": "snapshot", "input_bindings": bindings},
                    (case.snapshot_ref.artifact_id, case.request_ref.artifact_id), "fin",
                    "validation", "legacy_action.v1.0")
    assert "INVALID_REQUEST" in str(getattr(err.value, "problem", err.value))


# --------------------------------------------------------------------------
# checkpoint identity
# --------------------------------------------------------------------------


def test_snapshot_checkpoint_key_covers_each_snapshot_input():
    base = dict(snapshot_manifest_hash=fake_hash("manifest"), request_hash=fake_hash("request"),
                manifest_content_hash=fake_hash("tree"))

    def key(**changes):
        inputs = snapshot_cache_inputs(fake_hash("bindings"), **{**base, **changes})
        return cache_identity(kind="legacy_score", inputs=inputs, implementation="impl",
                              parameters="params", environment="env",
                              schema="legacy_action.v1.0", shard="default")

    reference = key()
    assert key() == reference
    assert key(snapshot_manifest_hash=fake_hash("manifest-2")) != reference
    assert key(request_hash=fake_hash("request-2")) != reference
    assert key(manifest_content_hash=fake_hash("tree-2")) != reference


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def _plan_files(case):
    from engine.v2.data.legacy_nightly_read_plan import NIGHTLY_CAPTURE_IMPLEMENTATION_REF
    from engine.v2.ops.fingerprints import file_hash

    fixture = case.live_store / "unused.txt"
    fixture.write_bytes(b"barrier read set")
    manifest = case.tmp / "legacy_manifest.json"
    # These barrier-only-kind jobs (legacy_finality/decisions/...) are
    # produced by the plan but never run in this file's tests (only
    # legacy_materialize runs, or nothing runs at all) -- the file_refs
    # below are placeholder PATHS satisfying the capture-inputs plan-time
    # guard's presence check, not files any worker here actually reads.
    tables = ("daily_market", "option_chains", "earnings_events", "trades")
    file_refs = tuple(
        LegacyFileRef(path=f"data/curated/{table}/year=2024/part-0000.parquet",
                      content_hash=file_hash(fixture), byte_size=fixture.stat().st_size)
        for table in tables
    ) + (LegacyFileRef(path="data/raw/fetch/orats/ab/placeholder.meta.json",
                       content_hash=file_hash(fixture), byte_size=fixture.stat().st_size),
        LegacyFileRef(path="unused.txt", content_hash=file_hash(fixture),
                      byte_size=fixture.stat().st_size))
    manifest.write_text(json.dumps(to_document(LegacyInputManifest(
        manifest_id="m1", file_refs=file_refs,
        table_contract_refs=(), registry_and_model_refs=("placeholder::sha256:" + "0" * 64,),
        calendar_ref="placeholder::sha256:" + "0" * 64,
        selected_session=SESSION, finality_receipt_refs=(), knowledge_mode_by_table={},
        availability_evidence_refs=(), read_set_complete=True,
        capture_implementation_ref=NIGHTLY_CAPTURE_IMPLEMENTATION_REF))))
    population = case.tmp / "population.json"
    population.write_text(json.dumps(["AAA|S1|2020-01-15"]))
    return ["plan", "nightly", "--as-of", SESSION, "--input-mode", "snapshot",
            "--snapshot-scope", "shadow",
            "--input-manifest", str(manifest), "--expected-population", str(population),
            "--tickers", "AAA,BBB", "--year-start", "2020", "--year-end", "2021"]


def _submitted_specs(case, plan_ref, key):
    result = dispatch(parser().parse_args(["submit", "--plan", plan_ref, "--idempotency-key", key]),
                      case.root, case.conn, case.clock)
    ids = [job["job_id"] for job in result["jobs"]]
    rows = {row[0]: json.loads(row[1]) for row in case.conn.execute(
        f"SELECT job_id, spec_json FROM jobs WHERE job_id IN ({','.join('?' * len(ids))})", ids)}
    return ids, rows


def _advance_head(case):
    repository = Repository(case.conn, case.store)
    records = {name: list(repository.fragment_records(case.snap, name)) for name in TABLES}
    contracts = {name: contract_for(name) for name in TABLES}
    ee = contracts["earnings_events"]
    records["earnings_events"].append(publish_and_inspect(case.store, ee, _ref(ee), [dict(
        event_id="EE3", ticker="AAA", event_date=datetime(2022, 3, 1), year=2022, session="AMC",
        **_EE_COMMON)], partition_key="2022"))
    dm = contracts["daily_market"]
    dm_2020 = [r for r in records["daily_market"] if r.partition_key == "2020"]
    hashes = {"daily_market": {"2020": partition_logical_hash(
        case.store, [r.object_ref for r in dm_2020], dm, _ref(dm), "2020")}}
    table_manifests = {name: manifests.dataset_manifest(
        _ref(contracts[name]), recs, knowledge_mode="reconstructed", coverage_receipt_refs=(RECEIPT,),
        availability_evidence_refs=(), partition_logical_hashes=hashes.get(name))
        for name, recs in records.items()}
    snap = manifests.snapshot_ref(table_manifests, calendar_version="cal.v1",
                                  source_priority_version="prio.v1", finality_receipt_refs=(RECEIPT,),
                                  parent_snapshot_id=case.snap.snapshot_id)
    head = case.conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads "
                             "WHERE scope='shadow'").fetchone()
    data_catalog.commit_snapshot(
        case.conn, scope="shadow", request_hash=fake_hash("advance"), contracts=list(contracts.values()),
        objects=[r.object_ref for recs in records.values() for r in recs],
        records=[r for recs in records.values() for r in recs], manifests=list(table_manifests.values()),
        snapshot=snap, expected_head_snapshot_id=head[0], expected_head_generation=head[1],
        receipt_id="r-advance", attempt_id="att-advance", fence=1, fence_check=lambda _c: None,
        clock=case.clock, store=case.store)
    return snap.snapshot_id


def test_snapshot_plan_pins_head_once_and_binds_all_three_artifacts(case, monkeypatch):
    import engine.v2.ops.snapshot_planning as planning
    calls = []
    real = planning.resolve_snapshot_head
    monkeypatch.setattr(planning, "resolve_snapshot_head",
                        lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    argv = _plan_files(case)
    planned = dispatch(parser().parse_args(argv), case.root, case.conn, case.clock)
    assert calls == ["shadow"]
    inputs = planned["plan"]["snapshot_inputs"]
    assert inputs["snapshot_ref_artifact_id"] == case.snapshot_ref.artifact_id
    assert inputs["snapshot_manifest_hash"] == case.snap.manifest_hash
    # P2-C03: pin_snapshot_inputs widens the evidence-scope YEARS by one at
    # the start (a walk-back can cross a year boundary the requested year
    # alone would miss) -- case.request was built with the un-widened
    # EVIDENCE_SCOPE, so the comparison request here must widen the same way.
    widened = _build_request(case.repository, case.snap, case.snapshot_object, case.store,
                             evidence_scope={"tickers": ["AAA", "BBB"], "years": [2019, 2020, 2021]})
    assert inputs["materialization_request_hash"] == widened.request_hash  # refs from the catalog

    ids, specs = _submitted_specs(case, planned["plan_ref"], "k1")
    by_kind = {spec["kind"]: (job_id, spec) for job_id, spec in specs.items()}
    materialize_id, materialize = by_kind["legacy_materialize"]
    assert materialize["dependency_job_ids"] == []
    assert set(materialize["parameters"]["input_bindings"]) == {"snapshot_ref.json",
                                                                "materialization_request.json"}
    for kind in ("legacy_score", "legacy_decision_replay"):
        _, spec = by_kind[kind]
        bindings = spec["parameters"]["input_bindings"]
        assert spec["parameters"]["input_mode"] == "snapshot"
        assert bindings["snapshot_ref.json"] == inputs["snapshot_ref_artifact_id"]
        assert bindings["materialization_request.json"] == inputs["materialization_request_ref"]
        assert bindings["materialization_manifest.json"] == materialize_id + "#" + MANIFEST
        assert "legacy_manifest.json" not in bindings
        assert materialize_id in spec["dependency_job_ids"]
    for kind, (_, spec) in by_kind.items():
        bindings = spec["parameters"]["input_bindings"]
        if kind not in ("legacy_score", "legacy_decision_replay", "legacy_materialize"):
            assert "input_mode" not in spec["parameters"]
            assert bindings["legacy_manifest.json"] == planned["plan"]["input_manifest_ref"]
        for binding in bindings.values():
            if str(binding).startswith("job_"):
                assert binding.split("#", 1)[0] in spec["dependency_job_ids"]

    retry_ids, _ = _submitted_specs(case, planned["plan_ref"], "k2")
    assert retry_ids == ids and calls == ["shadow"]

    advanced = _advance_head(case)
    record_reference_inputs(case, advanced, "r-advance-references")
    replanned = dispatch(parser().parse_args(argv), case.root, case.conn, case.clock)
    assert calls == ["shadow", "shadow"]
    assert replanned["plan"]["snapshot_inputs"]["snapshot_ref_artifact_id"] \
        != inputs["snapshot_ref_artifact_id"]
    new_ids, _ = _submitted_specs(case, replanned["plan_ref"], "k3")
    assert set(new_ids).isdisjoint(ids)


def test_snapshot_plan_refuses_a_snapshot_with_no_committed_reference_inputs(case):
    from engine.v2.data.errors import DataError
    from engine.v2.ops.errors import OpsError

    with pytest.raises(DataError) as missing:
        reference_inputs_for_snapshot(case.conn, scope="other-scope", snapshot_id=case.snap.snapshot_id)
    assert missing.value.code == "SNAPSHOT_NOT_READY"
    _advance_head(case)  # committed by a receipt that recorded no reference inputs
    with pytest.raises(OpsError) as err:
        dispatch(parser().parse_args(_plan_files(case)), case.root, case.conn, case.clock)
    assert err.value.problem.code == "INPUT_CHANGED"
    assert err.value.problem.details["data_code"] == "SNAPSHOT_NOT_READY"


def test_newer_reference_inputs_change_the_request_and_job_identity(case):
    argv = _plan_files(case)
    first = dispatch(parser().parse_args(argv), case.root, case.conn, case.clock)
    first_ids, first_specs = _submitted_specs(case, first["plan_ref"], "refs-1")
    record_reference_inputs(case, case.snap.snapshot_id, "r2-references",
                            registry=b'{"models": [], "retrained": true}')
    second = dispatch(parser().parse_args(argv), case.root, case.conn, case.clock)
    one, two = first["plan"]["snapshot_inputs"], second["plan"]["snapshot_inputs"]
    assert one["snapshot_ref_artifact_id"] == two["snapshot_ref_artifact_id"]
    assert one["materialization_request_hash"] != two["materialization_request_hash"]
    second_ids, second_specs = _submitted_specs(case, second["plan_ref"], "refs-2")
    snapshot_kinds = ("legacy_materialize", "legacy_score", "legacy_decision_replay")
    first_jobs = {job for job, spec in first_specs.items() if spec["kind"] in snapshot_kinds}
    second_jobs = {job for job, spec in second_specs.items() if spec["kind"] in snapshot_kinds}
    assert len(first_jobs) == len(snapshot_kinds) and first_jobs.isdisjoint(second_jobs)
    assert set(first_ids) != set(second_ids)


_FAKE_SNAPSHOT_INPUTS = {"snapshot_ref_artifact_id": "art_snapshot",
                         "materialization_request_ref": "art_request",
                         "scratch_estimate_bytes": 1024}


def test_default_legacy_graph_is_unchanged_by_the_input_mode_flag():
    plan = build_nightly_plan(str(REPO), SESSION)
    common = dict(tickers=("AAA", "BBB"), year_start=2020, year_end=2021, input_refs=("art_m",),
                  expected_population=("AAA|S1|2020-01-15",))
    default = [to_document(r) for r in build_legacy_job_requests(plan, **common)]
    explicit = [to_document(r) for r in build_legacy_job_requests(plan, input_mode="legacy", **common)]
    assert json.dumps(default, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert all("input_mode" not in r["job"]["parameters"] for r in default)
    assert all(r["job"]["parameters"]["input_bindings"]["legacy_manifest.json"] == "art_m"
               for r in default)
    assert "legacy_materialize" not in {r["job"]["kind"] for r in default}
    snapshot = build_legacy_job_requests(plan, input_mode="snapshot",
                                         snapshot_inputs=_FAKE_SNAPSHOT_INPUTS, **common)
    assert {r.job.kind for r in snapshot} == {r["job"]["kind"] for r in default} | {"legacy_materialize"}
    assert not {r.idempotency_key for r in snapshot} & {r["idempotency_key"] for r in default}


def test_snapshot_graph_binds_only_declared_dependencies():
    plan = build_nightly_plan(str(REPO), SESSION)
    requests = build_legacy_job_requests(plan, tickers=("AAA",), year_start=2025, year_end=2026,
                                         input_refs=("art_m",), input_mode="snapshot",
                                         snapshot_inputs=_FAKE_SNAPSHOT_INPUTS)
    violations = []
    for request in requests:
        declared = set(request.job.dependency_job_ids)
        for name, binding in (request.job.parameters.get("input_bindings") or {}).items():
            if str(binding).startswith("job_") and binding.split("#", 1)[0] not in declared:
                violations.append((request.job.kind, name))
    assert violations == []
    for request in requests:
        errors = registry().get(request.job.kind).validate
        assert not errors or not errors(request.job, __import__(
            "engine.v2.foundation", fromlist=["from_document"]).from_document(
                registry().get(request.job.kind).parameters, request.job.parameters))


# --------------------------------------------------------------------------
# in-process: worker half, root verification and refusal branches
# --------------------------------------------------------------------------


def _envelope(case, attempt_id):
    path = case.conn.execute("PRAGMA database_list").fetchone()[2]
    return {"attempt_id": attempt_id, "materialization": {
        "catalog_path": path, "store_root": str(case.store.root), "base": str(case.base)}}


def test_materialize_worker_writes_once_reuses_and_discards_a_losing_partial(case):
    from engine.v2.ops import materialization_worker as worker

    staging = case.tmp / "staging"
    staging.mkdir()
    (staging / "materialization_request.json").write_text(json.dumps(to_document(case.request)))
    first = worker.run_materialize({"expected_ids": ["legacy_materialize"]}, staging,
                                   _envelope(case, "att_one"))
    dest = materialization_root(case.base, case.request.request_hash)
    written = json.loads((staging / "materialization_manifest.json").read_text())
    assert first["reused"] is False and written["files"] == hash_tree(dest)
    before = _stats(dest)

    second = worker.run_materialize({"expected_ids": ["legacy_materialize"]}, staging,
                                    _envelope(case, "att_two"))
    assert second["reused"] is True and _stats(dest) == before
    assert json.loads((staging / "materialization_manifest.json").read_text()) == written

    # A concurrent attempt that loses the rename keeps nothing and writes nothing.
    assert worker._write_root(case.request, _envelope(case, "att_three")["materialization"],
                              dest, "att_three") is None
    assert _stats(dest) == before
    assert not list(case.base.glob(".*partial*"))


def test_verify_root_refuses_links_extras_writable_entries_and_absence(tmp_path):
    from engine.v2.ops.errors import OpsError
    from engine.v2.ops.fingerprints import file_hash
    from engine.v2.ops.snapshot_roots import partial_root, verify_root

    def build(name):
        root = tmp_path / name
        (root / "d").mkdir(parents=True)
        (root / "d" / "f").write_bytes(b"x")
        return root

    def lock(root):
        for path in sorted(root.rglob("*"), reverse=True):
            if not path.is_symlink():
                path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555)

    good = build("good")
    files = {"d/f": file_hash(good / "d" / "f")}
    lock(good)
    assert set(verify_root(good, files)) == {"d/f"}
    with pytest.raises(OpsError):
        verify_root(good, {"d/f": fake_hash("other")})
    with pytest.raises(OpsError):
        verify_root(good, {**files, "d/g": fake_hash("g")})
    with pytest.raises(OpsError):
        verify_root(tmp_path / "absent", files)
    linked = build("linked")
    (linked / "d" / "l").symlink_to(linked / "d" / "f")
    lock(linked)
    with pytest.raises(OpsError):
        verify_root(linked, files)
    writable = build("writable")
    lock(writable)
    (writable / "d" / "f").chmod(0o644)
    with pytest.raises(OpsError):
        verify_root(writable, files)
    with pytest.raises(OpsError):
        partial_root(tmp_path, "sha256:not-a-hash", "att")


def test_manifest_and_request_refusals(case):
    from engine.v2.ops.errors import OpsError
    from engine.v2.ops.snapshot_stages import (
        SnapshotLaunch,
        confirm_attempt,
        manifest_files,
        materialize_effect,
        request_from_artifact,
        request_mismatches,
    )

    good = {"schema_version": "legacy_materialization_manifest.v1.0",
            "request_hash": case.request.request_hash, "snapshot_id": case.snap.snapshot_id,
            "snapshot_manifest_hash": case.snap.manifest_hash, "files": {"a": fake_hash("a")}}
    assert manifest_files(good, case.request) == {"a": fake_hash("a")}
    for bad in ({**good, "schema_version": "x"}, {**good, "request_hash": fake_hash("r")},
                {**good, "files": {}}, [good]):
        with pytest.raises(OpsError):
            manifest_files(bad, case.request)
    assert "data/features/SNAPSHOT" in request_mismatches(case.repository, case.request,
                                                          {"stray.bin": fake_hash("s")})
    tampered = to_document(case.request)
    tampered["direct_scope"] = {"tickers": ["BBB"], "years": [2021]}
    tampered_ref = case.publish(tampered, "legacy_materialization_request.v1.0")
    with pytest.raises(OpsError):
        request_from_artifact(case.conn, case.store, tampered_ref.artifact_id)
    with pytest.raises(OpsError):
        request_from_artifact(case.conn, case.store, case.snapshot_ref.artifact_id)

    class Claim:
        attempt_id = "att_none"
        spec = JobSpec(kind="legacy_score", implementation_ref="i", spec_hash=None,
                       environment_ref="e", parameters={"input_mode": "snapshot"},
                       output_namespace="shadow", resource_class="legacy_score",
                       retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")

    with pytest.raises(OpsError):
        confirm_attempt(case.conn, case.store, Claim, None)
    launch = SnapshotLaunch(mode="snapshot", root=case.tmp, snapshot_artifact_id="a",
                            request_artifact_id="b", request_hash="c", snapshot_manifest_hash="d")
    with pytest.raises(OpsError):
        confirm_attempt(case.conn, case.store, Claim, launch)
    with pytest.raises(OpsError):
        materialize_effect(case.conn, case.store, Claim, [], launch)


def test_planning_refusals(case):
    from engine.v2.ops.errors import OpsError
    from engine.v2.ops.plans import nightly_plan
    from engine.v2.ops.snapshot_planning import direct_scope_for, pin_snapshot_inputs

    assert direct_scope_for(["AAA|S1|2020-01-15", "BBB|S2|2021-02-10"]) == {
        "tickers": ["AAA", "BBB"], "years": [2020, 2021]}
    with pytest.raises(OpsError):
        direct_scope_for(["AAA|2020-01-15"])
    with pytest.raises(OpsError):
        pin_snapshot_inputs(case.conn, case.store, "shadow", tickers=(), year_start=2020,
                            year_end=2021, expected_population=(), clock=case.clock)
    with pytest.raises(OpsError) as err:
        pin_snapshot_inputs(case.conn, case.store, "no-such-scope", tickers=("AAA",),
                            year_start=2020, year_end=2021,
                            expected_population=("AAA|S1|2020-01-15",), clock=case.clock)
    assert err.value.problem.code == "INPUT_CHANGED"
    with pytest.raises(OpsError):  # direct years (2025) outside the evidence years
        pin_snapshot_inputs(case.conn, case.store, "shadow", tickers=("AAA", "BBB"),
                            year_start=2020, year_end=2021,
                            expected_population=("AAA|S1|2025-01-15",), clock=case.clock)
    with pytest.raises(OpsError):
        dispatch(parser().parse_args(["plan", "nightly", "--as-of", SESSION, "--input-mode",
                                      "snapshot"]), case.root, case.conn, case.clock)
    with pytest.raises(OpsError):
        nightly_plan(str(REPO), SESSION, input_mode="snapshot")
    plan = build_nightly_plan(str(REPO), SESSION)
    for kwargs in ({"input_mode": "other"}, {"input_mode": "snapshot"},
                   {"input_mode": "snapshot", "snapshot_inputs": _FAKE_SNAPSHOT_INPUTS,
                    "include_prerequisites": True}):
        with pytest.raises(OpsError):
            build_legacy_job_requests(plan, tickers=("AAA",), year_start=2020, year_end=2021,
                                      **kwargs)


def test_planning_succeeds_when_trades_spans_more_tickers_and_earlier_years(case):
    """Real-shape regression for the heavy-run stage 9a defect: the fixture's
    committed ``trades`` table (TRADE_ROWS) spans {AAA, BBB} x {2020, 2021} --
    wider than a planned universe of just BBB over just 2021. Before the fix,
    ``evidence_scope_covers_trades`` compared trades's real span against the
    abstract evidence_scope (here {"BBB"} x {2020, 2021} -- AAA is outside
    it) and refused with INVALID_REQUEST, exactly the real 201-ticker/
    2023-2026 refusal the task brief reports. daily_market is whole_table by
    construction, so planning must now succeed: trades's span is covered by
    construction, regardless of how much narrower the planned board universe
    is."""
    from engine.v2.ops.snapshot_planning import pin_snapshot_inputs

    result = pin_snapshot_inputs(case.conn, case.store, "shadow", tickers=("BBB",), year_start=2021,
                                 year_end=2021, expected_population=("BBB|S1|2021-02-10",),
                                 clock=case.clock)
    assert result["snapshot_id"] == case.snap.snapshot_id
