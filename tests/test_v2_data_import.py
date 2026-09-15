"""P2-7/Task7b: the supervised §7 snapshot-import coordinator.

``plan_import`` enumeration/refusals are pure unit tests over a synthetic
private legacy root. The end-to-end cases run through a REAL ``Service``
(``tests.ops_support.TEST_POLICY``), exactly like
``tests/test_v2_ops_supervised_legacy.py`` — the only path that exercises the
real Phase 1 read-set pin, staging copy, worker subprocess and coordinator
commit together.

``build_legacy_store`` (this module's own "tiny monkeypatchable builder
path", task brief D16 note) writes a minimal but type-correct synthetic
Parquet tree matching every REAL ``build_legacy_mapping()`` contract exactly
— never real business data, never a real legacy rebuild. It is reused by
``tests/test_v2_data_rebuild_rollback.py``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import ArtifactRef  # noqa: E402
from engine.v2.data import legacy_mapping as data_legacy_mapping  # noqa: E402
from engine.v2.data import reference_inputs  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.errors import fail as fail_data  # noqa: E402
from engine.v2.data.import_snapshot import plan_import  # noqa: E402
from engine.v2.data.objects import PARQUET_FRAGMENT_SCHEMA_REF  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, CONTENT_HASH_PREFIX, SystemClock  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.checkpoints import artifact as load_artifact  # noqa: E402
from engine.v2.ops.snapshot_import import save_import_plan, submit_import  # noqa: E402
from engine.v2.ops.stages import registry  # noqa: E402
from engine.v2.ops.submission import NamespacePolicy  # noqa: E402
from engine.v2.ops.supervisor import LEASE_SECONDS, Service  # noqa: E402
from tests.ops_support import TEST_POLICY, FakeClock  # noqa: E402

POLICY = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
MAPPING = data_legacy_mapping.build_legacy_mapping()
SCOPE = "legacy_primary"

_ARROW_TYPES = {"string": pa.string(), "float64": pa.float64(), "int64": pa.int64(), "bool": pa.bool_()}


def _arrow_type(physical_type: str):
    if physical_type.startswith("timestamp["):
        return pa.timestamp(physical_type[len("timestamp["):-1])
    return _ARROW_TYPES[physical_type]


def _increasing(column: dict, i: int, year: int):
    physical = column["physical_type"]
    if physical == "string":
        return f"{column['name'][:3]}{i:06d}"
    if physical == "int64":
        return i
    if physical == "float64":
        return float(i)
    if physical == "bool":
        return bool(i % 2)
    return datetime(year, 1, 1) + timedelta(seconds=i)


def _fixed(column: dict, year: int):
    if column["name"] == "year":
        return year
    physical = column["physical_type"]
    if physical == "string":
        return f"fx_{column['name']}"
    if physical == "int64":
        return 0
    if physical == "float64":
        return 0.0
    if physical == "bool":
        return True
    return datetime(year, 1, 1)


def _synthetic_rows(contract_doc: dict, n: int, *, year: int, start: int = 0) -> list[dict]:
    """Type-correct, generic synthetic rows for any real ``TableContract`` doc.

    Every non-varying column gets a fixed, type-appropriate placeholder (the
    observation-time column always does, even when nullable, so a whole-table
    materialization read can derive its time bound from fragment records); the
    table's own LAST primary-key column increases strictly with the row
    index, so ``objects.inspect_staged_file``'s key-order check passes
    regardless of how many primary-key columns a table declares (this
    module's judgement call — see the module docstring).
    """
    primary_key = contract_doc["primary_key"]
    partition_columns = set(contract_doc["partition_columns"])
    # The column that varies per row is the LAST primary-key column that is
    # not itself a partition column (e.g. securities' primary key ends in
    # ``year``, which is fixed per partition file) — every other primary-key
    # column, partition column included, is held fixed within one file.
    non_partition_pk = [name for name in primary_key if name not in partition_columns]
    last_pk = non_partition_pk[-1]
    other_pk = set(primary_key) - {last_pk}
    rows = []
    for j in range(n):
        i = start + j
        row = {}
        for column in contract_doc["columns"]:
            name = column["name"]
            if name == last_pk:
                row[name] = _increasing(column, i, year)
            elif (name in other_pk or name in partition_columns or not column["nullable"]
                  or name == contract_doc.get("observation_time_column")):
                row[name] = _fixed(column, year)
            else:
                row[name] = None
        rows.append(row)
    return rows


def _table_from_rows(contract_doc: dict, rows: list[dict]) -> pa.Table:
    arrays = {}
    for column in contract_doc["columns"]:
        arrays[column["name"]] = pa.array([row.get(column["name"]) for row in rows],
                                          type=_arrow_type(column["physical_type"]))
    return pa.table(arrays)


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def build_legacy_store(root: Path, *, year: int = 2024, daily_market_parts: int = 1,
                       rows_per_part: int = 2, generated_at: str | None = None) -> None:
    """A synthetic legacy repo root: tables, feature files, SNAPSHOT and reference inputs.

    ``generated_at`` (task brief 2026-09-14 SEND-BACK): the SNAPSHOT's own
    ``generated_at`` date, which ``plan_import`` now reads as the manifest's
    ``selected_session`` (``import_snapshot.py::_check_snapshot_shape``) —
    defaults to a date tied to ``year``, never ``None``, since a real
    ``engine.data.manifest.write_snapshot`` payload always has one and
    ``_check_snapshot_shape`` now refuses a non-date value.
    """
    data = root / reference_inputs.DATA_DIR
    for name in data_legacy_mapping.TIER2_DATASETS:
        contract_doc = MAPPING["tables"][name]
        parts = daily_market_parts if name == "daily_market" else 1
        for part in range(parts):
            rows = _synthetic_rows(contract_doc, rows_per_part, year=year, start=part * rows_per_part)
            path = data / "curated" / name / f"year={year}" / f"part-{part:04d}.parquet"
            _write_parquet(path, _table_from_rows(contract_doc, rows))
    for name, relative in (("feature_panel", data_legacy_mapping.PANEL_RELATIVE_PATH),
                           ("tier4_forecasts", data_legacy_mapping.TIER4_RELATIVE_PATH)):
        contract_doc = MAPPING["tables"][name]
        rows = _synthetic_rows(contract_doc, rows_per_part, year=year)
        _write_parquet(data / relative, _table_from_rows(contract_doc, rows))
    keys = MAPPING["legacy_snapshot_metadata"]["expected_top_level_keys"]
    snapshot_path = root / reference_inputs.LEGACY_SNAPSHOT_PATH
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: None for key in keys}
    payload["generated_at"] = generated_at or f"{year}-01-15T00:00:00+00:00"
    snapshot_path.write_text(json.dumps(payload))
    write_reference_inputs(root, fold=f"{year}01")


#: The one champion Tier-4 feature model the synthetic registry declares.
REFERENCE_MODEL_ID = "size_v9"
_INPUTS = reference_inputs.LEGACY_REFERENCE_INPUTS_V1["inputs"]


def _synthetic_model_bytes() -> bytes:
    """A REAL joblib artifact (plain-dict payload, no GLOBAL opcode --
    nothing outside builtins is referenced), not literal placeholder bytes.
    ``capture_inputs.capture()`` now opcode-scans every pinned champion
    artifact (``fingerprints.verify_pinned_model_modules``), which needs a
    genuinely parseable joblib file even for a synthetic fixture."""
    buffer = io.BytesIO()
    joblib.dump({"kind": "synthetic_test_fixture"}, buffer, compress=3)
    return buffer.getvalue()


#: Built once at import time: real joblib bytes, deterministic, no engine.*
#: reference for :func:`write_reference_inputs`'s default ``model_bytes``.
_SYNTHETIC_MODEL_BYTES = _synthetic_model_bytes()


def _put(root: Path, relative: str, data: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def reference_artifact_path() -> str:
    return f"{_INPUTS['champion_artifact']['directory']}/{REFERENCE_MODEL_ID}.joblib"


def reference_cache_path(root: Path, fold: str = "202401") -> str:
    panel = root / reference_inputs.DATA_DIR / data_legacy_mapping.PANEL_RELATIVE_PATH
    digest = hashlib.sha256(panel.read_bytes()).hexdigest()[:12]
    return f"{reference_inputs.TIER4_SERVING_DIR}/{REFERENCE_MODEL_ID}_{fold}_{digest}.joblib"


def write_reference_inputs(root: Path, *, model_bytes: bytes = _SYNTHETIC_MODEL_BYTES,
                           fold: str = "202401") -> None:
    """Calendar, registry naming one champion artifact, structures, chooser pool,
    one Tier-4 serving cache and the two Tier-4-derived model outputs (task
    brief 2026-09-14: pnl_sim_history/recalibration_pairs) for the panel
    already written under ``root``."""
    _put(root, _INPUTS["calendar"]["path"], b"Price,Close\nTicker,^GSPC\nDate,\n2024-01-02,4742.83\n")
    _put(root, _INPUTS["structure_champions"]["path"], b"{}\n")
    _put(root, _INPUTS["chooser_analog_pool"]["path"], b"synthetic chooser analog pool")
    _put(root, _INPUTS["pnl_sim_history"]["path"], b"synthetic pnl_sim_history parquet bytes")
    _put(root, _INPUTS["recalibration_pairs"]["path"], b"synthetic recalibration_pairs parquet bytes")
    _put(root, reference_artifact_path(), model_bytes)
    registry = {"version": 1, "models": [
        {"id": REFERENCE_MODEL_ID, "champion": True, "produces": "pred_abs_move",
         "artifact": reference_artifact_path(),
         "artifact_sha256": hashlib.sha256(model_bytes).hexdigest()},
        {"id": "retired_v0", "champion": False, "produces": None,
         "artifact": f"{_INPUTS['champion_artifact']['directory']}/retired_v0.joblib",
         "artifact_sha256": "0" * 64},
    ]}
    _put(root, _INPUTS["model_registry"]["path"], json.dumps(registry).encode())
    _put(root, reference_cache_path(root, fold), b"synthetic serving cache")


# --------------------------------------------------------------------------
# plan_import: enumeration and refusals
# --------------------------------------------------------------------------


def test_plan_import_enumerates_declared_files(tmp_path):
    build_legacy_store(tmp_path, daily_market_parts=3, rows_per_part=2)
    plan = plan_import(tmp_path, scope=SCOPE, expected_head_snapshot_id=None, expected_head_generation=0)
    request = plan.snapshot_import_request
    assert set(request.table_sources) == set(data_legacy_mapping.DATASET_ORDER)
    assert len(request.table_sources["daily_market"]) == 3
    for name in data_legacy_mapping.TIER2_DATASETS:
        if name != "daily_market":
            assert len(request.table_sources[name]) == 1, name
    assert request.legacy_snapshot_source_ref.path == "data/features/SNAPSHOT"
    table_paths = {ref.path for refs in request.table_sources.values() for ref in refs}
    reference_paths = {ref.path for ref in plan.legacy_input_manifest.file_refs} - table_paths
    assert reference_paths == {*(spec["path"] for spec in _INPUTS.values() if "path" in spec),
                               reference_artifact_path(), reference_cache_path(tmp_path)}
    assert len(plan.legacy_input_manifest.file_refs) == len(table_paths) + len(reference_paths)


def test_plan_import_refuses_stray_file(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / "data" / "curated" / "securities" / "year=2024" / "notes.txt").write_text("x")
    with pytest.raises(DataError) as excinfo:
        plan_import(tmp_path, scope=SCOPE, expected_head_snapshot_id=None, expected_head_generation=0)
    assert excinfo.value.code == "CONTRACT_MISMATCH"


def test_plan_import_refuses_symlinked_part(tmp_path):
    build_legacy_store(tmp_path)
    target = tmp_path / "data" / "curated" / "securities" / "year=2024" / "part-0000.parquet"
    link = tmp_path / "data" / "curated" / "securities" / "year=2024" / "part-0001.parquet"
    link.symlink_to(target)
    with pytest.raises(DataError) as excinfo:
        plan_import(tmp_path, scope=SCOPE, expected_head_snapshot_id=None, expected_head_generation=0)
    assert excinfo.value.code == "INPUT_CHANGED"


def test_plan_import_refuses_missing_table(tmp_path):
    build_legacy_store(tmp_path)
    shutil.rmtree(tmp_path / "data" / "curated" / "trades")
    with pytest.raises(DataError) as excinfo:
        plan_import(tmp_path, scope=SCOPE, expected_head_snapshot_id=None, expected_head_generation=0)
    assert excinfo.value.code == "INPUT_CHANGED"


def test_plan_import_refuses_bad_snapshot_shape(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / reference_inputs.LEGACY_SNAPSHOT_PATH).write_text(json.dumps({"only_one_key": True}))
    with pytest.raises(DataError) as excinfo:
        plan_import(tmp_path, scope=SCOPE, expected_head_snapshot_id=None, expected_head_generation=0)
    assert excinfo.value.code == "CONTRACT_MISMATCH"


# --------------------------------------------------------------------------
# end-to-end, through a real Service
# --------------------------------------------------------------------------


def _run_until_terminal(service, conn, job_id, timeout=60,
                        states=("succeeded", "failed", "blocked", "cancelled")):
    deadline = time.monotonic() + timeout
    state = "queued"
    while time.monotonic() < deadline:
        service.tick()
        state = conn.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
        if state in states:
            return state
        time.sleep(0.05)
    return state


def _submit_and_run(root, store_root, conn, clock, *, idempotency_key, expected_head=None,
                    expected_generation=0, scope=SCOPE):
    store = ArtifactStore(root)
    plan = plan_import(store_root, scope=scope, expected_head_snapshot_id=expected_head,
                       expected_head_generation=expected_generation)
    plan_ref = save_import_plan(conn, store, plan, clock=clock)
    receipt = submit_import(conn, store, plan_ref.artifact_id, registry=registry(), policy=POLICY,
                            clock=clock, idempotency_key=idempotency_key, repo_root=ROOT)
    service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=ROOT,
                      store_root=store_root)
    try:
        service.start()
        state = _run_until_terminal(service, conn, receipt.job_id)
    finally:
        service.close()
    return receipt, state, plan


def _object_bytes(store, ref_doc):
    digest = ref_doc["content_hash"].removeprefix(CONTENT_HASH_PREFIX)
    ref = ArtifactRef(artifact_id=ref_doc["object_id"], content_hash=ref_doc["content_hash"],
                      schema_ref=PARQUET_FRAGMENT_SCHEMA_REF, byte_size=ref_doc["byte_size"],
                      storage_key=f"objects/{digest[:2]}/{digest}")
    return store.read_verified(ref)


def test_snapshot_import_end_to_end(tmp_path):
    root, store_root = tmp_path / "ops", tmp_path / "legacy_store"
    root.mkdir()
    store_root.mkdir()
    build_legacy_store(store_root, daily_market_parts=3, rows_per_part=2)

    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    try:
        receipt, state, plan = _submit_and_run(root, store_root, conn, clock, idempotency_key="import-1")
        failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                               (receipt.job_id,)).fetchone()[0]
        assert state == "succeeded", failure

        head = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope=?",
                            (SCOPE,)).fetchone()
        assert head is not None and head["generation"] == 1
        snapshot = Repository(conn).resolve(head["snapshot_id"])
        assert set(snapshot.table_versions) == set(data_legacy_mapping.DATASET_ORDER)

        daily_dvs = conn.execute(
            "SELECT partition_logical_hashes_json FROM data_dataset_versions WHERE dataset_version_id=?",
            (snapshot.table_versions["daily_market"].dataset_version_id,)).fetchone()[0]
        assert json.loads(daily_dvs) != {}, "daily_market's 3-part year needs a stored partition hash"

        store = ArtifactStore(root)
        attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                  (receipt.job_id,)).fetchone()[0]
        outputs = {row[0]: row[1] for row in conn.execute(
            "SELECT name, artifact_id FROM attempt_outputs WHERE attempt_id=?", (attempt_id,)).fetchall()}
        receipt_doc = json.loads(store.read_verified(
            load_artifact(conn, store, outputs["snapshot_import_receipt"])))
        legacy_bytes = _object_bytes(store, receipt_doc["legacy_snapshot_object_ref"])
        assert legacy_bytes == (store_root / reference_inputs.LEGACY_SNAPSHOT_PATH).read_bytes()

        objects_before = conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0]
        fragments_before = conn.execute("SELECT COUNT(*) FROM data_fragments").fetchone()[0]

        # Re-submitting the same import (now against the current head) reuses
        # every immutable row and leaves the head exactly where it is.
        receipt2, state2, _ = _submit_and_run(
            root, store_root, conn, clock, idempotency_key="import-2",
            expected_head=head["snapshot_id"], expected_generation=head["generation"])
        failure2 = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                                (receipt2.job_id,)).fetchone()[0]
        assert state2 == "succeeded", failure2
        head2 = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope=?",
                             (SCOPE,)).fetchone()
        assert (head2["snapshot_id"], head2["generation"]) == (head["snapshot_id"], head["generation"])
        assert conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0] == objects_before
        assert conn.execute("SELECT COUNT(*) FROM data_fragments").fetchone()[0] == fragments_before
    finally:
        conn.close()


def test_source_modified_after_planning_fails_input_changed(tmp_path):
    root, store_root = tmp_path / "ops", tmp_path / "legacy_store"
    root.mkdir()
    store_root.mkdir()
    build_legacy_store(store_root)

    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    try:
        store = ArtifactStore(root)
        plan = plan_import(store_root, scope=SCOPE, expected_head_snapshot_id=None,
                           expected_head_generation=0)
        plan_ref = save_import_plan(conn, store, plan, clock=clock)
        target = store_root / "data" / "curated" / "trades" / "year=2024" / "part-0000.parquet"
        target.write_bytes(target.read_bytes() + b"\x00tampered")
        receipt = submit_import(conn, store, plan_ref.artifact_id, registry=registry(), policy=POLICY,
                                clock=clock, idempotency_key="mod-1", repo_root=ROOT)
        service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=ROOT,
                          store_root=store_root)
        try:
            service.start()
            state = _run_until_terminal(service, conn, receipt.job_id)
        finally:
            service.close()
        assert state == "failed"
        failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                               (receipt.job_id,)).fetchone()[0]
        assert "INPUT_CHANGED" in failure
        assert conn.execute("SELECT COUNT(*) FROM data_snapshot_heads").fetchone()[0] == 0
    finally:
        conn.close()


def test_coordinator_fault_then_clean_retry_reuses_objects(tmp_path, monkeypatch):
    """A fault mid-commit leaves the head unchanged and records a failed
    ``SnapshotImportReceipt`` in its own transaction; resubmitting the exact
    same plan (same content, same ``request_hash``) then commits cleanly and
    republishes the SAME content-addressed objects rather than duplicating
    them.

    Judgement call: ``MANIFEST_CORRUPT`` (this test's injected data-layer
    code) is not retryable in ``contracts.operations.FAILURE_CODES`` — a
    genuinely non-retryable coordinator failure never gets a second attempt
    of the SAME job (``lifecycle.advance_job``: retry requires
    ``outcome.failure.retryable``). "Retry" here is therefore modeled as a
    second, independent submission of the identical plan — the resubmission
    path §7.3 itself calls out ("reuses an already verified object with the
    same content hash on retry") — rather than the job's own bounded-attempt
    mechanism, which this particular failure code never enters.
    """
    root, store_root = tmp_path / "ops", tmp_path / "legacy_store"
    root.mkdir()
    store_root.mkdir()
    build_legacy_store(store_root)

    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    try:
        store = ArtifactStore(root)
        plan = plan_import(store_root, scope=SCOPE, expected_head_snapshot_id=None,
                           expected_head_generation=0)
        plan_ref = save_import_plan(conn, store, plan, clock=clock)

        import engine.v2.ops.snapshot_promotion as sp_mod
        real_commit = sp_mod.commit_snapshot_for_attempt

        def _always_faulty(*args, **kwargs):
            raise fail_data("MANIFEST_CORRUPT", "injected fault for test")

        monkeypatch.setattr(sp_mod, "commit_snapshot_for_attempt", _always_faulty)
        receipt_a = submit_import(conn, store, plan_ref.artifact_id, registry=registry(), policy=POLICY,
                                  clock=clock, idempotency_key="fault-a", repo_root=ROOT)
        service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=ROOT,
                          store_root=store_root)
        try:
            service.start()
            state_a = _run_until_terminal(service, conn, receipt_a.job_id)
        finally:
            service.close()
        assert state_a == "failed"
        assert conn.execute("SELECT COUNT(*) FROM data_snapshot_heads").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM data_import_receipts WHERE status='failed'").fetchone()[0] == 1

        monkeypatch.setattr(sp_mod, "commit_snapshot_for_attempt", real_commit)
        receipt_b = submit_import(conn, store, plan_ref.artifact_id, registry=registry(), policy=POLICY,
                                  clock=clock, idempotency_key="fault-b", repo_root=ROOT)
        service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=ROOT,
                          store_root=store_root)
        try:
            service.start()
            state_b = _run_until_terminal(service, conn, receipt_b.job_id)
        finally:
            service.close()
        failure_b = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                                 (receipt_b.job_id,)).fetchone()[0]
        assert state_b == "succeeded", failure_b
        head = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope=?",
                            (SCOPE,)).fetchone()
        assert head["generation"] == 1
        objects_count = conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0]
        table_files = sum(len(refs) for refs in plan.snapshot_import_request.table_sources.values())
        assert objects_count == table_files + 1  # tables + legacy SNAPSHOT; references are per receipt
    finally:
        conn.close()


# --------------------------------------------------------------------------
# heavy-run stage 9a: lease keepalive, lost lease, no coordinator re-stream
# --------------------------------------------------------------------------


def _job_failure(conn, job_id):
    return conn.execute("SELECT failure_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]


def _daily_market_audit_inputs(conn, store):
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope=?",
                        (SCOPE,)).fetchone()[0]
    repository = Repository(conn, store)
    snapshot = repository.resolve(head)
    manifest = repository._manifest(conn, snapshot.table_versions["daily_market"].dataset_version_id)
    records = list(repository.fragment_records(snapshot, "daily_market"))
    return manifest, records, repository.table_contract(snapshot, "daily_market")


def test_import_keeps_its_lease_per_object_and_never_restreams(tmp_path, monkeypatch):
    """100 s of clock per published object; the coordinator streams no Parquet row."""
    import engine.v2.data.objects as objects_mod
    import engine.v2.ops.snapshot_promotion as sp_mod
    from engine.v2.data.manifests import verify_partition_hashes

    root, store_root = tmp_path / "ops", tmp_path / "legacy_store"
    root.mkdir()
    store_root.mkdir()
    build_legacy_store(store_root, daily_market_parts=3, rows_per_part=2)
    clock = FakeClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    published, streamed = [], []
    real_publish, real_stream = sp_mod.publish_legacy_file, objects_mod._stream_rows

    def slow_publish(*args, **kwargs):
        clock.advance(100)
        published.append(args[3].path)
        return real_publish(*args, **kwargs)

    def counting_stream(*args, **kwargs):
        streamed.append(1)
        return real_stream(*args, **kwargs)

    monkeypatch.setattr(sp_mod, "publish_legacy_file", slow_publish)
    monkeypatch.setattr(objects_mod, "_stream_rows", counting_stream)
    try:
        receipt, state, _ = _submit_and_run(root, store_root, conn, clock, idempotency_key="slow-1")
        assert state == "succeeded", _job_failure(conn, receipt.job_id)
        assert 100 * len(published) > 3 * LEASE_SECONDS
        assert streamed == [], "the commit path must not re-stream any partition"

        store = ArtifactStore(root)
        attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                  (receipt.job_id,)).fetchone()[0]
        output = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=? AND "
                              "name='snapshot_import_receipt'", (attempt_id,)).fetchone()[0]
        envelope = json.loads(store.read_verified(load_artifact(conn, store, output)))["envelope"]
        timings = envelope["coordinator_timings_ms"]
        assert set(timings) == {"publish", "build", "commit"}
        assert timings["publish"] >= 100_000 * len(published)

        # The audit is still callable: it agrees with the worker's partition
        # hash, and it catches a wrong one.
        manifest, records, contract = _daily_market_audit_inputs(conn, store)
        assert manifest.partition_logical_hashes, "3-part year needs a stored partition hash"
        verify_partition_hashes(store, manifest, records, contract)
        wrong = dataclasses.replace(manifest, partition_logical_hashes={
            key: CONTENT_HASH_PREFIX + "0" * 64 for key in manifest.partition_logical_hashes})
        with pytest.raises(DataError) as err:
            verify_partition_hashes(store, wrong, records, contract)
        assert err.value.code == "MANIFEST_CORRUPT"
    finally:
        conn.close()


def test_tampered_bytes_are_still_refused_at_publish(tmp_path, monkeypatch):
    import engine.v2.ops.snapshot_promotion as sp_mod

    root, store_root = tmp_path / "ops", tmp_path / "legacy_store"
    root.mkdir()
    store_root.mkdir()
    build_legacy_store(store_root, daily_market_parts=3)
    clock = FakeClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    real_publish = sp_mod.publish_legacy_file

    def tampering_publish(store, attempt_id, legacy_root, file_ref, **kwargs):
        if file_ref.path.endswith("curated/daily_market/year=2024/part-0001.parquet"):
            staged = Path(legacy_root) / file_ref.path
            staged.chmod(0o600)
            staged.write_bytes(staged.read_bytes() + b"tampered after inspection")
        return real_publish(store, attempt_id, legacy_root, file_ref, **kwargs)

    monkeypatch.setattr(sp_mod, "publish_legacy_file", tampering_publish)
    try:
        receipt, state, _ = _submit_and_run(root, store_root, conn, clock, idempotency_key="tamper")
        assert state == "failed"
        assert "INPUT_CHANGED" in _job_failure(conn, receipt.job_id)
        assert conn.execute("SELECT COUNT(*) FROM data_snapshot_heads").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0] == 0
    finally:
        conn.close()


def test_import_that_loses_its_lease_recovers_and_resubmits_cleanly(tmp_path, monkeypatch):
    import engine.v2.ops.snapshot_promotion as sp_mod
    from engine.v2.ops.lifecycle import request_cancel

    root, store_root = tmp_path / "ops", tmp_path / "legacy_store"
    root.mkdir()
    store_root.mkdir()
    build_legacy_store(store_root, daily_market_parts=3)
    clock = FakeClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    real_publish = sp_mod.publish_legacy_file

    def stalled_publish(*args, **kwargs):
        clock.advance(LEASE_SECONDS + 30)  # one object outlasts the whole lease
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(sp_mod, "publish_legacy_file", stalled_publish)
    try:
        store = ArtifactStore(root)
        plan = plan_import(store_root, scope=SCOPE, expected_head_snapshot_id=None,
                           expected_head_generation=0)
        plan_ref = save_import_plan(conn, store, plan, clock=clock)
        first = submit_import(conn, store, plan_ref.artifact_id, registry=registry(), policy=POLICY,
                              clock=clock, idempotency_key="stalled-1", repo_root=ROOT)
        service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=ROOT,
                          store_root=store_root)
        try:
            service.start()
            state = _run_until_terminal(service, conn, first.job_id,
                                        states=("retry_wait", "failed", "succeeded"))
        finally:
            service.close()
        assert state == "retry_wait", _job_failure(conn, first.job_id)
        attempt = conn.execute("SELECT state, process_state, failure_json FROM attempts "
                               "WHERE job_id=?", (first.job_id,)).fetchone()
        assert (attempt["state"], attempt["process_state"]) == ("failed", "verified_dead")
        assert json.loads(attempt["failure_json"])["code"] == "LEASE_LOST"
        assert conn.execute("SELECT COUNT(*) FROM data_import_receipts "
                            "WHERE status='failed'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0] == 0

        monkeypatch.setattr(sp_mod, "publish_legacy_file", real_publish)
        assert request_cancel(conn, first.job_id, None, clock=clock).state == "cancelled"
        second, state2, _ = _submit_and_run(root, store_root, conn, clock,
                                            idempotency_key="stalled-2")
        assert state2 == "succeeded", _job_failure(conn, second.job_id)
        table_files = sum(len(refs) for refs in plan.snapshot_import_request.table_sources.values())
        objects = conn.execute("SELECT COUNT(*), COUNT(DISTINCT content_hash) "
                               "FROM data_objects").fetchone()
        assert tuple(objects) == (table_files + 1, table_files + 1)
    finally:
        conn.close()
