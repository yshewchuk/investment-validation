"""Pinned legacy reference inputs for snapshot-backed scoring (guide §14, D13/D14).

* Tier 0: every ``LEGACY_REFERENCE_INPUTS_V1`` path equals the legacy constant
  that defines it, relative to ``engine.paths.ROOT``.
* ``plan_import`` resolution and refusals on a synthetic legacy root
  (``tests.test_v2_data_import.build_legacy_store``).
* Imports through a real ``Service``: the exact rows recorded per receipt, and
  a model-only change that reuses the snapshot while recording new refs.
* Schema v5: checksummed, idempotent, append-only, committed receipts only.
* End to end: import, ``ops plan nightly --input-mode snapshot``, submit, and
  the real ``legacy_materialize`` worker writes every reference file at its
  exact legacy path, byte-identical to the source.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from engine.v2.contracts import LegacyFileRef, LegacyInputManifest
from engine.v2.data import legacy_mapping
from engine.v2.data import reference_inputs as ri
from engine.v2.data import schema as data_schema
from engine.v2.data.errors import DataError
from engine.v2.data.import_snapshot import plan_import
from engine.v2.data.reference_catalog import (
    REFERENCE_KINDS,
    ReferenceInput,
    insert_reference_inputs,
    reference_inputs_for_snapshot,
)
from engine.v2.foundation import SystemClock, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.cli import dispatch, parser
from engine.v2.ops.errors import OpsError
from engine.v2.ops.lifecycle import request_cancel
from engine.v2.ops.migrations import Migration, migrate
from engine.v2.ops.snapshot_roots import default_materialization_base, materialization_root
from engine.v2.ops.stages import registry
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY
from tests.test_v2_data_catalog import _assert_immutable, build_chain, catalog, insert_receipt
from tests.test_v2_data_import import (
    REFERENCE_MODEL_ID,
    ROOT,
    _run_until_terminal,
    _submit_and_run,
    build_legacy_store,
    reference_artifact_path,
    reference_cache_path,
    write_reference_inputs,
)

INPUTS = ri.LEGACY_REFERENCE_INPUTS_V1["inputs"]
SCOPE = "shadow"


def _plan(root):
    return plan_import(root, scope=SCOPE, expected_head_snapshot_id=None, expected_head_generation=0)


def _refused(root) -> str:
    with pytest.raises(DataError) as err:
        _plan(root)
    return err.value.code


def _expected_paths(root: Path) -> set[str]:
    exact = {spec["path"] for spec in INPUTS.values() if spec["resolution"] == "exact"}
    return exact | {reference_artifact_path(), reference_cache_path(root)}


# --------------------------------------------------------------------------
# tier 0: paths come from the legacy constants
# --------------------------------------------------------------------------


def test_reference_input_paths_equal_their_legacy_constants():
    from engine import paths, score, structure_registry
    from engine.data.features import tier4
    from engine.models import registry as model_registry

    def rel(module, path):  # each module's own ``paths`` object, in case a test reloaded it
        return Path(path).relative_to(module.paths.ROOT).as_posix()

    assert set(INPUTS) == set(REFERENCE_KINDS)
    assert INPUTS["calendar"]["path"] == rel(score, paths.GSPC_DAILY)
    assert INPUTS["calendar"]["path"] == "earnings_predictions/data/raw/polygon/gspc_daily.csv"
    assert INPUTS["calendar"]["path"].startswith("earnings_predictions/")
    assert INPUTS["model_registry"]["path"] == rel(model_registry, model_registry.REGISTRY_PATH)
    assert INPUTS["structure_champions"]["path"] == rel(structure_registry,
                                                        structure_registry.CHAMPIONS_PATH)
    assert INPUTS["champion_artifact"]["directory"] == rel(model_registry, model_registry.ARTIFACT_DIR)
    assert INPUTS["tier4_serving_cache"]["directory"] == rel(tier4, tier4.SERVING_DIR)
    assert INPUTS["chooser_analog_pool"]["path"] == rel(
        score, score.paths.FEATURES / score.CHOOSER_ANALOG_POOL)
    assert INPUTS["legacy_snapshot"]["path"] == rel(score, score.paths.SNAPSHOT_FILE)
    assert ri.DATA_DIR == rel(score, score.paths.DATA)
    for data_relative, constant in ((legacy_mapping.PANEL_RELATIVE_PATH, score.paths.PANEL),
                                    (legacy_mapping.TIER4_RELATIVE_PATH, score.paths.TIER4),
                                    (legacy_mapping.SNAPSHOT_RELATIVE_PATH, score.paths.SNAPSHOT_FILE)):
        assert f"{ri.DATA_DIR}/{data_relative}" == rel(score, constant)


def test_kind_for_path_classifies_every_declared_input():
    for kind, spec in INPUTS.items():
        if spec["resolution"] == "exact":
            assert ri.kind_for_path(spec["path"]) == kind
    serving = INPUTS["tier4_serving_cache"]["directory"]
    assert ri.kind_for_path(f"{serving}/size_v1_4_202601_{'a' * 12}.joblib") == "tier4_serving_cache"
    assert ri.kind_for_path(f"{INPUTS['champion_artifact']['directory']}/x.joblib") == "champion_artifact"
    assert ri.kind_for_path(f"{serving}/notes.txt") is None
    assert ri.kind_for_path("data/features/panel.parquet") is None


# --------------------------------------------------------------------------
# plan_import resolution and refusals
# --------------------------------------------------------------------------


def test_plan_pins_exactly_the_reference_files_and_excludes_a_stale_cache(tmp_path):
    build_legacy_store(tmp_path)
    stale = f"{ri.TIER4_SERVING_DIR}/{REFERENCE_MODEL_ID}_202401_{'0' * 12}.joblib"
    (tmp_path / stale).write_bytes(b"cache for a different panel")
    manifest = _plan(tmp_path).legacy_input_manifest
    pinned = {ref.path for ref in manifest.file_refs if ri.kind_for_path(ref.path)}
    assert pinned == _expected_paths(tmp_path) and stale not in pinned
    assert manifest.calendar_ref.startswith(INPUTS["calendar"]["path"] + "::")
    assert {ref.split("::")[0] for ref in manifest.registry_and_model_refs} == (
        _expected_paths(tmp_path) - {INPUTS["calendar"]["path"], INPUTS["legacy_snapshot"]["path"]})


def test_missing_registry_referenced_artifact_is_refused(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / reference_artifact_path()).unlink()
    assert _refused(tmp_path) == "INPUT_CHANGED"


def test_missing_exact_reference_input_is_refused(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / INPUTS["calendar"]["path"]).unlink()
    assert _refused(tmp_path) == "INPUT_CHANGED"


def test_tier4_champion_without_a_cache_for_this_panel_is_stale(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / reference_cache_path(tmp_path)).rename(
        tmp_path / f"{ri.TIER4_SERVING_DIR}/{REFERENCE_MODEL_ID}_202401_{'0' * 12}.joblib")
    assert _refused(tmp_path) == "TIER4_CACHE_STALE"


def test_bad_registry_or_artifact_is_a_contract_mismatch(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / reference_artifact_path()).write_bytes(b"retrained without re-registering")
    assert _refused(tmp_path) == "CONTRACT_MISMATCH"
    registry_path = tmp_path / INPUTS["model_registry"]["path"]
    for artifact in ("../outside.joblib", "/abs/model.joblib",
                     f"{ri.TIER4_SERVING_DIR}/{REFERENCE_MODEL_ID}.joblib"):
        write_reference_inputs(tmp_path)
        document = json.loads(registry_path.read_text())
        document["models"][0]["artifact"] = artifact
        registry_path.write_text(json.dumps(document))
        assert _refused(tmp_path) == "CONTRACT_MISMATCH", artifact
    registry_path.write_text("[]")
    assert _refused(tmp_path) == "CONTRACT_MISMATCH"


# --------------------------------------------------------------------------
# imports through a real Service
# --------------------------------------------------------------------------


def _import(tmp_path, store_root, conn, clock, key, head=None, generation=0):
    receipt, state, plan = _submit_and_run(tmp_path / "ops", store_root, conn, clock,
                                           idempotency_key=key, expected_head=head,
                                           expected_generation=generation, scope=SCOPE)
    failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?", (receipt.job_id,)).fetchone()[0]
    assert state == "succeeded", failure
    return plan


def _rows(conn):
    return {row[0]: tuple(row[1:]) for row in conn.execute(
        "SELECT legacy_path, receipt_id, kind, content_hash, byte_size FROM data_import_reference_inputs")}


def _head(conn):
    return tuple(conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope=?",
                              (SCOPE,)).fetchone())


def _store_bytes(root: Path, content_hash: str) -> bytes:
    digest = content_hash.removeprefix("sha256:")
    return (root / "ops" / "objects" / digest[:2] / digest).read_bytes()


@pytest.fixture
def imported(tmp_path):
    store_root = tmp_path / "legacy_store"
    (tmp_path / "ops").mkdir()
    build_legacy_store(store_root)
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops" / "ops.sqlite", clock=clock)
    try:
        yield tmp_path, store_root, conn, clock
    finally:
        conn.close()


def test_import_records_exactly_the_expected_refs_per_receipt(imported):
    tmp_path, store_root, conn, clock = imported
    stale = f"{ri.TIER4_SERVING_DIR}/{REFERENCE_MODEL_ID}_202401_{'0' * 12}.joblib"
    (store_root / stale).write_bytes(b"cache for a different panel")
    _import(tmp_path, store_root, conn, clock, "refs-1")
    rows = _rows(conn)
    assert set(rows) == _expected_paths(store_root) and stale not in rows
    (receipt_id,) = {row[0] for row in rows.values()}
    status, snapshot_id = conn.execute("SELECT status, result_snapshot_id FROM data_import_receipts "
                                       "WHERE receipt_id=?", (receipt_id,)).fetchone()
    assert status == "committed" and snapshot_id == _head(conn)[0]
    for path, (_, kind, content_hash, byte_size) in rows.items():
        source = (store_root / path).read_bytes()
        assert kind == ri.kind_for_path(path) and byte_size == len(source)
        assert _store_bytes(tmp_path, content_hash) == source


def test_model_only_change_reuses_the_snapshot_and_records_new_refs(imported):
    tmp_path, store_root, conn, clock = imported
    _import(tmp_path, store_root, conn, clock, "model-1")
    head = _head(conn)
    first = reference_inputs_for_snapshot(conn, scope=SCOPE, snapshot_id=head[0])
    write_reference_inputs(store_root, model_bytes=b"synthetic size model v2")
    _import(tmp_path, store_root, conn, clock, "model-2", head=head[0], generation=head[1])
    assert _head(conn) == head
    second = reference_inputs_for_snapshot(conn, scope=SCOPE, snapshot_id=head[0])
    by_path = [{item.legacy_path: item.content_hash for item in refs} for refs in (first, second)]
    changed = {path for path in by_path[0] if by_path[0][path] != by_path[1][path]}
    assert changed == {reference_artifact_path(), INPUTS["model_registry"]["path"]}
    receipts = conn.execute("SELECT receipt_id FROM data_import_receipts WHERE status='committed' "
                            "AND result_snapshot_id=?", (head[0],)).fetchall()
    assert len(receipts) == 2
    assert len({row[0] for row in _rows_all(conn)}) == 2


def _rows_all(conn):
    return conn.execute("SELECT receipt_id, legacy_path FROM data_import_reference_inputs").fetchall()


# --------------------------------------------------------------------------
# schema v5
# --------------------------------------------------------------------------


def _reference(path="engine/models/registry.json", kind="model_registry"):
    return ReferenceInput(kind=kind, legacy_path=path, object_id="art_x",
                          content_hash="sha256:" + "ab" * 32, byte_size=3)


def test_v5_migration_is_checksummed_idempotent_and_append_only(tmp_path):
    conn, clock = catalog(tmp_path)
    versions = conn.execute("SELECT version, name, checksum FROM schema_versions WHERE owner='data' "
                            "ORDER BY version").fetchall()
    assert tuple(versions[-1])[:2] == (5, "import_reference_inputs")
    ids = build_chain(conn, clock)
    insert_reference_inputs(conn, ids["receipt_id"], [_reference()])
    _assert_immutable(conn, "data_import_reference_inputs", f"receipt_id = '{ids['receipt_id']}'",
                      "byte_size = 4")
    with pytest.raises(sqlite3.IntegrityError):
        insert_reference_inputs(conn, ids["receipt_id"], [_reference()])  # (receipt, path) is the key
    insert_receipt(conn, clock, receipt_id="r-failed", status="failed", result_snapshot_id=None)
    with pytest.raises(sqlite3.IntegrityError, match="committed import receipt"):
        insert_reference_inputs(conn, "r-failed", [_reference()])
    with pytest.raises(DataError):
        insert_reference_inputs(conn, ids["receipt_id"], [_reference("x", kind="unknown")])
    conn.close()

    reopened = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    assert reopened.execute("SELECT version, name, checksum FROM schema_versions WHERE owner='data' "
                            "ORDER BY version").fetchall() == versions
    reopened.close()
    edited = [Migration(v, n, s) for v, n, s in data_schema.MIGRATIONS]
    version, name, statements = data_schema.MIGRATIONS[4]
    edited[4] = Migration(version, name, statements + ("SELECT 1",))
    raw = sqlite3.connect(str(tmp_path / "catalog.sqlite"), isolation_level=None)
    try:
        with pytest.raises(OpsError) as err:
            migrate(raw, data_schema.OWNER, tuple(edited), clock=clock)
        assert err.value.problem.details["reason"] == "checksum_mismatch"
    finally:
        raw.close()


# --------------------------------------------------------------------------
# end to end: import -> plan nightly --input-mode snapshot -> real materialize worker
# --------------------------------------------------------------------------


def _plan_argv(tmp_path, store_root):
    tickers = sorted(set(pq.read_table(next((store_root / ri.DATA_DIR / "curated" / "trades").rglob(
        "*.parquet")), columns=["ticker"]).column("ticker").to_pylist()))
    fixture = store_root / INPUTS["structure_champions"]["path"]
    from engine.v2.data.legacy_nightly_read_plan import NIGHTLY_CAPTURE_IMPLEMENTATION_REF
    from engine.v2.ops.fingerprints import file_hash
    manifest = tmp_path / "legacy_manifest.json"
    # This plan's own barrier-only kinds (legacy_finality/decisions/...) are
    # all cancelled below before they ever run -- only legacy_materialize's
    # own snapshot_ref/materialization_request bindings matter to this test
    # (nightly.py:_stage_inputs never binds materialize to legacy_manifest.json
    # at all). The plan-time guard (engine.v2.data.legacy_nightly_read_plan,
    # capture_inputs deliverable) still runs at plan time regardless, so these
    # entries are placeholder PATHS satisfying its presence check, not files
    # this test's materialize worker ever reads.
    placeholder_refs = tuple(
        LegacyFileRef(path=f"{ri.DATA_DIR}/curated/{table}/year=2024/part-0000.parquet",
                      content_hash=file_hash(fixture), byte_size=fixture.stat().st_size)
        for table in ("daily_market", "option_chains", "earnings_events", "trades")
    ) + (
        LegacyFileRef(path="data/raw/fetch/orats/ab/placeholder.meta.json",
                      content_hash=file_hash(fixture), byte_size=fixture.stat().st_size),
        LegacyFileRef(path=INPUTS["structure_champions"]["path"], content_hash=file_hash(fixture),
                      byte_size=fixture.stat().st_size),
    )
    manifest.write_text(json.dumps(to_document(LegacyInputManifest(
        manifest_id="m1", file_refs=placeholder_refs,
        table_contract_refs=(), registry_and_model_refs=("placeholder::sha256:" + "0" * 64,),
        calendar_ref="placeholder::sha256:" + "0" * 64,
        selected_session="2026-09-12", finality_receipt_refs=(), knowledge_mode_by_table={},
        availability_evidence_refs=(), read_set_complete=True,
        capture_implementation_ref=NIGHTLY_CAPTURE_IMPLEMENTATION_REF))))
    population = tmp_path / "population.json"
    population.write_text(json.dumps([f"{tickers[0]}|S1|2024-01-15"]))
    return ["plan", "nightly", "--as-of", "2026-09-12", "--input-mode", "snapshot",
            "--snapshot-scope", SCOPE, "--input-manifest", str(manifest),
            "--expected-population", str(population), "--tickers", ",".join(tickers),
            "--year-start", "2024", "--year-end", "2024"]


def test_import_plan_submit_materializes_reference_files_byte_identical(imported):
    tmp_path, store_root, conn, clock = imported
    ops_root = tmp_path / "ops"
    _import(tmp_path, store_root, conn, clock, "e2e-import")
    planned = dispatch(parser().parse_args(_plan_argv(tmp_path, store_root)), ops_root, conn, clock)
    request_hash = planned["plan"]["snapshot_inputs"]["materialization_request_hash"]
    submitted = dispatch(parser().parse_args(["submit", "--plan", planned["plan_ref"],
                                              "--idempotency-key", "e2e"]), ops_root, conn, clock)
    kinds = {job["job_id"]: json.loads(conn.execute("SELECT spec_json FROM jobs WHERE job_id=?",
                                                    (job["job_id"],)).fetchone()[0])["kind"]
             for job in submitted["jobs"]}
    (materialize_id,) = [job for job, kind in kinds.items() if kind == "legacy_materialize"]
    for job in kinds:
        if job != materialize_id:
            request_cancel(conn, job, None, clock=clock)
    service = Service(conn, ops_root, registry(), TEST_POLICY, clock=clock, code_source=ROOT,
                      store_root=store_root)
    try:
        service.start()
        state = _run_until_terminal(service, conn, materialize_id, timeout=120)
    finally:
        service.close()
    failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?", (materialize_id,)).fetchone()[0]
    assert state == "succeeded", failure

    dest = materialization_root(default_materialization_base(ops_root), request_hash)
    rows = _rows(conn)
    assert set(rows) == _expected_paths(store_root)
    for path in rows:
        assert (dest / path).is_file() and not (dest / path).is_symlink()
        assert (dest / path).read_bytes() == (store_root / path).read_bytes(), path
