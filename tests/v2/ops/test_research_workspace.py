"""Experiment research workspace (platform PR 1): overlay isolation, closure and refusals.

Synthetic only: a tiny two-snapshot production catalog/store is built in ``tmp_path``.
(a) production catalog/store/heads hash identically after every workspace operation;
(b) identities re-resolve, and a missing or changed object refuses with a typed refusal;
(c) no supervisor or job launches from a workspace.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine.v2.data import catalog, manifests  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.foundation import ArtifactStore  # noqa: E402
from engine.v2.ops import executor  # noqa: E402
from engine.v2.ops.catalog import register_artifact, transaction  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.supervisor import Service  # noqa: E402
from engine.v2.ops.workspace import closure as closure_module  # noqa: E402
from engine.v2.ops.workspace import overlay as overlay_module  # noqa: E402
from engine.v2.ops.workspace.guard import refuse_workspace_execution  # noqa: E402
from engine.v2.ops.workspace.overlay import (  # noqa: E402
    IDENTITY_FILE,
    create_workspace,
    open_workspace,
)
from tests.data_scan_support import (  # noqa: E402
    RECEIPT,
    catalog_and_store,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)

_SEC = contract_for("securities")
_SEC_REF = contract_ref_for(_SEC)


def _row(ticker: str) -> dict:
    return dict(ticker=ticker, year=2024, first_date=None, last_date=None, mcap_usd=1.5e9,
                mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
                n_obs=250, src="orats")


def _commit(conn, clock, store, ticker, receipt_id, head, generation):
    record = publish_and_inspect(store, _SEC, _SEC_REF, [_row(ticker)], partition_key="2024")
    manifest = manifests.dataset_manifest(
        _SEC_REF, [record], knowledge_mode="reconstructed", coverage_receipt_refs=(RECEIPT,),
        availability_evidence_refs=())
    snap = manifests.snapshot_ref({"securities": manifest}, calendar_version="cal.v1",
                                  source_priority_version="prio.v1",
                                  finality_receipt_refs=(RECEIPT,), parent_snapshot_id=head)
    receipt = catalog.commit_snapshot(
        conn, scope="shadow", request_hash=fake_hash(receipt_id), contracts=[_SEC],
        objects=[record.object_ref], records=[record], manifests=[manifest], snapshot=snap,
        expected_head_snapshot_id=head, expected_head_generation=generation,
        receipt_id=receipt_id, attempt_id="att-" + receipt_id, fence=1,
        fence_check=lambda _c: None, clock=clock, store=store)
    return receipt.snapshot_ref, record


@pytest.fixture
def prod(tmp_path):
    root = tmp_path / "prod"
    root.mkdir()
    conn, clock, store = catalog_and_store(root)
    first, _ = _commit(conn, clock, store, "AAA", "r1", None, 0)
    snap, record = _commit(conn, clock, store, "BBB", "r2", first.snapshot_id, 1)
    conn.close()
    digest = record.object_ref.content_hash.removeprefix("sha256:")
    return SimpleNamespace(
        root=root, catalog=root / "catalog.sqlite", store=root / "store", snap=snap, clock=clock,
        tmp=tmp_path, object=root / "store" / "objects" / digest[:2] / digest)


def _create(prod, name="ws", **kwargs):
    return create_workspace(
        prod.tmp / name, production_catalog=prod.catalog, production_store=prod.store,
        snapshot_id=kwargs.pop("snapshot_id", prod.snap.snapshot_id),
        manifest_hash=kwargs.pop("manifest_hash", prod.snap.manifest_hash), clock=prod.clock,
        **kwargs)


def _reopen(prod, name="ws", **kwargs):
    return open_workspace(prod.tmp / name, production_catalog=kwargs.pop("catalog", prod.catalog),
                          production_store=kwargs.pop("store", prod.store), clock=prod.clock,
                          **kwargs)


def _state(prod):
    digest = hashlib.sha256(prod.catalog.read_bytes())
    for path in sorted(p for p in prod.store.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(prod.store)).encode() + path.read_bytes())
    ro = sqlite3.connect(prod.catalog.as_uri() + "?mode=ro", uri=True)
    heads = ro.execute("SELECT scope, snapshot_id, generation FROM data_snapshot_heads "
                       "ORDER BY scope").fetchall()
    ro.close()
    return digest.hexdigest(), heads


def _details(caught):
    return caught.value.problem.code, caught.value.problem.details.get("reason")


def _rewrite_without_triggers(prod, table, statement, args):
    conn = sqlite3.connect(prod.catalog)
    conn.execute("PRAGMA foreign_keys = OFF")
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' "
                                "AND tbl_name = ?", (table,)).fetchall():
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute(statement, args)
    conn.commit()
    conn.close()


def test_production_state_is_unchanged_by_every_operation(prod):
    before = _state(prod)
    with _create(prod) as ws:
        assert _state(prod) == before
        ref = ws.store.publish_bytes(b"{}", schema_ref="scratch_note.v1.0")
        with transaction(ws.conn):
            register_artifact(ws.conn, ref, None, prod.clock)
        ws.source.fragment_records(ws.source_snapshot(), "securities")
        ws.verify_before_seal()
        with pytest.raises(sqlite3.OperationalError):
            ws._source_conn.execute("DELETE FROM data_snapshot_heads")
        assert _state(prod) == before
        assert len([p for p in (ws.paths.store / "objects").rglob("*") if p.is_file()]) == 2
    with _reopen(prod) as again:
        again.verify_before_seal(deep=True)
    assert _state(prod) == before


def test_identities_resolve_identically_and_only_by_snapshot_id(prod):
    with _create(prod) as ws:
        ref = ws.source_snapshot()
        assert (ref.snapshot_id, ref.manifest_hash) == (prod.snap.snapshot_id, prod.snap.manifest_hash)
        with pytest.raises(DataError) as caught:
            ws.source.resolve("snap_other")
        assert caught.value.code == "SNAPSHOT_NOT_FOUND"
        for name in ("resolve_pinned", "resolve_full_pinned", "latest_dataset_version", "commit"):
            with pytest.raises(OpsError) as caught:
                getattr(ws.source, name)
            assert _details(caught) == ("WORKSPACE_REFUSED", "source_method_refused")
        digest = ws.closure.identity_digest
    with _reopen(prod) as again:
        assert again.source_snapshot().manifest_hash == prod.snap.manifest_hash
        assert again.closure.identity_digest == digest


@pytest.mark.parametrize("mutate,reason", [
    (lambda p: p.object.unlink(), "object_missing"),
    (lambda p: (os.chmod(p.object, 0o644), p.object.write_bytes(b"short")), "object_changed"),
])
def test_missing_or_changed_object_refuses_and_leaves_no_workspace(prod, mutate, reason):
    mutate(prod)
    with pytest.raises(DataError) as caught:
        _create(prod)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", reason)
    assert not (prod.tmp / "ws").exists()


def test_same_size_rewrite_is_caught_by_deep_verify_and_by_the_seal_check(prod):
    original = prod.object.read_bytes()
    flipped = bytes([original[0] ^ 1]) + original[1:]
    stat = prod.object.stat()
    os.chmod(prod.object, 0o644)
    with _create(prod) as ws:
        prod.object.write_bytes(flipped)
        os.utime(prod.object, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        with pytest.raises(DataError) as caught:
            ws.verify_before_seal()
        assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "closure_changed")
    with pytest.raises(DataError) as caught:
        _create(prod, "deep", deep=True)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "object_changed")
    with pytest.raises(DataError) as caught:
        _reopen(prod, deep=True)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "object_changed")


def test_identity_and_row_defects_refuse_with_typed_reasons(prod):
    with pytest.raises(DataError) as caught:
        _create(prod, "bad-hash", manifest_hash="sha256:" + "0" * 64)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "manifest_mismatch")
    with pytest.raises(DataError) as caught:
        _create(prod, "unknown", snapshot_id="snap_unknown")
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "snapshot_unresolved")
    _rewrite_without_triggers(prod, "data_snapshots", "DELETE FROM data_snapshots WHERE snapshot_id "
                              "= (SELECT parent_snapshot_id FROM data_snapshots WHERE snapshot_id = ?)",
                              (prod.snap.snapshot_id,))
    with pytest.raises(DataError) as caught:
        _create(prod, "no-parent")
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "parent_missing")


def test_missing_commit_receipt_refuses(prod):
    _rewrite_without_triggers(prod, "data_import_receipts",
                              "DELETE FROM data_import_receipts WHERE result_snapshot_id = ?",
                              (prod.snap.snapshot_id,))
    with pytest.raises(DataError) as caught:
        _create(prod)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "commit_receipt_missing")


def test_commit_receipt_ref_mismatch_refuses(prod):
    _rewrite_without_triggers(prod, "data_snapshots",
                              "UPDATE data_snapshots SET commit_receipt_ref = 'not-the-receipt' "
                              "WHERE snapshot_id = ?", (prod.snap.snapshot_id,))
    with pytest.raises(DataError) as caught:
        _create(prod)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "commit_receipt_ref_mismatch")


def test_roots_and_paths_are_refused_with_typed_reasons(prod):
    (prod.tmp / "real").mkdir()
    (prod.tmp / "link").symlink_to(prod.tmp / "real")
    (prod.tmp / "full").mkdir()
    (prod.tmp / "full" / "stray").write_text("x")
    cases = [(prod.root / "ws", "production_root"), (prod.store / "ws", "production_root"),
             (prod.root, "production_root"), (prod.tmp / "gone" / "ws", "parent_missing"),
             (prod.tmp / "link" / "ws", "symlink_path"), (prod.tmp / "full", "root_not_empty")]
    for root, reason in cases:
        with pytest.raises(OpsError) as caught:
            create_workspace(root, production_catalog=prod.catalog, production_store=prod.store,
                             snapshot_id=prod.snap.snapshot_id,
                             manifest_hash=prod.snap.manifest_hash, clock=prod.clock)
        assert _details(caught) == ("WORKSPACE_REFUSED", reason), root
    assert (prod.tmp / "full" / "stray").exists()


def test_reopen_refuses_symlink_escape_and_a_different_production(prod):
    _create(prod).close()
    scratch = prod.tmp / "ws" / "scratch"
    scratch.rmdir()
    scratch.symlink_to(prod.tmp / "real", target_is_directory=True)
    with pytest.raises(OpsError) as caught:
        _reopen(prod)
    assert _details(caught) == ("WORKSPACE_REFUSED", "layout_invalid")
    scratch.unlink()
    scratch.mkdir()
    other = prod.tmp / "other_store"
    other.mkdir()
    with pytest.raises(OpsError) as caught:
        _reopen(prod, store=other)
    assert _details(caught) == ("WORKSPACE_REFUSED", "production_mismatch")
    _reopen(prod).close()


def test_no_supervisor_or_job_launches_from_a_workspace(prod):
    with _create(prod) as ws:
        alias = prod.tmp / "alias"
        alias.symlink_to(ws.paths.root, target_is_directory=True)
        for path in (ws.paths.root, ws.paths.store, ws.paths.scratch / "deeper" / "x", alias):
            with pytest.raises(OpsError) as caught:
                Service(None, path, None, None, clock=None, code_source=prod.tmp)
            assert _details(caught) == ("WORKSPACE_REFUSED", "execution_in_workspace")
        with pytest.raises(OpsError) as caught:
            executor.launch(None, None, None, ws.store, prod.tmp, clock=None, boot_id="b")
        assert caught.value.code == "WORKSPACE_REFUSED"
        with pytest.raises(OpsError) as caught:
            executor.launch(None, None, None, ArtifactStore(prod.tmp / "elsewhere"),
                            ws.paths.scratch, clock=None, boot_id="b")
        assert caught.value.code == "WORKSPACE_REFUSED"
    refuse_workspace_execution(prod.root, prod.tmp / "elsewhere", None)


def test_foreign_snapshot_refs_and_queries_are_refused_on_the_source_view(prod):
    foreign = dataclasses.replace(prod.snap, snapshot_id="snap_foreign")
    with _create(prod) as ws:
        calls = (lambda: ws.source.table_contract(foreign, "securities"),
                 lambda: ws.source.fragment_records(foreign, "securities"),
                 lambda: ws.source.get_event(None, foreign),
                 lambda: ws.source.get_chain(None, foreign),
                 lambda: ws.source.get_price_series(None, foreign),
                 lambda: ws.source.get_close("AAA", "2024-01-02", "2024-01-03", foreign),
                 lambda: ws.source.explain_dependencies(None, snapshot_ref=foreign),
                 lambda: ws.source.explain_dependencies(SimpleNamespace(snapshot_id="snap_foreign"),
                                                        snapshot_ref=ws.source_snapshot()),
                 lambda: ws.source.scan(SimpleNamespace(snapshot_id="snap_foreign"),
                                        table_name="securities"),
                 lambda: ws.source.scan_population_bound("snap_foreign", table_name="securities"))
        for call in calls:
            with pytest.raises(DataError) as caught:
                call()
            assert caught.value.code == "SNAPSHOT_NOT_FOUND"
        forged = dataclasses.replace(prod.snap, table_versions={})
        for call in (lambda: ws.source.table_contract(forged, "securities"),
                     lambda: ws.source.fragment_records(forged, "securities")):
            with pytest.raises(DataError) as caught:
                call()
            assert caught.value.code == "SNAPSHOT_NOT_FOUND"
        assert ws.source.table_contract(ws.source_snapshot(), "securities").table_name == "securities"


def test_read_pins_land_only_in_the_workspace(prod):
    before = _state(prod)
    with _create(prod) as ws:
        ref = ws.record_read_pin("preflight", {"history_snapshot": "snap_x"}, clock=prod.clock)
        assert ws.store.read_verified(ref).startswith(b"{")
        assert (ws.paths.store / ref.storage_key).is_file()
        assert not (prod.store / ref.storage_key).exists()
        with pytest.raises(OpsError) as caught:
            ws.record_read_pin("", {}, clock=prod.clock)
        assert _details(caught) == ("WORKSPACE_REFUSED", "pin_name_invalid")
    assert _state(prod) == before


def test_closure_byte_and_row_caps_refuse_with_typed_reasons(prod, monkeypatch):
    # The probes query the migrated columns (data_fragments.input_receipt_refs_json from migration 2,
    # data_price_captures.contract_id from migration 9): a zero budget must refuse, not raise SQL errors.
    monkeypatch.setattr(closure_module, "_MAX_ROW_BYTES", 0)
    with pytest.raises(DataError) as caught:
        _create(prod, "bytes")
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "closure_bytes_over_cap")
    monkeypatch.setattr(closure_module, "_MAX_ROW_BYTES", 256 * 1024 * 1024)
    monkeypatch.setattr(closure_module, "_MAX_ROWS", 0)
    with pytest.raises(DataError) as caught:
        _create(prod, "rows")
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "closure_rows_over_cap")
    assert not (prod.tmp / "bytes").exists() and not (prod.tmp / "rows").exists()


def _receipt_id(prod):
    conn = sqlite3.connect(prod.catalog)
    try:
        row = conn.execute("SELECT receipt_id FROM data_import_receipts "
                           "WHERE result_snapshot_id = ?",
                           (prod.snap.snapshot_id,)).fetchone()
    finally:
        conn.close()
    assert row is not None
    return row[0]


def test_lineage_base_uncommitted_refuses(prod):
    _rewrite_without_triggers(
        prod, "data_receipt_lineage",
        "INSERT INTO data_receipt_lineage (receipt_id, base_receipt_id, kind) "
        "VALUES (?, ?, ?)",
        (_receipt_id(prod), "no-such-receipt", "price_history_capture"))
    with pytest.raises(DataError) as caught:
        _create(prod)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "lineage_receipt_missing")


def test_lineage_cycle_refuses(prod):
    receipt = _receipt_id(prod)
    _rewrite_without_triggers(
        prod, "data_receipt_lineage",
        "INSERT INTO data_receipt_lineage (receipt_id, base_receipt_id, kind) "
        "VALUES (?, ?, ?)",
        (receipt, receipt, "price_history_capture"))
    with pytest.raises(DataError) as caught:
        _create(prod)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "lineage_cycle")


def test_coverage_naming_uncommitted_receipt_refuses(prod):
    _rewrite_without_triggers(
        prod, "data_snapshot_coverage",
        "INSERT INTO data_snapshot_coverage (snapshot_id, table_name, coverage_id, "
        "import_receipt_id, coverage_hash, coverage_json) VALUES (?, ?, ?, ?, ?, ?)",
        (prod.snap.snapshot_id, "securities", "cov-test", "no-such-receipt",
         fake_hash("coverage"), "{}"))
    with pytest.raises(DataError) as caught:
        _create(prod)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "coverage_receipt_missing")


def test_capture_rows_change_the_closure_identity(prod):
    with _create(prod, "before") as ws:
        before = ws.closure.identity_digest
    _rewrite_without_triggers(
        prod, "data_price_captures",
        "INSERT INTO data_price_captures (capture_id, receipt_id, ticker, source_kind, "
        "source_hash, retrieved_at, outcome, rows_added, rows_tombstoned, created_at, "
        "contract_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("cap-test", _receipt_id(prod), "AAA", "orats", fake_hash("source"),
         "2024-01-02T00:00:00Z", "added", 1, 0, "2024-01-02T00:00:00Z",
         "price_history.v1"))
    with _create(prod, "after") as ws:
        after = ws.closure.identity_digest
    assert before != after


def test_reopen_refuses_malformed_identity_field_types(prod):
    with _create(prod, "typed") as ws:
        identity_path = ws.paths.root / IDENTITY_FILE
    identity = json.loads(identity_path.read_text())
    identity["snapshot_id"] = ["not", "a", "string"]
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(OpsError) as caught:
        _reopen(prod, "typed")
    assert _details(caught) == ("WORKSPACE_REFUSED", "identity_unreadable")


def test_reopen_refuses_a_different_schema_version_with_its_own_reason(prod):
    with _create(prod, "schema") as ws:
        identity_path = ws.paths.root / IDENTITY_FILE
    identity = json.loads(identity_path.read_text())
    if isinstance(identity["schema_version"], int):
        identity["schema_version"] += 1
    else:
        identity["schema_version"] = "other"
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(OpsError) as caught:
        _reopen(prod, "schema")
    assert _details(caught) == ("WORKSPACE_REFUSED", "schema_mismatch")


def test_unrelated_committed_receipt_does_not_change_the_closure_identity(prod):
    committed = _receipt_id(prod)
    copy = ("INSERT INTO data_import_receipts (receipt_id, attempt_id, fence, source_manifest_hash, "
            "result_snapshot_id, status, problem_json, registered_at, scope) "
            "SELECT ?, attempt_id, fence, source_manifest_hash, ?, status, problem_json, "
            "registered_at, scope FROM data_import_receipts WHERE receipt_id = ?")
    with _create(prod, "before") as ws:
        before = ws.closure.identity_digest
    _rewrite_without_triggers(prod, "data_import_receipts", copy,
                              ("second-commit", prod.snap.snapshot_id, committed))
    with _create(prod, "after") as ws:
        after = ws.closure.identity_digest
    assert after == before


def test_bounded_cursor_refuses_on_rows_and_bytes_and_covers_coverage_reads(prod, monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?)", [("abcd",)] * 3)
    bounded = closure_module._BoundedConn(conn)
    assert len(bounded.execute("SELECT v FROM t").fetchall()) == 3
    monkeypatch.setattr(closure_module, "_MAX_ROWS", 2)
    with pytest.raises(DataError) as caught:
        bounded.execute("SELECT v FROM t").fetchall()
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "closure_rows_over_cap")
    monkeypatch.setattr(closure_module, "_MAX_ROWS", 2_000_000)
    monkeypatch.setattr(closure_module, "_MAX_ROW_BYTES", 8)
    with pytest.raises(DataError) as caught:
        bounded.execute("SELECT v FROM t").fetchall()
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "closure_bytes_over_cap")
    # a read that used to be uncapped (coverage rows) now goes through the same cursor
    monkeypatch.setattr(closure_module, "_MAX_ROW_BYTES", 256 * 1024 * 1024)
    ro = sqlite3.connect(prod.catalog.as_uri() + "?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    monkeypatch.setattr(closure_module, "_MAX_ROWS", 0)
    with pytest.raises(DataError) as caught:
        closure_module._commit_receipts(closure_module._BoundedConn(ro), prod.snap.snapshot_id)
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "closure_rows_over_cap")
    ro.close()


def test_source_view_reads_go_through_the_bounded_cursor(prod, monkeypatch):
    with _create(prod) as ws:
        monkeypatch.setattr(closure_module, "_MAX_ROWS", 0)
        with pytest.raises(DataError) as caught:
            ws.source.fragment_records(ws.source_snapshot(), "securities")
        assert _details(caught)[1] == "closure_rows_over_cap"


def test_pin_resolution_in_create_goes_through_the_bounded_cursor(prod, monkeypatch):
    monkeypatch.setattr(overlay_module, "verify_snapshot_closure", lambda *a, **k: None)
    monkeypatch.setattr(closure_module, "_MAX_ROWS", 0)
    with pytest.raises(DataError) as caught:
        _create(prod, "pinbound")
    assert _details(caught)[1] == "closure_rows_over_cap"


def test_schema_compatibility_read_goes_through_the_bounded_cursor(prod, monkeypatch):
    monkeypatch.setattr(closure_module, "_MAX_ROWS", 0)
    with pytest.raises(DataError) as caught:
        _create(prod, "schemabound")
    assert _details(caught)[1] == "closure_rows_over_cap"


def test_single_value_over_the_sqlite_length_limit_refuses_with_the_byte_cap(prod, monkeypatch):
    monkeypatch.setattr(overlay_module, "_MAX_VALUE_BYTES", 200)
    with pytest.raises(DataError) as caught:
        _create(prod, "bigvalue")
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "closure_bytes_over_cap")
    assert not (prod.tmp / "bigvalue").exists()
    monkeypatch.undo()
    with _create(prod, "smallvalue") as ws:  # default 64 MiB cap: normal reads still work
        assert ws.closure.object_count >= 1


def test_malformed_public_arguments_refuse_with_typed_errors(prod):
    with pytest.raises(OpsError) as caught:
        _create(prod, "badid", snapshot_id=["not", "a", "string"])
    assert caught.value.code == "WORKSPACE_REFUSED"
    assert caught.value.problem.details.get("reason") == "argument_invalid"
    with pytest.raises(OpsError) as caught:
        open_workspace(None, production_catalog=prod.catalog, production_store=prod.store, clock=None)
    assert caught.value.problem.details.get("reason") == "argument_invalid"
    with _create(prod) as ws:
        for call in (lambda: ws.source.scan(None, table_name="securities"),
                     lambda: ws.source.table_contract(None, "securities")):
            with pytest.raises(DataError) as caught:
                call()
            assert caught.value.code == "SNAPSHOT_NOT_FOUND"
    with pytest.raises(DataError) as caught:
        closure_module.verify_snapshot_closure(None, None, ["x"], "h")
    assert _details(caught) == ("SNAPSHOT_CLOSURE_BROKEN", "argument_invalid")
    with pytest.raises(OpsError) as caught:
        overlay_module.SnapshotOnlyRepository(None, None, ["x"])
    assert caught.value.problem.details.get("reason") == "argument_invalid"


def test_storage_failures_leave_the_entry_points_as_typed_refusals(prod, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(overlay_module, "verify_snapshot_closure", broken)
    with pytest.raises(OpsError) as caught:
        _create(prod, "storagefail")
    assert caught.value.code == "WORKSPACE_REFUSED"
    assert caught.value.problem.details.get("reason") == "storage_failure"
    assert caught.value.problem.details.get("cause") == "ValueError"


def test_second_create_of_the_same_name_refuses_and_leaves_the_first_intact(prod):
    first = _create(prod, "same")
    identity_before = first.identity
    with pytest.raises(OpsError) as caught:
        _create(prod, "same")
    assert caught.value.code == "WORKSPACE_REFUSED"
    assert caught.value.problem.details.get("reason") == "root_not_empty"
    assert (prod.tmp / "same" / ".research_workspace").exists()
    first.close()
    with _reopen(prod, "same") as again:
        assert again.identity == identity_before
    assert not [p for p in prod.tmp.iterdir() if p.name.startswith(".creating-")]


def test_workspace_scope_repository_binds_reads_and_published_pin(prod):
    from engine.v2.foundation import content_hash
    from engine.v2.ops.workspace.scope import SCOPE_HEAD_VERSION, size_scope_repository

    before = _state(prod)
    scope = "x" * 100
    with _create(prod) as workspace:
        view = size_scope_repository(workspace, scope, clock=prod.clock)
        pinned = view.resolve_pinned(scope)
        assert (pinned.snapshot_id, pinned.manifest_hash) == (
            prod.snap.snapshot_id, prod.snap.manifest_hash)
        assert view.resolve(pinned.snapshot_id).snapshot_id == pinned.snapshot_id
        stored = json.loads(workspace.store.read_verified(view.read_pin_ref))
        binding = stored["pin"]
        assert binding["schema_version"] == SCOPE_HEAD_VERSION
        assert binding["scope"] == scope
        assert binding["snapshot_id"] == prod.snap.snapshot_id
        assert binding["manifest_hash"] == prod.snap.manifest_hash
        assert binding["binding_id"] == view.binding_id
        assert view.binding_id == content_hash({k: v for k, v in binding.items()
                                                if k != "binding_id"})
        with pytest.raises(OpsError) as caught:
            view.resolve_pinned("other")
        assert _details(caught) == ("WORKSPACE_REFUSED", "scope_mismatch")
    assert _state(prod) == before


def test_workspace_scope_repository_refuses_invalid_scope_before_workspace_access():
    from engine.v2.ops.workspace.scope import size_scope_repository

    class UnreadableWorkspace:
        def source_snapshot(self):
            raise AssertionError("invalid scope must be checked first")

    for scope in ("", " size-prep", "x" * 101, "bad\x00scope", "bad\ud800scope"):
        with pytest.raises(OpsError) as caught:
            size_scope_repository(UnreadableWorkspace(), scope, clock=None)
        assert _details(caught) == ("WORKSPACE_REFUSED", "scope_invalid")


def test_workspace_scope_repository_refuses_invalid_workspace():
    from engine.v2.ops.workspace.scope import size_scope_repository

    invalid = SimpleNamespace(snapshot_id="snap_a", manifest_hash="hash_a",
                              source=object(),
                              source_snapshot=lambda: None, record_read_pin=lambda *a, **k: None)
    with pytest.raises(OpsError) as caught:
        size_scope_repository(invalid, "size-prep", clock=object())
    assert _details(caught) == ("WORKSPACE_REFUSED", "workspace_invalid")


def test_workspace_scope_repository_refuses_changed_captured_pin(prod):
    from engine.v2.ops.workspace.scope import size_scope_repository

    with _create(prod) as workspace:
        original = workspace.source_snapshot()
        from engine.v2.data.repository import Repository
        previous = Repository(workspace._source_conn).resolve(original.parent_snapshot_id)
        view = size_scope_repository(workspace, "size-prep", clock=prod.clock)
        workspace.source._only = previous.snapshot_id
        workspace.snapshot_id = previous.snapshot_id
        workspace.manifest_hash = previous.manifest_hash
        with pytest.raises(DataError) as caught:
            view.resolve_pinned("size-prep")
        assert caught.value.code == "SNAPSHOT_CLOSURE_BROKEN"
        assert caught.value.problem.details["reason"] == "pin_mismatch"


def test_workspace_scope_repository_preserves_manifest_mismatch_reason(prod):
    from engine.v2.ops.workspace.scope import size_scope_repository

    with _create(prod) as workspace:
        view = size_scope_repository(workspace, "size-prep", clock=prod.clock)
        workspace.manifest_hash = "changed-manifest"
        with pytest.raises(DataError) as caught:
            view.resolve_pinned("size-prep")
        assert caught.value.code == "SNAPSHOT_CLOSURE_BROKEN"
        assert caught.value.problem.details["reason"] == "manifest_mismatch"


def test_workspace_scope_repository_types_snapshot_read_failure():
    from engine.v2.ops.workspace.scope import size_scope_repository

    class BrokenWorkspace:
        snapshot_id = "snap_a"
        manifest_hash = "hash_a"
        source = SimpleNamespace(resolve=lambda _snapshot_id: None)

        def source_snapshot(self):
            raise OSError("synthetic read failure")

        def record_read_pin(self, name, document, *, clock):
            raise AssertionError("snapshot failure must precede publication")

    with pytest.raises(OpsError) as caught:
        size_scope_repository(BrokenWorkspace(), "size-prep", clock=object())
    assert _details(caught) == ("WORKSPACE_REFUSED", "storage_failure")


def test_workspace_scope_repository_preserves_typed_read_pin_failure():
    from engine.v2.ops.errors import fail
    from engine.v2.ops.workspace.scope import size_scope_repository

    snapshot = SimpleNamespace(snapshot_id="snap_a", manifest_hash="hash_a")
    refusal = fail("RESOURCE_UNAVAILABLE", "synthetic typed failure")

    class TypedFailureWorkspace:
        snapshot_id = "snap_a"
        manifest_hash = "hash_a"
        source = SimpleNamespace(resolve=lambda _snapshot_id: snapshot)

        def source_snapshot(self):
            return snapshot

        def record_read_pin(self, name, document, *, clock):
            raise refusal

    with pytest.raises(OpsError) as caught:
        size_scope_repository(TypedFailureWorkspace(), "size-prep", clock=object())
    assert caught.value is refusal


def test_workspace_scope_repository_types_unexpected_read_pin_failure():
    from engine.v2.ops.workspace.scope import size_scope_repository

    snapshot = SimpleNamespace(snapshot_id="snap_a", manifest_hash="hash_a")

    class BrokenWorkspace:
        snapshot_id = "snap_a"
        manifest_hash = "hash_a"
        source = SimpleNamespace(resolve=lambda _snapshot_id: snapshot)

        def source_snapshot(self):
            return snapshot

        def record_read_pin(self, name, document, *, clock):
            raise OSError("synthetic catalog failure")

    with pytest.raises(OpsError) as caught:
        size_scope_repository(BrokenWorkspace(), "size-prep", clock=object())
    assert _details(caught) == ("WORKSPACE_REFUSED", "storage_failure")
