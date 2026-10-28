"""Experiment research workspace (platform PR 1): overlay isolation, closure and refusals.

Synthetic only: a tiny two-snapshot production catalog/store is built in ``tmp_path``.
(a) production catalog/store/heads hash identically after every workspace operation;
(b) identities re-resolve, and a missing or changed object refuses with a typed refusal;
(c) no supervisor or job launches from a workspace.
"""
from __future__ import annotations

import dataclasses
import hashlib
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
from engine.v2.ops.workspace.guard import refuse_workspace_execution  # noqa: E402
from engine.v2.ops.workspace.overlay import create_workspace, open_workspace  # noqa: E402
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
                 lambda: ws.source.scan(SimpleNamespace(snapshot_id="snap_foreign"),
                                        table_name="securities"),
                 lambda: ws.source.scan_population_bound("snap_foreign", table_name="securities"))
        for call in calls:
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
