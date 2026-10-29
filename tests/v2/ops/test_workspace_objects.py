"""Workspace closure object admission: exact named set, typed refusals, no production write."""
from __future__ import annotations

import pytest

from tests.data_scan_support import fake_hash
from tests.v2.ops.test_research_workspace import (  # noqa: F401
    _create,
    _details,
    _state,
    prod,
)

from engine.v2.ops.errors import OpsError
from engine.v2.ops.workspace.catalog_rows import copy_closure_rows
from engine.v2.ops.workspace.objects import admit_closure_objects


def _key(path):
    return "/".join(path.parts[-3:])


def _admitted(ws):
    return sorted(_key(p) for p in (ws.store.root / "objects").rglob("*") if p.is_file())


def _object_files(prod):
    return sorted(_key(p) for p in (prod.store / "objects").rglob("*") if p.is_file())


def test_admits_exactly_the_pinned_snapshot_objects(prod):
    before = _state(prod)
    with _create(prod) as ws:
        pin_objects = _admitted(ws)
        copy_closure_rows(ws)
        result = admit_closure_objects(ws)
        assert result["snapshot_id"] == prod.snap.snapshot_id
        assert (result["objects"], result["copied"], result["present"]) == (1, 1, 0)
        assert result["bytes"] == prod.object.stat().st_size
        admitted = ws.store.root / "objects" / prod.object.parent.name / prod.object.name
        assert admitted.read_bytes() == prod.object.read_bytes()
        # the ancestor snapshot's object is production data outside the named set
        assert len(_object_files(prod)) == 2
        assert sorted(set(_admitted(ws)) - set(pin_objects)) == [_key(prod.object)]
        assert not list((ws.store.root / "objects").rglob("*.part"))
        again = admit_closure_objects(ws)
        assert (again["copied"], again["present"], again["bytes"]) == (0, 1, 0)
    assert _state(prod) == before


def test_requires_copied_closure_rows(prod):
    with _create(prod) as ws:
        with pytest.raises(OpsError) as caught:
            admit_closure_objects(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "workspace_invalid")


def test_missing_source_object_refuses_with_typed_code(prod):
    with _create(prod) as ws:
        copy_closure_rows(ws)
        before = _admitted(ws)
        prod.object.unlink()
        with pytest.raises(OpsError) as caught:
            admit_closure_objects(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "object_missing")
        assert _admitted(ws) == before


def test_changed_source_object_refuses_and_leaves_no_partial(prod):
    with _create(prod) as ws:
        copy_closure_rows(ws)
        before = _admitted(ws)
        data = bytearray(prod.object.read_bytes())
        data[-1] ^= 0xFF
        prod.object.write_bytes(bytes(data))
        with pytest.raises(OpsError) as caught:
            admit_closure_objects(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "object_changed")
        assert _admitted(ws) == before
        assert not list((ws.store.root / "objects").rglob("*.part"))


def test_raw_receipt_rows_in_the_workspace_refuse_before_any_copy(prod):
    with _create(prod) as ws:
        copy_closure_rows(ws)
        ws.conn.execute(
            "INSERT INTO data_raw_receipts (raw_receipt_id, source, endpoint, request_hash, "
            "raw_hash, artifact_ref_json, response_kind, request_json, response_meta_json, "
            "received_at) VALUES ('raw_x', 's', 'e', ?, ?, '{}', 'complete', '{}', '{}', 't')",
            (fake_hash("q"), fake_hash("r")))
        ws.conn.commit()
        before = _admitted(ws)
        with pytest.raises(OpsError) as caught:
            admit_closure_objects(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "receipts_present")
        assert _admitted(ws) == before


def test_wrong_sized_destination_is_never_repaired(prod):
    with _create(prod) as ws:
        copy_closure_rows(ws)
        destination = ws.store.root / "objects" / prod.object.parent.name / prod.object.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"short")
        with pytest.raises(OpsError) as caught:
            admit_closure_objects(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "destination_conflict")
        assert destination.read_bytes() == b"short"


def test_same_size_corrupt_destination_is_refused(prod):
    with _create(prod) as ws:
        copy_closure_rows(ws)
        destination = ws.store.root / "objects" / prod.object.parent.name / prod.object.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * prod.object.stat().st_size)
        with pytest.raises(OpsError) as caught:
            admit_closure_objects(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "object_changed")
        assert destination.read_bytes() == b"x" * prod.object.stat().st_size


def test_source_object_that_grew_is_refused_without_partial(prod):
    with _create(prod) as ws:
        copy_closure_rows(ws)
        before = _admitted(ws)
        prod.object.write_bytes(prod.object.read_bytes() + b"extra")
        with pytest.raises(OpsError) as caught:
            admit_closure_objects(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "object_changed")
        assert _admitted(ws) == before
        assert not list((ws.store.root / "objects").rglob("*.part"))
