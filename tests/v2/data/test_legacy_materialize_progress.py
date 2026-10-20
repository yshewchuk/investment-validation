"""Bounded stderr progress for legacy materialization (#1018): never stdout, never per item."""
from __future__ import annotations

from engine.v2.data import legacy_adapter, legacy_materialization, materialize_progress
from engine.v2.data.repository import Repository
from tests.v2.data.test_legacy_materialization import (
    _build_request,
    _build_snapshot,
    _snapshot_object_ref,
)


def test_materialize_progress_goes_to_stderr_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(materialize_progress, "_PROGRESS_INTERVAL_S", 0.0)
    conn, store, snap = _build_snapshot(tmp_path)
    try:
        repository = Repository(conn, store)
        request = _build_request(repository, snap, _snapshot_object_ref(store), store)
        dest_root = tmp_path / "materialized"

        manifest = legacy_adapter.materialize(repository, store, request, dest_root)

        assert manifest
        captured = capsys.readouterr()
        assert "[materialize]" in captured.err
        assert "tables" in captured.err
        assert "[materialize]" not in captured.out
        # Start and done lines bracket the run with elapsed seconds.
        assert "start" in captured.err
        assert "done in" in captured.err
        # The final done==total tick for both write and validation phases.
        assert "6/6 tables" in captured.err
    finally:
        conn.close()
