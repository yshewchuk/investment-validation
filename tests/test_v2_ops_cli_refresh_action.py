"""Unit tests for ``engine.v2.ops.cli.refresh_action`` (no HTTP)."""
from __future__ import annotations

import json

from engine.v2.foundation import ArtifactStore, SystemClock
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.cli import refresh_action


def _ops_root(tmp_path):
    root = tmp_path
    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    return root, conn, clock


def _publish(root, conn, clock, value, schema_ref):
    store = ArtifactStore(root)
    ref = store.publish_bytes(json.dumps(value, sort_keys=True).encode(), schema_ref=schema_ref)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref.artifact_id


def test_refresh_action_missing_plan_ref(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    status, body = refresh_action(root, {}, clock=clock)
    assert status == 400
    conn.close()


def test_refresh_action_plan_ref_not_found(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    status, body = refresh_action(root, {"plan_ref": "nope"}, clock=clock)
    assert status == 404
    conn.close()


def test_refresh_action_rejects_non_nightly_plan(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    ref = _publish(root, conn, clock, {"kind": "artifact_check"},
                   schema_ref="operations_plan.v1.0")
    status, body = refresh_action(root, {"plan_ref": ref}, clock=clock)
    assert status == 400
    conn.close()


def test_refresh_action_duplicate_submit_is_side_effect_free(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    ref = _publish(root, conn, clock,
                   {"kind": "nightly", "blocked_prerequisites": ["upstream_job"]},
                   schema_ref="operations_plan.v1.0")
    first = refresh_action(root, {"plan_ref": ref}, clock=clock)
    second = refresh_action(root, {"plan_ref": ref}, clock=clock)
    assert first[0] == second[0] == 400
    assert first[1]["problem"]["code"] == "INVALID_REQUEST"
    assert first[1] == second[1]
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    conn.close()