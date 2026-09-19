from engine.v2.foundation import ArtifactStore, SystemClock, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.cli import whatif_action, whatif_result_action
from engine.v2.ops.checkpoints import register_artifact, artifact as load_artifact
from engine.v2.ops.catalog import transaction


def _ops_root(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    return tmp_path, conn, clock


def _payload():
    return {"request": {"event_id": "evt-1", "calendar_revision": "c",
                        "strategy_version": "STR-THRU", "deployment_id": "d",
                        "decision_clock_id": "entry-close", "requested_decision_at": "2026-09-16",
                        "snapshot_id": "s", "mode": "replay", "fill_model": {}},
            "native_inputs": {"context": {}, "features": {}, "forecast": {}, "geometry": None,
                              "pricing": None, "analogs": {}, "simulation": {}, "gate": {},
                              "chooser": {}, "diagnostics": {}, "source_ref": "s",
                              "stage_receipts": []}}


def test_whatif_action_rejects_malformed_payload(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    status, body = whatif_action(root, {"request": {}}, clock=clock)
    assert status == 400
    conn.close()


def test_whatif_action_submits_and_is_idempotent(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    payload = _payload()
    first_status, first_body = whatif_action(root, payload, clock=clock)
    second_status, second_body = whatif_action(root, payload, clock=clock)
    assert first_status == second_status == 202
    assert first_body["job_id"] == second_body["job_id"]
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    conn.close()


def test_whatif_result_action_reports_queued_before_a_worker_runs(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    payload = _payload()
    status, body = whatif_action(root, payload, clock=clock)
    assert status == 202
    fetch_status, fetch_body = whatif_result_action(root, body["job_id"], clock=clock)
    assert fetch_status == 202
    assert fetch_body["state"] == "queued"
    conn.close()


def test_whatif_result_action_unknown_job_is_an_ops_error_status(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    status, body = whatif_result_action(root, "job_doesnotexist", clock=clock)
    assert status == 400  # INVALID_REQUEST/"unknown job" -> validation category -> 400
    conn.close()