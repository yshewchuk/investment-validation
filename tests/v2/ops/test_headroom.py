"""Headroom check: status tiers, projection, scratch limits, CLI exit codes.

Synthetic manifest through a tmp ops catalog, plus hand-inserted
attempts/resource_reservations rows for the memory limits; no market data and
no snapshots -- the price-history entry covers its projection through a stub
repository.
"""
import json
import types

import pytest

from engine.v2.data.price_history_query import _RESULT_CAP
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import cli, headroom
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from tests.ops_support import FakeClock

MINIMUM_SCRATCH_BYTES = profile_named(DEFAULT_POLICY, "legacy_score").scratch_bytes


@pytest.mark.parametrize("pct,expected", [
    (0.0, "OK"), (80.0, "OK"), (80.1, "WARN"), (95.0, "WARN"), (95.1, "FAIL"), (150.0, "FAIL"),
])
def test_status_boundaries(pct, expected):
    assert headroom.status_for(pct) == expected


def test_project_days_growing():
    assert headroom.project_days([(0.0, 10), (1.0, 20)], 30) == pytest.approx(1.0)


def test_project_days_flat_or_declining_is_none():
    assert headroom.project_days([(0.0, 10), (1.0, 10)], 30) is None
    assert headroom.project_days([(0.0, 20), (1.0, 10)], 30) is None


def test_project_days_needs_two_distinct_dates():
    assert headroom.project_days([(0.0, 10)], 30) is None
    assert headroom.project_days([(0.0, 10), (0.0, 20)], 30) is None


@pytest.fixture
def catalog(tmp_path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    store = ArtifactStore(tmp_path)
    yield conn, store, clock
    conn.close()


def publish(conn, store, clock, document, schema_ref="operations_plan.v1.0"):
    ref = store.publish_bytes(json.dumps(document, sort_keys=True).encode(), schema_ref=schema_ref)
    register_artifact(conn, ref, None, clock)
    return ref.artifact_id


def nightly_plan_document(manifest_id=None, **extra):
    plan = {"kind": "nightly", "input_mode": "legacy", "session": "2026-10-23",
            "year_start": 2024, "year_end": 2026}
    if manifest_id:
        plan["input_manifest_ref"] = manifest_id
    plan.update(extra)
    return plan


def manifest_document(total_bytes):
    return {"file_refs": [{"byte_size": total_bytes}]}


def report_for(catalog, plan, **kwargs):
    conn, store, _clock = catalog
    return headroom.headroom_report(conn, store.root, plan, **kwargs)


def test_scratch_reports_one_entry_per_staging_profile(catalog):
    conn, store, clock = catalog
    total = MINIMUM_SCRATCH_BYTES // 2
    manifest_id = publish(conn, store, clock, manifest_document(total), "legacy_input_manifest.v1.0")
    document = report_for(catalog, nightly_plan_document(manifest_id))
    by_name = {item["name"]: item for item in document["limits"]}
    assert set(by_name) >= {
        "scratch_bytes:legacy_score", "scratch_bytes:validation",
        "scratch_bytes:legacy_rebuild", "scratch_bytes:model_evidence",
        "scratch_bytes:projection"}
    entry = by_name["scratch_bytes:legacy_score"]
    assert entry["usage"] == total
    assert entry["limit"] == MINIMUM_SCRATCH_BYTES
    assert entry["status"] == "OK"


def test_overall_status_and_exit_code_are_the_worst(catalog):
    conn, store, clock = catalog
    manifest_id = publish(conn, store, clock,
                          manifest_document(int(0.85 * MINIMUM_SCRATCH_BYTES)),
                          "legacy_input_manifest.v1.0")
    warn = report_for(catalog, nightly_plan_document(manifest_id))
    assert (warn["status"], warn["exit_code"]) == ("WARN", 1)
    assert all(item["status"] in ("OK", "WARN", "UNMEASURED") for item in warn["limits"])
    assert any(item["pct"] == 85.0 for item in warn["limits"])

    manifest_id = publish(conn, store, clock,
                          manifest_document(int(0.99 * MINIMUM_SCRATCH_BYTES)),
                          "legacy_input_manifest.v1.0")
    fail = report_for(catalog, nightly_plan_document(manifest_id))
    assert (fail["status"], fail["exit_code"]) == ("FAIL", 2)


def test_plan_without_any_scratch_input_is_unmeasured_not_ok(catalog):
    document = report_for(catalog, nightly_plan_document())
    assert document["status"] == "UNMEASURED"
    assert document["exit_code"] == 1
    scratch = [item for item in document["limits"] if item["name"].startswith("scratch_bytes")]
    assert len(scratch) == 1
    assert scratch[0]["status"] == "UNMEASURED"
    assert scratch[0]["pct"] is None
    assert "neither" in scratch[0]["detail"]


def test_non_nightly_plan_scratch_is_unmeasured(catalog):
    document = report_for(catalog, {"kind": "experiment"})
    scratch = next(item for item in document["limits"] if item["name"] == "scratch_bytes")
    assert scratch["status"] == "UNMEASURED"
    assert "not a nightly plan" in scratch["detail"]


def test_price_history_skipped_without_snapshot_inputs(catalog):
    document = report_for(catalog, nightly_plan_document())
    entry = next(item for item in document["limits"]
                 if item["name"] == "price_history_result_rows")
    assert entry["limit"] == _RESULT_CAP
    assert entry["unit"] == "rows"
    assert entry["status"] == "UNMEASURED"
    assert entry["pct"] is None
    assert entry["projected_days_to_limit"] is None
    assert "skipped" in entry["detail"]


def test_not_covered_lists_the_static_names(catalog):
    document = report_for(catalog, nightly_plan_document())
    assert "chains._RESULT_CAP (2_000_000)" in document["not_covered"]
    assert "serving.projections._LOOKUP_RESULT_CAP" in document["not_covered"]
    assert "models.inventory MAX_ARTIFACT_BYTES/MAX_REGISTRY_BYTES" in document["not_covered"]
    assert "iv_crush_inputs.MAX_TARGET_OBJECT_BYTES" in document["not_covered"]
    assert "event_history.MAX_OUTCOME_OBJECT_BYTES" in document["not_covered"]
    assert len(document["not_covered"]) == 11


def test_rss_cap_entry_only_when_requested(catalog):
    document = report_for(catalog, nightly_plan_document())
    assert not [item for item in document["limits"] if item["name"] == "rss_cap_gb"]
    document = report_for(catalog, nightly_plan_document(), max_rss_gb=1.5)
    entry = next(item for item in document["limits"] if item["name"] == "rss_cap_gb")
    assert entry["limit"] == int(1.5 * 1024 ** 3)
    assert entry["usage"] == 0
    assert entry["status"] == "OK"


def test_cli_headroom_returns_document_exit_code(tmp_path, catalog, capsys):
    conn, store, clock = catalog
    manifest_id = publish(conn, store, clock,
                          manifest_document(int(0.99 * MINIMUM_SCRATCH_BYTES)),
                          "legacy_input_manifest.v1.0")
    plan_id = publish(conn, store, clock, nightly_plan_document(manifest_id))
    capsys.readouterr()
    assert cli.main(["--root", str(tmp_path), "headroom", "--plan", plan_id]) == 2
    printed = capsys.readouterr()
    document = json.loads(printed.out)
    assert document["schema"] == "headroom.v1"
    assert document["status"] == "FAIL"
    lines = [line for line in printed.err.splitlines() if line.startswith("[headroom] ")]
    assert any(line.startswith("[headroom] FAIL scratch_bytes:") for line in lines)
    assert any(line.startswith("[headroom] start") for line in lines)
    assert any(line.startswith("[headroom] done in ") for line in lines)


# ------------------------------------------------------------- refusal (C)


def _headroom_args(conn, store, clock, plan_id, **extra):
    args = types.SimpleNamespace(plan=plan_id, max_rss_gb=None, history=5)
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def test_unknown_plan_id_refuses_invalid_request(catalog):
    conn, store, clock = catalog
    with pytest.raises(OpsError) as excinfo:
        headroom.headroom_command(_headroom_args(conn, store, clock, "no-such-plan"),
                                  store.root, conn, clock)
    assert excinfo.value.code == "INVALID_REQUEST"


def test_non_object_plan_json_refuses_invalid_request(catalog):
    conn, store, clock = catalog
    plan_id = publish(conn, store, clock, [1, 2, 3])
    with pytest.raises(OpsError) as excinfo:
        headroom.headroom_command(_headroom_args(conn, store, clock, plan_id),
                                  store.root, conn, clock)
    assert excinfo.value.code == "INVALID_REQUEST"


# ---------------------------------------------------- real rows (memory, G)


def _seed_finished_attempt(conn, *, attempt_id, profile, memory_bytes, peak,
                           ended_at, epoch="ep-1", boot="boot-1"):
    conn.execute("INSERT OR IGNORE INTO supervisor_epochs "
                 "(epoch_id, boot_id, pid, started_at) VALUES (?, ?, 1, ?)",
                 (epoch, boot, ended_at))
    conn.execute("INSERT INTO jobs (job_id, namespace, idempotency_key, request_digest, "
                 "principal, kind, spec_json, resource_class, checkpoint_contract_ref, "
                 "retry_json, state, priority, max_attempts, created_at, updated_at) "
                 "VALUES (?, 'ns', ?, 'd', 'op', 'k', '{}', 'rc', 'cc', '{}', "
                 "'succeeded', 0, 1, ?, ?)", (attempt_id, attempt_id, ended_at, ended_at))
    conn.execute("INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, "
                 "supervisor_epoch, host_boot_id, state, process_state, resources_json, "
                 "created_at, lease_expires_at, ended_at, memory_peak_bytes) "
                 "VALUES (?, ?, 1, 1, ?, ?, 'succeeded', 'verified_dead', '{}', ?, ?, ?, ?)",
                 (attempt_id, attempt_id, epoch, boot, ended_at, ended_at, ended_at, peak))
    conn.execute("INSERT INTO resource_reservations (attempt_id, policy_version, profile, "
                 "memory_bytes, scratch_bytes, heavy, disk_heavy, measured, created_at) "
                 "VALUES (?, 'v1', ?, ?, 0, 0, 0, 0, ?)",
                 (attempt_id, profile, memory_bytes, ended_at))
    conn.commit()


def test_memory_entries_group_by_profile_and_scope_rss_to_latest_epoch(catalog):
    conn, store, clock = catalog
    _seed_finished_attempt(conn, attempt_id="a1", profile="legacy_score",
                           memory_bytes=1000, peak=900, ended_at="2026-10-01T00:00:00Z")
    _seed_finished_attempt(conn, attempt_id="b1", profile="validation", memory_bytes=2000,
                           peak=500, ended_at="2026-10-02T00:00:00Z", epoch="ep-2",
                           boot="boot-2")
    _seed_finished_attempt(conn, attempt_id="b2", profile="validation", memory_bytes=2000,
                           peak=600, ended_at="2026-10-03T00:00:00Z", epoch="ep-2",
                           boot="boot-2")
    by_name = {item["name"]: item
               for item in headroom._memory_entries(conn, max_rss_gb=1.0)}
    assert by_name["memory_peak:legacy_score"]["usage"] == 900
    assert by_name["memory_peak:legacy_score"]["limit"] == 1000
    assert by_name["memory_peak:validation"]["usage"] == 600  # MAX over the profile group
    # the rss peak follows the most recent finished attempt's epoch (ep-2), not a
    # global max: b1/b2 (500, 600) count, a1's higher 900 in ep-1 does not.
    assert by_name["rss_cap_gb"]["usage"] == 600
    assert by_name["rss_cap_gb"]["limit"] == int(1.0 * 1024 ** 3)


# ------------------------------------------------- price history (projection)


class _StubRef:
    table_versions = {"price_history": 1}


class _StubRecord:
    def __init__(self, partition_key, row_count):
        self.partition_key = partition_key
        self.row_count = row_count


class _StubRepository:
    def __init__(self, conn, store, pin_snapshots=True):
        pass

    def resolve(self, snapshot_id):
        return _StubRef()

    def fragment_records(self, ref, table):
        return [_StubRecord("AAPL", 1000), _StubRecord("MSFT", 400)]


def test_price_history_projection_from_history_points(catalog, monkeypatch):
    monkeypatch.setattr(headroom, "Repository", _StubRepository)
    monkeypatch.setattr(headroom, "_history_points",
                        lambda conn, repository, snapshot_id, history: [(0.0, 100), (10.0, 1000)])
    document = report_for(catalog, nightly_plan_document(snapshot_inputs={"snapshot_id": "snap"}))
    entry = next(item for item in document["limits"]
                 if item["name"] == "price_history_result_rows")
    assert entry["status"] == "OK"
    assert entry["usage"] == 1000  # largest partition (AAPL)
    assert entry["projected_days_to_limit"] is not None

