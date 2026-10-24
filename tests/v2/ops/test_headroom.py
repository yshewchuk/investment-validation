"""Headroom check: status tiers, scratch limits, CLI exit codes.

Synthetic manifest through a tmp ops catalog, plus hand-inserted
attempts/resource_reservations rows for the memory limits; no market data and
no snapshots.
"""
import json
import types

import pytest

from engine.v2.foundation import ArtifactStore
from engine.v2.ops import cli, headroom
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import register_artifact, registered_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from tests.ops_support import FakeClock

MINIMUM_SCRATCH_BYTES = profile_named(DEFAULT_POLICY, "legacy_score").scratch_bytes
MATERIALIZE_SCRATCH_BYTES = profile_named(DEFAULT_POLICY, "materialize").scratch_bytes


@pytest.mark.parametrize("pct,expected", [
    (0.0, "OK"), (80.0, "OK"), (80.1, "WARN"), (95.0, "WARN"), (95.1, "FAIL"), (150.0, "FAIL"),
])
def test_status_boundaries(pct, expected):
    assert headroom.status_for(pct) == expected


@pytest.mark.parametrize("usage,limit,status,pct", [
    (8004, 10000, "WARN", 80.0),  # unrounded 80.04% rounds DOWN to 80.0 for display
    (9504, 10000, "FAIL", 95.0),  # unrounded 95.04% rounds DOWN to 95.0 for display
    (8000, 10000, "OK", 80.0),
    (9500, 10000, "WARN", 95.0),
])
def test_entry_classifies_the_unrounded_ratio(usage, limit, status, pct):
    """Status tiers run on the exact ratio; the 1-decimal pct is display only."""
    entry = headroom._entry("t", usage, limit, "bytes", "")
    assert (entry["status"], entry["pct"]) == (status, pct)


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


def test_mixed_statuses_aggregate_to_the_worst_exit_two(catalog):
    conn, store, clock = catalog
    manifest_id = publish(conn, store, clock, manifest_document(MINIMUM_SCRATCH_BYTES // 4),
                          "legacy_input_manifest.v1.0")
    _seed_finished_attempt(conn, attempt_id="ok1", profile="legacy_score",
                           memory_bytes=1000, peak=500, ended_at="2026-10-01T00:00:00Z")
    ok = report_for(catalog, nightly_plan_document(manifest_id))
    assert {item["status"] for item in ok["limits"]} == {"OK"}
    assert (ok["status"], ok["exit_code"]) == ("OK", 0)
    _seed_finished_attempt(conn, attempt_id="warn1", profile="validation",
                           memory_bytes=1000, peak=850, ended_at="2026-10-02T00:00:00Z")
    _seed_finished_attempt(conn, attempt_id="fail1", profile="materialize",
                           memory_bytes=1000, peak=990, ended_at="2026-10-03T00:00:00Z")
    mixed = report_for(catalog, nightly_plan_document(manifest_id))
    statuses = {item["status"] for item in mixed["limits"]}
    assert {"OK", "WARN", "FAIL"} <= statuses
    assert (mixed["status"], mixed["exit_code"]) == ("FAIL", 2)


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


def test_snapshot_plan_scratch_entry_uses_pinned_estimate(catalog):
    estimate = MATERIALIZE_SCRATCH_BYTES // 4
    document = report_for(catalog, nightly_plan_document(
        input_mode="snapshot", snapshot_inputs={"scratch_estimate_bytes": estimate}))
    scratch = [item for item in document["limits"]
               if item["name"].startswith("scratch_bytes:")]
    assert [item["name"] for item in scratch] == ["scratch_bytes:materialize"]
    entry = scratch[0]
    assert entry["usage"] == estimate
    assert entry["limit"] == MATERIALIZE_SCRATCH_BYTES
    assert entry["status"] == "OK"
    assert "snapshot scratch_estimate_bytes" in entry["detail"]


@pytest.mark.parametrize("manifest", [
    {},                                      # file_refs missing (must not default to 0)
    {"file_refs": {"byte_size": 5}},         # not a list
    {"file_refs": [{"size": 5}]},            # item lacks byte_size (no KeyError)
    {"file_refs": ["ref"]},                  # item is not an object
    {"file_refs": [{"byte_size": True}]},    # bool is not an int here
    {"file_refs": [{"byte_size": -1}]},      # negative
    {"file_refs": [{"byte_size": "5"}]},     # non-int
    {"file_refs": [{"byte_size": 5.5}]},     # non-int
], ids=["missing", "not-a-list", "no-byte-size", "non-object-item",
         "bool", "negative", "str", "float"])
def test_malformed_manifest_scratch_input_refuses_invalid_request(catalog, manifest):
    conn, store, clock = catalog
    manifest_id = publish(conn, store, clock, manifest, "legacy_input_manifest.v1.0")
    with pytest.raises(OpsError) as excinfo:
        report_for(catalog, nightly_plan_document(manifest_id))
    assert excinfo.value.code == "INVALID_REQUEST"


def test_empty_manifest_file_refs_is_zero_usage_ok(catalog):
    conn, store, clock = catalog
    manifest_id = publish(conn, store, clock, {"file_refs": []},
                          "legacy_input_manifest.v1.0")
    document = report_for(catalog, nightly_plan_document(manifest_id))
    scratch = [item for item in document["limits"] if item["name"].startswith("scratch_bytes")]
    assert scratch and all(item["usage"] == 0 and item["status"] == "OK" for item in scratch)


@pytest.mark.parametrize("snapshot_inputs", [
    None, {},                                      # estimate missing (must not default to 0)
    {"scratch_estimate_bytes": None},
    {"scratch_estimate_bytes": True},
    {"scratch_estimate_bytes": -1},
    {"scratch_estimate_bytes": "1024"},
    {"scratch_estimate_bytes": 1.5},
], ids=["absent", "empty", "null", "bool", "negative", "str", "float"])
def test_bad_snapshot_scratch_estimate_refuses_invalid_request(catalog, snapshot_inputs):
    plan = nightly_plan_document(input_mode="snapshot")
    if snapshot_inputs is not None:
        plan["snapshot_inputs"] = snapshot_inputs
    with pytest.raises(OpsError) as excinfo:
        report_for(catalog, plan)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.message == \
        "snapshot scratch_estimate_bytes must be a non-negative integer"


def test_not_covered_lists_the_static_names(catalog):
    document = report_for(catalog, nightly_plan_document())
    assert document["not_covered"] == list(headroom._NOT_COVERED)
    assert len(document["not_covered"]) == len(headroom._NOT_COVERED)
    for name in headroom._NOT_COVERED:
        assert name in document["not_covered"]


def test_rss_cap_entry_only_when_requested(catalog):
    document = report_for(catalog, nightly_plan_document())
    assert not [item for item in document["limits"] if item["name"] == "rss_cap_gb"]
    peak = next(item for item in document["limits"] if item["name"] == "memory_peak")
    assert peak["status"] == "UNMEASURED"
    assert "no finished attempt has a recorded memory peak" in peak["detail"]
    document = report_for(catalog, nightly_plan_document(), max_rss_gb=1.5)
    entry = next(item for item in document["limits"] if item["name"] == "rss_cap_gb")
    assert entry["limit"] == int(1.5 * 1024 ** 3)
    assert entry["status"] == "UNMEASURED"
    assert entry["pct"] is None
    assert "no finished attempt has a recorded memory peak" in entry["detail"]
    assert document["status"] == "UNMEASURED"
    assert document["exit_code"] == 1


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
    args = types.SimpleNamespace(plan=plan_id, max_rss_gb=None)
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def test_unknown_plan_id_refuses_invalid_request(catalog):
    conn, store, clock = catalog
    with pytest.raises(OpsError) as excinfo:
        headroom.headroom_command(_headroom_args(conn, store, clock, "no-such-plan"),
                                  store.root, conn, clock)
    assert excinfo.value.code == "INVALID_REQUEST"


def test_oversize_ref_metadata_refuses_invalid_request(catalog, monkeypatch):
    """registered_artifact reads the ref row bounded by _MAX_METADATA_BYTES;
    an oversize (or malformed) ref row maps to the same unreadable-artifact
    INVALID_REQUEST as an unknown id, not to RESOURCE_LIMIT_EXCEEDED."""
    conn, store, clock = catalog
    plan_id = publish(conn, store, clock, nightly_plan_document())
    monkeypatch.setattr(headroom, "_MAX_METADATA_BYTES", 8)
    with pytest.raises(OpsError) as excinfo:
        headroom.headroom_command(_headroom_args(conn, store, clock, plan_id),
                                  store.root, conn, clock)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.message == "the plan artifact is not readable"


def test_non_object_plan_json_refuses_invalid_request(catalog):
    conn, store, clock = catalog
    plan_id = publish(conn, store, clock, [1, 2, 3])
    with pytest.raises(OpsError) as excinfo:
        headroom.headroom_command(_headroom_args(conn, store, clock, plan_id),
                                  store.root, conn, clock)
    assert excinfo.value.code == "INVALID_REQUEST"


def test_tampered_plan_artifact_refuses_integrity_failed(catalog):
    conn, store, clock = catalog
    plan_id = publish(conn, store, clock, nightly_plan_document())
    path = store.verify(registered_artifact(conn, plan_id))
    path.chmod(0o600)
    path.write_bytes(b'{"kind": "tampered"}')
    with pytest.raises(OpsError) as excinfo:
        headroom.headroom_command(_headroom_args(conn, store, clock, plan_id),
                                  store.root, conn, clock)
    assert excinfo.value.code == "INTEGRITY_FAILED"


def test_oversize_json_artifact_refuses_resource_limit(catalog, monkeypatch):
    conn, store, clock = catalog
    plan_id = publish(conn, store, clock, nightly_plan_document())
    monkeypatch.setattr(headroom, "_MAX_JSON_BYTES", 1)
    byte_accesses = []

    def _no_byte_access(*args, **kwargs):
        byte_accesses.append("byte access")
        raise AssertionError("oversize refusal must not hash or read artifact bytes")
    # headroom_command builds its own ArtifactStore(root), so the class methods
    # are the ones actually used; an instance patch on this fixture store is not.
    monkeypatch.setattr(ArtifactStore, "verify", _no_byte_access)
    monkeypatch.setattr(ArtifactStore, "read_verified", _no_byte_access)
    with pytest.raises(OpsError) as excinfo:
        headroom.headroom_command(_headroom_args(conn, store, clock, plan_id),
                                  store.root, conn, clock)
    assert excinfo.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert excinfo.value.problem.message == "the plan artifact exceeds 1 bytes"
    assert byte_accesses == []  # verify/read_verified were never called


@pytest.mark.parametrize("plan", [[1, 2, 3], None])
def test_headroom_report_refuses_non_object_plan(catalog, plan):
    with pytest.raises(OpsError) as excinfo:
        report_for(catalog, plan)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.message == "plan must be a JSON object"


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), float("-inf"),
                                 0, 0.0, -1, -0.5, "1.5", 2, []])
def test_headroom_report_validates_max_rss_gb_directly(catalog, bad):
    """The engine-level guard is independent of cli.py's argparse type: every
    non-positive, non-finite or non-float value is refused as INVALID_REQUEST."""
    with pytest.raises(OpsError) as excinfo:
        report_for(catalog, nightly_plan_document(), max_rss_gb=bad)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.message == \
        "max_rss_gb must be a finite number greater than 0"


def test_headroom_report_accepts_valid_max_rss_gb(catalog):
    conn, store, _clock = catalog
    assert headroom.headroom_report(conn, store.root, nightly_plan_document(),
                                    max_rss_gb=None)["exit_code"] == 1  # UNMEASURED limits
    by_name = {item["name"]: item
               for item in report_for(catalog, nightly_plan_document(),
                                      max_rss_gb=0.25)["limits"]}
    assert by_name["rss_cap_gb"]["limit"] == int(0.25 * 1024 ** 3)


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


def test_rss_cap_uses_the_latest_epoch_peak_not_the_global_max(catalog):
    """Two supervisor epochs share one profile: the older epoch's higher peak
    must not feed the rss cap; only the epoch of the latest finished attempt does."""
    conn, _store, _clock = catalog
    _seed_finished_attempt(conn, attempt_id="a1", profile="legacy_score",
                           memory_bytes=10000, peak=9000, ended_at="2026-10-01T00:00:00Z")
    _seed_finished_attempt(conn, attempt_id="a2", profile="legacy_score",
                           memory_bytes=10000, peak=1000, ended_at="2026-10-02T00:00:00Z",
                           epoch="ep-2", boot="boot-2")
    by_name = {item["name"]: item
               for item in headroom._memory_entries(conn, max_rss_gb=1.0)}
    assert by_name["memory_peak:legacy_score"]["usage"] == 9000  # both epochs count
    assert by_name["rss_cap_gb"]["usage"] == 1000  # only ep-2, the latest epoch
    assert by_name["rss_cap_gb"]["status"] == "OK"


def test_entries_have_no_projection_yet(catalog):
    """No covered limit has growth history: the key is present and always None."""
    conn, store, clock = catalog
    manifest_id = publish(conn, store, clock, manifest_document(1000),
                          "legacy_input_manifest.v1.0")
    _seed_finished_attempt(conn, attempt_id="a1", profile="legacy_score",
                           memory_bytes=1000, peak=900, ended_at="2026-10-01T00:00:00Z")
    document = report_for(catalog, nightly_plan_document(manifest_id))
    assert document["limits"]
    for item in document["limits"]:
        assert item["projected_days_to_limit"] is None

