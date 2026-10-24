"""Headroom check: status tiers, projection, scratch limits, CLI exit codes.

Synthetic manifest through a tmp ops catalog only; no market data, no
attempts rows, no snapshots -- the price-history and memory limits keep
their documented skip behaviour here.
"""
import json

import pytest

from engine.v2.data.price_history_query import _RESULT_CAP
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import cli, headroom
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import register_artifact
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
    assert all(item["status"] in ("OK", "WARN") for item in warn["limits"])
    assert any(item["pct"] == 85.0 for item in warn["limits"])

    manifest_id = publish(conn, store, clock,
                          manifest_document(int(0.99 * MINIMUM_SCRATCH_BYTES)),
                          "legacy_input_manifest.v1.0")
    fail = report_for(catalog, nightly_plan_document(manifest_id))
    assert (fail["status"], fail["exit_code"]) == ("FAIL", 2)


def test_plan_without_any_scratch_input_skips_with_note(catalog):
    document = report_for(catalog, nightly_plan_document())
    assert document["status"] == "OK"
    scratch = [item for item in document["limits"] if item["name"].startswith("scratch_bytes")]
    assert len(scratch) == 1
    assert "neither" in scratch[0]["detail"]


def test_price_history_skipped_without_snapshot_inputs(catalog):
    document = report_for(catalog, nightly_plan_document())
    entry = next(item for item in document["limits"]
                 if item["name"] == "price_history_result_rows")
    assert entry["limit"] == _RESULT_CAP
    assert entry["unit"] == "rows"
    assert entry["status"] == "OK"
    assert entry["projected_days_to_limit"] is None
    assert "skipped" in entry["detail"]


def test_not_covered_lists_the_static_names(catalog):
    document = report_for(catalog, nightly_plan_document())
    assert "chains._RESULT_CAP (2_000_000)" in document["not_covered"]
    assert "serving.projections._LOOKUP_RESULT_CAP" in document["not_covered"]
    assert len(document["not_covered"]) == 8


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
