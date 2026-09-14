"""D19/P2-C02: the render memory report -- synthetic, seeded catalog rows only.

No real Scorer, no real nightly: every attempt/job/reservation row below is
inserted directly against the real ops schema (``engine.v2.ops.bootstrap.
open_catalog``), so the report is exercised against the exact columns
production code reads (``attempts.memory_peak_bytes``, ``started_at``,
``ended_at``, ``resource_reservations.memory_bytes``) without needing a real
Scorer/FeatureContext load anywhere in this file.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from checks.rearchitecture_phase2_render_memory import (
    KEEP_THRESHOLD,
    main,
    render_memory_report,
)
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.profiles import GIB

ROOT = Path(__file__).resolve().parents[1]


def _seed_job(conn, job_id, kind):
    conn.execute(
        "INSERT INTO jobs (job_id, namespace, idempotency_key, request_digest, principal, "
        "kind, spec_hash, spec_json, resource_class, checkpoint_contract_ref, retry_json, "
        "state, priority, max_attempts, created_at, updated_at, fence, attempt_count) "
        "VALUES (?, 'shadow', ?, 'digest', 'operator', ?, NULL, '{}', 'projection', "
        "'legacy_action.v1.0', '{}', 'succeeded', 0, 1, 't0', 't0', 1, 1)",
        (job_id, job_id, kind))


def _seed_attempt(conn, attempt_id, job_id, *, epoch_id, started_at, ended_at, peak_bytes):
    conn.execute(
        "INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, supervisor_epoch, "
        "host_boot_id, state, process_state, resources_json, created_at, started_at, "
        "lease_expires_at, ended_at, exit_code, memory_peak_bytes) VALUES "
        "(?, ?, 1, 1, ?, 'boot', 'succeeded', 'exited', '{}', ?, ?, ?, ?, 0, ?)",
        (attempt_id, job_id, epoch_id, started_at, started_at, ended_at, ended_at, peak_bytes))


def _seed_reservation(conn, attempt_id, *, profile, memory_bytes):
    conn.execute(
        "INSERT INTO resource_reservations (attempt_id, policy_version, profile, memory_bytes, "
        "scratch_bytes, heavy, disk_heavy, measured, created_at) VALUES "
        "(?, 'v1', ?, ?, 0, 0, 0, 0, 't0')",
        (attempt_id, profile, memory_bytes))


@pytest.fixture
def seeded_root(tmp_path):
    """One render attempt (2 GiB reservation), one overlapping neighbor, one
    attempt that ended before the render attempt started (not overlapping).
    """
    root = tmp_path / "ops_root"
    root.mkdir()
    conn = open_catalog(root / "catalog.sqlite", clock=_clock())
    conn.execute("INSERT INTO supervisor_epochs VALUES ('epoch1', 'boot', 1, 't0', NULL)")
    _seed_job(conn, "job_render", "legacy_render")
    _seed_job(conn, "job_neighbor", "legacy_score")
    _seed_job(conn, "job_before", "legacy_finality")
    _seed_attempt(conn, "att_render", "job_render", epoch_id="epoch1",
                 started_at="2026-09-14T10:00:00.000000Z",
                 ended_at="2026-09-14T10:30:00.000000Z", peak_bytes=int(1.4 * GIB))
    _seed_reservation(conn, "att_render", profile="projection", memory_bytes=2 * GIB)
    _seed_attempt(conn, "att_neighbor", "job_neighbor", epoch_id="epoch1",
                 started_at="2026-09-14T10:10:00.000000Z",
                 ended_at="2026-09-14T10:20:00.000000Z", peak_bytes=int(0.5 * GIB))
    _seed_attempt(conn, "att_before", "job_before", epoch_id="epoch1",
                 started_at="2026-09-14T09:00:00.000000Z",
                 ended_at="2026-09-14T09:30:00.000000Z", peak_bytes=int(0.2 * GIB))
    conn.commit()
    conn.close()
    return root


def _clock():
    from engine.v2.foundation import SystemClock
    return SystemClock()


def test_overlap_list_includes_only_the_concurrent_attempt(seeded_root):
    report = render_memory_report(seeded_root, "job_render")
    assert report["attempt_id"] == "att_render"
    overlap_ids = {row["attempt_id"] for row in report["overlapping_attempts"]}
    assert overlap_ids == {"att_neighbor"}
    assert report["reservation"]["memory_bytes"] == 2 * GIB
    assert report["peak_rss_bytes"] == int(1.4 * GIB)


def test_recommendation_keep_under_threshold(seeded_root):
    report = render_memory_report(seeded_root, "job_render")
    assert report["peak_to_reservation_ratio"] == pytest.approx(1.4 / 2)
    assert report["peak_to_reservation_ratio"] <= KEEP_THRESHOLD
    assert report["recommendation"] == "keep"


def test_recommendation_flags_over_threshold(tmp_path):
    root = tmp_path / "ops_root"
    root.mkdir()
    conn = open_catalog(root / "catalog.sqlite", clock=_clock())
    conn.execute("INSERT INTO supervisor_epochs VALUES ('epoch1', 'boot', 1, 't0', NULL)")
    _seed_job(conn, "job_render", "legacy_render")
    _seed_attempt(conn, "att_render", "job_render", epoch_id="epoch1",
                 started_at="2026-09-14T10:00:00.000000Z",
                 ended_at="2026-09-14T10:30:00.000000Z", peak_bytes=int(1.9 * GIB))
    _seed_reservation(conn, "att_render", profile="projection", memory_bytes=2 * GIB)
    conn.commit()
    conn.close()

    report = render_memory_report(root, "job_render")
    assert report["recommendation"] == "flag_for_decision"
    assert report["overlapping_attempts"] == []


def test_missing_job_raises_clear_error(tmp_path):
    root = tmp_path / "ops_root"
    root.mkdir()
    conn = open_catalog(root / "catalog.sqlite", clock=_clock())
    conn.close()
    with pytest.raises(SystemExit, match="no succeeded attempt"):
        render_memory_report(root, "job_missing")


def test_cli_prints_json_only(seeded_root, capsys):
    exit_code = main(["--root", str(seeded_root), "--render-job", "job_render"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    doc = json.loads(out)
    assert doc["recommendation"] == "keep"
    assert "\n" not in out


def test_cli_subprocess_prints_only_one_json_line(seeded_root):
    result = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "rearchitecture_phase2_render_memory.py"),
         "--root", str(seeded_root), "--render-job", "job_render"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    json.loads(lines[0])
