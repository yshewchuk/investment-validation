#!/usr/bin/env python3
"""P2-C02: render memory report.

"Measure the render profile now that it constructs a Scorer ... Record
render peak memory and other running jobs; retain or change its 2 GiB
reservation from evidence, not from the small synthetic tests. Never change
limits beneath an active attempt." (P2-C02, quoted in the D19 brief.)

This script only READS the ops catalog (``<root>/catalog.sqlite``, the same
layout ``engine.v2.ops.cli`` opens at ``--root``). It never writes to
``resource_reservations`` and never edits ``engine/v2/ops/profiles.py`` --
the reservation decision is the evidence-reading OPERATOR's, made from a
real nightly's numbers, never from this script and never from a synthetic
test (guide/brief: "retain or change ... from evidence, not from the small
synthetic tests").

Usage::

    python3 checks/rearchitecture_phase2_render_memory.py \\
        --root data/operations --render-job job_abc123
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.foundation import SystemClock
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named

#: peak <= this share of the reservation -> "keep"; above it -> "flag_for_decision".
#: Stated explicitly in every report this script prints (deliverable 3).
KEEP_THRESHOLD = 0.80
RECOMMENDATION_RULE = (
    f"peak_rss_bytes <= {KEEP_THRESHOLD:.0%} of reserved memory_bytes -> keep; "
    "otherwise -> flag_for_decision"
)


def _latest_succeeded_attempt(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT attempt_id, job_id, attempt_number, started_at, ended_at, "
        "memory_peak_bytes FROM attempts WHERE job_id = ? AND state = 'succeeded' "
        "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no succeeded attempt found for render job {job_id!r}")
    return row


def _reservation(conn: sqlite3.Connection, attempt_id: str) -> dict | None:
    row = conn.execute(
        "SELECT profile, memory_bytes, policy_version FROM resource_reservations "
        "WHERE attempt_id = ?", (attempt_id,)).fetchone()
    return dict(row) if row is not None else None


def _overlapping_attempts(conn: sqlite3.Connection, attempt_id: str, started_at: str,
                          ended_at: str | None) -> list[dict]:
    """Every OTHER attempt whose run window overlapped this one's.

    A null ``ended_at`` (still running, or abandoned without a clean close)
    is treated as "open until now" on both sides of the comparison, so an
    unterminated neighbor still counts as contention rather than being
    silently excluded.
    """
    window_end = ended_at or "9999-12-31T23:59:59.999999Z"
    rows = conn.execute(
        "SELECT a.attempt_id, a.job_id, j.kind, a.started_at, a.ended_at, "
        "a.memory_peak_bytes, r.profile, r.memory_bytes AS reserved_bytes "
        "FROM attempts a JOIN jobs j ON j.job_id = a.job_id "
        "LEFT JOIN resource_reservations r ON r.attempt_id = a.attempt_id "
        "WHERE a.attempt_id != ? AND a.started_at IS NOT NULL "
        "AND a.started_at < ? AND (a.ended_at IS NULL OR a.ended_at > ?) "
        "ORDER BY a.started_at", (attempt_id, window_end, started_at)).fetchall()
    return [dict(row) for row in rows]


def _recommendation(peak_bytes: int | None, reserved_bytes: int | None) -> tuple[str, float | None]:
    if peak_bytes is None or reserved_bytes in (None, 0):
        return "undetermined", None
    ratio = peak_bytes / reserved_bytes
    return ("keep" if ratio <= KEEP_THRESHOLD else "flag_for_decision"), ratio


def render_memory_report(root: str | Path, render_job: str) -> dict:
    """Assemble the report; pure read, no writes anywhere."""
    conn = open_catalog(Path(root) / "catalog.sqlite", clock=SystemClock())
    try:
        attempt = _latest_succeeded_attempt(conn, render_job)
        reservation = _reservation(conn, attempt["attempt_id"])
        others = _overlapping_attempts(conn, attempt["attempt_id"], attempt["started_at"],
                                       attempt["ended_at"])
        reserved_bytes = reservation["memory_bytes"] if reservation else None
        policy_default = profile_named(DEFAULT_POLICY, "projection").memory_bytes
        recommendation, ratio = _recommendation(attempt["memory_peak_bytes"], reserved_bytes)
        return {
            "schema_version": "phase2_render_memory_report.v1.0",
            "render_job": render_job,
            "attempt_id": attempt["attempt_id"],
            "attempt_number": attempt["attempt_number"],
            "started_at": attempt["started_at"],
            "ended_at": attempt["ended_at"],
            "peak_rss_bytes": attempt["memory_peak_bytes"],
            "reservation": reservation,
            "current_policy_projection_memory_bytes": policy_default,
            "peak_to_reservation_ratio": ratio,
            "recommendation": recommendation,
            "recommendation_rule": RECOMMENDATION_RULE,
            "overlapping_attempts": others,
        }
    finally:
        conn.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="the ops root (holds catalog.sqlite)")
    parser.add_argument("--render-job", required=True, help="the legacy_render job id")
    args = parser.parse_args(argv)
    report = render_memory_report(args.root, args.render_job)
    print(json.dumps(report, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
