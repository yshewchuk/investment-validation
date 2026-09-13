"""Current operational health is independent of an immutable board release."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from engine.v2.foundation import format_timestamp, fsync_directory
from engine.v2.ops.catalog import dumps, transaction


def record_check(conn, occurrence, kind, ok, receipt):
    with transaction(conn):
        conn.execute("INSERT INTO health_observations VALUES (?,?,?,?) "
                     "ON CONFLICT(occurrence,kind) DO UPDATE SET ok=excluded.ok,receipt_json=excluded.receipt_json",
                     (occurrence, kind, None if ok is None else int(ok), dumps(receipt)))


def budget_streak(conn):
    rows = conn.execute("SELECT occurrence,ok,receipt_json FROM health_observations "
                        "WHERE kind='engineering' ORDER BY occurrence DESC").fetchall()
    failures, unknown, first = 0, [], None
    for row in rows:
        if row["ok"] == 1:
            break
        if row["ok"] is None:
            unknown.append(row["occurrence"])
        else:
            failures += 1
            first = row["occurrence"]
    return {"ok": bool(rows) and rows[0]["ok"] == 1, "first_failed_on": first,
            "consecutive_nights": failures, "unknown_occurrences": unknown,
            "latest": json.loads(rows[0]["receipt_json"]) if rows else None,
            "override": None}


def health(conn, *, clock, executor_mode="watchdog"):
    jobs = [dict(row) for row in conn.execute(
        "SELECT job_id,kind,state,created_at,updated_at,queue_reason_json FROM jobs "
        "WHERE state NOT IN ('succeeded','cancelled') ORDER BY created_at")]
    current = conn.execute("SELECT release_id,occurrence,delivered_at FROM releases "
                           "WHERE delivered_at IS NOT NULL ORDER BY occurrence DESC LIMIT 1").fetchone()
    withheld = conn.execute("SELECT release_id,occurrence FROM releases WHERE eligible=0 "
                            "ORDER BY occurrence DESC LIMIT 1").fetchone()
    return {"schema_version": "operations_health.v1.0", "generated_at": format_timestamp(clock.now()),
            "executor_mode": executor_mode, "containment": "best_effort" if executor_mode == "watchdog" else "kernel",
            "jobs": jobs, "watermarks": [dict(row) for row in conn.execute("SELECT * FROM watermarks")],
            "current_release": dict(current) if current else None,
            "withheld_release": dict(withheld) if withheld else None,
            "code_budgets": budget_streak(conn), "activation": "shadow_only"}


def write_health(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    temporary.write_text(json.dumps(document))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
