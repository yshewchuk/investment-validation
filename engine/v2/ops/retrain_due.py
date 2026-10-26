"""Monthly retrain: the due-month decision, durable records and the pool-sidecar gate
(#805 slice 2; contract: engine/v2/models/ARCHITECTURE.md section 2.3 "Trigger (D9)").

``due_decision`` is a pure function of the session date, the base release's
``retrain_cutoff`` (None until slice 3 puts one in the catalog) and the months that already
have a durable record. Nothing here guesses: a malformed date raises.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

STEPS = ("train", "stage", "accept", "check", "promote")
#: Completed months examined when the base has a ``retrain_cutoff``.
MAX_LOOKBACK_MONTHS = 12
#: First month whose retrain is wired (models ARCHITECTURE section 2.3 Rollout: the first
#: affected nightly is 2026-11-02, for October). Earlier completed months are never due, so
#: the unavailable chain cannot block the October sessions after deployment.
FIRST_ELIGIBLE_MONTH = "2026-10"
_MARKER_DIR = "retrain_superseded"
_SCAN_LIMIT = 400


def _month_end(year: int, month: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def _completed_months(session: date, count: int) -> list[tuple[str, date]]:
    """Newest first: calendar months whose last day is strictly before ``session``."""
    year, month = session.year, session.month
    out = []
    while len(out) < count:
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        out.append((f"{year:04d}-{month:02d}", _month_end(year, month)))
    return out


def due_decision(session: str, base_retrain_cutoff: str | None, recorded: frozenset,
                 first_month: str = FIRST_ELIGIBLE_MONTH) -> dict:
    """The plan's ``retrain`` section: ``{"state": "not_due", "reason": ...}`` or
    ``{"state": "due", "month", "m_end", "superseded": [older due months]}``."""
    day = date.fromisoformat(session)
    if base_retrain_cutoff is None:
        candidates = _completed_months(day, 1)
    else:
        cutoff = date.fromisoformat(base_retrain_cutoff)
        candidates = [item for item in _completed_months(day, MAX_LOOKBACK_MONTHS)
                      if item[1] + timedelta(days=1) > cutoff]
    candidates = [item for item in candidates if item[0] >= first_month]
    if not candidates:
        return {"state": "not_due",
                "reason": "before_first_eligible_month" if _completed_months(day, 1)[0][0] < first_month
                else "base_retrain_cutoff_current"}
    open_months = [item for item in candidates if item[0] not in recorded]
    if not open_months:
        return {"state": "not_due", "reason": "month_already_recorded"}
    target = open_months[0]  # newest first
    return {"state": "due", "month": target[0], "m_end": target[1].isoformat(),
            "superseded": sorted(item[0] for item in open_months[1:])}


def read_recorded(conn, ops_root, session: str | None = None) -> frozenset:
    """Months with a durable success (a succeeded ``retrain-promote`` job) or a
    ``superseded`` marker file; with ``session`` only markers inside the decision's
    12-month window are read."""
    months = set()
    rows = conn.execute(
        "SELECT idempotency_key FROM jobs WHERE namespace = 'shadow' AND state = 'succeeded' "
        "AND idempotency_key >= ? AND idempotency_key < ? ORDER BY idempotency_key DESC LIMIT ?",
        ("retrain-promote:", "retrain-promote;", _SCAN_LIMIT)).fetchall()
    for row in rows:
        parts = row["idempotency_key"].split(":")
        if parts[0] == "retrain-promote" and len(parts) >= 4:
            months.add(parts[2])
    marker_dir = Path(ops_root) / _MARKER_DIR
    if session is not None:
        # Bounded: only the months a decision can examine are probed, never the whole archive.
        months.update(month for month, _ in _completed_months(date.fromisoformat(session),
                                                              MAX_LOOKBACK_MONTHS)
                      if (marker_dir / f"{month}.json").exists())
    elif marker_dir.is_dir():
        months.update(path.stem for path in marker_dir.glob("????-??.json"))
    return frozenset(months)


def write_superseded(ops_root, months, target: str, session: str) -> None:
    """Write each older due month's marker once (an existing marker is left as it is)."""
    marker_dir = Path(ops_root) / _MARKER_DIR
    for month in months:
        path = marker_dir / f"{month}.json"
        if path.exists():
            continue
        marker_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"month": month, "superseded_by": target,
                                   "session": session}, sort_keys=True))
        os.replace(tmp, path)


def decide(conn, ops_root, session: str, base_release_id: str, *,
           base_retrain_cutoff: str | None = None,
           first_month: str = FIRST_ELIGIBLE_MONTH) -> dict:
    """Evaluate the decision at plan-build time, write markers for older due months and
    log the outcome to stderr. The returned section is pinned into the plan."""
    decision = due_decision(session, base_retrain_cutoff, read_recorded(conn, ops_root, session), first_month)
    decision["base"] = base_release_id
    if decision["state"] == "due":
        write_superseded(ops_root, decision["superseded"], decision["month"], session)
    print(f"[retrain] session={session} {json.dumps(decision, sort_keys=True)}",
          file=sys.stderr, flush=True)
    return decision


def chain_blocks_pool_sidecar(conn, session: str, *, log: bool = True) -> bool:
    """True while this session's ``restage`` or retrain job is nonterminal or failed: the
    pool-nightly sidecar must submit nothing (models ARCHITECTURE section 2.3 Promotion).
    One bounded query per step, scoped to the session before the limit."""
    for step in ("restage", *STEPS):
        prefix = f"retrain-{step}:{session}"
        row = conn.execute(
            "SELECT idempotency_key, state FROM jobs WHERE namespace = 'shadow' "
            "AND idempotency_key >= ? AND idempotency_key < ? AND state != 'succeeded' LIMIT 1",
            (prefix, prefix + ";")).fetchone()
        if row is not None:
            if log:
                print(f"[retrain] pool sidecar held: {row['idempotency_key']} is {row['state']}",
                      file=sys.stderr, flush=True)
            return True
    return False
