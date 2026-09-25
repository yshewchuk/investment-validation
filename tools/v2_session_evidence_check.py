#!/usr/bin/env python3
"""P6-6 session evidence completeness, scoped to one qualified session.

Each Phase 6 capability row declares, in ``tools/phase6_capabilities.toml``,
the one real artifact that counts as its evidence:

* ``job:<kind>`` -- a succeeded attempt of that job kind whose ``started_at``
  falls inside the session window; when no attempt matches, a delivered
  ``outbox`` row of the same kind (the outbox table has no timestamp column,
  so that fallback is reported with ``window_checked: false``);
* ``route:<METHOD> <path>`` -- a 2xx row for that method/path in this
  session's own route-probe receipt
  (``<evidence-dir>/route_probe/<session>-route_probe.json``);
* ``cli:<tool>`` -- a resource-measurement record under
  ``<evidence-dir>/resource_measurement/`` whose ``command`` names that tool,
  whose ``exit_code`` is 0, that was not killed, and that started inside the
  window;
* ``exempt`` -- no evidence required (any ``missing``/``dormant-historical``
  disposition is exempt too);
* ``open`` -- a known gap: reported as OPEN and it fails the verdict while any
  remain, but it is never reported as uncovered evidence.

This replaces the old union-every-``capabilities_covered`` scan, which counted
FAIL receipts as coverage (Opus review #1). A row is covered only by its own
declared evidence, in this session's window. A broken JSON receipt is reported
under ``unreadable_evidence_files`` and never silently counted; a row whose
declaration has no ``evidence`` key fails closed, exactly like the old
"no disposition = not exempt" rule.

Exit 0 iff no row is uncovered and no row is open, else 1. ``--session``,
``--window-start``, ``--window-end`` and ``--catalog`` are all required: there
is deliberately no "look at everything ever" mode.

Usage::

    python3 tools/v2_session_evidence_check.py --session 2026-09-25 \\
        --window-start 2026-09-25T00:00:00Z --window-end 2026-09-25T23:59:59Z \\
        --catalog data/operations/catalog.sqlite [--json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.phase6_inventory import DECLARATIONS, build_document, load_declarations  # noqa: E402

__all__ = [
    "EXEMPT_DISPOSITIONS",
    "DEFAULT_EVIDENCE_DIR",
    "ROUTE_PROBE_DIR",
    "RESOURCE_MEASUREMENT_DIR",
    "parse_instant",
    "command_matches",
    "check",
    "main",
]

EXEMPT_DISPOSITIONS = frozenset({"missing", "dormant-historical"})
DEFAULT_EVIDENCE_DIR = Path("reports/phase6_evidence")
ROUTE_PROBE_DIR = "route_probe"
RESOURCE_MEASUREMENT_DIR = "resource_measurement"

#: The catalog's own timestamp wire form (engine/v2/foundation/clock.py), so a
#: normalized window bound compares lexicographically against stored values.
_WIRE = "%Y-%m-%dT%H:%M:%S.%fZ"
_JOB_ATTEMPT_SQL = (
    "SELECT 1 FROM attempts JOIN jobs ON attempts.job_id = jobs.job_id "
    "WHERE jobs.kind = ? AND attempts.state = 'succeeded' "
    "AND attempts.started_at BETWEEN ? AND ? LIMIT 1"
)
_OUTBOX_SQL = "SELECT 1 FROM outbox WHERE kind = ? AND state = 'delivered' LIMIT 1"


def parse_instant(text: str) -> datetime:
    """One operator-supplied instant; a naive value is refused, never guessed.

    Accepts the RFC 3339 forms an operator types (``...Z`` or an explicit
    offset) and normalizes to UTC. The catalog's own wire form is UTC, so a
    naive value would silently mean the host's local time -- refused instead.
    """
    value = text.strip()
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{text!r} has no timezone; pass an RFC 3339 UTC instant ending in Z")
    return parsed.astimezone(timezone.utc)


def _wire(text: str) -> str:
    return parse_instant(text).strftime(_WIRE)


def _is_2xx(status) -> bool:
    return isinstance(status, int) and not isinstance(status, bool) and 200 <= status < 300


def _is_clean_exit(record: dict) -> bool:
    """Exit 0 AND not killed: a watchdog kill must never pass on ``exit_code``
    alone, whatever combination the recorder might some day emit."""
    exit_code = record.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
        return False
    return record.get("killed") is False


def command_matches(command, tool: str) -> bool:
    """True when ``tool`` names the recorded command's own argv.

    ``cli:`` evidence has two shapes and neither may be guessed at: a tool path
    (``tools/v2_fill_quality.py`` -- one argv element) and an ops subcommand
    (``ops ledger calibrate`` -- matched token by token, in order, against
    ``python3 -m engine.v2.ops ledger calibrate``).
    """
    if not isinstance(command, list):
        return False
    argv = [part for part in command if isinstance(part, str)]
    parts = tool.split()
    if not parts or len(parts) > len(argv):
        return False
    return any(all(parts[offset] in argv[start + offset] for offset in range(len(parts)))
               for start in range(len(argv) - len(parts) + 1))


def _in_window(started_at, start: datetime, end: datetime) -> bool:
    if not isinstance(started_at, str):
        return False
    try:
        moment = parse_instant(started_at)
    except ValueError:
        return False
    return start <= moment <= end


def _detail(row_id, evidence, status, source, window_checked, detail) -> dict:
    return {
        "id": row_id,
        "evidence": evidence,
        "status": status,
        "source": source,
        "window_checked": window_checked,
        "detail": detail,
    }


def _job_evidence(conn, kind: str, start_wire: str, end_wire: str):
    """``(covered, source, window_checked, detail)`` for one ``job:<kind>``.

    Attempts first (that kind, succeeded, inside the window), then the outbox
    (delivered effect of the same kind). The outbox table has no timestamp
    column at all (``engine/v2/ops/schema_runtime.py``), so an outbox hit is
    recorded with ``window_checked: false`` -- the limitation is named on the
    row, never hidden.
    """
    try:
        hit = conn.execute(_JOB_ATTEMPT_SQL, (kind, start_wire, end_wire)).fetchone()
        if hit is not None:
            return True, "job", True, f"succeeded {kind!r} attempt inside the window"
        hit = conn.execute(_OUTBOX_SQL, (kind,)).fetchone()
    except sqlite3.Error as exc:
        return False, "job", True, f"catalog query failed: {exc}"
    if hit is not None:
        return True, "outbox", False, (
            f"delivered outbox effect {kind!r}; the outbox table carries no timestamp, "
            "so the session window was NOT checked")
    return False, "job", True, (
        f"no succeeded {kind!r} attempt in the window and no delivered outbox effect")


def _load_route_receipt(path: Path, session: str):
    """``(receipt, detail)``; a receipt for another session is refused here so
    one session's probe can never cover another's rows."""
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError:
        return None, f"no route-probe receipt at {path}"
    except (OSError, ValueError) as exc:
        return None, f"unreadable route-probe receipt {path}: {exc}"
    if not isinstance(document, dict):
        return None, f"route-probe receipt {path} is not a JSON object"
    if document.get("session") != session:
        return None, (f"route-probe receipt {path} names session "
                      f"{document.get('session')!r}, not {session!r}")
    return document, None


def _route_covered(receipt: dict, method: str, path: str) -> bool:
    for row in receipt.get("routes", []):
        if not isinstance(row, dict) or row.get("method") != method:
            continue
        if path not in (row.get("path"), row.get("declared_path")):
            continue
        if _is_2xx(row.get("status")):
            return True
    return False


def _cli_evidence(records, tool: str, start: datetime, end: datetime):
    """``(covered, detail)`` across every resource-measurement record."""
    for record in records:
        if not isinstance(record, dict) or not command_matches(record.get("command"), tool):
            continue
        if not _is_clean_exit(record):
            continue
        if not _in_window(record.get("started_at"), start, end):
            continue
        return True, (f"resource measurement of {tool!r} exited 0, was not killed, "
                      "and started inside the window")
    return False, (f"no exit-0, not-killed resource measurement of {tool!r} "
                   "inside the window")


def _load_resource_records(directory: Path):
    """``(records, unreadable)`` from ``resource_measurement/*.json``.

    An unparsable file is named rather than fatal, and a file that parses to
    something other than an object carries no record.
    """
    records: list[dict] = []
    unreadable: list[str] = []
    if not directory.is_dir():
        return records, unreadable
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            unreadable.append(str(path))
            continue
        if isinstance(document, dict):
            records.append(document)
        else:
            unreadable.append(str(path))
    return records, unreadable


def check(*, session: str, window_start: str, window_end: str, catalog: Path,
          declarations=None, evidence_dir=DEFAULT_EVIDENCE_DIR, root=ROOT) -> dict:
    """Resolve every declaration row's own evidence inside one session window.

    Returns the summary plus a per-row ``rows`` detail list (each carrying
    ``source``, ``window_checked`` and a human ``detail``), so a caller can see
    exactly why a row is covered or not -- never just a boolean.
    """
    root = Path(root)
    evidence_path = Path(evidence_dir)
    if not evidence_path.is_absolute():
        evidence_path = root / evidence_path
    start, end = parse_instant(window_start), parse_instant(window_end)
    if end < start:
        raise ValueError(f"--window-end {window_end!r} is before --window-start {window_start!r}")
    catalog_path = Path(catalog)
    if not catalog_path.is_absolute():
        catalog_path = root / catalog_path

    declarations_path = root / (declarations or DECLARATIONS)
    declared = load_declarations(declarations_path)
    evidence_by_id = {row.get("id"): row.get("evidence") for row in declared.get("row", [])}
    rows = build_document(root=root, declarations=declarations)["rows"]
    evidence = {row["id"]: evidence_by_id.get(row["id"]) for row in rows}

    job_needed = any(isinstance(ev, str) and ev.startswith("job:") for ev in evidence.values())
    route_needed = any(isinstance(ev, str) and ev.startswith("route:") for ev in evidence.values())
    cli_needed = any(isinstance(ev, str) and ev.startswith("cli:") for ev in evidence.values())

    conn = None
    catalog_error = None
    if job_needed:
        try:
            conn = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            catalog_error = str(exc)

    route_receipt = None
    route_detail = None
    if route_needed:
        route_path = evidence_path / ROUTE_PROBE_DIR / f"{session}-route_probe.json"
        route_receipt, route_detail = _load_route_receipt(route_path, session)

    records: list[dict] = []
    unreadable: list[str] = []
    if cli_needed:
        records, unreadable = _load_resource_records(evidence_path / RESOURCE_MEASUREMENT_DIR)

    details: list[dict] = []
    uncovered: list[str] = []
    rows_open: list[str] = []
    exempt = covered = 0
    start_wire, end_wire = _wire(window_start), _wire(window_end)
    job_cache: dict[str, tuple] = {}
    try:
        for row in rows:
            row_id = row["id"]
            declared_evidence = evidence[row_id]
            disposition = row.get("disposition")
            if declared_evidence == "open":
                rows_open.append(row_id)
                details.append(_detail(row_id, declared_evidence, "open", None, None,
                                       "declared open (known gap, not evidence)"))
                continue
            if declared_evidence == "exempt" or disposition in EXEMPT_DISPOSITIONS:
                exempt += 1
                why = ("declared exempt" if declared_evidence == "exempt"
                       else f"disposition {disposition!r}")
                details.append(_detail(row_id, declared_evidence, "exempt", None, None, why))
                continue
            if not isinstance(declared_evidence, str) or ":" not in declared_evidence:
                uncovered.append(row_id)
                details.append(_detail(row_id, declared_evidence, "uncovered", None, None,
                                       "no usable evidence field (fails closed)"))
                continue
            source_kind, _, argument = declared_evidence.partition(":")
            if source_kind == "job":
                if conn is None:
                    ok, source, window_checked = False, "job", None
                    detail = f"catalog unavailable: {catalog_error}"
                else:
                    if argument not in job_cache:
                        job_cache[argument] = _job_evidence(conn, argument, start_wire, end_wire)
                    ok, source, window_checked, detail = job_cache[argument]
            elif source_kind == "route":
                method, _, path = argument.partition(" ")
                if route_receipt is None:
                    ok, source, window_checked, detail = False, "route", False, route_detail
                else:
                    ok = _route_covered(route_receipt, method, path)
                    source, window_checked = "route", False
                    detail = (f"{method} {path} answered 2xx in this session's route probe"
                              if ok else
                              f"{method} {path} has no 2xx row in this session's route probe")
            elif source_kind == "cli":
                ok, detail = _cli_evidence(records, argument, start, end)
                source, window_checked = "cli", True
            else:
                ok, source, window_checked = False, None, None
                detail = f"unknown evidence kind {source_kind!r} (fails closed)"
            if ok:
                covered += 1
                details.append(_detail(row_id, declared_evidence, "covered", source,
                                       window_checked, detail))
            else:
                uncovered.append(row_id)
                details.append(_detail(row_id, declared_evidence, "uncovered", source,
                                       window_checked, detail))
    finally:
        if conn is not None:
            conn.close()

    return {
        "session": session,
        "window": [window_start, window_end],
        "rows_total": len(rows),
        "rows_exempt": exempt,
        "rows_open": rows_open,
        "rows_covered": covered,
        "rows_uncovered": uncovered,
        "rows": details,
        "unreadable_evidence_files": unreadable,
        "verdict": "PASS" if not uncovered and not rows_open else "FAIL",
    }


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--session", required=True,
                        help="the qualified session id; also names its route-probe receipt")
    parser.add_argument("--window-start", required=True,
                        help="RFC 3339 UTC instant; no default 'everything ever' mode")
    parser.add_argument("--window-end", required=True, help="RFC 3339 UTC instant, inclusive")
    parser.add_argument("--catalog", required=True, type=Path,
                        help="the real ops catalog (opened read-only) for job:<kind> evidence")
    parser.add_argument("--json", action="store_true", help="print only the JSON document")
    parser.add_argument("--declarations", type=Path, default=None,
                        help="override tools/phase6_capabilities.toml")
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = check(session=args.session, window_start=args.window_start,
                       window_end=args.window_end, catalog=args.catalog,
                       declarations=args.declarations, evidence_dir=args.evidence_dir)
    except ValueError as exc:
        print(f"session-evidence: refused: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"phase6 session evidence: {result['session']} "
              f"[{result['window'][0]} .. {result['window'][1]}]: "
              f"{result['rows_total']} rows, {result['rows_exempt']} exempt, "
              f"{len(result['rows_open'])} open, {result['rows_covered']} covered, "
              f"{len(result['rows_uncovered'])} uncovered -> {result['verdict']}")
        for row_id in result["rows_uncovered"]:
            print(f"  UNCOVERED {row_id}")
        for row_id in result["rows_open"]:
            print(f"  OPEN {row_id}")
        for path in result["unreadable_evidence_files"]:
            print(f"  UNREADABLE {path}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
