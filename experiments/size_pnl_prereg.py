#!/usr/bin/env python3
"""Pre-registration guard for EXP-188 (SIZE decision-level P&L re-score).

``engine.evaluate`` records ``spec_hash_checked=False`` when no PLANNED row
exists and does not refuse the run, so a plain run would silently skip the
registration check.
This guard fails closed instead: every path either returns a receipt whose
``spec_hash_checked`` is True, or raises an ``OpsError``. It never returns a
receipt that says the check was skipped.

    python3 -m experiments.size_pnl_prereg --check      # verify, print receipt
    python3 -m experiments.size_pnl_prereg --register   # append PLANNED row once
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from engine.v2.ops.errors import OpsError, fail
from experiments import lib

EXPERIMENT_ID = "EXP-188"
SPEC_PATH = Path(__file__).resolve().parent / "EXP-188_size_decision_pnl_rescore" / "spec.yaml"


def _refuse(reason, message):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", f"{reason}: {message}",
                details={"reason": reason})


def _read_bounded(path, limit, *, first_line=False) -> str:
    """Read a regular file without blocking on FIFOs and without reading past ``limit``.

    With ``first_line`` the text is cut to the first non-blank line within the bytes
    read (leading blank or whitespace-only lines are skipped, matching
    ``engine.evaluate``), and the size check is skipped, so a grown log is never read
    in full. If no non-blank line occurs within the read, the result is "".
    Callers catch OSError/ValueError.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"not a regular file: {path}")
        if not first_line and st.st_size > limit:
            raise ValueError(f"file larger than {limit} bytes: {path}")
        chunks = []
        remaining = limit
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1 << 16))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if first_line:
        data = next((line for line in data.split(b"\n") if line.strip()), b"")
    return data.decode("utf-8")


def _load(spec_path):
    """Return ``(spec, hash)`` for a readable EXP-188 spec mapping, else refuse."""
    try:
        spec = yaml.safe_load(_read_bounded(spec_path, 1 << 20))
        if not isinstance(spec, dict):
            raise ValueError("spec did not parse to a mapping")
    except (OSError, ValueError, yaml.YAMLError):
        raise _refuse("spec_unreadable", "spec is missing, unparseable or not a mapping") from None
    if spec.get("id") != EXPERIMENT_ID:
        raise _refuse("spec_id_mismatch", f"spec id is not {EXPERIMENT_ID}")
    return spec, lib.spec_hash(spec)


def _planned_hashes(ledger_path):
    """Spec hashes of every PLANNED row for EXP-188, in file order."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    try:
        text = _read_bounded(path, 16 << 20)
    except (OSError, ValueError):
        raise _refuse("ledger_malformed", "ledger is unreadable or not a regular file") from None
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = reader.fieldnames or []
    if any(column not in header for column in lib.LEDGER_COLUMNS):
        raise _refuse("ledger_malformed", "ledger header lacks a required column")
    return [row.get("spec_hash") for row in reader
            if row.get("id") == EXPERIMENT_ID and row.get("stage") == "planned"]


def _parse_ts(value):
    """A timezone-aware pandas Timestamp (naive means UTC), or None if unparseable."""
    if not isinstance(value, (str, datetime)):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp


def _first_run_ts(run_log_path):
    """The ``ts`` of the first run-log line, or None if that line is blank or the log is empty."""
    try:
        first = _read_bounded(run_log_path, 1 << 16, first_line=True)
    except (OSError, ValueError):
        raise _refuse("run_log_malformed", "run log is unreadable") from None
    if not first.strip():
        return None
    try:
        record = json.loads(first)
    except ValueError:
        record = None
    stamp = _parse_ts(record.get("ts")) if isinstance(record, dict) else None
    if stamp is None:
        raise _refuse("run_log_malformed", "first run-log line carries no parseable ts")
    return stamp


def _checked_stamp(spec):
    """The parsed ``preregistered_at`` of the spec, or refuse if it is missing or unparseable."""
    prereg = _parse_ts(spec.get("preregistered_at"))
    if prereg is None:
        raise _refuse("preregistered_at_missing", "spec has no parseable preregistered_at")
    return prereg


def _chronology(spec, spec_path, run_log_path):
    """Return ``(prereg, first_ts)``: the checked stamp and the first run-log ts (or None).

    Refuses if the stamp is missing or unparseable, if the run log is malformed, or if
    the stamp is later than the first run. ``run_log_path`` is only read if it exists.
    """
    prereg = _checked_stamp(spec)
    first_ts = None
    run_log = Path(run_log_path)
    if run_log.exists():
        first_ts = _first_run_ts(run_log)
        if first_ts is not None and prereg > first_ts:
            raise _refuse("preregistered_after_run", "preregistered_at is after the first run")
    return prereg, first_ts


def require_preregistration(spec_path=SPEC_PATH, ledger_path=None, run_log_path=None) -> dict:
    """Verify the spec is registered before any run; return a receipt or raise.

    ``lib.LEDGER_PATH`` is read at call time, so tests can monkeypatch it.
    """
    spec_path = Path(spec_path)
    ledger = lib.LEDGER_PATH if ledger_path is None else ledger_path
    run_log = (spec_path.parent / "results" / "run_log.jsonl" if run_log_path is None
               else Path(run_log_path))

    spec, digest = _load(spec_path)
    planned = _planned_hashes(ledger)
    if not planned:
        raise _refuse("planned_row_missing", f"no PLANNED row for {EXPERIMENT_ID} in the ledger")
    if len(planned) > 1:
        raise _refuse("planned_row_ambiguous", f"{len(planned)} PLANNED rows for {EXPERIMENT_ID}")
    if planned[0] != digest:
        raise _refuse("spec_hash_mismatch", "spec differs from its PLANNED row")

    prereg, first_ts = _chronology(spec, spec_path, run_log)

    return {
        "spec_hash": digest,
        "spec_hash_checked": True,
        "preregistered_at": str(prereg),
        "first_run_ts": None if first_ts is None else str(first_ts),
    }


def register_planned(spec_path=SPEC_PATH, ledger_path=None, today=None,
                     run_log_path=None) -> bool:
    """Append the PLANNED row once. True if appended, False if already registered.

    Runs the same stamp and run-log chronology check as ``require_preregistration``
    before any ledger write.
    """
    ledger = lib.LEDGER_PATH if ledger_path is None else ledger_path
    spec_path = Path(spec_path)
    run_log = (spec_path.parent / "results" / "run_log.jsonl" if run_log_path is None
               else Path(run_log_path))
    spec, digest = _load(spec_path)
    _chronology(spec, spec_path, run_log)
    planned = _planned_hashes(ledger)
    if planned == [digest]:
        return False
    if planned:
        raise _refuse("planned_row_conflict", f"ledger holds a different PLANNED hash for "
                      f"{EXPERIMENT_ID}")
    lib.ledger_append([{
        "id": EXPERIMENT_ID,
        "spec_hash": digest,
        "date": today or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "stage": "planned",
        "oos_mean_mid": "",
        "sharpe_trade": "",
        "promoted": "False",
    }], path=ledger)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="verify registration and print the receipt as JSON")
    mode.add_argument("--register", action="store_true",
                      help="append the PLANNED row if it is not already present")
    args = parser.parse_args(argv)
    try:
        if args.check:
            print(json.dumps(require_preregistration()))
        else:
            print("registered" if register_planned() else "already registered")
    except OpsError as exc:
        print(f"refused: {exc.code} {exc.problem.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
