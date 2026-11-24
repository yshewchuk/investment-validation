#!/usr/bin/env python3
"""Pre-registration guard for EXP-189 (production short-vol width tuning, M0 only).

Fail-closed, modelled on ``experiments.size_pnl_prereg``. Every path either returns a receipt whose
``spec_hash_checked`` is True, or raises an ``OpsError``. It never returns a receipt that says the check
was skipped, and it cannot no-op: with no PLANNED row it refuses (``planned_row_missing``).

    python3 -m experiments.size_width_prereg --check    # verify, print receipt

This module never writes experiments/LEDGER.csv: it has no register function and no code path that
appends to the ledger. The PLANNED row is registered by the supervisor once the spec is on main.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

from engine.v2.ops.errors import OpsError
from experiments import lib
from experiments.size_pnl_prereg import _chronology, _read_bounded, _refuse

EXPERIMENT_ID = "EXP-189"
SPEC_PATH = Path(__file__).resolve().parent / "EXP-189_short_vol_width_tuning" / "spec.yaml"


def load_spec(spec_path=SPEC_PATH):
    """Return ``(spec, hash)`` for a readable EXP-189 spec mapping, else refuse."""
    try:
        spec = yaml.safe_load(_read_bounded(spec_path, 1 << 20))
        if not isinstance(spec, dict):
            raise ValueError("spec did not parse to a mapping")
    except (OSError, ValueError, yaml.YAMLError):
        raise _refuse("spec_unreadable", "spec is missing, unparseable or not a mapping") from None
    if spec.get("id") != EXPERIMENT_ID:
        raise _refuse("spec_id_mismatch", f"spec id is not {EXPERIMENT_ID}")
    if not isinstance(spec.get("amendments", []), list):
        raise _refuse("spec_unreadable", "spec amendments is not a list")
    return spec, lib.spec_hash(spec)


def check_id_free(experiments_dir=None):
    """Refuse unless exactly one ``EXP-189_*`` directory exists and no other spec claims id EXP-189.

    Returns the one ``EXP-189_*`` directory.
    """
    base = Path(experiments_dir) if experiments_dir is not None else Path(__file__).resolve().parent
    dirs = [p for p in base.glob(f"{EXPERIMENT_ID}_*") if p.is_dir()]
    if len(dirs) != 1:
        raise _refuse("experiment_id_conflict", f"{len(dirs)} directories named {EXPERIMENT_ID}_* found, expected 1")
    own = dirs[0]
    for spec_file in base.glob("*/spec.yaml"):
        if spec_file.parent == own:
            continue
        try:
            other = yaml.safe_load(_read_bounded(spec_file, 1 << 20))
        except (OSError, ValueError, yaml.YAMLError):
            continue  # a spec that does not parse cannot carry the id
        if isinstance(other, dict) and other.get("id") == EXPERIMENT_ID:
            raise _refuse("experiment_id_conflict", f"{spec_file} also carries id {EXPERIMENT_ID}")
    return own


def _planned_hashes(ledger_path):
    """Spec hashes of every PLANNED row for EXP-189, in file order. A missing ledger gives an empty list."""
    try:
        frame = lib.ledger_read(Path(ledger_path))
    except (lib.LedgerError, ValueError):
        raise _refuse("ledger_malformed", "ledger is unreadable or malformed") from None
    rows = frame[(frame["id"] == EXPERIMENT_ID) & (frame["stage"] == "planned")]
    return rows["spec_hash"].tolist()


def require_preregistration(spec_path=SPEC_PATH, ledger_path=None, run_log_path=None) -> dict:
    """Verify the spec is registered before any run; return a receipt or raise.

    The ledger must hold exactly one PLANNED row for EXP-189 (no amendments are declared), equal to the
    spec hash. ``lib.LEDGER_PATH`` is read at call time, so tests can monkeypatch it.
    """
    spec_path = Path(spec_path)
    ledger = lib.LEDGER_PATH if ledger_path is None else ledger_path
    run_log = (spec_path.parent / "results" / "run_log.jsonl" if run_log_path is None
               else Path(run_log_path))

    spec, digest = load_spec(spec_path)
    expected = len(spec.get("amendments", [])) + 1
    planned = _planned_hashes(ledger)
    if not planned:
        raise _refuse("planned_row_missing", f"no PLANNED row for {EXPERIMENT_ID} in the ledger")
    if len(planned) != expected:
        raise _refuse("planned_row_ambiguous", f"{len(planned)} PLANNED rows for {EXPERIMENT_ID}, "
                      f"expected {expected} (one per amendment plus the spec)")
    if planned[-1] != digest:
        raise _refuse("spec_hash_mismatch", "spec differs from its PLANNED row")

    prereg, first_ts = _chronology(spec, spec_path, run_log)

    return {
        "spec_hash": digest,
        "spec_hash_checked": True,
        "preregistered_at": str(prereg),
        "first_run_ts": None if first_ts is None else str(first_ts),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify EXP-189 pre-registration (read-only).")
    parser.add_argument("--check", action="store_true", required=True,
                        help="verify registration and print the receipt as JSON")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(require_preregistration()))
    except OpsError as exc:
        print(f"refused: {exc.code} {exc.problem.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
