"""Pre-registration guard for EXP-190 (Stage A). Fail-closed: a receipt with spec_hash_checked True, or OpsError.
Refuses stage_b_pending before the ledger is read. Never writes LEDGER.csv (no register function). Run with --check.
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import yaml

from engine.v2.ops.errors import OpsError
from experiments import lib
from experiments.size_pnl_prereg import _chronology, _read_bounded, _refuse

EXPERIMENT_ID = "EXP-190"
SPEC_PATH = Path(__file__).resolve().parent / "EXP-190_size_quantile_move" / "spec.yaml"


def load_spec(spec_path=SPEC_PATH):
    """Return ``(spec, hash)`` for a readable EXP-190 spec mapping, else refuse."""
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


def _may_claim(spec_file):
    """True unless an unparseable sibling spec is readable, at most 4 MiB and free of the id in its raw bytes."""
    try:
        with open(spec_file, "rb") as handle:
            raw = handle.read((4 << 20) + 1)
    except OSError:
        return True
    return len(raw) > 4 << 20 or b"EXP-190" in raw


def check_id_free(experiments_dir=None):
    """Refuse unless exactly one EXP-190_* directory exists and no other spec claims the id; return it."""
    base = Path(experiments_dir) if experiments_dir is not None else Path(__file__).resolve().parent
    dirs = [p for p in base.glob(f"{EXPERIMENT_ID}_*") if p.is_dir()]
    if len(dirs) != 1:
        raise _refuse("experiment_id_conflict", f"{len(dirs)} directories named {EXPERIMENT_ID}_* found, expected 1")
    for spec_file in base.glob("*/spec.yaml"):
        if spec_file.parent == dirs[0]:
            continue
        try:
            other = yaml.safe_load(_read_bounded(spec_file, 1 << 20))
        except (OSError, ValueError, yaml.YAMLError):
            if _may_claim(spec_file):
                raise _refuse("experiment_id_conflict",
                              f"{spec_file} is unreadable and may claim {EXPERIMENT_ID}") from None
            continue
        if isinstance(other, dict) and other.get("id") == EXPERIMENT_ID:
            raise _refuse("experiment_id_conflict", f"{spec_file} also carries id {EXPERIMENT_ID}")
    return dirs[0]


def _planned_hashes(ledger_path):
    """Spec hashes of every PLANNED EXP-190 row in file order; a missing ledger gives []."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    try:
        text = _read_bounded(path, 16 << 20)
    except (OSError, ValueError):
        raise _refuse("ledger_malformed", "ledger is unreadable, not a regular file or larger than 16 MiB") from None
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if any(column not in (reader.fieldnames or []) for column in lib.LEDGER_COLUMNS):
        raise _refuse("ledger_malformed", "ledger header lacks a required column")
    try:
        return [row.get("spec_hash") for row in reader
                if row.get("id") == EXPERIMENT_ID and row.get("stage") == "planned"]
    except csv.Error:
        raise _refuse("ledger_malformed", "ledger is unreadable or malformed") from None


def require_preregistration(spec_path=SPEC_PATH, ledger_path=None, run_log_path=None, experiments_dir=None) -> dict:
    """Return a receipt that the spec is registered, or raise (order: spec, id, stage_b, then the ledger).

    The ledger needs one PLANNED row per spec and per amendment, the last equal to the spec hash.
    """
    spec_path = Path(spec_path)
    ledger = lib.LEDGER_PATH if ledger_path is None else ledger_path
    run_log = spec_path.parent / "results" / "run_log.jsonl" if run_log_path is None else Path(run_log_path)
    spec, digest = load_spec(spec_path)
    check_id_free(experiments_dir)
    stage_b = spec.get("stage_b")
    if not (isinstance(stage_b, dict) and stage_b.get("status") == "settled"):
        raise _refuse("stage_b_pending", "stage_b is not settled: part 2 has not merged")
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
    return {"spec_hash": digest, "spec_hash_checked": True, "preregistered_at": str(prereg),
            "first_run_ts": None if first_ts is None else str(first_ts)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify EXP-190 pre-registration (read-only).")
    parser.add_argument("--check", action="store_true", required=True, help="verify and print the receipt as JSON")
    parser.parse_args(argv)
    try:
        print(json.dumps(require_preregistration()))
    except OpsError as exc:
        print(f"refused: {exc.code} {exc.problem.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
