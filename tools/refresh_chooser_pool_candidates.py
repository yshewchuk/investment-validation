#!/usr/bin/env python3
"""Append one exit month of fresh EXP-137 candidates to the stored table."""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.v2.models.training.fold_store import TrainingRefused
from engine.v2.models.training.receipts import ReceiptIssue

CANDIDATES = ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str/results/candidates.parquet"
KEY = ("event_id", "arm", "variant", "fill_alpha")


def refuse(code: str, detail: str) -> None:
    raise TrainingRefused((ReceiptIssue(str(CANDIDATES), code, detail),))


def require_month(value: str) -> str:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        refuse("invalid_exit_month", "month must be canonical YYYY-MM")
    return value


def append_candidates(path: Path, fresh: pd.DataFrame) -> int:
    if not path.is_file():
        refuse("candidate_table_missing", "stored EXP-137 candidate table is required")
    try:
        old = pd.read_parquet(path)
    except Exception as exc:
        refuse("candidate_table_read_failed", f"{type(exc).__name__}: {exc}")
    if any(column not in old for column in (*KEY, "event_date", "entry_date")):
        refuse("candidate_table_invalid", "stored table lacks candidate identity or dates")
    if any(column not in fresh for column in (*KEY, "event_date", "entry_date")):
        refuse("candidate_build_invalid", "fresh candidates lack identity or dates")
    event_dates = pd.to_datetime(fresh["event_date"], errors="coerce")
    entry_dates = pd.to_datetime(fresh["entry_date"], errors="coerce")
    if event_dates.isna().any() or entry_dates.isna().any() or (event_dates > entry_dates).any():
        refuse("candidate_causality_failed", "forecast/residual event date must be <= entry date")
    fresh = fresh.drop_duplicates(list(KEY), keep="first")
    old_keys = pd.MultiIndex.from_frame(old[list(KEY)])
    new_keys = pd.MultiIndex.from_frame(fresh[list(KEY)])
    unseen = fresh.loc[~new_keys.isin(old_keys)]
    if len(unseen):
        try:
            pd.concat([old, unseen], ignore_index=True).to_parquet(path, index=False)
        except Exception as exc:
            refuse("candidate_table_write_failed", f"{type(exc).__name__}: {exc}")
    return len(unseen)


def refresh(month: str, path: Path = CANDIDATES) -> int:
    month = require_month(month)
    if not path.is_file():
        refuse("candidate_table_missing", "stored EXP-137 candidate table is required")
    started = time.monotonic()
    print(f"[chooser-refresh] start exit_month={month}", file=sys.stderr, flush=True)
    try:
        builder_path = ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str/build.py"
        spec = importlib.util.spec_from_file_location("chooser_exp137_builder", builder_path)
        if spec is None or spec.loader is None:
            refuse("candidate_builder_missing", str(builder_path))
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        built = builder.build_all(force=True, limit_exit_month=month)
        added = append_candidates(path, built["trades"])
    except TrainingRefused:
        raise
    except Exception as exc:
        refuse("candidate_build_failed", f"{type(exc).__name__}: {exc}")
    print(f"[chooser-refresh] done added={added} elapsed_s={time.monotonic()-started:.1f}",
          file=sys.stderr, flush=True)
    return added


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, help="candidate exit month (YYYY-MM)")
    args = parser.parse_args(argv)
    refresh(args.month)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
