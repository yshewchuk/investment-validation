#!/usr/bin/env python3
"""Append one exit month of EXP-137 candidates; explicitly invoked by the monthly supervisor, with no automatic job caller in this slice."""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import resource
import stat
import tempfile
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANDIDATE_ARCHIVE_MAX_BYTES = 512 * 1024 * 1024
from engine.v2.models.training.fold_store import TrainingRefused
from engine.v2.models.training.receipts import ReceiptIssue

CANDIDATES = ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str/results/candidates.parquet"
KEY = ("event_id", "arm", "variant", "fill_alpha")


def refuse(code: str, detail: str) -> None:
    raise TrainingRefused((ReceiptIssue(str(CANDIDATES), code, detail),))


def require_month(value: str) -> str:
    try:
        parsed = date.fromisoformat(value + "-01")
    except (TypeError, ValueError):
        refuse("invalid_exit_month", "month must be canonical YYYY-MM")
    if not value.isascii() or parsed.strftime("%Y-%m") != value:
        refuse("invalid_exit_month", "month must be canonical YYYY-MM")
    return value


def _read_candidate_table(path: Path) -> pd.DataFrame:
    if not hasattr(os, "O_NOFOLLOW"):
        refuse("candidate_table_read_failed", "safe no-follow reads are unsupported")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        refuse("candidate_table_missing", "stored EXP-137 candidate table is required")
    except OSError as exc:
        refuse("candidate_table_read_failed", f"{type(exc).__name__}: {exc}")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            refuse("candidate_table_invalid", "stored candidate path is not a regular file")
        if info.st_size > CANDIDATE_ARCHIVE_MAX_BYTES:
            refuse("candidate_table_too_large", "stored candidate table exceeds the bounded read limit")
        source = os.fdopen(fd, "rb")
        fd = None
        with source:
            payload = source.read(CANDIDATE_ARCHIVE_MAX_BYTES + 1)
        if len(payload) > CANDIDATE_ARCHIVE_MAX_BYTES:
            refuse("candidate_table_too_large", "stored candidate table exceeds the bounded read limit")
        return pd.read_parquet(io.BytesIO(payload))
    except TrainingRefused:
        raise
    except Exception as exc:
        refuse("candidate_table_read_failed", f"{type(exc).__name__}: {exc}")
    finally:
        if fd is not None:
            os.close(fd)


def append_candidates(path: Path, fresh: pd.DataFrame) -> int:
    old = _read_candidate_table(path)
    if any(column not in old for column in (*KEY, "event_date", "entry_date")):
        refuse("candidate_table_invalid", "stored table lacks candidate identity or dates")
    if fresh.empty:
        return 0
    if any(column not in fresh for column in (*KEY, "event_date", "entry_date")):
        refuse("candidate_build_invalid", "fresh candidates lack identity or dates")
    event_dates = pd.to_datetime(fresh["event_date"], errors="coerce")
    entry_dates = pd.to_datetime(fresh["entry_date"], errors="coerce")
    sessions = fresh.get("session", pd.Series("", index=fresh.index)).astype(str).str.upper()
    future_non_bmo = (event_dates > entry_dates) & ~sessions.eq("BMO")
    post_event_bmo = sessions.eq("BMO") & entry_dates.ge(event_dates)
    if event_dates.isna().any() or entry_dates.isna().any() or future_non_bmo.any() or post_event_bmo.any():
        refuse("candidate_causality_failed", "entry must precede BMO events; other event dates must be <= entry")
    fresh = fresh.drop_duplicates(list(KEY), keep="first")
    old_keys = pd.MultiIndex.from_frame(old[list(KEY)])
    new_keys = pd.MultiIndex.from_frame(fresh[list(KEY)])
    unseen = fresh.loc[~new_keys.isin(old_keys)]
    if len(unseen):
        temp_path = None
        try:
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            temp_path = Path(temp_name)
            os.close(fd)
            pd.concat([old, unseen], ignore_index=True).to_parquet(temp_path, index=False)
            os.chmod(temp_path, path.stat().st_mode)
            os.replace(temp_path, path)
        except Exception as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            refuse("candidate_table_write_failed", f"{type(exc).__name__}: {exc}")
    return len(unseen)


def refresh(month: str, path: Path = CANDIDATES) -> int:
    month = require_month(month)
    if not path.is_file():
        refuse("candidate_table_missing", "stored EXP-137 candidate table is required")
    started = time.monotonic()
    print(f"[chooser-refresh] start exit_month={month} "
          f"RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB",
          file=sys.stderr, flush=True)
    try:
        builder_path = ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str/build.py"
        spec = importlib.util.spec_from_file_location("chooser_exp137_builder", builder_path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        built = builder.build_all(force=True, limit_exit_month=month)
        added = append_candidates(path, built["trades"])
    except TrainingRefused:
        raise
    except Exception as exc:
        refuse("candidate_build_failed", f"{type(exc).__name__}: {exc}")
    print(f"[chooser-refresh] done added={added} elapsed_s={time.monotonic()-started:.1f} "
          f"skips={built.get('meta', {}).get('skips', {})} "
          f"RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB",
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
