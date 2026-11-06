#!/usr/bin/env python3
"""Append one effective-exit month of EXP-137 candidate rows for an explicit supervisor run."""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import resource
import stat
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.paths import ROOT
from engine.v2.models.training.fold_store import TrainingRefused
from engine.v2.models.training.receipts import ReceiptIssue
from engine.fills import MIN_MEANINGFUL_COST

CANDIDATES = ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str/results/candidates.parquet"
KEY = ("event_id", "arm", "variant", "fill_alpha")
SETTLEMENT_FIELDS = ("exit_date", "spot_exit", "dte_exit", "exit_arb_ok", "exit_value", "exit_value_expiry", "entry_cost", "expiry", "spot_expiry", "pnl", "ret")
MAX_BYTES = 512 * 1024 * 1024


def refuse(code: str, detail: str, path: Path = CANDIDATES) -> None:
    raise TrainingRefused((ReceiptIssue(str(path), code, detail),))


def require_month(value: str, path: Path = CANDIDATES) -> str:
    try:
        parsed = date.fromisoformat(value + "-01")
    except (TypeError, ValueError):
        refuse("invalid_exit_month", "month must be canonical YYYY-MM", path)
    if not value.isascii() or parsed.strftime("%Y-%m") != value:
        refuse("invalid_exit_month", "month must be canonical YYYY-MM", path)
    return value


def read_table(path: Path) -> tuple[pd.DataFrame, bytes]:
    if not hasattr(os, "O_NOFOLLOW"):
        refuse("candidate_table_read_failed", "safe no-follow reads are unsupported", path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        refuse("candidate_table_missing", "stored EXP-137 candidate table is required", path)
    except OSError as exc:
        refuse("candidate_table_read_failed", f"{type(exc).__name__}: {exc}", path)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            refuse("candidate_table_invalid", "stored candidate path is not a regular file", path)
        if info.st_size > MAX_BYTES:
            refuse("candidate_table_too_large", "stored candidate table exceeds read limit", path)
        with os.fdopen(fd, "rb") as source:
            fd = None
            payload = source.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            refuse("candidate_table_too_large", "stored candidate table exceeds read limit", path)
        return pd.read_parquet(io.BytesIO(payload)), payload
    except TrainingRefused:
        raise
    except Exception as exc:
        refuse("candidate_table_read_failed", f"{type(exc).__name__}: {exc}", path)
    finally:
        if fd is not None:
            os.close(fd)


def append_candidates(path: Path, fresh: pd.DataFrame) -> int:
    old, original_bytes = read_table(path)
    if list(old.columns) != list(fresh.columns):
        refuse("candidate_schema_mismatch", "fresh and stored candidate columns differ", path)
    required = (*KEY, "event_date", "entry_date", "pnl", "ret", *SETTLEMENT_FIELDS)
    if any(column not in old for column in required):
        refuse("candidate_schema_mismatch", "candidate identity or date columns are missing", path)
    event, entry, exits = (pd.to_datetime(pd.concat([old[name], fresh[name]]), errors="coerce") for name in ("event_date", "entry_date", "exit_date"))
    sessions = pd.concat([old.get("session", pd.Series("", index=old.index)), fresh.get("session", pd.Series("", index=fresh.index))]).astype(str).str.upper()
    if event.isna().any() or entry.isna().any() or exits.isna().any() or exits.lt(entry).any() or (event.lt(entry) & ~sessions.eq("BMO")).any() or (sessions.eq("BMO") & entry.ge(event)).any():
        refuse("candidate_causality_failed", "event, entry and exit dates must satisfy session causality", path)
    repaired_fresh = ~fresh["exit_arb_ok"].fillna(True).astype(bool) | fresh["exit_value"].lt(0)
    if fresh.loc[~repaired_fresh, ["exit_value", "entry_cost"]].isna().any(axis=None): refuse("candidate_causality_failed", "non-repaired fresh rows require exit_value and entry_cost", path)
    if fresh.loc[repaired_fresh, ["exit_value_expiry", "entry_cost"]].isna().any(axis=None): refuse("candidate_causality_failed", "repaired fresh rows require exit_value_expiry and entry_cost", path)
    fresh_expiry = pd.to_datetime(fresh["expiry"], errors="coerce")
    if fresh_expiry.loc[repaired_fresh].isna().any() or fresh_expiry.loc[repaired_fresh].lt(pd.to_datetime(fresh.loc[repaired_fresh, "entry_date"], errors="coerce")).any(): refuse("candidate_causality_failed", "repaired fresh rows require expiry on or after entry", path)
    fresh.loc[repaired_fresh, "exit_value"] = fresh.loc[repaired_fresh, "exit_value_expiry"]
    fresh["pnl"] = fresh["exit_value"] - fresh["entry_cost"]
    fresh["ret"] = fresh["pnl"].div(fresh["entry_cost"]).where(fresh["entry_cost"].gt(MIN_MEANINGFUL_COST))
    fresh = fresh.drop_duplicates(list(KEY), keep="first")
    new_keys, old_keys = (pd.MultiIndex.from_frame(frame[list(KEY)]) for frame in (fresh, old))
    repaired = ~old["exit_arb_ok"].fillna(True).astype(bool) | old["exit_value"].lt(0)
    expiry = pd.to_datetime(old["expiry"], errors="coerce")
    retained = repaired & ~old_keys.isin(new_keys) & old["exit_value_expiry"].notna() & old["entry_cost"].notna()
    if expiry.loc[retained].isna().any(): refuse("candidate_causality_failed", "retained repaired rows require expiry date", path)
    if expiry.loc[retained].lt(pd.to_datetime(old.loc[retained, "entry_date"], errors="coerce")).any(): refuse("candidate_causality_failed", "retained expiry must not precede entry", path)
    history_changed = retained & old["exit_date"].ne(expiry)
    old.loc[retained, "exit_date"] = expiry.loc[retained]
    updates = fresh[list(KEY) + list(SETTLEMENT_FIELDS)].rename(
        columns={name: f"_fresh_{name}" for name in SETTLEMENT_FIELDS})
    columns = old.columns
    old = old.merge(updates, on=list(KEY), how="left", sort=False)
    matched = old["_fresh_exit_date"].notna()
    changed = pd.Series(False, index=old.index)
    for name in SETTLEMENT_FIELDS:
        fresh_value = old.pop(f"_fresh_{name}")
        changed |= matched & old[name].ne(fresh_value) & ~(old[name].isna() & fresh_value.isna())
        old[name] = fresh_value.where(matched, old[name])
    old = old[columns]
    unseen = fresh.loc[~new_keys.isin(old_keys)]
    if unseen.empty and not changed.any() and not history_changed.any():
        return 0
    temp_path = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temp_path = Path(name)
        os.close(fd)
        pd.concat([old, unseen], ignore_index=True).to_parquet(temp_path, index=False)
        os.chmod(temp_path, path.stat().st_mode)
        with temp_path.open("rb") as staged:
            os.fsync(staged.fileno())
        path.with_suffix(path.suffix + ".bak").write_bytes(original_bytes)
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        refuse("candidate_table_write_failed", f"{type(exc).__name__}: {exc}", path)
    return len(unseen)


def refresh(month: str, path: Path = CANDIDATES) -> int:
    month = require_month(month, path)
    if not path.is_file():
        refuse("candidate_table_missing", "stored EXP-137 candidate table is required", path)
    started = time.monotonic()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[chooser-refresh] start month={month} RSS={rss:.0f}MB", file=sys.stderr, flush=True)
    try:
        builder_path = ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str/build.py"
        spec = importlib.util.spec_from_file_location("chooser_exp137_builder", builder_path)
        if spec is None or spec.loader is None:
            raise ImportError("could not load EXP-137 builder")
        builder = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = builder
        spec.loader.exec_module(builder)
        built = builder.build_all(force=True, limit_exit_month=month)
        added = append_candidates(path, built["trades"])
    except TrainingRefused:
        raise
    except Exception as exc:
        refuse("candidate_build_failed", f"{type(exc).__name__}: {exc}", path)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[chooser-refresh] done added={added} elapsed_s={time.monotonic()-started:.1f} RSS={rss:.0f}MB",
          file=sys.stderr, flush=True)
    return added


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, help="effective candidate exit month (YYYY-MM)")
    refresh(parser.parse_args(argv).month)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
