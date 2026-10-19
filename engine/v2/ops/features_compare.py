"""Numeric comparison of a rebuilt features table against the pinned one (snapshot mode)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COMPARISON_ID = "numeric.v1"
FEATURES_RTOL = 1e-9
FEATURES_ATOL = 1e-12
MAX_REPORTED_COLUMNS = 5
# Per-row provenance stamps: checked for form (non-null sha256 hex), never value-compared,
# because an incremental tier4 build legitimately keeps carried rows at an older panel hash.
PROVENANCE_COLUMNS = {"tier4": ("tier3_snapshot",)}
_SHA256_RE = r"[0-9a-f]{64}"
# Row cap on the stamp-column decode of a byte-identical table (about 25x today's tier4 rows);
# a larger table is refused rather than decoded.
STAMP_MAX_ROWS = 5_000_000
# Decoded-size cap per stamp column (footer-reported uncompressed bytes): 64 chars plus overhead per row.
STAMP_MAX_BYTES = STAMP_MAX_ROWS * 96
TABLE_KEYS = {"panel": ("ticker", "date"), "tier4": ("ticker", "event_date")}
IDENTICAL = {"verdict": "match", "max_abs_diff": 0.0, "max_rel_diff": 0.0,
             "n_columns_differing": 0, "mismatches": []}


def _structural(mismatches):
    return {"verdict": "mismatch", "mismatches": mismatches,
            "max_abs_diff": 0.0, "max_rel_diff": 0.0, "n_columns_differing": 0}


def _schema_entry(rebuilt, pinned):
    if list(rebuilt.columns) == list(pinned.columns) and \
            [str(t) for t in rebuilt.dtypes] == [str(t) for t in pinned.dtypes]:
        return None
    rebuilt_names, pinned_names = set(rebuilt.columns), set(pinned.columns)
    common = [c for c in rebuilt.columns if c in pinned_names]
    return {"reason": "schema",
            "rebuilt_only": [c for c in rebuilt.columns if c not in pinned_names][:MAX_REPORTED_COLUMNS],
            "pinned_only": [c for c in pinned.columns if c not in rebuilt_names][:MAX_REPORTED_COLUMNS],
            "dtype_changed": [c for c in common
                              if str(rebuilt.dtypes[c]) != str(pinned.dtypes[c])][:MAX_REPORTED_COLUMNS]}


def _compare_float(a, b):
    """``(n_pattern, n_bad, max_abs, max_rel)`` for two float64 arrays."""
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    n_pattern = int(np.count_nonzero(~same & (~np.isfinite(a) | ~np.isfinite(b))))
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return n_pattern, 0, 0.0, 0.0
    d = np.abs(a[mask] - b[mask])
    bm = b[mask]
    nonzero = bm != 0
    max_abs = float(d.max())
    max_rel = float((d[nonzero] / np.abs(bm[nonzero])).max()) if nonzero.any() else 0.0
    n_bad = int(np.count_nonzero(d > FEATURES_ATOL + FEATURES_RTOL * np.abs(bm)))
    return n_pattern, n_bad, max_abs, max_rel


def _bad_stamp_count(series):
    return int((~series.astype("string").str.fullmatch(_SHA256_RE).fillna(False).astype(bool)).sum())


def _stamp_cap_refusal(parquet_file, columns):
    """The mismatch entry for a stamp column too large to decode, from the footer alone; else None."""
    md = parquet_file.metadata
    if columns and md.num_rows > STAMP_MAX_ROWS:
        return {"reason": "stamp_rows_over_cap", "n_rows": md.num_rows}
    for c in columns:
        size = sum(md.row_group(i).column(j).total_uncompressed_size
                   for i in range(md.num_row_groups) for j in range(md.num_columns)
                   if md.row_group(i).column(j).path_in_schema == c)
        if size > STAMP_MAX_BYTES:
            return {"reason": "stamp_bytes_over_cap", "column": c, "n_bytes": size}
    return None


def _footer_refusal(name, path):
    """``(stamp_columns_present, cap_refusal_or_None)`` for one table file, from the parquet footer alone."""
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(path)
    cols = [c for c in PROVENANCE_COLUMNS.get(name, ()) if c in set(meta.schema_arrow.names)]
    return cols, _stamp_cap_refusal(meta, cols)


def _cap_refusals(name, paths):
    """Cap refusal entries (tagged with the side) for every table file whose stamp column is too large."""
    found = []
    for side, path in paths:
        refusal = _footer_refusal(name, path)[1]
        if refusal:
            found.append(dict(refusal, side=side))
    return found


def compare_identical(name, path) -> dict:
    """Result for a byte-identical pair of tables: nothing to diff, but provenance stamps
    must still be well formed. Reads only the stamp columns (never the whole table)."""
    cols, refusal = _footer_refusal(name, path)
    if refusal:
        return dict(IDENTICAL, verdict="mismatch", mismatches=[refusal])
    out = []
    for c in cols:
        # the pair is one file, so count each bad row once per side, as compare_tables does
        n_bad = 2 * _bad_stamp_count(pd.read_parquet(path, columns=[c])[c])
        if n_bad:
            out.append({"reason": "provenance_stamp", "column": c, "n_rows": n_bad})
    return dict(IDENTICAL, verdict="mismatch" if out else "match", mismatches=out)


def _stamp_mismatches(name, rebuilt, pinned):
    """Provenance-stamp columns are checked for form (non-null sha256 hex) on both sides."""
    out = []
    for c in [c for c in PROVENANCE_COLUMNS.get(name, ()) if c in rebuilt.columns]:
        n_bad = sum(_bad_stamp_count(f[c]) for f in (rebuilt, pinned))
        if n_bad:
            out.append({"reason": "provenance_stamp", "column": c, "n_rows": n_bad})
    stamps = list(PROVENANCE_COLUMNS.get(name, ()))
    return out, rebuilt.drop(columns=stamps, errors="ignore"), pinned.drop(columns=stamps, errors="ignore")


def _precheck(name, rebuilt_path, pinned_path):
    """Structural result when a table is missing or a stamp column is over its cap; else None."""
    if not rebuilt_path.exists() or not pinned_path.exists():
        return _structural([{"reason": "table_missing", "rebuilt": rebuilt_path.exists(),
                             "pinned": pinned_path.exists()}])
    capped = _cap_refusals(name, (("rebuilt", rebuilt_path), ("pinned", pinned_path)))
    return _structural(capped) if capped else None


def compare_tables(name, rebuilt_path, pinned_path) -> dict:
    """Compare two tables of the same contract; never raises for a data
    difference, and every value in the result is json-safe (never a cell value)."""
    rebuilt_path, pinned_path = Path(rebuilt_path), Path(pinned_path)
    early = _precheck(name, rebuilt_path, pinned_path)
    if early is not None:
        return early
    rebuilt = pd.read_parquet(rebuilt_path)
    pinned = pd.read_parquet(pinned_path)
    schema = _schema_entry(rebuilt, pinned)
    if schema is not None:
        return _structural([schema])
    if len(rebuilt) != len(pinned):
        return _structural([{"reason": "row_count", "rebuilt": len(rebuilt),
                             "pinned": len(pinned)}])
    keys = [{"reason": "key", "column": c} for c in TABLE_KEYS.get(name, ())
            if c not in rebuilt.columns or not rebuilt[c].equals(pinned[c])]
    if keys:
        return _structural(keys)
    max_abs = max_rel = 0.0
    n_columns = 0
    mismatches, rebuilt, pinned = _stamp_mismatches(name, rebuilt, pinned)
    for c in rebuilt.columns:
        if pd.api.types.is_float_dtype(rebuilt[c]):
            a = rebuilt[c].to_numpy(dtype="float64")
            b = pinned[c].to_numpy(dtype="float64")
            n_pattern, n_bad, col_abs, col_rel = _compare_float(a, b)
            if n_pattern:
                mismatches.append({"reason": "nan_pattern", "column": c, "n_rows": n_pattern})
                continue
            max_abs = max(max_abs, col_abs)
            max_rel = max(max_rel, col_rel)
            if col_abs > 0:
                n_columns += 1
            if n_bad:
                mismatches.append({"reason": "float_beyond_tolerance", "column": c,
                                   "n_rows": n_bad, "max_abs_diff": col_abs,
                                   "max_rel_diff": col_rel})
        else:
            na_r, na_p = rebuilt[c].isna().to_numpy(dtype=bool), pinned[c].isna().to_numpy(dtype=bool)
            both = ~na_r & ~na_p
            n_diff = int(np.count_nonzero(na_r != na_p)) + int((rebuilt[c][both] != pinned[c][both]).sum())
            if n_diff:
                mismatches.append({"reason": "non_float", "column": c, "n_rows": n_diff})
    return {"verdict": "mismatch" if mismatches else "match",
            "mismatches": mismatches[:MAX_REPORTED_COLUMNS],
            "max_abs_diff": max_abs, "max_rel_diff": max_rel,
            "n_columns_differing": n_columns}
