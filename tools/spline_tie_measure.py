"""Read-only tie measurement for design 6(c).

The supervisor runs this under ``tools/bounded_run.py``; owners and CI never do.
It resolves one stored v11 panel, imputes each test year's training rows exactly
as the recipe does, and reports the largest single-value share and the distinct
quantile-knot count per ``SPLINE_COLUMNS`` column. Ties are measured separately
per horizon: production fits build one training matrix per ``horizon_sessions``,
so pooling horizons would hide ties. The training window is the event-year
approximation ``2010..Y-1``, not production membership cutoffs. Opens the
catalog with the run's own ``connect(..., must_exist=True)`` (SQLite WAL
pragmas); run it on a private copy root.

Usage: to read a stored panel the tool must run with the PINNED code tree of the
run that produced it on ``PYTHONPATH`` (the panel context embeds the code closure
and environment), e.g. ``cd <pinned worktree> && PYTHONPATH=<pinned worktree>
python3 <this repo>/tools/spline_tie_measure.py --store-root ... --panel-request
...``; do not use ``-I``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import stat
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer

__all__ = ["year_of", "split_by_horizon", "measure", "format_table", "main"]

_MAX_REQUEST_BYTES = 64 * 1024 * 1024


def _load(name, relative_path):
    """Load a module from this tree by path, independent of ``PYTHONPATH``."""
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_spline_recipe = _load(
    "_spline_recipe", "engine/v2/models/magnitude_elasticnet_spline_recipe.py")
_spline_step = _load(
    "_spline_step", "engine/v2/models/training/_magnitude_spline_step.py")
SPLINE_COLUMNS = _spline_recipe.SPLINE_COLUMNS
distinct_knots = _spline_step.distinct_knots


def _read_request(path):
    """Load the panel request JSON with a bounded read."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit("panel request must be a regular file")
        if info.st_size > _MAX_REQUEST_BYTES:
            raise SystemExit("panel request exceeds 64 MiB")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            payload = handle.read(_MAX_REQUEST_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    return json.loads(payload)


def _rss_mb():
    """Current peak resident set size in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def year_of(event_id):
    """Event ids end in ``_YYYY-MM-DD``."""
    return int(event_id[-10:-6])


def split_by_horizon(rows, columns):
    """Build one training matrix and year vector per ``horizon_sessions``."""
    columns = list(columns)
    grouped = {}
    for row in rows:
        if "horizon_sessions" not in row:
            raise ValueError("row without horizon_sessions")
        grouped.setdefault(row["horizon_sessions"], []).append(row)
    groups = {}
    for horizon, horizon_rows in grouped.items():
        matrix = np.full((len(horizon_rows), len(columns)), np.nan, dtype=float)
        years = np.empty(len(horizon_rows), dtype=int)
        for index, row in enumerate(horizon_rows):
            years[index] = year_of(row["event_id"])
            for column_index, name in enumerate(columns):
                value = row.get(name)
                if value is not None:
                    matrix[index, column_index] = value
        groups[horizon] = (matrix, years)
    return groups


def measure(matrix, years, columns, test_years, first_year=2010, ks=(4, 6, 8)):
    """Measure imputed training-row ties for each column across the test years."""
    matrix = np.asarray(matrix, dtype=float)
    years = np.asarray(years, dtype=int)
    test_years = list(test_years)
    if not test_years:
        raise ValueError("test_years is empty")
    result = {name: {"max_share": 0.0, "worst_year": None,
                     "min_distinct": {k: None for k in ks},
                     "tied_years": {k: [] for k in ks}} for name in columns}
    for year in test_years:
        training = matrix[(years >= first_year) & (years <= year - 1)]
        if training.size == 0:
            raise ValueError(f"no training rows before {year}")
        imputed = SimpleImputer(
            strategy="median", keep_empty_features=True).fit_transform(training)
        for index, name in enumerate(columns):
            column = imputed[:, index]
            row = result[name]
            _, counts = np.unique(column, return_counts=True)
            share = float(counts.max() / column.size)
            if share > row["max_share"]:
                row["max_share"], row["worst_year"] = share, year
            for k in ks:
                distinct = distinct_knots(column, k)
                if row["min_distinct"][k] is None or distinct < row["min_distinct"][k]:
                    row["min_distinct"][k] = distinct
                if distinct < k:
                    row["tied_years"][k].append(year)
    for row in result.values():
        for k in ks:
            if row["min_distinct"][k] is None:
                row["min_distinct"][k] = k
    return result


def format_table(result, ks):
    """Render one line per column plus a final tied-column summary."""
    middle = ks[len(ks) // 2]
    header = ["column", "max_share", "worst_year"]
    header += [f"min_distinct@{k}" for k in ks]
    header += [f"tied_years@{k}" for k in ks]
    header += [f"tied@{middle}"]
    lines = [" ".join(header)]
    tied_columns = []
    for name, row in result.items():
        if any(row["tied_years"][k] for k in ks):
            tied_columns.append(name)
        fields = [name, f"{row['max_share']:.4f}", str(row["worst_year"])]
        fields += [str(row["min_distinct"][k]) for k in ks]
        fields += [str(len(row["tied_years"][k])) for k in ks]
        fields += [", ".join(str(year) for year in row["tied_years"][middle]) or "-"]
        lines.append(" ".join(fields))
    lines.append("tied columns at any k: " + (", ".join(tied_columns) or "none"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--panel-request", required=True)
    parser.add_argument("--test-years", nargs=2, type=int, default=(2014, 2025),
                        metavar=("LO", "HI"))
    parser.add_argument("--first-year", type=int, default=2010)
    parser.add_argument("--ks", nargs="+", type=int, default=(4, 6, 8))
    args = parser.parse_args(argv)
    if args.test_years[0] > args.test_years[1]:
        parser.error("--test-years LO must be <= HI")

    from engine.v2.data.repository import Repository
    from engine.v2.foundation import ArtifactStore
    from engine.v2.ops.catalog import connect
    from experiments.size_search_binding import read_panel, validate_panel_request

    started = time.monotonic()
    print("[spline-tie] stage=read-panel start", file=sys.stderr)
    conn = connect(f"{args.store_root}/catalog.sqlite", must_exist=True)
    repository = Repository(conn, ArtifactStore(args.store_root), pin_snapshots=True)
    request = _read_request(args.panel_request)
    validate_panel_request(request)
    source = repository.resolve(request["price_history_snapshot"])
    panel = read_panel(repository, source, request, columns=list(SPLINE_COLUMNS))
    rows = panel["rows"]
    print(f"[spline-tie] stage=read-panel done in {time.monotonic() - started:.1f}s "
          f"rows={len(rows)} rss={_rss_mb():.1f}MB", file=sys.stderr)
    code_tree = Path(
        sys.modules["experiments.size_search_binding"].__file__).resolve().parents[1]
    print(f"[spline-tie] code tree for engine/experiments: {code_tree}",
          file=sys.stderr)

    columns = list(SPLINE_COLUMNS)
    groups = split_by_horizon(rows, columns)
    counts = ", ".join(f"{horizon}={len(group[0])}"
                       for horizon, group in sorted(groups.items()))
    print(f"[spline-tie] horizons={sorted(groups)} rows={counts}",
          file=sys.stderr)

    started = time.monotonic()
    print("[spline-tie] stage=measure start", file=sys.stderr)
    for horizon in sorted(groups):
        matrix, years = groups[horizon]
        result = measure(matrix, years, columns,
                         range(args.test_years[0], args.test_years[1] + 1),
                         first_year=args.first_year, ks=tuple(args.ks))
        print(f"horizon_sessions={horizon}")
        print(format_table(result, tuple(args.ks)))
    print(f"[spline-tie] stage=measure done in {time.monotonic() - started:.1f}s "
          f"rss={_rss_mb():.1f}MB", file=sys.stderr)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[spline-tie] peak RSS {peak_mb:.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
