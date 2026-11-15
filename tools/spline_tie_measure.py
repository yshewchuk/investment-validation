"""Read-only tie measurement for design 6(c).

The supervisor runs this under ``tools/bounded_run.py``; owners and CI never do.
It resolves one stored v11 panel, imputes each test year's training rows exactly
as the recipe does, and reports the largest single-value share and the distinct
quantile-knot count per ``SPLINE_COLUMNS`` column. No store writes.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import numpy as np
from sklearn.impute import SimpleImputer

from engine.v2.models.magnitude_elasticnet_spline_recipe import SPLINE_COLUMNS
from engine.v2.models.training._magnitude_spline_step import distinct_knots

__all__ = ["year_of", "measure", "format_table", "main"]


def year_of(event_id):
    """Event ids end in ``_YYYY-MM-DD``."""
    return int(event_id[-10:-6])


def measure(matrix, years, columns, test_years, first_year=2010, ks=(4, 6, 8)):
    """Measure imputed training-row ties for each column across the test years."""
    matrix = np.asarray(matrix, dtype=float)
    years = np.asarray(years, dtype=int)
    result = {name: {"max_share": 0.0, "worst_year": None,
                     "min_distinct": {k: None for k in ks},
                     "tied_years": {k: [] for k in ks}} for name in columns}
    for year in test_years:
        training = matrix[(years >= first_year) & (years <= year - 1)]
        if training.size == 0:
            continue
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

    from engine.v2.data.repository import Repository
    from engine.v2.foundation import ArtifactStore
    from engine.v2.ops.catalog import connect
    from experiments.size_search_binding import read_panel

    started = time.monotonic()
    print("[spline-tie] stage=read-panel start", file=sys.stderr)
    conn = connect(f"{args.store_root}/catalog.sqlite", must_exist=True)
    repository = Repository(conn, ArtifactStore(args.store_root), pin_snapshots=True)
    with open(args.panel_request, encoding="utf-8") as handle:
        request = json.load(handle)
    source = repository.resolve(request["price_history_snapshot"])
    panel = read_panel(repository, source, request, columns=list(SPLINE_COLUMNS))
    rows = panel["rows"]
    print(f"[spline-tie] stage=read-panel done in {time.monotonic() - started:.1f}s "
          f"rows={len(rows)}", file=sys.stderr)

    columns = list(SPLINE_COLUMNS)
    matrix = np.full((len(rows), len(columns)), np.nan, dtype=float)
    years = np.empty(len(rows), dtype=int)
    for index, row in enumerate(rows):
        years[index] = year_of(row["event_id"])
        for column_index, name in enumerate(columns):
            value = row.get(name)
            if value is not None:
                matrix[index, column_index] = value

    started = time.monotonic()
    print("[spline-tie] stage=measure start", file=sys.stderr)
    result = measure(matrix, years, columns,
                     range(args.test_years[0], args.test_years[1] + 1),
                     first_year=args.first_year, ks=tuple(args.ks))
    print(f"[spline-tie] stage=measure done in {time.monotonic() - started:.1f}s",
          file=sys.stderr)
    print(format_table(result, tuple(args.ks)))

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[spline-tie] peak RSS {peak_mb:.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
