"""Derive HMOSXL/HMOSXLP columns for stored fit rows (slice 4 fit-path wiring)."""
from __future__ import annotations

import sys
import time

import numpy as np

from engine.v2.research.derived_features import LOG_COLUMNS, derive_columns, derived_names
from experiments.memory_guard import rss_bytes

_LOG_NAMES = frozenset(derived_names("HMOSXL"))
_PRODUCT_NAMES = frozenset(derived_names("HMOSXLP")) - _LOG_NAMES


def derived_set(feature_columns) -> str | None:
    """Return the derived set a projected column list names, or None when it names none."""
    columns = set(feature_columns)
    if columns & _PRODUCT_NAMES:
        return "HMOSXLP"
    if columns & _LOG_NAMES:
        return "HMOSXL"
    return None


def with_derived(rows, feature_columns) -> list[dict]:
    """Return rows with the derived columns of their set, or the same rows when stored-only."""
    columns = tuple(feature_columns)
    set_name = derived_set(columns)
    if set_name is None:
        return rows
    started = time.monotonic()
    present = set(columns)
    sources = {name: np.asarray([row[name] for row in rows], dtype=float)
               for name in LOG_COLUMNS if name in present}
    derived = derive_columns(sources, set_name)
    output = [{**row, **{name: None if np.isnan(values[index]) else float(values[index])
                         for name, values in derived.items()}}
              for index, row in enumerate(rows)]
    counts = {name: int(np.isnan(values).sum()) for name, values in derived.items()}
    nulls = {name: count for name, count in sorted(counts.items()) if count}
    print(f"[derive] set={set_name} rows={len(rows)} derived={len(derived)} "
          f"null_cols={len(nulls)} nulls={nulls} elapsed={time.monotonic() - started:.1f}s "
          f"rss={rss_bytes() // (1024 * 1024)}MiB", file=sys.stderr)
    return output
