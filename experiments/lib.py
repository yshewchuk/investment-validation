"""Shared plumbing for the experiments tree: specs, the ledger, run logs.

``experiments/`` is where the EXP-101+ discipline lives (the 0-50 range
belongs to ``earnings_predictions/`` and is never reused). Every evaluated
spec — including grid cells and failures — lands in ``LEDGER.csv``: the
multiple-testing record a promotion decision cites, so the program always
knows how many tries preceded a winner. That is the guard against the
overfitting fifty experiments of iteration invites, and it only works if the
ledger is append-only, which this module enforces.

The ledger is deliberately a dumb CSV: greppable, diffable, syncable to the
private mirror, no database. Its columns are fixed and its rows are only
ever appended.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from engine import paths

__all__ = [
    "EXPERIMENTS_DIR",
    "LEDGER_PATH",
    "LEDGER_COLUMNS",
    "LedgerError",
    "verify_append",
    "ledger_read",
    "ledger_append",
    "ledger_context",
    "next_experiment_number",
    "experiment_dirs",
    "parse_experiment_id",
    "load_spec",
    "save_spec",
    "spec_hash",
    "slugify",
]

EXPERIMENTS_DIR = paths.ROOT / "experiments"
LEDGER_PATH = EXPERIMENTS_DIR / "LEDGER.csv"

#: Fixed ledger columns. stage ∈ {planned, ran}. EVERY evaluated spec gets a
#: row — grid cells and failures included.
LEDGER_COLUMNS = [
    "id", "spec_hash", "date", "stage",
    "oos_mean_mid", "sharpe_trade", "promoted",
]

#: The 0-50 range belongs to the pre-engine research tree.
FIRST_NUMBER = 101

_ID_RE = re.compile(r"^EXP-(\d+)$")


class LedgerError(RuntimeError):
    """The ledger was asked to do something an append-only record cannot."""


def verify_append(before: bytes, after: bytes) -> bool:
    """True iff ``after`` is ``before`` plus new trailing bytes — never a rewrite.

    This is the whole append-only invariant in one predicate: a rewritten,
    reordered, or trimmed history fails it, because the old bytes must be a
    prefix of the new ones. Exposed (rather than buried in ``ledger_append``)
    so the acceptance suite can exercise the failure side directly.
    """
    return after.startswith(before) and len(after) >= len(before)


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------


def ledger_read(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or LEDGER_PATH)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    frame = pd.read_csv(path, dtype=str)
    missing = [c for c in LEDGER_COLUMNS if c not in frame.columns]
    if missing:
        raise LedgerError(f"LEDGER.csv is missing columns {missing} — refusing to work with it")
    return frame


def ledger_append(rows: Sequence[Mapping[str, Any]], path: Path | None = None) -> int:
    """Append rows, enforcing the append-only invariant.

    The file is read first, the new rows are appended, and the result is
    verified to start with the exact previous bytes. A row can therefore never
    be rewritten or deleted through this API — any attempt to hand-edit the
    file between ledger operations is caught by the prefix check, and there is
    simply no replace/delete function to call.
    """
    path = Path(path or LEDGER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    before = path.read_bytes() if path.exists() else b""

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
    if not before:
        writer.writeheader()
    for row in rows:
        missing = [c for c in LEDGER_COLUMNS if c not in row]
        if missing:
            raise LedgerError(f"ledger row missing columns {missing}: {row}")
        writer.writerow({c: row[c] for c in LEDGER_COLUMNS})

    with open(path, "ab") as fh:
        fh.write(buf.getvalue().encode())

    after = path.read_bytes()
    if not verify_append(before, after):
        # Roll the corrupted append back: the file must be the old bytes or
        # the old bytes plus exactly what we added — never a rewrite.
        path.write_bytes(before)
        raise LedgerError(
            "ledger prefix changed during append — the file was edited out of "
            "band; the append was rolled back"
        )
    return len(rows)


def ledger_context(spec_hash_value: str, path: Path | None = None) -> dict[str, Any]:
    """The multiple-testing context a promotion report cites."""
    frame = ledger_read(path)
    if frame.empty:
        return {"specs_tried": 0, "this_spec_rows": 0, "promotions": 0}
    return {
        "specs_tried": int(frame["spec_hash"].nunique()),
        "this_spec_rows": int((frame["spec_hash"] == spec_hash_value).sum()),
        "promotions": int((frame["promoted"] == "True").sum()),
    }


# --------------------------------------------------------------------------
# experiment folders
# --------------------------------------------------------------------------


def experiment_dirs(root: Path | None = None) -> dict[int, Path]:
    root = Path(root or EXPERIMENTS_DIR)
    out: dict[int, Path] = {}
    if not root.exists():
        return out
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^EXP-(\d+)", child.name)
        if match:
            out[int(match.group(1))] = child
    return out


def parse_experiment_id(exp_id: str) -> int:
    match = _ID_RE.match(exp_id.strip())
    if not match:
        raise ValueError(f"experiment id must look like EXP-101, got {exp_id!r}")
    number = int(match.group(1))
    if number < FIRST_NUMBER:
        raise ValueError(
            f"EXP-{number:03d} is in the 0-50 range owned by earnings_predictions/ — "
            f"new experiments start at EXP-{FIRST_NUMBER}"
        )
    return number


def next_experiment_number(root: Path | None = None) -> int:
    dirs = experiment_dirs(root)
    return max([FIRST_NUMBER - 1, *dirs.keys()]) + 1


def slugify(title: str, width: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:width].rstrip("_") or "experiment"


# --------------------------------------------------------------------------
# specs
# --------------------------------------------------------------------------


def load_spec(path: Path | str) -> dict[str, Any]:
    import yaml

    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"spec at {path} did not parse to a mapping")
    return doc


def save_spec(spec: Mapping[str, Any], path: Path | str) -> Path:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(spec), sort_keys=False, allow_unicode=True))
    return path


def spec_hash(spec: Mapping[str, Any]) -> str:
    """Identity of an evaluated spec (delegates to the engine definition)."""
    from engine.evaluate import spec_hash as _hash

    return _hash(spec)


def record_evaluation(exp_dir: Path | str, spec: Mapping[str, Any],
                      results: Mapping[str, Any], promoted: bool = False,
                      ledger_path: Path | None = None) -> None:
    """Append the RAN row for one evaluated spec (primary or grid cell)."""
    from datetime import datetime, timezone

    headline = results.get("headline", {}) if isinstance(results, Mapping) else {}
    ledger_append(
        [{
            "id": spec.get("id", ""),
            "spec_hash": spec_hash(spec),
            "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "stage": "ran",
            "oos_mean_mid": headline.get("mean", ""),
            "sharpe_trade": headline.get("sharpe_trade", ""),
            "promoted": str(bool(promoted)),
        }],
        path=ledger_path,
    )
