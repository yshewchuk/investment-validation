"""Shared plumbing for the experiments tree: specs, the ledger, run logs.

``experiments/`` is where the EXP-101+ discipline lives (the 0-50 range
belongs to ``earnings_predictions/`` and is never reused). Every evaluated
real-run spec — including grid cells and failures — lands in ``LEDGER.csv``: the
multiple-testing record a promotion decision cites, so the program always
knows how many tries preceded a winner. That is the guard against the
overfitting fifty experiments of iteration invites, and it only works if the
ledger is append-only, which this module enforces. Smoke/subset grid runs
use ``record=False`` to leave this multiple-testing record unchanged.

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
from collections.abc import MutableMapping
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
    "record_evaluation",
    "record_evaluation_result",
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


def ledger_ensure(path: Path | None = None) -> Path:
    """Create a header-only ledger if none exists.

    Called at the start of every writer, so the program ledger exists from
    the first write onward: reports must be able to distinguish "no
    experiments tried yet" (N/A) from "the ledger is missing" (FAIL), and
    promotion rule (e) needs the file to be there from the first experiment
    onward. Deliberately NOT called at import time — importing this module
    must never create a file in the checkout.
    """
    path = Path(path or LEDGER_PATH)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        csv.DictWriter(buf, fieldnames=LEDGER_COLUMNS).writeheader()
        path.write_bytes(buf.getvalue().encode())
    return path


def ledger_read(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or LEDGER_PATH)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    frame = pd.read_csv(path, dtype=str)
    missing = [c for c in LEDGER_COLUMNS if c not in frame.columns]
    if missing:
        raise LedgerError(f"LEDGER.csv is missing columns {missing} — refusing to work with it")
    return frame


def _ledger_fieldnames(path: Path) -> list[str]:
    """The fieldnames one append must use: the file's own header if it has
    one, else the fixed new-ledger header. An existing ledger keeps its own
    header byte-for-byte and is never rewritten or corrupted."""
    if path.exists():
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), None)
        if header:
            return header
    return list(LEDGER_COLUMNS)


def ledger_append(rows: Sequence[Mapping[str, Any]], path: Path | None = None) -> int:
    """Append rows, enforcing the append-only invariant.

    The file is read first, the new rows are appended, and the result is
    verified to start with the exact previous bytes. A row can therefore never
    be rewritten or deleted through this API — any attempt to hand-edit the
    file between ledger operations is caught by the prefix check, and there is
    simply no replace/delete function to call.
    """
    path = Path(path or LEDGER_PATH)
    ledger_ensure(path)
    before = path.read_bytes() if path.exists() else b""
    fieldnames = _ledger_fieldnames(path)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    if not before:
        writer.writeheader()
    for row in rows:
        missing = [c for c in LEDGER_COLUMNS if c not in row]
        if missing:
            raise LedgerError(f"ledger row missing columns {missing}: {row}")
        writer.writerow({c: row.get(c, "") for c in fieldnames})

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


def metrics_path(run_dir: Path | str, spec: Mapping[str, Any]) -> Path:
    """The spec-hash-named primary metrics artifact for one spec."""
    return Path(run_dir) / "results" / f"metrics_{spec_hash(spec)[:12]}.json"


def receipt_path(run_dir: Path | str, spec: Mapping[str, Any]) -> Path:
    """The recording receipt binding a spec's metrics artifact to a run."""
    return Path(run_dir) / "results" / f"receipt_{spec_hash(spec)[:12]}.json"


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_evaluation(exp_dir: Path | str, spec: Mapping[str, Any],
                      results: Mapping[str, Any], promoted: bool = False,
                      ledger_path: Path | None = None,
                      publish_receipt: bool = True) -> None:
    """Append this run's ``ran`` row, refresh the metrics artifact's accuracy
    checklist against that exact ledger, finalize it to ``"recorded"`` and
    publish the receipt binding the run ID, spec hash and SHA-256 of the exact
    bytes promotion will later read.

    The caller's mutable ``results`` mapping is finalized in place to the same
    ``checklist``, ``checklist_fails`` and ``recording_mode="recorded"`` values
    the artifact carries, before the receipt is published — so an in-memory
    caller sees the recorded verdict without re-reading disk.

    ``publish_receipt=False`` is the ledger-only legacy mode for outcomes with
    no primary metrics artifact (e.g. a null result where the incumbent won):
    it appends the ``ran`` row and returns without validating, finalizing or
    receipting anything, so the row alone can never authorize promotion."""
    from datetime import datetime, timezone

    exp_dir = Path(exp_dir)
    ledger_path = Path(ledger_path or LEDGER_PATH)
    caller_results = results if isinstance(results, MutableMapping) else None
    results = results if isinstance(results, Mapping) else {}
    headline = results.get("headline", {}) or {}
    run_id = str(results.get("run_id", "") or "")
    ledger_append(
        [{
            "id": spec.get("id", ""),
            "spec_hash": spec_hash(spec),
            "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "stage": "ran",
            "oos_mean_mid": headline.get("mean", ""),
            "sharpe_trade": headline.get("sharpe_trade", ""),
            "promoted": str(bool(promoted)),
            "run_id": run_id,
        }],
        path=ledger_path,
    )
    if not publish_receipt:
        return

    metrics = metrics_path(exp_dir, spec)
    if not metrics.is_file():
        raise LedgerError(
            f"appended a ran row for run {run_id[:12]}… but found no metrics "
            f"artifact at {metrics}; refusing to publish an unbound receipt")
    artifact = json.loads(metrics.read_text())
    requested_hash = spec_hash(spec)
    artifact_spec_hash = str((artifact or {}).get("spec_hash", "") or "")
    if artifact_spec_hash != requested_hash:
        raise LedgerError(
            f"metrics artifact {metrics.name} carries spec_hash "
            f"{artifact_spec_hash[:12]}… but the recorded spec is "
            f"{requested_hash[:12]}… — refusing to bind a receipt to a copied "
            "foreign artifact")
    artifact_run_id = str((artifact or {}).get("run_id", "") or "")
    if artifact_run_id != run_id:
        raise LedgerError(
            f"metrics artifact {metrics.name} carries run_id {artifact_run_id[:12]}… "
            f"but the recorded run is {run_id[:12]}… — refusing to bind a receipt "
            "to a foreign artifact")
    # The checklist's ledger item depends on the ``ran`` row this call just
    # appended, so recompute it here (against the exact ledger) and store the
    # final verdict in the artifact the receipt's digest covers.
    from engine.report import accuracy_checklist

    checklist = accuracy_checklist(artifact, spec, ledger_path=ledger_path)
    finalized = [
        {"name": item.name, "status": item.status, "evidence": item.evidence}
        for item in checklist
    ]
    artifact["checklist"] = finalized
    artifact["checklist_fails"] = sum(1 for item in checklist if item.status == "FAIL")
    artifact["recording_mode"] = "recorded"
    metrics.write_text(json.dumps(artifact, indent=1, default=str))
    # Finalize the caller-held result to exactly what the artifact carries, and
    # do it before the receipt binds those bytes: no metrics write may follow
    # receipt publication.
    if caller_results is not None:
        caller_results["checklist"] = [dict(item) for item in finalized]
        caller_results["checklist_fails"] = artifact["checklist_fails"]
        caller_results["recording_mode"] = "recorded"
    receipt_path(exp_dir, spec).write_text(json.dumps({
        "run_id": run_id,
        "spec_hash": spec_hash(spec),
        "metrics_sha256": file_sha256(metrics),
    }, indent=1))


def record_evaluation_result(result: Any, spec: Mapping[str, Any],
                             *, promoted: bool = False,
                             ledger_path: Path | None = None,
                             publish_receipt: bool = True) -> None:
    """Record one evaluated run from its own ``EvalResult``.

    The run's metrics artifact lives in ``result.run_dir`` — the arm/cell
    directory for a secondary, not the primary's folder — so the receipt binds
    the artifact that run actually wrote. ``record_evaluation`` finalizes the
    same ``result.results`` mapping in place.
    """
    run_dir = getattr(result, "run_dir", None)
    if run_dir is None:
        raise LedgerError(
            "cannot record an EvalResult with no run_dir: a receipt must bind "
            "the directory that owns the run's metrics")
    results = getattr(result, "results", None)
    if not isinstance(results, Mapping):
        raise LedgerError("EvalResult carries no results mapping to record")
    record_evaluation(
        run_dir, spec, results, promoted=promoted, ledger_path=ledger_path,
        publish_receipt=publish_receipt)


ARMS_DIR = "arms"
ARMS_INDEX = "ARMS.md"


def evaluate_with_grid(spec: Mapping[str, Any], trades: pd.DataFrame, run_dir: Path | str,
                       *, ledger_path: Path | None = None, record: bool = True,
                       **evaluate_kwargs: Any):
    """Evaluate the primary spec, then every grid cell as a secondary arm.

    The primary keeps ``run_dir/REPORT.md`` and ``run_dir/figures/``; each
    secondary writes its report and figures under
    ``run_dir/arms/<spec_hash[:12]>/``, so no secondary can overwrite the
    headline evidence. ``run_dir/ARMS.md`` indexes the arms, marking the
    preregistered primary. Returns the primary's ``EvalResult``.
    ``record=False`` skips all experiment-ledger appends for smoke/subset
    runs; evaluation artifacts and preregistration checks are unchanged.
    """
    from engine.evaluate import evaluate

    # The helper owns where each arm's report goes; refuse before any artifact
    # or ledger row exists rather than fail on the first secondary arm.
    if "report_dir" in evaluate_kwargs:
        raise ValueError("evaluate_with_grid assigns report_dir for each arm")
    run_dir = Path(run_dir)
    # Removed first and written last, so a run that dies part-way never leaves
    # an index from an earlier run describing arms this run did not finish.
    (run_dir / ARMS_INDEX).unlink(missing_ok=True)
    result = evaluate(spec, trades, run_dir=run_dir, **evaluate_kwargs)
    if record:
        record_evaluation(run_dir, spec, result.results, ledger_path=ledger_path)
    arms = [("primary", "preregistered primary", result)]

    for key, values in (spec.get("grid") or {}).items():
        for value in values:
            cell = dict(spec)
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"][key] = value
            # Grid cells legitimately differ from the registered primary spec;
            # the label exempts them from the spec-hash continuity check —
            # they are secondary results, never the headline.
            cell["grid_cell"] = True
            arm_dir = run_dir / ARMS_DIR / spec_hash(cell)[:12]
            cell_result = evaluate(cell, trades, run_dir=run_dir,
                                   report_dir=arm_dir, **evaluate_kwargs)
            if record:
                record_evaluation(run_dir, cell, cell_result.results, ledger_path=ledger_path)
            arms.append(("secondary", f"{key}={value}", cell_result))

    lines = [f"# {spec.get('id')} — arms", "",
             "Exactly one arm is the preregistered primary; every other arm is "
             "a secondary grid cell and is never the headline.", ""]
    for role, label, arm in arms:
        rel = Path(arm.report_path).relative_to(run_dir).as_posix() if arm.report_path else "(no report)"
        lines.append(f"- **{role}** — {label}: [{rel}]({rel})")
    (run_dir / ARMS_INDEX).write_text("\n".join(lines) + "\n")
    return result
