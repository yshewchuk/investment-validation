"""On-disk fold artifacts for the training job: atomic publish, verify, resume.

A fold is built in ``folds/<fold_id>.partial-<pid>/`` and published by one
``os.replace`` to ``folds/<fold_id>/`` after its ``COMPLETE.json`` (the
sha256 of every file) is written, so a fold directory exists only when the
fold is complete. See :mod:`.job` for the layout and the resume rules.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib

from .receipts import ReceiptIssue

__all__ = ["FoldOutcome", "TrainingJobResult", "TrainingRefused"]

MEMBERSHIP_FILE = "membership_receipt.json"
LABEL_FILE = "label_availability_receipt.json"
ESTIMATOR_FILE = "estimator.joblib"
PREDICTIONS_FILE = "predictions.parquet"
COMPLETE_FILE = "COMPLETE.json"


class TrainingRefused(RuntimeError):
    def __init__(self, issues: tuple[ReceiptIssue, ...]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{i.code} at {i.path}: {i.detail}" for i in self.issues))


@dataclass(frozen=True)
class FoldOutcome:
    fold_id: str
    status: str  # "fitted" | "resumed" | "skipped" | "planned"
    n_train: int
    n_test: int


@dataclass(frozen=True)
class TrainingJobResult:
    recipe_id: str
    out_dir: Path
    outcomes: tuple[FoldOutcome, ...]

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)


def refuse(path: str, code: str, detail: str):
    raise TrainingRefused((ReceiptIssue(path, code, detail),))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def open_job(out_dir, identity: dict) -> Path:
    """Create the job directory, or confirm it holds this same job."""
    out_dir = Path(out_dir)
    manifest = out_dir / "job.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored != identity:
            changed = sorted(k for k in identity if stored.get(k) != identity[k])
            refuse("$.job", "RESUME_MISMATCH",
                   f"{out_dir} holds a job with different {changed}; use a new out_dir")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest, identity)
    folds = out_dir / "folds"
    folds.mkdir(exist_ok=True)
    for stale in folds.glob("*.partial-*"):  # an interrupted fold; never published
        shutil.rmtree(stale)
    return out_dir


def verify_complete(fdir: Path, membership: dict, label: dict) -> None:
    """A published fold must still match its hashes and the data's receipts."""
    where = f"$.folds.{membership['fold_id']}"
    marker = fdir / COMPLETE_FILE
    if not marker.is_file():
        refuse(where, "ARTIFACT_INCOMPLETE", f"{fdir} has no {COMPLETE_FILE}")
    for name, digest in json.loads(marker.read_text())["files"].items():
        path = fdir / name
        if not path.is_file() or sha256_file(path) != digest:
            refuse(f"{where}.{name}", "ARTIFACT_CORRUPT", f"{path} does not match {COMPLETE_FILE}")
    for name, fresh in ((MEMBERSHIP_FILE, membership), (LABEL_FILE, label)):
        if json.loads((fdir / name).read_text()) != json.loads(json.dumps(fresh)):
            refuse(f"{where}.{name}", "RECEIPT_MISMATCH",
                   f"stored {name} no longer matches the data; the fold is not the one on disk")


def start_fold(folds_dir: Path, fold_id: str, membership: dict, label: dict) -> Path:
    partial = folds_dir / f"{fold_id}.partial-{os.getpid()}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    write_json(partial / MEMBERSHIP_FILE, membership)
    write_json(partial / LABEL_FILE, label)
    return partial


def dump_estimator(partial: Path, model) -> None:
    joblib.dump(model, partial / ESTIMATOR_FILE)


def publish_fold(partial: Path, fold_id: str, status: str) -> None:
    files = {p.name: sha256_file(p) for p in sorted(partial.iterdir()) if p.is_file()}
    write_json(partial / COMPLETE_FILE, {"fold_id": fold_id, "status": status, "files": files})
    os.replace(partial, partial.parent / fold_id)  # the commit
