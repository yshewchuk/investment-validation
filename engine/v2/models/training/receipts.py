"""Per-fold training-membership and label-availability receipts, and their check.

A receipt is a small JSON document written beside a fold artifact. It binds
the fold to the exact recipe (fingerprint) and dataset (fingerprint), names
its members by a hash of their keys, and records when their labels became
observable relative to the fold cutoff. :func:`receipt_issues` is the check:
it re-derives both receipts from the data and refuses, by code, when

* ``FUTURE_MEMBER`` — a training row's membership time is at or after the
  fold cutoff;
* ``FUTURE_LABEL`` — a training label became observable at or after
  ``cutoff + label.max_days_after_cutoff`` days (legacy's own tolerance: a
  print on the fold's last day is realized at the next session);
* ``LABEL_TIME_MISSING`` — a training row has a label but no availability
  date, so availability cannot be shown;
* ``UPSTREAM_IN_SAMPLE`` — an upstream model's prediction on a training row
  came from a fold that started after the row, i.e. a model that trained on
  it (in-sample leakage, the full-refit champion included);
* ``UPSTREAM_LINEAGE_MISSING`` / ``UPSTREAM_MODEL_MISMATCH`` — an upstream
  prediction with no fold lineage, or produced by a different model id;
* ``RECEIPT_MISMATCH`` — a stored receipt no longer matches what the data
  re-derives (used when resuming).

Labels past the cutoff but inside the tolerance are not failures; the
receipt counts them (``n_labels_after_cutoff``) so the legacy overlap is on
the record rather than hidden. Upstream columns with ``lineage="unrecorded"``
are counted as ``n_unverified``, never as verified.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.v2.foundation import content_hash

from .folds import FoldPlan, PreparedDataset
from .recipes import TrainingRecipe, recipe_fingerprint

__all__ = [
    "LABEL_RECEIPT_V1",
    "MEMBERSHIP_RECEIPT_V1",
    "ReceiptIssue",
    "fold_receipts",
    "receipt_issues",
]

MEMBERSHIP_RECEIPT_V1 = "p5.training_membership_receipt.v1"
LABEL_RECEIPT_V1 = "p5.label_availability_receipt.v1"


@dataclass(frozen=True, order=True)
class ReceiptIssue:
    path: str
    code: str
    detail: str


def _iso(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _keys_hash(frame: pd.DataFrame, columns, rows: np.ndarray) -> str:
    """Order-independent hash of the member keys."""
    if len(rows) == 0:
        return "sha256:" + hashlib.sha256(b"").hexdigest()
    part = frame.iloc[rows][list(columns)].astype(str)
    joined = part.agg("\x1f".join, axis=1).sort_values(kind="stable")
    return "sha256:" + hashlib.sha256("\x1e".join(joined).encode()).hexdigest()


def _time_range(times: pd.Series):
    valid = times.dropna()
    return (_iso(valid.min()), _iso(valid.max())) if len(valid) else (None, None)


def fold_receipts(recipe: TrainingRecipe, prepared: PreparedDataset, fold: FoldPlan,
                  *, dataset_fp: str) -> tuple[dict, dict]:
    """``(membership, label_availability)`` receipts for one fold."""
    frame = prepared.frame
    train = fold.train
    times = pd.to_datetime(frame[recipe.membership_time_column], errors="coerce")
    tmin, tmax = _time_range(times.iloc[train])
    cutoff = fold.cutoff
    membership = {
        "schema_version": MEMBERSHIP_RECEIPT_V1,
        "recipe_id": recipe.recipe_id,
        "recipe_fingerprint": recipe_fingerprint(recipe),
        "dataset_fingerprint": dataset_fp,
        "fold_id": fold.fold_id,
        "fold_kind": fold.kind,
        "cutoff": _iso(cutoff),
        "skipped": fold.skipped,
        "n_dataset_rows": prepared.n_source_rows,
        "n_recipe_rows": int(len(frame)),
        "n_incomplete_rows": int((~prepared.complete).sum()),
        "n_train": int(len(train)),
        "n_test": int(len(fold.test)),
        "train_keys_hash": _keys_hash(frame, recipe.key_columns, train),
        "test_keys_hash": _keys_hash(frame, recipe.key_columns, fold.test),
        "membership_time_column": recipe.membership_time_column,
        "membership_time_min": tmin,
        "membership_time_max": tmax,
        "n_members_at_or_after_cutoff": (
            int((times.iloc[train] >= cutoff).sum()) if cutoff is not None else 0),
        "feature_order_hash": content_hash(list(recipe.features)),
        "target": recipe.target.column,
        "missing_mask": recipe.missing_mask,
        "filters_hash": content_hash([[f.column, f.op, f.value] for f in recipe.filters]),
        "seeds": list(recipe.estimator.seeds),
    }
    label = _label_receipt(recipe, frame, fold)
    label["dataset_fingerprint"] = dataset_fp
    return membership, label


def _label_receipt(recipe: TrainingRecipe, frame: pd.DataFrame, fold: FoldPlan) -> dict:
    rule = recipe.label
    train = fold.train
    out = {
        "schema_version": LABEL_RECEIPT_V1,
        "recipe_id": recipe.recipe_id,
        "fold_id": fold.fold_id,
        "cutoff": _iso(fold.cutoff),
        "label_time_column": rule.time_column,
        "max_days_after_cutoff": rule.max_days_after_cutoff,
        "n_labels": int(len(train)),
    }
    if rule.time_column not in frame.columns:
        out.update({"label_time_column_present": False, "n_label_time_missing": int(len(train))})
    else:
        label_times = pd.to_datetime(frame[rule.time_column], errors="coerce").iloc[train]
        lo, hi = _time_range(label_times)
        out.update({"label_time_column_present": True,
                    "n_label_time_missing": int(label_times.isna().sum()),
                    "label_time_min": lo, "label_time_max": hi})
        if fold.cutoff is not None:
            out["n_labels_after_cutoff"] = int((label_times >= fold.cutoff).sum())
            if rule.max_days_after_cutoff is None:
                out["n_labels_beyond_allowance"] = None
                out["allowance"] = "unbounded (legacy sets no label-availability bound)"
            else:
                limit = fold.cutoff + pd.Timedelta(days=rule.max_days_after_cutoff)
                out["n_labels_beyond_allowance"] = int((label_times >= limit).sum())
                out["allowance"] = f"label time < cutoff + {rule.max_days_after_cutoff} days"
    out["upstream"] = [_upstream_receipt(recipe, frame, fold, dep) for dep in recipe.upstream]
    return out


def _upstream_receipt(recipe, frame, fold, dep) -> dict:
    train = fold.train
    present = [c for c in dep.columns if c in frame.columns and c in recipe.features]
    if present:
        values = frame[present].apply(pd.to_numeric, errors="coerce").iloc[train]
        has_value = np.isfinite(values.to_numpy(dtype=float)).any(axis=1)
    else:
        has_value = np.zeros(len(train), dtype=bool)
    row = {"produces": dep.produces, "lineage": dep.lineage, "columns": list(present),
           "n_rows_with_value": int(has_value.sum())}
    if dep.lineage != "tier4_monthly_oos":
        row.update({"n_verified": 0, "n_unverified": int(has_value.sum())})
        return row
    if dep.fold_column not in frame.columns or dep.model_id_column not in frame.columns:
        row.update({"lineage_columns_present": False, "n_verified": 0,
                    "n_lineage_missing": int(has_value.sum())})
        return row
    member_time = pd.to_datetime(frame[recipe.membership_time_column], errors="coerce").iloc[train]
    fold_start = pd.to_datetime(frame[dep.fold_column], errors="coerce").iloc[train]
    model_ids = frame[dep.model_id_column].iloc[train]
    lineage_missing = has_value & fold_start.isna().to_numpy()
    in_sample = has_value & (fold_start > member_time).fillna(False).to_numpy()
    mismatch = has_value & fold_start.notna().to_numpy() & (
        model_ids.astype(str).to_numpy() != str(dep.model_id))
    starts = fold_start[has_value]
    row.update({
        "lineage_columns_present": True,
        "expected_model_id": dep.model_id,
        "n_lineage_missing": int(lineage_missing.sum()),
        "n_in_sample": int(in_sample.sum()),
        "n_model_mismatch": int(mismatch.sum()),
        "n_verified": int((has_value & ~lineage_missing & ~in_sample & ~mismatch).sum()),
        "upstream_fold_start_max": _iso(starts.max()) if starts.notna().any() else None,
    })
    return row


def receipt_issues(recipe: TrainingRecipe, membership: dict, label: dict) -> tuple[ReceiptIssue, ...]:
    """The refusals a fold's receipts imply; empty means the fold may be fit."""
    issues: list[ReceiptIssue] = []
    fold = membership["fold_id"]

    def add(path, code, detail):
        issues.append(ReceiptIssue(f"$.folds.{fold}.{path}", code, detail))

    if membership["recipe_fingerprint"] != recipe_fingerprint(recipe):
        add("recipe_fingerprint", "RECEIPT_MISMATCH", "receipt was written for a different recipe")
    if membership["n_members_at_or_after_cutoff"]:
        add("membership", "FUTURE_MEMBER",
            f"{membership['n_members_at_or_after_cutoff']} training row(s) at or after cutoff "
            f"{membership['cutoff']}")
    if not label.get("label_time_column_present"):
        add("label", "LABEL_TIME_MISSING", f"dataset has no {label['label_time_column']!r} column")
    elif label["n_label_time_missing"]:
        add("label", "LABEL_TIME_MISSING",
            f"{label['n_label_time_missing']} training label(s) without an availability date")
    if label.get("n_labels_beyond_allowance"):
        add("label", "FUTURE_LABEL",
            f"{label['n_labels_beyond_allowance']} training label(s) observable only after "
            f"{label['allowance']} (cutoff {label['cutoff']})")
    for dep in label.get("upstream", []):
        where = f"upstream.{dep['produces']}"
        if dep.get("lineage_columns_present") is False and dep.get("n_lineage_missing"):
            add(where, "UPSTREAM_LINEAGE_MISSING",
                f"{dep['n_lineage_missing']} row(s) carry {dep['produces']} with no fold lineage columns")
            continue
        if dep.get("n_lineage_missing"):
            add(where, "UPSTREAM_LINEAGE_MISSING",
                f"{dep['n_lineage_missing']} row(s) carry {dep['produces']} with no fold_start "
                "(a full-refit or unknown producer)")
        if dep.get("n_in_sample"):
            add(where, "UPSTREAM_IN_SAMPLE",
                f"{dep['n_in_sample']} training row(s) got {dep['produces']} from a fold that "
                "started after the row (the upstream model trained on it)")
        if dep.get("n_model_mismatch"):
            add(where, "UPSTREAM_MODEL_MISMATCH",
                f"{dep['n_model_mismatch']} row(s) carry {dep['produces']} from a model other than "
                f"{dep.get('expected_model_id')}")
    return tuple(sorted(issues))
