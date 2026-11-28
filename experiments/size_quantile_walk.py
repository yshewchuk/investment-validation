"""EXP-190 Q3a part 2 (design sections 4, 7): the quantile walk-forward and its fit receipts. No production caller
yet (Q6 driver is the first); a caller fitting real data must first pass `size_quantile_prereg.require_preregistration`."""
import json
import math
import os
import tempfile
from pathlib import Path

import sklearn

from engine.v2.foundation import content_hash, ensure_directory, fsync_directory
from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, decision, stage
from experiments import lib
from experiments.size_pnl_prereg import _read_bounded
from experiments.size_quantile_binding import bind_grid
from experiments.size_quantile_fits import arm_id, configurations, fit_fold, select_configuration
from experiments.size_quantile_folds import annual_folds
from experiments.size_quantile_sources import training_labels

SCHEMA = "size_quantile_fit.v1"
SOURCE = "Q_raw"
MAX_RECEIPT_BYTES = 64 << 20
FIXED = ("schema_version", "inputs", "levels", "configuration", "selection_losses", "arm_id", "event_ids")


def _labels(frame):
    return [(str(e), float(m)) for e, m in zip(frame["event_id"], frame["absolute_move_pct"])]


def _refuse(path, reason, message):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", f"{reason}: {path.name}: {message}", details={"reason": reason})


def _valid_outcome(document, keys, rows, width):
    status, failure, raw = document["status"], document["failure"], document["raw"]
    if status == "fit_failed":
        return raw == {} and isinstance(failure, str) and len(failure) <= 200
    if status != "ok" or failure is not None or not isinstance(raw, dict) or set(raw) != keys:
        return False
    return all(isinstance(m, list) and len(m) == rows and all(
        isinstance(r, list) and len(r) == width and all(
            type(v) is int or (type(v) is float and math.isfinite(v)) for v in r) for r in m)
        for m in raw.values())


def _load(path, expected, keys):
    try:
        document = json.loads(_read_bounded(path, MAX_RECEIPT_BYTES))
        stored = document.pop("content_hash")
        if content_hash(document) != stored:
            raise ValueError("content hash does not verify")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise _refuse(path, "fit_receipt_unreadable", "the receipt is unreadable or altered") from None
    if document.get("inputs") != json.loads(json.dumps(expected["inputs"])):
        raise _refuse(path, "fit_receipt_changed", "the receipt was fitted for different inputs")
    fixed_ok = set(document) == {*FIXED, "status", "failure", "raw"} and all(
        content_hash(document[k]) == content_hash(expected[k]) for k in FIXED)
    if not fixed_ok or not _valid_outcome(document, keys, len(expected["event_ids"]), len(expected["levels"])):
        raise _refuse(path, "fit_receipt_unreadable", "the receipt is not one this walk could write")
    return {**document, "content_hash": stored}


def _publish(path, body, keys):
    document = {**body, "content_hash": content_hash(body)}
    data = json.dumps(document, sort_keys=True, allow_nan=False).encode()
    if len(data) > MAX_RECEIPT_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", f"{path.name} exceeds {MAX_RECEIPT_BYTES} bytes")
    receipt = json.loads(data)
    ensure_directory(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)  # atomic, never overwrites
        except FileExistsError:
            return _load(path, body, keys)
        fsync_directory(path.parent)
        return receipt
    finally:
        os.unlink(tmp)


def _walk(sealed, spec, years, out_dir):
    bind_grid(spec, sealed["plan"])
    folds = annual_folds(sealed, years)
    ordinals = [int(y) for y in years]
    if any(b - a != 1 for a, b in zip(ordinals, ordinals[1:])):
        raise fail("INVALID_EXPERIMENT_SPEC", "years must be consecutive", details={"reason": "years_not_contiguous"})
    directory, receipts = Path(out_dir), {}
    keys = {f"{a}x{b}" for a, b in configurations(spec)}
    progress = Progress("quantile walk-forward", len(years))
    event_dates = {e["event_id"]: e["event_date"] for e in sealed["context"]["target_events"]}
    for index, year in enumerate(years):
        fold = folds[year]
        cutoff = fold["training_labels_before"]
        with stage(f"quantile walk {year}"):
            # Strictly sequential: Y's receipt is on disk before any later year reads labels for Y's evaluation events.
            requested = [e for y in years[:index] for e in folds[y]["evaluation_event_ids"]]
            # An event dated on or after Y's cutoff has no label available before it; the label reader refuses it.
            earlier = [e for e in requested if event_dates[e] < cutoff]
            selection = _labels(training_labels(sealed, earlier, cutoff)) if earlier else []
            train = _labels(training_labels(sealed, fold["training_event_ids"], cutoff))
            inputs = {"sealed_binding": sealed["binding"], "spec_hash": lib.spec_hash(spec),
                      "sklearn_version": sklearn.__version__, "year": year, "training_labels_before": cutoff,
                      "training_labels_hash": content_hash([list(p) for p in train]),
                      "selection_labels_hash": content_hash([list(p) for p in selection]),
                      "evaluation_event_ids_hash": content_hash(list(fold["evaluation_event_ids"])),
                      "prior_receipt_hashes": [r["content_hash"] for r in receipts.values()]}
            path = directory / f"fit_{year}.json"
            configuration, losses = select_configuration(spec, sealed, list(receipts.values()), dict(selection))
            decision("quantile walk", f"year={year} configuration={configuration[0]}x{configuration[1]} "
                     f"training={len(train)} evaluation={len(fold['evaluation_event_ids'])} "
                     f"selection_labels={len(selection)} selection_not_before_cutoff={len(requested) - len(earlier)}")
            fixed = {"schema_version": SCHEMA, "inputs": inputs, "levels": list(spec["levels"]),
                     "configuration": list(configuration), "selection_losses": losses,
                     "arm_id": arm_id(spec, sealed, SOURCE, configuration),
                     "event_ids": list(fold["evaluation_event_ids"])}
            fit_fold(spec, sealed, train, fold["evaluation_event_ids"], validate_only=True)
            if path.exists():
                receipt = _load(path, fixed, keys)
                decision("quantile walk", f"year={year} replay, no refit")
            else:
                try:
                    raw = fit_fold(spec, sealed, train, fold["evaluation_event_ids"])["raw"]
                    status, failure = "ok", None
                except ValueError as exc:
                    status, raw, failure = "fit_failed", {}, str(exc)[:200]
                    decision("quantile walk", f"year={year} fit_failed: {failure}")
                receipt = _publish(path, {**fixed, "status": status, "failure": failure, "raw": raw}, keys)
            receipts[year] = receipt
        progress.tick()
        if receipt["status"] != "ok":
            decision("quantile walk", f"year={year} {receipt['status']}: later years are not run")
            break
    return receipts


def fit_walk_forward(sealed, spec, years, out_dir):
    """Return {year: receipt} for consecutive years through the first fit_failed year; replay/write fit_<year>.json."""
    try:
        return _walk(sealed, spec, years, out_dir)
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "the quantile walk-forward ran out of memory") from None
