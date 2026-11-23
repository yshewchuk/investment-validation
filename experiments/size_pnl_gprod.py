"""EXP-188 Amendment 3 G_PROD: production's STR-THRU gate, refit ONCE through 2024 (`size_pnl_gate_fold fit`), applied per model to 2025 straddle candidates; design section 'Amendment 3'. Sensitivity only, never a verdict; no refit, no threshold change. The fold artifact and the inputs table are read only after their sha256 matches the expected one.
"""
import io
import os
import stat
import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine.v2.ops.errors import fail
from experiments import label_availability, size_pnl_gate_fold as fold, size_pnl_pool as pool, size_pnl_straddle as straddle

RULE = straddle.GATE_RULE
MODEL_COLS = ("pred_abs_move", "pred_abs_move_p10", "pred_abs_move_p90", "pred_abs_move_sd", "forecast_edge")
MAX_FOLD_BYTES = 16 << 20
MAX_INPUTS_BYTES = 256 << 20
MAX_INPUT_ROWS = 5_000_000
MAX_INPUT_DECODED_BYTES = 2 << 30
_INVALID = "INVALID_EXPERIMENT_SPEC"
_BINDING_KEYS = ("gate_fold", "gate_inputs")
_SHA = re.compile(r"[0-9a-f]{64}")


def _refuse(reason, message):
    return fail(_INVALID, message, details={"reason": reason})


@dataclass(frozen=True, eq=False)
class Gate:
    """A verified G_PROD gate: the fold's model, features and threshold, the per-event inputs and both sha256s."""
    model: object
    features: tuple
    threshold: float
    inputs: pd.DataFrame
    fold_sha256: str
    inputs_sha256: str


def _read_verified(item, what, max_bytes):
    """The verified bytes of one gate file. Its path is opened without following links and hashed before any parse."""
    if not (isinstance(item, dict) and set(item) == {"path", "sha256"} and isinstance(item["path"], str) and item["path"].strip()
            and isinstance(item["sha256"], str) and _SHA.fullmatch(item["sha256"])):
        raise _refuse("gate_binding_invalid", f"gate_{what} must hold exactly a path and a sha256")
    try:
        fd = os.open(item["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, ValueError):  # a NUL byte in the path raises ValueError
        raise _refuse(f"gate_{what}_missing", f"the gate {what} is missing or not a plain file") from None
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise _refuse(f"gate_{what}_invalid", f"the gate {what} is not a regular file within its size cap")
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise _refuse(f"gate_{what}_invalid", f"the gate {what} is larger than its size cap")
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise _refuse(f"gate_{what}_sha_mismatch", f"the gate {what} does not match its expected sha256")
    return data


def _valid_fold(payload):
    """True when the fold payload is a registered G_PROD gate: features include MODEL_COLS, the registered threshold, a legal train_through."""
    if not (isinstance(payload, dict) and {"model", "features", "threshold", "train_through"} <= set(payload)):
        return False
    features, through = payload["features"], payload["train_through"]
    try:
        threshold = float(payload["threshold"])
    except (TypeError, ValueError):
        return False
    return (isinstance(features, (list, tuple)) and all(isinstance(f, str) and f.strip() for f in features)
            and len(set(features)) == len(features) and set(MODEL_COLS) <= set(features)
            and threshold == fold.REGISTERED_THRESHOLD
            and isinstance(through, int) and not isinstance(through, bool) and through == fold.MAX_TRAIN_THROUGH
            and callable(getattr(payload["model"], "predict", None)))


def load_gate(binding):
    """The verified G_PROD gate for the configured gate entries, or None when neither is configured. Both files are verified before either is parsed."""
    present = [k for k in _BINDING_KEYS if k in binding]
    if not present:
        return None
    if len(present) == 1:
        raise _refuse("gate_config_incomplete", "gate_fold and gate_inputs must be configured together")
    raw_fold = _read_verified(binding["gate_fold"], "fold", MAX_FOLD_BYTES)
    raw_inputs = _read_verified(binding["gate_inputs"], "inputs", MAX_INPUTS_BYTES)
    import joblib  # lazy: only a configured gate needs it
    # the fold is sha256-pinned (a supervisor-built HGB artifact of a few hundred KB) and capped at 16 MiB; zlib bounds the decoded size at about 1032x the file, and joblib has no pre-decode limit
    try:
        payload = joblib.load(io.BytesIO(raw_fold))
    except Exception:
        raise _refuse("gate_fold_invalid", "the gate fold artifact cannot be loaded") from None
    if not _valid_fold(payload):
        raise _refuse("gate_fold_invalid", "the gate fold artifact is not the registered gate (features, threshold or train_through)")
    features = list(payload["features"])
    try:
        meta = pq.ParquetFile(io.BytesIO(raw_inputs)).metadata
    except Exception:
        raise _refuse("gate_inputs_invalid", "the gate inputs cannot be read as parquet") from None
    if meta.num_rows > MAX_INPUT_ROWS or sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)) > MAX_INPUT_DECODED_BYTES:
        raise _refuse("gate_inputs_invalid", "the gate inputs decode to more than the size cap")
    try:
        table = pd.read_parquet(io.BytesIO(raw_inputs))
    except Exception:
        raise _refuse("gate_inputs_invalid", "the gate inputs cannot be read as parquet") from None
    if not (table.columns.is_unique and set(table.columns) == {"event_id", *features} and table["event_id"].is_unique):
        raise _refuse("gate_inputs_invalid", "the gate inputs must hold exactly event_id and the gate features, with unique event ids")
    try:
        inputs = table.assign(event_id=table["event_id"].astype(str)).set_index("event_id")[features].astype(float)
    except (TypeError, ValueError):
        raise _refuse("gate_inputs_invalid", "the gate inputs hold a non-numeric feature") from None
    return Gate(payload["model"], tuple(features), float(payload["threshold"]), inputs,
                binding["gate_fold"]["sha256"], binding["gate_inputs"]["sha256"])


def receipt_body(gate):
    """The G_PROD part of the outer receipt: the rule, both verified hashes and the threshold."""
    return {"rule": RULE, "fold_sha256": gate.fold_sha256, "inputs_sha256": gate.inputs_sha256, "threshold": gate.threshold}


def _scores(gate, events, pool_pred, pool_res):
    """Event id (str) -> (ret, reason): the fold's predicted ret with reason None, or (nan, reason) when undetermined."""
    from engine.data.features import tier4
    ids = events["event_id"].astype(str).to_numpy()
    fc = events["pred_abs_move"].to_numpy(dtype=float)
    im = events["implied_move"].to_numpy(dtype=float)
    X = gate.inputs.reindex(ids)  # NaN rows for ids the table lacks
    absent = ~np.isin(ids, gate.inputs.index.to_numpy())
    p10, p90, sd, _n = tier4.interval_for(fc, pool_pred, pool_res, floor=0.0)  # production's band rule and MIN_RESIDUALS
    X = X.assign(pred_abs_move=fc, pred_abs_move_p10=p10, pred_abs_move_p90=p90, pred_abs_move_sd=sd, forecast_edge=fc - im)
    values = X[list(gate.features)].to_numpy(dtype=float)
    ok = np.isfinite(values).all(axis=1)
    ret = np.full(len(ids), np.nan)
    if ok.any():
        ret[ok] = np.asarray(gate.model.predict(values[ok]), dtype=float)
    ok &= np.isfinite(ret)
    out = {}
    for i, eid in enumerate(ids):
        if ok[i]:
            out[eid] = (float(ret[i]), None)
        elif absent[i]:
            out[eid] = (float("nan"), "gate_input_missing")
        elif np.isnan(p10[i]) and np.isfinite(fc[i]):
            out[eid] = (float("nan"), "thin_pool")
        else:
            out[eid] = (float("nan"), "gate_input_nan")
    return out


def month_decider(gate, table, labels, start_m):
    """The per-model, per-month decision callable: decide(events, chunk) -> decision rows for G_PROD.

    The pool is the model's own strictly-earlier, printed-label pool. ``labels`` carries the `available` column; only labels
    printed strictly before the month start enter the pool. Undetermined (a missing input, a NaN input,
    a thin pool) is never counted as "no".
    """
    labels = labels[label_availability.printed_before(labels["available"], start_m.date().isoformat())]
    history = pool.pool_history(table[table["event_date"] < start_m], labels).dropna(subset=["pred_abs_move", "err_move"])
    pred = history["pred_abs_move"].to_numpy(dtype=float)
    res = history["err_move"].to_numpy(dtype=float)

    def decide(events, chunk):
        scores = _scores(gate, events, pred, res)

        def verdict(ev):
            ret, reason = scores[str(ev["event_id"])]
            if reason is not None:
                return "undetermined", reason
            return ("taken", None) if ret >= gate.threshold else ("not_taken", "below_gate_threshold")

        return straddle.decide(events, pd.DataFrame({"event_date": [], "edge": []}), RULE, chunk, gate=verdict)

    return decide
