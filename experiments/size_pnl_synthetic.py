"""EXP-188 SY1: synthetic-predictor generator (design guides/size_pnl_synthetic_calibration_design.md sections 2-3). Pure numpy/pandas over admitted inner-year labels; no store or raw table reader. Every output is SYNTHETIC and feeds no roster, tuning receipt, winner or board decision.
"""
import hashlib
import sys
import time

import numpy as np
import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments import size_pnl_readers as readers
from experiments.size_pnl_tuning import SYNTHETIC_PREFIX

_DENIED = "HOLDOUT_ACCESS_DENIED"
OUT_OF_SCOPE = "SYNTHETIC_OUT_OF_SCOPE"
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_FAMILIES = ("oracle", "median_noise", "mean_noise", "regression_direction", "shrinkage", "level_shift")


def _is_real(value):
    """True for a finite real number that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _grid_ok(values, needed, ok):
    """A list of reals passing ``ok`` with no duplicates; non-empty iff ``needed`` (the family is in the arm)."""
    values = [] if values is None else values
    if not isinstance(values, list) or bool(values) != needed:
        return False
    return all(_is_real(v) and ok(v) for v in values) and len(set(values)) == len(values)


def _valid(arm):
    """Whether ``arm`` is a well-formed synthetic_arm block."""
    if not isinstance(arm, dict) or not (isinstance(arm.get("seed_tag"), str) and arm["seed_tag"]):
        return False
    years, families = arm.get("inner_years"), arm.get("families")
    if not (isinstance(years, list) and len(years) == 2 and all(_is_int(y) for y in years)
            and 2023 <= years[0] <= years[1] <= 2024):
        return False
    if not (isinstance(families, list) and families and all(isinstance(f, str) and f in _FAMILIES for f in families)
            and len(set(families)) == len(families)):
        return False
    return (_grid_ok(arm.get("spread_ladder"), True, lambda v: v > 0)
            and _grid_ok(arm.get("shrinkage_L"), "shrinkage" in families, lambda v: 0 < v <= 1)
            and _grid_ok(arm.get("level_shift"), "level_shift" in families, lambda v: v > -1))


def _checked(arm):
    """``arm`` itself when it is well formed; otherwise the synthetic_arm_invalid refusal."""
    if not _valid(arm):
        raise fail("INVALID_EXPERIMENT_SPEC", "the spec's synthetic_arm block is missing or invalid",
                   details={"reason": "synthetic_arm_invalid"})
    return arm


def arm_from_spec(spec):
    """The validated synthetic_arm block of the parsed spec; refuses when it is missing or malformed."""
    return _checked(spec.get("synthetic_arm") if isinstance(spec, dict) else None)


def predictors(arm):
    """The ordered (family, params) predictor list for the arm's families (19 under the default spec)."""
    _checked(arm)
    out = []
    for name in _FAMILIES:
        if name not in arm["families"]:
            continue
        if name == "oracle":
            out.append({"family": name, "params": {}})
        elif name == "shrinkage":
            out += [{"family": name, "params": {"L": L}} for L in arm["shrinkage_L"]]
        elif name == "level_shift":
            out += [{"family": name, "params": {"shift": k}} for k in arm["level_shift"]]
        else:
            out += [{"family": name, "params": {"s": s}} for s in arm["spread_ladder"]]
    return out


def model_id(arm, predictor):
    """The stable synthetic model id: SYNTHETIC_PREFIX plus the hash of the predictor's definition.

    Refuses unless ``arm`` is well formed and ``predictor`` is one of ``predictors(arm)``.
    """
    _checked(arm)
    if not isinstance(predictor, dict) or predictor not in predictors(arm):
        raise fail("INVALID_EXPERIMENT_SPEC", "the synthetic predictor is not one of the arm's predictors",
                   details={"reason": "synthetic_predictor_invalid"})
    return SYNTHETIC_PREFIX + content_hash({"source": "synthetic", "family": predictor["family"],
                                            "params": predictor["params"], "seed_tag": arm["seed_tag"]})


def _z(event_ids, seed_tag):
    """One standard normal per event, seeded by the event id and seed tag only (common to every noisy predictor)."""
    return np.array([np.random.default_rng(int.from_bytes(
        hashlib.sha256(f"{event_id}|{seed_tag}".encode()).digest()[:8], "big")).standard_normal()
        for event_id in event_ids], dtype=float)


def _prior_mean(dates, y):
    """Mean of y over events dated strictly before each event's calendar quarter; NaN when there is none."""
    order = np.argsort(dates.to_numpy(), kind="mergesort")
    csum = np.concatenate([[0.0], np.cumsum(y[order])])
    cut = np.searchsorted(dates.to_numpy()[order], dates.to_period("Q").to_timestamp().to_numpy(), side="left")
    return np.where(cut > 0, csum[cut] / np.maximum(cut, 1), np.nan)


def _forecast(pred, y, z, prior):
    """The predictor's forecast of y per event; never clipped, NaN where undetermined."""
    fam, p = pred["family"], pred["params"]
    if fam == "oracle":
        return y
    if fam == "shrinkage":
        return (1 - p["L"]) * y + p["L"] * prior
    if fam == "level_shift":
        return y * (1 + p["shift"])
    s = p["s"]
    if fam == "median_noise":
        return y * np.exp(s * z)
    if fam == "mean_noise":
        return y * np.exp(s * z - s * s / 2)
    return y * np.exp(s * z + s * s / 2)


def _dates_ok(labels):
    """Whether event_date converts with pd.to_datetime with no NaT."""
    try:
        return not pd.to_datetime(labels["event_date"]).isna().any()
    except (ValueError, TypeError):
        return False


def _labels_ok(labels):
    """Whether labels is a non-empty frame with finite numeric abs_move, valid event_date, and a unique string event_id index."""
    return (isinstance(labels, pd.DataFrame) and {"abs_move", "event_date"} <= set(labels.columns)
            and len(labels) > 0 and labels.index.name == "event_id" and labels.index.is_unique
            and all(isinstance(i, str) for i in labels.index)
            and pd.api.types.is_numeric_dtype(labels["abs_move"])
            and np.isfinite(labels["abs_move"].to_numpy(dtype=float)).all() and _dates_ok(labels))


def _guard(admitted, labels, arm):
    """Refuses unadmitted ids, label dates not equal to the admitted date, and admitted dates outside the inner years."""
    events = admitted["events"]
    if not labels.index.isin(events["event_id"].to_numpy()).all():
        raise fail(_DENIED, "a synthetic label id is not admitted",
                   details={"reason": OUT_OF_SCOPE, "why": "event_not_admitted"})
    admitted_dates = pd.Series(pd.to_datetime(events["event_date"]).to_numpy(), index=events["event_id"].to_numpy())
    want = admitted_dates.reindex(labels.index)
    mismatch = pd.to_datetime(labels["event_date"]).to_numpy() != want.to_numpy()
    if mismatch.any():
        raise fail(_DENIED, "a synthetic label date differs from its admitted date",
                   details={"reason": OUT_OF_SCOPE, "why": "date_not_admitted", "events": int(mismatch.sum())})
    y0, y1 = arm["inner_years"]
    inside = ((want >= pd.Timestamp(f"{y0}-01-01")) & (want < pd.Timestamp(f"{y1 + 1}-01-01"))).to_numpy()
    if not inside.all():
        raise fail(_DENIED, "a synthetic event date is outside the inner years",
                   details={"reason": OUT_OF_SCOPE, "why": "date_out_of_scope", "events": int((~inside).sum())})


def _entry(pred, mid, ctx, f):
    """One ready entry: the kept table, drop counts, binding and metrics for predictor ``pred``."""
    ids, dates, y, crush = ctx["ids"], ctx["dates"], ctx["y"], ctx["crush"]
    finite, has = np.isfinite(f), ~np.isnan(crush)
    keep = finite & has
    n1, n2 = int((finite & ~has).sum()), int((~finite).sum())
    table = pd.DataFrame({"event_date": dates[keep], "pred_move": f[keep], "pred_crush": crush[keep]},
                         index=pd.Index(ids[keep], name="event_id"))
    if len(table) + n1 + n2 != len(ids) or not table.index.isin(ids).all():
        raise fail(_DENIED, "the synthetic output ids differ from the input ids",
                   details={"reason": OUT_OF_SCOPE, "why": "output_ids_differ"})
    err = f[keep] - y[keep]
    binding = content_hash({"model_id": mid, "production": ctx["binding"], "labels": ctx["labels_hash"]})
    return {"status": "ready", "source": "synthetic", "family": pred["family"], "params": pred["params"],
            "table": table, "binding": binding, "dropped": {"no_crush_forecast": n1, "undetermined_mean": n2},
            "metrics": {"n": int(keep.sum()), "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "mae": float(np.mean(np.abs(err)))}}


def generate(admitted, labels, production, arm):
    """Synthetic forecast entries keyed by model_id, built only from the admitted inner-year labels."""
    _checked(arm)
    start = time.monotonic()
    if not isinstance(admitted, readers.Admitted):
        raise fail("HOLDOUT_ACCESS_DENIED", "the synthetic generator accepts only an admission issued by admit_events",
                   details={"reason": "admission_not_issued"})
    if not isinstance(production, readers.Production):
        raise fail(_CONFLICT, "the synthetic generator accepts only a production issued by read_production",
                   details={"reason": "production_not_issued"})
    prod = readers._checked_production(production["frame"], admitted)
    table_hash = production.get("table_hash")
    if not readers._is_hash(table_hash) or readers._production_binding(table_hash, prod) != production["binding"]:
        raise fail(_CONFLICT, "the production binding does not match its frame and table hash",
                   details={"reason": "production_binding_mismatch"})
    if not _labels_ok(labels):
        raise fail(_CONFLICT, "the synthetic labels are malformed", details={"reason": "synthetic_labels_malformed"})
    _guard(admitted, labels, arm)
    labels = labels.sort_index(kind="mergesort")
    ids = labels.index
    dates = pd.DatetimeIndex(pd.to_datetime(labels["event_date"]))
    y = labels["abs_move"].to_numpy(dtype=float)
    labels_hash = content_hash([[str(i), d.isoformat(), float(v)] for i, d, v in zip(ids, dates, y)])
    ctx = {"ids": ids, "dates": dates, "y": y, "binding": production["binding"], "labels_hash": labels_hash,
           "crush": prod["pred_crush"].reindex(ids).to_numpy(dtype=float)}
    z, prior = _z(ids, arm["seed_tag"]), _prior_mean(dates, y)
    out = {}
    for pred in predictors(arm):
        mid = model_id(arm, pred)
        assert mid not in out, "synthetic model ids must be unique"
        out[mid] = _entry(pred, mid, ctx, _forecast(pred, y, z, prior))
    print(f"[pnl_synthetic] events={len(ids)} predictors={len(out)} excluded={admitted['excluded']} "
          f"elapsed={time.monotonic() - start:.1f}s", file=sys.stderr)
    return out
