"""EXP-189 M0 error model: log-scale OLS fit, fake-forecast draws and calibration check (T1).

Design: ``guides/size_width_tuning_design.md`` section 4. The model is
``log f = L + b * (log y - mean_log_y) + s * z``, fitted to paired (forecast, label) rows where both
are strictly positive and finite. Nothing is imputed; a fit with too few rows or a label with zero
log-variance is refused. ``calibrate`` reports whether fake forecasts reproduce M0's RMSE and MAE
within ``TOLERANCE`` but never raises for that: the caller omits the calibrated arm. Pure numpy:
no I/O and no logging.
"""
import math

import numpy as np

from engine.v2.ops.errors import fail

MIN_EVENTS = 300
TOLERANCE = 0.05


def paired(forecast, label):
    """``(mask, dropped)``: mask is True where both values are finite and > 0."""
    f = np.asarray(forecast, dtype=float)
    y = np.asarray(label, dtype=float)
    if f.ndim != 1 or y.ndim != 1 or f.shape != y.shape:
        raise ValueError("forecast and label must be 1-D arrays of equal length")
    f_ok = np.isfinite(f) & (f > 0)
    y_ok = np.isfinite(y) & (y > 0)
    dropped = {
        "nonpositive_forecast": int(np.sum(~f_ok)),
        "nonpositive_label": int(np.sum(f_ok & ~y_ok)),
    }
    return f_ok & y_ok, dropped


def _refuse(reason, n):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", "M0 inputs insufficient for error-model fit",
                details={"reason": reason, "n": n})


def fit(forecast, label):
    """Log-scale OLS of ``log f`` on centred ``log y`` over the paired rows."""
    mask, dropped = paired(forecast, label)
    n = int(np.sum(mask))
    if n < MIN_EVENTS:
        raise _refuse("m0_inputs_insufficient", n)
    lf = np.log(np.asarray(forecast, dtype=float)[mask])
    ly = np.log(np.asarray(label, dtype=float)[mask])
    if float(np.ptp(ly)) == 0.0:
        raise _refuse("m0_inputs_insufficient", n)
    mean_log_y = float(np.mean(ly))
    cy = ly - mean_log_y
    sxx = float(np.sum(cy * cy))
    L = float(np.mean(lf))
    b = float(np.sum(cy * (lf - L)) / sxx)
    resid = lf - (L + b * cy)
    s = math.sqrt(float(np.sum(resid * resid)) / (n - 2))
    r = resid - np.mean(resid)
    m2 = float(np.mean(r * r))
    if m2 > 0.0:
        skew = float(np.mean(r ** 3)) / m2 ** 1.5
        excess = float(np.mean(r ** 4)) / m2 ** 2 - 3.0
    else:
        skew, excess = 0.0, 0.0
    return {"L": L, "b": b, "s": s, "n": n, "mean_log_y": mean_log_y, "dropped": dropped,
            "skew": skew, "excess_kurtosis": excess}


def fake_forecast(model, y, z):
    """``exp(L + b * (log y - mean_log_y) + s * z)``, never clipped."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    return np.exp(model["L"] + model["b"] * (np.log(y) - model["mean_log_y"]) + model["s"] * z)


def _rel(value, base):
    return value / base - 1.0 if base != 0 else math.inf


def calibrate(model, forecast, label, z):
    """Compare M0 and the mean over draws of fake forecasts on the same kept rows."""
    f = np.asarray(forecast, dtype=float)
    y = np.asarray(label, dtype=float)
    mask, _ = paired(f, y)
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] != f.shape[0] or not np.all(np.isfinite(z)):
        raise ValueError("z must be finite with shape (N, len(forecast)), N >= 1")
    fk, yk, zk = f[mask], y[mask], z[:, mask]
    err_m0 = fk - yk
    rmse_m0 = float(np.sqrt(np.mean(err_m0 * err_m0)))
    mae_m0 = float(np.mean(np.abs(err_m0)))
    err = fake_forecast(model, yk, zk) - yk
    rmse_fake = float(np.mean(np.sqrt(np.mean(err * err, axis=1))))
    mae_fake = float(np.mean(np.mean(np.abs(err), axis=1)))
    rmse_rel = _rel(rmse_fake, rmse_m0)
    mae_rel = _rel(mae_fake, mae_m0)
    ok = abs(rmse_rel) <= TOLERANCE and abs(mae_rel) <= TOLERANCE
    return {"ok": bool(ok), "reason": None if ok else "fake_not_calibrated",
            "rmse_m0": rmse_m0, "rmse_fake": rmse_fake, "mae_m0": mae_m0, "mae_fake": mae_fake,
            "rmse_rel": rmse_rel, "mae_rel": mae_rel}


def t1_markdown(model, calibration):
    """Report table T1 for the fitted M0 model and its calibration check."""
    d = model["dropped"]
    status = "PASS" if calibration["ok"] else f"FAIL ({calibration['reason']})"
    return "\n".join([
        "### T1 error model (M0, 2023-24 inner)",
        "",
        "| L | b | s | n |",
        "|---|---|---|---|",
        f"| {model['L']:.4f} | {model['b']:.4f} | {model['s']:.4f} | {model['n']} |",
        "",
        f"Dropped: nonpositive_forecast={d['nonpositive_forecast']}, "
        f"nonpositive_label={d['nonpositive_label']}; "
        f"residual skew={model['skew']:.4f}, excess kurtosis={model['excess_kurtosis']:.4f}.",
        "",
        f"Calibration: {status} rmse_rel={calibration['rmse_rel']:+.3f}, "
        f"mae_rel={calibration['mae_rel']:+.3f}",
        "",
    ])
