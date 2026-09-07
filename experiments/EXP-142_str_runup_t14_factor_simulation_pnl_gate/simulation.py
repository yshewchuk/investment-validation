"""Factor models and option repricing for EXP-142.

Every public function is deterministic. Model fits and residual pools receive
only rows dated before the outer test year; the caller records that audit.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


SEED = 20260829
MIN_POOL = 250
MIN_VOL = 0.01
MIN_SPOT_FRACTION = 0.0001
BASIS_LOW = 0.5
BASIS_HIGH = 2.0

TARGETS = (
    "target_log_exern",
    "target_log1p_event_var",
    "target_log1p_abs_spot",
)
SCORE_ARMS = (
    "sim_surface_point",
    "sim_surface_swarm",
    "sim_joint",
    "sim_joint_independent",
)


def factor_model():
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=SEED,
    )


def black_scholes_call(spot, strike, years, vol):
    spot = np.asarray(spot, dtype=float)
    vol = np.maximum(np.asarray(vol, dtype=float), MIN_VOL)
    if float(years) <= 0:
        return np.maximum(spot - float(strike), 0.0)
    root_t = np.sqrt(float(years))
    d1 = (np.log(spot / float(strike)) + 0.5 * vol**2 * float(years)) / (vol * root_t)
    d2 = d1 - vol * root_t
    return spot * norm.cdf(d1) - float(strike) * norm.cdf(d2)


def black_scholes_put(spot, strike, years, vol):
    spot = np.asarray(spot, dtype=float)
    vol = np.maximum(np.asarray(vol, dtype=float), MIN_VOL)
    if float(years) <= 0:
        return np.maximum(float(strike) - spot, 0.0)
    root_t = np.sqrt(float(years))
    d1 = (np.log(spot / float(strike)) + 0.5 * vol**2 * float(years)) / (vol * root_t)
    d2 = d1 - vol * root_t
    return float(strike) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def black_scholes_straddle(spot, strike, years, vol):
    return (
        black_scholes_call(spot, strike, years, vol)
        + black_scholes_put(spot, strike, years, vol)
    )


def implied_straddle_vol(spot, strike, years, value):
    values = (float(spot), float(strike), float(years), float(value))
    if not np.isfinite(values).all() or min(values) <= 0:
        return float("nan")
    intrinsic = abs(float(spot) - float(strike))
    if float(value) <= intrinsic + 1e-8:
        return float("nan")
    objective = lambda vol: float(
        black_scholes_straddle(float(spot), float(strike), float(years), vol)
        - float(value)
    )
    try:
        if objective(5.0) < 0:
            return float("nan")
        return float(brentq(objective, 0.001, 5.0, maxiter=100))
    except (ValueError, RuntimeError):
        return float("nan")


def event_variance(iv30_pct, exern_iv30_pct):
    iv = pd.to_numeric(iv30_pct, errors="coerce").astype(float) / 100.0
    ex = pd.to_numeric(exern_iv30_pct, errors="coerce").astype(float) / 100.0
    return np.maximum((iv**2 - ex**2) * (30.0 / 365.0), 0.0)


def parse_leg_state(raw):
    doc = json.loads(raw) if isinstance(raw, str) else raw
    entry = doc.get("entry") or []
    exits = doc.get("exit") or []
    if len(entry) < 2 or len(exits) < 2:
        return {
            "spot_entry_leg": np.nan,
            "spot_exit_actual": np.nan,
            "strike_leg": np.nan,
            "dte_entry_leg": np.nan,
            "dte_exit": np.nan,
            "entry_mid_legs": np.nan,
            "exit_mid_legs": np.nan,
        }
    return {
        "spot_entry_leg": float(doc.get("spot_entry", np.nan)),
        "spot_exit_actual": float(doc.get("spot_exit", np.nan)),
        "strike_leg": float(entry[0].get("strike", np.nan)),
        "dte_entry_leg": float(entry[0].get("dte", np.nan)),
        "dte_exit": float(exits[0].get("dte", np.nan)),
        "entry_mid_legs": float(sum(float(leg.get("price", np.nan)) for leg in entry)),
        "exit_mid_legs": float(sum(float(leg.get("price", np.nan)) for leg in exits)),
    }


def add_targets(frame):
    out = frame.copy()
    out["entry_event_var"] = event_variance(out["iv30"], out["exern_iv30"])
    out["exit_event_var"] = event_variance(out["exit_iv30"], out["exit_exern_iv30"])
    out["abs_spot_move_pct"] = (
        100.0 * np.abs(np.log(out["spot_exit_actual"] / out["spot_entry_leg"]))
    )
    out["spot_sign"] = np.sign(
        np.log(out["spot_exit_actual"] / out["spot_entry_leg"])
    )
    out["spot_sign"] = out["spot_sign"].where(out["spot_sign"] != 0, 1.0)
    out["target_log_exern"] = np.log(
        pd.to_numeric(out["exit_exern_iv30"], errors="coerce").clip(lower=0.01)
    )
    out["target_log1p_event_var"] = np.log1p(
        pd.to_numeric(out["exit_event_var"], errors="coerce").clip(lower=0.0)
    )
    out["target_log1p_abs_spot"] = np.log1p(
        pd.to_numeric(out["abs_spot_move_pct"], errors="coerce").clip(lower=0.0)
    )

    entry_surface = np.sqrt(
        (pd.to_numeric(out["exern_iv30"], errors="coerce") / 100.0) ** 2
        + out["entry_event_var"] / (pd.to_numeric(out["dte_entry_leg"], errors="coerce") / 365.0)
    )
    observed_iv = [
        implied_straddle_vol(s, k, d / 365.0, value)
        for s, k, d, value in zip(
            out["spot_entry_leg"],
            out["strike_leg"],
            out["dte_entry_leg"],
            out["entry_mid_legs"],
        )
    ]
    out["entry_contract_iv"] = observed_iv
    out["entry_iv_basis"] = (
        pd.to_numeric(out["entry_contract_iv"], errors="coerce") / entry_surface
    ).clip(BASIS_LOW, BASIS_HIGH)
    return out


def _inverse_target(name, values):
    arr = np.asarray(values, dtype=float)
    if name == "target_log_exern":
        return np.exp(arr)
    return np.maximum(np.expm1(arr), 0.0)


def _safe_corr(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def fit_outer_models(train, test, features, log: Callable[[str], None] = print):
    x_train = train[list(features)].to_numpy(dtype=float)
    x_test = test[list(features)].to_numpy(dtype=float)
    groups = train["ticker"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("factor residual cross-fit has fewer than two ticker groups")
    splitter = GroupKFold(n_splits=n_splits)
    oof = {target: np.full(len(train), np.nan) for target in TARGETS}
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, groups=groups), 1):
        log(f"factor residual fold {fold}/{n_splits}: fit {len(fit_idx):,}, validate {len(val_idx):,}")
        for target in TARGETS:
            model = factor_model().fit(x_train[fit_idx], train[target].to_numpy(dtype=float)[fit_idx])
            oof[target][val_idx] = model.predict(x_train[val_idx])

    predictions = {}
    metrics = {}
    for target in TARGETS:
        model = factor_model().fit(x_train, train[target].to_numpy(dtype=float))
        pred = model.predict(x_test)
        predictions[target] = pred
        actual_level = _inverse_target(target, test[target])
        pred_level = _inverse_target(target, pred)
        metrics[target] = {
            "n": int(len(test)),
            "r": _safe_corr(actual_level, pred_level),
            "mae": float(np.mean(np.abs(actual_level - pred_level))),
            "bias": float(np.mean(pred_level - actual_level)),
        }

    residuals = pd.DataFrame({
        target: train[target].to_numpy(dtype=float) - oof[target]
        for target in TARGETS
    })
    residuals["spot_sign"] = train["spot_sign"].to_numpy(dtype=float)
    residuals["mcap_log"] = train["mcap_log"].to_numpy(dtype=float)
    residuals["im"] = train["im"].to_numpy(dtype=float)
    residuals = residuals.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return predictions, residuals, metrics


def _bin_edges(values, quantiles):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return np.array([])
    return np.unique(np.quantile(arr, quantiles))


def _bin(value, edges):
    if not np.isfinite(value) or len(edges) < 2:
        return -1
    return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2))


def _pool_for(row, residuals, mcap_edges, im_edges):
    m_bin = _bin(float(row["mcap_log"]), mcap_edges)
    i_bin = _bin(float(row["im"]), im_edges)
    r_m = np.array([_bin(v, mcap_edges) for v in residuals["mcap_log"]])
    r_i = np.array([_bin(v, im_edges) for v in residuals["im"]])
    both = np.flatnonzero((r_m == m_bin) & (r_i == i_bin))
    if len(both) >= MIN_POOL:
        return both, "mcap_x_im"
    mcap = np.flatnonzero(r_m == m_bin)
    if len(mcap) >= MIN_POOL:
        return mcap, "mcap"
    return np.arange(len(residuals)), "global"


def _seed(event_id, arm):
    raw = hashlib.sha256(f"{event_id}|{arm}|142".encode()).digest()[:8]
    return int.from_bytes(raw, "big")


def _price_return(row, exern_pct, event_var, move_pct, sign):
    exern = np.asarray(exern_pct, dtype=float) / 100.0
    variance = np.asarray(event_var, dtype=float)
    move = np.asarray(move_pct, dtype=float)
    direction = np.asarray(sign, dtype=float)
    spot = float(row["spot_entry_leg"]) * np.exp(direction * move / 100.0)
    spot = np.maximum(spot, float(row["spot_entry_leg"]) * MIN_SPOT_FRACTION)
    years = float(row["dte_exit"]) / 365.0
    total_vol = np.sqrt(np.maximum(exern**2 + variance / years, MIN_VOL**2))
    total_vol = np.maximum(total_vol * float(row["entry_iv_basis"]), MIN_VOL)
    value = black_scholes_straddle(
        spot, float(row["strike_leg"]), years, total_vol
    )
    return (value - float(row["entry_cost"])) / float(row["entry_cost"])


def simulate_test_rows(test, predictions, residuals, draws=4000):
    mcap_edges = _bin_edges(residuals["mcap_log"], [0.0, 1 / 3, 2 / 3, 1.0])
    im_edges = _bin_edges(residuals["im"], np.linspace(0.0, 1.0, 6))
    rows = []
    fallback = {"mcap_x_im": 0, "mcap": 0, "global": 0}
    for pos, (_, row) in enumerate(test.iterrows()):
        pool, pool_kind = _pool_for(row, residuals, mcap_edges, im_edges)
        fallback[pool_kind] += 1
        pred_ex = float(predictions["target_log_exern"][pos])
        pred_ev = float(predictions["target_log1p_event_var"][pos])
        pred_move = float(predictions["target_log1p_abs_spot"][pos])
        record = {
            "event_id": str(row["event_id"]),
            "ticker": str(row["ticker"]),
            "event_date": pd.Timestamp(row["event_date"]),
            "year": int(pd.Timestamp(row["event_date"]).year),
            "realized_ret": float(row["ret"]),
            "pool_n": int(len(pool)),
            "pool_kind": pool_kind,
        }

        point = _price_return(
            row,
            _inverse_target("target_log_exern", [pred_ex]),
            _inverse_target("target_log1p_event_var", [pred_ev]),
            np.array([0.0]),
            np.array([1.0]),
        )
        record["sim_surface_point"] = float(point[0])
        record["sim_surface_point_pwin"] = float(point[0] > 0)

        for arm in ("sim_surface_swarm", "sim_joint", "sim_joint_independent"):
            rng = np.random.default_rng(_seed(record["event_id"], arm))
            ix = rng.choice(pool, size=int(draws), replace=True)
            if arm == "sim_joint_independent":
                ix_ex = rng.choice(pool, size=int(draws), replace=True)
                ix_ev = rng.choice(pool, size=int(draws), replace=True)
                ix_move = rng.choice(pool, size=int(draws), replace=True)
                ix_sign = rng.choice(pool, size=int(draws), replace=True)
            else:
                ix_ex = ix_ev = ix_move = ix_sign = ix
            exern = _inverse_target(
                "target_log_exern",
                pred_ex + residuals["target_log_exern"].to_numpy()[ix_ex],
            )
            ev = _inverse_target(
                "target_log1p_event_var",
                pred_ev + residuals["target_log1p_event_var"].to_numpy()[ix_ev],
            )
            if arm == "sim_surface_swarm":
                move = np.zeros(int(draws))
                sign = np.ones(int(draws))
            else:
                move = _inverse_target(
                    "target_log1p_abs_spot",
                    pred_move + residuals["target_log1p_abs_spot"].to_numpy()[ix_move],
                )
                sign = residuals["spot_sign"].to_numpy()[ix_sign]
            simulated = _price_return(row, exern, ev, move, sign)
            record[arm] = float(np.mean(simulated))
            record[f"{arm}_pwin"] = float(np.mean(simulated > 0))
            record[f"{arm}_p10"] = float(np.quantile(simulated, 0.10))
            record[f"{arm}_p90"] = float(np.quantile(simulated, 0.90))
            record[f"{arm}_sd"] = float(np.std(simulated))
        rows.append(record)
    return pd.DataFrame(rows), fallback


def oracle_reprice(frame):
    values = []
    for _, row in frame.iterrows():
        ret = _price_return(
            row,
            np.array([float(row["exit_exern_iv30"])]),
            np.array([float(row["exit_event_var"])]),
            np.array([float(row["abs_spot_move_pct"])]),
            np.array([float(row["spot_sign"])]),
        )
        values.append(float(ret[0]))
    out = frame[["event_id", "entry_cost", "exit_value", "ret"]].copy()
    out["oracle_ret"] = values
    out["oracle_value"] = out["entry_cost"] * (1.0 + out["oracle_ret"])
    out["value_abs_error_fraction"] = (
        (out["oracle_value"] - out["exit_value"]).abs()
        / out["exit_value"].abs().clip(lower=0.05)
    )
    return out
