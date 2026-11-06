#!/usr/bin/env python3
"""Exploratory SIZE target transforms; not promotion evidence."""
import argparse, json, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

H = ["n_prior", "mean_prior_move", "mean_prior_abs_move", *[f"ema{k}_prior_{s}move" for k in (2, 4, 8, 12) for s in ("", "abs_")], "signed_streak"]
MO = ["spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5", "spy_vol20", "spy_vol60", "spy_vol252", "spy_vol20_rel252", "dist_high", "dist_ema", "ret5", "ret10", "ret20", "or_implied", "or_skewing", "or_contango", "or_rvol30", "or_exern30", "or_iv30", "or_iee", "or_fwd90_30", "or_fexern90_30", "or_exern_z252", "mcap_log", "mcap_usd", "pre_iv30", "pre_iv10", "pre_exern_iv30", "pre_exern_iv10"]
EPS = 1e-6


def transform(y, mu, sigma, kind):
    y, mu, sigma = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    if kind == "raw": return y
    if kind == "residual": return y - mu
    if kind == "z": return (y - mu) / sigma
    if kind == "log_ratio": return np.log((y + EPS) / (mu + EPS))
    raise ValueError(kind)


def inverse(z, mu, sigma, kind):
    z, mu, sigma = np.asarray(z), np.asarray(mu), np.asarray(sigma)
    if kind == "raw": return z
    if kind == "residual": return z + mu
    if kind == "z": return z * sigma + mu
    if kind == "log_ratio": return (mu + EPS) * np.exp(np.clip(z, -30, 30)) - EPS
    raise ValueError(kind)


def history_stats(frame, train_ids):
    """Prior-only population mean/std from retained panel rows; std uses ddof=0."""
    mu, sig = np.zeros(len(frame)), np.zeros(len(frame))
    for _, g in frame.groupby("ticker", sort=False):
        prior = []
        for i, row in g.sort_values("date").iterrows():
            if prior: mu[i], sig[i] = np.mean(prior), np.std(prior, ddof=0)
            if pd.notna(row.abs_move): prior.append(float(row.abs_move))
    training = frame.event_id.isin(train_ids).to_numpy()
    floor = max(float(np.median(sig[training])) * 0.5, 1e-6)
    return mu, np.maximum(sig, floor), floor


def _ridge_loo_residuals(y, fitted, x, alpha=1):
    design = np.c_[x, np.ones(len(x))]; penalty = np.eye(design.shape[1]) * alpha; penalty[-1, -1] = 0
    leverage = np.einsum("ij,ji->i", design, np.linalg.solve(design.T@design+penalty, design.T))
    return (np.asarray(y)-fitted) / np.maximum(1-leverage, 1e-9)


def _key(ticker, date):
    return f"{ticker}_{pd.Timestamp(date):%Y-%m-%d}"


def _blocks_ci(y, pred, baseline, quarters, tickers, seed=41):
    groups = [np.flatnonzero((quarters == q) & (tickers == t)) for q, t in sorted(set(zip(quarters, tickers)))]
    rng, out = np.random.default_rng(seed), []
    for _ in range(500):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        out.append([np.mean(np.abs(pred[ix]-y[ix])-np.abs(baseline[ix]-y[ix])), np.sqrt(np.mean((pred[ix]-y[ix])**2))-np.sqrt(np.mean((baseline[ix]-y[ix])**2))])
    return np.quantile(out, [0.025, 0.975], axis=0).T.tolist()


def _distribution(pred, residual, scale, y, seed=72):
    rng = np.random.default_rng(seed)
    draws = pred[:, None] + rng.choice(residual, (len(pred), 200), replace=True) * scale[:, None]
    result = {}
    for c in (0.5, 0.8, 0.9):
        lo, hi = np.quantile(draws, [(1-c)/2, (1+c)/2], axis=1)
        result[str(int(c*100))] = {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi-lo))}
    s = np.sort(draws, axis=1); n = s.shape[1]
    crps = np.mean(np.abs(draws-y[:, None]), axis=1) - np.sum((2*np.arange(1,n+1)-n-1)*s, axis=1)/(n*n)
    result["crps"] = float(np.mean(crps))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None); ap.add_argument("--report", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.panel is None:
        import engine.paths as paths
        a.panel = str(paths.PANEL)
    start = time.time()
    def log(msg):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
        print(f"[size-transform] {msg}; elapsed={time.time()-start:.1f}s peak_rss={rss:.2f}GiB", file=sys.stderr, flush=True)
    log("start report and panel selection")
    d = json.loads([l for l in open(a.report) if l.startswith('{')][-1])
    retained = set(d["frozen_eligibility"]["retained_event_ids"])
    paired = d["paired"]["paired_rows"]
    del d
    test_ids = {r["event_id"] for r in paired}
    if not test_ids <= retained or len(paired) != 751: raise ValueError("paired population mismatch")
    ds0 = ds.dataset(a.panel, format="parquet"); names = set(ds0.schema.names)
    features = {"H": [x for x in H if x in names], "MO-": [x for x in MO if x in names]}
    if len(features["H"]) != len(H) or not features["MO-"]: raise ValueError("required panel features unavailable")
    cols = list(dict.fromkeys(["ticker", "date", "abs_move", *H, *MO])); expr = None
    for eid in retained:
        ticker, day = eid.rsplit("_", 1); term = (ds.field("ticker") == ticker) & (ds.field("date") == pd.Timestamp(day))
        expr = term if expr is None else expr | term
    panel = ds0.to_table(columns=[c for c in cols if c in names], filter=expr).to_pandas().reset_index(drop=True)
    panel["event_id"] = [_key(t, d) for t, d in zip(panel.ticker, panel.date)]
    if set(panel.event_id) != retained or panel.event_id.duplicated().any(): raise ValueError("panel population mismatch")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    train_ids = set(panel.loc[panel.date.dt.year <= 2024, "event_id"]) - test_ids
    mu, sigma, floor = history_stats(panel, train_ids); panel["mu"], panel["sigma"] = mu, sigma
    tr = panel[panel.event_id.isin(train_ids) & panel.abs_move.notna()].copy()
    pmap = {r["event_id"]: r for r in paired}
    te = panel[panel.event_id.isin(test_ids)].copy().set_index("event_id").loc[list(pmap)].reset_index()
    ytr = tr.abs_move.to_numpy(float); y = np.array([pmap[k]["absolute_move_pct"] for k in te.event_id], float)
    base = np.array([pmap[k]["history_mean"] for k in te.event_id], float); incumbent = np.array([pmap[k]["incumbent"] for k in te.event_id], float)
    quarters = np.array([pmap[k]["quarter"] for k in te.event_id]); tickers = te.ticker.to_numpy()
    result = {"n_train": len(tr), "n_test": len(te), "sigma_floor_pct": floor, "label_note": "Panel abs_move is used only for training labels and causal mu/sigma; paired absolute_move_pct scores 2025.", "features": features, "baselines": {}}
    for name, pred in (("history_mean", base), ("v1.4", incumbent)):
        result["baselines"][name] = {"mae": float(np.mean(np.abs(pred-y))), "rmse": float(np.sqrt(np.mean((pred-y)**2)))}
    models = {"ridge": Ridge(alpha=1), "elasticnet": ElasticNet(alpha=.01, l1_ratio=.75, max_iter=10000), "hgb": HistGradientBoostingRegressor(max_leaf_nodes=7, random_state=0)}
    arms, dist, total, done = {}, {}, 24, 0; log("start fixed model arms")
    for fs, fcols in features.items():
        med = tr[fcols].median().fillna(0); xt = tr[fcols].fillna(med); xv = te[fcols].fillna(med)
        for kind in ("raw", "residual", "z", "log_ratio"):
            target = transform(ytr, tr.mu, tr.sigma, kind)
            for mn, estimator in models.items():
                pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
                pipe.fit(xt, target); fit = pipe.predict(xt); pred = inverse(pipe.predict(xv), te.mu, te.sigma, kind)
                key = f"{fs}/{kind}/{mn}"; err = pred-y; mae = np.mean(np.abs(err)); rmse = np.sqrt(np.mean(err**2))
                arms[key] = {"mae": float(mae), "rmse": float(rmse), "delta_vs_history_mean": {"mae": float(mae-np.mean(np.abs(base-y))), "rmse": float(rmse-np.sqrt(np.mean((base-y)**2)))}, "delta_ci95": _blocks_ci(y, pred, base, quarters, tickers)}
                if fs == "H" and kind == "raw" and mn == "ridge":
                    x = pipe[:-1].transform(xt); resid = _ridge_loo_residuals(ytr, fit, x)
                    dist["unscaled"] = _distribution(pipe.predict(xv), resid, np.ones(len(te)), y)
                    dist["sigma_scaled"] = _distribution(pipe.predict(xv), resid/tr.sigma.to_numpy(), te.sigma.to_numpy(), y)
                done += 1
                if done % 4 == 0: log(f"fit {done}/{total}; rate={done/(time.time()-start):.2f}/s eta={(total-done)/max(done/(time.time()-start),1e-9):.1f}s")
    result["arms"], result["raw_ridge_H_distribution"] = arms, dist
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    log(f"complete train={len(tr)} test={len(te)} output={out/'summary.json'}")


if __name__ == "__main__": main()
