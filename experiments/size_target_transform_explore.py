#!/usr/bin/env python3
"""Exploratory SIZE target transforms; not promotion evidence."""
import argparse, json, os, resource, stat, sys, time
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
EPS = 1e-6


def score_metrics(y, pred):
    e = np.asarray(pred)-np.asarray(y); return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2)))


def transform(y, mu, sigma, kind):
    y, mu, sigma = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    if kind in ("raw", "residual"): return y if kind == "raw" else y-mu
    if kind == "z": return (y-mu)/sigma
    if kind == "log_ratio": return np.log((y+EPS)/(mu+EPS))
    raise ValueError(kind)


def inverse(z, mu, sigma, kind):
    z, mu, sigma = np.asarray(z), np.asarray(mu), np.asarray(sigma)
    if kind in ("raw", "residual"): return z if kind == "raw" else z+mu
    if kind == "z": return z*sigma+mu
    if kind == "log_ratio": return (mu+EPS)*np.exp(np.clip(z, -30, 30))-EPS
    raise ValueError(kind)


def history_stats(frame, train_ids):
    """Rebuild H and prior-only mu/sigma from retained signed moves."""
    mu, sig = np.zeros(len(frame)), np.zeros(len(frame)); feat = pd.DataFrame(0., index=frame.index, columns=H)
    for _, g in frame.groupby("ticker", sort=False):
        moves, labels = [], []
        for i, r in g.sort_values("date").iterrows():
            if moves:
                p = pd.Series(moves); sign = np.sign(moves[-1]); streak = 0
                for v in reversed(moves):
                    if np.sign(v) != sign: break
                    streak += 1
                emas = [v for k in (2, 4, 8, 12) for v in (p.ewm(span=k, adjust=False).mean().iloc[-1], p.abs().ewm(span=k, adjust=False).mean().iloc[-1])]
                feat.loc[i] = [len(moves), p.mean(), p.abs().mean(), *emas, sign*streak]
                mu[i], sig[i] = np.mean(labels), np.std(labels, ddof=0)
            if pd.notna(r.move): moves.append(float(r.move))
            if pd.notna(r.abs_move): labels.append(float(r.abs_move))
    train = frame.event_id.isin(train_ids).to_numpy(); floor = max(float(np.median(sig[train]))*.5, 1e-6)
    return mu, np.maximum(sig, floor), floor, feat


def ridge_loo_residuals(y, fitted, x, alpha=1):
    d = np.c_[x, np.ones(len(x))]; p = np.eye(d.shape[1])*alpha; p[-1, -1] = 0
    h = np.einsum("ij,ji->i", d, np.linalg.solve(d.T@d+p, d.T))
    return (np.asarray(y)-fitted)/np.maximum(1-h, 1e-9)


def validate_panel_population(retained, test_ids, panel_ids, duplicates):
    missing = retained-panel_ids
    if duplicates or not test_ids <= panel_ids or len(missing) > .02*len(retained): raise ValueError("panel population mismatch")
    return missing


def block_bootstrap_ci(y, pred, baseline, quarters, tickers, seed=41):
    groups = [np.flatnonzero((quarters == q)&(tickers == t)) for q, t in sorted(set(zip(quarters, tickers)))]
    rng, out = np.random.default_rng(seed), []
    for _ in range(500):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        out.append([np.mean(np.abs(pred[ix]-y[ix])-np.abs(baseline[ix]-y[ix])), np.sqrt(np.mean((pred[ix]-y[ix])**2))-np.sqrt(np.mean((baseline[ix]-y[ix])**2))])
    return np.quantile(out, [.025, .975], axis=0).T.tolist()


def distribution_metrics(pred, residual, scale, y, seed=72):
    rng = np.random.default_rng(seed)
    draws = np.maximum(pred[:, None]+rng.choice(residual, (len(pred), 200), replace=True)*scale[:, None], 0.)
    out = {}
    for c in (.5, .8, .9):
        lo, hi = np.quantile(draws, [(1-c)/2, (1+c)/2], axis=1)
        out[str(int(c*100))] = {"coverage": float(np.mean((y>=lo)&(y<=hi))), "mean_width": float(np.mean(hi-lo))}
    s = np.sort(draws, axis=1); n = s.shape[1]
    out["crps"] = float(np.mean(np.mean(np.abs(draws-y[:, None]), axis=1)-np.sum((2*np.arange(1,n+1)-n-1)*s, axis=1)/(n*n)))
    return out


def _key(t, d): return f"{t}_{pd.Timestamp(d):%Y-%m-%d}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--panel"); ap.add_argument("--report", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.panel: import engine.paths as paths; a.panel = str(paths.PANEL)
    start = time.time()
    def log(msg): print(f"[size-transform] {msg}; elapsed={time.time()-start:.1f}s peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024):.2f}GiB", file=sys.stderr, flush=True)
    log("start report and panel selection"); fd = os.open(a.report, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK); st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_size > 64<<20: os.close(fd); raise ValueError("report must be a regular file under 64 MiB")
    with os.fdopen(fd, encoding="utf-8") as f: raw = f.read((64<<20)+1)
    if len(raw) > 64<<20: raise ValueError("report grew beyond 64 MiB")
    d = json.loads([l for l in raw.splitlines() if l.startswith("{")][-1])
    retained = set(d["frozen_eligibility"]["retained_event_ids"]); paired = d["paired"]["paired_rows"]; del d, raw
    test_ids = {r["event_id"] for r in paired}
    if len(paired) != 751 or len(test_ids) != len(paired) or not test_ids <= retained: raise ValueError("paired population mismatch")
    ticks = {x.rsplit("_", 1)[0] for x in retained}; dataset = ds.dataset(a.panel, format="parquet")
    kept, rows, size = [], 0, 0
    for b in dataset.scanner(columns=["ticker", "date", "move", "abs_move"], filter=ds.field("ticker").isin(list(ticks)), batch_size=256, use_threads=False).to_batches():
        if b.num_rows > 256 or b.nbytes > 64<<20: raise ValueError("decoded panel batch exceeds budget")
        f = b.to_pandas(); f["event_id"] = [_key(t, d) for t, d in zip(f.ticker, f.date)]; f = f[f.event_id.isin(retained)]
        rows += len(f); size += int(f.memory_usage(deep=True).sum())
        if rows > len(retained) or size > 64<<20: raise ValueError("selected panel exceeds row/byte budget")
        if len(f): kept.append(f)
    panel = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["ticker", "date", "move", "abs_move", "event_id"])
    ids = set(panel.event_id); missing = validate_panel_population(retained, test_ids, ids, panel.event_id.duplicated().any())
    log(f"panel rows={len(panel)} retained={len(retained)} missing={len(missing)}")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    train_ids = set(panel.loc[panel.date.dt.year<=2024, "event_id"])-test_ids
    mu, sigma, floor, feat = history_stats(panel, train_ids); panel["mu"], panel["sigma"] = mu, sigma; panel[H] = feat
    tr = panel[panel.event_id.isin(train_ids)&panel.abs_move.notna()].copy(); pmap = {r["event_id"]: r for r in paired}
    te = panel[panel.event_id.isin(test_ids)].copy().set_index("event_id").loc[list(pmap)].reset_index()
    ytr = tr.abs_move.to_numpy(float); y = np.array([pmap[k]["absolute_move_pct"] for k in te.event_id], float)
    base = np.array([pmap[k]["history_mean"] for k in te.event_id], float); inc = np.array([pmap[k]["incumbent"] for k in te.event_id], float)
    q = np.array([pmap[k]["quarter"] for k in te.event_id]); tickers = te.ticker.to_numpy()
    out = {"n_train": len(tr), "n_test": len(te), "missing_panel_events": len(missing), "sigma_floor_pct": floor, "feature_note": "H recomputed from retained prior moves; MO- omitted because legacy market/options cutoffs are unsafe.", "label_note": "Panel abs_move is used for training and causal mu/sigma; paired absolute_move_pct scores test.", "features": {"H": H}, "baselines": {}}
    for name, pred in (("history_mean", base), ("v1.4", inc)):
        mae, rmse = score_metrics(y, pred); out["baselines"][name] = {"mae": mae, "rmse": rmse}
    models = {"ridge": Ridge(alpha=1), "elasticnet": ElasticNet(alpha=.01, l1_ratio=.75, max_iter=10000), "hgb": HistGradientBoostingRegressor(max_leaf_nodes=7, random_state=0)}
    bmae, brmse = score_metrics(y, base); arms, dist, done = {}, {}, 0; log("start fixed H model arms")
    med = tr[H].median().fillna(0); xt, xv = tr[H].fillna(med), te[H].fillna(med)
    for kind in ("raw", "residual", "z", "log_ratio"):
        target = transform(ytr, tr.mu, tr.sigma, kind)
        for mn, est in models.items():
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est); pipe.fit(xt, target)
            pred = inverse(pipe.predict(xv), te.mu, te.sigma, kind); mae, rmse = score_metrics(y, pred); key = f"H/{kind}/{mn}"
            arms[key] = {"mae": mae, "rmse": rmse, "delta_vs_history_mean": {"mae": mae-bmae, "rmse": rmse-brmse}, "delta_ci95": block_bootstrap_ci(y, pred, base, q, tickers)}
            if kind == "raw" and mn == "ridge":
                fit = pipe.predict(xt); x = pipe[:-1].transform(xt); r = ridge_loo_residuals(ytr, fit, x)
                dist["unscaled"] = distribution_metrics(pipe.predict(xv), r, np.ones(len(te)), y)
                dist["sigma_scaled"] = distribution_metrics(pipe.predict(xv), r/tr.sigma.to_numpy(), te.sigma.to_numpy(), y)
            done += 1
            if done%3 == 0: log(f"fit {done}/12; rate={done/(time.time()-start):.2f}/s ETA={(12-done)/max(done/(time.time()-start),1e-9):.1f}s")
    out["arms"], out["raw_ridge_H_distribution"] = arms, dist; dest = Path(a.out); dest.mkdir(parents=True, exist_ok=True)
    (dest/"summary.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n"); log(f"complete train={len(tr)} test={len(te)} output={dest/'summary.json'}")


if __name__ == "__main__": main()
