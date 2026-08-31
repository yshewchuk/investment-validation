#!/usr/bin/env python3
"""EXP-114 — learn the error scale instead of picking buckets.

    python3 experiments/EXP-114_size_v2_0_step_5_learn_the_error_scale_i/run.py

Four uncertainty models over ONE set of point predictions per champion, so the
point estimate is fixed by construction and only the interval differs:

  global                one flat residual pool — as shipped
  by_prediction_decile  EXP-112's winner, hand-picked, carried as the benchmark
  sigma_model           a fitted sigma(x): scale learned from the features
  quantile_regression   residual quantiles predicted directly

The reason for the primary is not elegance. EXP-112's bucketing was chosen by
hand and a diagnostic afterwards showed one of its two dimensions was nearly the
weakest variable available, and that the other only leads for the size model —
the prediction ranks 1 of 15 as an error-scale predictor there, 19 of 42 for
implied_t1, and 11 of 43 for the STR-THRU gate where `dte_entry` dominates. A
fitted sigma(x) discovers the conditioning per model rather than being told it.

Causality is enforced the same way throughout: for test year Y the pool, the
sigma model and the quantile models are fitted only on out-of-sample residuals
from years before Y.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.models.training.common import walk_forward  # noqa: E402
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent
DECILES = 10
MIN_POOL = 250
#: sigma(x) must not be trusted below this; a variance model on a thin fold is
#: noise dressed as precision.
MIN_SIGMA_FIT = 2_000


def _sigma_fit(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=200, random_state=0
    ).fit(X, y)


def _sigma_oof(X, y, folds: int = 5):
    """Out-of-fold sigma-hat for the STANDARDISATION pool.

    `sigma_model` fits sigma on the training fold and then predicts back onto
    that same fold. A GBM tracks its own training rows more closely than unseen
    ones, so those sigma_p are too good, `res_past / sigma_p` comes out too
    narrow, and the intervals inherit that narrowness at test time — while the
    test-fold sigma_n is honest. The arm was biased toward undercoverage by
    construction, which is what its cov50 0.54 / cov80 0.74 shape on the
    implied model looks like: too fat in the middle, too thin in the tails.

    Cross-fitting removes that asymmetry — every standardising sigma is
    predicted by a model that did not see the row. The folds shuffle across
    years INSIDE the training window, which is not a temporal split; it does
    not need to be, because the window already contains only years before the
    test year, so nothing from the future reaches the pool either way.
    """
    from sklearn.model_selection import KFold

    oof = np.empty(len(X), dtype=float)
    for train_idx, held_idx in KFold(n_splits=folds, shuffle=True,
                                     random_state=0).split(X):
        oof[held_idx] = _sigma_fit(X[train_idx], y[train_idx]).predict(X[held_idx])
    return oof


def _quantile_fit(X, y, q):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, learning_rate=0.06, max_iter=200, random_state=0
    ).fit(X, y)


#: CRPS is quadratic in the sample size, so the empirical pool is subsampled.
CRPS_SAMPLE = 300


def _prepare_pool(pool: np.ndarray) -> tuple:
    """Everything about a pool that does not depend on the residual read from it.

    Sorted values for the PIT, the four interval quantiles, and the CRPS
    subsample with its pairwise term. All three were being recomputed per event
    against pools of ~88k residuals; they are properties of the pool alone.
    """
    ordered = np.sort(pool)
    sub = (pool if len(pool) <= CRPS_SAMPLE
           else np.random.default_rng(0).choice(pool, CRPS_SAMPLE, replace=False))
    sub = np.sort(sub)
    spread = 0.5 * float(np.abs(sub[:, None] - sub[None, :]).mean())
    return (np.quantile(ordered, [0.10, 0.90, 0.25, 0.75]), ordered, sub, spread)


def evaluate_arm(base: pd.DataFrame, features, arm: str) -> pd.DataFrame:
    """Walk the years forward; everything an arm needs is fitted on earlier years."""
    rows = []
    for year in sorted(base["year"].unique()):
        past, now = base[base["year"] < year], base[base["year"] == year]
        if len(past) < MIN_POOL or now.empty:
            continue
        res_past = (past["actual"] - past["pred"]).to_numpy(dtype=float)
        Xp = past[features].to_numpy(dtype=float)
        Xn = now[features].to_numpy(dtype=float)
        pred_n = now["pred"].to_numpy(dtype=float)
        act_n = now["actual"].to_numpy(dtype=float)

        if arm == "quantile_regression":
            if len(past) < MIN_SIGMA_FIT:
                continue
            lo = _quantile_fit(Xp, res_past, 0.10).predict(Xn)
            hi = _quantile_fit(Xp, res_past, 0.90).predict(Xn)
            lo50 = _quantile_fit(Xp, res_past, 0.25).predict(Xn)
            hi50 = _quantile_fit(Xp, res_past, 0.75).predict(Xn)
            r = act_n - pred_n
            for i in range(len(now)):
                rows.append({"year": int(year), "pred": float(pred_n[i]),
                             "in80": bool(lo[i] <= r[i] <= hi[i]),
                             "in50": bool(lo50[i] <= r[i] <= hi50[i]),
                             "pit": np.nan, "crps": np.nan,
                             "width80": float(hi[i] - lo[i])})
            continue

        # Every arm below reduces to (pool, scale) per event, and the scale is 1
        # for all but sigma_model. Keeping the scale separate is what makes the
        # memory bounded: see the note in the sigma_model branch.
        scales = np.ones(len(now))
        if arm == "global":
            pools = [res_past] * len(now)
        elif arm == "by_prediction_decile":
            edges = np.unique(np.quantile(past["pred"], np.linspace(0, 1, DECILES + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            pb = pd.cut(past["pred"], edges, labels=False, include_lowest=True)
            nb = pd.cut(now["pred"], edges, labels=False, include_lowest=True)
            by = {b: res_past[(pb == b).to_numpy()] for b in pb.dropna().unique()}
            pools = [by.get(b) if by.get(b) is not None and len(by.get(b, [])) >= MIN_POOL
                     else res_past for b in nb]
        elif arm in ("sigma_model", "sigma_model_oof"):
            if len(past) < MIN_SIGMA_FIT:
                continue
            model = _sigma_fit(Xp, np.abs(res_past))
            # The ONLY difference between the two sigma arms: how the pool that
            # gets standardised is scaled. sigma_n is fitted on all of `past`
            # and predicted on unseen `now` in both.
            raw_p = (_sigma_oof(Xp, np.abs(res_past)) if arm == "sigma_model_oof"
                     else model.predict(Xp))
            sig_p = np.clip(raw_p, 1e-6, None)
            sig_n = np.clip(model.predict(Xn), 1e-6, None)
            z = res_past / sig_p              # standardised residuals
            # ONE standardised pool, scaled per event at read time rather than
            # materialised. Building `z * s` for each event held a full-length
            # array per test row -- ~8k x 88k floats, 5.6 GB, and the OOM killer
            # took it twice. The transform is exact, not an approximation: s is
            # positive so scaling is monotone, quantiles and widths scale by s,
            # and both the PIT and the CRPS of `z*s` against a residual r equal
            # those of `z` against `r/s` (CRPS is positively homogeneous).
            pools = [z] * len(now)
            scales = sig_n
        else:
            raise ValueError(arm)

        # The pool is one repeated object for every arm but the decile one, so
        # everything pool-shaped is memoised by identity rather than recomputed
        # per event. Without this each row rescanned ~88k residuals for its PIT
        # and redrew its own CRPS subsample.
        cache: dict[int, tuple] = {}

        for i in range(len(now)):
            pool = pools[i]
            scale = float(scales[i])
            r = (act_n[i] - pred_n[i]) / scale
            prepared = cache.get(id(pool))
            if prepared is None:
                prepared = cache[id(pool)] = _prepare_pool(pool)
            (lo80, hi80, lo50, hi50), ordered, sub, spread = prepared
            rows.append({"year": int(year), "pred": float(pred_n[i]),
                         "in80": bool(lo80 <= r <= hi80), "in50": bool(lo50 <= r <= hi50),
                         "pit": float(np.searchsorted(ordered, r, side="right") / len(ordered)),
                         "crps": scale * (float(np.abs(sub - r).mean()) - spread),
                         "width80": float((hi80 - lo80) * scale)})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    from scipy import stats as st

    # The width PERCENTILES, not just the mean. The whole point of conditioning
    # is that the interval stops being the same for every event, and a mean
    # width cannot show that: a flat pool and a well-conditioned one can share
    # it exactly. p10/p90 is what separates "+/- 5 for everything" from "+/- 2
    # for AAPL and +/- 9 for the illiquid name with no quote".
    w = df["width80"]
    out = {"n": int(len(df)),
           "coverage_80": float(df["in80"].mean()),
           "coverage_50": float(df["in50"].mean()),
           "coverage_80_error": float(abs(df["in80"].mean() - 0.80)),
           "mean_width80": float(w.mean()),
           "width80_p10": float(w.quantile(0.10)),
           "width80_p50": float(w.quantile(0.50)),
           "width80_p90": float(w.quantile(0.90)),
           "width80_spread": float(w.quantile(0.90) - w.quantile(0.10))}
    pit = df["pit"].dropna().to_numpy(dtype=float)
    out["pit_ks"] = float(st.kstest(pit, "uniform").statistic) if len(pit) else None
    crps = df["crps"].dropna()
    out["crps"] = float(crps.mean()) if len(crps) else None
    return out


def base_for_size():
    from engine.models.training import size_model
    from experiments import size_lab

    F = list(size_model.FEATURES)
    panel = size_lab.prepare_panel()
    num = panel[F + ["abs_move"]].apply(pd.to_numeric, errors="coerce")
    panel = panel[np.isfinite(num.to_numpy(dtype=float)).all(axis=1)]
    wf = walk_forward(panel, F, "abs_move", size_model.fit, first_test_year=2013)
    b = wf.frame[["ticker", "date", "pred", "abs_move"]].merge(
        panel[["ticker", "date"] + F], on=["ticker", "date"])
    b = b.rename(columns={"abs_move": "actual"})
    b["year"] = pd.to_datetime(b["date"]).dt.year
    return b, F


def base_for_implied(sample_events: int = 12_000):
    from engine.dashboard.model_evidence import _daily_subset
    from engine.features import add_absolute_features, add_quote_indicators, load_panel
    from engine.models.training import implied_t1
    from engine.models.training.train_all import _events_with_session

    ev = _events_with_session().sample(sample_events, random_state=7)
    panel = add_quote_indicators(add_absolute_features(load_panel()))
    daily = _daily_subset(ev["ticker"].unique(),
                          years=sorted(pd.to_datetime(ev["event_date"]).dt.year.unique().tolist()))
    ds = implied_t1.build_dataset(ev, panel=panel, daily=daily)
    F = list(implied_t1.FEATURES)
    num = ds[F + [implied_t1.TARGET]].apply(pd.to_numeric, errors="coerce")
    ds = ds[np.isfinite(num.to_numpy(dtype=float)).all(axis=1)]
    wf = walk_forward(ds, F, implied_t1.TARGET, implied_t1.fit, first_test_year=2015)
    # The implied_t1 dataset is keyed on `event_date` and already carries the
    # `year` walk_forward split on; the size panel uses `date`. Normalising here
    # rather than assuming, so both models reach `evaluate_arm` in one shape.
    b = wf.frame.rename(columns={implied_t1.TARGET: "actual", "event_date": "date"})
    b["year"] = b["year"].astype(int)
    return b[["ticker", "date", "pred", "actual", "year"] + F], F


#: The first four are pre-registered in spec.yaml. `sigma_model_oof` is NOT —
#: it was added after the pre-registered arms had run, to separate "sigma(x) is
#: the wrong idea" from "sigma(x) was standardised in-sample". spec.yaml is left
#: untouched so its hash still fingerprints what was registered in advance, and
#: the report carries the deviation explicitly.
PREREGISTERED_ARMS = ("global", "by_prediction_decile", "sigma_model",
                      "quantile_regression")
POST_HOC_ARMS = ("sigma_model_oof",)
ARMS = PREREGISTERED_ARMS + POST_HOC_ARMS


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    out = {
        "spec_hash": lib.spec_hash(spec),
        "preregistered_arms": list(PREREGISTERED_ARMS),
        "post_hoc_arms": {
            "sigma_model_oof": (
                "Added after the pre-registered arms had run. sigma_model "
                "standardises its residual pool with in-sample sigma-hat, which "
                "biases it toward undercoverage; this arm is identical except "
                "that the standardising sigma-hat is cross-fitted. Reported "
                "alongside the primary, not in place of it."
            )
        },
        "models": {},
    }

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    for label, builder in (("size_v1_4", base_for_size),
                           ("opf_implied_t1_gbm", base_for_implied)):
        print(f"\n[EXP-114] {label}: building the shared point predictions", flush=True)
        # The point predictions are a walk-forward refit costing minutes, and
        # they are identical across all four arms by construction. Cached, so a
        # crash in one arm does not re-pay for the estimate none of them change.
        cache = results_dir / f"base_{label}.parquet"
        if cache.exists():
            base = pd.read_parquet(cache)
            features = [c for c in base.columns
                        if c not in ("ticker", "date", "pred", "actual", "year")]
            print(f"[EXP-114] {label}: reusing cached point predictions", flush=True)
        else:
            base, features = builder()
            base.to_parquet(cache, index=False)
        print(f"[EXP-114] {label}: {len(base):,} OOS rows, {len(features)} features", flush=True)
        model_out, scored = {}, {}
        for arm in ARMS:
            df = evaluate_arm(base, features, arm)
            if df.empty:
                model_out[arm] = {"available": False}
                continue
            scored[arm] = df
            # Persisted so the report can be rewritten, or a figure redrawn,
            # without re-paying for the arms.
            df.to_parquet(results_dir / f"scored_{label}_{arm}.parquet", index=False)
            model_out[arm] = summarise(df)
            s = model_out[arm]
            print(f"  {arm:22s} cov80 {s['coverage_80']:.4f} (err {s['coverage_80_error']:.4f}) "
                  f"cov50 {s['coverage_50']:.4f} "
                  f"PIT-KS {s['pit_ks'] if s['pit_ks'] is None else round(s['pit_ks'],4)} "
                  f"CRPS {s['crps'] if s['crps'] is None else round(s['crps'],4)}", flush=True)
        err = (base["pred"] - base["actual"]).abs()
        model_out["_point_estimate"] = {"mae": float(err.mean()),
                                        "r": float(np.corrcoef(base["pred"], base["actual"])[0, 1])}
        out["models"][label] = model_out
        write_figures(label, scored, model_out, HERE / "figures" / label)
        # Flushed per model rather than once at the end: the second model costs
        # as long as the first, and a crash there used to discard both.
        (results_dir / "metrics.json").write_text(json.dumps(out, indent=1, default=str))
        del base, scored

    return 0


def write_figures(label, scored, summary, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colours = {"global": "#8b949e", "by_prediction_decile": "#d29922",
               "sigma_model": "#3fb950", "quantile_regression": "#58a6ff",
               "sigma_model_oof": "#bc8cff"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for arm, df in scored.items():
        d = df.assign(b=pd.qcut(df["pred"], DECILES, labels=False, duplicates="drop"))
        g = d.groupby("b").agg(pred=("pred", "mean"), cov=("in80", "mean"), w=("width80", "mean"))
        ax1.plot(g["pred"], g["cov"], "o-", color=colours[arm], label=arm)
        ax2.plot(g["pred"], g["w"], "o-", color=colours[arm], label=arm)
    ax1.axhline(0.80, ls="--", color="#d0342c", label="target 0.80")
    ax1.set_xlabel("prediction"); ax1.set_ylabel("share inside the 80% interval")
    ax1.set_title(f"{label} — coverage by prediction decile"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.set_xlabel("prediction"); ax2.set_ylabel("80% interval width")
    ax2.set_title("interval width — a flat line means unconditional"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "coverage.png", dpi=110); plt.close(fig)

    have = [(a, d) for a, d in scored.items() if d["pit"].notna().any()]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(4.2*len(have), 4), sharey=True)
        for ax, (arm, df) in zip(np.atleast_1d(axes), have):
            ax.hist(df["pit"].dropna(), bins=20, range=(0,1), color=colours[arm], alpha=0.85)
            ax.axhline(df["pit"].notna().sum()/20, ls="--", color="#d0342c")
            ax.set_title(f"{arm}\nKS {summary[arm]['pit_ks']:.4f}", fontsize=10)
            ax.set_xlabel("PIT"); ax.grid(alpha=0.3)
        fig.suptitle(f"{label} — PIT, flat means calibrated", y=1.02)
        fig.tight_layout(); fig.savefig(out_dir / "pit.png", dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
