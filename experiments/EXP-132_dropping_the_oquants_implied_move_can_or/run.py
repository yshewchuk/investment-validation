#!/usr/bin/env python3
"""EXP-132 — can ORATS history replace the programme's last oquants feature?

    python3 experiments/EXP-132_dropping_the_oquants_implied_move_can_or/run.py

Every champion carries exactly one oquants-derived input,
``mean_prior_implied_move``. This asks whether an ORATS-derived equivalent
serves as well — not whether it is the same quantity, which EXP-122 already
answered (it is not: oquants is E|move| to 3%, ORATS is 1.55x a model-free
straddle).

Three things this run refuses to do, each because it would flatter the answer:

* score arms on different rows. The replacement covers 96.1% of the panel
  against the incumbent's 99.5%, so scoring each on what it happens to have
  drops the hardest rows from one side. Every arm here runs on the
  INTERSECTION.
* call a difference "noise" without measuring noise. The incumbent is refit at
  a second seed, and that spread is the bar every comparison is read against.
* pool the five champions. A mean across models hides the one that broke.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from engine.features import load_panel  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from engine.models.training import implied_t1 as implied_mod  # noqa: E402
from engine.models.training import iv_crush as crush_mod  # noqa: E402
from engine.models.training import size_model as size_mod  # noqa: E402
from engine.models.training.common import (  # noqa: E402
    SEED, decile_spread, regression_metrics, walk_forward,
)
from experiments import lib  # noqa: E402

RESULTS = HERE / "results"
INCUMBENT = "mean_prior_implied_move"
REPLACEMENT = "mean_prior_or_implied"
EMAS = ("ema4_prior_or_implied", "ema8_prior_or_implied", "ema12_prior_or_implied")
FIRST_TEST_YEAR = 2013


def add_orats_history(panel: pd.DataFrame) -> pd.DataFrame:
    """``mean_prior_or_implied`` and its EMAs — the ORATS-side history block.

    Built exactly as ``history_features`` builds the incumbent: per ticker, over
    events STRICTLY BEFORE the one being scored, with no lookahead and a null
    where there are no priors. The two therefore differ only in which series
    they average, which is what makes the comparison about the SERIES.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    prior = out.groupby("ticker")["or_implied"].shift(1)
    out[REPLACEMENT] = prior.groupby(out["ticker"]).expanding().mean().reset_index(level=0, drop=True)
    for span, name in zip((4, 8, 12), EMAS):
        out[name] = (
            prior.groupby(out["ticker"])
            .apply(lambda s, sp=span: s.ewm(span=sp, adjust=True, min_periods=sp).mean())
            .reset_index(level=0, drop=True)
        )
    return out


def swap(features, *, use, extra=()):
    """The champion's feature list with the incumbent replaced (or dropped)."""
    out = [f for f in features if f != INCUMBENT]
    if use:
        out.append(use)
    return out + list(extra)


def score(frame, features, target, fit, *, seed=SEED, label=""):
    result = walk_forward(frame, features, target, fit,
                          first_test_year=FIRST_TEST_YEAR, seed=seed)
    m = dict(result.metrics)
    m["decile_spread"] = decile_spread(result.frame[target], result.frame["pred"])
    m["label"] = label
    m["by_year"] = {int(r["year"]): float(r["mae"]) for _, r in result.by_year.iterrows()} \
        if not result.by_year.empty else {}
    return m


def compare(base: dict, other: dict) -> dict:
    """Years improved and a sign test — the consistency this programme promotes on."""
    ya, yb = base.get("by_year") or {}, other.get("by_year") or {}
    shared = sorted(set(ya) & set(yb))
    if not shared:
        return {"years": 0}
    wins = sum(1 for y in shared if yb[y] < ya[y])
    diffs = np.array([yb[y] - ya[y] for y in shared])
    out = {"years": len(shared), "years_improved": wins,
           "share_improved": wins / len(shared),
           "mean_delta_mae": float(diffs.mean())}
    try:
        from scipy.stats import wilcoxon
        out["wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if len(shared) >= 6 else None
    except Exception:
        out["wilcoxon_p"] = None
    return out


def attach_history(data: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join the ORATS history columns onto a dataset built outside the panel.

    `gate.build_dataset` and `implied_t1.build_dataset` assemble their own
    frames through `entry_feature_frame`, so a column added to the PANEL never
    reaches them. That is the same trap `pre_iv30` sprang — a change made in
    the obvious place, correct there, and absent from a sibling builder that
    consumes the same events by a different route.

    Joined on (ticker, event_date) because these history features are properties
    of the EVENT, identical for every decision day and every fill alpha
    belonging to it.
    """
    cols = ["ticker", "date", REPLACEMENT, *EMAS]
    have = [c for c in cols if c in panel.columns]
    src = panel[have].rename(columns={"date": "event_date"}).drop_duplicates(
        subset=["ticker", "event_date"])
    out = data.copy()
    out["event_date"] = pd.to_datetime(out["event_date"])
    src["event_date"] = pd.to_datetime(src["event_date"])
    return out.merge(src, on=["ticker", "event_date"], how="left")


def gate_lift(frame, features, target, *, seed=SEED, top_fraction=0.20):
    """What a gate is actually for: the lift its top slice buys over the base.

    MAE is the wrong measure here and reporting it alone would let a feature
    look neutral while changing which trades get taken. A gate does not need
    calibrated levels — it needs the trades it likes to beat the trades it does
    not, so the comparison is mean return of the passed set against the
    ungated base, plus the top-minus-bottom decile spread.
    """
    result = walk_forward(frame, features, target, gate_mod.fit,
                          first_test_year=2020, seed=seed)
    scored = result.frame
    if scored.empty:
        return {"n": 0}
    cut = float(np.quantile(scored["pred"], 1.0 - top_fraction))
    passed = scored[scored["pred"] >= cut]
    base_mean = float(scored[target].mean())
    gated_mean = float(passed[target].mean()) if len(passed) else float("nan")
    return {
        "n": int(len(scored)),
        "mae": result.metrics.get("mae"),
        "r": result.metrics.get("r"),
        "base_mean_ret": base_mean,
        "gated_mean_ret": gated_mean,
        "gate_lift": gated_mean - base_mean,
        "gated_n": int(len(passed)),
        "decile_spread": decile_spread(scored[target], scored["pred"]),
        "by_year": {int(r["year"]): float(r["mae"]) for _, r in result.by_year.iterrows()}
                   if not result.by_year.empty else {},
    }


def gate_jobs(panel):
    """The two gate champions, on their own replayed trades."""
    from engine.data import store

    trades = store.read_table("trades")
    trades = trades[trades["provenance"].astype(str) == "engine.replay"]
    out = []
    for strategy in ("STR-RUNUP", "STR-THRU"):
        rows = trades[trades["strategy"] == strategy]
        if rows.empty:
            continue
        data = attach_history(gate_mod.build_dataset(rows, panel=panel), panel)
        if data.empty:
            continue
        out.append((f"gate_{strategy.lower().replace('-', '_')}", data,
                    gate_mod.TARGET, list(gate_mod.FEATURES)))
    return out


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    RESULTS.mkdir(exist_ok=True)
    started = time.time()
    panel = add_orats_history(load_panel())
    results: dict = {"spec_hash": lib.spec_hash(spec), "models": {}}

    jobs = []
    # size — the one at risk: an equal-weight OLS + MLP blend, and only the
    # linear half is scale-sensitive.
    jobs.append(("size_v1_4", size_mod.prepare(panel), size_mod.TARGET,
                 list(size_mod.FEATURES), size_mod.fit))
    # iv_crush — reads the panel directly, target from Tier 2.
    jobs.append(("iv_crush_v1_gbm", crush_mod.prepare(panel), crush_mod.TARGET,
                 list(crush_mod.FEATURES), crush_mod.fit))

    for name, data, target, features, fit in jobs:
        # The MAIN intersection deliberately excludes the EMAs. They need 12
        # prior events and cover only 66% of the panel, so requiring them here
        # would shrink EVERY arm to the EMA's coverage and answer a different
        # question on a smaller, easier sample.
        need = [INCUMBENT, REPLACEMENT, target, *features]
        have = data.dropna(subset=[c for c in need if c in data.columns])
        rows_full = int(data[data[INCUMBENT].notna()].shape[0])
        block = {"rows_intersection": int(len(have)), "rows_incumbent_only": rows_full,
                 "coverage_cost": rows_full - int(len(have))}
        print(f"\n[EXP-132] {name}: {len(have):,} rows on the intersection "
              f"(incumbent alone reaches {rows_full:,})", flush=True)

        base = score(have, features, target, fit, label="incumbent")
        base2 = score(have, features, target, fit, seed=SEED + 1, label="incumbent_seed2")
        block["incumbent"] = base
        block["incumbent_second_seed"] = base2
        block["seed_noise_mae"] = abs((base2.get("mae") or 0) - (base.get("mae") or 0))

        for arm, feats in (
            ("orats", swap(features, use=REPLACEMENT)),
            ("drop_entirely", swap(features, use=None)),
        ):
            m = score(have, feats, target, fit, label=arm)
            m["vs_incumbent"] = compare(base, m)
            m["delta_mae"] = (m.get("mae") or 0) - (base.get("mae") or 0)
            # Direction matters. `abs(delta) > noise` means DISTINGUISHABLE, not
            # worse — and the first version of this printed "WORSE" over three
            # arms that had improved. MAE is lower-is-better, so a negative
            # delta outside the noise band is a win.
            m["within_seed_noise"] = abs(m["delta_mae"]) <= block["seed_noise_mae"]
            m["verdict"] = ("within noise" if m["within_seed_noise"]
                            else ("BETTER" if m["delta_mae"] < 0 else "WORSE"))
            block[arm] = m
            print(f"   {arm:14} MAE {m.get('mae'):.4f}  Δ {m['delta_mae']:+.4f}  "
                  f"(seed noise {block['seed_noise_mae']:.4f})  "
                  f"years improved {m['vs_incumbent'].get('years_improved')}/"
                  f"{m['vs_incumbent'].get('years')}", flush=True)

        # The EMA arm gets its OWN intersection and its OWN incumbent baseline
        # refit on those same rows, because it cannot be compared to a number
        # computed on 1.5x the sample.
        ema_rows = data.dropna(subset=[c for c in [*need, *EMAS] if c in data.columns])
        if len(ema_rows) > 2000:
            ema_base = score(ema_rows, features, target, fit, label="incumbent_on_ema_rows")
            ema_arm = score(ema_rows, swap(features, use=REPLACEMENT, extra=EMAS),
                            target, fit, label="ema_variant")
            ema_arm["vs_incumbent"] = compare(ema_base, ema_arm)
            ema_arm["delta_mae"] = (ema_arm.get("mae") or 0) - (ema_base.get("mae") or 0)
            block["ema_rows"] = int(len(ema_rows))
            block["incumbent_on_ema_rows"] = ema_base
            block["ema_variant"] = ema_arm
            print(f"   {'ema_variant':14} MAE {ema_arm.get('mae'):.4f}  "
                  f"Δ {ema_arm['delta_mae']:+.4f} vs an incumbent refit on the SAME "
                  f"{len(ema_rows):,} rows", flush=True)
        results["models"][name] = block

    # -- the gates, judged on LIFT rather than on MAE ------------------------
    results["implied_t1"] = {}
    # implied_t1 is the awkward one: its TARGET is an implied move, and one of
    # its inputs is a running mean of a DIFFERENT vendor's implied move. If the
    # substitution interacts with anything rather than being neutral, here.
    try:
        from engine.models.training.train_all import _events_with_session

        events = _events_with_session()
        im = attach_history(
            implied_mod.build_dataset(events, panel=panel, decision_days=(14,)), panel)
        need = [INCUMBENT, REPLACEMENT, implied_mod.TARGET, *implied_mod.FEATURES]
        have = im.dropna(subset=[c for c in need if c in im.columns])
        if len(have) > 2000:
            print(f"\n[EXP-132] opf_implied_t1_gbm: {len(have):,} rows (T-14 only)", flush=True)
            feats = list(implied_mod.FEATURES)
            base = score(have, feats, implied_mod.TARGET, implied_mod.fit, label="incumbent")
            base2 = score(have, feats, implied_mod.TARGET, implied_mod.fit,
                          seed=SEED + 1, label="incumbent_seed2")
            blk = {"rows": int(len(have)), "incumbent": base, "incumbent_second_seed": base2,
                   "seed_noise_mae": abs((base2.get("mae") or 0) - (base.get("mae") or 0))}
            for arm, f2 in (("orats", swap(feats, use=REPLACEMENT)),
                            ("drop_entirely", swap(feats, use=None))):
                m = score(have, f2, implied_mod.TARGET, implied_mod.fit, label=arm)
                m["vs_incumbent"] = compare(base, m)
                m["delta_mae"] = (m.get("mae") or 0) - (base.get("mae") or 0)
                m["within_seed_noise"] = abs(m["delta_mae"]) <= blk["seed_noise_mae"]
                m["verdict"] = ("within noise" if m["within_seed_noise"]
                                else ("BETTER" if m["delta_mae"] < 0 else "WORSE"))
                blk[arm] = m
                print(f"   {arm:14} MAE {m['mae']:.4f}  Δ {m['delta_mae']:+.4f}  "
                      f"(noise {blk['seed_noise_mae']:.4f})  {m['verdict']}", flush=True)
            results["implied_t1"] = blk
    except Exception as exc:
        results["implied_t1"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(f"[EXP-132] implied_t1 arm failed: {type(exc).__name__}: {exc}", flush=True)

    results["gates"] = {}
    for name, data, target, features in gate_jobs(panel):
        need = [INCUMBENT, REPLACEMENT, target, *features]
        have = data.dropna(subset=[c for c in need if c in data.columns])
        if len(have) < 500:
            results["gates"][name] = {"skipped": f"only {len(have)} usable rows"}
            continue
        print(f"\n[EXP-132] {name}: {len(have):,} rows on the intersection", flush=True)
        base = gate_lift(have, features, target)
        base2 = gate_lift(have, features, target, seed=SEED + 1)
        block = {"rows": int(len(have)), "incumbent": base,
                 "incumbent_second_seed": base2,
                 "seed_noise_lift": abs((base2.get("gate_lift") or 0) - (base.get("gate_lift") or 0))}
        for arm, feats in (("orats", swap(features, use=REPLACEMENT)),
                           ("drop_entirely", swap(features, use=None))):
            m = gate_lift(have, feats, target)
            m["delta_lift"] = (m.get("gate_lift") or 0) - (base.get("gate_lift") or 0)
            m["within_seed_noise"] = abs(m["delta_lift"]) <= block["seed_noise_lift"]
            m["verdict"] = ("within noise" if m["within_seed_noise"]
                            else ("BETTER" if m["delta_lift"] > 0 else "WORSE"))
            block[arm] = m
            print(f"   {arm:14} lift {100*m['gate_lift']:+.3f}%  Δ {100*m['delta_lift']:+.3f}pp  "
                  f"(seed noise {100*block['seed_noise_lift']:.3f}pp)  {m['verdict']}", flush=True)
        results["gates"][name] = block

    results["elapsed_s"] = round(time.time() - started, 1)
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=1, default=str))

    print(f"\n{'model':20}{'arm':16}{'MAE':>9}{'delta':>9}{'noise':>9}{'yrs+':>8}{'verdict':>12}")
    for name, b in results["models"].items():
        for arm in ("orats", "drop_entirely", "ema_variant"):
            m = b.get(arm) or {}
            if not m:
                continue
            v = m.get("vs_incumbent") or {}
            print(f"{name:20}{arm:16}{m.get('mae', float('nan')):>9.4f}"
                  f"{m.get('delta_mae', float('nan')):>+9.4f}{b['seed_noise_mae']:>9.4f}"
                  f"{str(v.get('years_improved'))+'/'+str(v.get('years')):>8}"
                  f"{m.get('verdict', '?'):>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
