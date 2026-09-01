#!/usr/bin/env python3
"""EXP-117 Stage 4 — retrain the size champion on the validated target.

Like-for-like, registered rule: both arms run walk-forward on IDENTICAL rows
of the ORIGINAL 2,936-ticker oquants panel. The treatment arm differs only in
the target values: events of the Stage-1 sample tickers where oquants was the
adjudicated outlier get the consensus value (mean of the two agreeing
independent-ish sources); every other row keeps the oquants target. Extended-
panel metrics are a separate question and never reported beside these.

    arm A  size_v1_4 architecture on the incumbent oquants target
    arm B  the same architecture on the corrected target

History features (mean/EMA of prior moves, signed_streak) are recomputed from
the corrected moves for the affected tickers, because they are functions of
the target history — reusing them would leak the old target into the new one.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/investing-plan")
sys.path.insert(0, str(ROOT))

from engine.data.features import panel as panel_mod  # noqa: E402
from engine.models.training import size_model  # noqa: E402
from engine.models.training.common import walk_forward  # noqa: E402

HERE = Path(__file__).resolve().parent
TOL = 0.5
FIRST_TEST_YEAR = 2013
#: registered no-degradation margins (arm B vs arm A)
MAX_MAE_WORSENING_PP = 0.01
MAX_R_LOSS = 0.002

report: dict = {"generated_at": pd.Timestamp.now("UTC").isoformat()}


def log(msg: str) -> None:
    print(f"[stage4] {msg}", flush=True)


def consensus_target(stage1: pd.DataFrame) -> pd.DataFrame:
    """Per-event corrected signed move, per the discovered-dependence rule.

    Polygon-covered events (2024-09+): Polygon is the truth anchor (backed by
    yfinance to 0.000 median); oquants deviating >0.5pp takes the Polygon
    value. Pre-Polygon: the registered agreeing-pair consensus over {oquants,
    ORATS, yfinance}; oquants is corrected only when it is the outlier of a
    pair it is not part of (i.e. ORATS+yfinance agree against it). Every other
    event keeps the oquants value.
    """
    out = stage1[["ticker", "date", "oq_move", "move_orats", "move_pg",
                  "move_yf_raw", "verdict"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    oq = out["oq_move"].to_numpy(dtype=float)
    orats = out["move_orats"].to_numpy(dtype=float)
    pg = out["move_pg"].to_numpy(dtype=float)
    yf = out["move_yf_raw"].to_numpy(dtype=float)

    corrected = oq.copy()
    n_pg = n_pair = 0
    for i in range(len(out)):
        if np.isfinite(pg[i]):
            # |value| beyond the model's (0,200) bound is dropped by prepare()
            # in BOTH arms, so correcting it would only desync the row sets
            if abs(oq[i] - pg[i]) > TOL and abs(pg[i]) <= 200.0:
                corrected[i] = pg[i]
                n_pg += 1
            continue
        # pre-Polygon: ORATS+yfinance agree against oquants
        if (np.isfinite(orats[i]) and np.isfinite(yf[i])
                and abs(orats[i] - yf[i]) <= TOL
                and abs(oq[i] - orats[i]) > TOL and abs(oq[i] - yf[i]) > TOL):
            corrected[i] = (orats[i] + yf[i]) / 2.0
            n_pair += 1
    out["corrected_move"] = corrected
    log(f"consensus target: {n_pg} corrected via Polygon, {n_pair} via "
        f"orats+yfinance pair, of {len(out):,} events")
    return out[["ticker", "date", "corrected_move"]]


def build_corrected_moves_dir(corrections: pd.DataFrame, tmp_dir: Path) -> int:
    """Rewrite the sample tickers' moves files with corrected values."""
    from engine import paths

    tmp_dir.mkdir(parents=True, exist_ok=True)
    corr = {(r.ticker, pd.Timestamp(r.date).normalize()): r.corrected_move
            for r in corrections.itertuples()}
    affected = set(corrections["ticker"])
    n_files = 0
    for path in sorted(paths.RAW_OQUANTS_MOVES.glob("moves_*.json")):
        ticker = path.name[len("moves_"):-len(".json")]
        if ticker not in affected:
            continue
        doc = json.loads(path.read_text())
        data = doc.get("data") or {}
        dates = data.get("dates") or []
        moves = list(data.get("realized_moves") or [])
        if not dates or len(moves) != len(dates):
            continue
        changed = 0
        for i, d in enumerate(dates):
            key = (ticker, pd.Timestamp(d).normalize())
            if key in corr and np.isfinite(corr[key]):
                if abs(moves[i] - corr[key]) > 1e-9:
                    moves[i] = float(corr[key])
                    changed += 1
        if changed:
            data["realized_moves"] = moves
            data["abs_realized_moves"] = [abs(m) for m in moves]
            doc["data"] = data
            doc["exp117_corrected"] = changed
            (tmp_dir / path.name).write_text(json.dumps(doc))
            n_files += 1
    log(f"corrected moves files written: {n_files}")
    return n_files


def rebuild_panel_corrected(tmp_moves: Path) -> pd.DataFrame:
    """Tier-3 panel with corrected events block for the sample tickers."""
    from engine.features import load_panel

    base = load_panel()
    log(f"base panel: {len(base):,} rows")
    corrected_events = panel_mod.build_events(moves_dir=tmp_moves)
    tickers = set(corrected_events["ticker"])  # only tickers actually rebuilt
    log(f"rebuilt events for {len(tickers)} tickers: {len(corrected_events):,} rows")

    keep = [c for c in base.columns
            if c not in corrected_events.columns and c not in ("ticker", "k")]
    base_part = base[~base["ticker"].isin(tickers)]
    replaced = base[base["ticker"].isin(tickers)][keep + ["ticker", "k"]].merge(
        corrected_events, on=["ticker", "k"], how="inner",
    )
    panel = pd.concat([base_part, replaced], ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # signed_streak is a function of the move history — recompute it on the
    # corrected tickers with the panel's own recursion.
    signs = np.sign(panel["move"].to_numpy(dtype=float))
    tk_arr = panel["ticker"].to_numpy()
    n = len(panel)
    length_before = np.zeros(n, dtype=int)
    sign_before = np.zeros(n, dtype=int)
    run, prev_sign = 0, 0
    for i in range(n):
        if i == 0 or tk_arr[i] != tk_arr[i - 1]:
            length_before[i], sign_before[i] = 0, 0
            run, prev_sign = 1, signs[i]
            continue
        length_before[i], sign_before[i] = run, prev_sign
        if signs[i] == prev_sign and signs[i] != 0:
            run += 1
        else:
            run, prev_sign = 1, signs[i]
    panel["signed_streak"] = length_before * sign_before
    panel["ema12r_abs"] = panel["ema12_prior_abs_move"].where(
        panel["n_prior"] >= 12, panel["mean_prior_abs_move"])
    return panel


def main() -> None:
    stage1_path = HERE / "results" / "stage1_events.parquet"
    stage1 = pd.read_parquet(stage1_path)
    corrections = consensus_target(stage1)

    tmp_moves = Path("/tmp/exp117/corrected_moves")
    n_files = build_corrected_moves_dir(corrections, tmp_moves)
    report["n_moves_files_corrected"] = n_files
    panel_b = rebuild_panel_corrected(tmp_moves)

    from engine.features import load_panel
    panel_a = load_panel()

    # identical rows: complete on features+target in BOTH arms (same feature
    # list, so this is the same mask — assert it).
    feats = list(size_model.FEATURES)
    prep_a = size_model.prepare(panel_a)
    prep_b = size_model.prepare(panel_b)
    mask_a = np.isfinite(prep_a[feats + ["abs_move"]].to_numpy(dtype=float)).all(axis=1)
    mask_b = np.isfinite(prep_b[feats + ["abs_move"]].to_numpy(dtype=float)).all(axis=1)
    assert int(mask_a.sum()) == int(mask_b.sum()), "row sets diverged"
    prep_a, prep_b = prep_a[mask_a].reset_index(drop=True), prep_b[mask_b].reset_index(drop=True)
    report["n_rows_like_for_like"] = int(len(prep_a))

    target_changed = (prep_a["abs_move"] - prep_b["abs_move"]).abs() > 1e-6
    report["n_rows_target_changed"] = int(target_changed.sum())
    report["target_change_pct"] = round(float(target_changed.mean()) * 100, 3)

    started = time.time()
    log("arm A: incumbent target ...")
    res_a = walk_forward(prep_a, feats, "abs_move", size_model.fit,
                         first_test_year=FIRST_TEST_YEAR)
    log("arm B: corrected target ...")
    res_b = walk_forward(prep_b, feats, "abs_move", size_model.fit,
                         first_test_year=FIRST_TEST_YEAR)
    report["elapsed_s"] = round(time.time() - started, 1)

    arms = {}
    for name, res in (("A_incumbent", res_a), ("B_corrected", res_b)):
        m = res.metrics
        arms[name] = {
            "r": round(m["r"], 5), "mae": round(m["mae"], 5),
            "rmse": round(m["rmse"], 5), "bias": round(m["bias"], 5),
            "n": m["n"], "oos_years": m["oos_years"],
        }
    report["arms"] = arms

    bya = res_a.by_year.set_index("year")
    byb = res_b.by_year.set_index("year")
    join = bya[["mae", "r"]].join(byb[["mae", "r"]], lsuffix="_a", rsuffix="_b")
    join["mae_diff"] = join["mae_b"] - join["mae_a"]
    report["per_year"] = {
        int(y): {k: round(float(v), 5) for k, v in row.items()}
        for y, row in join.iterrows()
    }
    years_improved = int((join["mae_diff"] < 0).sum())
    report["years_mae_improved"] = f"{years_improved}/{len(join)}"

    from scipy.stats import wilcoxon
    if len(join) >= 6:
        w = wilcoxon(join["mae_b"].values, join["mae_a"].values)
        report["wilcoxon_mae_p"] = round(float(w.pvalue), 5)
    # drop-best-year robustness
    worst = join["mae_diff"].idxmin()
    rest = join.drop(index=worst)
    report["drop_best_year_mean_mae_diff"] = round(float(rest["mae_diff"].mean()), 5)

    d = res_b.metrics["mae"] - res_a.metrics["mae"]
    dr = res_b.metrics["r"] - res_a.metrics["r"]
    verdict = d <= MAX_MAE_WORSENING_PP and dr >= -MAX_R_LOSS
    report["no_degradation"] = bool(verdict)
    report["verdict"] = (
        f"MAE {res_a.metrics['mae']:.4f} -> {res_b.metrics['mae']:.4f} "
        f"(delta {d:+.4f}pp, margin {MAX_MAE_WORSENING_PP}); "
        f"r {res_a.metrics['r']:.4f} -> {res_b.metrics['r']:.4f} "
        f"(delta {dr:+.4f}, margin {MAX_R_LOSS}): "
        f"{'NO DEGRADATION' if verdict else 'DEGRADED'}"
    )

    res_a.frame.to_parquet(HERE / "results" / "stage4_arm_a.parquet")
    res_b.frame.to_parquet(HERE / "results" / "stage4_arm_b.parquet")
    (HERE / "results" / "stage4_results.json").write_text(json.dumps(report, indent=1))
    log(report["verdict"])
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
