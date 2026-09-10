#!/usr/bin/env python3
"""EXP-167: menu width (5/7/8 families) on a partial-inclusive universe with
the deviation head, cut0 hygiene gate, and per-secured-dollar ordering."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = ROOT / "experiments/EXP-161_dynamic_short_vol_neural_structure_picker/run.py"
MARGIN163 = ROOT / "experiments/EXP-163_two_headed_level_and_deviation_dyn_sv/margin163.py"
TIER4 = ROOT / "data/features/tier4_forecasts.parquet"
FIRST_TEST_YEAR = 2020
SEEDS = (20260908, 20260909, 20260910, 20260911, 20260912)
MIN_FIT_ROWS = 500
MIN_OFFERED = 2
MENU5 = {
    "N7q2_-1_-1_1": "TWIN-P", "N5q2_-2_1": "TWIN-P5", "N4q0_-1_1": "CND-PS",
    "N3q-2_1": "BFLY-P", "N5q-4_1_1": "BFLY-P5",
}
MENU7 = {**MENU5, "N7q-2_1_-1_1": "NOTCH7", "N7q-2_-1_1_1": "RAMP7"}
MENU8 = {**MENU7, "N5q-2_-1_2": "CTR5"}
MENUS = {"menu5": MENU5, "menu7": MENU7, "menu8": MENU8}
INCUMBENT = "incumbent_resolver"
PRIMARY = "menu7"
ARMS = (INCUMBENT, "menu5", "menu7", "menu8")
BREAKEVEN_COLS = ("breakeven_down_room_forecast", "breakeven_up_room_forecast",
                  "max_profit_pct_spot", "max_profit_over_cost", "max_profit_over_secured")
SHAPE_COLS = tuple(f"shape_pnl_{side}_m{i}" for side in ("down", "up") for i in (1, 2, 3, 4))
TIER4_COLS = ("pred_abs_move_p10", "pred_abs_move_p90", "tier4_pred_abs_move_sd", "pred_abs_move_resid_n",
              "pred_im_t1_d14", "pred_im_t1_d14_p10", "pred_im_t1_d14_p90",
              "pred_runup_abs_move_d14", "pred_runup_abs_move_d14_p10", "pred_runup_abs_move_d14_p90",
              "pred_runup_abs_move_d14_sd", "tier4_forecast_edge")
POLICY = dict(start_equity=200_000.0, cap=0.66, target_share=0.25)
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine.evaluate import evaluate
from experiments import common, lib


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module("exp167_base", SOURCE)
margin163 = load_module("exp167_margin163", MARGIN163)
FULL = base.ARMS["nn_all_categories"]


def log(message: str) -> None:
    print(f"[EXP-167 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def load_data_menu(menu: dict):
    columns = [
        "arm", "event_id", "ticker", "event_date", "entry_date", "exit_date",
        "fill_alpha", "entry_cost", "exit_value", "exit_value_expiry", "exit_arb_ok",
        "spot_entry", "exp_pnl_sim", "exp_pnl_sim_select", "pred_abs_move",
        "pred_abs_move_sd", "width_over_forecast", "half_width_pct_spot",
        "anchor_over_spot", "n_legs", "n_admissible", "dte_entry", "rel_spread",
        "quote_repaired", "wide_market", "legs", "common_universe",
    ]
    raw = pd.read_parquet(base.CANDIDATES, columns=columns)
    raw = raw[raw["arm"].isin(menu)].copy()
    raw["strategy"] = raw["arm"].map(menu)
    for col in ("event_date", "entry_date", "exit_date"):
        raw[col] = pd.to_datetime(raw[col]).dt.normalize()
    raw = base.conditional_exit(raw)
    raw["event_id"] = raw["event_id"].astype(str)
    raw["candidate_id"] = raw["event_id"] + "|" + raw["strategy"]
    raw["entry_cost_pct"] = 100.0 * raw["entry_cost"] / raw["spot_entry"]
    raw["year"] = raw["event_date"].dt.year
    mid = raw[np.isclose(raw["fill_alpha"].astype(float), 0.5)].copy()
    if mid.duplicated(["event_id", "strategy"]).any():
        raise RuntimeError("midpoint candidates are not unique per event and structure")
    offered = mid.groupby("event_id")["strategy"].agg(lambda x: len(set(x)))
    keep_ids = set(offered[offered >= MIN_OFFERED].index)
    mid = mid[mid["event_id"].isin(keep_ids)].copy()
    raw = raw[raw["event_id"].isin(keep_ids)].copy()
    panel_cols = ["ticker", "date", "mcap_usd", *base.CATEGORIES["history"], *base.CATEGORIES["market"]]
    from engine.features import load_panel
    panel = load_panel()[panel_cols].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.drop_duplicates(["ticker", "date"], keep="last")
    mid = mid.merge(panel, left_on=["ticker", "event_date"], right_on=["ticker", "date"], how="left", validate="many_to_one").drop(columns="date")
    for col in ("mcap_usd", *base.CATEGORIES["history"], *base.CATEGORIES["market"]):
        mid[col] = pd.to_numeric(mid[col], errors="coerce")
    mid["mcap_log"] = np.log(mid["mcap_usd"].where(mid["mcap_usd"] > 0))
    for col in ("quote_repaired", "wide_market"):
        mid[col] = mid[col].fillna(False).astype(float)
    for strategy in menu.values():
        mid[f"is_{strategy.lower().replace('-', '_')}"] = (mid["strategy"] == strategy).astype(float)
    mid = mid.sort_values(["entry_date", "event_id", "strategy"]).reset_index(drop=True)
    log(f"Loaded menu of {len(menu)}: {len(mid):,} midpoint candidates on {mid.event_id.nunique():,} events (>= {MIN_OFFERED} offered)")
    return mid, raw


def _exit_cash(entry_legs, S: np.ndarray) -> np.ndarray:
    total = np.zeros_like(S)
    for leg in entry_legs:
        sign = -1.0 if leg["side"] == "sell" else 1.0
        K, q = float(leg["strike"]), float(leg["qty"])
        intr = np.maximum(K - S, 0.0) if leg["right"] == "P" else np.maximum(S - K, 0.0)
        total = total + sign * q * intr
    return total


def _row_schematics(legs_raw, spot, entry_cost, pred_move_pct, pred_move_sd_pct, secured):
    try:
        legs = json.loads(legs_raw) if isinstance(legs_raw, str) else legs_raw
        entry = legs["entry"]
        strikes = sorted({float(leg["strike"]) for leg in entry})
        knots = np.unique(np.concatenate([np.array([0.0, spot]), np.array(strikes)]))
        pnl = _exit_cash(entry, knots) - entry_cost
        pnl_inf = -entry_cost
        crossings = []
        vals = np.concatenate([pnl, [pnl_inf]])
        pts = np.concatenate([knots, [np.inf]])
        for i in range(len(vals) - 1):
            if (vals[i] <= 0.0 < vals[i + 1]) or (vals[i + 1] <= 0.0 < vals[i]):
                if np.isfinite(pts[i + 1]) or np.isfinite(pts[i]):
                    if vals[i + 1] != vals[i]:
                        t = -vals[i] / (vals[i + 1] - vals[i])
                        crossings.append(pts[i] + t * (pts[i + 1] - pts[i]))
        max_profit = float(np.concatenate([pnl, [pnl_inf]]).max())
        down_room = up_room = np.nan
        if pred_move_pct and pred_move_pct > 0:
            move = spot * float(pred_move_pct) / 100.0
            below = [c for c in crossings if c < spot]
            above = [c for c in crossings if c > spot]
            if below:
                down_room = (spot - max(below)) / move
            if above:
                up_room = (min(above) - spot) / move
        sd = float(pred_move_sd_pct) if pred_move_sd_pct and pred_move_sd_pct > 0 else 0.1 * float(pred_move_pct or 0.0)
        p = float(pred_move_pct or 0.0)
        grid_fracs = [max(p - sd, 0.0), p, p + sd, p + 2 * sd]
        shape = {}
        for side, sgn in (("down", -1.0), ("up", 1.0)):
            for i, m in enumerate(grid_fracs, start=1):
                f = min(m / 100.0, 0.9)
                S = spot * (1.0 + sgn * f)
                pnl_at = float(_exit_cash(entry, np.array([S]))[0] - entry_cost)
                shape[f"shape_pnl_{side}_m{i}"] = 100.0 * pnl_at / spot
        return {
            "breakeven_down_room_forecast": down_room,
            "breakeven_up_room_forecast": up_room,
            "max_profit_pct_spot": 100.0 * max_profit / spot,
            "max_profit_over_cost": (max_profit / entry_cost) if entry_cost and entry_cost > 0.05 else np.nan,
            "max_profit_over_secured": (100.0 * max_profit / secured) if secured and secured > 0 else np.nan,
            **shape,
        }
    except Exception:
        return {**{c: np.nan for c in BREAKEVEN_COLS}, **{c: np.nan for c in SHAPE_COLS}}


def build_schematics(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    secured = out["legs"].map(margin163.secured_per_contract).astype(float)
    for col in (*BREAKEVEN_COLS, *SHAPE_COLS):
        out[col] = np.nan
    n = len(out)
    for pos, (legs, spot, cost, pm, psd, sec) in enumerate(zip(
            out["legs"], out["spot_entry"], out["entry_cost"],
            out["pred_abs_move"], out["pred_abs_move_sd"], secured)):
        feats = _row_schematics(legs, float(spot), float(cost), pm, psd, float(sec) if np.isfinite(sec) else 0.0)
        for k, v in feats.items():
            out.at[out.index[pos], k] = v
        if pos and pos % 5000 == 0:
            log(f"Schematic features {pos:,}/{n:,}")
    log(f"Schematic features done for {n:,} candidates")
    return out


def join_tier4(dataset: pd.DataFrame) -> pd.DataFrame:
    t4 = pd.read_parquet(TIER4)
    t4["event_date"] = pd.to_datetime(t4["event_date"]).dt.normalize()
    keep = [c for c in TIER4_COLS if c != "tier4_forecast_edge"]
    raw_keep = ["pred_abs_move_sd" if c == "tier4_pred_abs_move_sd" else c for c in keep]
    t4 = t4[["ticker", "event_date", *raw_keep]].rename(columns={"pred_abs_move_sd": "tier4_pred_abs_move_sd"})
    t4 = t4.drop_duplicates(["ticker", "event_date"], keep="last")
    out = dataset.merge(t4, on=["ticker", "event_date"], how="left", validate="many_to_one")
    out["tier4_forecast_edge"] = out["pred_abs_move"] - out["or_implied"]
    return out


def menu_features(menu: dict) -> tuple:
    extra = tuple(f"is_{s.lower().replace('-', '_')}" for s in menu.values()
                  if f"is_{s.lower().replace('-', '_')}" not in FULL)
    return tuple(FULL) + extra + BREAKEVEN_COLS + SHAPE_COLS + TIER4_COLS


def fit_head(X: np.ndarray, y: np.ndarray):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(y))).fit(y.reshape(-1, 1))
    models, stats = [], []
    for seed in SEEDS:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                         early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=50, random_state=seed),
        ).fit(X, qt.transform(y.reshape(-1, 1)).ravel())
        est = model.named_steps["mlpregressor"]
        models.append(model)
        stats.append({"seed": seed, "n_iter": int(est.n_iter_), "loss": float(est.loss_)})
    return models, stats


def predict(models, X: np.ndarray) -> np.ndarray:
    return np.mean([model.predict(X) for model in models], axis=0)


def within_event_spearman(frame: pd.DataFrame, score_col: str) -> float:
    rhos = []
    for _, g in frame.dropna(subset=[score_col, "pnl"]).groupby("event_id"):
        if len(g) >= 3:
            rhos.append(spearmanr(g[score_col], g["pnl"]).statistic)
    return float(np.nanmean(rhos)) if rhos else float("nan")


def generate_menu(menu_name: str, dataset: pd.DataFrame):
    results_dir = RESULTS / menu_name
    dev_cache, choices_cache, diag_path = (results_dir / "dev_scores.parquet",
                                           results_dir / "choices.parquet",
                                           results_dir / "fold_diagnostics.json")
    if all(p.exists() for p in (dev_cache, choices_cache, diag_path)):
        dev, choices = pd.read_parquet(dev_cache), pd.read_parquet(choices_cache)
        for frame in (dev, choices):
            frame["event_date"] = pd.to_datetime(frame["event_date"])
        return dev, choices, json.loads(diag_path.read_text())
    features = menu_features(MENUS[menu_name])
    event_mean = dataset.groupby("event_id")["pnl"].transform("mean")
    dataset = dataset.assign(dev_target=dataset["pnl"] - event_mean)
    dev_parts, choice_parts, diagnostics = [], [], []
    for year in range(FIRST_TEST_YEAR, int(dataset["year"].max()) + 1):
        train, test = dataset[dataset["year"] < year], dataset[dataset["year"] == year].copy()
        fold = {"year": int(year), "train": int(len(train)), "test": int(len(test))}
        test["score"] = np.nan
        train_ok = np.isfinite(train[list(features) + ["dev_target"]].to_numpy(float)).all(1)
        fit = train.loc[train_ok]
        if len(fit) >= MIN_FIT_ROWS:
            models, seed_stats = fit_head(fit[list(features)].to_numpy(float), fit["dev_target"].to_numpy(float))
            test_ok = np.isfinite(test[list(features)].to_numpy(float)).all(1)
            if test_ok.any():
                test.loc[test_ok, "score"] = predict(models, test.loc[test_ok, list(features)].to_numpy(float))
            fold.update({"fit": int(len(fit)), "scoreable": int(test_ok.sum()),
                         "within_event_spearman": within_event_spearman(test, "score"),
                         "mean_epochs": float(np.mean([s["n_iter"] for s in seed_stats]))})
        piece = test[["candidate_id", "event_id", "ticker", "event_date", "strategy", "pnl"]].rename(columns={"pnl": "realized_pnl"}).reset_index(drop=True)
        piece["score"] = test["score"].to_numpy()
        joined = test.reset_index(drop=True)
        truth = joined.sort_values(["event_id", "pnl", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": "oracle_structure"})
        incumbent = joined.sort_values(["event_id", "exp_pnl_sim", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": "incumbent_structure"})
        head_choice = joined.dropna(subset=["score"]).sort_values(["event_id", "score", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": "head_structure"})
        ev = test[["event_id", "ticker", "event_date"]].drop_duplicates("event_id")
        offered_count = test.groupby("event_id")["strategy"].nunique().rename("offered")
        ev = ev.merge(offered_count, on="event_id").merge(truth, on="event_id", how="left").merge(incumbent, on="event_id", how="left").merge(head_choice, on="event_id", how="left")
        dev_parts.append(piece)
        choice_parts.append(ev)
        diagnostics.append(fold)
        log(f"{menu_name} fold {year}: rho={fold.get('within_event_spearman', float('nan')):+.3f}")
    dev = pd.concat(dev_parts, ignore_index=True).sort_values(["event_date", "candidate_id"])
    choices = pd.concat(choice_parts, ignore_index=True).sort_values(["event_date", "event_id"])
    results_dir.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(dev_cache, index=False)
    choices.to_parquet(choices_cache, index=False)
    write_json(diag_path, diagnostics)
    return dev, choices, diagnostics


def build_book(mid: pd.DataFrame, choices: pd.DataFrame, arm: str):
    if arm == INCUMBENT:
        gate = base.incumbent_gate(mid)
        picked = choices[["event_id", "incumbent_structure"]].merge(gate[["event_id", "traded"]], on="event_id", how="left")
        picked = picked.rename(columns={"incumbent_structure": "picked_structure"})
        priority = None
    else:
        picked = choices[["event_id", "head_structure"]].rename(columns={"head_structure": "picked_structure"})
        priority = "expected_per_secured"
    out = mid.merge(picked.dropna(subset=["picked_structure"]), on="event_id", how="inner")
    traded_col = "traded" if arm == INCUMBENT else None
    if arm == INCUMBENT:
        out = out[(out["strategy"] == out["picked_structure"]) & out["traded"].fillna(False)].copy()
    else:
        hyg = (out["rel_spread"] <= 0.25) & (out["mcap_usd"] >= 10e9)
        out = out[(out["strategy"] == out["picked_structure"]) & hyg.fillna(False)].copy()
    out = out.drop_duplicates(["event_id", "strategy"])
    out["secured_per_contract"] = out["legs"].map(margin163.secured_per_contract)
    out["expected_per_secured"] = out["exp_pnl_sim"] * 100.0 / out["secured_per_contract"]
    return out.sort_values(["entry_date", "event_id"]).reset_index(drop=True), priority


def account_score(book: pd.DataFrame, priority):
    funded = margin163.simulate(book, start_equity=POLICY["start_equity"], cap=POLICY["cap"],
                                target_share=POLICY["target_share"], priority_col=priority)
    placed = funded[funded["funded"]].copy()
    if placed.empty:
        return funded, {"wanted": int(len(funded)), "funded": 0}
    start = POLICY["start_equity"]
    final = float(funded.attrs["final_equity"])
    span = (pd.to_datetime(funded["exit_date"]).max() - pd.to_datetime(funded["entry_date"]).min()).days
    years = span / 365.25
    unfunded = funded[~funded["funded"]]
    return funded, {
        "wanted": int(len(funded)), "funded": int(len(placed)),
        "peak_concurrency": int(funded["concurrency"].max()),
        "final_equity": final, "profit_usd": final - start,
        "cagr": float((final / start) ** (1 / years) - 1) if years > 0 else float("nan"),
        "max_secured_usd": float((funded["secured_before"] + funded["contracts"] * funded["secured_per_contract"]).max()),
        "defined_risk_failures": int((placed["ret"] < -1.0 - 1e-8).sum()),
        "unfunded_wanted": int(len(unfunded)),
        "unfunded_mean_realized_pnl": float(unfunded["pnl"].mean()) if len(unfunded) else 0.0,
        "funded_mean_realized_pnl": float(placed["pnl"].mean()),
    }


def structure_diagnostics(dev: pd.DataFrame, choices: pd.DataFrame, funded_ids: set) -> dict:
    out = {}
    per_cand = dev.set_index(["event_id", "strategy"])["realized_pnl"]
    prec = choices.dropna(subset=["head_structure", "oracle_structure"])
    for strategy in sorted(choices["head_structure"].dropna().unique()):
        offered = int(dev.loc[dev["strategy"] == strategy, "event_id"].nunique())
        picks = choices[choices["head_structure"] == strategy]
        vals, funded_picks = [], 0
        for r in picks.itertuples():
            key = (r.event_id, strategy)
            if key in per_cand.index and r.event_id in funded_ids:
                vals.append(float(per_cand[key]))
                funded_picks += 1
        out[strategy] = {
            "offered_events": offered,
            "pick_events": int(len(picks)),
            "funded_picks": funded_picks,
            "mean_realized_when_picked_funded": float(np.mean(vals)) if vals else float("nan"),
            "pick_precision_vs_oracle": float((prec[prec["head_structure"] == strategy]["oracle_structure"] == strategy).mean()) if len(picks) else float("nan"),
        }
    return out


def report_sections(accounts: dict, structure_diag: dict, choices_by_menu: dict, funded_picks: dict) -> list[dict]:
    rows = []
    for arm in ARMS:
        a = accounts[arm]
        rows.append([arm, f"{a.get('wanted', 0):,}", f"{a.get('funded', 0):,}", f"{a.get('unfunded_wanted', 0):,}",
                     f"{a.get('funded_mean_realized_pnl', 0):+.3f}", f"{a.get('unfunded_mean_realized_pnl', 0):+.3f}",
                     f"{a.get('peak_concurrency', 0):,}", f"${a.get('final_equity', float('nan')):,.0f}", f"{a.get('cagr', float('nan')):.2%}"])
    m7, m5 = accounts.get(PRIMARY, {}), accounts.get("menu5", {})
    base_line = accounts.get(INCUMBENT, {})
    restored = [s for s in ("NOTCH7", "RAMP7") if s in structure_diag]
    menu_mean = m7.get("funded_mean_realized_pnl", float("nan"))
    restored_ok = all(
        structure_diag[s]["mean_realized_when_picked_funded"] == structure_diag[s]["mean_realized_when_picked_funded"]
        and structure_diag[s]["mean_realized_when_picked_funded"] >= menu_mean
        for s in restored
    ) if restored else False
    checks = {
        "menu7_beats_menu5": m7.get("final_equity", -np.inf) > m5.get("final_equity", np.inf),
        "restored_structures_realize_at_least_menu_mean": restored_ok,
        "menu7_beats_incumbent_resolver": m7.get("final_equity", -np.inf) > base_line.get("final_equity", np.inf),
        "no_defined_risk_failure": all(a.get("defined_risk_failures", 1) == 0 for a in accounts.values()),
    }
    struct_rows = []
    for strategy, s in sorted(structure_diag.items()):
        struct_rows.append([strategy, f"{s['offered_events']:,}", f"{s['pick_events']:,}", f"{s['funded_picks']:,}",
                            f"{s['mean_realized_when_picked_funded']:+.3f}", f"{s['pick_precision_vs_oracle']:.1%}"])
    return [
        {"title": "Menu width on a partial-inclusive universe", "body": [
            "Universes are partial-inclusive (events offering at least two of the menu's structures with valid midpoints). Every arm except the incumbent baseline uses the cut0 hygiene gate (rel_spread <= 0.25, mcap >= $10B on the chosen structure) and expected-PnL-per-secured-dollar ordering. The deviation head is retrained per menu on the event-demeaned-over-offered target.",
            "menu7 adds the two seven-strike families EXP-141 dropped (NOTCH7, RAMP7 - old-chooser precision 12.6 and 16.4 percent against a 12.5 percent chance baseline). menu8 also returns the centre-five (CTR5, 12.3 percent).",
        ]},
        {"title": "Account by arm", "note": "$200,000 start, 66% secured cap, 25% headroom target. The incumbent_resolver row is the program baseline rebuilt on the menu5 partial universe.", "columns": ["arm", "wanted", "funded", "unfunded", "funded mean", "unf mean", "peak open", "final equity", "CAGR"], "align": ["---"] + ["---:"] * 8, "rows": rows},
        {"title": "Per-structure diagnostics (menu7)", "note": "offered = events where the structure has a valid midpoint row; picks = head argmax choices; funded = picks the gate and account could trade; precision = share of the structure's picks that were the offered-menu oracle.", "columns": ["structure", "offered", "picks", "funded", "mean realized when funded", "precision vs oracle"], "align": ["---"] + ["---:"] * 5, "rows": struct_rows},
        {"title": "Primary checks", "columns": ["check", "status"], "align": ["---", "---"], "rows": [[name, "PASS" if value else "FAIL"] for name, value in checks.items()]},
    ]


def arm_spec(spec: dict, arm: str) -> dict:
    out = deepcopy(spec)
    if arm != PRIMARY:
        out["primary_spec"]["evaluated_arm"] = arm
        out["grid_cell"] = True
        out["promotion_target"] = None
    return out


def plan(spec: dict) -> None:
    digest, ledger = lib.spec_hash(spec), lib.ledger_read()
    if not (ledger["spec_hash"] == digest).any():
        lib.ledger_append([{"id": spec["id"], "spec_hash": digest, "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                            "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}])
        log("Registered primary specification in LEDGER.csv")


def main() -> None:
    global RESULTS, SEEDS, FIRST_TEST_YEAR
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        RESULTS = HERE / "results_smoke"
        SEEDS = (SEEDS[0],)
        FIRST_TEST_YEAR = 2025
    spec = lib.load_spec(HERE / "spec.yaml")
    if not args.no_ledger and not args.smoke:
        plan(spec)
    data, choices_by_menu, dev_by_menu = {}, {}, {}
    for menu_name, menu in MENUS.items():
        mid, raw = load_data_menu(menu)
        if args.smoke:
            mid = mid[mid["year"] >= FIRST_TEST_YEAR].copy()
        dataset = base.add_causal_analogs(mid)
        dataset = build_schematics(dataset)
        dataset = join_tier4(dataset)
        dev, choices, diagnostics = generate_menu(menu_name, dataset)
        data[menu_name] = (mid, raw)
        dev_by_menu[menu_name], choices_by_menu[menu_name] = dev, choices
        log(f"{menu_name}: {len(choices):,} OOS events, {int(choices['head_structure'].notna().sum()):,} head choices")
    spy = common.load_spy_daily()
    accounts, funded_books, funded_picks = {}, {}, {}
    for arm in ARMS:
        menu_name = "menu5" if arm == INCUMBENT else arm
        mid, raw = data[menu_name]
        book, priority = build_book(mid, choices_by_menu[menu_name], arm)
        funded, account = account_score(book, priority)
        accounts[arm] = account
        funded_books[arm] = base.selected_all_alphas(raw, funded)
        funded_picks[arm] = set(funded.loc[funded["funded"], "event_id"])
        log(f"{arm}: wanted={account.get('wanted', 0):,} funded={account.get('funded', 0):,} final=${account.get('final_equity', float('nan')):,.0f}")
    structure_diag = structure_diagnostics(dev_by_menu["menu7"], choices_by_menu["menu7"], funded_picks["menu7"])
    write_json(RESULTS / "structure_diagnostics_menu7.json", structure_diag)
    evaluations = {}
    for arm in ARMS:
        cell, run_dir = arm_spec(spec, arm), HERE if arm == PRIMARY else HERE / "arms" / arm
        extra = (lambda result: report_sections(accounts, structure_diag, choices_by_menu, funded_picks)) if arm == PRIMARY else [
            {"title": "Arm accounting", "body": [f"Arm: {arm}. Final equity ${accounts[arm].get('final_equity', float('nan')):,.0f}; funded {accounts[arm].get('funded', 0):,}/{accounts[arm].get('wanted', 0):,}."]}
        ]
        result = evaluate(cell, funded_books[arm], gate=None, run_dir=run_dir, spy_daily=spy,
                          input_files=[base.CANDIDATES, TIER4, *[(RESULTS / m / f).resolve() for m in MENUS for f in ("dev_scores.parquet", "choices.parquet")]],
                          extra_sections=extra, write_report=True)
        evaluations[arm] = result.results
        log(f"{arm}: evaluated, mean={result.results['headline'].get('mean', float('nan')):+.3f}")
    write_json(RESULTS / "comparison.json", {"policy": POLICY, "account_metrics": accounts,
                                             "structure_diagnostics_menu7": structure_diag,
                                             "evaluation_headlines": {k: v["headline"] for k, v in evaluations.items()}})
    if not args.no_ledger and not args.smoke:
        lib.record_evaluation(HERE, spec, evaluations[PRIMARY])
    print(f"[EXP-167] report: {HERE / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
