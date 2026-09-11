#!/usr/bin/env python3
"""EXP-180: fund the DYN-SV book by a model's predicted return per margin,
instead of exp_pnl_sim per margin -- holding the chooser's own picks fixed at
the deployed champion's choices (dyn_sv_chooser_v1_1 / EXP-169).

Single variable: the funding-PRIORITY column margin163.simulate sorts
same-day competitors by. Everything else -- which structure gets picked per
event, the hygiene gate, the capital-allocation walk itself, the funding
policy -- is unchanged from EXP-169's own champion book.

The auxiliary model predicts `ret` directly (not the chooser's event-demeaned,
quantile-transformed ranking target, which has no portable scale) on the
FULL candidate population via the same expanding-annual-fold walk as the
chooser, so every offered candidate in every OOS event (2020-2026) gets an
out-of-fold predicted return -- avoiding the coverage gap a chosen-only
training set would have in its earliest fold.

Run under tools/bounded_run.py; see spec.yaml's resource_notes.
"""
from __future__ import annotations

import gc
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
E169_DIR = ROOT / "experiments/EXP-169_menu7prime_confirmation"
E169_RUN = E169_DIR / "run.py"

FIRST_TEST_YEAR = 2020
SEEDS = (20260908, 20260909, 20260910, 20260911, 20260912)
MIN_FIT_ROWS = 500
PRIMARY_ARM = "menu7p_mcap10"

STARTED = time.monotonic()
sys.path.insert(0, str(ROOT))
from engine.evaluate import evaluate  # noqa: E402
from experiments import common, lib  # noqa: E402


def log(msg: str) -> None:
    print(f"[EXP-180 {time.monotonic() - STARTED:,.0f}s] {msg}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plan(spec: dict) -> None:
    digest, ledger = lib.spec_hash(spec), lib.ledger_read()
    if not (ledger["spec_hash"] == digest).any():
        lib.ledger_append([{
            "id": spec["id"], "spec_hash": digest,
            "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False",
        }])
        log("Registered specification in LEDGER.csv")


# -- the auxiliary return-predicting head, mirroring e169.fit_head/generate --
# but on the RAW ret scale (no event-demeaning, no quantile transform): the
# whole point is a portable magnitude comparable to exp_pnl_sim's own units.

def fit_return_head(X: np.ndarray, y: np.ndarray):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models, stats = [], []
    for seed in SEEDS:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                         early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=50, random_state=seed),
        ).fit(X, y)
        est = model.named_steps["mlpregressor"]
        models.append(model)
        stats.append({"seed": seed, "n_iter": int(est.n_iter_), "loss": float(est.loss_)})
    return models, stats


def predict_return(models, X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def _release_free_pages() -> None:
    """Same trick as engine/score.py::_release_free_pages: glibc keeps freed
    heap in its own arenas rather than returning it, so five MLP fits per
    fold can accumulate RSS that Python itself has already finished with.
    Best-effort; a platform without malloc_trim is slightly fatter, not
    broken."""
    import ctypes
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def generate_return_predictions(dataset: pd.DataFrame, features: tuple) -> tuple[pd.DataFrame, list]:
    cache = RESULTS / "return_head_predictions.parquet"
    diag_path = RESULTS / "return_head_diagnostics.json"
    if cache.exists() and diag_path.exists():
        log("cached return-head predictions found, skipping retrain")
        out = pd.read_parquet(cache)
        out["event_date"] = pd.to_datetime(out["event_date"])
        return out, json.loads(diag_path.read_text())

    # Per-fold checkpoints: a memory-cap kill mid-walk (as happened once
    # during this experiment, at fold 2025 under a too-tight cap) loses only
    # the in-flight fold, not the five before it. Each fold's own MLP fits
    # are also freed before the next one starts, rather than accumulating
    # five folds x five seeds of pipeline objects across the whole walk.
    fold_dir = RESULTS / "fold_cache"
    fold_dir.mkdir(parents=True, exist_ok=True)
    parts, diagnostics = [], []
    for year in range(FIRST_TEST_YEAR, int(dataset["year"].max()) + 1):
        fold_parquet, fold_json = fold_dir / f"{year}.parquet", fold_dir / f"{year}.json"
        if fold_parquet.exists() and fold_json.exists():
            test = pd.read_parquet(fold_parquet)
            test["event_date"] = pd.to_datetime(test["event_date"])
            fold = json.loads(fold_json.read_text())
            log(f"Fold {year}: cached, rho={fold.get('spearman', float('nan')):+.3f} n_test={fold['test']:,}")
            parts.append(test)
            diagnostics.append(fold)
            continue

        train = dataset[dataset["year"] < year]
        test = dataset[dataset["year"] == year].copy()
        fold = {"year": int(year), "train": int(len(train)), "test": int(len(test))}
        test["predicted_ret"] = np.nan
        train_ok = np.isfinite(train[list(features) + ["ret"]].to_numpy(float)).all(1)
        fit = train.loc[train_ok]
        if len(fit) >= MIN_FIT_ROWS:
            models, seed_stats = fit_return_head(
                fit[list(features)].to_numpy(float), fit["ret"].to_numpy(float))
            test_ok = np.isfinite(test[list(features)].to_numpy(float)).all(1)
            if test_ok.any():
                test.loc[test_ok, "predicted_ret"] = predict_return(
                    models, test.loc[test_ok, list(features)].to_numpy(float))
            rho = (spearmanr(test.loc[test_ok, "predicted_ret"], test.loc[test_ok, "ret"]).statistic
                  if test_ok.sum() >= 10 else float("nan"))
            fold.update({"fit": int(len(fit)), "scoreable": int(test_ok.sum()),
                        "spearman": float(rho),
                        "mean_epochs": float(np.mean([s["n_iter"] for s in seed_stats]))})
            del models  # five StandardScaler+MLPRegressor pipelines, freed before the next fold fits five more
        del train, fit
        _release_free_pages()
        result = test[["candidate_id", "event_id", "ticker", "event_date", "strategy",
                       "ret", "pnl", "entry_cost", "predicted_ret"]]
        result.to_parquet(fold_parquet, index=False)
        write_json(fold_json, fold)
        parts.append(result)
        diagnostics.append(fold)
        log(f"Fold {year}: rho={fold.get('spearman', float('nan')):+.3f} n_test={fold['test']:,}")

    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(cache, index=False)
    write_json(diag_path, diagnostics)
    return out, diagnostics


def account_score_with_priority(margin163, book: pd.DataFrame, priority_col: str, policy: dict) -> tuple[pd.DataFrame, dict]:
    """e169.account_score, parameterized on the priority column — e169's own
    version hardcodes 'expected_per_secured', so it can't be reused for the
    candidate order."""
    funded = margin163.simulate(book, start_equity=policy["start_equity"], cap=policy["cap"],
                                target_share=policy["target_share"], priority_col=priority_col)
    placed = funded[funded["funded"]].copy()
    if placed.empty:
        return funded, {"wanted": int(len(funded)), "funded": 0}
    start = policy["start_equity"]
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


def per_structure_wanted_vs_funded(book: pd.DataFrame) -> dict:
    out = {}
    for structure, g in book.groupby("strategy"):
        placed = g[g["funded"]]
        out[structure] = {
            "wanted": int(len(g)), "funded": int(len(placed)),
            "wanted_mean_realized_pnl": float(g["pnl"].mean()) if len(g) else float("nan"),
            "funded_mean_realized_pnl": float(placed["pnl"].mean()) if len(placed) else float("nan"),
        }
    return out


def accuracy_table(dev: pd.DataFrame, champ_choices: pd.DataFrame, diagnostics: list) -> list[dict]:
    chosen = champ_choices.dropna(subset=["head_structure"])[["event_id", "head_structure"]]
    dev_chosen = dev.merge(chosen, on="event_id").query("strategy == head_structure")
    by_year = {int(f["year"]): f for f in diagnostics}
    rows = []
    for year in sorted(by_year):
        g_all = dev[pd.to_datetime(dev["event_date"]).dt.year == year].dropna(subset=["predicted_ret", "ret"])
        g_chosen = dev_chosen[pd.to_datetime(dev_chosen["event_date"]).dt.year == year].dropna(subset=["predicted_ret", "ret"])
        rows.append({
            "year": year,
            "full_population_rho": by_year[year].get("spearman", float("nan")),
            "chosen_only_rho": float(spearmanr(g_chosen["predicted_ret"], g_chosen["ret"]).statistic) if len(g_chosen) >= 10 else float("nan"),
            "n_chosen": int(len(g_chosen)),
        })
    all_chosen = dev_chosen.dropna(subset=["predicted_ret", "ret"])
    rows.append({
        "year": "all",
        "full_population_rho": float(np.nanmean([r["full_population_rho"] for r in rows])),
        "chosen_only_rho": float(spearmanr(all_chosen["predicted_ret"], all_chosen["ret"]).statistic) if len(all_chosen) >= 10 else float("nan"),
        "n_chosen": int(len(all_chosen)),
    })
    return rows


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    plan(spec)

    e169 = load_module("exp180_e169", E169_RUN)
    mid, raw = e169.load_data_menu(e169.MENU)
    log(f"loaded {len(mid):,} candidates")

    dataset = e169.base.add_causal_analogs(mid)
    dataset = e169.build_schematics(dataset)
    dataset = e169.join_tier4(dataset)
    features = e169.menu_features(e169.MENU)
    log(f"dataset built, {len(features)} features")

    dev, diagnostics = generate_return_predictions(dataset, features)
    log(f"return head: {dev['predicted_ret'].notna().sum():,}/{len(dev):,} scored")

    champ_choices = pd.read_parquet(E169_DIR / "results/choices.parquet")
    acc_table = accuracy_table(dev, champ_choices, diagnostics)
    log(f"accuracy table: full-pop all-years rho={acc_table[-1]['full_population_rho']:+.4f}, "
        f"chosen-only all-years rho={acc_table[-1]['chosen_only_rho']:+.4f}")

    per_arm = {}
    for arm in e169.ARMS:
        book = e169.build_book(mid, champ_choices, arm)
        book = book.merge(dev[["event_id", "strategy", "predicted_ret"]], on=["event_id", "strategy"], how="left")
        book["predicted_priority"] = book["predicted_ret"] * book["entry_cost"] * 100.0 / book["secured_per_contract"]

        rank_agreement_pooled = float(spearmanr(
            book["expected_per_secured"], book["predicted_priority"], nan_policy="omit").statistic)

        results = {}
        for name, col in (("incumbent", "expected_per_secured"), ("candidate", "predicted_priority")):
            funded, account = account_score_with_priority(e169.margin163, book.copy(), col, e169.POLICY)
            wf = per_structure_wanted_vs_funded(funded)
            results[name] = {"account": account, "wanted_vs_funded": wf, "funded_frame": funded}
            log(f"{arm}/{name}: wanted={account.get('wanted', 0):,} funded={account.get('funded', 0):,} "
                f"final=${account.get('final_equity', float('nan')):,.0f}")
        per_arm[arm] = {"incumbent": results["incumbent"], "candidate": results["candidate"],
                        "rank_agreement_pooled": rank_agreement_pooled}
        log(f"{arm}: priority rank agreement (pooled Spearman) = {rank_agreement_pooled:+.3f}")

    write_json(RESULTS / "full_comparison.json", {
        "accuracy_table": acc_table,
        "per_arm": {arm: {
            "incumbent": {"account": v["incumbent"]["account"], "wanted_vs_funded": v["incumbent"]["wanted_vs_funded"]},
            "candidate": {"account": v["candidate"]["account"], "wanted_vs_funded": v["candidate"]["wanted_vs_funded"]},
            "rank_agreement_pooled": v["rank_agreement_pooled"],
        } for arm, v in per_arm.items()},
    })

    # -- reports, per arm --------------------------------------------------
    spy = common.load_spy_daily()

    def fmt_pct(x): return f"{x:+.1%}" if np.isfinite(x) else "n/a"
    def fmt_pnl(x): return f"{x:+.3f}" if np.isfinite(x) else "n/a"
    def fmt_usd(x): return f"${x:,.0f}" if np.isfinite(x) else "n/a"

    def sections_for(arm: str):
        v = per_arm[arm]
        ia, ka = v["incumbent"]["account"], v["candidate"]["account"]
        account_rows = [
            ["wanted", f"{ia.get('wanted', 0):,}", f"{ka.get('wanted', 0):,}"],
            ["funded", f"{ia.get('funded', 0):,}", f"{ka.get('funded', 0):,}"],
            ["final equity", fmt_usd(ia.get('final_equity', float('nan'))), fmt_usd(ka.get('final_equity', float('nan')))],
            ["CAGR", f"{ia.get('cagr', float('nan')):.2%}", f"{ka.get('cagr', float('nan')):.2%}"],
            ["peak concurrency", f"{ia.get('peak_concurrency', 0):,}", f"{ka.get('peak_concurrency', 0):,}"],
            ["defined-risk failures", f"{ia.get('defined_risk_failures', 0):,}", f"{ka.get('defined_risk_failures', 0):,}"],
            ["funded mean realized", fmt_pnl(ia.get('funded_mean_realized_pnl', float('nan'))), fmt_pnl(ka.get('funded_mean_realized_pnl', float('nan')))],
            ["unfunded mean realized", fmt_pnl(ia.get('unfunded_mean_realized_pnl', float('nan'))), fmt_pnl(ka.get('unfunded_mean_realized_pnl', float('nan')))],
        ]
        account_section = {
            "title": f"Account — {arm} (same picks; incumbent = exp_pnl_sim/margin order, candidate = predicted_ret/margin order)",
            "columns": ["metric", "incumbent order", "candidate order"], "align": ["---", "---:", "---:"],
            "rows": account_rows,
        }

        wf_rows, contrib_rows = [], []
        total_i = total_k = 0.0
        for s in sorted(set(v["incumbent"]["wanted_vs_funded"]) | set(v["candidate"]["wanted_vs_funded"])):
            iw = v["incumbent"]["wanted_vs_funded"].get(s, {})
            kw = v["candidate"]["wanted_vs_funded"].get(s, {})
            wf_rows.append([
                s, f"{iw.get('wanted', 0):,}/{iw.get('funded', 0):,}", f"{kw.get('wanted', 0):,}/{kw.get('funded', 0):,}",
                fmt_pnl(iw.get("funded_mean_realized_pnl", float("nan"))), fmt_pnl(kw.get("funded_mean_realized_pnl", float("nan"))),
            ])
            ic = iw.get("funded", 0) * iw.get("funded_mean_realized_pnl", 0.0)
            kc = kw.get("funded", 0) * kw.get("funded_mean_realized_pnl", 0.0)
            total_i += ic; total_k += kc
            contrib_rows.append([s, f"{ic:+.1f}", f"{kc:+.1f}", f"{kc - ic:+.1f}"])
        contrib_rows.sort(key=lambda r: float(r[3].replace("+", "")))
        contrib_rows.append(["**total**", f"{total_i:+.1f}", f"{total_k:+.1f}", f"{total_k - total_i:+.1f}"])

        wf_section = {
            "title": f"Per structure — funded count and mean realized PnL, {arm}",
            "columns": ["structure", "incumbent wanted/funded", "candidate wanted/funded",
                       "incumbent funded mean", "candidate funded mean"],
            "align": ["---"] + ["---:"] * 4, "rows": wf_rows,
        }
        contrib_section = {
            "title": f"Aggregate funded-PnL contribution by structure, {arm} "
                    "(funded count x mean realized; same picks and same wanted pool both orders — "
                    "only WHICH members of each structure's pool clear capital differs)",
            "columns": ["structure", "incumbent", "candidate", "delta"],
            "align": ["---", "---:", "---:", "---:"], "rows": contrib_rows,
        }
        return [account_section, wf_section, contrib_section]

    def accuracy_section():
        rows = [[str(r["year"]), fmt_pnl(r["full_population_rho"]), fmt_pnl(r["chosen_only_rho"]), f"{r['n_chosen']:,}"]
               for r in acc_table]
        return {
            "title": "Auxiliary return-head accuracy — full candidate population vs the chosen-only subset that competes for funding",
            "note": "full_population_rho = Spearman(predicted_ret, realized ret) over every offered candidate that year. "
                   "chosen_only_rho = the same, restricted to the champion's own head pick per event — the population "
                   "that actually competes for capital.",
            "columns": ["year", "full population ρ", "chosen-only ρ", "n chosen"],
            "align": ["---"] + ["---:"] * 3, "rows": rows,
        }

    def agreement_section():
        rows = [[arm, f"{per_arm[arm]['rank_agreement_pooled']:+.3f}"] for arm in e169.ARMS]
        return {
            "title": "Priority rank agreement — incumbent (exp_pnl_sim/margin) vs candidate (predicted_ret/margin)",
            "note": "Spearman correlation between the two priority columns, pooled over the arm's wanted book. "
                   "Low agreement means the two orders are funding substantially different events with the same capital; "
                   "the account-level delta should track this number, not the accuracy table alone.",
            "columns": ["arm", "pooled rank agreement (ρ)"], "align": ["---", "---:"], "rows": rows,
        }

    for arm in e169.ARMS:
        candidate_funded_book = e169.base.selected_all_alphas(raw, per_arm[arm]["candidate"]["funded_frame"])
        run_dir = HERE if arm == PRIMARY_ARM else HERE / "arms" / arm
        cell = deepcopy(spec)
        if arm != PRIMARY_ARM:
            cell["primary_spec"] = dict(cell["primary_spec"], evaluated_arm=arm)
            cell["grid_cell"] = True
            cell["promotion_target"] = None
        extra = sections_for(arm) + [accuracy_section(), agreement_section()]
        result = evaluate(
            cell, candidate_funded_book, gate=None, run_dir=run_dir, spy_daily=spy,
            tail_shock=common.abs_move_tail_shock,
            input_files=[e169.base.CANDIDATES, e169.TIER4, E169_DIR / "results/choices.parquet",
                        RESULTS / "return_head_predictions.parquet"],
            extra_sections=extra, write_report=True,
        )
        log(f"{arm}: report written, mean={result.results['headline'].get('mean', float('nan')):+.3f}")
        if arm == PRIMARY_ARM:
            lib.record_evaluation(HERE, spec, result.results, promoted=False)

    log("done")


if __name__ == "__main__":
    main()
