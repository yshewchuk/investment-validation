#!/usr/bin/env python3
"""EXP-138 — does a better-shaped move distribution pick better structures?

    python3 experiments/EXP-138_a_skewed_move_weibull_scaled_simulation/run.py

The registered primary is not a return. It is the chooser's **top-1 hit rate**
on the common universe: of the eight families priced on an event, how often is
the one with the highest simulated expected P&L the one that realized best.
That is a direct test of the simulator and is not confounded by funding, by the
gate, or by which families happen to be available.

Under the additive-and-clipped draw EXP-137 measured 24.7% against a 12.5%
chance baseline. If the clip's spike at zero was biasing the ranking — and it
sits exactly where centre-peaked families pay most and twin-peaked families pay
least — then reshaping the marginal should move that number and should pull the
centre-peaked share of the book down with it.

Four arms, differing in one thing only. ``ratio`` is the benchmark that matters:
it resamples the pool's own realized/predicted ratios and assumes nothing, so a
fitted curve that cannot beat it has bought an assumption and nothing else.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E133 = ROOT / "experiments" / "EXP-133_every_symmetric_put_structure_the_ladder"
E134 = ROOT / "experiments" / "EXP-134_priced_right_funded_and_held_structure_s"
for p in (ROOT, E133, E134, HERE):
    sys.path.insert(0, str(p))

from engine.evaluate import evaluate, trade_stats  # noqa: E402
from experiments import common as ecommon, lib  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


margin = _load("e138_margin", E134 / "margin.py")
e138 = _load("e138_build", HERE / "build.py")
e133run = _load("e138_e133run", E133 / "run.py")
e134run = _load("e138_e134run", E134 / "run.py")
import family as fammod  # noqa: E402

LABEL = {f.key: f for f in fammod.FAMILIES}
FAMILIES = list(e138.FAMILIES)
MODELS = list(e138.MODELS)
PRIMARY, REFERENCE, BENCHMARK = "weibull", "additive", "ratio"
RESULTS = HERE / "results"
MID = 0.5


def prepared(model: str, tallies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(all alphas, mid rows)`` for ONE move model.

    Loaded a shard at a time and never for more than one model at once. The
    first version of this build held every model's rows in memory and was
    killed by the OOM killer; nothing downstream needs them together.
    """
    raw = e133run.attach(e138.load_model(model), tallies)
    rows = e134run.apply_exit(raw, "conditional")
    return rows, rows[np.isclose(rows["fill_alpha"].astype(float), MID)]


def hit_rate(mid: pd.DataFrame) -> dict:
    """The primary: was the predicted-best family the realized-best?

    Restricted to the common universe, where all eight families are priced
    under every move model, so "best of eight" is the same question on every
    event and comparable across arms.
    """
    cu = mid[mid["common_universe"].fillna(False).astype(bool)]
    full = cu.groupby("event_id")["arm"].nunique() == len(FAMILIES)
    cu = cu[cu["event_id"].isin(full[full].index)]
    if cu.empty:
        return {}
    pred = cu.pivot(index="event_id", columns="arm", values="exp_pnl_sim_select")[FAMILIES]
    real = cu.pivot(index="event_id", columns="arm", values="ret")[FAMILIES]
    chosen = pred.values.argmax(axis=1)
    truth = real.values.argmax(axis=1)
    rank = (-real.values).argsort(axis=1).argsort(axis=1)[np.arange(len(chosen)), chosen] + 1
    rho = np.array([stats.spearmanr(pred.values[i], real.values[i]).statistic
                    for i in range(len(pred))])
    rho = rho[np.isfinite(rho)]
    picked = real.values[np.arange(len(chosen)), chosen]
    oracle = real.values.max(axis=1)
    rnd = real.values.mean(axis=1)
    centre = np.array([not LABEL[FAMILIES[i]].twin_peaked for i in chosen])
    return {
        "events": int(len(pred)),
        "top1": float((chosen == truth).mean()),
        "top3": float((rank <= 3).mean()),
        "mean_rank": float(rank.mean()),
        "spearman": float(rho.mean()),
        "spearman_pos": float((rho > 0).mean()),
        "ret_chooser": float(picked.mean()),
        "ret_oracle": float(oracle.mean()),
        "ret_random": float(rnd.mean()),
        "spread_captured": float((picked.mean() - rnd.mean())
                                 / (oracle.mean() - rnd.mean())),
        "centre_share": float(centre.mean()),
        "sim_minus_real": float((cu["exp_pnl_sim"] - cu["ret"]).mean()),
    }


def family_stats(mid: pd.DataFrame, fam: str, *, gated: bool) -> dict:
    rows = mid[mid["arm"] == fam]
    rows = rows[rows["common_universe"].fillna(False).astype(bool)]
    if rows.empty:
        return {}
    if gated:
        decided = e133run.apply_gate(rows)
        rows = rows[rows["event_id"].isin(set(decided.loc[decided["traded"], "event_id"]))]
    if rows.empty:
        return {}
    cost = pd.to_numeric(rows["entry_cost"], errors="coerce")
    pnl = pd.to_numeric(rows["pnl"], errors="coerce")
    per_year = rows.groupby(pd.to_datetime(rows["event_date"]).dt.year)["ret"].mean()
    return {
        "n": int(len(rows)), "mean": float(rows["ret"].mean()),
        "return_on_capital": float(pnl.sum() / cost.sum()) if cost.sum() else float("nan"),
        "sharpe_trade": float(trade_stats(rows["ret"], rows["event_date"])["sharpe_trade"]),
        "years_positive": int((per_year > 0).sum()), "years": int(per_year.size),
        "worse_than_debit": int((pnl < -cost - 1e-9).sum()),
    }


def best_of_family(mid: pd.DataFrame) -> pd.DataFrame:
    """One row per event: the family this move model predicted would do best."""
    cu = mid[mid["common_universe"].fillna(False).astype(bool)]
    if cu.empty:
        return cu
    idx = cu.groupby("event_id")["exp_pnl_sim_select"].idxmax()
    return cu.loc[idx]


def main(record: bool = True) -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    built = e138.build_all()
    tallies = built["tallies"]

    hits, fams, books, keep_ids = {}, {}, {}, {}
    for model in MODELS:
        _, mid = prepared(model, tallies)
        hits[model] = hit_rate(mid)
        for fam in FAMILIES:
            for g in (False, True):
                s = family_stats(mid, fam, gated=g)
                if s:
                    fams[f"{model}|{fam}|{'gated' if g else 'ungated'}"] = s
        books[model] = best_of_family(mid)
        keep_ids[model] = set(zip(books[model]["event_id"], books[model]["arm"]))
        h = hits[model]
        if h:
            print(f"[EXP-138] {model:9s} top-1 {100*h['top1']:5.1f}%  top-3 "
                  f"{100*h['top3']:5.1f}%  rank {h['mean_rank']:.2f}  rho "
                  f"{h['spearman']:+.3f}  centre {100*h['centre_share']:5.1f}%  "
                  f"chooser {100*h['ret_chooser']:+6.2f}%  gap "
                  f"{100*h['sim_minus_real']:+6.1f}pp", flush=True)

    (RESULTS / "move_model_comparison.json").write_text(
        json.dumps({"hit_rates": hits, "families": fams,
                    "meta": built["meta"]}, indent=1, default=str))

    p, r, b = hits.get(PRIMARY, {}), hits.get(REFERENCE, {}), hits.get(BENCHMARK, {})
    pb, rb = books.get(PRIMARY), books.get(REFERENCE)

    def roc(bk):
        if bk is None or bk.empty:
            return float("nan")
        c = pd.to_numeric(bk["entry_cost"], errors="coerce")
        return float(pd.to_numeric(bk["pnl"], errors="coerce").sum() / c.sum())

    acceptance = {
        "better_chooser": bool(p.get("top1", -1) > r.get("top1", -1)),
        "not_at_a_cost": bool(roc(pb) >= roc(rb)),
        "beats_the_empirical_benchmark": bool(p.get("top1", -1) >= b.get("top1", -1)),
        "defined_risk_holds": bool(pb is not None and not pb.empty and
                                   ((pd.to_numeric(pb["pnl"], errors="coerce")
                                     < -pd.to_numeric(pb["entry_cost"], errors="coerce")
                                     - 1e-9).sum() == 0)),
    }
    (RESULTS / "acceptance.json").write_text(json.dumps(acceptance, indent=1))
    print(f"\n[EXP-138] weibull top-1 {100*p.get('top1', float('nan')):.1f}% vs "
          f"additive {100*r.get('top1', float('nan')):.1f}% vs empirical ratio "
          f"{100*b.get('top1', float('nan')):.1f}%", flush=True)
    print(f"[EXP-138] best-of-family return on capital: weibull "
          f"{100*roc(pb):+.1f}% vs additive {100*roc(rb):+.1f}%", flush=True)
    print(f"[EXP-138] acceptance: {acceptance}", flush=True)

    # Only the primary and the reference get a full evaluation. Thirty-two
    # Monte Carlo passes would add twenty minutes of reports nobody reads; the
    # comparison this experiment turns on is the table above.
    spy = ecommon.load_spy_daily()
    already = set(lib.ledger_read().query("stage == 'ran'")["spec_hash"])
    for model in (PRIMARY, REFERENCE):
        bk = books.get(model)
        if bk is None or bk.empty:
            continue
        allr, _ = prepared(model, tallies)
        keys = keep_ids[model]
        rows = allr[[(e, a) in keys for e, a in zip(allr["event_id"], allr["arm"])]]
        del allr
        if rows.empty:
            continue
        is_primary = model == PRIMARY
        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["move_model"] = f"grid cell: {model}"
            cell["grid_cell"] = True
        run_dir = HERE if is_primary else HERE / "arms" / model
        for sub in ("results", "figures"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        result = evaluate(
            cell, rows, gate=None, run_dir=run_dir, repricer=None,
            tail_shock=ecommon.abs_move_tail_shock, spy_daily=spy,
            input_files=[RESULTS / "candidates.parquet"],
            extra_sections=lambda rr, k=model: sections(hits, fams, k),
            write_report=True)
        if record and lib.spec_hash(cell) not in already:
            lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-138] {model}: report {result.report_path}", flush=True)


def sections(hits, fams, cell):
    hit_rows = [[
        f"**{m}**" if m == cell else m,
        f"{hits[m]['events']:,}", f"{100*hits[m]['top1']:.1f}%",
        f"{100*hits[m]['top3']:.1f}%", f"{hits[m]['mean_rank']:.2f}",
        f"{hits[m]['spearman']:+.3f}", f"{100*hits[m]['spearman_pos']:.1f}%",
        f"{100*hits[m]['centre_share']:.1f}%",
        f"{100*hits[m]['ret_chooser']:+.2f}%",
        f"{100*hits[m]['spread_captured']:.1f}%",
        f"{100*hits[m]['sim_minus_real']:+.1f}pp",
    ] for m in MODELS if hits.get(m)]

    fam_rows = []
    for fam in FAMILIES:
        row = [fam, "twin" if LABEL[fam].twin_peaked else "centre"]
        for m in MODELS:
            s = fams.get(f"{m}|{fam}|ungated")
            row.append(f"{100*s['return_on_capital']:+.1f}%" if s else "—")
        fam_rows.append(row)

    return [
        {
            "title": "Does a better-shaped move distribution choose better?",
            "note": ("The registered primary. Common universe, all eight families "
                     "priced under every model, so 'best of eight' is the same "
                     "question everywhere. Chance is 12.5% on top-1, 37.5% on "
                     "top-3, and a mean rank of 4.5. `centre` is the share of "
                     "picks that are centre-peaked — the bias the additive "
                     "clip's spike at zero is expected to create."),
            "columns": ["move model", "events", "top-1", "top-3", "mean rank",
                        "rho", "rho>0", "centre", "chooser return",
                        "spread captured", "sim − real"],
            "align": ["---"] + ["---:"] * 10,
            "rows": hit_rows,
        },
        {
            "title": "Return on capital per family, ungated, by move model",
            "note": ("Same events, same structures, same prices. Only the "
                     "simulated move's distribution differs, so any column-wise "
                     "difference is attributable to it and to nothing else."),
            "columns": ["family", "payoff"] + MODELS,
            "align": ["---", "---"] + ["---:"] * len(MODELS),
            "rows": fam_rows,
        },
    ]


if __name__ == "__main__":
    main(record="--no-ledger" not in sys.argv)
