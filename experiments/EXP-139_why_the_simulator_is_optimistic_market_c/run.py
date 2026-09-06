#!/usr/bin/env python3
"""EXP-139 — four arms over one set of priced candidates.

    python3 experiments/EXP-139_why_the_simulator_is_optimistic_market_c/run.py

  uncorrected  EXP-137's expectation, drawn from the pool as it stands.
  cap          the same candidates simulated from a pool conditioned on the
               event's market-cap bucket as well as its predicted-move decile.
  debias       uncorrected, minus a per-family correction estimated on a
               TRAILING window of strictly earlier events.
  both         cap, minus the same correction.

The first two differ only in the pool; the last two add a correction that is
arithmetic on top, so no third simulation is needed. `cap` is the registered
primary because it addresses a cause; `debias` treats the symptom and is
labelled as such wherever it appears.
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
E137 = ROOT / "experiments" / "EXP-137_one_book_per_family_which_enumerated_str"
for p in (ROOT, E133, E134, E137, HERE):
    sys.path.insert(0, str(p))

from engine.evaluate import evaluate  # noqa: E402
from experiments import common as ecommon, lib  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


e139 = _load("e139_build", HERE / "build.py")
e137 = _load("e139_e137build", E137 / "build.py")
e133run = _load("e139_e133run", E133 / "run.py")
e134run = _load("e139_e134run", E134 / "run.py")
import family as fammod  # noqa: E402
import cap_pool as cp  # noqa: E402

LABEL = {f.key: f for f in fammod.FAMILIES}
FAMILIES = list(e139.FAMILIES)
RESULTS = HERE / "results"
MID = 0.5
#: Trailing window for the per-family correction, and the floor below which no
#: correction is estimated. Both registered before the run.
DEBIAS_MONTHS, DEBIAS_MIN = 12, 100


def panel(frame: pd.DataFrame, tallies: pd.DataFrame) -> pd.DataFrame:
    """Mid rows on the common universe, with the registered conditional exit."""
    # `attach` merges the securities table to apply the market-cap floor, and
    # this build already carries an mcap_usd column of its own — leaving both
    # in produces mcap_usd_x / _y and the floor silently stops being applied.
    # `cap_bucket` is what this experiment needs and it survives the drop.
    frame = frame.drop(columns=["mcap_usd"], errors="ignore")
    rows = e134run.apply_exit(e133run.attach(frame, tallies), "conditional")
    mid = rows[np.isclose(rows["fill_alpha"].astype(float), MID)]
    mid = mid[mid["common_universe"].fillna(False).astype(bool)]
    full = mid.groupby("event_id")["arm"].nunique() == len(FAMILIES)
    return mid[mid["event_id"].isin(full[full].index)].copy()


def debias(mid: pd.DataFrame) -> pd.Series:
    """Per-family (simulated − realized), from strictly earlier events only.

    Causal by construction: the correction applied to an event in month M is
    estimated on that family's events in [M − 12 months, M). A correction
    fitted on the full sample would be reading the answer.
    """
    m = mid.copy()
    m["event_date"] = pd.to_datetime(m["event_date"])
    m["gap"] = m["exp_pnl_sim_select"] - m["ret"]
    m["month"] = m["event_date"].dt.to_period("M")
    out = pd.Series(0.0, index=m.index)
    for fam, sub in m.groupby("arm"):
        s = sub.sort_values("event_date")
        for month in s["month"].unique():
            start = month.to_timestamp()
            window = s[(s["event_date"] >= start - pd.DateOffset(months=DEBIAS_MONTHS))
                       & (s["event_date"] < start)]
            if len(window) >= DEBIAS_MIN:
                out.loc[s.index[s["month"] == month]] = window["gap"].mean()
    return out


def score(mid: pd.DataFrame, adj: pd.Series | None = None) -> dict:
    """Hit rate and calibration for one arm."""
    m = mid.copy()
    m["pred"] = m["exp_pnl_sim_select"] - (adj if adj is not None else 0.0)
    pred = m.pivot(index="event_id", columns="arm", values="pred")[FAMILIES]
    real = m.pivot(index="event_id", columns="arm", values="ret")[FAMILIES]
    chosen, truth = pred.values.argmax(1), real.values.argmax(1)
    rank = (-real.values).argsort(1).argsort(1)[np.arange(len(chosen)), chosen] + 1
    picked = real.values[np.arange(len(chosen)), chosen]
    tw = np.array([LABEL[f].twin_peaked for f in FAMILIES])
    gaps = {f: float((m.loc[m["arm"] == f, "pred"] - m.loc[m["arm"] == f, "ret"]).mean())
            for f in FAMILIES}
    return {
        # Kept so the report can trace WHICH events changed hands between arms
        # and whether the change helped — a hit rate alone says an arm is
        # different, not how.
        "_chosen": chosen, "_truth": truth, "_real": real.values,
        "_events": pred.index.to_numpy(),
        "events": int(len(pred)), "top1": float((chosen == truth).mean()),
        "top3": float((rank <= 3).mean()), "mean_rank": float(rank.mean()),
        "ret_chooser": float(picked.mean()),
        "ret_oracle": float(real.values.max(1).mean()),
        "ret_random": float(real.values.mean(1).mean()),
        "twin_share": float(tw[chosen].mean()),
        "mean_gap": float(np.mean(list(gaps.values()))),
        "max_gap_spread": float(max(gaps.values()) - min(gaps.values())),
        "gaps": gaps,
        "picked": {FAMILIES[i]: int((chosen == i).sum()) for i in range(len(FAMILIES))},
        "was_best": {FAMILIES[i]: int((truth == i).sum()) for i in range(len(FAMILIES))},
    }


def main(record: bool = True) -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    b139 = e139.build_all()
    b137 = e137.build_all()
    cap_mid = panel(e139.load_all(), b139["tallies"])
    unc_mid = panel(pd.read_parquet(E137 / "results" / "candidates.parquet"),
                    b137["tallies"])
    # Compare on the events BOTH arms priced, so a difference cannot come from
    # one arm having seen a wider universe than the other.
    shared = set(cap_mid["event_id"]) & set(unc_mid["event_id"])
    cap_mid = cap_mid[cap_mid["event_id"].isin(shared)]
    unc_mid = unc_mid[unc_mid["event_id"].isin(shared)]
    print(f"[EXP-139] {len(shared):,} events priced by both arms", flush=True)

    # Computed once and reused for both the comparison and the per-arm books;
    # the trailing estimate is the expensive part of this file.
    adj_unc, adj_cap = debias(unc_mid), debias(cap_mid)
    arms = {
        "uncorrected": score(unc_mid),
        "cap": score(cap_mid),
        "debias": score(unc_mid, adj_unc),
        "both": score(cap_mid, adj_cap),
    }
    for k, s in arms.items():
        print(f"[EXP-139] {k:12s} top-1 {100*s['top1']:5.1f}%  top-3 {100*s['top3']:5.1f}%  "
              f"rank {s['mean_rank']:.2f}  twin {100*s['twin_share']:5.1f}%  "
              f"mean gap {100*s['mean_gap']:+6.1f}pp  spread {100*s['max_gap_spread']:5.1f}pp  "
              f"chooser {100*s['ret_chooser']:+6.2f}%", flush=True)

    base = arms["uncorrected"]
    p = arms["cap"]
    acceptance = {
        "less_optimistic": bool(all(abs(p["gaps"][f]) <= abs(base["gaps"][f]) + 1e-9
                                    for f in FAMILIES)),
        "better_chooser": bool(p["top1"] > base["top1"]),
        "not_at_a_cost": bool(p["ret_chooser"] >= base["ret_chooser"]),
        "treatment_actually_applied": bool(b139["meta"]["cap_conditioned_share"] >= 0.5),
    }
    print(f"[EXP-139] acceptance: {acceptance}", flush=True)
    (RESULTS / "arm_comparison.json").write_text(
        json.dumps({"arms": {k: {kk: vv for kk, vv in v.items()
                                 if not kk.startswith("_")}
                             for k, v in arms.items()},
                    "acceptance": acceptance, "meta": b139["meta"]},
                   indent=1, default=str))

    print(f"\n{'family':>14} {'payoff':>7} " + "".join(f"{k:>14}" for k in arms))
    for f in FAMILIES:
        print(f"{f:>14} {'twin' if LABEL[f].twin_peaked else 'centre':>7} "
              + "".join(f"{100*arms[k]['gaps'][f]:>13.1f}pp" for k in arms))

    # An experiment without a REPORT.md generated by engine.report does not
    # exist (guides/README.md #9, AGENTS.md "Finishing an experiment"). The
    # primary's best-of-family book goes through the standard evaluation so the
    # record carries a provenance block, an alpha sweep and real headline
    # numbers rather than a comparison table alone.
    spy = ecommon.load_spy_daily()
    already = set(lib.ledger_read().query("stage == 'ran'")["spec_hash"])
    # Each SOURCE is loaded once; `debias` and `both` reuse the same rows as
    # `uncorrected` and `cap` respectively and differ only in which family the
    # adjusted prediction picks.
    sources = {}
    for src, frame, tal in (("cap", e139.load_all(), b139["tallies"]),
                            ("unc", pd.read_parquet(
                                E137 / "results" / "candidates.parquet"),
                             b137["tallies"])):
        sources[src] = e134run.apply_exit(
            e133run.attach(frame.drop(columns=["mcap_usd"], errors="ignore"), tal),
            "conditional")
    del frame

    for name, mid, adj, src in (("cap", cap_mid, None, "cap"),
                                ("uncorrected", unc_mid, None, "unc"),
                                ("debias", unc_mid, adj_unc, "unc"),
                                ("both", cap_mid, adj_cap, "cap")):
        m = mid.copy()
        m["pred"] = m["exp_pnl_sim_select"] - (adj if adj is not None else 0.0)
        pick = m.loc[m.groupby("event_id")["pred"].idxmax()]
        allr = sources[src]
        keys = set(zip(pick["event_id"], pick["arm"]))
        book = allr[[(e, a) in keys for e, a in zip(allr["event_id"], allr["arm"])]]
        if book.empty:
            continue
        is_primary = name == "cap"
        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["treatment"] = f"grid cell: {name}"
            cell["grid_cell"] = True
        run_dir = HERE if is_primary else HERE / "arms" / name
        for sub in ("results", "figures"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        result = evaluate(
            cell, book, gate=None, run_dir=run_dir, repricer=None,
            tail_shock=ecommon.abs_move_tail_shock, spy_daily=spy,
            input_files=[RESULTS / "build_meta.json"],
            extra_sections=lambda rr, k=name: sections(arms, b139, k),
            write_report=True)
        if record and lib.spec_hash(cell) not in already:
            lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-139] {name}: report {result.report_path}", flush=True)


def sections(arms, built, cell):
    order = [k for k in ("uncorrected", "cap", "debias", "both") if k in arms]
    base = arms["uncorrected"]
    F = FAMILIES

    def sel_table(arm):
        """Predicted-best against actually-best, per family, for one arm."""
        a = arms[arm]
        ch, tr, real = a["_chosen"], a["_truth"], a["_real"]
        out = []
        for i, f in enumerate(F):
            s_, t_ = ch == i, tr == i
            prec = 100 * (s_ & t_).sum() / max(s_.sum(), 1)
            rec = 100 * (s_ & t_).sum() / max(t_.sum(), 1)
            out.append([
                f, "twin" if LABEL[f].twin_peaked else "centre",
                f"{s_.sum():,}", f"{t_.sum():,}",
                f"{s_.sum() / max(t_.sum(), 1):.2f}",
                f"{prec:.1f}%", f"{rec:.1f}%",
                f"{100 * real[s_, i].mean():+.1f}%" if s_.sum() else "—",
            ])
        return out

    def flips(arm):
        """Against the uncorrected arm: which picks changed, and did it help?"""
        a = arms[arm]
        ch0, ch1, real = base["_chosen"], a["_chosen"], a["_real"]
        moved = ch0 != ch1
        if not moved.any():
            return None
        idx = np.arange(len(ch0))
        r0 = real[idx, ch0]
        r1 = real[idx, ch1]
        better = (r1 > r0) & moved
        worse = (r1 < r0) & moved
        return {
            "n": int(moved.sum()), "share": float(moved.mean()),
            "better": int(better.sum()), "worse": int(worse.sum()),
            "delta": float(r1[moved].mean() - r0[moved].mean()),
            "was_right_before": int((base["_truth"] == ch0)[moved].sum()),
            "right_after": int((a["_truth"] == ch1)[moved].sum()),
        }
    rows = [[
        f"**{k}**" if k == cell else k,
        f"{100*arms[k]['top1']:.1f}%", f"{100*arms[k]['top3']:.1f}%",
        f"{arms[k]['mean_rank']:.2f}", f"{100*arms[k]['twin_share']:.1f}%",
        f"{100*arms[k]['mean_gap']:+.1f}pp", f"{100*arms[k]['max_gap_spread']:.1f}pp",
        f"{100*arms[k]['ret_chooser']:+.2f}%",
    ] for k in order if k in arms]
    fam_rows = [[f, "twin" if LABEL[f].twin_peaked else "centre"]
                + [f"{100*arms[k]['gaps'][f]:+.1f}pp" for k in order]
                for f in FAMILIES]
    m = built["meta"]
    flip_rows = []
    for k in order:
        if k == "uncorrected":
            continue
        fl = flips(k)
        if not fl:
            continue
        flip_rows.append([
            k, f"{fl['n']:,}", f"{100*fl['share']:.1f}%",
            f"{fl['better']:,}", f"{fl['worse']:,}",
            f"{100*fl['delta']:+.2f}pp",
            f"{fl['was_right_before']:,}", f"{fl['right_after']:,}",
        ])

    net_rows = []
    for i, f in enumerate(F):
        row = [f, "twin" if LABEL[f].twin_peaked else "centre",
               f"{(base['_truth'] == i).sum():,}"]
        for k in order:
            row.append(f"{(arms[k]['_chosen'] == i).sum():,}")
        net_rows.append(row)

    outcome_rows = [[
        f"**{k}**" if k == cell else k,
        f"{100*arms[k]['ret_chooser']:+.2f}%",
        f"{100*arms[k]['ret_oracle']:+.2f}%",
        f"{100*arms[k]['ret_random']:+.2f}%",
        f"{100*(arms[k]['ret_chooser'] - arms[k]['ret_random']) / (arms[k]['ret_oracle'] - arms[k]['ret_random']):.1f}%",
    ] for k in order]

    per_arm_selection = []
    for k in order:
        per_arm_selection.append({
            "title": f"Selections under `{k}`" + (" — the registered primary"
                                                 if k == "cap" else ""),
            "note": ("`predicted` is how often this arm named the family best; "
                     "`was best` is how often it realized best — the same in "
                     "every arm, since only the prediction moves. `ratio` above "
                     "1 means the arm over-calls the family. `precision` is how "
                     "often a call was right; `recall` how much of that family's "
                     "wins it caught."),
            "columns": ["family", "payoff", "predicted", "was best", "ratio",
                        "precision", "recall", "mean return when picked"],
            "align": ["---", "---"] + ["---:"] * 6,
            "rows": sel_table(k),
        })

    return [
        {
            "title": "Four arms, one set of priced candidates",
            "note": ("`cap` conditions the residual pool on the event's "
                     "market-cap bucket as well as its predicted-move decile — "
                     "a cause. `debias` subtracts a per-family correction "
                     "estimated on a trailing window — a symptom treatment, "
                     "labelled as such. Chance on top-1 is 12.5%; twin-peaked "
                     "families actually win 36.7% of the time."),
            "columns": ["arm", "top-1", "top-3", "mean rank", "twin share",
                        "mean gap", "gap spread", "chooser return"],
            "align": ["---"] + ["---:"] * 7,
            "rows": rows,
            "body": ["", f"Cap conditioning was actually applied to "
                     f"{100*m['cap_conditioned_share']:.1f}% of draws "
                     f"(median pool {m['median_pool_size']:,.0f} rows); the rest "
                     "fell back to decile-only, which is the existing behaviour."],
        },
        {
            "title": "Simulated minus realized, per family",
            "note": ("The quantity an argmax ranks on. What matters for a "
                     "ranking is not the level but the SPREAD between "
                     "families — a common bias cancels, a differential one "
                     "reorders."),
            "columns": ["family", "payoff"] + order,
            "align": ["---", "---"] + ["---:"] * len(order),
            "rows": fam_rows,
        },
        {
            "title": "Where the choice ended up, by arm",
            "note": ("How many events each family was picked for, against how "
                     "often it actually won. `was best` is fixed — only the "
                     "predictions move — so this table is the whole story of "
                     "what each adjustment did to the selection."),
            "columns": ["family", "payoff", "was best"] + order,
            "align": ["---", "---"] + ["---:"] * (len(order) + 1),
            "rows": net_rows,
        },
        {
            "title": "What the adjustments actually changed",
            "note": ("Only the events whose pick MOVED against the uncorrected "
                     "arm. `better`/`worse` compare the realized return of the "
                     "new pick against the old one on the same event, so this "
                     "is whether the change paid — not whether the arm is "
                     "different."),
            "columns": ["arm", "picks changed", "share", "better", "worse",
                        "mean change", "right before", "right after"],
            "align": ["---"] + ["---:"] * 7,
            "rows": flip_rows,
            "body": ["", "An adjustment that moves many picks with `better` and "
                     "`worse` near parity has added variance and no skill, "
                     "which is what a symptom treatment looks like from the "
                     "inside."],
        },
        {
            "title": "What the choice was worth, by arm",
            "note": ("`oracle` takes the realized best, `random` the mean of all "
                     "eight — the ceiling and the floor. `captured` is the share "
                     "of that gap the arm's chooser actually collected."),
            "columns": ["arm", "chooser", "oracle", "random", "captured"],
            "align": ["---"] + ["---:"] * 4,
            "rows": outcome_rows,
        },
    ] + per_arm_selection


if __name__ == "__main__":
    main(record="--no-ledger" not in sys.argv)
