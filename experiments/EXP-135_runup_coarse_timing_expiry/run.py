"""Matched coarse-grid comparisons and standard engine evaluations for EXP-135."""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_job import HERE, RESULTS, OFFSETS, log, write_json, request_history
from build import RULES
from engine.evaluate import evaluate, build_equity
from experiments import lib

def clustered_interval(values, dates, seed=135, draws=2000):
    frame = pd.DataFrame({"value":np.asarray(values,float),
                           "week":pd.to_datetime(dates).dt.to_period("W").astype(str).to_numpy()})
    frame = frame.dropna()
    if len(frame) < 2:
        return [None,None]
    grouped = frame.groupby("week").value.agg(["sum","count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(0,draws,200):
        idx = rng.integers(0,len(grouped),size=(min(200,draws-len(estimates)),len(grouped)))
        estimates.extend((sums[idx].sum(axis=1)/counts[idx].sum(axis=1)).tolist())
    return np.quantile(estimates,[.025,.975]).tolist()

def stats(frame):
    mid = frame[np.isclose(frame.fill_alpha,.5)]
    result = dict(n=len(mid))
    for alpha,label in ((0.,"worst"),(.25,"quarter"),(.5,"mid"),(.75,"three_quarters"),(1.,"best")):
        f = frame[np.isclose(frame.fill_alpha,alpha)]
        result[label] = float(f.ret_net.mean()) if len(f) else None
    if mid.empty:
        return result
    result.update(gross_mid=float(mid.ret.mean()), median=float(mid.ret_net.median()),
         win_rate=float((mid.ret_net>0).mean()),
         capital_weighted=float((mid.exit_value_net-mid.entry_cost_net).sum()/mid.entry_cost_net.sum()),
         net_per_capital_day=float((mid.exit_value_net-mid.entry_cost_net).sum()/
             (mid.entry_cost_net*(mid.exit_date-mid.entry_date).dt.days).sum()),
         ci95=clustered_interval(mid.ret_net,mid.event_date),
         mean_exit_dte=float(mid.dte_exit.mean()) if "dte_exit" in mid else None)
    means = [result[x] for x in ("worst","quarter","mid","three_quarters","best")]
    if means[0]>=0:
        result["breakeven_alpha"] = 0.
    elif means[-1]<0:
        result["breakeven_alpha"] = None
    else:
        for i in range(1,5):
            if means[i]>=0:
                result["breakeven_alpha"] = (i-1)*.25 + .25*(-means[i-1])/(means[i]-means[i-1])
                break
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-harness",action="store_true")
    parser.add_argument("--no-ledger",action="store_true")
    args = parser.parse_args()
    trades = pd.read_parquet(RESULTS / "grid_trades.parquet")
    coverage = json.loads((RESULTS / "coverage.json").read_text())
    history = request_history()
    planned = json.loads((RESULTS / "pull_plan.json").read_text())
    finished = {r["signature"] for r in history if r["status"] in ("ok","split","absent")}
    import hashlib
    unprocessed = [p for p in planned["jobs"] if hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() not in finished]
    if unprocessed:
        raise RuntimeError(f"Acquisition incomplete: {len(unprocessed)} entry requests remain")
    if coverage["new_exit_calls"]:
        raise RuntimeError("Exit repair plan has unprocessed requests")
    mid = trades[np.isclose(trades.fill_alpha,.5)].copy()
    counts = mid.groupby("event_id").size()
    matched_ids = set(counts[counts==12].index)
    matched = trades[trades.event_id.isin(matched_ids)].copy()
    if not matched_ids:
        raise RuntimeError("No fully matched events; report coverage before interpreting timing")
    log(f"Fully matched: {len(matched_ids)}/{coverage['events']} events")

    summaries, yearly, pairs, expiry_pairs, sensitivities = [],[],[],[],[]
    windows = {"all":(2018,2026),"discovery":(2018,2023),"validation":(2024,2026)}
    for rule in RULES:
        within = set(mid[mid.rule==rule].groupby("event_id").size().loc[lambda s:s==4].index)
        for offset in OFFSETS:
            base = trades[(trades.rule==rule)&(trades.offset==offset)]
            for cohort, ids in (("matched12",matched_ids),("matched4",within),("available",None)):
                cell = base if ids is None else base[base.event_id.isin(ids)]
                for window,(lo,hi) in windows.items():
                    f = cell[cell.event_date.dt.year.between(lo,hi)]
                    summaries.append(dict(rule=rule,offset=offset,cohort=cohort,window=window,**stats(f)))
            cell = matched[(matched.rule==rule)&(matched.offset==offset)]
            for year, f in cell.groupby(cell.event_date.dt.year):
                yearly.append(dict(rule=rule,offset=offset,year=int(year),**stats(f)))
            m = cell[np.isclose(cell.fill_alpha,.5)]
            for label, event_ids in (
                ("no_repaired_quotes",set(m.loc[~m.quote_repaired,"event_id"])),
                ("entry_spread_le_10pct",set(m.loc[m.entry_spread_pct<=.10,"event_id"])),
                ("entry_spread_le_25pct",set(m.loc[m.entry_spread_pct<=.25,"event_id"])),
                ("exclude_top_1pct_winners",set(m.loc[m.ret_net<=m.ret_net.quantile(.99),"event_id"]))):
                sensitivities.append(dict(rule=rule,offset=offset,slice=label,**stats(cell[cell.event_id.isin(event_ids)])))
        r = mid[(mid.rule==rule)&mid.event_id.isin(matched_ids)]
        pivot = r.pivot(index="event_id",columns="offset",values="ret_net")
        dates = r.drop_duplicates("event_id").set_index("event_id").event_date
        for earlier,later in zip(OFFSETS[:-1],OFFSETS[1:]):
            delta = pivot[later]-pivot[earlier]
            for window,(lo,hi) in windows.items():
                ds = dates.reindex(delta.index)
                keep = ds.dt.year.between(lo,hi)
                d,ds = delta[keep],ds[keep]
                ci = clustered_interval(d,ds)
                pairs.append(dict(rule=rule,earlier=earlier,later=later,window=window,n=len(d),
                             later_minus_earlier=float(d.mean()),ci95=ci))
        log(f"Summaries and paired intervals complete: {rule}")

    expiry_comparisons = (
        ("first_post_event", "second_post_event"),
        ("first_post_event", "exit_plus_14_calendar_days"),
        ("second_post_event", "exit_plus_14_calendar_days"),
    )
    matched_mid = matched[np.isclose(matched.fill_alpha,.5)]
    for offset in OFFSETS:
        at_offset = matched_mid[matched_mid.offset==offset]
        pivot = at_offset.pivot(index="event_id",columns="rule",values="ret_net")
        dates = at_offset.drop_duplicates("event_id").set_index("event_id").event_date
        for reference, challenger in expiry_comparisons:
            delta = pivot[challenger]-pivot[reference]
            same_expiry = at_offset.pivot(index="event_id",columns="rule",values="expiry")
            same = same_expiry[challenger] == same_expiry[reference]
            for window,(lo,hi) in windows.items():
                ds = dates.reindex(delta.index)
                keep = ds.dt.year.between(lo,hi)
                d,ds = delta[keep],ds[keep]
                expiry_pairs.append(dict(
                    offset=offset,reference=reference,challenger=challenger,
                    window=window,n=len(d),same_expiry_share=float(same[keep].mean()),
                    challenger_minus_reference=float(d.mean()),
                    ci95=clustered_interval(d,ds)))

    control = pd.read_parquet(RESULTS/"control.parquet")
    control = control[control.event_id.isin(matched_ids)].copy()
    control["entry_cost_net"] = control.entry_cost+.013
    control["exit_value_net"] = control.exit_value-.013
    control["ret_net"] = (control.exit_value_net-control.entry_cost_net)/control.entry_cost_net
    control_stats = {window:stats(control[control.event_date.dt.year.between(lo,hi)])
                     for window,(lo,hi) in windows.items()}
    for name,data in (("summary",summaries),("by_year",yearly),
                      ("paired_differences",pairs),("expiry_differences",expiry_pairs),
                      ("sensitivities",sensitivities)):
        write_json(RESULTS/(name+".json"),data)
        pd.DataFrame(data).to_csv(RESULTS/(name+".csv"),index=False)
    write_json(RESULTS/"control_summary.json",control_stats)

    spec = lib.load_spec(HERE/"spec.yaml")
    ledger = lib.ledger_read()
    already = set(ledger.loc[ledger.stage == "ran", "spec_hash"])
    if not args.skip_harness:
        for rule in RULES:
            for offset in OFFSETS:
                log(f"Engine evaluation: {rule}, T-{offset}")
                cell = matched[(matched.rule==rule)&(matched.offset==offset)].copy()
                cell_spec = dict(spec,grid_cell=True,primary_spec=dict(spec["primary_spec"],entry_sessions=offset,expiry_rules=rule))
                run_dir = HERE/"cells"/f"{rule}_t{offset}"
                run_dir.mkdir(parents=True,exist_ok=True)
                # Engine accounting uses quote-only returns so the transaction
                # log reconciles exactly against legs. Net-fee comparisons above
                # are the research headline and are explicitly distinguished.
                result = evaluate(cell_spec,cell,run_dir=run_dir,mc_paths=200,
                                  fractions=(.02,.05),write_report=True,
                                  input_files=[RESULTS/"grid_trades.parquet"])
                cell_hash = lib.spec_hash(cell_spec)
                if not args.no_ledger and cell_hash not in already:
                    lib.record_evaluation(run_dir,cell_spec,result.results)
                    already.add(cell_hash)
    plot(summaries,yearly)
    report(summaries,pairs,expiry_pairs,control_stats,coverage,len(matched_ids),history)
    log("Engine evaluation: registered incumbent control for root report")
    root_result = evaluate(
        spec, control, run_dir=HERE, mc_paths=200, fractions=(.02,.05),
        write_report=True,
        input_files=[RESULTS/"control.parquet", RESULTS/"grid_trades.parquet",
                     RESULTS/"summary.json", RESULTS/"paired_differences.json",
                     RESULTS/"expiry_differences.json", RESULTS/"sensitivities.json"],
        extra_sections=comparison_sections(
            summaries, pairs, expiry_pairs, sensitivities, control_stats,
            coverage, len(matched_ids)))
    root_rows = ledger[(ledger.spec_hash == lib.spec_hash(spec)) & (ledger.stage == "ran")]
    root_has_metrics = bool(
        len(root_rows) and
        (root_rows.oos_mean_mid.notna() & root_rows.sharpe_trade.notna()).any())
    if not args.no_ledger and not root_has_metrics:
        lib.record_evaluation(HERE,spec,root_result.results)
    log(f"Finished: {root_result.report_path}")

def comparison_sections(summaries,pairs,expiry_pairs,sensitivities,control,coverage,n):
    """Put the complete timing decision inside the generated report."""
    def pct(value):
        return "n/a" if value is None else f"{100*value:+.2f}%"

    matched_all = {
        (s["rule"],s["offset"]):s for s in summaries
        if s["cohort"]=="matched12" and s["window"]=="all"
    }
    matched_validation = {
        (s["rule"],s["offset"]):s for s in summaries
        if s["cohort"]=="matched12" and s["window"]=="validation"
    }
    result_rows = []
    for rule in RULES:
        for offset in OFFSETS:
            s = matched_all[(rule,offset)]
            v = matched_validation[(rule,offset)]
            result_rows.append([
                rule,f"T-{offset}",f"{s['n']:,}",pct(s["worst"]),pct(s["mid"]),
                pct(s["best"]),pct(s["median"]),pct(s["capital_weighted"]),
                pct(v["mid"]),
            ])

    timing_rows = []
    for p in pairs:
        if p["window"] == "discovery":
            continue
        lo,hi = p["ci95"]
        timing_rows.append([
            p["rule"],f"T-{p['earlier']} to T-{p['later']}",p["window"],
            f"{p['n']:,}",pct(p["later_minus_earlier"]),
            f"[{pct(lo)}, {pct(hi)}]",
        ])

    expiry_rows = []
    for p in expiry_pairs:
        if p["window"] == "discovery":
            continue
        lo,hi = p["ci95"]
        expiry_rows.append([
            f"T-{p['offset']}",f"{p['reference']} to {p['challenger']}",p["window"],
            f"{p['n']:,}",pct(p["challenger_minus_reference"]),
            f"[{pct(lo)}, {pct(hi)}]",f"{100*p['same_expiry_share']:.1f}%",
        ])

    sensitivity_rows = []
    for s in sensitivities:
        if s["offset"] not in (7,3):
            continue
        sensitivity_rows.append([
            s["rule"],f"T-{s['offset']}",s["slice"],f"{s['n']:,}",
            pct(s["mid"]),pct(s.get("capital_weighted")),
        ])

    t3 = [matched_all[(rule,3)]["mid"] for rule in RULES]
    t7 = [matched_all[(rule,7)]["mid"] for rule in RULES]
    t14 = [matched_all[(rule,14)]["mid"] for rule in RULES]
    t3_validation = [matched_validation[(rule,3)]["mid"] for rule in RULES]
    control_all = control["all"]
    directional = [
        f"- Fully matched coverage is {n:,} of {coverage['events']:,} frozen events across all 12 cells.",
        f"- T-3 returned {pct(min(t3))} to {pct(max(t3))}; T-7 returned {pct(min(t7))} to {pct(max(t7))}; T-14 returned {pct(min(t14))} to {pct(max(t14))}.",
        f"- T-3 in 2024-26 ranged from {pct(min(t3_validation))} to {pct(max(t3_validation))}, which supports refining the late region without establishing a live optimum.",
        f"- The incumbent T-14 minimum-30-DTE control returned {pct(control_all['mid'])} at modeled mid fills after commissions.",
        "- No expiry rule consistently dominated on paired events. Removing the largest 1% of winners reduced every cell mean to approximately zero or below.",
        "- Returns in these tables include $0.65 per option contract per side. Bid and ask fills are modeled EOD marks, not achieved executions.",
    ]
    return [
        {
            "title":"Directional decision",
            "note":"Exploratory comparison only. No model or trading rule was promoted.",
            "body":directional,
        },
        {
            "title":"Matched timing and expiry grid",
            "note":"Expiry was selected from the T-21 chain and fixed across entry dates. Each entry chose its own ATM strike and held it through the common pre-print exit.",
            "columns":["expiry rule","entry","n","worst","mid","best","median","capital weighted","mid 2024-26"],
            "align":["---","---:","---:","---:","---:","---:","---:","---:","---:"],
            "rows":result_rows,
        },
        {
            "title":"Paired adjacent timing differences",
            "note":"Positive means entering later improved net return on the same event. Intervals use 2,000 resamples clustered by earnings week and have no multiplicity adjustment.",
            "columns":["expiry rule","change","window","n","later minus earlier","95% interval"],
            "align":["---","---","---","---:","---:","---:"],
            "rows":timing_rows,
        },
        {
            "title":"Paired expiry differences",
            "note":"Positive means the challenger improved net return on the same event and entry date.",
            "columns":["entry","reference to challenger","window","n","difference","95% interval","same contract"],
            "align":["---:","---","---","---:","---:","---:","---:"],
            "rows":expiry_rows,
        },
        {
            "title":"Execution and right-tail sensitivity at T-7 and T-3",
            "columns":["expiry rule","entry","slice","n","mean","capital weighted"],
            "align":["---","---:","---","---:","---:","---:"],
            "rows":sensitivity_rows,
        },
    ]

def plot(summaries,yearly):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = pd.DataFrame(summaries)
    f = f[(f.cohort=="matched12")&(f.window.isin(["discovery","validation"]))]
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,window in zip(axes,["discovery","validation"]):
        for rule in RULES:
            r = f[(f.window==window)&(f.rule==rule)].set_index("offset").reindex(OFFSETS)
            mean = r.mid.to_numpy()*100
            ci = np.array(r.ci95.tolist(),float)*100
            ax.errorbar(OFFSETS,mean,yerr=np.maximum(0,np.array([mean-ci[:,0],ci[:,1]-mean])),
                        marker="o",capsize=3,label=rule)
        ax.axhline(0,color="gray",linewidth=.8)
        ax.set_xticks(OFFSETS)
        ax.invert_xaxis()
        ax.set_title(window.capitalize())
        ax.set_xlabel("Trading sessions before pre-print exit")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean return after commissions, mid fills (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Same earnings events and fixed expiries across entry dates")
    fig.tight_layout()
    figures = HERE/"figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures/"timing_expiry.png",dpi=180)
    fig.savefig(figures/"timing_expiry.svg")
    plt.close(fig)

def report(summaries,pairs,expiry_pairs,control,coverage,n,history):
    def pct(x):
        return "n/a" if x is None else f"{100*x:+.2f}%"
    matched_all = {
        (s["rule"], s["offset"]): s for s in summaries
        if s["cohort"] == "matched12" and s["window"] == "all"
    }
    matched_validation = {
        (s["rule"], s["offset"]): s for s in summaries
        if s["cohort"] == "matched12" and s["window"] == "validation"
    }
    t3 = [matched_all[(rule, 3)]["mid"] for rule in RULES]
    t7 = [matched_all[(rule, 7)]["mid"] for rule in RULES]
    t14 = [matched_all[(rule, 14)]["mid"] for rule in RULES]
    t21_capital = [matched_all[(rule, 21)]["capital_weighted"] for rule in RULES]
    t3_validation = [matched_validation[(rule, 3)]["mid"] for rule in RULES]
    rows = ["# EXP-135: coarse pre-earnings timing and expiry comparison","",
        "Exploratory comparison. No model or trading rule was promoted.","",
        f"Frozen cohort: {coverage['events']:,} events; all 12 cells priced on {n:,} matched events.",
        "Expiry is selected from the T-21 chain and held fixed across entry dates. Each entry selects its own ATM strike and holds it through the common pre-print exit.",
        "Returns below include $0.65 per option contract per side ($2.60 round trip per straddle). Bid/ask fills are modeled, not achieved executions.","",
        "## Directional read","",
        f"- The useful entry region is late. T-3 returned {pct(min(t3))} to {pct(max(t3))} across the expiry rules; T-7 returned {pct(min(t7))} to {pct(max(t7))}. T-14 returned {pct(min(t14))} to {pct(max(t14))}.",
        f"- T-3 remained directionally mixed but near zero in 2024-26: {pct(min(t3_validation))} to {pct(max(t3_validation))}. This supports refining the late region, but does not establish a live optimum.",
        f"- T-21 had positive equal-weighted means, while capital-weighted returns were {pct(min(t21_capital))} to {pct(max(t21_capital))}. Its apparent edge is concentrated in cheap contracts and the right tail.",
        "- No expiry rule consistently dominated on paired events. The nearest post-event expiry had the highest equal-weighted T-3 mean; the second and 14-calendar-day rules had cleaner capital-weighted and tight-spread results.",
        "- Removing each cell's largest 1% of winners reduced every mean to approximately zero or below. Any next-stage gate must preserve access to the right tail rather than optimize win rate alone.","",
        "![Timing and expiry](figures/timing_expiry.png)","",
        "## Matched results","",
        "| Expiry rule | Entry | n | Worst | Mid | Best | Mid median | Mid 2024-26 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for rule in RULES:
        for offset in OFFSETS:
            def find(window):
                return next(s for s in summaries if s["rule"]==rule and s["offset"]==offset and s["cohort"]=="matched12" and s["window"]==window)
            s,v = find("all"),find("validation")
            rows.append(f"| {rule} | -{offset} | {s['n']} | {pct(s['worst'])} | {pct(s['mid'])} | {pct(s['best'])} | {pct(s.get('median'))} | {pct(v['mid'])} |")
    c = control["all"]
    rows += ["",f"Existing T-14 / minimum-30-DTE control on the same events: worst {pct(c['worst'])}, mid {pct(c['mid'])}, best {pct(c['best'])}.","",
        "## Adjacent timing differences","",
        "Positive means entering later improved net return. Intervals use 2,000 resamples clustered by earnings week. These are exploratory intervals without multiplicity adjustment; an interval crossing zero does not establish equivalence.","",
        "| Expiry rule | Earlier -> later | Window | Difference | 95% interval |",
        "|---|---|---|---:|---|"]
    for p in pairs:
        if p["window"]=="discovery":
            continue
        lo,hi = p["ci95"]
        rows.append(f"| {p['rule']} | {p['earlier']} -> {p['later']} | {p['window']} | {pct(p['later_minus_earlier'])} | [{pct(lo)}, {pct(hi)}] |")
    rows += ["","## Paired expiry differences","",
        "Positive means the challenger expiry improved return on the same event and entry date. A high same-expiry share means the two labels often resolve to the same contract.","",
        "| Entry | Reference -> challenger | Window | Difference | 95% interval | Same contract |",
        "|---:|---|---|---:|---|---:|"]
    for p in expiry_pairs:
        if p["window"]=="discovery":
            continue
        lo,hi = p["ci95"]
        rows.append(f"| -{p['offset']} | {p['reference']} -> {p['challenger']} | {p['window']} | {pct(p['challenger_minus_reference'])} | [{pct(lo)}, {pct(hi)}] | {100*p['same_expiry_share']:.1f}% |")
    calls = sum(r.get("network_calls",0) for r in history)
    rows += ["","## Coverage and interpretation","",
        f"- Logged acquisition attempts: {calls:,}. Remaining unavailable exit cells: {coverage['missing_exit_cells']:,}.",
        "- All-available and within-expiry matched cohorts are in results/summary.csv. Compare them with the fully matched cohort before attributing an effect to timing.",
        "- Results by year are in results/by_year.csv; repaired-quote, tight-spread and large-winner sensitivity results are in results/sensitivities.csv.",
        "- Capital-days efficiency is descriptive, not an annualized compounded return. Engine equity curves account for overlapping trades with a 100% deployment cap, but mark open positions at entry cost and use fractional contracts.",
        "- Engine cell evaluations use quote-only returns for exact transaction reconciliation. The comparison tables use returns after commissions.",
        "- The 2024-26 window was reserved from configuration selection in this experiment, but has been examined in earlier program research and is not a pristine holdout.",
        "- Historical calendar revisions, current-listed-name coverage and conditioning on baseline replay availability limit generalization. EOD data cannot validate immediately-before-close order timing or achieved fills.",
        "- The expiry rules can select the same expiry. A flat or noisy neighboring-date comparison warrants a finer follow-up only where results persist across years and execution assumptions.","",
        "Reproduce: data_job.py --dry-run; data_job.py --confirm; build.py; data_job.py --confirm --repair-plan results/exit_repair_plan.json; build.py; run.py.",""]
    (HERE/"COMPARISON.md").write_text("\n".join(rows))
if __name__ == "__main__":
    main()
