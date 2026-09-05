import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path("experiments/EXP-130_a_monthly_trailing_cutoff_for_the_p_l_ga")
sys.path.insert(0,"."); sys.path.insert(0,str(HERE))
import run as r
FIG = HERE/"figures"; FIG.mkdir(exist_ok=True)

sim = pd.read_csv(r.EXP129/"results"/"simulated.csv"); sim["event_date"]=pd.to_datetime(sim.event_date)
trades = pd.read_parquet(r.TRADES); trades["event_date"]=pd.to_datetime(trades.event_date)
windows=[2,3,4,6,9,12,18,24,36]
rows=[]
for w in windows:
    ids,cuts,un = r.monthly_trailing(sim, window_months=w, quantile=0.20)
    b=r.book(trades, ids, f"w{w}")
    if b.get("n"): rows.append({"w":w,"n":b["n"],"ungated":un,"cagr":b["cagr"],
        "sharpe":b["sharpe_trade"],"dd":b["max_drawdown"],"be":b["breakeven_alpha"]})
sweep=pd.DataFrame(rows)
aids,acuts,aun = r.annual_prior(sim, quantile=0.20); ab=r.book(trades,aids,"annual")

fig,axes=plt.subplots(1,4,figsize=(18,4.2))
for ax,(f,t,sc,better) in zip(axes,[("cagr","CAGR (%)",100,"up"),("sharpe","Sharpe per trade",1,"up"),
                                    ("dd","Max drawdown (%)",100,"down"),("ungated","Ungateable events",1,"down")]):
    ax.plot(sweep.w, sc*sweep[f], "o-", color="#1f77b4")
    if f!="ungated":
        ref = {"cagr":ab["cagr"],"sharpe":ab["sharpe_trade"],"dd":ab["max_drawdown"]}[f]
        ax.axhline(sc*ref, color="#d62728", linestyle="--", label="annual rule")
    else:
        ax.axhline(aun, color="#d62728", linestyle="--", label="annual rule")
    ax.axvspan(2,5,color="grey",alpha=0.16)
    ax.set_xlabel("trailing window (months)"); ax.set_title(t); ax.grid(alpha=.25); ax.legend(fontsize=8)
axes[0].annotate("shaded: window cannot\nclear the 100-event floor\n— coverage collapse,\nnot adaptation",
                 xy=(3.4,4), fontsize=7.5, color="#444")
fig.suptitle("EXP-130 — window sweep. CAGR and Sharpe peak at 6-9 months, but the bootstrap says "
             "no window is distinguishable from any other.", fontsize=10)
fig.tight_layout(rect=(0,0,1,0.9)); fig.savefig(FIG/"window_sweep.png", dpi=130); plt.close(fig)

d=json.load(open(HERE/"results"/"metrics.json"))
fig,ax=plt.subplots(figsize=(11,4.2))
for lab,colour in (("primary_monthly_12m_top20","#1f77b4"),("monthly_6m_top20","#2ca02c"),
                   ("annual_prior_years_top20","#d62728")):
    c=pd.DataFrame(d["cutoffs"][lab])
    if c.empty: continue
    ax.step(pd.PeriodIndex(c.month,freq="M").to_timestamp(), c.cutoff, where="post",
            color=colour, label=f"{lab}  (within-year SD {d['adaptation'][lab] or 0:.4f})")
ax.set_ylabel("exp_pnl_sim cutoff"); ax.set_xlabel("month")
ax.set_title("EXP-130 — the cutoff series: adaptation is real and measurable; it just does not pay")
ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG/"cutoff_series.png", dpi=130); plt.close(fig)
print("figures written"); print(sweep.to_string(index=False))
