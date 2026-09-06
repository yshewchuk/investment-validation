#!/usr/bin/env python3
"""EXP-137 — eight books, compared like for like.

    python3 experiments/EXP-137_one_book_per_family_which_enumerated_str/run.py

Four readings of every family, because each answers a different question and
the differences between them are the result:

  UNGATED, all events        is this family alive at all? The unselected
                             population — no gate, no chooser between families.
  GATED, all events          the book anyone would actually run.
  UNGATED, common universe   the eight families on IDENTICAL events, so the
                             comparison is like for like rather than eight
                             different samples.
  GATED, common universe     the same, gated.

A family positive ungated and negative gated has a GATE problem, not a
structure problem, and nothing but the pair distinguishes them. A family that
looks good on all events and bad on the common universe was being carried by
the events its rivals could not reach.

Families are never presented as one flat ranking. Six of the eight are
centre-peaked and want a quiet print; two are twin-peaked and want the modal
move. Those are two theses, not eight variants of one.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E133 = ROOT / "experiments" / "EXP-133_every_symmetric_put_structure_the_ladder"
E134 = ROOT / "experiments" / "EXP-134_priced_right_funded_and_held_structure_s"
for p in (ROOT, E133, E134, HERE):
    sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine.evaluate import evaluate, trade_stats  # noqa: E402
from experiments import common as ecommon, lib  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


margin = _load("e137_margin", E134 / "margin.py")
e137build = _load("e137_build", HERE / "build.py")
e133run = _load("e137_e133run", E133 / "run.py")
e134run = _load("e137_e134run", E134 / "run.py")
import family as fammod  # noqa: E402

RESULTS = HERE / "results"
MID = 0.5
PRIMARY = "N4q0_-1_1"
FAMILIES = e137build.FAMILIES
LABEL = {f.key: f for f in fammod.FAMILIES}


def book(trades: pd.DataFrame, fam: str, *, gated: bool, common: bool) -> pd.DataFrame:
    """One family's mid rows under one gate setting and one slice."""
    rows = e134run.apply_exit(trades[trades["arm"] == fam], "conditional")
    if common:
        rows = rows[rows["common_universe"].fillna(False).astype(bool)]
    mid = rows[np.isclose(rows["fill_alpha"].astype(float), MID)]
    if mid.empty:
        return mid
    if not gated:
        return mid.copy()
    decided = e133run.apply_gate(mid)
    return mid[mid["event_id"].isin(set(decided.loc[decided["traded"], "event_id"]))].copy()


def stats(mid: pd.DataFrame) -> dict:
    """Per-family statistics, before and after the account can pay for it."""
    if mid.empty:
        return {}
    cost = pd.to_numeric(mid["entry_cost"], errors="coerce")
    pnl = pd.to_numeric(mid["pnl"], errors="coerce")
    per_year = mid.groupby(pd.to_datetime(mid["event_date"]).dt.year)["ret"].mean()
    m = mid.copy()
    m["secured_per_contract"] = m["legs"].map(margin.secured_per_contract)
    funded = margin.simulate(m)
    ok = funded[funded["funded"]]
    return {
        "n": int(len(mid)),
        "tickers": int(mid["ticker"].nunique()),
        "mean": float(mid["ret"].mean()),
        "median": float(mid["ret"].median()),
        "win": float((mid["ret"] > 0).mean()),
        "sharpe_trade": float(trade_stats(mid["ret"], mid["event_date"])["sharpe_trade"]),
        "return_on_capital": float(pnl.sum() / cost.sum()) if cost.sum() else float("nan"),
        "years_positive": int((per_year > 0).sum()),
        "years": int(per_year.size),
        "worse_than_debit": int((pnl < -cost - 1e-9).sum()),
        "width_over_forecast": float(mid["width_over_forecast"].median()),
        "beyond_wing": float((mid["landed_over_half"] >= 1.0).mean()),
        "held_share": float(mid["held_to_expiry"].mean()),
        "funded": int(len(ok)),
        "secured_median": float(ok["secured_per_contract"].median()) if len(ok) else float("nan"),
        "final_equity": float(funded.attrs.get("final_equity", float("nan"))),
    }


def payoff_figure(fam: str, offsets, out: Path) -> None:
    """Terminal payoff of one family, in units of the anchor spacing.

    Drawn from the family's own contract vector rather than from a stored
    picture, so the chart cannot drift away from what the code trades. The
    x-axis is the move away from the anchor; the y-axis is what the structure
    is worth at expiry, which for every enumerated family is non-negative and
    zero outside the outermost strikes.
    """
    f = LABEL[fam]
    K = np.array([0.0] + [d for d in offsets] + [-d for d in offsets])
    q = np.array([f.q0] + list(f.tail) + list(f.tail), dtype=float)
    keep = q != 0
    K, q = K[keep], q[keep]
    span = max(offsets) * 1.35
    grid = np.linspace(-span, span, 2001)
    pay = np.array([(q * np.maximum(K - x, 0.0)).sum() for x in grid])

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(grid, pay, lw=2.2, color="#1f77b4")
    ax.axhline(0, lw=0.9, color="#444")
    ax.fill_between(grid, 0, pay, alpha=0.12, color="#1f77b4")
    for k, qty in zip(K, q):
        ax.axvline(k, ls=":", lw=0.9, color="#999")
        ax.annotate(f"{qty:+.0f}", (k, ax.get_ylim()[1]), ha="center",
                    va="top", fontsize=9,
                    color="#2ca02c" if qty > 0 else "#d62728")
    ax.set_xlabel("move away from the anchor, in units of the spacing a")
    ax.set_ylabel("payoff at expiry")
    ax.set_title(f"{fam}  q=({f.q0},{','.join(str(x) for x in f.tail)})  "
                 f"d={tuple(offsets)}  — {'twin-peaked' if f.twin_peaked else 'centre-peaked'}")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def ascii_payoff(fam: str, offsets) -> list[str]:
    """The same payoff as text, so the shape survives anywhere the PNG does not."""
    f = LABEL[fam]
    K = np.array([0.0] + [d for d in offsets] + [-d for d in offsets])
    q = np.array([f.q0] + list(f.tail) + list(f.tail), dtype=float)
    # A family with no contract at its own axis (the four-strike condor) has no
    # leg there, and printing a "+0" row invents one.
    keep = q != 0
    K, q = K[keep], q[keep]
    order = np.argsort(K)
    K, q = K[order], q[order]
    pts = np.array([(q * np.maximum(K - x, 0.0)).sum() for x in K])
    hi = max(pts.max(), 1e-9)
    rows = ["```", f"payoff at expiry — {fam}, spacing d={tuple(offsets)}", ""]
    for level in range(10, -1, -1):
        y = hi * level / 10
        line = "".join("#" if v >= y - 1e-9 else " " for v in
                       np.interp(np.linspace(K.min(), K.max(), 61), K, pts))
        rows.append(f"{y:6.2f}a |{line}")
    rows.append("       +" + "-" * 61)
    rows.append("        " + f"{K.min():+.0f}a".ljust(30) + "0" + f"{K.max():+.0f}a".rjust(30))
    rows.append("")
    rows.append("strike offset : " + "  ".join(f"{k:+.0f}a" for k in K))
    rows.append("contracts     : " + "  ".join(f"{v:+3.0f} " for v in q))
    rows.append("payoff        : " + "  ".join(f"{v:4.1f}" for v in pts))
    rows.append("```")
    return rows


def sample_trade(mid: pd.DataFrame, fam: str) -> tuple[list[str], dict | None]:
    """One real trade from this family's own book, chosen as the MEDIAN result.

    Deliberately not the best trade. A structure guide illustrated with its own
    top decile teaches the reader the wrong thing about what the structure
    normally does, so the row picked is the one whose return sits closest to
    the family's median.
    """
    if mid.empty:
        return ["*No traded example — this family funded nothing.*"], None
    target = mid["ret"].median()
    row = mid.iloc[(mid["ret"] - target).abs().argsort().iloc[0]]
    doc = json.loads(row["legs"])
    ex = {l["name"]: l for l in doc["exit"]}
    lines = [
        f"**{row['ticker']} — earnings {pd.Timestamp(row['event_date']).date()}**  ",
        f"Spot {row['spot_entry']:.2f} at entry → {row['spot_exit']:.2f} the day after "
        f"the print ({100*(row['spot_exit']/row['spot_entry'] - 1):+.1f}%). "
        f"Predicted move {row['pred_abs_move']:.1f}% ± {row['pred_abs_move_sd']:.1f}. "
        f"Anchor {row['anchor']:.2f}, half-width {row['half_width_pct_spot']:.1f}% of spot "
        f"({row['width_over_forecast']:.2f}× the forecast).",
        "",
        "| leg | side | qty | strike | entry bid/ask | paid | exit bid/ask | received |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for l in sorted(doc["entry"], key=lambda l: -l["strike"]):
        x = ex[l["name"]]
        lines.append(
            f"| {l['name']} | {l['side']} | {l['qty']:.0f} | {l['strike']:.2f} | "
            f"{l['bid']:.2f} / {l['ask']:.2f} | {l['price']:.3f} | "
            f"{x['bid']:.2f} / {x['ask']:.2f} | {x['price']:.3f} |")
    held = bool(row.get("held_to_expiry", False))
    lines += [
        "",
        f"Net debit **${row['entry_cost']:.2f}** per contract "
        f"(× 100 = ${100*row['entry_cost']:.0f}). "
        + (f"The post-print curve violated no-arbitrage, so this one was **held to "
           f"expiry** and settled at its terminal payoff of ${row['exit_value']:.2f}."
           if held else
           f"The post-print curve was arithmetically consistent, so it was **closed** "
           f"at ${row['exit_value']:.2f}."),
        f"Result **{100*row['ret']:+.1f}%** on the debit — this family's median trade.",
    ]
    return lines, {"ticker": row["ticker"], "ret": float(row["ret"])}


def main(record: bool = True) -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    built = e137build.build_all()
    trades = e133run.attach(built["trades"], built["tallies"])
    events = built["events"]

    results: dict[str, dict] = {}
    for fam in FAMILIES:
        for gated in (False, True):
            for cu in (False, True):
                s = stats(book(trades, fam, gated=gated, common=cu))
                if s:
                    tag = f"{fam}|{'gated' if gated else 'ungated'}|{'common' if cu else 'all'}"
                    s["offered_events"] = int(events[f"offers_{fam}"].sum())
                    results[tag] = s
        print(f"[EXP-137] {fam}: scored", flush=True)

    (RESULTS / "family_books.json").write_text(json.dumps(results, indent=1, default=str))
    pd.DataFrame(results).T.to_csv(RESULTS / "family_books.csv")

    prim = f"{PRIMARY}|gated|common"
    twin = f"N5q2_-2_1|gated|common"
    p, t = results.get(prim, {}), results.get(twin, {})
    acceptance = {
        "beats_twin_p5_family": bool(
            p.get("return_on_capital", -9) > t.get("return_on_capital", -9)
            and p.get("years_positive", -1) >= t.get("years_positive", -1)),
        "alive_ungated": bool(
            results.get(f"{PRIMARY}|ungated|all", {}).get("return_on_capital", -9) > 0),
        "defined_risk_holds": results.get(f"{PRIMARY}|gated|all", {}).get(
            "worse_than_debit", 1) == 0,
        "universe_floor": results.get(f"{PRIMARY}|gated|all", {}).get("n", 0) >= 250,
    }
    (RESULTS / "acceptance.json").write_text(json.dumps(acceptance, indent=1))
    print(f"[EXP-137] acceptance: {acceptance}", flush=True)

    spy = ecommon.load_spy_daily()
    already = set(lib.ledger_read().query("stage == 'ran'")["spec_hash"])
    for fam in FAMILIES:
        rows = e134run.apply_exit(trades[trades["arm"] == fam], "conditional")
        mid = rows[np.isclose(rows["fill_alpha"].astype(float), MID)]
        if mid.empty:
            continue
        decided = e133run.apply_gate(mid)
        keep = set(decided.loc[decided["traded"], "event_id"])
        full = rows[rows["event_id"].isin(keep)]
        if full.empty:
            continue
        is_primary = fam == PRIMARY
        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["structure"] = f"grid cell: family {fam}"
            cell["grid_cell"] = True
        run_dir = HERE if is_primary else HERE / "arms" / fam
        for sub in ("results", "figures"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        result = evaluate(
            cell, full, gate=None, run_dir=run_dir, repricer=None,
            tail_shock=ecommon.abs_move_tail_shock, spy_daily=spy,
            input_files=[RESULTS / "candidates.parquet"],
            # The GATED mid rows, not the ungated ones: the sample trade must
            # come from the book this report is about.
            extra_sections=lambda rr, k=fam, m=full[np.isclose(
                full["fill_alpha"].astype(float), MID)], d=run_dir: sections(
                    results, events, built, k, m, d),
            write_report=True)
        if record and lib.spec_hash(cell) not in already:
            lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-137] {fam}: report {result.report_path}", flush=True)


def _median_offsets(mid: pd.DataFrame, fam: str) -> tuple[int, ...]:
    """This book's most common spacing RATIO, for drawing the shape.

    Every trade picks its own width, so no single geometry describes the book.
    The modal reduced ratio is the honest representative and the caption says
    so rather than implying the width is fixed.
    """
    ratios = mid["shape_key"].dropna().str.split("@").str[-1]
    if ratios.empty:
        t = LABEL[fam].tiers
        return (1, 2, 4)[:t] if t == 3 else ((1, 3) if t == 2 else (1,))
    return tuple(int(x) for x in ratios.mode().iloc[0].split(":"))


def sections(results, events, built, cell, mid=None, run_dir=None):
    """The four readings, grouped by payoff shape rather than ranked flat."""
    def rows_for(gated, cu):
        out = []
        for shape in (False, True):                      # centre first, then twin
            group = [f for f in FAMILIES if LABEL[f].twin_peaked == shape]
            for fam in group:
                s = results.get(f"{fam}|{'gated' if gated else 'ungated'}|"
                                f"{'common' if cu else 'all'}")
                if not s:
                    continue
                out.append([
                    ("**" + fam + "**") if fam == cell else fam,
                    "twin" if shape else "centre",
                    f"{s['n']:,}", f"{s['tickers']:,}",
                    f"{100*s['mean']:+.1f}%", f"{100*s['median']:+.1f}%",
                    f"{100*s['return_on_capital']:+.1f}%",
                    f"{s['sharpe_trade']:.2f}",
                    f"{s['years_positive']}/{s['years']}",
                    f"{s['worse_than_debit']}",
                ])
        return out

    cols = ["family", "payoff", "n", "tickers", "mean", "median", "on capital",
            "Sharpe", "years+", "loses>debit"]
    align = ["---", "---"] + ["---:"] * 8

    offered = [[fam, "twin" if LABEL[fam].twin_peaked else "centre",
                LABEL[fam].label,
                f"{int(events[f'offers_{fam}'].sum()):,}",
                f"{100*events[f'offers_{fam}'].mean():.1f}%"]
               for fam in FAMILIES]

    # The structure this report is about, drawn and traded. Placed first
    # because a reader who does not know what the shape IS cannot read any of
    # the tables below it.
    shape_sections = []
    if mid is not None and run_dir is not None:
        offs = _median_offsets(mid, cell)
        png = Path(run_dir) / "figures" / "payoff.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        try:
            payoff_figure(cell, offs, png)
            img = [f"![payoff](figures/{png.name})", ""]
        except Exception:                                        # noqa: BLE001
            img = []
        trade_lines, _ = sample_trade(mid, cell)
        shape_sections = [
            {
                "title": f"The structure — {cell}",
                "note": (f"{LABEL[cell].label}. "
                         f"{'Twin-peaked' if LABEL[cell].twin_peaked else 'Centre-peaked'}: "
                         + ("it pays most when the stock moves about one spacing "
                            "either way, which is the modal earnings move."
                            if LABEL[cell].twin_peaked else
                            "it pays most when the stock does not move, so it is "
                            "short the event.")
                         + f" Drawn at d={tuple(offs)}, this book's median spacing ratio; "
                         "the width itself is chosen per event."),
                "columns": None, "rows": None,
                "body": img + ascii_payoff(cell, offs),
            },
            {
                "title": "A sample trade — the median result, not the best",
                "columns": None, "rows": None,
                "body": trade_lines,
            },
        ]

    return shape_sections + [
        {
            "title": "Is this family alive? — every event it can carry, NO gate",
            "note": ("The unselected population. No gate and no chooser between "
                     "families: each trades everything its own rules admit. A "
                     "family that cannot make money here is dead, whatever its "
                     "share of EXP-134's argmax was."),
            "columns": cols, "align": align, "rows": rows_for(False, False),
        },
        {
            "title": "The book anyone would run — trailing six-month top 20%",
            "note": ("A family positive above and negative here has a GATE "
                     "problem, not a structure problem. The pair is what tells "
                     "them apart."),
            "columns": cols, "align": align, "rows": rows_for(True, False),
        },
        {
            "title": "Like for like — the common universe, ungated",
            "note": (f"The {built['meta'].get('common_universe_events', 0):,} events "
                     "on which ALL EIGHT families are tradeable, so every row "
                     "covers identical events. This is the comparison the "
                     "primary is read on; the tables above compare eight "
                     "different samples however carefully each was run."),
            "columns": cols, "align": align, "rows": rows_for(False, True),
        },
        {
            "title": "Like for like — the common universe, gated",
            "columns": cols, "align": align, "rows": rows_for(True, True),
        },
        {
            "title": "How often each family is offered at all",
            "note": ("Share-of-argmax and ability-to-trade are different things "
                     "and EXP-134 could not separate them. Three strikes fit "
                     "almost any ladder; seven rarely do."),
            "columns": ["family", "payoff", "shape", "events offered", "share"],
            "align": ["---", "---", "---", "---:", "---:"],
            "rows": offered,
        },
    ]


if __name__ == "__main__":
    main(record="--no-ledger" not in sys.argv)
