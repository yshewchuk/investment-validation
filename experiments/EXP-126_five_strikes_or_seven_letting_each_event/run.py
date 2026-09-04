#!/usr/bin/env python3
"""EXP-126 — five strikes or seven, chosen per event.

Run:  python3 experiments/EXP-126_five_strikes_or_seven_letting_each_event/run.py

Primary is the CHOOSER: per event, whichever of the three shapes passes the
registered entry filters with the highest reward:risk. The three single-shape
arms and the fit-based chooser run as labelled grid cells — they are the
benchmarks the primary has to beat on universe size, not results that may be
promoted in its place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.evaluate import evaluate  # noqa: E402
from experiments import common, lib  # noqa: E402

import shapes as shapes_mod  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MID = 0.5

#: EXP-125's primary, for the report's continuity line. Its universe is the
#: number this experiment exists to move.
EXP125_EVENTS, EXP125_TICKERS, EXP125_MEAN = 90, 79, 0.1182


def mid_rows(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[np.isclose(trades["fill_alpha"].astype(float), MID)]


def choose(mids: pd.DataFrame, by: str) -> pd.DataFrame:
    """One row per event: the passing shape that wins on ``by``.

    ``by="rr"`` is the registered primary — highest reward:risk. ``by="fit"``
    is the other registered criterion — the shape whose snapped geometry lands
    closest, in percent of spot, to the move the forecast asked for. Ties go to
    the seven-strike shape: it is the incumbent and its coverage is widest.
    """
    ok = mids[mids["passes"].fillna(False).astype(bool)].copy()
    ok["_incumbent"] = (ok["shape"] != "seven").astype(int)
    if by == "rr":
        ok = ok.sort_values(["rr", "_incumbent"], ascending=[False, True])
    elif by == "fit":
        ok = ok.sort_values(["fit_err", "rr", "_incumbent"],
                            ascending=[True, False, True])
    else:
        raise ValueError(by)
    return ok.groupby("event_id", sort=False).head(1)


def arm_rows(built: dict[str, pd.DataFrame], picks: pd.DataFrame) -> pd.DataFrame:
    """Every fill alpha for the (event, shape) pairs an arm decided to trade."""
    wanted = set(zip(picks["event_id"], picks["shape"]))
    parts = []
    for key, trades in built.items():
        keep = trades[[(e, s) in wanted
                       for e, s in zip(trades["event_id"], trades["shape"])]]
        if len(keep):
            parts.append(keep)
    return pd.concat(parts, ignore_index=True) if parts else built["seven"].head(0)


def funnel(mids: pd.DataFrame) -> dict:
    """What each filter costs, per arm. A drop with no reason is a defect."""
    n = int(mids["event_id"].nunique())
    return {
        "priced": n,
        "tickers_priced": int(mids["ticker"].nunique()),
        "f_reward": int(mids["f_reward"].fillna(False).sum()),
        "f_spread": int(mids["f_spread"].fillna(False).sum()),
        "f_mcap": int(mids["f_mcap"].fillna(False).sum()),
        "passes": int(mids["passes"].fillna(False).sum()),
        "rr_median": _f(mids["rr"].median()),
        "fit_err_median": _f(mids["fit_err"].median()),
    }


def traded(mid: pd.DataFrame) -> dict:
    """The numbers the acceptance criteria are written against."""
    if mid.empty:
        return {}
    cost = pd.to_numeric(mid["cost"], errors="coerce")
    pnl = pd.to_numeric(mid["pnl"], errors="coerce")
    per_year = mid.groupby("year")["ret"].mean()
    return {
        "n": int(len(mid)),
        "tickers": int(mid["ticker"].nunique()),
        "mean": _f(mid["ret"].mean()),
        "median": _f(mid["ret"].median()),
        "win": _f((mid["ret"] > 0).mean()),
        "return_on_capital": _f(pnl.sum() / cost.sum()),
        "net_pnl_per_spot": _f((pnl / pd.to_numeric(mid["spot_entry"],
                                                    errors="coerce")).sum()),
        "rr_median": _f(mid["rr"].median()),
        "rel_spread_median": _f(mid["rel_spread"].median()),
        "peak_pct_spot_median": _f(mid["peak_pct_spot"].median()),
        "fit_err_median": _f(mid["fit_err"].median()),
        "dead": _f(mid["dead"].mean()),
        "centre": _f(mid["centre"].mean()),
        "years_positive": int((per_year > 0).sum()),
        "years": int(per_year.size),
        "shapes": {k: int(v) for k, v in mid["shape"].value_counts().items()},
    }


def _f(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def sections(summary: dict, funnels: dict, arm: str) -> list[dict]:
    order = ["choose_rr", "choose_fit", "seven", "five_wide", "five_tight"]
    rows = []
    for key in order:
        s = summary.get(key)
        if not s:
            continue
        rows.append([
            f"**{key}**" if key == arm else key,
            f"{s['n']:,}", f"{s['tickers']:,}",
            f"{100*s['mean']:+.2f}%", f"{100*s['return_on_capital']:+.2f}%",
            f"{100*s['win']:.1f}%", f"{s['rr_median']:.2f}",
            f"{s['peak_pct_spot_median']:.2f}%", f"{100*s['dead']:.1f}%",
            f"{100*s['centre']:.1f}%", f"{s['years_positive']}/{s['years']}",
        ])
    # EXP-125 reported median `w`; this table reports the payoff CENTRE, which
    # for the seven-strike shape is 1.5w. 2.86% x 1.5 = 4.29% is the same
    # geometry read off a different reference point, so the published figure is
    # converted here rather than dropped into a column that does not mean it.
    rows.append(["EXP-125 seven (prior snapshot)", f"{EXP125_EVENTS}", f"{EXP125_TICKERS}",
                 f"{100*EXP125_MEAN:+.2f}%", "+5.52%", "64.4%", "—", "4.29%",
                 "16.7%", "—", "6/9"])

    fun = [[key, f"{f['priced']:,}", f"{f['tickers_priced']:,}", f"{f['f_reward']:,}",
            f"{f['f_spread']:,}", f"{f['f_mcap']:,}", f"{f['passes']:,}",
            f"{f['rr_median']:.2f}", f"{f['fit_err_median']:.2f}"]
           for key, f in funnels.items()]

    picks = summary.get(arm, {}).get("shapes", {})
    return [
        {
            "title": "Every arm, on one universe — the row count is the point",
            "note": (
                "TWIN-P reaches 90 trades because `cost < w` rejects 98.6% of "
                "priced events. The five-strike shapes are dominated payoffs — "
                "pointwise below the seven-strike tent once scaled to a common "
                "peak — so no-arbitrage puts their reward:risk at or above its "
                "on the same event, and the reward term can only pass more "
                "often. What that costs is coverage, which is the `dead` and "
                "`centre` columns: the share of prints landing beyond a wing, "
                "and the share landing at the anchor where `five_tight` alone "
                "pays nothing."
            ),
            "columns": ["arm", "n", "tickers", "mean/trade", "on capital", "win",
                        "reward:risk", "payoff centre %spot", "dead", "centre",
                        "years+"],
            "align": ["---"] + ["---:"] * 10,
            "rows": rows,
        },
        {
            "title": "Entry-rule funnel — which filter each shape actually pays",
            "columns": ["shape", "priced", "tickers", "reward>risk", "spread<=25%",
                        "mcap>=$10B", "all three", "median r:r", "median fit err"],
            "align": ["---"] + ["---:"] * 8,
            "rows": fun,
            "body": ["", "`fit err` is |payoff centre / forecast − 1| after the "
                     "ladder snap: how far the geometry the ticker could actually "
                     "carry sits from the move the forecast asked us to sit on."],
        },
        {
            "title": f"What the chooser picked ({arm})",
            "columns": ["shape", "trades"],
            "align": ["---", "---:"],
            "rows": [[k, f"{v:,}"] for k, v in sorted(picks.items(),
                                                      key=lambda kv: -kv[1])],
            "body": ["", "A chooser that always picks the same shape is not a "
                     "chooser; it is that shape, and should be read as one."],
        },
    ]


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    spy = common.load_spy_daily()
    built = shapes_mod.build_all()

    mids = pd.concat([mid_rows(t) for t in built.values()], ignore_index=True)
    funnels = {key: funnel(mid_rows(built[key])) for key in built}

    arms: dict[str, pd.DataFrame] = {}
    for key in built:
        m = mid_rows(built[key])
        arms[key] = m[m["passes"].fillna(False).astype(bool)][["event_id", "shape"]]
    arms["choose_rr"] = choose(mids, "rr")[["event_id", "shape"]]
    arms["choose_fit"] = choose(mids, "fit")[["event_id", "shape"]]

    rows = {key: arm_rows(built, picks) for key, picks in arms.items()}
    summary = {key: traded(mid_rows(ev)) for key, ev in rows.items()}
    (RESULTS / "arm_summary.json").write_text(
        json.dumps({"arms": summary, "funnel": funnels}, indent=1, default=str))

    for key, s in summary.items():
        if s:
            print(f"[EXP-126] {key}: {s['n']:,} trades on {s['tickers']:,} tickers, "
                  f"mean {100*s['mean']:+.2f}%, on capital "
                  f"{100*s['return_on_capital']:+.2f}%", flush=True)

    already_ran = set(lib.ledger_read().query("stage == 'ran'")["spec_hash"])

    # Primary first, then the grid cells, so the ledger reads in that order.
    for key in ["choose_rr", "choose_fit", "seven", "five_wide", "five_tight"]:
        ev = rows.get(key)
        if ev is None or ev.empty:
            print(f"[EXP-126] {key}: nothing traded", flush=True)
            continue
        is_primary = key == "choose_rr"
        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["structure"] = f"grid cell: {key}"
            cell["grid_cell"] = True
        # Every arm gets its OWN report and figures. Writing only the primary's
        # left the arm that mattered most — five_wide, the one that beat the
        # incumbent on capital — with nothing but a metrics blob, which is not
        # a result anybody can read. Grid cells land under `arms/<key>/` so
        # they cannot overwrite the primary's REPORT.md or its figures.
        arm_dir = HERE if is_primary else HERE / "arms" / key
        for sub in ("results", "figures"):
            (arm_dir / sub).mkdir(parents=True, exist_ok=True)
        result = evaluate(
            cell, ev, gate=None, run_dir=arm_dir,
            # Per-event widths: a shifted-date reprice would need the forecast
            # re-derived at the shifted date, so the same exemption EXP-125 took.
            repricer=None,
            tail_shock=common.abs_move_tail_shock, spy_daily=spy,
            input_files=[shapes_mod.SHAPES[s].trades_path()
                         for s in sorted(set(ev["shape"]))],
            extra_sections=lambda r, k=key: sections(summary, funnels, k),
            write_report=True,
        )
        # The ledger is a multiple-testing record, not a run log: re-running an
        # identical spec to regenerate its report is not a new test, so its row
        # is appended once and only once.
        if lib.spec_hash(cell) in already_ran:
            print(f"[EXP-126] {key}: ledger row already recorded, not duplicated",
                  flush=True)
        else:
            lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-126] {key}: report {result.report_path}", flush=True)


if __name__ == "__main__":
    main()
