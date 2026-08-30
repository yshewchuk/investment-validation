#!/usr/bin/env python3
"""EXP-102 — CAL-P risk mechanics: defined-risk claim, assignment exposure.

Run:  python3 experiments/EXP-102_cal_p_risk_mechanics_defined_risk_claim/run.py

Descriptive measurement, no gate, no promotion target. The standard
evaluation suite runs (breakeven alpha, per-year, regimes, MC, tail
injection, deployment); the risk-mechanics analyses the plan pre-registers
are appended by this driver.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from engine import paths  # noqa: E402
from engine.data import store  # noqa: E402
from engine.evaluate import evaluate  # noqa: E402
from experiments import common, lib  # noqa: E402

HERE = Path(__file__).resolve().parent
STRATEGY = "CAL-P"
ZEROCOST_PATH = HERE / "results" / "zero_cost_dryrun.json"

#: Losses beyond the debit: ret < -1.0 means the close cost more than the
#: debit paid, which a defined-risk structure cannot do.
DEBIT_EXCEEDED = -1.0


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    print(f"[{spec['id']}] loading engine trades …", flush=True)
    trades = common.load_engine_trades(STRATEGY)
    print(f"[{spec['id']}] {len(trades):,} rows / "
          f"{trades['event_id'].nunique():,} events", flush=True)

    spy = common.load_spy_daily()
    repricer = common.make_repricer(STRATEGY)
    input_files = sorted((paths.CURATED / "trades").glob("year=*/part-*.parquet"))

    mid = trades[np.isclose(trades["fill_alpha"].astype(float), 0.5)].copy()

    # The risk-mechanics analyses ARE the experiment; they render through the
    # generator (Phase 4 §A8) rather than being appended to REPORT.md, so the
    # defined-risk falsification carries the report's units and reaches §0.
    def required_outputs(result):
        mechanics = risk_mechanics(mid)
        (HERE / "results" / "risk_mechanics.json").write_text(
            json.dumps(mechanics, indent=1, default=str))
        return appendix_sections(spec, result, mechanics)

    result = evaluate(
        spec, trades, gate=None, run_dir=HERE,
        repricer=repricer, tail_shock=common.calp_tail_shock, spy_daily=spy,
        input_files=input_files,
        extra_sections=required_outputs,
    )
    lib.record_evaluation(HERE, spec, result.results)
    print(f"[{spec['id']}] report: {result.report_path}", flush=True)


# --------------------------------------------------------------------------
# the risk-mechanics analyses (pre-registered required outputs)
# --------------------------------------------------------------------------


def _parse_legs(mid: pd.DataFrame) -> pd.DataFrame:
    """Front/back contract details out of the stored legs blob."""
    rows = []
    for t in mid.itertuples(index=False):
        out: dict = {"event_id": t.event_id, "ticker": t.ticker,
                     "event_date": t.event_date, "exit_date": t.exit_date,
                     "entry_cost": t.entry_cost, "exit_value": t.exit_value,
                     "ret": t.ret}
        doc = json.loads(t.legs) if isinstance(t.legs, str) else {}
        out["spot_entry"] = _f(doc.get("spot_entry"))
        out["spot_exit"] = _f(doc.get("spot_exit"))
        entry = {leg["name"]: leg for leg in (doc.get("entry") or [])}
        exit_ = {leg["name"]: leg for leg in (doc.get("exit") or [])}
        front = entry.get("front_put") or {}
        out["strike_front"] = _f(front.get("strike"))
        out["expiry_front"] = front.get("expiry")
        out["dte_front_entry"] = front.get("dte")
        xf = exit_.get("front_put") or {}
        out["front_exit_bid"] = _f(xf.get("bid"))
        out["front_exit_ask"] = _f(xf.get("ask"))
        xb = exit_.get("back_put") or {}
        out["back_exit_bid"] = _f(xb.get("bid"))
        out["back_exit_ask"] = _f(xb.get("ask"))
        rows.append(out)
    return pd.DataFrame(rows)


def _f(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def risk_mechanics(mid: pd.DataFrame) -> dict:
    legs = _parse_legs(mid)
    legs["year"] = pd.to_datetime(legs["event_date"]).dt.year
    legs["realized_move"] = legs["spot_exit"] / legs["spot_entry"] - 1.0

    # Mcap buckets from the securities table (ticker-year mcap).
    sec = store.read_table("securities", years=range(2018, 2027),
                           columns=["ticker", "year", "mcap_usd"])
    legs = legs.merge(sec, on=["ticker", "year"], how="left")
    legs["mcap_bucket"] = pd.cut(
        legs["mcap_usd"], bins=[-1, 1e9, 10e9, np.inf],
        labels=["<1B", "1-10B", ">10B"])

    # -- 1. max-loss distribution vs net debit -----------------------------
    exceeded = legs[legs["ret"] < DEBIT_EXCEEDED].copy()
    exceeded = classify_exceedances(exceeded)
    by_year_exceed = exceeded.groupby("year").size().to_dict()
    by_bucket = {}
    for bucket, g in legs.groupby("mcap_bucket", observed=True):
        b = g[g["ret"] < DEBIT_EXCEEDED]
        by_bucket[str(bucket)] = {
            "n": int(len(g)),
            "n_exceeded": int(len(b)),
            "worst_ret": float(g["ret"].min()) if len(g) else None,
            "p01_ret": float(np.percentile(g["ret"], 1)) if len(g) else None,
        }
    classification_counts = exceeded["classification"].value_counts().to_dict()

    # -- 2. assignment exposure -------------------------------------------
    itm_depth = (legs["strike_front"] - legs["spot_exit"]) / legs["strike_front"]
    legs["itm_depth"] = itm_depth
    assignment = {
        "n": int(len(legs)),
        "itm_at_post_print_close": float((itm_depth > 0).mean()),
        "itm_by_gt_5pct": float((itm_depth > 0.05).mean()),
        "itm_by_gt_10pct": float((itm_depth > 0.10).mean()),
        "median_itm_depth": float(itm_depth.median()),
        "front_dte_at_entry": legs["dte_front_entry"].describe().to_dict()
        if legs["dte_front_entry"].notna().any() else {},
    }
    pin = pin_risk_at_front_expiry(legs)
    assignment["pin_risk_at_front_expiry"] = pin

    # -- 3. zero_cost selection --------------------------------------------
    zero_cost = None
    if ZEROCOST_PATH.exists():
        doc = json.loads(ZEROCOST_PATH.read_text())
        res = (doc.get("results") or [{}])[0]
        zero_cost = {
            "zero_cost_dropped": int((res.get("skipped") or {}).get("zero_cost", 0)),
            "planned": res.get("planned"),
            "replayable": res.get("replayable"),
            "priced": res.get("priced"),
            "note": ("events dropped because the structure prices at a credit at "
                     "some fill alpha — the 4,736-trade universe is conditioned on "
                     "surviving as a debit at the best fill"),
        }
    else:
        zero_cost = {"note": "zero_cost dry-run not yet available "
                             "(run engine.build_trades --strategy CAL-P --dry-run)"}

    # -- 4. tail-injection companion: the crushed-back ruin bound ------------
    #
    # The harness tail shock (common.calp_tail_shock) doubles the move and
    # keeps each leg's quoted time value — for a SAME-STRIKE calendar the
    # doubled intrinsic then cancels between the legs, so the mechanical
    # shock alone does not compound the loss. The adverse variant answers the
    # ruin question the shock exists for: if the doubled down move ALSO
    # crushed the back leg's time value to zero (deep-ITM puts trade near
    # parity), the exit value is -front_tv and the loss is the debit PLUS the
    # front leg's remaining time value. Computed over the worst-move trades.
    worst_moves = legs["realized_move"].nsmallest(
        max(1, int(round(len(legs) * 0.01)))).index
    sub = legs.loc[worst_moves]
    front_tv = np.maximum(
        (sub["front_exit_bid"] + sub["front_exit_ask"]) / 2.0
        - np.maximum(sub["strike_front"] - sub["spot_exit"], 0.0), 0.0)
    bound = -1.0 - front_tv / sub["entry_cost"]
    crushed_bound = {
        "n_worst_move_trades": int(len(sub)),
        "worst_bound_ret": float(bound.min()) if len(bound) else None,
        "worst_bound_trade": sub.loc[bound.idxmin(), "ticker"] if len(bound) else None,
        "note": ("exit_value = -front_tv when the back leg's time value is "
                 "crushed to zero at the doubled move; loss = debit + front tv"),
    }

    return {
        "debit_exceeded": {
            "n": int(len(exceeded)),
            "share": float(len(exceeded) / len(legs)) if len(legs) else None,
            "worst_ret": float(legs["ret"].min()),
            "worst_trade": legs.loc[legs["ret"].idxmin(),
                                    ["ticker", "event_date", "entry_cost",
                                     "exit_value", "ret", "realized_move"]]
            .to_dict() if len(legs) else {},
            "by_year": {str(int(k)): int(v) for k, v in by_year_exceed.items()},
            "by_mcap_bucket": by_bucket,
            "classification": classification_counts,
            "trades": exceeded[
                ["ticker", "event_date", "year", "entry_cost", "exit_value",
                 "ret", "realized_move", "mcap_usd", "classification"]
            ].sort_values("ret").to_dict("records"),
        },
        "assignment_exposure": assignment,
        "crushed_back_ruin_bound": crushed_bound,
        "zero_cost": zero_cost,
    }


def classify_exceedances(exceeded: pd.DataFrame) -> pd.DataFrame:
    """Each debit-exceeding trade: real loss, quote artifact, or stale chain.

    Three artifact signatures are checkable from the store:
    * the exit contract's chain row carries a repaired crossed quote;
    * the front put was bought back below its intrinsic at the exit spot —
      a quote no rational market would fill (stale exit chain);
    * an exit leg quoted through a very wide market (rel spread > 50%).
    Anything matching none of these is a real loss.
    """
    if exceeded.empty:
        exceeded["classification"] = pd.Series(dtype=str)
        return exceeded

    # Chain evidence only for the exceedance trades: filter per year partition
    # down to their tickers and exit dates (the full table is 15M rows and
    # would not fit in memory alongside everything else this run holds).
    exc_tickers = set(exceeded["ticker"])
    exc_exit_dates = set(pd.to_datetime(exceeded["exit_date"]).dt.normalize())
    years = sorted(pd.to_datetime(exceeded["event_date"]).dt.year.unique())
    cols = ["ticker", "obs_date", "expiry", "strike", "right",
            "bid", "ask", "quote_repaired"]
    frames = []
    for _year, chunk in store.iter_table("option_chains", years=years, columns=cols):
        chunk = chunk[chunk["ticker"].isin(exc_tickers)]
        if chunk.empty:
            continue
        chunk["obs_date"] = pd.to_datetime(chunk["obs_date"])
        chunk = chunk[chunk["obs_date"].isin(exc_exit_dates)]
        if len(chunk):
            frames.append(chunk)
    chains = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=cols))
    if len(chains):
        chains["expiry"] = pd.to_datetime(chains["expiry"])
    chain_idx = {
        (r.ticker, r.obs_date, r.expiry, round(r.strike, 4), r.right): r
        for r in chains.itertuples(index=False)
    }

    classes = []
    for row in exceeded.itertuples(index=False):
        reasons = []
        exit_date = pd.Timestamp(row.exit_date)
        expiry = pd.Timestamp(row.expiry_front) if row.expiry_front else pd.NaT
        key = (row.ticker, exit_date, expiry,
               round(float(row.strike_front), 4) if np.isfinite(row.strike_front) else None,
               "P")
        chain_row = chain_idx.get(key)
        if chain_row is not None and bool(chain_row.quote_repaired):
            reasons.append("crossed_quote_repaired")

        spot_exit = row.spot_exit
        if np.isfinite(row.strike_front) and np.isfinite(spot_exit):
            intrinsic = max(row.strike_front - spot_exit, 0.0)
            front_mid = np.nan
            if np.isfinite(row.front_exit_bid) and np.isfinite(row.front_exit_ask):
                front_mid = 0.5 * (row.front_exit_bid + row.front_exit_ask)
                spread = row.front_exit_ask - row.front_exit_bid
                if front_mid > 0 and spread / front_mid > 0.5:
                    reasons.append("wide_exit_quotes")
            # Stale exit chain: buying back the short put below intrinsic, or
            # the long back put quoted below ITS intrinsic, cannot be a
            # rational quote.
            if intrinsic > 0 and np.isfinite(front_mid) and front_mid < 0.95 * intrinsic:
                reasons.append("front_exit_below_intrinsic")

        classes.append("data_artifact: " + "+".join(reasons) if reasons else "real_loss")
    exceeded = exceeded.copy()
    exceeded["classification"] = classes
    return exceeded


def pin_risk_at_front_expiry(legs: pd.DataFrame) -> dict:
    """Spot vs front strike at the FRONT expiry — the pin/assignment window."""
    tickers = set(legs["ticker"])
    expiries = pd.to_datetime(legs["expiry_front"], errors="coerce")
    years = sorted({d.year for d in expiries.dropna()})
    frames = []
    for year, chunk in store.iter_table(
            "daily_market", years=years, columns=["ticker", "date", "spot"]):
        chunk = chunk[chunk["ticker"].isin(tickers)]
        if len(chunk):
            frames.append(chunk)
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "date", "spot"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["ticker", "date"])

    spot_at = np.full(len(legs), np.nan)
    by_ticker = {t: g for t, g in daily.groupby("ticker", sort=False)}
    for i, (idx, row) in enumerate(legs.iterrows()):
        exp = expiries.loc[idx]
        g = by_ticker.get(row["ticker"])
        if g is None or pd.isna(exp):
            continue
        dates = g["date"].to_numpy()
        # numpy 2.5 refuses searchsorted(ns-array, Timestamp); pass datetime64.
        j = int(np.searchsorted(dates, pd.Timestamp(exp).to_datetime64(),
                                side="right")) - 1
        if j >= 0:
            spot_at[i] = float(g["spot"].to_numpy()[j])

    legs = legs.copy()
    legs["spot_front_expiry"] = spot_at
    ok = legs["spot_front_expiry"].notna() & legs["strike_front"].notna()
    depth = (legs.loc[ok, "strike_front"] - legs.loc[ok, "spot_front_expiry"]) / \
        legs.loc[ok, "strike_front"]
    pin_band = depth.abs() <= 0.02
    return {
        "n_measured": int(ok.sum()),
        "itm_at_front_expiry": float((depth > 0).mean()) if ok.sum() else None,
        "pin_within_2pct": float(pin_band.mean()) if ok.sum() else None,
        "itm_by_gt_10pct_at_expiry": float((depth > 0.10).mean()) if ok.sum() else None,
    }


# --------------------------------------------------------------------------
# appendix
# --------------------------------------------------------------------------


def appendix_sections(spec, result, m: dict) -> list[dict]:
    """The pre-registered required outputs, as generator sections."""
    d = m["debit_exceeded"]
    a = m["assignment_exposure"]
    z = m["zero_cost"]
    head = result.results["headline"]
    tail = result.results["stress"].get("tail_injection", {})
    sections: list[dict] = []

    sections.append({
        "title": "Max-loss distribution vs net debit",
        "note": (
            f"Trades losing MORE than the net debit (ret < -100%): **{d['n']:,} of "
            f"{a['n']:,} ({d['share']:.1%})** at mid fills. Worst realized trade: "
            f"**{d['worst_ret'] * 100:.1f}%** "
            f"({d['worst_trade'].get('ticker')} "
            f"{str(d['worst_trade'].get('event_date'))[:10]}, paid "
            f"{d['worst_trade'].get('entry_cost'):.3f}, close cost "
            f"{-d['worst_trade'].get('exit_value'):.3f}). The defined-risk claim "
            "(max loss = net debit) is falsified unless every exceedance below "
            "classifies as a data artifact."),
        "columns": ["classification", "count"],
        "align": ["---", "---:"],
        "rows": [[k, f"{v:,}"] for k, v in sorted(d["classification"].items(),
                                                  key=lambda kv: -kv[1])],
        "body": ["", "Per mcap bucket:", "",
                 "| bucket | n | exceeded | worst ret | p01 ret |",
                 "|---|---:|---:|---:|---:|"]
                + [f"| {bucket} | {row['n']:,} | {row['n_exceeded']:,} | "
                   f"{row['worst_ret'] * 100:.1f}% | {row['p01_ret'] * 100:.1f}% |"
                   for bucket, row in d["by_mcap_bucket"].items()]
                + ["", "Per-year exceedance counts: "
                   + ", ".join(f"{y}: {n}" for y, n in sorted(d["by_year"].items()))],
        "promote_to_verdict": True,
        "verdict_row": ("Is CAL-P defined-risk?",
                        f"**No** — {d['n']:,} of {a['n']:,} trades ({d['share']:.1%}) "
                        f"lost more than the debit; worst {d['worst_ret'] * 100:.1f}%",
                        "§8.5.1"),
        "falsifies": "every exceedance classifying as a data artifact rather than a "
                     "real loss.",
    })

    pin = a.get("pin_risk_at_front_expiry", {})
    assignment_body = [
        f"Front-leg DTE at entry: median {a['front_dte_at_entry'].get('50%')}, range "
        f"{a['front_dte_at_entry'].get('min')}-{a['front_dte_at_entry'].get('max')} "
        "— the structure as priced is a 2-4 DTE front, not a 1 DTE front.",
    ]
    if pin.get("n_measured"):
        assignment_body += [
            "",
            f"At the FRONT expiry (pin/assignment window, n={pin['n_measured']:,}): "
            f"ITM {pin['itm_at_front_expiry']:.1%}, within ±2% pin band "
            f"{pin['pin_within_2pct']:.1%}, ITM by >10% "
            f"{pin['itm_by_gt_10pct_at_expiry']:.1%}.",
        ]
    sections.append({
        "title": "Assignment exposure",
        "note": (
            f"Short front put ITM at the post-print close: "
            f"**{a['itm_at_post_print_close']:.1%}** (ITM by >5%: "
            f"{a['itm_by_gt_5pct']:.1%}, by >10%: {a['itm_by_gt_10pct']:.1%}; "
            f"median depth {a['median_itm_depth'] * 100:.1f}%)."),
        "body": assignment_body,
        "promote_to_verdict": True,
        "verdict_row": ("How exposed is the short leg to assignment?",
                        f"**{a['itm_at_post_print_close']:.1%}** ITM at the post-print "
                        f"close ({a['itm_by_gt_10pct']:.1%} by more than 10%)", "§8.5.2"),
    })

    if "zero_cost_dropped" in z:
        zero_body = [
            f"`build_trades --strategy CAL-P --dry-run`: planned {z['planned']:,}, "
            f"with both chains {z['replayable']:,}, priced {z['priced']:,} — "
            f"**{z['zero_cost_dropped']:,} events dropped** because the calendar "
            "prices at a credit at some fill alpha. The priced universe is "
            "conditioned on surviving as a debit at the BEST fill — the cheapest "
            "calendars are systematically excluded from every number in this report."]
    else:
        zero_body = [z["note"]]
    sections.append({"title": "The zero-cost selection", "body": zero_body})

    if tail.get("available") is False:
        tail_body = [f"NOT RUN: {tail.get('note')}"]
    else:
        mc5 = tail.get("mc", {}).get("0.05", {})
        cb = m.get("crushed_back_ruin_bound", {})
        tail_body = [
            f"The {tail.get('n_shocked')} trades whose exit quotes changed after "
            f"re-pricing (of the worst-1%-by-move set): worst trade "
            f"{tail.get('base_worst_trade') * 100:.1f}% → "
            f"**{tail.get('shocked_worst_trade') * 100:.1f}%** of the debit. "
            f"MC P(loss) at 5% under the shocked sequence: "
            f"**{mc5.get('p_loss'):.3f}**, terminal p05 "
            f"{mc5.get('terminal_p05'):.2f}x.",
            "",
            "Mechanics note: the shock keeps each leg's quoted time value, and for a "
            "SAME-STRIKE calendar the doubled intrinsic cancels between the legs — "
            "the mechanical shock alone therefore does not compound the loss. The "
            "ruin bound is the adverse variant in which the doubled down move also "
            "crushes the back leg's time value to zero: loss = debit + front-leg time "
            f"value, worst **{cb.get('worst_bound_ret', 0) * 100:.1f}%** of the debit "
            f"({cb.get('worst_bound_trade')}). The empirical tail "
            f"({d['worst_ret'] * 100:.1f}%, already in the base distribution) shows "
            "what real quote dynamics did without any shock.",
        ]
    sections.append({"title": "Tail injection (mandatory — has_short_leg)",
                     "body": tail_body})

    sections.append({
        "title": "Breakeven alpha, deployment, and the consequence for the Sep-1 spend",
        "body": [
            "Gated universe = full unselected set (no gate in this experiment). "
            f"Breakeven alpha **{head.get('breakeven_alpha')}** vs the plan's quoted "
            f"0.475. Peak deployment {head['deployment']['peak']:.2f}x (cap "
            f"{head['deployment']['cap']}), worst cash "
            f"{head['deployment']['worst_cash']:.2%}, max concurrency "
            f"{head.get('max_concurrency')}.",
            "",
            "If the debit-exceeding tail above is real (not artifact), the CAL-P "
            "structure is NOT defined-risk in practice and the put-side chain budget "
            "should move to exit-chain coverage per the experiment plan.",
        ]})
    return sections


if __name__ == "__main__":
    main()
