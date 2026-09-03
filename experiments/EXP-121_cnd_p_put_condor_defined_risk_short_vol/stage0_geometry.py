#!/usr/bin/env python3
"""EXP-121 stage 0 — does a symmetric condor exist on real strike grids?

    python3 experiments/EXP-121_.../stage0_geometry.py

Registered in spec.yaml BEFORE it ran. It looks at strike grids, spot, and the
oquants implied move. It does not price anything, does not read an exit chain,
and therefore cannot see whether a width made money — which is the whole reason
the primary width is allowed to be chosen from its output.

Strike selection goes through the structure's own StrikeSelector objects, not a
re-derivation of them, so what stage 0 measures is literally what the replay
will resolve.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import replay as replay_mod  # noqa: E402
from engine.build_trades import event_universe  # noqa: E402
from engine.features import load_panel  # noqa: E402
from engine.replay import _clean, load_chain_index  # noqa: E402
from engine.structures import (  # noqa: E402
    ChainSnapshot,
    StructureError,
    put_condor,
)
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage0_geometry.json"

#: Registered in spec.yaml before this ran.
CANDIDATE_WIDTHS = (0.025, 0.05, 0.075, 0.10)
RESOLVE_FLOOR = 0.50


def geometry_row(snapshot: ChainSnapshot, widths=CANDIDATE_WIDTHS) -> dict:
    """Resolve the four strikes at every candidate width. No pricing."""
    structure = put_condor()
    legs = {leg.name: leg for leg in structure.legs}
    out: dict = {}

    try:
        expiry = legs["short_lo"].expiry.select(
            snapshot.rows, snapshot.event_date, snapshot.session)
    except StructureError:
        for w in widths:
            out[f"fail_{w}"] = "no_post_event_expiry"
        return out
    at_expiry = snapshot.rows[snapshot.rows["expiry"] == expiry]
    spot = snapshot.spot_price
    out["expiry"] = expiry
    out["dte"] = int(at_expiry["dte"].iloc[0]) if len(at_expiry) else None
    out["spot"] = spot
    out["n_put_strikes"] = int(
        at_expiry.loc[at_expiry["right"] == "P", "strike"].nunique())

    # short_lo is width-independent; resolving it once also separates "this
    # ladder has nothing at the money" from "this ladder cannot carry a wing".
    resolved_base: dict[str, float] = {}
    try:
        resolved_base["short_lo"] = legs["short_lo"].strike.select(
            at_expiry, spot, {}, right="P")
    except StructureError:
        for w in widths:
            out[f"fail_{w}"] = "no_bracket_below"
        return out
    out["short_lo"] = resolved_base["short_lo"]

    order = ("short_hi", "long_lo", "long_hi")
    cause = {"short_hi": "no_listed_offset", "long_lo": "no_listed_mirror_low",
             "long_hi": "no_listed_mirror_high"}
    for w in widths:
        structure_w = put_condor(width=w)
        legs_w = {leg.name: leg for leg in structure_w.legs}
        resolved = dict(resolved_base)
        failed = None
        for name in order:
            try:
                resolved[name] = legs_w[name].strike.select(
                    at_expiry, spot, resolved, right="P")
            except StructureError:
                failed = cause[name]
                break
        if failed is not None:
            out[f"fail_{w}"] = failed
            continue
        spacing = resolved["short_hi"] - resolved["short_lo"]
        out[f"fail_{w}"] = None
        out[f"spacing_{w}"] = spacing
        out[f"spacing_pct_{w}"] = 100.0 * spacing / spot
        out[f"k1_{w}"] = resolved["long_lo"]
        out[f"k4_{w}"] = resolved["long_hi"]
    return out


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    assert tuple(spec["stage0"]["candidate_widths"]) == CANDIDATE_WIDTHS
    started = time.time()

    events = event_universe()
    structure = put_condor()
    plan = replay_mod.plan_events(structure, events)
    available = replay_mod.available_chain_keys()
    frame = plan.frame
    has_entry = np.array(
        [(t, d) in available for t, d in zip(frame["ticker"], frame["entry_date"])])
    has_exit = np.array(
        [(t, d) in available for t, d in zip(frame["ticker"], frame["exit_date"])])
    print(f"[stage0] planned {len(frame):,}; entry chain {has_entry.sum():,}; "
          f"entry+exit {int((has_entry & has_exit).sum()):,}", flush=True)
    frame = frame[has_entry].reset_index(drop=True)
    frame["has_exit_chain"] = has_exit[has_entry]

    rows: list[dict] = []
    for year, block in frame.groupby(frame["entry_date"].dt.year, sort=True):
        keys = {(t, d) for t, d in zip(block["ticker"], block["entry_date"])}
        index = load_chain_index(keys, progress_every=0)
        for plan_row in block.to_dict("records"):
            chain = index.get(plan_row["ticker"], plan_row["entry_date"])
            if chain is None or chain.empty:
                continue
            chain = _clean(chain)
            if chain.empty:
                continue
            try:
                snapshot = ChainSnapshot(
                    ticker=plan_row["ticker"], obs_date=plan_row["entry_date"],
                    event_date=plan_row["event_date"], rows=chain,
                    session=plan_row["session"])
                geom = geometry_row(snapshot)
            except StructureError:
                continue
            rows.append({
                "event_id": plan_row["event_id"], "ticker": plan_row["ticker"],
                "event_date": plan_row["event_date"], "year": int(year),
                "has_exit_chain": bool(plan_row["has_exit_chain"]), **geom,
            })
        del index
        print(f"[stage0] {year}: {len(rows):,} events measured, "
              f"{time.time() - started:.0f}s", flush=True)

    geom = pd.DataFrame(rows)
    geom.to_parquet(HERE / "results" / "stage0_geometry.parquet", index=False)

    # The implied move the width rule sizes against: oquants `implied_move`,
    # registered in spec.yaml (EXP-120 measured it as the better-calibrated of
    # the two series). Joined on (ticker, event date) like every other panel read.
    panel = load_panel()[["ticker", "date", "implied_move", "or_implied"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    geom = geom.merge(panel, left_on=["ticker", "event_date"],
                      right_on=["ticker", "date"], how="left")

    report = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "n_events_with_entry_chain": int(len(geom)),
        "n_with_exit_chain": int(geom["has_exit_chain"].sum()),
        "median_put_strikes_at_expiry": float(geom["n_put_strikes"].median()),
        "implied_move_pct": {
            "oquants_median": _median(geom["implied_move"]),
            "orats_median": _median(geom["or_implied"]),
            "n_oquants": int(geom["implied_move"].notna().sum()),
        },
        "widths": {},
    }
    for w in CANDIDATE_WIDTHS:
        fail = geom[f"fail_{w}"]
        ok = fail.isna()
        spacing = geom.loc[ok, f"spacing_pct_{w}"]
        report["widths"][str(w)] = {
            "resolved": int(ok.sum()),
            "resolvability": float(ok.mean()),
            "fail_causes": {k: int(v) for k, v in Counter(fail[~ok]).items()},
            "spacing_pct_median": _median(spacing),
            "spacing_pct_p25": _q(spacing, 0.25),
            "spacing_pct_p75": _q(spacing, 0.75),
            "spacing_over_implied_median": _median(
                spacing / geom.loc[ok, "implied_move"]),
            "resolvability_by_year": {
                int(y): float(g.mean())
                for y, g in ok.groupby(geom["year"])
            },
        }

    report["width_rule"] = _apply_width_rule(report)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["width_rule"], indent=1), flush=True)
    print(f"[stage0] {OUT} in {time.time() - started:.0f}s", flush=True)


def _apply_width_rule(report: dict) -> dict:
    """The registered rule, applied mechanically to stage 0's own output."""
    implied = report["implied_move_pct"]["oquants_median"]
    trace = []
    chosen = None
    for w in CANDIDATE_WIDTHS:
        block = report["widths"][str(w)]
        a = block["resolvability"] >= RESOLVE_FLOOR
        b = (block["spacing_pct_median"] is not None
             and implied is not None
             and block["spacing_pct_median"] >= implied)
        trace.append({"width": w, "resolvability": block["resolvability"],
                      "spacing_pct_median": block["spacing_pct_median"],
                      "meets_resolvability": bool(a), "meets_spacing": bool(b)})
        if a and b and chosen is None:
            chosen = w
    fallback = None
    if chosen is None:
        eligible = [t["width"] for t in trace if t["meets_resolvability"]]
        fallback = max(eligible) if eligible else None
        chosen = fallback
    return {
        "median_implied_move_pct_oquants": implied,
        "resolve_floor": RESOLVE_FLOOR,
        "trace": trace,
        "chosen_width": chosen,
        "chosen_by_fallback": fallback is not None,
    }


def _median(series) -> float | None:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.median()) if len(v) else None


def _q(series, q: float) -> float | None:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.quantile(q)) if len(v) else None


if __name__ == "__main__":
    main()
