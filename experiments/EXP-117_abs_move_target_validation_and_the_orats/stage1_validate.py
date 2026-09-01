#!/usr/bin/env python3
"""EXP-117 Stage 1 — triangulate the earnings-move target against Polygon.

For every event of every sampled ticker, recompute the session-aware move from
Polygon unadjusted daily closes (fetched by stage1_pull.py into Tier 1), then
adjudicate each event among the three measurements:

    oquants realized_moves      (the incumbent target)
    ORATS daily_market.spot     (Stage-0 recomputation, move_orats)
    Polygon unadjusted close    (the independent vendor)

Consensus rule (registered): any two of the three within 0.5pp define the
consensus for that event; a source deviating from consensus is wrong on that
event. Independent-vs-independent disagreement (Polygon vs ORATS) is ambiguity,
reported, not adjudicated.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/investing-plan")
sys.path.insert(0, str(ROOT))

from engine.data.fetch import Fetcher, cache_key  # noqa: E402

HERE = Path(__file__).resolve().parent
FROM = "2006-01-01"
TO = "2026-09-01"
TOL = 0.5  # pp, consensus tolerance
report: dict = {"generated_at": pd.Timestamp.now("UTC").isoformat(), "tolerance_pp": TOL}


def log(msg: str) -> None:
    print(f"[stage1] {msg}", flush=True)


def load_polygon_series(tickers: list[str]) -> tuple[dict, dict]:
    """(unadjusted series dict, per-ticker status) from the Tier-1 cache."""
    f = Fetcher()
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    status: dict[str, str] = {}
    started = time.time()
    for i, tk in enumerate(tickers):
        if i % 50 == 0:
            log(f"polygon load {i}/{len(tickers)}, {time.time()-started:.0f}s")
        endpoint = f"v2/aggs/ticker/{tk}/range/1/day/{FROM}/{TO}"
        params = {"adjusted": "false", "limit": 50000}
        key = cache_key("polygon", endpoint, params)
        path = f.body_path("polygon", key)
        if not path.exists():
            status[tk] = "not_fetched"
            continue
        try:
            with gzip.open(path, "rb") as fh:
                payload = json.loads(fh.read())
        except (OSError, ValueError) as exc:
            status[tk] = f"unreadable:{exc}"
            continue
        rows = payload.get("results") or []
        if not rows:
            status[tk] = "empty"
            continue
        ts = np.array([r["t"] for r in rows], dtype="int64")
        closes = np.array([r["c"] for r in rows], dtype="float64")
        # bar timestamps are within the UTC trade date; take the UTC date
        days = (ts // 86_400_000).astype("datetime64[D]").astype("datetime64[ns]")
        order = np.argsort(days, kind="stable")
        days, closes = days[order], closes[order]
        keep = np.isfinite(closes) & (closes > 0)
        days, closes = days[keep], closes[keep]
        uniq, idx = np.unique(days, return_index=True)  # last occurrence per day
        last_idx = np.empty(len(uniq), dtype=int)
        # np.unique return_index gives FIRST occurrence; rebuild for last
        _, rev = np.unique(days, return_inverse=True)
        for j in range(len(days)):
            last_idx[rev[j]] = j
        series[tk] = (days[last_idx], closes[last_idx])
        status[tk] = "ok"
    log(f"polygon series loaded: {sum(1 for v in status.values() if v == 'ok')}/{len(tickers)} ok")
    return series, status


def polygon_moves(events: pd.DataFrame, series: dict) -> pd.DataFrame:
    started = time.time()
    n = len(events)
    move = np.full(n, np.nan)
    reason = np.full(n, "", dtype=object)
    tickers = events["ticker"].to_numpy()
    dates = events["date"].to_numpy()
    sessions = events["session"].astype(object).where(events["session"].notna(), "").to_numpy()
    for i in range(n):
        sess = sessions[i]
        if sess not in ("BMO", "AMC"):
            reason[i] = "no_session"
            continue
        ser = series.get(tickers[i])
        if ser is None:
            reason[i] = "no_series"
            continue
        sd, sc = ser
        t = dates[i]
        if sess == "BMO":
            j_pre = int(np.searchsorted(sd, t, side="left")) - 1
            j_post = int(np.searchsorted(sd, t, side="left"))
        else:
            j_pre = int(np.searchsorted(sd, t, side="right")) - 1
            j_post = int(np.searchsorted(sd, t, side="right"))
        if j_pre < 0 or j_post >= len(sd):
            reason[i] = "before_coverage" if j_pre < 0 and len(sd) and t < sd[0] else "missing_close"
            continue
        gap = (sd[j_post] - sd[j_pre]) / np.timedelta64(1, "D")
        if gap > 5:
            reason[i] = "wide_gap"
            continue
        p, q = sc[j_pre], sc[j_post]
        move[i] = (q / p - 1.0) * 100.0
    out = events.copy()
    out["move_pg"] = move
    out["reason_pg"] = reason
    log(f"polygon moves computed: {np.isfinite(move).sum():,}/{n}, {time.time()-started:.0f}s")
    return out


def adjudicate(events: pd.DataFrame) -> pd.DataFrame:
    """Per event: which source(s) deviate from the two-way consensus."""
    oq = events["oq_move"].to_numpy(dtype=float)
    orats = events["move_orats"].to_numpy(dtype=float)
    pg = events["move_pg"].to_numpy(dtype=float)

    n = len(events)
    verdict = np.full(n, "", dtype=object)
    for i in range(n):
        vals = {"oquants": oq[i], "orats": orats[i], "polygon": pg[i]}
        have = {k: v for k, v in vals.items() if np.isfinite(v)}
        if len(have) < 2:
            verdict[i] = "insufficient"
            continue
        keys = list(have)
        pairs = [
            (keys[a], keys[b])
            for a in range(len(keys)) for b in range(a + 1, len(keys))
            if abs(have[keys[a]] - have[keys[b]]) <= TOL
        ]
        if not pairs:
            verdict[i] = "ambiguous"  # no two sources agree: definition/data problem
            continue
        agreed = set()
        for a, b in pairs:
            agreed |= {a, b}
        outliers = [k for k in keys if k not in agreed]
        if not outliers:
            verdict[i] = "all_agree"
        elif len(outliers) == 1:
            verdict[i] = f"outlier:{outliers[0]}"
        else:
            verdict[i] = "ambiguous"
    out = events.copy()
    out["verdict"] = verdict
    return out


def main() -> None:
    sample = json.loads((HERE / "results" / "stage1_sample.json").read_text())
    tickers = sample["tickers"]
    ev = pd.read_parquet(HERE / "results" / "stage0_events.parquet")
    ev = ev[ev["ticker"].isin(set(tickers))].copy()
    log(f"sample events: {len(ev):,} on {ev['ticker'].nunique()} tickers")

    series, status = load_polygon_series(tickers)
    report["polygon_coverage"] = {
        k: int(sum(1 for v in status.values() if v == k))
        for k in sorted(set(status.values()))
    }

    ev = polygon_moves(ev, series)
    report["polygon_reasons"] = {
        str(k): int(v) for k, v in ev["reason_pg"].value_counts().items() if k
    }

    scored = ev[np.isfinite(ev["move_pg"])].copy()
    report["events_with_polygon_move"] = int(len(scored))

    ev = adjudicate(ev)
    scored = ev[np.isfinite(ev["move_pg"])].copy()
    report["verdict_counts"] = {
        str(k): int(v) for k, v in scored["verdict"].value_counts().items()
    }
    n_scored = len(scored)
    oq_out = scored["verdict"] == "outlier:oquants"
    orats_out = scored["verdict"] == "outlier:orats"
    pg_out = scored["verdict"] == "outlier:polygon"
    ambig = scored["verdict"] == "ambiguous"
    all_ok = scored["verdict"] == "all_agree"
    report["rates_pct"] = {
        "all_agree": round(all_ok.mean() * 100, 3),
        "outlier_oquants": round(oq_out.mean() * 100, 3),
        "outlier_orats": round(orats_out.mean() * 100, 3),
        "outlier_polygon": round(pg_out.mean() * 100, 3),
        "ambiguous": round(ambig.mean() * 100, 3),
    }

    # the headline acceptance number: oquants vs the independent consensus
    indep_consensus = scored[
        np.isfinite(scored["move_pg"]) & np.isfinite(scored["move_orats"])
        & ((scored["move_pg"] - scored["move_orats"]).abs() <= TOL)
    ]
    if len(indep_consensus):
        dev = (indep_consensus["oq_move"] - indep_consensus["move_pg"]).abs()
        report["oquants_vs_independent_consensus"] = {
            "n": int(len(indep_consensus)),
            "within_0.5pp_pct": round(float((dev <= TOL).mean()) * 100, 3),
            "within_1pp_pct": round(float((dev <= 1.0).mean()) * 100, 3),
            "n_oquants_deviates": int((dev > TOL).sum()),
        }

    # stratified disagreement for the report
    scored["year"] = scored["date"].dt.year

    def decade(y):
        if y <= 2009:
            return "2007-2009"
        if y <= 2014:
            return "2010-2014"
        if y <= 2019:
            return "2015-2019"
        return "2020-2026"

    scored["decade"] = scored["year"].map(decade)
    strat = {}
    for key, g in scored.groupby(["decade", "session"]):
        strat[f"{key[0]} {key[1]}"] = {
            "n": int(len(g)),
            "all_agree_pct": round(float((g["verdict"] == "all_agree").mean()) * 100, 2),
            "oquants_outlier_pct": round(float((g["verdict"] == "outlier:oquants").mean()) * 100, 2),
            "ambiguous_pct": round(float((g["verdict"] == "ambiguous").mean()) * 100, 2),
        }
    report["stratified"] = strat

    # pairwise agreement matrix (median abs diff + within-tol)
    pair = {}
    for a, b in (("oq_move", "move_orats"), ("oq_move", "move_pg"), ("move_orats", "move_pg")):
        m = np.isfinite(scored[a]) & np.isfinite(scored[b])
        d = (scored.loc[m, a] - scored.loc[m, b]).abs()
        pair[f"{a}_vs_{b}"] = {
            "n": int(m.sum()),
            "median_abs_diff_pp": round(float(d.median()), 4),
            "within_0.5pp_pct": round(float((d <= 0.5).mean()) * 100, 2),
            "gt_5pp_pct": round(float((d > 5).mean()) * 100, 3),
            "corr": round(float(np.corrcoef(scored.loc[m, a], scored.loc[m, b])[0, 1]), 5),
        }
    report["pairwise"] = pair

    scored.to_parquet(HERE / "results" / "stage1_events.parquet")
    (HERE / "results" / "stage1_results.json").write_text(
        json.dumps(report, indent=1, default=str))
    log("done")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
