#!/usr/bin/env python3
"""EXP-117 Stage 0 — session-aware recomputation of the earnings-move target.

Recomputes the realized earnings move for every oquants panel event from the
price series already on disk, under the session rule the program claims oquants
uses (EXP-000 / schemas.CONVENTIONS["realized_move"]):

    BMO: close(t-1) -> close(t)     i.e. last close strictly before t -> first close on/after t
    AMC: close(t)   -> close(t+1)   i.e. last close on/before t -> first close strictly after t

Sources recomputed from (all on disk, zero API calls):
    orats   Tier-2 daily_market.spot (ORATS summaries stockPrice, unadjusted)
    yf_raw  yfinance close_raw (split-adjusted, not dividend-adjusted)
    yf_adj  yfinance close_adj (split + dividend adjusted)

Compares each against oquants realized_moves and reports where the
disagreement lives. Also reproduces the naive all-AMC computation whose
0.7489 correlation the plan cites, to pin down what drove it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/investing-plan")
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data import store  # noqa: E402

OUT = Path("/tmp/exp117/stage0_results.json")
MAX_GAP_CALENDAR_DAYS = 5  # P->Q window wider than this = halt/gap, excluded from headline

report: dict = {"generated_at": pd.Timestamp.now("UTC").isoformat()}


def log(msg: str) -> None:
    print(f"[stage0] {msg}", flush=True)


def load_oquants_events() -> pd.DataFrame:
    started = time.time()
    rows = []
    files = sorted(paths.RAW_OQUANTS_MOVES.glob("moves_*.json"))
    for i, path in enumerate(files):
        if i % 500 == 0:
            log(f"oquants moves {i}/{len(files)} files, {len(rows)} rows, {time.time()-started:.0f}s")
        doc = json.loads(path.read_text())
        ticker = doc.get("ticker") or path.name[len("moves_"):-len(".json")]
        data = doc.get("data") or {}
        dates = data.get("dates") or []
        moves = data.get("realized_moves") or []
        if not dates or len(dates) != len(moves):
            continue
        for k, (d, m) in enumerate(zip(dates, moves)):
            rows.append((ticker, k, d, m))
    out = pd.DataFrame(rows, columns=["ticker", "k", "date", "oq_move"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["oq_abs"] = out["oq_move"].abs()
    log(f"oquants: {len(out):,} events on {out['ticker'].nunique():,} tickers, {time.time()-started:.0f}s")
    return out


def load_sessions() -> pd.DataFrame:
    ev = store.read_table(
        "earnings_events",
        columns=["ticker", "event_date", "session", "session_src"],
    )
    ev = ev.rename(columns={"event_date": "date"})
    log(f"sessions: {len(ev):,} calendar rows, {ev['session'].notna().sum():,} with session")
    return ev


def recompute_for_series(
    events: pd.DataFrame,
    series_by_ticker: dict[str, tuple[np.ndarray, np.ndarray]],
    label: str,
) -> pd.DataFrame:
    """Session-aware move per event from one price-source dict.

    series_by_ticker[ticker] = (dates_ns_sorted, closes). The anchor dates are
    the name's own observed closes: a day the name has no close cannot anchor
    either side of the window.
    """
    started = time.time()
    tickers = events["ticker"].to_numpy()
    dates = events["date"].to_numpy()
    sessions = events["session"].astype(object).where(events["session"].notna(), "").to_numpy()
    n = len(events)
    move = np.full(n, np.nan)
    gap_days = np.full(n, np.nan)
    reason = np.full(n, "", dtype=object)

    done = 0
    for i in range(n):
        if i and i % 50000 == 0:
            log(f"{label}: {i}/{n} events, {time.time()-started:.0f}s")
        sess = sessions[i]
        if sess not in ("BMO", "AMC"):
            reason[i] = "no_session"
            continue
        ser = series_by_ticker.get(tickers[i])
        if ser is None:
            reason[i] = "no_series"
            continue
        sd, sc = ser
        t = dates[i]
        if sess == "BMO":
            j_pre = int(np.searchsorted(sd, t, side="left")) - 1   # last close < t
            j_post = int(np.searchsorted(sd, t, side="left"))      # first close >= t
        else:
            j_pre = int(np.searchsorted(sd, t, side="right")) - 1  # last close <= t
            j_post = int(np.searchsorted(sd, t, side="right"))     # first close > t
        if j_pre < 0 or j_post >= len(sd):
            reason[i] = "missing_close"
            continue
        p, q = sc[j_pre], sc[j_post]
        if not np.isfinite(p) or not np.isfinite(q) or p <= 0:
            reason[i] = "bad_price"
            continue
        gap = (sd[j_post] - sd[j_pre]) / np.timedelta64(1, "D")
        gap_days[i] = gap
        if gap > MAX_GAP_CALENDAR_DAYS:
            reason[i] = "wide_gap"
            continue
        move[i] = (q / p - 1.0) * 100.0
        done += 1

    out = events.copy()
    out[f"move_{label}"] = move
    out[f"gap_days_{label}"] = gap_days
    out[f"reason_{label}"] = reason
    log(f"{label}: recomputed {done:,}/{n} events, {time.time()-started:.0f}s")
    return out


def load_daily_market_spot() -> dict:
    started = time.time()
    dm = store.read_table("daily_market", columns=["ticker", "date", "spot"])
    dm = dm[dm["spot"].notna()]
    log(f"daily_market: {len(dm):,} spot rows, {time.time()-started:.0f}s")
    out = {}
    started = time.time()
    for i, (tk, g) in enumerate(dm.groupby("ticker", sort=False)):
        if i % 500 == 0:
            log(f"daily_market group {i}, {time.time()-started:.0f}s")
        g = g.sort_values("date")
        out[str(tk)] = (g["date"].to_numpy(), g["spot"].to_numpy(dtype=float))
    log(f"daily_market: {len(out):,} ticker series built, {time.time()-started:.0f}s")
    return out


def load_yfinance() -> tuple[dict, dict]:
    started = time.time()
    raw, adj = {}, {}
    files = sorted(paths.RAW_YF.glob("px_*.csv"))
    for i, path in enumerate(files):
        if i % 500 == 0:
            log(f"yfinance {i}/{len(files)} files, {time.time()-started:.0f}s")
        ticker = path.name[len("px_"):-len(".csv")]
        try:
            px = pd.read_csv(path, parse_dates=["date"])
        except (ValueError, OSError):
            continue
        px = px.sort_values("date")
        d = px["date"].to_numpy()
        if "close_raw" in px.columns:
            v = px["close_raw"].to_numpy(dtype=float)
            ok = np.isfinite(v) & (v > 0)
            if ok.any():
                raw[ticker] = (d[ok], v[ok])
        if "close_adj" in px.columns:
            v = px["close_adj"].to_numpy(dtype=float)
            ok = np.isfinite(v) & (v > 0)
            if ok.any():
                adj[ticker] = (d[ok], v[ok])
    log(f"yfinance: {len(raw):,} raw / {len(adj):,} adj series, {time.time()-started:.0f}s")
    return raw, adj


def compare(events: pd.DataFrame, col: str, tag: str) -> dict:
    """oquants vs one recomputed column, on their common finite ground."""
    both = events[np.isfinite(events["oq_move"]) & np.isfinite(events[col])].copy()
    d = both[col] - both["oq_move"]
    ad = d.abs()
    rel = ad / both["oq_abs"].clip(lower=0.5)  # relative to a floor, small moves are noisy in ratio
    corr = float(np.corrcoef(both["oq_move"], both[col])[0, 1]) if len(both) > 2 else float("nan")
    out = {
        "n": int(len(both)),
        "corr_signed": round(corr, 6),
        "corr_abs": float(np.corrcoef(both["oq_abs"], both[col].abs())[0, 1]) if len(both) > 2 else None,
        "median_abs_diff_pp": round(float(ad.median()), 4),
        "mean_abs_diff_pp": round(float(ad.mean()), 4),
        "pct_within_0.1pp": round(float((ad <= 0.1).mean()) * 100, 2),
        "pct_within_0.5pp": round(float((ad <= 0.5).mean()) * 100, 2),
        "pct_within_1pp": round(float((ad <= 1.0).mean()) * 100, 2),
        "pct_diff_gt_5pp": round(float((ad > 5.0).mean()) * 100, 3),
        "median_ratio_recomputed_over_oq": round(float((both[col].abs() / both["oq_abs"].clip(lower=0.01)).median()), 4),
        "median_oq_abs": round(float(both["oq_abs"].median()), 4),
        "median_recomputed_abs": round(float(both[col].abs().median()), 4),
    }
    log(f"compare {tag}: n={out['n']:,} corr={corr:.4f} within0.5pp={out['pct_within_0.5pp']}%")
    return out


def naive_all_amc(events: pd.DataFrame, series_by_ticker: dict) -> pd.DataFrame:
    """The v1.0 reproduction: close(event date) -> next close, one rule for all."""
    started = time.time()
    n = len(events)
    move = np.full(n, np.nan)
    tickers = events["ticker"].to_numpy()
    dates = events["date"].to_numpy()
    for i in range(n):
        ser = series_by_ticker.get(tickers[i])
        if ser is None:
            continue
        sd, sc = ser
        t = dates[i]
        j0 = int(np.searchsorted(sd, t, side="right")) - 1  # close on/after... no: last close <= t
        j1 = j0 + 1
        if j0 < 0 or j1 >= len(sd):
            continue
        p, q = sc[j0], sc[j1]
        if np.isfinite(p) and np.isfinite(q) and p > 0:
            move[i] = (q / p - 1.0) * 100.0
    out = events.copy()
    out["move_naive"] = move
    log(f"naive all-AMC done, {np.isfinite(move).sum():,} events, {time.time()-started:.0f}s")
    return out


def main() -> None:
    events = load_oquants_events()
    sessions = load_sessions()
    events = events.merge(
        sessions[["ticker", "date", "session", "session_src"]],
        on=["ticker", "date"], how="left",
    )
    report["events_total"] = int(len(events))
    report["session_counts"] = {str(k): int(v) for k, v in events["session"].value_counts(dropna=False).items()}

    report["reason_counts"] = {}

    log("loading daily_market spot ...")
    orats_spot = load_daily_market_spot()
    events = recompute_for_series(events, orats_spot, "orats")
    report["reason_counts"]["orats"] = {
        str(k): int(v) for k, v in events["reason_orats"].value_counts().items() if k
    }

    log("loading yfinance series ...")
    yf_raw, yf_adj = load_yfinance()
    events = recompute_for_series(events, yf_raw, "yf_raw")
    events = recompute_for_series(events, yf_adj, "yf_adj")
    report["reason_counts"]["yf_raw"] = {
        str(k): int(v) for k, v in events["reason_yf_raw"].value_counts().items() if k
    }

    events = naive_all_amc(events, orats_spot)

    for col, tag in (
        ("move_orats", "session-aware vs oquants (ORATS spot)"),
        ("move_yf_raw", "session-aware vs oquants (yfinance raw)"),
        ("move_yf_adj", "session-aware vs oquants (yfinance adj)"),
        ("move_naive", "NAIVE all-AMC vs oquants (ORATS spot)"),
    ):
        report[f"compare_{col}"] = compare(events, col, tag)

    # stratify the session-aware ORATS comparison by session and decade
    both = events[np.isfinite(events["oq_move"]) & np.isfinite(events["move_orats"])].copy()
    both["decade"] = (both["date"].dt.year // 10) * 10
    strat = {}
    for key, g in both.groupby(["session", "decade"]):
        ad = (g["move_orats"] - g["oq_move"]).abs()
        strat[f"{key[0]} {key[1]}s"] = {
            "n": int(len(g)),
            "pct_within_0.5pp": round(float((ad <= 0.5).mean()) * 100, 2),
            "pct_diff_gt_5pp": round(float((ad > 5.0).mean()) * 100, 3),
            "median_abs_diff": round(float(ad.median()), 4),
        }
    report["stratified_orats"] = strat

    # big disagreements: where does oquants differ from all three recomputations?
    big = both[
        ((both["move_orats"] - both["oq_move"]).abs() > 5.0)
    ].copy()
    report["n_big_disagreements_orats"] = int(len(big))
    big["decade"] = (big["date"].dt.year // 10) * 10
    report["big_disagreements_by_decade"] = {
        str(k): int(v) for k, v in big["decade"].value_counts().sort_index().items()
    }
    report["big_disagreements_by_session"] = {
        str(k): int(v) for k, v in big["session"].value_counts(dropna=False).items()
    }

    # split candidates: recomputed move beyond 45% (rare for earnings)
    for col in ("move_orats", "move_yf_raw"):
        mask = events[col].abs() > 45
        n_cand = int(mask.sum())
        if n_cand:
            sub = events[mask]
            agree = (sub["oq_move"] - sub[col]).abs() <= 5
            report[f"split_candidates_{col}"] = {
                "n": n_cand,
                "oquants_also_big": int(agree.sum()),
                "oquants_small": int((~agree).sum()),
            }

    events.to_parquet("/tmp/exp117/stage0_events.parquet")
    OUT.write_text(json.dumps(report, indent=1, default=str))
    log(f"wrote {OUT}")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
