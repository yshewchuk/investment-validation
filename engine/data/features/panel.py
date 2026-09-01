"""Tier 3 — the causal event panel.

Rebuilds the master panel that every current verdict rests on, from Tier 2,
with the same feature definitions as the pre-engine research so the migration
test can reconcile the two row for row.

**Leak discipline is the whole point of this module.** For event ``k`` of a
ticker, every feature uses only information available strictly before the
print:

* history features use events ``0..k-1`` only — the EMA recursion is seeded on
  the first event and stepped forward, so at event ``k`` it reflects nothing
  from ``k`` onward;
* price and market-state features are read at the last close *strictly before*
  the event date (``searchsorted(..., "left") - 1``);
* nothing is normalized cross-sectionally or on full-sample statistics.

Four feature blocks, matching the four legacy build stages:

======================  ==================================================
block                    source
======================  ==================================================
``events``               oquants moves (realized + implied move history)
``regime``               S&P 500 daily series (market state)
``runup``                yfinance daily closes (distance/return features)
``orats``                Tier-2 ``daily_market``
======================  ==================================================

One deliberate divergence from the legacy build: market cap. The legacy
``or_mcap_log`` applied a single ×1e6 to every pre-2026-03-11 row, which is
wrong for the billions era before 2017-06-28. This panel carries the corrected
``mcap_log``, and the migration test reports the difference as a known, expected
delta rather than papering over it.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.data import store

__all__ = [
    "SPANS",
    "MIN_HISTORY",
    "ORATS_FEATURES",
    "history_features",
    "build_events",
    "add_regime_features",
    "add_runup_features",
    "add_orats_features",
    "build_panel",
    "PANEL_COLUMNS",
]

SPANS = (2, 4, 8, 12)
#: An event is admitted only once the ticker has this many prior events, so the
#: history features are never computed from a near-empty window.
MIN_HISTORY = 4

#: ``tier-2 daily_market column -> panel column``. The Tier-2 normalizer already
#: applies the ×100 vol-point convention, so no multiplier is needed here.
ORATS_FEATURES = {
    "implied_move": "or_implied",
    "skew": "or_skewing",
    "contango": "or_contango",
    "rvol30": "or_rvol30",
    "exern_iv30": "or_exern30",
    "iv30": "or_iv30",
    "iee": "or_iee",
    "fwd90_30": "or_fwd90_30",
    "fexern90_30": "or_fexern90_30",
}

PANEL_COLUMNS = (
    ["ticker", "k", "date", "quarter", "move", "abs_move", "implied_move", "n_prior",
     "mean_prior_move", "mean_prior_abs_move"]
    + [f"ema{s}_prior_move" for s in SPANS]
    + [f"ema{s}_prior_abs_move" for s in SPANS]
    + ["mean_prior_implied_move", "year",
       "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol20",
       "signed_streak", "ema12r_abs", "dist_high", "dist_ema", "ret5", "ret10", "ret20"]
    + list(ORATS_FEATURES.values())
    + ["or_exern_z252", "mcap_log", "mcap_usd", "mcap_asof"]
)


def _log(message: str) -> None:
    print(f"  [panel] {message}", flush=True)


# --------------------------------------------------------------------------
# block 1 — the causal event panel from oquants moves
# --------------------------------------------------------------------------


def _causal_ema(history: list[float], span: int) -> float | None:
    """EMA over events strictly before the current one.

    Reproduces the legacy recursion exactly: seed on the first prior event and
    step forward with ``a = 2/(span+1)``, returning None until there are at
    least ``span`` prior events. ``pandas.ewm`` is *not* substitutable here —
    it differs in seeding and in the ``adjust=True`` weighting, and the panel's
    stored values follow this recursion.
    """
    if len(history) < span:
        return None
    a = 2.0 / (span + 1.0)
    ema = history[0]
    for value in history[1:]:
        ema = a * value + (1.0 - a) * ema
    return ema


def history_features(
    prior_moves: Sequence[float],
    prior_abs: Sequence[float],
    prior_implied: Sequence[float | None],
) -> dict[str, float | None]:
    """Event-history features for the event that follows the given history.

    The single definition of the panel's history block. :func:`build_events`
    calls it while walking a ticker's events, and ``engine.features`` calls it
    with a ticker's realized history to produce the same features for an event
    that has not happened yet. Two implementations of this recursion would drift
    the moment either was touched, and the live scorer diverging from the
    research panel is precisely the failure Phase 1's replay test exists to
    catch — so there is one.

    ``prior_*`` must contain events strictly before the one being scored.
    """
    known_implied = [x for x in prior_implied if x is not None and not pd.isna(x)]
    out: dict[str, float | None] = {
        "n_prior": len(prior_moves),
        "mean_prior_move": float(np.mean(prior_moves)) if len(prior_moves) else None,
        "mean_prior_abs_move": float(np.mean(prior_abs)) if len(prior_abs) else None,
        "mean_prior_implied_move": float(np.mean(known_implied)) if known_implied else None,
    }
    for span in SPANS:
        out[f"ema{span}_prior_move"] = _causal_ema(list(prior_moves), span)
        out[f"ema{span}_prior_abs_move"] = _causal_ema(list(prior_abs), span)
    return out


def build_events(moves_dir: Path | None = None,
                 extra_moves_dirs: Sequence[Path] = ()) -> pd.DataFrame:
    """The base causal panel: one row per admitted (ticker, event).

    ``extra_moves_dirs`` hold synthesized oquants-format files — the EXP-117
    universe extension for tickers oquants does not carry (target provenance:
    COMPUTED, see engine/data/pulls/computed_moves.py). A ticker present in
    the primary dir is never shadowed by a synthesized one.
    """
    moves_dir = moves_dir or paths.RAW_OQUANTS_MOVES
    files = sorted(moves_dir.glob("moves_*.json"))
    if extra_moves_dirs:
        seen = {p.name for p in files}
        for extra in extra_moves_dirs:
            extra_path = Path(extra)
            if not extra_path.exists():
                continue
            files += sorted(p for p in extra_path.glob("moves_*.json")
                            if p.name not in seen)
    if not files:
        raise FileNotFoundError(f"no oquants moves files under {moves_dir}")

    rows: list[dict] = []
    skipped_empty = 0
    started = time.time()
    for i, path in enumerate(files):
        if i % 500 == 0:
            _log(f"events {i}/{len(files)} files, {len(rows)} rows, {time.time()-started:.0f}s")
        doc = json.loads(path.read_text())
        ticker = doc.get("ticker") or path.name[len("moves_") : -len(".json")]
        data = doc.get("data") or {}
        dates = data.get("dates") or []
        if not dates:
            skipped_empty += 1
            continue
        moves = data.get("realized_moves") or []
        abs_moves = data.get("abs_realized_moves") or []
        implied = data.get("implied_moves") or []
        quarters = data.get("quarters") or []
        n = len(dates)
        if not (len(moves) == len(abs_moves) == len(implied) == len(quarters) == n):
            _log(f"SKIP {ticker}: ragged arrays")
            continue

        prior_moves: list[float] = []
        prior_abs: list[float] = []
        prior_implied: list = []
        for k in range(n):
            if k >= MIN_HISTORY:
                row = {
                    "ticker": ticker,
                    "k": k,
                    "date": dates[k],
                    "quarter": quarters[k],
                    "move": moves[k],
                    "abs_move": abs_moves[k],
                    "implied_move": implied[k],
                    "year": int(str(dates[k])[:4]),
                    **history_features(prior_moves, prior_abs, prior_implied),
                }
                rows.append(row)
            prior_moves.append(moves[k])
            prior_abs.append(abs_moves[k])
            prior_implied.append(implied[k])

    _log(f"events: {len(rows)} rows from {len(files)} files ({skipped_empty} empty)")
    if not rows:
        # Shape matters even when empty: a caller that goes on to add regime and
        # run-up features must not hit a KeyError instead of an empty result.
        columns = ["ticker", "k", "date", "quarter", "move", "abs_move", "implied_move",
                   "n_prior", "mean_prior_move", "mean_prior_abs_move",
                   "mean_prior_implied_move", "year"]
        columns += [f"ema{s}_prior_move" for s in SPANS]
        columns += [f"ema{s}_prior_abs_move" for s in SPANS]
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
        empty["ticker"] = pd.Series(dtype="object")
        empty["date"] = pd.Series(dtype="datetime64[ns]")
        return empty
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# block 2 — market regime at the last pre-event close
# --------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _gspc_series(path_str: str, mtime: float) -> pd.DataFrame:
    """The S&P daily series, read once per process.

    The panel build calls this once, but ``engine.features.live_features`` calls
    it per scored event — hundreds of times for one dashboard refresh — and
    re-parsing a twenty-year daily CSV each time is pure waste. Keyed on mtime so
    a refreshed file is picked up rather than served stale.
    """
    raw = pd.read_csv(path_str, skiprows=3, header=None)
    raw.columns = ["date", "adj", "close", "high", "low", "open", "volume"][: raw.shape[1]]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    return raw.dropna(subset=["date"]).sort_values("date")


def add_regime_features(df: pd.DataFrame, gspc_path: Path | None = None) -> pd.DataFrame:
    """S&P 500 state as of the last close strictly before each event."""
    gspc_path = Path(gspc_path or paths.GSPC_DAILY)
    raw = _gspc_series(str(gspc_path), gspc_path.stat().st_mtime)

    closes = raw["close"].to_numpy(dtype=float)
    dates = raw["date"].to_numpy()
    # SIMPLE returns with ddof=1 — i.e. `pct_change().rolling(20).std()`. This
    # is the legacy definition, recovered by search and reproduced exactly; log
    # returns or ddof=0 shift every value by ~2.6%. Worth pinning down rather
    # than approximating: `spy_vol20` is one of the nine champion-model features.
    simple_ret = closes[1:] / closes[:-1] - 1.0

    out = df.copy()
    n = len(out)
    cols = {c: np.full(n, np.nan) for c in
            ("spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol20")}

    event_dates = out["date"].to_numpy()
    idx = np.searchsorted(dates, event_dates, side="left") - 1
    for i, j in enumerate(idx):
        if j < 0 or j >= len(closes):
            continue
        spot = closes[j]
        if j >= 21:
            cols["spy_ret21"][i] = (spot / closes[j - 21] - 1.0) * 100
        if j >= 63:
            cols["spy_ret63"][i] = (spot / closes[j - 63] - 1.0) * 100
        if j >= 252:
            cols["spy_ret252"][i] = (spot / closes[j - 252] - 1.0) * 100
            # 252 observations ending at j (j-251 .. j inclusive).
            cols["spy_dd252"][i] = (spot / closes[j - 251 : j + 1].max() - 1.0) * 100
        if j >= 20:
            cols["spy_vol20"][i] = simple_ret[j - 20 : j].std(ddof=1) * np.sqrt(252) * 100
    for name, values in cols.items():
        out[name] = values
    _log(f"regime: {int(np.isfinite(cols['spy_vol20']).sum())}/{n} events with market state")
    return out


# --------------------------------------------------------------------------
# block 3 — run-up / distance features from the ticker's own price history
# --------------------------------------------------------------------------


def _yf_history_from_tier1(ticker: str) -> pd.DataFrame | None:
    """Price history from the Tier-1 yfinance cache, for tickers the legacy
    ``px_{T}.csv`` tree does not carry (the EXP-117 universe extension).

    Same series the synthesized moves were computed from — split-adjusted
    Close, validated exact against Polygon in EXP-117 — shaped like a legacy
    px frame (``date``, ``close_adj``). Dividends are not adjusted; that only
    matters for the run-up features of these tickers, never for a price.
    """
    import gzip
    import io

    from engine.data.fetch import Fetcher, cache_key

    try:
        fetcher = Fetcher()
        key = cache_key("yfinance", "history", {"ticker": ticker, "period": "max"})
        path = fetcher.body_path("yfinance", key)
        if not path.exists():
            return None
        with gzip.open(path, "rb") as fh:
            frame = pd.read_csv(io.BytesIO(fh.read()))
    except (OSError, ValueError, EOFError):
        return None
    if frame.empty or "Close" not in frame.columns:
        return None
    dates = pd.to_datetime(frame[frame.columns[0]], errors="coerce", utc=True)
    dates = dates.dt.tz_localize(None)
    closes = pd.to_numeric(frame["Close"], errors="coerce")
    out = pd.DataFrame({"date": dates, "close_adj": closes})
    out = out.dropna().sort_values("date")
    return out if len(out) else None


def add_runup_features(df: pd.DataFrame, px_dir: Path | None = None) -> pd.DataFrame:
    """Streak, distance-from-extreme, and short-horizon return features."""
    px_dir = px_dir or paths.RAW_YF
    out = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # signed_streak: length of the run of same-signed moves that ended strictly
    # before this event, signed by that run's direction.
    signs = np.sign(out["move"].to_numpy(dtype=float))
    tickers = out["ticker"].to_numpy()
    n = len(out)
    length_before = np.zeros(n, dtype=int)
    sign_before = np.zeros(n, dtype=int)
    run, prev_sign = 0, 0
    for i in range(n):
        if i == 0 or tickers[i] != tickers[i - 1]:
            length_before[i], sign_before[i] = 0, 0
            run, prev_sign = 1, signs[i]
            continue
        length_before[i], sign_before[i] = run, prev_sign
        if signs[i] == prev_sign and signs[i] != 0:
            run += 1
        else:
            run, prev_sign = 1, signs[i]
    out["signed_streak"] = length_before * sign_before

    # The 12-span EMA of |move| where enough history exists, else the mean.
    out["ema12r_abs"] = out["ema12_prior_abs_move"].where(
        out["n_prior"] >= 12, out["mean_prior_abs_move"]
    )

    for col in ("dist_high", "dist_ema", "ret5", "ret10", "ret20"):
        out[col] = np.nan

    groups = list(out.groupby("ticker", sort=True))
    started = time.time()
    covered = 0
    for gi, (ticker, group) in enumerate(groups):
        if gi % 500 == 0:
            _log(f"runup {gi}/{len(groups)} tickers, {time.time()-started:.0f}s")
        path = px_dir / f"px_{ticker}.csv"
        px = None
        if path.exists() and path.stat().st_size >= 50:
            try:
                px = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
            except (ValueError, OSError):
                px = None
        if px is None or len(px) < 300 or "close_adj" not in px.columns:
            px = _yf_history_from_tier1(ticker)
        if px is None or len(px) < 300 or "close_adj" not in px.columns:
            continue
        closes = px["close_adj"].to_numpy(dtype=float)
        pdates = px["date"].to_numpy()
        ema252 = pd.Series(closes).ewm(span=252, adjust=False).mean().to_numpy()
        high252 = pd.Series(closes).rolling(252, min_periods=120).max().to_numpy()

        size = len(group)
        dist_high = np.full(size, np.nan)
        dist_ema = np.full(size, np.nan)
        r5 = np.full(size, np.nan)
        r10 = np.full(size, np.nan)
        r20 = np.full(size, np.nan)
        for j, event_date in enumerate(group["date"].to_numpy()):
            idx = int(np.searchsorted(pdates, event_date, side="left")) - 1
            if idx < 0:
                continue
            if idx >= 252 and np.isfinite(high252[idx]) and np.isfinite(ema252[idx]) and ema252[idx] > 0:
                dist_high[j] = (closes[idx] / high252[idx] - 1.0) * 100
                dist_ema[j] = (closes[idx] / ema252[idx] - 1.0) * 100
            if idx >= 20 and closes[idx - 20] > 0:
                r20[j] = (closes[idx] / closes[idx - 20] - 1.0) * 100
                r10[j] = (closes[idx] / closes[idx - 10] - 1.0) * 100
                r5[j] = (closes[idx] / closes[idx - 5] - 1.0) * 100
        out.loc[group.index, "dist_high"] = dist_high
        out.loc[group.index, "dist_ema"] = dist_ema
        out.loc[group.index, "ret5"] = r5
        out.loc[group.index, "ret10"] = r10
        out.loc[group.index, "ret20"] = r20
        covered += 1

    _log(f"runup: price history found for {covered}/{len(groups)} tickers")
    return out


# --------------------------------------------------------------------------
# block 4 — ORATS state at the last pre-event close, from Tier 2
# --------------------------------------------------------------------------


def add_orats_features(df: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join Tier-2 ``daily_market`` state as of the last row before each event.

    The as-of rule is ``searchsorted(dates, event_date, "left") - 1``: the last
    EOD row *strictly before* the event date. For a BMO print that is the
    previous close, which is exactly right. For an AMC print the event-date
    close is also pre-print and is therefore admissible, but the legacy panel
    used the prior close for both, so this reproduces that — a deliberate
    conservatism (never a leak; at worst one session stale on AMC names) that
    Phase 1 can revisit with the session-aware anchors in ``engine.calendar``.
    """
    if daily is None:
        needed = ["ticker", "date", "mcap_usd", "mcap_log", "src_iv", *ORATS_FEATURES.keys()]
        _log("reading tier-2 daily_market …")
        daily = store.read_table("daily_market", columns=needed)
    _log(f"daily_market: {len(daily):,} rows")

    out = df.copy()
    n = len(out)
    targets = {panel_col: np.full(n, np.nan) for panel_col in ORATS_FEATURES.values()}
    targets["or_exern_z252"] = np.full(n, np.nan)
    targets["mcap_log"] = np.full(n, np.nan)
    targets["mcap_usd"] = np.full(n, np.nan)
    # The date the market cap was actually observed. Provenance worth carrying:
    # it says how stale the figure is, and it is what decides which ORATS unit
    # era the raw value came from — the event date is the wrong key for that
    # when an event sits on or just after an era boundary.
    mcap_asof = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    positions: dict[str, list[tuple[int, np.datetime64]]] = {}
    for i, (ticker, event_date) in enumerate(
        zip(out["ticker"].to_numpy(), out["date"].to_numpy())
    ):
        positions.setdefault(ticker, []).append((i, event_date))

    daily = daily.sort_values(["ticker", "date"])
    known_tickers = set(daily["ticker"].unique())
    started = time.time()
    no_prior = 0
    missing_ticker = sum(
        len(rows) for ticker, rows in positions.items() if ticker not in known_tickers
    )
    for gi, (ticker, group) in enumerate(daily.groupby("ticker", sort=True)):
        if ticker not in positions:
            continue
        if gi % 500 == 0:
            _log(f"orats {gi} tickers, {time.time()-started:.0f}s")
        # The IV block resolves over rows that came from ORATS summaries. Rows
        # contributed only by cores (a market-cap observation on a date
        # summaries has no row for) carry no IV, and must not become the
        # as-of answer for an IV feature.
        iv_rows = (
            group[group["src_iv"].notna()] if "src_iv" in group.columns else group
        )
        if iv_rows.empty:
            iv_rows = group.iloc[0:0]
        dates = iv_rows["date"].to_numpy()
        columns = {c: iv_rows[c].to_numpy(dtype=float) for c in ORATS_FEATURES}
        exern = columns["exern_iv30"]

        # Market cap gets its own as-of index, over the rows that actually have
        # one. The IV block comes from ORATS summaries and the cap from ORATS
        # cores, and the two series do not share every date: tying the cap to
        # the summaries observation would return a stale figure whenever
        # summaries has a gap, or none at all where it has no row. "Most recent
        # cap available before the print" is both the legacy behaviour and the
        # right one.
        has_mcap = np.isfinite(group["mcap_usd"].to_numpy(dtype=float))
        mcap_dates = group["date"].to_numpy()[has_mcap]
        mcap_log_vals = group["mcap_log"].to_numpy(dtype=float)[has_mcap]
        mcap_usd_vals = group["mcap_usd"].to_numpy(dtype=float)[has_mcap]

        for i, event_date in positions[ticker]:
            j = int(np.searchsorted(dates, event_date, side="left")) - 1 if len(dates) else -1
            if j < 0:
                no_prior += 1
            else:
                for src_col, panel_col in ORATS_FEATURES.items():
                    targets[panel_col][i] = columns[src_col][j]
            if len(mcap_dates):
                jm = int(np.searchsorted(mcap_dates, event_date, side="left")) - 1
                if jm >= 0:
                    targets["mcap_log"][i] = mcap_log_vals[jm]
                    targets["mcap_usd"][i] = mcap_usd_vals[jm]
                    mcap_asof[i] = mcap_dates[jm]
            # z-score of ex-earnings IV against its own trailing 252 sessions,
            # strictly before j — a within-name standardization, so it cannot
            # leak cross-sectional information from the future.
            window = exern[max(0, j - 252) : j]
            window = window[np.isfinite(window)]
            if len(window) >= 60 and np.isfinite(exern[j]):
                std = window.std()
                if std > 0:
                    targets["or_exern_z252"][i] = (exern[j] - window.mean()) / std

    for name, values in targets.items():
        out[name] = values
    out["mcap_asof"] = mcap_asof
    coverage = float(np.isfinite(targets["or_implied"]).mean())
    _log(
        f"orats: implied-move coverage {coverage:.3f}; "
        f"{no_prior} events with no prior daily row; {missing_ticker} events on unknown tickers"
    )
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def build_panel(daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full Tier-3 causal panel, built from Tier 1 (moves/prices) and Tier 2."""
    _log("block 1/4 — causal events from oquants moves")
    panel = build_events(extra_moves_dirs=(paths.COMPUTED_MOVES,))
    _log("block 2/4 — market regime")
    panel = add_regime_features(panel)
    _log("block 3/4 — run-up and distance features")
    panel = add_runup_features(panel)
    _log("block 4/4 — ORATS state from tier 2")
    panel = add_orats_features(panel, daily=daily)

    ordered = [c for c in PANEL_COLUMNS if c in panel.columns]
    extra = [c for c in panel.columns if c not in ordered]
    panel = panel[ordered + extra].sort_values(["ticker", "date"]).reset_index(drop=True)
    _log(f"panel complete: {len(panel):,} rows × {len(panel.columns)} columns")
    return panel
