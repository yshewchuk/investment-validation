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
from engine.calendar import AMC
from engine.data import store

__all__ = [
    "SPANS",
    "MIN_HISTORY",
    "ORATS_FEATURES",
    "ANCHOR_COLUMNS",
    "history_features",
    "build_events",
    "add_regime_features",
    "add_runup_features",
    "add_orats_features",
    "add_pre_print_vol",
    "add_implied_history",
    "PRE_PRINT_FEATURES",
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

#: Tier-2 vol terms read at the SESSION-AWARE pre-print close. Defined here
#: because :data:`PANEL_COLUMNS` names them; the builder is below.
PRE_PRINT_FEATURES = {
    "iv30": "pre_iv30",
    "iv10": "pre_iv10",
    "exern_iv30": "pre_exern_iv30",
    "exern_iv10": "pre_exern_iv10",
}

PANEL_COLUMNS = (
    ["ticker", "k", "date", "quarter", "move", "abs_move", "n_prior",
     "mean_prior_move", "mean_prior_abs_move"]
    + [f"ema{s}_prior_move" for s in SPANS]
    + [f"ema{s}_prior_abs_move" for s in SPANS]
    + ["mean_prior_or_implied", "year",
       "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol20",
       "signed_streak", "ema12r_abs", "dist_high", "dist_ema", "ret5", "ret10", "ret20"]
    + list(ORATS_FEATURES.values())
    + ["or_exern_z252", "mcap_log", "mcap_usd", "mcap_asof"]
    + list(PRE_PRINT_FEATURES.values())
)

#: Per-block observation dates written by the three market-state builders: the
#: row each block was actually read at, given whatever ``as_of_column`` the
#: caller anchored on. They are provenance for the live path's stamps, not
#: features, and :func:`build_panel` drops them (see there for why).
ANCHOR_COLUMNS = ("regime_asof", "runup_asof", "orats_asof")


def _anchor_index(
    series_dates: np.ndarray,
    event_dates: np.ndarray,
    as_of_dates: np.ndarray | None,
) -> np.ndarray:
    """Row index each market block is read at: the earlier of two ceilings.

    **Strictly before the event date** — the panel's rule. For a BMO print that
    is the last pre-print close exactly; for an AMC print the event-date close
    would also be admissible, but the legacy panel used the prior close for both
    and every stored number follows that. It is a modelling choice with its own
    experiment to justify changing it, so it stays a hard ceiling here.

    **On or before the decision date** — ``daily_state_frame``'s rule, and the
    right one for an as-of that is a close we would trade at: that close's own
    quotes are known to us then.

    Taking the ``min`` composes them. A decision at or after the panel's anchor
    resolves to the panel's anchor, so every existing value is unchanged and the
    AMC conservatism survives; a decision *earlier* than it resolves to the
    decision's own row, which is the whole point. Using the decision's ceiling
    alone would silently push BMO names one session staler than the panel and
    break the panel/live equivalence check.
    """
    event_idx = np.searchsorted(series_dates, event_dates, side="left") - 1
    if as_of_dates is None:
        return event_idx
    as_of_idx = np.searchsorted(series_dates, as_of_dates, side="right") - 1
    return np.minimum(event_idx, as_of_idx)


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

    The implied-move history is NOT here. It averages ``or_implied``, which a
    LATER block produces, so it cannot be computed from the moves files this
    one reads — see :func:`add_implied_history`.
    """
    out: dict[str, float | None] = {
        "n_prior": len(prior_moves),
        "mean_prior_move": float(np.mean(prior_moves)) if len(prior_moves) else None,
        "mean_prior_abs_move": float(np.mean(prior_abs)) if len(prior_abs) else None,
    }
    for span in SPANS:
        out[f"ema{span}_prior_move"] = _causal_ema(list(prior_moves), span)
        out[f"ema{span}_prior_abs_move"] = _causal_ema(list(prior_abs), span)
    return out


def build_events(moves_dir: Path | None = None,
                 extra_moves_dirs: Sequence[Path] = ()) -> pd.DataFrame:
    """The base causal panel: one row per admitted (ticker, event).

    ``extra_moves_dirs`` hold synthesized oquants-format files — originally the
    EXP-117 universe extension for tickers oquants does not carry (target
    provenance: COMPUTED, see engine/data/pulls/computed_moves.py).

    **Merged per EVENT, not per file.** It used to be per file: a ticker present
    in the primary dir was skipped entirely in the extras. That is right for a
    universe extension and wrong for a FORWARD gap, which is the case that
    matters now — the oquants cache is fetched periodically and has no fetcher
    in this repository, so it lags. On 2026-09-05 it ended at 2026-08-31 while
    Tier 2 held prints through 09-04, and a file-level merge meant no
    synthesized file could supply them for the 1,900 tickers oquants carries.

    **The synthesized source wins outright.** Both ``move``/``abs_move`` and
    ``implied_move`` come from it wherever it has the event; oquants supplies
    only what it does not cover.

    Realized move: session-aware close-to-close on yfinance, validated by
    EXP-117 at 99.5% within 0.5pp against Polygon truth, and found by the
    2026-09-05 arbitration to match oquants to the cent on 92.5% of the events
    where oquants and ORATS spot disagreed. Computing it ourselves removes a
    vendor from the critical path without changing what the number means.

    Implied move: ORATS ``daily_market.implied_move`` at the last pre-print
    close. This is NOT the same quantity oquants quotes — EXP-122 measured
    oquants at E|move| to 3% and ORATS at 1.55x a model-free straddle — so the
    switch shifts ``mean_prior_implied_move`` by a systematic ~+1.95pp across
    the whole panel and every champion must be retrained. EXP-132 is what
    licenses it: the ORATS-derived history was BETTER for iv_crush and
    implied_t1, within noise for both gates, and worse for size_v1_4 by 0.0036
    against its own 0.0033 seed-noise band.

    A per-field split was tried first — computed realized, oquants implied —
    and rejected. It looked conservative and was worse: oquants covers history
    and not the present, so implied_move would have switched series at exactly
    the boundary between the training data and the live board. A discontinuity
    there is harder to reason about than a uniform shift, and it sits where it
    does the most damage.

    The merge is per EVENT rather than per file so a forward gap can be closed
    at all: the oquants cache has no fetcher in this repository and lags.
    """
    moves_dir = moves_dir or paths.RAW_OQUANTS_MOVES
    files = [(p, "oquants") for p in sorted(moves_dir.glob("moves_*.json"))]
    for extra in extra_moves_dirs:
        extra_path = Path(extra)
        if extra_path.exists():
            files += [(p, "computed") for p in sorted(extra_path.glob("moves_*.json"))]
    if not files:
        raise FileNotFoundError(f"no oquants moves files under {moves_dir}")

    started = time.time()
    skipped_empty = 0
    # PASS 1 — merge the per-ticker series, field by field. The history block
    # is NOT built here: `mean_prior_move` and its EMAs are functions of prior
    # REALIZED moves, so recomputing those means the history has to be derived
    # from the merged series rather than from either source's own.
    merged: dict[str, dict[str, dict]] = {}
    for i, (path, origin) in enumerate(files):
        if i % 500 == 0:
            _log(f"events {i}/{len(files)} files, {time.time()-started:.0f}s")
        doc = json.loads(path.read_text())
        ticker = doc.get("ticker") or path.name[len("moves_") : -len(".json")]
        data = doc.get("data") or {}
        dates = data.get("dates") or []
        if not dates:
            skipped_empty += 1
            continue
        moves = data.get("realized_moves") or []
        abs_moves = data.get("abs_realized_moves") or []
        quarters = data.get("quarters") or []
        n = len(dates)
        if not (len(moves) == len(abs_moves) == len(quarters) == n):
            _log(f"SKIP {ticker}: ragged arrays")
            continue
        book = merged.setdefault(ticker, {})
        for k in range(n):
            day = str(dates[k])[:10]
            rec = book.setdefault(day, {"date": dates[k], "quarter": quarters[k]})
            # The synthesized source wins both fields where it has the event.
            if origin == "computed" or "move" not in rec:
                rec["move"], rec["abs_move"] = moves[k], abs_moves[k]
                rec["src"] = origin

    # PASS 2 — history from the merged series, in date order.
    rows: list[dict] = []
    recomputed = 0
    for ticker, book in merged.items():
        prior_moves: list[float] = []
        prior_abs: list[float] = []
        for day in sorted(book):
            rec = book[day]
            k = len(prior_moves)
            if k >= MIN_HISTORY:
                rows.append({
                    "ticker": ticker,
                    "k": k,
                    "date": rec["date"],
                    "quarter": rec.get("quarter"),
                    "move": rec.get("move"),
                    "abs_move": rec.get("abs_move"),
                    "year": int(str(rec["date"])[:4]),
                    **history_features(prior_moves, prior_abs),
                })
                recomputed += rec.get("src") == "computed"
            prior_moves.append(rec.get("move"))
            prior_abs.append(rec.get("abs_move"))

    rows.sort(key=lambda r: (r["ticker"], str(r["date"])))
    _log(f"events: {len(rows)} rows from {len(files)} files ({skipped_empty} empty; "
         f"{recomputed} events from the synthesized source)")
    if not rows:
        # Shape matters even when empty: a caller that goes on to add regime and
        # run-up features must not hit a KeyError instead of an empty result.
        columns = ["ticker", "k", "date", "quarter", "move", "abs_move",
                   "n_prior", "mean_prior_move", "mean_prior_abs_move", "year"]
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


def add_regime_features(
    df: pd.DataFrame,
    gspc_path: Path | None = None,
    as_of_column: str = "date",
) -> pd.DataFrame:
    """S&P 500 state as of the last close strictly before each ``as_of_column``.

    ``as_of_column`` defaults to ``"date"`` — the event date — which is the
    panel's own convention and leaves the Tier-3 build byte-identical. A caller
    scoring a decision taken *earlier* than the last pre-print close must pass
    the column holding that decision date instead: anchoring on the event date
    would hand it market state it could not have seen, and because
    :func:`engine.features._stamps` derives its stamp from the decision date
    independently of the value, ``assert_causal`` would not catch it.

    The row date actually used is returned in ``regime_asof``, so the stamp can
    be the observation's own date rather than an assumption about it.
    """
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

    anchor = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    idx = _anchor_index(
        dates,
        out["date"].to_numpy(),
        out[as_of_column].to_numpy() if as_of_column != "date" else None,
    )
    for i, j in enumerate(idx):
        if j < 0 or j >= len(closes):
            continue
        anchor[i] = dates[j]
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
    out["regime_asof"] = anchor
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


def add_runup_features(
    df: pd.DataFrame,
    px_dir: Path | None = None,
    as_of_column: str = "date",
) -> pd.DataFrame:
    """Streak, distance-from-extreme, and short-horizon return features.

    ``signed_streak`` and ``ema12r_abs`` are recursions over *prior events* and
    do not read a date at all. The price-anchored block — ``dist_high``,
    ``dist_ema``, ``ret5/10/20`` — is read at the last close strictly before
    ``as_of_column``; see :func:`add_regime_features` for why that column has to
    be the decision date and not the event date when the two differ. The close
    actually used comes back in ``runup_asof``.

    Ordering stays on ``date`` regardless: the streak recursion walks the
    ticker's events in event order, which is not what ``as_of_column`` means.
    """
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
    out["runup_asof"] = np.datetime64("NaT", "ns")

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
        anchor = np.full(size, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        row_idx = _anchor_index(
            pdates,
            group["date"].to_numpy(),
            group[as_of_column].to_numpy() if as_of_column != "date" else None,
        )
        for j, idx in enumerate(row_idx):
            idx = int(idx)
            if idx < 0:
                continue
            anchor[j] = pdates[idx]
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
        out.loc[group.index, "runup_asof"] = anchor
        covered += 1

    _log(f"runup: price history found for {covered}/{len(groups)} tickers")
    return out


# --------------------------------------------------------------------------
# block 4 — ORATS state at the last pre-event close, from Tier 2
# --------------------------------------------------------------------------


def add_orats_features(
    df: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    as_of_column: str = "date",
) -> pd.DataFrame:
    """Join Tier-2 ``daily_market`` state as of the last row before each event.

    The as-of rule is ``searchsorted(dates, as_of, "left") - 1``: the last EOD
    row *strictly before* ``as_of_column``, which defaults to the event date.
    For a BMO print that is the previous close, which is exactly right. For an
    AMC print the event-date close is also pre-print and is therefore
    admissible, but the legacy panel used the prior close for both, so this
    reproduces that — a deliberate conservatism (never a leak; at worst one
    session stale on AMC names) that Phase 1 can revisit with the session-aware
    anchors in ``engine.calendar``.

    That conservatism stops being conservative once the decision moves earlier
    than the last pre-print close: anchoring on the event date then reaches
    *forward* of the decision. Pass the decision-date column as
    ``as_of_column`` in that case. The IV row actually used comes back in
    ``orats_asof``; the market cap keeps its own ``mcap_asof``, because the two
    series do not share every date.
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
    orats_asof = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    # Both ceilings travel per row: the event date (the panel's rule) and the
    # decision date (the caller's). `_anchor_index` composes them per series,
    # because the IV series and the market-cap series do not share every date.
    positions: dict[str, list[tuple[int, np.datetime64, np.datetime64]]] = {}
    for i, (ticker, event_date, as_of_date) in enumerate(
        zip(
            out["ticker"].to_numpy(),
            out["date"].to_numpy(),
            out[as_of_column].to_numpy(),
        )
    ):
        positions.setdefault(ticker, []).append((i, event_date, as_of_date))

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

        rows = positions[ticker]
        events = np.array([r[1] for r in rows], dtype="datetime64[ns]")
        as_ofs = None if as_of_column == "date" else np.array(
            [r[2] for r in rows], dtype="datetime64[ns]"
        )
        iv_idx = _anchor_index(dates, events, as_ofs) if len(dates) else np.full(len(rows), -1)
        mcap_idx = (
            _anchor_index(mcap_dates, events, as_ofs)
            if len(mcap_dates)
            else np.full(len(rows), -1)
        )

        for pos, (i, event_date, _as_of) in enumerate(rows):
            j = int(iv_idx[pos])
            if j < 0:
                no_prior += 1
            else:
                orats_asof[i] = dates[j]
                for src_col, panel_col in ORATS_FEATURES.items():
                    targets[panel_col][i] = columns[src_col][j]
                # z-score of ex-earnings IV against its own trailing 252
                # sessions, strictly before j — a within-name standardization,
                # so it cannot leak cross-sectional information from the future.
                #
                # LEAK FIXED 2026-09-02. This block used to sit OUTSIDE the
                # `else`, so it also ran for events with no prior daily row.
                # There `j == -1`, which makes `exern[j]` the LAST row of the
                # ticker's series — routinely years AFTER the print — and
                # `exern[max(0, -253):-1]` the entire history rather than a
                # trailing window. That is a straight future leak, and it
                # populated 507 of the 116,432 non-null values (0.44%), spread
                # over 1989-2025: every event that precedes its own ticker's
                # ORATS coverage.
                #
                # `checks/phase0_migration.py::verify_z252_delta` — the
                # reference implementation this builder is supposed to
                # reproduce — has carried `if j < 0: continue` all along, which
                # is what says this was an oversight rather than a definition.
                #
                # No model has ever consumed the column (verified against every
                # registry entry, champion and retired), so nothing downstream
                # moves. `engine.features.QUARANTINED_FEATURES` now keeps it out
                # of the model-input surface so nothing can start.
                #
                # TODO(2026-Q4): delete `or_exern_z252` outright — this block,
                # its `targets` entry, and its name in PANEL_COLUMNS — together
                # with the KnownDelta and verify_z252_delta in
                # checks/phase0_migration.py that exist only to reconcile it.
                # Needs a Tier-3 rebuild, so it wants its own change, not a
                # ride-along. Until then every panel built before 2026-09-02
                # still carries the leaked values on those 507 rows.
                window = exern[max(0, j - 252) : j]
                window = window[np.isfinite(window)]
                if len(window) >= 60 and np.isfinite(exern[j]):
                    std = window.std()
                    if std > 0:
                        targets["or_exern_z252"][i] = (exern[j] - window.mean()) / std
            if len(mcap_dates):
                jm = int(mcap_idx[pos])
                if jm >= 0:
                    targets["mcap_log"][i] = mcap_log_vals[jm]
                    targets["mcap_usd"][i] = mcap_usd_vals[jm]
                    mcap_asof[i] = mcap_dates[jm]

    for name, values in targets.items():
        out[name] = values
    out["mcap_asof"] = mcap_asof
    out["orats_asof"] = orats_asof
    coverage = float(np.isfinite(targets["or_implied"]).mean())
    _log(
        f"orats: implied-move coverage {coverage:.3f}; "
        f"{no_prior} events with no prior daily row; {missing_ticker} events on unknown tickers"
    )
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def add_pre_print_vol(
    df: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    as_of_column: str = "date",
) -> pd.DataFrame:
    """Vol terms at the LAST SESSION BEFORE THE PRINT, session-aware.

    The ORATS block above anchors strictly before ``event_date`` for both
    sessions — a deliberate legacy conservatism that is never a leak and is one
    session stale on AMC names. These columns are the version that is not
    stale: for a BMO print the anchor is the session before ``event_date``,
    because that morning's print has already moved the event-date close; for an
    AMC print it is ``event_date`` itself, because the print lands after that
    close. Verified over 146,774 events: BMO anchors strictly before at
    100.0000%, AMC on the event date at 99.59% (the rest simply did not quote
    that day) and on-or-before at 100%.

    **Why these have to be panel columns and not a scorer-time rebuild.** The
    ``iv_crush`` model consumes them. Pairing pre- and post-print vol out of
    Tier 2 costs a read of ~9M rows and — decisively — is impossible for an
    event that has not printed. Without them in the panel that model can be
    trained and cannot be SERVED, which on 2026-09-05 left every forward
    TWIN-P5 row ungated and took the strategy off the board.

    Pre-print, causal, and a deterministic function of Tier 2 — which is the
    definition of a Tier-3 column. The realized crush that pairs with them is
    an OUTCOME and stays out, beside ``abs_move`` in spirit but computed by the
    model that needs it.
    """
    if daily is None:
        needed = ["ticker", "date", *PRE_PRINT_FEATURES.keys()]
        _log("reading tier-2 daily_market for pre-print vol …")
        daily = store.read_table("daily_market", columns=needed)
    if events is None:
        events = store.read_table(
            "earnings_events", columns=["ticker", "event_date", "session"])

    sessions = {
        (str(t), pd.Timestamp(d).normalize()): str(sn)
        for t, d, sn in zip(events["ticker"], events["event_date"], events["session"])
        if sn is not None and not pd.isna(sn)
    }
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    sizes = daily.groupby("ticker", sort=True).size()
    starts = sizes.cumsum().shift(1).fillna(0).astype(int)
    spans = {t: (int(a), int(a) + int(n))
             for t, a, n in zip(sizes.index, starts.values, sizes.values)}
    dates = daily["date"].to_numpy()

    out = df.copy()
    n_rows = len(out)
    idx = np.full(n_rows, -1)
    # `as_of_column` exists for the same reason the other blocks have one: a
    # decision taken earlier than the last pre-print close must not anchor on
    # the event date, which would reach FORWARD of the decision. For a forward
    # event the searchsorted below then lands on the newest close available,
    # which is the right estimate of a pre-print quote that does not exist yet.
    when_col = as_of_column if as_of_column in out.columns else "date"
    for i, (ticker, when) in enumerate(zip(out["ticker"].to_numpy(),
                                           pd.to_datetime(out[when_col]).to_numpy())):
        span = spans.get(str(ticker))
        if span is None:
            continue
        lo, hi = span
        session = sessions.get((str(ticker), pd.Timestamp(when).normalize()), AMC)
        # AMC prints after the close, so that close is still pre-print; BMO
        # prints before it, so the last clean quote is the session before.
        side = "right" if session == AMC else "left"
        j = np.searchsorted(dates[lo:hi], when, side=side) - 1
        if j >= 0:
            idx[i] = lo + j

    found = idx >= 0
    for source, column in PRE_PRINT_FEATURES.items():
        values = daily[source].to_numpy(dtype=float)
        out[column] = np.where(found, values[np.where(found, idx, 0)], np.nan)
    _log(f"pre-print vol: {int(found.sum()):,}/{n_rows:,} events anchored")
    return out


def add_implied_history(df: pd.DataFrame) -> pd.DataFrame:
    """``mean_prior_or_implied`` — the running mean of prior quoted implied moves.

    A separate block, and it has to be, because it averages ``or_implied``,
    which :func:`add_orats_features` produces. The event-history block runs
    first and cannot see it.

    **This replaced an oquants-derived feature and the column behind it.** The
    panel used to carry ``implied_move`` — the oquants quote — and average it
    into ``mean_prior_implied_move``. Two things retired that on 2026-09-05:

    * The vendor cannot serve it. Its cache has no fetcher in this repository,
      it lags (2026-08-31 while Tier 2 held prints through 09-04), and it does
      not cover every ticker — so the column blocked Tier 3 from advancing and
      carried two different quantities depending on the ticker.
    * EXP-132 measured the substitution across all five champions: the
      ORATS-derived history was BETTER for iv_crush and implied_t1, within
      noise for both gates, and worse for size_v1_4 by 0.0036 against its own
      0.0033 seed-noise band.

    The two series are NOT the same quantity — EXP-122 put oquants at E|move|
    to 3% and ORATS at 1.55x a model-free straddle — so this shifts the feature
    by a systematic ~+1.95pp and every champion was retrained for it. That is
    the cost; the benefit is a panel that can advance without a vendor.

    Strictly prior, expanding, per ticker: the same recursion the event-history
    block uses, so the two remain comparable.
    """
    out = df.sort_values(["ticker", "date"]).copy()
    prior = out.groupby("ticker")["or_implied"].shift(1)
    out["mean_prior_or_implied"] = (
        prior.groupby(out["ticker"]).expanding().mean().reset_index(level=0, drop=True)
    )
    have = out["mean_prior_or_implied"].notna()
    _log(f"implied history: {int(have.sum()):,}/{len(out):,} events "
         f"({have.mean():.1%}) have a prior quoted implied move")
    return out


def build_panel(daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full Tier-3 causal panel, built from Tier 1 (moves/prices) and Tier 2."""
    _log("block 1/4 — causal events from oquants moves")
    panel = build_events(extra_moves_dirs=(paths.COMPUTED_MOVES,))
    _log("block 2/4 — market regime")
    panel = add_regime_features(panel)
    _log("block 3/4 — run-up and distance features")
    panel = add_runup_features(panel)
    _log("block 4/6 — ORATS state from tier 2")
    panel = add_orats_features(panel, daily=daily)
    _log("block 5/6 — session-aware pre-print vol")
    panel = add_pre_print_vol(panel)
    _log("block 6/6 — implied-move history from ORATS")
    panel = add_implied_history(panel)

    # The per-block anchor dates are provenance for a *caller-supplied* as-of,
    # and the panel's as-of is always the event date, which every row already
    # carries. Dropping them keeps Tier 3 byte-identical to the pre-`as_of_column`
    # build — the property that proves this refactor changed no historical
    # number — and keeps `PANEL_FEATURE_COLUMNS` from acquiring three date
    # columns that no model can consume.
    panel = panel.drop(columns=[c for c in ANCHOR_COLUMNS if c in panel.columns])

    ordered = [c for c in PANEL_COLUMNS if c in panel.columns]
    extra = [c for c in panel.columns if c not in ordered]
    panel = panel[ordered + extra].sort_values(["ticker", "date"]).reset_index(drop=True)
    _log(f"panel complete: {len(panel):,} rows × {len(panel.columns)} columns")
    return panel
