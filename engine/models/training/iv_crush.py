"""Champion `iv_crush` model — how far 30-day implied vol falls across the print.

Target: ``100 * (post_iv30 / pre_iv30 - 1)``, where the pre-print close is the
last session strictly before the print (the session before ``event_date`` for a
BMO name, ``event_date`` itself for AMC) and the post-print close is the next
session after it. Measured on 145,798 events; iv30 falls at 83.2% of prints,
median −13.68%.

**The first signed target in the programme, and that is load-bearing.** Every
other feature model predicts a magnitude — an absolute move, an implied move —
so Tier 4's interval machinery floored both band bounds at zero. A hard floor
here would clip every band on the large majority of rows to ``[0, 0]``, and
``[0, 0]`` is not inverted, so no existing check would object. This producer
declares ``interval_floor=None``; see :func:`engine.data.features.tier4.interval_for`.

**The target lives on both sides of the print, so it is not a panel column.**
Tier 3 holds only pre-print state by construction — that is its leak rule
working, not a gap. The crush is an OUTCOME, like ``abs_move``, and it is read
from Tier 2 here and joined to the panel row that describes the event.

**What EXP-128 measured, and what it did not.** Walk-forward OOS over 71,864
events, 2014-2026: MAE 8.93pp against 11.12pp for the best model-free baseline,
RMSE 15.73 against 18.18, r 0.502, better in 13 of 13 years, top-vs-bottom
predicted-decile spread 31.0pp, 80% band covering 81.5%.

It **failed** its own registered coverage floor — 71,864 scored rows against a
pre-registered 80,000 — because the arm consumed every numeric panel column and
walk-forward drops a row if any one of them is non-finite. A curated feature
list would recover most of those rows and is the obvious next iteration. The
model is materialised anyway, by the user's explicit call, with the shortfall
recorded rather than argued away.

**The ablation is the useful finding.** ORATS publishes ``exern_iv30``, an
ex-earnings 30-day vol at the same close, which looked like a strong free
baseline (r 0.670 over the full sample). Dropping it costs 0.014pp of MAE. On
the rows the model is actually scored on it is worse than a constant (MAE 12.25
against 11.64) and its correlation collapses to 0.166 — the 0.670 was tail
alignment on a heavy-tailed target, not signal. Whatever this model knows, it
does not come from that column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from engine.models.training.common import SEED, fit_final, log, walk_forward

__all__ = [
    "FEATURES",
    "TARGET",
    "MAX_GAP_DAYS",
    "fit",
    "crush_frame",
    "prepare",
    "PANEL_VOL_COLUMNS",
    "train",
]

TARGET = "crush_pct_iv30"

#: Calendar days allowed between the pre- and post-print sessions. Five spans a
#: weekend plus a holiday; beyond it the ticker stopped quoting across the
#: print, so the "post" reading is a different regime's quote wearing the right
#: date. Dropped rather than imputed — a missing quote is not a zero crush, and
#: forward-filling one reports the crush as exactly zero on precisely the
#: illiquid names where it is largest.
MAX_GAP_DAYS = 5

#: The pre-print vol terms this model reads, which are TIER-3 PANEL COLUMNS.
#: Listed so `prepare` can refuse a stale panel loudly instead of failing later
#: as an unservable champion.
PANEL_VOL_COLUMNS = ("pre_iv30", "pre_iv10", "pre_exern_iv30", "pre_exern_iv10")

#: The feature list EXP-128 measured: every numeric panel column that is not an
#: outcome, a key, quarantined, or unavailable live.
#:
#: Declared explicitly rather than derived at fit time. A derived list changes
#: silently when the panel gains a column, and the registry would then hold a
#: champion whose recorded features are not the ones it was fit on — the exact
#: drift ``tier4.size_feature_model`` cross-checks against.
FEATURES: tuple[str, ...] = (
    "n_prior", "mean_prior_move", "mean_prior_abs_move", "ema2_prior_move",
    "ema4_prior_move", "ema8_prior_move", "ema12_prior_move", "ema2_prior_abs_move",
    "ema4_prior_abs_move", "ema8_prior_abs_move", "ema12_prior_abs_move",
    "mean_prior_or_implied", "spy_ret21", "spy_ret63", "spy_ret252",
    "spy_dd252", "spy_vol20", "signed_streak", "ema12r_abs", "dist_high",
    "dist_ema", "ret5", "ret10", "ret20", "or_implied", "or_skewing",
    "or_contango", "or_rvol30", "or_exern30", "or_iv30", "or_iee", "or_fwd90_30",
    "or_fexern90_30", "mcap_log", "abs_dist_high", "abs_dist_ema", "has_implied_quote",
    "pre_iv30", "pre_iv10", "pre_exern_iv30", "pre_exern_iv10",
)


def fit(X, y, seed: int = SEED):
    return HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=300, random_state=seed
    ).fit(X, y)


def _anchor_rows(events: pd.DataFrame, daily: pd.DataFrame):
    """Row positions in ``daily`` of each event's pre- and post-print closes.

    Positional rather than a date join: the two anchors are adjacent ROWS of one
    ticker's own series, and "the next session this ticker actually quoted" is
    not date arithmetic — a name that stopped trading has no next session, and
    that is a fact about the name rather than a date to compute.
    """
    sizes = daily.groupby("ticker", sort=True).size()
    starts = sizes.cumsum().shift(1).fillna(0).astype(int)
    spans = {t: (s, s + n) for t, s, n in zip(sizes.index, starts.values, sizes.values)}
    dates = daily["date"].to_numpy()

    pre = np.full(len(events), -1)
    post = np.full(len(events), -1)
    for i, (ticker, event_date, session) in enumerate(
        zip(events["ticker"].to_numpy(), events["event_date"].to_numpy(),
            events["session"].to_numpy())
    ):
        span = spans.get(ticker)
        if span is None:
            continue
        lo, hi = span
        # BMO prints before `event_date`'s close, so that close already knows.
        # AMC prints after it, so it does not.
        side = "right" if session == "AMC" else "left"
        j = np.searchsorted(dates[lo:hi], event_date, side=side) - 1
        if j < 0 or j + 1 >= (hi - lo):
            continue
        pre[i], post[i] = lo + j, lo + j + 1
    return pre, post


def crush_frame(events=None, daily=None) -> pd.DataFrame:
    """``(ticker, event_date)`` with the realized crush and its pre-print inputs."""
    from engine.data import store

    if events is None:
        events = store.read_table(
            "earnings_events", columns=["ticker", "event_date", "session"]
        )
    if daily is None:
        daily = store.read_table(
            "daily_market",
            columns=["ticker", "date", "iv10", "iv30", "exern_iv10", "exern_iv30"],
        )
    events = events.dropna(subset=["session"]).copy()
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.sort_values(["ticker", "event_date"]).reset_index(drop=True)
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)

    pre_i, post_i = _anchor_rows(events, daily)
    paired = pre_i >= 0
    out = events.loc[paired, ["ticker", "event_date"]].reset_index(drop=True)
    pre_i, post_i = pre_i[paired], post_i[paired]

    dates = daily["date"].to_numpy()
    gap = (dates[post_i] - dates[pre_i]).astype("timedelta64[D]").astype(float)
    for column, source in (("pre_iv30", "iv30"), ("pre_iv10", "iv10"),
                           ("pre_exern_iv30", "exern_iv30"),
                           ("pre_exern_iv10", "exern_iv10")):
        out[column] = daily[source].to_numpy()[pre_i]
    post_iv30 = daily["iv30"].to_numpy()[post_i]

    pre_v = out["pre_iv30"].to_numpy(dtype=float)
    ok = (
        np.isfinite(pre_v) & np.isfinite(post_iv30) & (pre_v > 0) & (gap <= MAX_GAP_DAYS)
    )
    # Divide only where the denominator is real: `np.where` would evaluate the
    # ratio everywhere and warn on the rows it is about to discard.
    ratio = np.full(len(out), np.nan)
    np.divide(post_iv30, pre_v, out=ratio, where=ok)
    out["post_iv30"] = np.where(ok, post_iv30, np.nan)
    out[TARGET] = np.where(ok, 100.0 * (ratio - 1.0), np.nan)
    return out


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """The panel row for each event, joined to that event's realized crush.

    **Only the TARGET comes from Tier 2 now.** The four pre-print vol terms this
    model reads — ``pre_iv30``, ``pre_iv10``, ``pre_exern_iv30``,
    ``pre_exern_iv10`` — are Tier-3 panel columns as of 2026-09-05, and that
    move is what makes the model servable at all: they are pre-print, causal and
    deterministic from Tier 2, and while they lived only in this module's
    pairing they could be reconstructed for an event that had PRINTED and never
    for one that had not. The board scores events that have not printed, so
    every forward row came back ungated and TWIN-P5 went dark.

    The realized crush stays out of Tier 3 deliberately: it is an OUTCOME, like
    ``abs_move``, and it is read here from the two closes that bracket the
    print. Tier 3 holding only the pre-print side is the leak rule working, not
    a gap in the panel.

    The join is INNER on the target. An event with no usable ``(pre, post)``
    pair has nothing to learn from — not a row this model declines to score, a
    row it has nothing to say about, which Tier 4 records as a NULL forecast.
    """
    missing = [c for c in PANEL_VOL_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(
            f"the panel is missing {missing} — rebuild Tier 3 "
            "(`python3 -m engine.data.rebuild --table panel`). Without them this "
            "model can be trained and cannot be served."
        )
    crush = crush_frame()[["ticker", "event_date", TARGET]]
    joined = panel.merge(
        crush, left_on=["ticker", "date"], right_on=["ticker", "event_date"], how="inner"
    )
    return joined.drop(columns=["event_date"])


def train(dataset: pd.DataFrame, *, seed: int = SEED, first_test_year: int = 2013):
    result = walk_forward(
        dataset, FEATURES, TARGET, fit, first_test_year=first_test_year, seed=seed
    )
    log(
        f"iv_crush: OOS n={result.metrics['n']:,} r={result.metrics['r']:.4f} "
        f"mae={result.metrics['mae']:.4f}"
    )
    model = fit_final(dataset, FEATURES, TARGET, fit, seed=seed)
    return model, result
