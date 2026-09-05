"""The earnings IV crush as a target: how it is read, and what it costs to read.

One row per event, built from Tier 2 rather than Tier 3, because the quantity
lives on BOTH sides of the print and Tier 3 by construction holds only the
pre-print side. That is not a gap in the panel — it is the panel's leak rule
working. The crush is an OUTCOME, exactly like ``abs_move``, and it belongs
next to the outcome rather than among the features.

**Which two closes.** The pre-print close is the last session strictly before
the print: the session before ``event_date`` for a BMO name, ``event_date``
itself for an AMC one. The post-print close is the next session after that.
That pairing is what makes a BMO and an AMC event describe the same thing —
"the last quote that did not know, and the first that did" — and reading it as
"the day before and the day after ``event_date``" instead would put the AMC
target one session late and quietly measure a day of decay as crush.

**What is dropped, and why not imputed.** A pair needs both readings finite,
``pre > 0``, and at most :data:`MAX_GAP_DAYS` calendar days between the two
sessions. A gap over that means the ticker stopped quoting across the print, so
the "post" reading is a different regime's quote wearing the right date. There
is no defensible fill for it: a missing quote is not a zero crush, and a
forward-filled one is the pre-print number, which would report the crush as
exactly zero on precisely the illiquid names where it is largest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data import store

__all__ = ["MAX_GAP_DAYS", "IV_COLUMNS", "build_crush_frame", "baselines"]

#: Calendar days allowed between the pre- and post-print sessions. Five spans a
#: normal weekend plus a holiday; beyond it the ticker was not quoting.
MAX_GAP_DAYS = 5

#: The vol terms a crush is measured on. ``iv30`` is the primary — better
#: covered and less noisy; ``iv10`` is the horizon a first-post-event expiry
#: actually lives at, and crushes harder.
IV_COLUMNS = ("iv30", "iv10")


def _anchor_rows(events: pd.DataFrame, daily: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row positions in ``daily`` of each event's pre- and post-print closes.

    ``-1`` where the ticker has no such pair. Positional rather than a merge
    because the two anchors are adjacent ROWS of one ticker's series, and
    "the next session this ticker actually quoted" is not expressible as a date
    arithmetic — a name that stopped trading has no next session, and that is
    a fact about the name, not a date to compute.
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


def build_crush_frame(
    events: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    *,
    log=print,
) -> pd.DataFrame:
    """``(ticker, event_date)`` with the realized crush on every term in IV_COLUMNS.

    Columns: ``pre_<term>``, ``post_<term>``, ``crush_pct_<term>`` (the ratio, in
    percent) and ``crush_pp_<term>`` (the difference, in vol points), plus
    ``pre_exern_iv30`` — the ex-earnings 30-day vol at the same pre-print close,
    which is the structural baseline this experiment has to beat.
    """
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

    pre, post = _anchor_rows(events, daily)
    paired = pre >= 0
    log(f"[crush] {paired.sum():,} of {len(events):,} events have an adjacent "
        f"(pre, post) session pair")

    out = events.loc[paired, ["ticker", "event_date", "session"]].reset_index(drop=True)
    pre, post = pre[paired], post[paired]
    dates = daily["date"].to_numpy()
    out["pre_date"] = dates[pre]
    out["post_date"] = dates[post]
    out["gap_days"] = (dates[post] - dates[pre]).astype("timedelta64[D]").astype(float)

    for term in IV_COLUMNS:
        values = daily[term].to_numpy()
        out[f"pre_{term}"] = values[pre]
        out[f"post_{term}"] = values[post]
    out["pre_exern_iv30"] = daily["exern_iv30"].to_numpy()[pre]
    out["pre_exern_iv10"] = daily["exern_iv10"].to_numpy()[pre]

    usable = out["gap_days"] <= MAX_GAP_DAYS
    log(f"[crush] {int((~usable).sum()):,} dropped for a gap over {MAX_GAP_DAYS} days")
    out = out[usable].reset_index(drop=True)

    for term in IV_COLUMNS:
        pre_v = out[f"pre_{term}"].to_numpy(dtype=float)
        post_v = out[f"post_{term}"].to_numpy(dtype=float)
        ok = np.isfinite(pre_v) & np.isfinite(post_v) & (pre_v > 0)
        # Divide only where the denominator is real. `np.where` would evaluate
        # the ratio everywhere first and warn on the pre == 0 rows it is about
        # to discard — noise that trains a reader to ignore numpy warnings.
        ratio = np.full(len(out), np.nan)
        np.divide(post_v, pre_v, out=ratio, where=ok)
        out[f"crush_pp_{term}"] = np.where(ok, post_v - pre_v, np.nan)
        out[f"crush_pct_{term}"] = np.where(ok, 100.0 * (ratio - 1.0), np.nan)
        log(f"[crush] {term}: {int(ok.sum()):,} usable, median "
            f"{np.nanmedian(out[f'crush_pct_{term}']):.2f}%")

    out["year"] = out["event_date"].dt.year
    return out


def baselines(frame: pd.DataFrame, target: str = "crush_pct_iv30") -> dict[str, np.ndarray]:
    """The model-free predictors the primary has to beat, on ``frame``'s rows.

    Three, and the spec requires clearing the BEST OF THEM ON EACH MEASURE
    SEPARATELY rather than a single named opponent. They disagree about which
    is better: the structural anchor wins RMSE and correlation, the constant
    wins MAE, and the blend wins MAE outright while keeping the correlation.
    Registering one would let the model choose its opponent after seeing the
    results, which is the defect that cost this programme mean-per-trade.

    ``constant`` uses the FULL-SAMPLE median and so is not itself causal. That
    is deliberate and it makes the bar harder, not easier: the model must beat
    a baseline that has already seen the answer.
    """
    truth = frame[target].to_numpy(dtype=float)
    term = "iv30" if target.endswith("iv30") else "iv10"
    pre = frame[f"pre_{term}"].to_numpy(dtype=float)
    exern = frame[f"pre_exern_{term}"].to_numpy(dtype=float)
    constant = np.full(len(frame), float(np.nanmedian(truth)))

    # The structural anchor has to be in the TARGET's units, and the two
    # supported targets are in different ones. Keying only off the term - which
    # is what the first version did - compared a percent change against a vol
    # level for the `level` arm and reported a meaningless win.
    if target.startswith("crush_pct_"):
        ratio = np.full(len(frame), np.nan)
        np.divide(exern, pre, out=ratio,
                  where=np.isfinite(exern) & np.isfinite(pre) & (pre > 0))
        structural = 100.0 * (ratio - 1.0)
    elif target.startswith("post_"):
        # Predicting the post-print LEVEL: the ex-earnings vol at the pre-print
        # close IS the structural estimate of it, with no transformation.
        structural = np.where(np.isfinite(exern), exern, np.nan)
    else:
        raise ValueError(
            f"no baseline is defined for target {target!r} — add one rather than "
            "letting a mismatched-units comparison report a win"
        )
    return {
        "constant": constant,
        "structural": structural,
        "blend": 0.5 * constant + 0.5 * structural,
    }
