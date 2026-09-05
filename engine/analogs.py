"""The empirical layer: what actually happened to trades like this one.

The model layer says what a champion predicts. This says what a matched set of
real, historically-priced trades did. They are reported side by side and never
averaged, because when they disagree that disagreement *is* the finding — a
model extrapolating past its evidence and a thin empirical set look identical
once you take their mean.

**Matching.** Four dimensions, from the guide: market-cap bucket, the ticker's
implied move against its own history, days to expiry at entry, and absolute
moneyness. Terciles for the implied dimension are cut on the eligible pool
itself rather than on fixed thresholds, so "rich for this name" tracks the
regime instead of a number written down in 2026.

**Widening.** Below :data:`MIN_ANALOGS` matches the buckets are dropped in a
fixed order — moneyness, then DTE, then implied tercile — and the number
dropped is reported. Fixed order matters: a ladder that widened along whichever
dimension yielded the most matches would be selecting the comparison set by its
answer.

**Causality.** Only trades that had *closed* before the decision date are
eligible. Scoring a 2019 event on 2024 trades would make every backtest look
prescient, and it is the easiest leak in the whole system to introduce by
accident, so it is enforced here rather than left to callers.

**Determinism.** The bootstrap seed is derived from the snapshot hash and the
request, so the same question against the same data returns the same interval,
byte for byte, on any machine.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "MIN_ANALOGS",
    "WIDENING_ORDER",
    "MCAP_EDGES",
    "DTE_BANDS",
    "MONEYNESS_BANDS",
    "AnalogSet",
    "AnalogMatcher",
    "bucket_frame",
]

#: Below this, an empirical distribution is an anecdote. The guide's threshold.
MIN_ANALOGS = 30

#: Dimensions are dropped in this order, most-specific first. Moneyness goes
#: first because the evidence base is ATM-centric anyway; the implied tercile
#: goes last because it is the dimension most predictive of the return.
WIDENING_ORDER = ("moneyness_band", "dte_band", "implied_tercile")

#: Market-cap buckets in USD. The 1–10B slice is the plan's claimed +5.3% pocket
#: and is kept as its own bucket for exactly that reason.
MCAP_EDGES = (1e9, 1e10)
MCAP_LABELS = ("<1B", "1-10B", ">=10B")

DTE_BANDS = ((1, 3), (4, 10), (11, 25), (26, 45))
DTE_LABELS = ("1-3", "4-10", "11-25", "26-45")

#: |strike/spot − 1| in percent. The ATM band is the only one the current
#: evidence actually covers; the others exist so a non-ATM request is matched
#: honestly rather than silently answered with ATM trades.
MONEYNESS_BANDS = (2.0, 5.0)
MONEYNESS_LABELS = ("ATM", "2-5%", ">5%")


def _bucket(values, edges, labels) -> np.ndarray:
    """Label each value by which side of ``edges`` it falls on.

    Half-open upward: ``value < edges[0]`` is the first label, and a value equal
    to an edge belongs to the bucket above it. A non-finite value gets ``None``,
    which the matcher reads as "this dimension cannot be matched on" rather than
    as a bucket of its own.
    """
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(np.asarray(edges, dtype=float), values, side="right")
    out = np.array(labels, dtype=object)[np.clip(idx, 0, len(labels) - 1)]
    out[~np.isfinite(values)] = None
    return out


def _dte_band(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), None, dtype=object)
    for (lo, hi), label in zip(DTE_BANDS, DTE_LABELS):
        out[(values >= lo) & (values <= hi)] = label
    return out


def bucket_frame(
    trades: pd.DataFrame, *, implied_edges: tuple[float, float] | None = None
) -> pd.DataFrame:
    """Attach the four matching dimensions to a trade frame.

    ``implied_ratio`` is the event's quoted implied move over the ticker's own
    prior mean — "rich or cheap *for this name*", which is what makes the
    dimension comparable across a $12 biotech and a mega cap.
    """
    out = trades.copy()

    def column(*names: str) -> pd.Series:
        """The first of ``names`` present, else an all-NaN series of the right length.

        A trade frame reaches here in several shapes — enriched, partly enriched,
        or empty on a fresh install before any replay has run. A missing column
        must yield an unmatchable dimension, not an exception: ``pd.to_numeric``
        on a missing key returns a bare scalar, which then fails on the first
        Series operation with an error naming neither the column nor the cause.
        """
        for name in names:
            if name in out.columns:
                return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype="float64")

    out["mcap_bucket"] = _bucket(column("mcap_usd"), MCAP_EDGES, MCAP_LABELS)
    out["dte_band"] = _dte_band(column("dte_entry"))

    spot = column("spot_entry")
    strike = column("strike")
    with np.errstate(divide="ignore", invalid="ignore"):
        moneyness = (strike / spot - 1.0).abs() * 100.0
    out["moneyness_pct"] = moneyness
    out["moneyness_band"] = _bucket(moneyness, MONEYNESS_BANDS, MONEYNESS_LABELS)

    # Measured at the *entry* date where the caller supplied it. For a structure
    # that enters two weeks before the print, the implied move quoted at entry
    # and the one quoted at the last pre-print close are different numbers, and
    # matching a request's entry-date reading against the trades' event-date one
    # would put them in different buckets for no reason. `or_implied` remains the
    # fallback for callers that only have the event-level figure.
    implied = column("implied_at_entry", "or_implied")
    prior = column("mean_prior_or_implied")
    ratio = implied / prior.replace(0, np.nan)
    out["implied_ratio"] = ratio
    if implied_edges is None:
        finite = ratio[np.isfinite(ratio)]
        implied_edges = (
            tuple(np.quantile(finite, [1 / 3, 2 / 3])) if len(finite) >= 30 else (0.9, 1.1)
        )
    out["implied_tercile"] = _bucket(ratio, implied_edges, ("low", "mid", "high"))
    out.attrs["implied_edges"] = tuple(float(e) for e in implied_edges)
    return out


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class AnalogSet:
    """The matched empirical distribution behind one score."""

    strategy: str
    alpha: float
    n: int
    mean: float | None
    median: float | None
    win_rate: float | None
    p10: float | None
    p90: float | None
    ci_low: float | None
    ci_high: float | None
    widened: int
    #: Dimensions that had no value to match on at all — distinct from
    #: ``dropped``, which is what widening gave up deliberately to find enough
    #: trades. Non-empty means no comparison was possible, not that it was loose.
    unavailable: tuple[str, ...] = ()
    buckets: dict = field(default_factory=dict)
    dropped: tuple[str, ...] = ()
    thin: bool = False
    years: tuple[int, ...] = ()

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 6) if isinstance(v, float) and np.isfinite(v) else v

        return {
            "n_analogs": self.n,
            "mean": r(self.mean),
            "median": r(self.median),
            "win_rate": r(self.win_rate),
            "p10": r(self.p10),
            "p90": r(self.p90),
            "ci_low": r(self.ci_low),
            "ci_high": r(self.ci_high),
            "widened": self.widened,
            "dropped": list(self.dropped),
            "unavailable": list(self.unavailable),
            "thin": self.thin,
            "buckets": self.buckets,
            "years": list(self.years),
        }


def _empty(strategy: str, alpha: float, buckets: dict, widened: int, dropped,
           *, unavailable: tuple[str, ...] = ()) -> AnalogSet:
    return AnalogSet(
        strategy=strategy, alpha=alpha, n=0, mean=None, median=None, win_rate=None,
        p10=None, p90=None, ci_low=None, ci_high=None, widened=widened,
        unavailable=unavailable,
        buckets=buckets, dropped=tuple(dropped), thin=True,
    )


# --------------------------------------------------------------------------
# the matcher
# --------------------------------------------------------------------------


class AnalogMatcher:
    """Matches a request against a bucketed trade population.

    Built once per scoring run over the engine-replayed trades — never over the
    legacy S1/S2/S3 rows, which are a single worst-case fill and were specified
    differently from the three program structures.
    """

    def __init__(self, trades: pd.DataFrame, *, snapshot: str = ""):
        self.snapshot = snapshot
        self.trades = bucket_frame(trades) if "mcap_bucket" not in trades.columns else trades
        self.implied_edges = self.trades.attrs.get("implied_edges", (0.9, 1.1))
        if "exit_date" in self.trades.columns:
            self.trades["exit_date"] = pd.to_datetime(self.trades["exit_date"])
        # The (strategy, alpha) pool is re-derived on every match otherwise; over
        # a full calendar that is a six-figure row scan per event. Cached once.
        self._pools: dict[tuple[str, float], pd.DataFrame] = {}
        # Causal pools: the as_of-filtered, causally re-bucketed pool depends
        # only on (strategy, alpha, as_of), and board rows share all three.
        # Without this cache the quantile + re-bucket cost is paid per row —
        # ~19s on a 3,120-row board.
        self._causal_pools: dict[tuple[str, float, pd.Timestamp],
                                 tuple[pd.DataFrame, tuple[float, float] | None]] = {}
        #: Cache ceiling, in entries. Sized from the workload, not from a round
        #: number: a full three-week board (3,120 rows) generates **34** distinct
        #: keys — 31 entry dates x 2 scoreable strategies — so 64 clears the
        #: working set outright and the cap never binds where the cache pays.
        #:
        #: The ceiling matters because an entry is not small. Each one holds a
        #: filtered, re-bucketed copy of the (strategy, alpha) pool — measured at
        #: 6.2 MB on average and ~9 MB for recent dates, where few trades have
        #: been excluded. At the previous 256 that is **1.6 GB**, which took the
        #: Scorer from 2.5 GB to 4.1 GB on a 7.8 GB box — most of the headroom
        #: the phase-1 suite had just recovered by not loading a second
        #: FeatureContext.
        #:
        #: Nothing was gained for it. The paths that would fill 256 keys —
        #: `recalibrate.build_pairs` (~1,000 scattered decision dates), the
        #: calibration sampler (300) — barely repeat an as_of, so they get almost
        #: no hits regardless; the slots above the board's working set are pure
        #: cost. 64 keeps the whole benefit at ~575 MB worst case.
        self.MAX_CAUSAL_CACHE = 64

    # -- request buckets ---------------------------------------------------

    def buckets_for(
        self,
        *,
        mcap_usd: float | None,
        dte: float | None,
        moneyness_pct: float | None,
        implied_ratio: float | None,
    ) -> dict:
        return {
            "mcap_bucket": _bucket([mcap_usd if mcap_usd is not None else np.nan],
                                   MCAP_EDGES, MCAP_LABELS)[0],
            "dte_band": _dte_band([dte if dte is not None else np.nan])[0],
            "moneyness_band": _bucket(
                [moneyness_pct if moneyness_pct is not None else np.nan],
                MONEYNESS_BANDS, MONEYNESS_LABELS,
            )[0],
            "implied_tercile": _bucket(
                [implied_ratio if implied_ratio is not None else np.nan],
                self.implied_edges, ("low", "mid", "high"),
            )[0],
            # The raw ratio travels with the buckets so match() can re-bucket
            # it against CAUSAL tercile edges (derived from trades already
            # closed at as_of) instead of the population edges above — which
            # were fit on all years, future ones included.
            "implied_ratio": implied_ratio,
        }

    # -- matching ----------------------------------------------------------

    def match(
        self,
        strategy: str,
        buckets: dict,
        *,
        alpha: float,
        as_of=None,
        min_analogs: int = MIN_ANALOGS,
        bootstrap: int = 2000,
        request_key: str = "",
    ) -> AnalogSet:
        """Matched returns, widening the buckets until there are enough."""
        key = (strategy, round(float(alpha), 4))
        base = self._pools.get(key)
        if base is None:
            base = self.trades[
                (self.trades["strategy"] == strategy)
                & np.isclose(self.trades["fill_alpha"].astype(float), float(alpha))
            ]
            self._pools[key] = base
        pool = base
        if as_of is not None:
            # Closed strictly before the decision: a trade still open on the day
            # we decide has not yet told us anything about how it went. The
            # causally filtered AND re-bucketed pool depends only on
            # (strategy, alpha, as_of), so it is cached — many board rows share
            # the triple, and recomputing the quantile + bucket per row cost
            # ~19s on a 3,120-row board.
            ts = pd.Timestamp(as_of).normalize()
            cache_key = (strategy, round(float(alpha), 4), ts)
            cached = self._causal_pools.get(cache_key)
            if cached is not None:
                pool, edges = cached
            else:
                pool = pool[pool["exit_date"] < ts]
                edges = None
                if "implied_ratio" in pool.columns and len(pool):
                    finite = pool["implied_ratio"][np.isfinite(pool["implied_ratio"])]
                    if len(finite) >= 30:
                        edges = tuple(float(e) for e in np.quantile(finite, [1 / 3, 2 / 3]))
                    else:
                        edges = (0.9, 1.1)
                    # Causal terciles: the implied-ratio bucket edges must come
                    # from the same closed-before-as_of pool, not from the whole
                    # population — population edges bake in future years (a 2019
                    # request bucketed by thresholds derived partly from 2024
                    # data is a look-ahead). Both the pool and the request are
                    # bucketed on the causal edges so the labels align.
                    pool = pool.assign(
                        implied_tercile=_bucket(pool["implied_ratio"], edges,
                                                ("low", "mid", "high"))
                    )
                if len(self._causal_pools) >= self.MAX_CAUSAL_CACHE:
                    self._causal_pools.pop(next(iter(self._causal_pools)))
                self._causal_pools[cache_key] = (pool, edges)
            ratio = buckets.get("implied_ratio")
            if ratio is not None and edges is not None:
                buckets = dict(buckets)
                buckets["implied_tercile"] = _bucket([ratio], edges,
                                                     ("low", "mid", "high"))[0]

        # A dimension with no value cannot match on, and must be COUNTED as
        # dropped rather than quietly skipped. The loop below used to `continue`
        # past a None bucket, so a row with no option chain — no `dte_entry`,
        # no `strike`, therefore no `dte_band` and no `moneyness_band` — matched
        # on the remaining two, succeeded on the FIRST pass, and reported
        # `widened: 0`. The board then showed the strategy's own base rate
        # (STR-THRU +0.0270, win 0.388, matched set up to 17,666 — the entire
        # population) wearing a badge that said nothing had been dropped.
        #
        # The number was never the problem; the label was. Refusing to answer
        # was tried and was worse: the analog layer exists to answer when the
        # model layer cannot, and a thin or absent match is the ONLY signal that
        # the model is extrapolating past its evidence. Suppressing it removes
        # the warning along with the estimate.
        unavailable = [d for d in ("mcap_bucket", *WIDENING_ORDER)
                       if buckets.get(d) is None]

        if len(unavailable) == 4:
            # No dimension has a value — no chain, no size, no implied quote.
            # Matching on zero dimensions would return the strategy's base
            # rate wearing an empty bucket label, the exact lie this layer was
            # rewritten to stop telling; the honest answer is an empty match.
            return self._summarize(
                self.trades.iloc[0:0], strategy, alpha, buckets, 0,
                list(unavailable), bootstrap=bootstrap,
                min_analogs=min_analogs, request_key=request_key,
                unavailable=tuple(unavailable),
            )

        active = [d for d in ("mcap_bucket", *WIDENING_ORDER) if d not in unavailable]
        dropped: list[str] = list(unavailable)
        for widened in range(len(WIDENING_ORDER) + 1 - len(unavailable)):
            mask = np.ones(len(pool), dtype=bool)
            for dimension in active:
                want = buckets.get(dimension)
                if want is None:
                    continue
                mask &= (pool[dimension] == want).to_numpy()
            matched = pool[mask]
            remaining = [d for d in WIDENING_ORDER if d not in dropped]
            if len(matched) >= min_analogs or not remaining:
                return self._summarize(
                    matched, strategy, alpha, buckets, len(dropped), dropped,
                    bootstrap=bootstrap, min_analogs=min_analogs,
                    request_key=request_key, unavailable=tuple(unavailable),
                )
            drop = remaining[0]
            active.remove(drop)
            dropped.append(drop)
        raise AssertionError("unreachable")  # pragma: no cover

    def _summarize(
        self, matched, strategy, alpha, buckets, widened, dropped, *,
        bootstrap, min_analogs, request_key, unavailable=(),
    ) -> AnalogSet:
        returns = pd.to_numeric(matched.get("ret"), errors="coerce").to_numpy(dtype=float)
        returns = returns[np.isfinite(returns)]
        if returns.size == 0:
            return _empty(strategy, alpha, buckets, widened, dropped,
                          unavailable=unavailable)

        thin = returns.size < min_analogs
        ci_low = ci_high = None
        if not thin and bootstrap:
            # The scorer does not invent an interval for a set too thin to
            # support one; `thin` is reported instead and the dashboard renders
            # it as low-confidence.
            rng = np.random.default_rng(_seed(self.snapshot, strategy, alpha, buckets, request_key))
            draws = rng.choice(returns, size=(bootstrap, returns.size), replace=True)
            means = draws.mean(axis=1)
            ci_low, ci_high = (float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95)))

        years = ()
        if "event_date" in matched.columns:
            years = tuple(sorted(pd.to_datetime(matched["event_date"]).dt.year.unique().tolist()))

        return AnalogSet(
            strategy=strategy,
            alpha=float(alpha),
            n=int(returns.size),
            mean=float(returns.mean()),
            median=float(np.median(returns)),
            win_rate=float((returns > 0).mean()),
            p10=float(np.quantile(returns, 0.10)),
            p90=float(np.quantile(returns, 0.90)),
            ci_low=ci_low,
            ci_high=ci_high,
            widened=widened,
            buckets=dict(buckets),
            dropped=tuple(dropped),
            thin=thin,
            years=years,
            unavailable=tuple(unavailable),
        )


def _seed(snapshot: str, strategy: str, alpha: float, buckets: dict, request_key: str) -> int:
    """Deterministic seed from the data snapshot and the request.

    Derived rather than fixed so two different requests do not share a bootstrap
    realization, and derived from the *snapshot* so the same request against
    rebuilt data is a different draw — which is honest, because it is a
    different sample.
    """
    payload = "|".join(
        [snapshot, strategy, f"{alpha:.4f}", request_key]
        + [f"{k}={buckets.get(k)}" for k in sorted(buckets)]
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
