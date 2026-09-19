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
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable

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
    "AnalogEvidenceHook",
    "bucket_frame",
    "match_frame",
]

AnalogEvidenceHook = Callable[[dict[str, Any]], None]

#: Below this, an empirical distribution is an anecdote. The guide's threshold.
MIN_ANALOGS = 30

#: Combined byte budget for `AnalogMatcher`'s causal-family cache
#: (`_causal_pools` + `_causal_row_caches` + `phase4_recipe_cache`,
#: evicted together — see `AnalogMatcher.MAX_CAUSAL_CACHE`'s comment for why
#: a fixed key-count cap cannot bound this: per-key cost tracks how large
#: that as_of's matched population is, and the 2026-09-18 diagnostic runs
#: measured it varying roughly 10x key to key (run 3: +5 keys -> +770 MB;
#: elsewhere in the same run, +30 keys -> +350 MB). 600 MB leaves headroom
#: under the 5.5 GB bounded-run cap alongside the ~2 GB fixed panel/context
#: baseline and the matcher's OTHER (small, bounded) caches.
#:
#: 2026-09-18, extended: an instrumented strict-capture run measured the
#: forward pass's analog step alone costing net +1.48 GB of real RSS while
#: this budget stayed at 600 MB — this budget was blind to the POPULATION
#: family (`_pools` + `_population_row_caches`), which is keyed only on
#: (strategy, alpha), never evicted, and which `AnalogMatcher.match`
#: documents in FULL (`_evidence_rows` over the whole, uncausal population,
#: not just a matched subset) the first time a phase4-traced candidate
#: touches a given key. Small key cardinality (bounded by the number of
#: strategies, not by event count) but each key can be a five- or six-figure
#: row population — real, resident, and previously invisible to this
#: number. Both families now share this ONE budget and total (see
#: `AnalogMatcher._evict_analog_cache_until_under_budget`), so this name
#: measures actual combined resident cost, not just the causal slice of it.
#:
#: 2026-09-19, lowered 600 MB -> 400 MB: a synthetic benchmark
#: (scratch/bench_analog_budget.py, untracked -- one population, one
#: strategy/alpha, 200 distinct as_of cutoffs, each causal pool sized to
#: the matcher's own accounting) measured ~2.25 MB/causal-key, so 600 MB
#: holds ~266 concurrent keys and 400 MB ~177. Sweeping the same 200 keys
#: twice: at 600 MB nothing evicts; at 400 MB, 19 of 200 evict and force a
#: quantile+bucket recompute on the second pass, costing ~21-28 ms/evicted
#: key against a ~80 ms/call cost (`_summarize`'s bootstrap) that runs
#: regardless of cache state -- roughly a 3% steady-state slowdown in the
#: worst case measured. A real capture's causal-key cardinality (a handful
#: of strategies x `engine.replay.ALPHA_GRID` (5) x the boundary events'
#: own as_of dates, typically under a few dozen) sits well under 177, so
#: 400 MB is very unlikely to force any recompute in practice while
#: freeing ~200 MB of headroom under the bounded-run cap.
CAUSAL_CACHE_BUDGET_BYTES = 400 * 1024 * 1024

#: `_causal_row_caches`/`phase4_recipe_cache` grow lazily, one row at a time,
#: as candidates touch rows -- unlike `_causal_pools`' own DataFrame (whose
#: exact byte cost `pool.memory_usage(deep=True).sum()` gives directly at
#: insertion time), their EVENTUAL size for a key is not known when that
#: key is first cached. `len(pool)` -- the key's full causal-filtered
#: population -- is used as a conservative upper bound (a widened match can
#: reach the entire population; see `AnalogMatcher.match`'s "matched set up
#: to 17,666 -- the entire population" note), multiplied by an average
#: bytes/row measured on the 2026-09-18 diagnostic run (280cf7c, run 4):
#: 506,512 new rows between two checkpoints, `_causal_row_caches` grew
#: 199.01 MB (393 B/row), `phase4_recipe_cache` grew 302.84 MB (598 B/row).
#: Overestimating (most keys never touch their whole population) means
#: evicting a bit earlier than strictly necessary, never later -- the safe
#: direction for a memory cap.
CAUSAL_ROW_CACHE_BYTES_PER_ROW = 393
RECIPE_CACHE_BYTES_PER_ROW = 598

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


def _json_value(value):
    """Return a detached, JSON-safe representation of an evidence value."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _freeze(value):
    """Recursively lock a ``_json_value`` structure against in-place mutation.

    Only ever applied to output that is already dict/list/scalar (that is
    what ``_json_value`` produces — it never emits a bare tuple), so the
    mapping is exact: ``dict`` -> ``MappingProxyType``, ``list`` -> ``tuple``.
    Used for the one row cache that every documented block (population, the
    causal pool, every widening step) is built FROM and that persists for the
    matcher's whole life, shared across every candidate that reuses a bucket
    — corrupting it in place would corrupt every OTHER candidate's evidence
    silently. Paired with :func:`_thaw` at the point a caller needs an
    independent, JSON-``dumps``-able copy back.
    """
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    """Invert :func:`_freeze`: a fresh, independent, mutable copy."""
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _evidence_rows(
    frame: pd.DataFrame, *, cache: dict[Any, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Copy rows and assign deterministic identities for one evidence block.

    ``cache``, keyed by each row's pandas index, is optional and lets several
    calls whose frames are nested/overlapping VIEWS of the same underlying
    trade population — as ``match()``'s population, causal pool, and every
    widening step are, AND as several match() calls for different candidates
    that happen to share a (strategy, alpha[, as_of]) bucket are — share one
    row's ``(digest, values)`` instead of each rebuilding it. Without it, a
    wide bucket that widens through several steps re-runs ``to_dict`` +
    ``json.dumps`` + ``sha256`` on the SAME rows once per step and retains an
    independent nested-dict copy of each row per call; measured on a 50k-row
    population widening to a 3,714-row match, that duplication is what turns
    a ~450 MB traced peak into a >1 GB RSS spike inside `AnalogMatcher.match`
    — big enough, on a process already resident at several GB, to exhaust an
    address-space cap before `_summarize`'s own (much smaller) bootstrap
    array can allocate. The digest and ``row_id`` numbering are unaffected:
    ``occurrence`` counting stays local to this call, so identical frames
    still produce byte-identical output whether or not a cache is supplied.

    A cache passed by ``AnalogMatcher`` persists for the matcher's whole
    life, so it is shared across DIFFERENT ``match()`` calls too — a strict
    capture run rescores the same boundary event's pinned/strike/coarse-ladder
    variants, which usually share a bucket. Each cache entry's ``values`` is
    therefore stored `_freeze`-locked (read-only, recursively): mutating it in
    place would silently corrupt every OTHER candidate that reused it, not
    just the caller's own copy. Every row this function RETURNS still gets an
    independent, ordinary, mutable, directly ``json.dumps``-able dict/list
    (via `_thaw`) — the lock protects the shared cache, not the caller.
    """
    row_cache: dict[Any, dict[str, Any]] = {} if cache is None else cache

    def _compute(source_index, series) -> dict[str, Any]:
        values = _json_value(series.to_dict())
        identity_payload = {
            "source_index": _json_value(source_index),
            "values": values,
        }
        encoded = json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        return {
            "digest": digest,
            "source_index": identity_payload["source_index"],
            "values": _freeze(values),
        }

    # `.iterrows()` boxes every column of every row into a fresh Series (a
    # real cost on a wide frame), so a cache hit must skip it, not just skip
    # the json/hash work below it. Safe only when the frame's own index has
    # no duplicate labels — otherwise `.loc[missing]` could return more rows
    # than requested, so duplicate-indexed frames fall back to the original
    # row-by-row path (still cache-aware, just without the `.iterrows()` skip).
    if not frame.index.has_duplicates:
        missing = [idx for idx in frame.index if idx not in row_cache]
        if missing:
            # dict.fromkeys: unique, order-preserving — `.loc` on a label list
            # with a repeat would otherwise fetch that label's row twice.
            subset = frame.loc[list(dict.fromkeys(missing))]
            for source_index, series in subset.iterrows():
                row_cache[source_index] = _compute(source_index, series)
    else:
        for source_index, series in frame.iterrows():
            if source_index not in row_cache:
                row_cache[source_index] = _compute(source_index, series)

    rows = []
    occurrences: dict[str, int] = {}
    for source_index in frame.index:
        cached = row_cache[source_index]
        digest = cached["digest"]
        occurrence = occurrences.get(digest, 0)
        occurrences[digest] = occurrence + 1
        row_id = f"sha256:{digest}:{occurrence}"
        rows.append({
            "row_id": row_id,
            "source_index": cached["source_index"],
            "values": _thaw(cached["values"]),
        })
    return {
        "row_ids": [row["row_id"] for row in rows],
        "rows": rows,
    }


class _AnalogMatchEvidence:
    """Collect opt-in matching evidence without affecting the default path.

    ``population``/``causal`` are rebuilt fresh on every call (no matcher-
    level cache of the DOCUMENTED block itself — see `AnalogMatcher.__init__`
    for why that layer was removed). What persists across candidates is only
    the ROW-LEVEL cache (`_population_row_caches`/`_causal_row_caches`,
    passed in as ``row_cache``): the expensive part of documenting a row —
    `to_dict` + `json.dumps` + `sha256`, run once per distinct pandas index —
    is memoized there, `_freeze`-locked, for the matcher's whole life. What
    this class rebuilds per call is comparatively cheap: an O(rows) loop of
    dict lookups against that warm cache plus one `_thaw` per row, needed
    because every consumer of an evidence document (this repo's whole test
    suite, and the `AnalogMatcher.match` docstring) is entitled to a plain,
    independently mutable, directly `json.dumps`-able dict back — the SAME
    guarantee `selected`/`contributing` below have always given.
    """

    def __init__(
        self,
        *,
        hook: AnalogEvidenceHook,
        strategy: str,
        alpha: float,
        snapshot: str,
        cutoff,
        request_key: str,
        bucket_query: dict,
        population: pd.DataFrame,
        row_cache: dict[Any, dict[str, Any]] | None = None,
    ):
        self.hook = hook
        self.strategy = strategy
        self.alpha = alpha
        self.snapshot = snapshot
        self.cutoff = (
            pd.Timestamp(cutoff).normalize().isoformat() if cutoff is not None else None
        )
        self.request_key = request_key
        self.bucket_query = _json_value(bucket_query)
        # Row-level cache, scoped to POPULATION content only (see the class
        # docstring below for why this must not be shared with the causal
        # pool's own row cache). Used for this one _evidence_rows() call,
        # then `set_causal` swaps `self._row_cache` to the causal-scoped one
        # for every call that follows (add_step, emit) — they all read
        # `pool`/`matched`, which are causal-provenance frames, never
        # `population`.
        self._row_cache: dict[Any, dict[str, Any]] = (
            {} if row_cache is None else row_cache
        )
        self.population = _evidence_rows(population, cache=self._row_cache)
        self.causal = None
        self.effective_bucket_query = None
        self.causal_implied_edges = None
        self.widening_steps: list[dict[str, Any]] = []

    def set_causal(
        self,
        pool: pd.DataFrame,
        bucket_query: dict,
        implied_edges: tuple[float, float] | None,
        *,
        row_cache: dict[Any, dict[str, Any]] | None = None,
    ) -> None:
        # Swap to a row cache scoped to THIS causal_key from here on: `pool`
        # carries an `implied_tercile` recomputed from causal (as_of-
        # dependent) edges via `.assign()` in `match()`, which is DIFFERENT
        # content for the same row index than the population's own
        # `implied_tercile` (population-wide edges) — and different again
        # across two causal pools with different `as_of`. Reusing the
        # population's row cache here, or one causal_key's cache for
        # another, would silently serve a row with the wrong tercile label
        # (a real bug caught in review: content, not just cost, depends on
        # which frame/as_of a row's cached value came from).
        self._row_cache = {} if row_cache is None else row_cache
        self.causal = _evidence_rows(pool, cache=self._row_cache)
        self.effective_bucket_query = _json_value(bucket_query)
        self.causal_implied_edges = _json_value(implied_edges)

    def add_step(
        self,
        active: list[str] | tuple[str, ...],
        dropped: list[str] | tuple[str, ...],
        matched: pd.DataFrame,
        *,
        accepted: bool,
    ) -> None:
        block = _evidence_rows(matched, cache=self._row_cache)
        self.widening_steps.append({
            "step": len(self.widening_steps),
            "active_dimensions": list(active),
            "dropped_dimensions": list(dropped),
            "row_ids": block["row_ids"],
            "count": len(block["row_ids"]),
            "accepted": accepted,
        })

    def emit(self, matched: pd.DataFrame, result: AnalogSet) -> None:
        numeric_returns = pd.to_numeric(matched.get("ret"), errors="coerce")
        finite = np.isfinite(numeric_returns.to_numpy(dtype=float))
        contributing = matched.loc[finite]
        self.hook({
            "schema_version": "analog_match_evidence.v1",
            "strategy": self.strategy,
            "alpha": self.alpha,
            "snapshot": self.snapshot,
            "cutoff": self.cutoff,
            "request_key": self.request_key,
            "bucket_query": self.bucket_query,
            "effective_bucket_query": self.effective_bucket_query,
            "causal_implied_edges": self.causal_implied_edges,
            "population": self.population,
            "causal": self.causal,
            "widening_steps": self.widening_steps,
            "selected": _evidence_rows(matched, cache=self._row_cache),
            "contributing": _evidence_rows(contributing, cache=self._row_cache),
            "result": result.as_dict(),
        })


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
        #
        # 2026-09-18: this cache and `_population_row_caches` below are now
        # priced and evicted alongside the causal family, under the SAME
        # `CAUSAL_CACHE_BUDGET_BYTES` total — see that constant's docstring.
        # `_population_pool_order` tracks recency across ONLY this small
        # (strategy, alpha)-keyed family, separately from `_causal_pools`'
        # own dict-order recency, because the two key shapes (2-tuple vs.
        # 3-tuple) are eviction-ordered independently:
        # `_evict_analog_cache_until_under_budget` empties the (usually much
        # larger, cheaper-to-rebuild) causal family first, and only reaches
        # into this one if the total is still over budget once causal is
        # gone — avoiding repeated, expensive full-population re-documentation
        # thrash on a key space this small.
        self._pools: dict[tuple[str, float], pd.DataFrame] = {}
        self._population_pool_order: dict[tuple[str, float], None] = {}
        # Causal pools: the as_of-filtered, causally re-bucketed pool depends
        # only on (strategy, alpha, as_of), and board rows share all three.
        # Without this cache the quantile + re-bucket cost is paid per row —
        # ~19s on a 3,120-row board.
        self._causal_pools: dict[tuple[str, float, pd.Timestamp],
                                 tuple[pd.DataFrame, tuple[float, float] | None]] = {}
        # Row-level evidence cache — opt-in, populated only when a caller
        # supplies an evidence_hook (Phase 4 strict capture), never on the
        # default scoring path. Persists for the matcher's whole life,
        # exactly like `_pools`/`_causal_pools` above, and is keyed the SAME
        # way, so candidates that rescore the same boundary event's variants
        # (which usually share a bucket) reuse one row's cached
        # `(digest, values)` instead of each re-serializing it — see
        # `_evidence_rows`' docstring for the measured effect.
        #
        # This is now the ONLY documented-content cache `AnalogMatcher` owns.
        # An earlier version also cached the fully-built, per-key DOCUMENTED
        # `population`/`causal` blocks themselves (`_documented_pools`/
        # `_documented_causal`) — removed 2026-09-18: a 40-forward-event
        # strict capture visits close to one distinct causal block per event,
        # and each retained block held its OWN independently `_thaw`ed
        # (deep-copied) values on top of what THIS row cache already holds
        # frozen, measured at 480.9 MB / 38 keys = 12.7 MB/key of pure
        # duplication for `_documented_causal` alone (plus 113.2 MB, flat, in
        # `_documented_pools`) in the run that diagnosed this file's other
        # fix. Nothing downstream needed that retained copy to persist past
        # the ONE candidate that built it: `_AnalogMatchEvidence.population`/
        # `.causal` are now rebuilt fresh on every call — cheap, an O(rows)
        # loop of lookups against this already-warm cache plus one `_thaw`
        # per row, not the `to_dict`+`json.dumps`+`sha256` work the row cache
        # itself memoizes — and the one caller that DOES need cross-candidate
        # sharing of a causal block's content (`Phase4TraceCollector
        # .capture_analog_inputs`) gets it from `phase4_recipe_cache` below,
        # which was already a separate, correctly-shared cache.
        #
        # Row caches are scoped PER documented-block key, not shared
        # globally by pandas index. Content, not just cost, differs by
        # provenance: `match()` recomputes `implied_tercile` on the causal
        # pool from CAUSAL (as_of-dependent) edges via `.assign()`, which
        # differs from the population's own `implied_tercile` (population-
        # wide edges, set once in `bucket_frame`) for the SAME row index —
        # and differs again between two causal pools at different `as_of`.
        # A single index-keyed cache shared across all of these would serve
        # a row with whichever tercile label happened to be cached first,
        # silently wrong for every other caller. One row cache per
        # population key and one per causal key keeps each key's rows
        # self-consistent while still sharing across every candidate that
        # matches the SAME key.
        self._population_row_caches: dict[tuple[str, float], dict[Any, dict[str, Any]]] = {}
        self._causal_row_caches: dict[
            tuple[str, float, pd.Timestamp | None], dict[Any, dict[str, Any]]
        ] = {}
        #: Phase 4 strict-capture only: `engine.score.Phase4TraceCollector
        #: .capture_analog_inputs` derives a trimmed, bucket-dimension-only
        #: projection of the FULL causal block above (every row, not just the
        #: matched subset) for its replayable recipe, and used to rebuild that
        #: projection — and retain a fresh copy of it — for every candidate,
        #: even candidates sharing this matcher's own (strategy, alpha, as_of)
        #: causal block by reference. A 40-forward-event strict capture visits
        #: close to one distinct causal block per event, so the growth was
        #: roughly linear in candidates scored, not bounded by anything: the
        #: measured driver of the forward-pass RSS climb this cache fixes.
        #: Keyed and evicted in lockstep with `_causal_pools`/
        #: `_causal_row_caches` above (same key shape); the trace collector
        #: only ever reads/writes through `capture_analog_inputs`, never
        #: iterates it directly.
        #: Value shape is `(population_hash, documented_rows)` — see
        #: `Phase4TraceCollector.capture_analog_inputs` for what populates it.
        #: This is the one row-content cache below `_causal_pools` that is
        #: NOT redundant with the row cache above: it holds a different,
        #: smaller, 5-field-per-row PROJECTION (not the full ~11-field row),
        #: and is the one place cross-candidate sharing of that projection
        #: actually happens — see the block comment above for what USED to
        #: also live here (the full-row `_documented_causal` cache) and why
        #: it was removed instead of kept alongside this one.
        self.phase4_recipe_cache: dict[
            tuple[str, float, pd.Timestamp | None], tuple[str | None, Any]
        ] = {}
        #: 2026-09-18, revised: a fixed KEY-COUNT ceiling cannot bound this
        #: cache's memory, because per-key cost is not fixed. The comment
        #: this replaced quoted "16.6 MB/key combined" from one run's
        #: average — but a real 40-forward-event strict capture (run 3)
        #: measured +5 keys costing +770 MB in one step and, later in the
        #: SAME run, +30 keys costing +350 MB in another: roughly 10x
        #: variance in per-key cost depending on how large that as_of's
        #: matched population is. A 64-key cap sized to the SMALL end of
        #: that range does nothing to bound a run that happens to visit the
        #: large end; that gap is what let a 5.5 GB bounded-run cap breach
        #: even after `_documented_pools`/`_documented_causal` were removed
        #: and every candidate's checkpoint content stopped being pinned in
        #: `tools/capture_tier0_corpus.py`'s `candidates` list (see that
        #: module's spill mechanism).
        #:
        #: Eviction is now byte-aware and least-recently-used instead:
        #: `_causal_pools`' own DataFrame is priced exactly
        #: (`memory_usage(deep=True)`) at insertion; `_causal_row_caches`/
        #: `phase4_recipe_cache`, which grow lazily row-by-row and are not
        #: yet full at insertion time, are priced by `len(pool)` (the key's
        #: whole causal population — a conservative upper bound; a widened
        #: match can reach it entirely) times a measured average bytes/row
        #: (`CAUSAL_ROW_CACHE_BYTES_PER_ROW`, `RECIPE_CACHE_BYTES_PER_ROW`,
        #: module level). `_evict_analog_cache_until_under_budget` pops the
        #: least-recently-touched key — `match()` now refreshes a key's
        #: position on every cache HIT, not just on insert — until the
        #: combined estimate is back under `CAUSAL_CACHE_BUDGET_BYTES`
        #: (600 MB). `MAX_CAUSAL_CACHE` remains as a loose, much higher
        #: backstop against key-count/dict overhead in a degenerate
        #: many-tiny-keys workload; the byte budget is what actually binds
        #: for every workload measured so far.
        #:
        #: Eviction only costs RECOMPUTATION, never a different answer: a
        #: cache miss on an evicted key re-filters/re-buckets `_causal_pools`'
        #: entry (the ~19s/3,120-row-board cost this cache exists to avoid
        #: paying per candidate, scaled down to one as_of) and re-runs
        #: `to_dict`/`json.dumps`/`sha256` for whichever rows the NEXT
        #: candidate on that key touches (`_causal_row_caches`' own memoized
        #: work) — bounded, one-time-per-revisit costs, not a content
        #: change: a re-served row is byte-identical to the one evicted.
        self.MAX_CAUSAL_CACHE = 512
        self._causal_cache_bytes: dict[
            tuple[str, float, pd.Timestamp | None], int
        ] = {}
        self._causal_cache_total_bytes = 0

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

    # -- analog-family cache eviction ----------------------------------------

    def _evict_analog_cache_until_under_budget(self) -> None:
        """Pop the least-recently-touched key — causal family first, then
        population — until the combined estimated size is back at or under
        `CAUSAL_CACHE_BUDGET_BYTES`.

        `_causal_pools` is iterated in insertion order, which `match()` keeps
        equal to recency order by deleting and re-inserting a key on every
        cache HIT (see the call site) — so `next(iter(...))` is always the
        least-recently-used key, not just the oldest-inserted one. The
        population family's own `_population_pool_order` gets the identical
        treatment. Eviction never changes an answer: a re-visited key
        rebuilds byte-identical content (see `MAX_CAUSAL_CACHE`'s comment for
        the recompute cost).

        Causal keys are emptied FIRST, even though both families share one
        budget check: there are usually many more of them, each far cheaper
        to rebuild than a population key's full-population documentation
        (see `CAUSAL_CACHE_BUDGET_BYTES`'s docstring) — so a workload that
        fits by evicting a handful of causal entries never pays a population
        re-documentation cost it did not need to.
        """
        while (self._causal_pools
               and (self._causal_cache_total_bytes > CAUSAL_CACHE_BUDGET_BYTES
                    or len(self._causal_pools) > self.MAX_CAUSAL_CACHE)):
            evicted = next(iter(self._causal_pools))
            self._causal_pools.pop(evicted)
            # Kept in lockstep with _causal_pools: a row cache or
            # recipe-cache entry for a pool that no longer exists would grow
            # unbounded, never evicted, for a scan that touches many as_of
            # dates (recalibrate.build_pairs, the calibration sampler).
            self._causal_row_caches.pop(evicted, None)
            self.phase4_recipe_cache.pop(evicted, None)
            self._causal_cache_total_bytes -= self._causal_cache_bytes.pop(evicted, 0)
        while (self._population_pool_order
               and self._causal_cache_total_bytes > CAUSAL_CACHE_BUDGET_BYTES):
            evicted = next(iter(self._population_pool_order))
            del self._population_pool_order[evicted]
            self._pools.pop(evicted, None)
            self._population_row_caches.pop(evicted, None)
            self._causal_cache_total_bytes -= self._causal_cache_bytes.pop(evicted, 0)

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
        evidence_hook: AnalogEvidenceHook | None = None,
    ) -> AnalogSet:
        """Matched returns, widening the buckets until there are enough.

        ``evidence_hook`` is opt-in and receives one self-contained evidence
        document after a successful match. Default callers take the existing
        path and allocate no row evidence. Every row's underlying trade
        value is protected at the source (`_evidence_rows`' row cache
        `_freeze`-locks it) no matter how many places reuse it.

        ``population``, ``causal``, ``selected``, and ``contributing`` are
        ALL rebuilt fresh per call — ordinary, independently mutable
        defensive copies, safe for a caller to hold onto and mutate without
        affecting any other candidate. What is actually shared BY REFERENCE
        across every candidate that matches the same (strategy, alpha[,
        as_of]) bucket on this matcher is one level down, invisible to the
        evidence document itself: the ROW cache each of those four blocks is
        built FROM (`_population_row_caches`/`_causal_row_caches`) memoizes
        the expensive part — `to_dict` + `json.dumps` + `sha256` per pandas
        row — so a strict-trace capture rescoring one boundary event's
        pinned/strike/coarse-ladder variants (which usually share a bucket)
        pays that cost once, not once per candidate. (An earlier version of
        this matcher also cached ``population``/``causal`` themselves by
        reference, at the block level — removed 2026-09-18 after it was
        measured to retain a fully independent, deep-copied duplicate of
        every row on top of what the row cache already held; see
        `AnalogMatcher.__init__`'s comment on `_causal_row_caches`. Content
        is unaffected: two candidates sharing a key still get byte-identical
        ``population``/``causal`` JSON, just as two independent objects
        instead of one shared one — `Phase4TraceCollector
        .capture_analog_inputs`, the one caller that needs cross-candidate
        sharing of a causal block's content, gets it from its own
        `phase4_recipe_cache`, unaffected by this change.)
        """
        key = (strategy, round(float(alpha), 4))
        base = self._pools.get(key)
        if base is None:
            base = self.trades[
                (self.trades["strategy"] == strategy)
                & np.isclose(self.trades["fill_alpha"].astype(float), float(alpha))
            ]
            self._pools[key] = base
            # Priced and evicted alongside the causal family — see
            # `CAUSAL_CACHE_BUDGET_BYTES`'s docstring for why this key was
            # previously invisible to that budget despite being real,
            # resident memory. Charged unconditionally at first creation
            # (not gated on `evidence_hook`), matching the causal branch's
            # own conservative philosophy below: an untraced caller still
            # creates and keeps this same `_pools[key]` entry, so the worst
            # case (a later traced caller fully documents it) is priced in
            # from the start rather than only once that later call arrives.
            row_count = len(base)
            estimate = (
                int(base.memory_usage(deep=True).sum())
                + row_count * CAUSAL_ROW_CACHE_BYTES_PER_ROW
            )
            self._causal_cache_bytes[key] = estimate
            self._causal_cache_total_bytes += estimate
            self._evict_analog_cache_until_under_budget()
        else:
            # Refresh recency exactly as `_causal_pools`' own cache-hit
            # branch does, below.
            del self._population_pool_order[key]
        self._population_pool_order[key] = None
        pool = base
        evidence = (
            _AnalogMatchEvidence(
                hook=evidence_hook,
                strategy=strategy,
                alpha=float(alpha),
                snapshot=self.snapshot,
                cutoff=as_of,
                request_key=request_key,
                bucket_query=buckets,
                population=base,
                row_cache=self._population_row_caches.setdefault(key, {}),
            )
            if evidence_hook is not None else None
        )
        # No as_of -> the "causal" pool is just the population, unfiltered.
        # Keyed with a `None` third slot rather than left un-cacheable, so
        # repeated no-as_of candidates on the same bucket still share one
        # documented block instead of each re-serializing the population a
        # second time under a different evidence field name.
        causal_key = (strategy, round(float(alpha), 4), None)
        causal_edges = None
        if as_of is not None:
            # Closed strictly before the decision: a trade still open on the day
            # we decide has not yet told us anything about how it went. The
            # causally filtered AND re-bucketed pool depends only on
            # (strategy, alpha, as_of), so it is cached — many board rows share
            # the triple, and recomputing the quantile + bucket per row cost
            # ~19s on a 3,120-row board.
            ts = pd.Timestamp(as_of).normalize()
            cache_key = (strategy, round(float(alpha), 4), ts)
            causal_key = cache_key
            cached = self._causal_pools.get(cache_key)
            if cached is not None:
                pool, edges = cached
                # Refresh recency: a plain dict already preserves insertion
                # order, and deleting + re-inserting the SAME key moves it to
                # the end (most-recently-used), which is all the LRU eviction
                # below needs — no OrderedDict required.
                del self._causal_pools[cache_key]
                self._causal_pools[cache_key] = cached
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
                self._causal_pools[cache_key] = (pool, edges)
                row_count = len(pool)
                estimate = (
                    int(pool.memory_usage(deep=True).sum())
                    + row_count * CAUSAL_ROW_CACHE_BYTES_PER_ROW
                    + row_count * RECIPE_CACHE_BYTES_PER_ROW
                )
                self._causal_cache_bytes[cache_key] = estimate
                self._causal_cache_total_bytes += estimate
                self._evict_analog_cache_until_under_budget()
            ratio = buckets.get("implied_ratio")
            if ratio is not None and edges is not None:
                buckets = dict(buckets)
                buckets["implied_tercile"] = _bucket([ratio], edges,
                                                     ("low", "mid", "high"))[0]
            causal_edges = edges
        if evidence is not None:
            evidence.set_causal(
                pool, buckets, causal_edges,
                row_cache=self._causal_row_caches.setdefault(causal_key, {}),
            )

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
            result = self._summarize(
                self.trades.iloc[0:0], strategy, alpha, buckets, 0,
                list(unavailable), bootstrap=bootstrap,
                min_analogs=min_analogs, request_key=request_key,
                unavailable=tuple(unavailable),
            )
            if evidence is not None:
                empty = self.trades.iloc[0:0]
                evidence.add_step((), unavailable, empty, accepted=True)
                evidence.emit(empty, result)
            return result

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
            accepted = len(matched) >= min_analogs or not remaining
            if evidence is not None:
                evidence.add_step(active, dropped, matched, accepted=accepted)
            if accepted:
                result = self._summarize(
                    matched, strategy, alpha, buckets, len(dropped), dropped,
                    bootstrap=bootstrap, min_analogs=min_analogs,
                    request_key=request_key, unavailable=tuple(unavailable),
                )
                if evidence is not None:
                    evidence.emit(matched, result)
                return result
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
        # SORTED, because the bootstrap below samples by INDEX. `rng.choice`
        # with a fixed seed picks the same positions every time, so the same
        # analogs arriving in a different row order produce a different
        # resample — and `matched`'s order is not guaranteed, it falls out of
        # how the trades were gathered and merged.
        #
        # Every other statistic here is order-invariant (mean, median, win
        # rate, quantiles), which is exactly why this hid: on 2026-09-11 the
        # board and its own re-score agreed on n_analogs, exp_pnl_analog and
        # win_analog and disagreed on ci_low/ci_high, for the same analog set.
        # Sorting makes the interval a function of the multiset, which is what
        # it was always supposed to be.
        returns = np.sort(returns)
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


def match_frame(
    trades: pd.DataFrame,
    matcher: AnalogMatcher,
    *,
    strategy: str,
    alpha: float = 0.5,
    as_of_column: str = "entry_date",
    min_analogs: int = MIN_ANALOGS,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Per-row matched analog statistics for a whole trade frame.

    One :meth:`AnalogMatcher.match` call per row (``bootstrap=0`` — a batch
    caller wants the point estimates, not a confidence interval, and skipping
    the bootstrap is what keeps a full-universe pass fast), with ``as_of`` set
    to that ROW's own decision date. ``match`` already restricts its pool to
    trades closed strictly before that date, so the cutoff is per-row, not a
    caller-supplied global one — which is what makes this safe to compute
    ONCE, outside any walk-forward loop, and reuse in every fold: a fold
    boundary drawn later can never un-close a trade that had already closed
    before a given row's own decision date.

    ``trades`` must already carry :func:`bucket_frame`'s columns
    (``mcap_bucket``, ``dte_band``, ``moneyness_band``, ``implied_ratio``) —
    callers that built ``matcher`` from an already-bucketed frame already have
    them; call :func:`bucket_frame` on ``trades`` first otherwise.

    Returns one row per unique ``event_id`` at the requested ``alpha``:
    ``event_id``, ``analog_mean``, ``analog_win_rate``, ``analog_n``,
    ``analog_widened``, ``analog_thin``.
    """
    mid = trades[np.isclose(pd.to_numeric(trades["fill_alpha"]), alpha)]
    mid = mid.drop_duplicates("event_id")
    records: list[dict] = []
    n = len(mid)
    for i, row in enumerate(mid.itertuples(index=False)):
        buckets = {
            "mcap_bucket": row.mcap_bucket,
            "dte_band": row.dte_band,
            "moneyness_band": row.moneyness_band,
            "implied_tercile": row.implied_tercile,
            "implied_ratio": row.implied_ratio,
        }
        aset = matcher.match(
            strategy, buckets, alpha=alpha, as_of=getattr(row, as_of_column),
            bootstrap=0, min_analogs=min_analogs, request_key=str(row.event_id),
        )
        records.append({
            "event_id": row.event_id,
            "analog_mean": aset.mean,
            "analog_win_rate": aset.win_rate,
            "analog_n": aset.n,
            "analog_widened": aset.widened,
            "analog_thin": aset.thin,
        })
        if progress_every and ((i + 1) % progress_every == 0 or i + 1 == n):
            print(f"  [analogs] {i + 1:,}/{n:,} matched", flush=True)
    return pd.DataFrame.from_records(
        records,
        columns=["event_id", "analog_mean", "analog_win_rate", "analog_n",
                 "analog_widened", "analog_thin"],
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
