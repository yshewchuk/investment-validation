"""Memory regression coverage for the Phase 4 strict-capture OOM.

Symptom (measured on real captures, not reproduced here — see
guides/rearchitecture_phase4_scoring.md history / commit message): a full
Phase 4 strict capture died with `_ArrayMemoryError` inside
`AnalogMatcher._summarize`'s bootstrap line, under BOTH a 5.5 GB and a 6.5 GB
address-space cap, while RSS stayed ~3.4 GB. Measured on synthetic inputs of
the same shape (this file, `tracemalloc` + `resource.getrusage`):

  * `_summarize`'s bootstrap alone (n=3,714 returns, bootstrap=2,000) traces
    ~119 MB peak — nowhere near 1 GB, and comfortably under the 200 MB bound
    asserted below.
  * The actual >1 GB-class spike is inside `AnalogMatcher.match`'s opt-in
    evidence path (`_AnalogMatchEvidence`, wired through `_score_analogs`'s
    `evidence_hook` — set ONLY when `Phase4TraceCollector` traces a score,
    i.e. only during strict capture, never in ordinary/nightly scoring).
    `match()` builds up to five overlapping/nested `_evidence_rows` blocks
    per request (population, the causal pool, one per widening step, plus
    `emit`'s `selected`/`contributing`) — each one, before this fix,
    independently re-serialized and separately retained the SAME rows.

The fix (`_evidence_rows`'s `cache` parameter, threaded through
`_AnalogMatchEvidence`) shares one row's `(digest, values)` — and, for
duplicate-free frames, skips `DataFrame.iterrows()` entirely — across all of
those overlapping calls within one `match()` invocation. It touches only the
opt-in evidence path: `_summarize`'s RNG stream, its inputs, and therefore
`mean`/`median`/`win_rate`/`p10`/`p90`/`ci_low`/`ci_high` are untouched by
construction (the cache lives on `_AnalogMatchEvidence`, never on the
`returns` array or the `rng.choice` call), and are proven identical below
whether or not the evidence path runs.

**Retention across candidates** (added after review, mirroring
`engine.pnl_sim.ResidualPool.documented_population` /
`engine.score._Predocumented`, fixed for the same class of bug in commit
53d2f8a): a strict-trace capture rescores one boundary event's pinned/
strike/coarse-ladder variants many times, and they usually share a
(strategy, alpha[, as_of]) bucket. The row-level cache (`_evidence_rows`'
`cache=`) moved from per-`match()`-call to per-`AnalogMatcher`
(matcher-lifetime, `_population_row_caches`/`_causal_row_caches`), so
population/causal/widening/selected/contributing blocks that only PARTIALLY
overlap across candidates all still share a row's `(digest, values)` instead
of each re-serializing it. Mutation safety: each row cache entry's
``values`` is `_freeze`-locked (recursively read-only) the moment it is
computed — corrupting the ONE place a row's value is built would otherwise
corrupt every candidate that ever reuses it — and `_evidence_rows` always
`_thaw`s a fresh, independent, plain, `json.dumps`-able copy for whatever it
returns.

**2026-09-18 revision:** an earlier version of this fix ALSO cached the
fully-built, per-key DOCUMENTED `population`/`causal` blocks themselves
(`_documented_pools`/`_documented_causal`), sharing them BY REFERENCE across
every `match()` call on the same key — the shape this file originally
tested below. Measured on a real 40-forward-event strict capture, that
extra layer turned out to be pure duplication: each retained block held its
OWN independently `_thaw`ed (deep-copied) values on top of what the row
cache above already held frozen, worth 480.9 MB / 38 keys = 12.7 MB/key
(`_documented_causal`) plus 113.2 MB flat (`_documented_pools`) — nothing
downstream ever needed that copy to outlive the ONE candidate that built
it. It was removed: `population`/`causal` are now rebuilt fresh on every
`match()` call — cheap (an O(rows) loop of lookups against the already-warm
row cache, not the `to_dict`/`json.dumps`/`sha256` work that cache
memoizes), ordinary, independently mutable, safe to hold onto and mutate,
exactly like `selected`/`contributing` always were. The row-level cache
above is now the ONLY thing that persists across candidates; a caller that
explicitly retains many candidates' full `population`/`causal` blocks
simultaneously (unlike this repo's one real caller, `Phase4TraceCollector`,
which extracts a trimmed, separately-cached projection via
`capture_analog_inputs` and lets the rest become garbage) will see memory
scale with how many it keeps — see
`test_population_and_causal_row_values_share_by_reference_not_whole_block`
and `test_retention_across_20_same_bucket_candidates_stays_bounded` below
for what is and is not still bounded.
"""
from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from engine.analogs import AnalogMatcher, _evidence_rows


def _trades(n: int, *, seed: int = 0, n_extra_cols: int = 8) -> pd.DataFrame:
    """A single-bucket trade population: every row lands in the same
    mcap/dte/moneyness/implied bucket, so `match()` never needs to widen —
    isolating the evidence path's own cost from the widening ladder's.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "strategy": "STR-THRU",
        "fill_alpha": 0.5,
        "ret": rng.normal(0, 0.3, n),
        "mcap_usd": 5e9,
        "dte_entry": 5,
        "spot_entry": 100.0,
        "strike": 100.0,
        "or_implied": 5.0,
        "mean_prior_or_implied": 5.0,
        "event_date": pd.Timestamp("2015-01-01") + pd.to_timedelta(
            rng.integers(0, 2000, n), unit="D"),
        "exit_date": pd.Timestamp("2015-01-05") + pd.to_timedelta(
            rng.integers(0, 2000, n), unit="D"),
    })
    for c in range(n_extra_cols):
        frame[f"extra_{c}"] = rng.normal(0, 1, n)
    return frame


def _match(trades: pd.DataFrame, *, bootstrap: int, evidence_hook=None,
           min_analogs: int = 30):
    matcher = AnalogMatcher(trades.copy(), snapshot="synthetic-snapshot")
    buckets = matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0,
                                   implied_ratio=1.0)
    return matcher.match(
        "STR-THRU", buckets, alpha=0.5, as_of="2024-01-01",
        min_analogs=min_analogs, bootstrap=bootstrap,
        request_key="req-fixed", evidence_hook=evidence_hook,
    )


def _fields(result):
    return (
        result.n, result.mean, result.median, result.win_rate, result.p10,
        result.p90, result.ci_low, result.ci_high, result.widened,
        result.thin,
    )


@pytest.mark.parametrize("n", [3714, 30, 31, 29, 1])
def test_evidence_path_is_bit_identical_to_default_path(n):
    """The evidence hook must not perturb the bootstrap stream or stats.

    n=3714 is the crash's own bucket size; 30/31 straddle MIN_ANALOGS (the
    thin/not-thin boundary, where a one-row shift changes whether the
    bootstrap runs at all); 29 stays thin (no bootstrap, ci_low/high None);
    1 is the degenerate single-row case.
    """
    trades = _trades(n, seed=n)

    default = _match(trades, bootstrap=2000, evidence_hook=None)

    captured = []
    hooked = _match(trades, bootstrap=2000, evidence_hook=captured.append)

    assert _fields(hooked) == _fields(default)
    if n >= 30:
        assert len(captured) == 1
        assert hooked.ci_low is not None and hooked.ci_high is not None
    else:
        assert hooked.thin is True
        assert hooked.ci_low is None and hooked.ci_high is None


def test_summarize_bootstrap_peak_memory_under_200mb_for_n_3714():
    """`_summarize`'s own bootstrap (the line the crash traceback names) —
    exercised through the DEFAULT path (no evidence hook, i.e. what every
    ordinary/nightly score already runs) — traces well under 200 MB for the
    crash's own bucket size. This is the empirical basis for concluding
    `_summarize` was never the >1 GB contributor; `match`'s evidence path was.
    """
    trades = _trades(3714, seed=1)

    gc.collect()
    tracemalloc.start()
    result = _match(trades, bootstrap=2000, evidence_hook=None)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.n == 3714
    assert result.ci_low is not None
    assert peak < 200 * 1_000_000, f"traced peak {peak / 1e6:.1f} MB >= 200 MB"


def test_evidence_rows_cache_matches_uncached_content():
    """Sharing a cache across nested/overlapping frames must not change what
    gets reported — only how much work it costs to report it.
    """
    population = _trades(500, seed=2)
    causal = population.iloc[:400]
    step = causal.iloc[:150]

    uncached_population = _evidence_rows(population)
    uncached_causal = _evidence_rows(causal)
    uncached_step = _evidence_rows(step)

    shared_cache: dict = {}
    cached_population = _evidence_rows(population, cache=shared_cache)
    cached_causal = _evidence_rows(causal, cache=shared_cache)
    cached_step = _evidence_rows(step, cache=shared_cache)

    assert cached_population == uncached_population
    assert cached_causal == uncached_causal
    assert cached_step == uncached_step
    # The cache actually got reused across all three nested calls: no more
    # unique entries than the largest (population) has rows.
    assert len(shared_cache) == len(population)


def test_evidence_rows_duplicate_index_falls_back_and_matches():
    """A frame with a duplicate index label must still produce the same
    content as the no-cache path (the `.iterrows()`-skip fast path is only
    safe for a unique index; this proves the fallback is not just safe but
    correct).
    """
    frame = _trades(20, seed=3)
    frame.index = pd.Index([0] * 5 + list(range(5, 20)))
    assert frame.index.has_duplicates

    uncached = _evidence_rows(frame)
    cached = _evidence_rows(frame, cache={})
    assert cached == uncached
    assert len(cached["row_ids"]) == len(frame)


def test_evidence_rows_cache_cuts_peak_memory_for_overlapping_calls():
    """Directional check: on a population wide/large enough for the
    duplication to matter, sharing one cache across the population + causal
    + widening-step-shaped sequence a real `match()` call makes must use
    meaningfully less peak memory than repeating the same calls independently
    (the pre-fix shape). Not tied to an absolute bound — machines vary — only
    to the relative improvement the cache is supposed to buy.
    """
    n = 8000
    population = _trades(n, seed=4, n_extra_cols=14)
    causal = population.iloc[: int(n * 0.9)]
    steps = [causal.iloc[:size] for size in
             (int(n * 0.05), int(n * 0.15), int(n * 0.4), len(causal))]

    def independent_calls():
        # Every block kept alive at once, as `_AnalogMatchEvidence` keeps
        # self.population / self.causal / self.widening_steps alive for the
        # life of one `match()` call — the OLD (pre-fix) call shape, since
        # each call below gets no cache and builds its own throwaway one.
        retained = [_evidence_rows(population), _evidence_rows(causal)]
        retained += [_evidence_rows(step) for step in steps]
        return retained

    def shared_cache_calls():
        cache: dict = {}
        retained = [
            _evidence_rows(population, cache=cache),
            _evidence_rows(causal, cache=cache),
        ]
        retained += [_evidence_rows(step, cache=cache) for step in steps]
        return retained

    gc.collect()
    tracemalloc.start()
    kept_before = independent_calls()
    _current, peak_before = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del kept_before
    gc.collect()

    gc.collect()
    tracemalloc.start()
    kept_after = shared_cache_calls()
    _current, peak_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del kept_after

    assert peak_after < peak_before * 0.9, (
        f"peak_after={peak_after/1e6:.1f}MB not meaningfully below "
        f"peak_before={peak_before/1e6:.1f}MB"
    )


def test_default_scoring_path_never_builds_evidence(monkeypatch):
    """Guard against regressing the existing invariant: ordinary scoring
    (no evidence_hook) must never call `_evidence_rows` at all, cache or not.
    """
    import engine.analogs as analogs_module

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("default matching must not build evidence")

    monkeypatch.setattr(analogs_module, "_evidence_rows", fail_if_called)
    trades = _trades(50, seed=5)
    result = _match(trades, bootstrap=0, evidence_hook=None)
    assert result.n == 50


# -- retention across candidates (mirrors the residual-pool fix, 53d2f8a) --


def _population_and_causal_hashes(matcher: AnalogMatcher, buckets: dict) -> tuple[str, str]:
    import json

    captured = []
    matcher.match(
        "STR-THRU", buckets, alpha=0.5, as_of="2024-01-01", min_analogs=30,
        bootstrap=0, request_key="req-hash", evidence_hook=captured.append,
    )
    evidence = captured[0]
    pop_json = json.dumps(evidence["population"], sort_keys=True)
    causal_json = json.dumps(evidence["causal"], sort_keys=True)
    return pop_json, causal_json


def test_documented_population_and_causal_content_is_independent_of_sharing():
    """Sharing must not change WHAT is reported — two matchers built from the
    identical trades, one queried once and one queried 20 times (so its
    documented caches are exercised, not left empty), must report byte-
    identical population/causal JSON for the same bucket.
    """
    trades = _trades(3714, seed=42)
    buckets = AnalogMatcher(trades.copy()).buckets_for(
        mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)

    fresh_matcher = AnalogMatcher(trades.copy(), snapshot="synthetic-snapshot")
    fresh_pop_json, fresh_causal_json = _population_and_causal_hashes(fresh_matcher, buckets)

    reused_matcher = AnalogMatcher(trades.copy(), snapshot="synthetic-snapshot")
    for _ in range(20):
        reused_pop_json, reused_causal_json = _population_and_causal_hashes(
            reused_matcher, buckets)

    assert reused_pop_json == fresh_pop_json
    assert reused_causal_json == fresh_causal_json


def test_population_and_causal_row_values_share_by_reference_not_whole_block():
    """Since 2026-09-18, `population`/`causal` are rebuilt fresh on every
    `match()` call — same (strategy, alpha) no longer means the identical
    block object on a second candidate (that whole-block cache was the
    duplication removed; see the module docstring). What DOES still persist,
    invisibly, is the row-level cache each block is built FROM: a fresh
    `population`/`causal` still means byte-identical content, and the
    underlying per-row FROZEN `values` object the matcher holds is never
    rebuilt for a row it has already seen on this key.
    """
    trades = _trades(500, seed=6)
    matcher = AnalogMatcher(trades.copy(), snapshot="synthetic-snapshot")
    buckets = matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
    key = ("STR-THRU", 0.5)

    captured_a, captured_b = [], []
    matcher.match("STR-THRU", buckets, alpha=0.5, as_of="2024-01-01",
                   min_analogs=30, bootstrap=0, evidence_hook=captured_a.append)
    row_cache = matcher._population_row_caches[key]
    frozen_before = {idx: entry["values"] for idx, entry in row_cache.items()}
    size_before = len(row_cache)

    matcher.match("STR-THRU", buckets, alpha=0.5, as_of="2024-01-01",
                   min_analogs=30, bootstrap=0, evidence_hook=captured_b.append)

    # Content is byte-identical either way...
    assert captured_a[0]["population"] == captured_b[0]["population"]
    assert captured_a[0]["causal"] == captured_b[0]["causal"]
    # ...but each call gets its own independent, mutation-safe object now —
    # population/causal are ordinary defensive copies, exactly like
    # selected/contributing always were (test_analog_evidence.py's
    # test_evidence_rows_are_defensive_copies pins this at the value level).
    assert captured_a[0]["population"] is not captured_b[0]["population"]
    assert captured_a[0]["causal"] is not captured_b[0]["causal"]
    assert captured_a[0]["selected"] is not captured_b[0]["selected"]

    # The one thing that DOES persist by reference: the row cache neither
    # regrew nor rebuilt any row's frozen value for the second candidate.
    assert row_cache is matcher._population_row_caches[key]
    assert len(row_cache) == size_before
    assert all(
        row_cache[idx]["values"] is frozen for idx, frozen in frozen_before.items()
    )

    # A different bucket key must not reuse the first key's row cache at all.
    captured_c = []
    matcher.match("STR-THRU", buckets, alpha=0.75, as_of="2024-01-01",
                   min_analogs=30, bootstrap=0, evidence_hook=captured_c.append)
    assert ("STR-THRU", 0.75) in matcher._population_row_caches
    assert matcher._population_row_caches[("STR-THRU", 0.75)] is not row_cache


def test_evidence_row_cache_is_read_only():
    """The foundational per-row cache must reject in-place mutation outright
    (not merely rely on nobody trying) -- it is the one place a row's value
    is ever built, shared underneath every documented block.
    """
    trades = _trades(10, seed=8)
    matcher = AnalogMatcher(trades.copy(), snapshot="synthetic-snapshot")
    buckets = matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
    captured = []
    matcher.match("STR-THRU", buckets, alpha=0.5, as_of="2024-01-01",
                   min_analogs=1, bootstrap=0, evidence_hook=captured.append)

    assert len(matcher._causal_row_caches) > 0
    row_cache = next(iter(matcher._causal_row_caches.values()))
    assert len(row_cache) > 0
    cached_entry = next(iter(row_cache.values()))
    with pytest.raises(TypeError):
        cached_entry["values"]["ret"] = 999.0

    # The document handed to the hook, meanwhile, is an ordinary mutable
    # dict/list -- _thaw ran before it was returned.
    captured[0]["selected"]["rows"][0]["values"]["ret"] = 999.0


def test_row_caches_stay_bounded_across_20_same_bucket_candidates():
    """20 candidates sharing one bucket on ONE persistent matcher (how
    `Scorer.matcher` is actually used across a capture run) must not grow
    the row-level caches past ONE entry each, and must not rebuild an
    already-cached row's frozen value for candidate #2..20. (The
    per-candidate `population`/`causal` OUTPUT is independently rebuilt for
    every one of the 20 now — see
    `test_population_and_causal_row_values_share_by_reference_not_
    whole_block` — so a "kept 20 independent evidence blocks" memory
    comparison, the shape this test used before 2026-09-18, no longer shows
    the fix's effect: what stayed bounded is the expensive per-row work
    underneath that output, not the output itself.)
    """
    n = 3000
    trades = _trades(n, seed=9, n_extra_cols=12)
    matcher = AnalogMatcher(trades.copy())
    buckets = matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)
    causal_key = ("STR-THRU", 0.5, pd.Timestamp("2024-01-01").normalize())

    frozen_ids_by_call = []
    for _ in range(20):
        captured = []
        matcher.match(
            "STR-THRU", buckets, alpha=0.5, as_of="2024-01-01",
            min_analogs=30, bootstrap=0, evidence_hook=captured.append,
        )
        causal_row_cache = matcher._causal_row_caches[causal_key]
        frozen_ids_by_call.append(
            {idx: id(entry["values"]) for idx, entry in causal_row_cache.items()}
        )

    assert len(matcher._population_row_caches) == 1
    assert len(matcher._causal_row_caches) == 1
    # Every one of the 20 calls saw the SAME row count and the SAME frozen
    # value objects for every row -- nothing was rebuilt for candidate #2..20.
    assert all(ids == frozen_ids_by_call[0] for ids in frozen_ids_by_call[1:])


def test_memory_across_n_distinct_as_of_keys_well_under_a_retained_extra_copy():
    """Synthetic reproduction of the REAL bug's growth axis: N DISTINCT
    as_of keys, not N candidates sharing one key (that shape was already
    bounded -- see the test above). A 40-forward-event strict capture visits
    close to one distinct causal block PER EVENT, so key count, not
    candidate count, was what grew unbounded in the real run (measured:
    `_causal_pools` reached 38 keys, well under the 64-entry cap, before the
    box's 5.5 GB ceiling killed it).

    This does not re-run the real capture (too heavy for a unit test); it
    reproduces the SHAPE of the duplication the removed `_documented_causal`
    cache used to add on top of today's row caches -- one independently
    `_thaw`ed full copy of every causal row, retained for the life of the
    run -- and asserts today's matcher (without that extra copy) uses well
    under what visiting the same N keys would cost WITH it.
    """
    n_keys = 8
    rows_per_key = 2500
    trades = _trades(rows_per_key, seed=11, n_extra_cols=10)
    buckets = AnalogMatcher(trades.copy()).buckets_for(
        mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)

    def visit_n_keys(matcher: AnalogMatcher, *, extra_copy: bool) -> list:
        kept = []
        for month in range(1, n_keys + 1):
            as_of = f"2024-{month:02d}-01"
            captured = []
            matcher.match(
                "STR-THRU", buckets, alpha=0.5, as_of=as_of,
                min_analogs=30, bootstrap=0, request_key=f"req-{month}",
                evidence_hook=captured.append,
            )
            kept.append(captured[0])
            if extra_copy:
                causal_key = ("STR-THRU", 0.5, pd.Timestamp(as_of).normalize())
                pool, _ = matcher._causal_pools[causal_key]
                row_cache = matcher._causal_row_caches[causal_key]
                # The extra retained copy `_documented_causal` used to add:
                # an independent `_thaw` of the SAME causal pool, over the
                # SAME (already-warm) row cache, kept alive for the run
                # instead of discarded like this function's local variable.
                kept.append(_evidence_rows(pool, cache=row_cache))
        return kept

    gc.collect()
    tracemalloc.start()
    kept_today = visit_n_keys(AnalogMatcher(trades.copy()), extra_copy=False)
    _current, peak_today = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del kept_today
    gc.collect()

    tracemalloc.start()
    kept_with_extra_copy = visit_n_keys(AnalogMatcher(trades.copy()), extra_copy=True)
    _current, peak_with_extra_copy = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del kept_with_extra_copy

    # Directional, not tied to an absolute bound (machines vary, and the
    # OTHER blocks a candidate retains -- population/selected/contributing,
    # unaffected by this fix and identical in both runs -- dilute how much
    # of the total one extra causal-sized copy per key can visibly move):
    # measured margin on this fixture is ~12%; 0.93 leaves headroom without
    # accepting a result indistinguishable from noise.
    assert peak_today < peak_with_extra_copy * 0.93, (
        f"peak_today={peak_today/1e6:.1f}MB not meaningfully under "
        f"peak_with_extra_copy={peak_with_extra_copy/1e6:.1f}MB "
        f"across {n_keys} distinct as_of keys"
    )


# -- row-cache content-provenance bug (review finding, fixed after fd34bcd) --
#
# The matcher-lifetime row cache in fd34bcd was keyed by pandas index alone.
# But `match()` recomputes `implied_tercile` on the causal pool from CAUSAL
# (as_of-dependent) edges via `.assign()` — different content for the same
# row index than the population's own `implied_tercile` (population-wide
# edges from `bucket_frame`), and different again between two causal pools
# at different `as_of`. An index-keyed cache shared across those frames
# silently serves whichever tercile label was cached first. The fix scopes
# one row cache per population key and one per causal key.


def _trades_with_trending_implied_ratio(n: int, *, seed: int = 0) -> pd.DataFrame:
    """Same single mcap/dte/moneyness bucket as `_trades`, but `implied_ratio`
    trends up with `event_date`, so: (a) the population-wide tercile edges
    (fit over ALL rows) differ from any as_of-restricted causal pool's edges,
    and (b) two causal pools with different `as_of` cutoffs — different
    prefixes of this trend — get different edges from EACH OTHER too. Both
    are needed to make a stale/wrong-provenance cache hit observable.
    """
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    event_date = pd.Timestamp("2015-01-01") + pd.to_timedelta(order, unit="D")
    frame = pd.DataFrame({
        "strategy": "STR-THRU",
        "fill_alpha": 0.5,
        "ret": rng.normal(0, 0.3, n),
        "mcap_usd": 5e9,
        "dte_entry": 5,
        "spot_entry": 100.0,
        "strike": 100.0,
        # Trends from ~1.0 to ~3.0 over the span; mean_prior fixed at 1.0, so
        # implied_ratio == or_implied trends the same way.
        "or_implied": 1.0 + 2.0 * (order / max(n - 1, 1)),
        "mean_prior_or_implied": 1.0,
        "event_date": event_date,
        "exit_date": event_date + pd.Timedelta(days=1),
    })
    return frame


def _causal_tercile_edges(trades: pd.DataFrame, as_of: str) -> tuple[float, float]:
    """Independent reference: the SAME computation `match()` does inline,
    recomputed from scratch (no cache of any kind) for one `as_of`.
    """
    ts = pd.Timestamp(as_of).normalize()
    causal = trades[trades["exit_date"] < ts]
    ratio = causal["or_implied"] / causal["mean_prior_or_implied"]
    finite = ratio[np.isfinite(ratio)]
    if len(finite) < 30:
        return (0.9, 1.1)
    return tuple(float(e) for e in np.quantile(finite, [1 / 3, 2 / 3]))


def _values_by_source_index(block: dict) -> dict:
    return {row["source_index"]: row["values"] for row in block["rows"]}


def test_population_then_causal_in_one_match_preserves_implied_tercile():
    """Population is documented first (in `_AnalogMatchEvidence.__init__`),
    causal second (in `set_causal`) -- both touch the same row indices. The
    causal block's `implied_tercile` must reflect CAUSAL edges, not whatever
    the population's own row cache happened to compute first.
    """
    from engine.analogs import _bucket

    trades = _trades_with_trending_implied_ratio(200, seed=1)
    matcher = AnalogMatcher(trades.copy(), snapshot="synthetic-snapshot")
    as_of = "2015-06-01"  # roughly the midpoint of the 200-day span
    buckets = matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=2.0)

    captured = []
    matcher.match("STR-THRU", buckets, alpha=0.5, as_of=as_of, min_analogs=1,
                   bootstrap=0, evidence_hook=captured.append)
    evidence = captured[0]

    causal_edges = _causal_tercile_edges(trades, as_of)
    population_values = _values_by_source_index(evidence["population"])
    causal_values = _values_by_source_index(evidence["causal"])

    saw_a_mismatch_opportunity = False
    for source_index, values in causal_values.items():
        ratio = values["implied_ratio"]
        expected = _bucket([ratio], causal_edges, ("low", "mid", "high"))[0]
        assert values["implied_tercile"] == expected, (
            f"row {source_index}: causal tercile {values['implied_tercile']!r} "
            f"!= expected {expected!r} from causal edges {causal_edges}"
        )
        population_tercile = population_values[source_index]["implied_tercile"]
        if population_tercile != expected:
            saw_a_mismatch_opportunity = True
    # If every row's population and causal tercile happened to agree, this
    # fixture would not distinguish the fix from the bug -- fail loudly
    # rather than pass for the wrong reason.
    assert saw_a_mismatch_opportunity, (
        "fixture did not exercise a population/causal tercile disagreement"
    )


def test_two_candidates_different_as_of_get_correct_own_tercile_labels():
    """Candidate B, scored on the SAME matcher right after candidate A
    (sharing a bucket, different `as_of`, overlapping causal rows), must
    report ITS OWN causal edges' tercile for every row -- not A's, and not
    the population's. Checked against the independently-computed ground
    truth (not against a second "fresh matcher" run: population is always
    documented before causal for EVERY matcher, fresh or not, so a fresh
    reference suffers the identical population-poisons-causal contamination
    and would agree with a buggy shared result for the wrong reason -- see
    test_documented_population_and_causal_content_is_independent_of_sharing
    for that byte-identical-to-a-fresh-matcher property instead).
    """
    from engine.analogs import _bucket

    trades = _trades_with_trending_implied_ratio(200, seed=2)
    buckets = AnalogMatcher(trades.copy()).buckets_for(
        mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=2.0)
    as_of_a, as_of_b = "2015-04-01", "2015-09-01"
    edges_a = _causal_tercile_edges(trades, as_of_a)
    edges_b = _causal_tercile_edges(trades, as_of_b)
    assert edges_a != edges_b, "fixture must give the two as_of dates different edges"

    shared_matcher = AnalogMatcher(trades.copy(), snapshot="synthetic-snapshot")

    def run(matcher, as_of):
        captured = []
        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=as_of, min_analogs=1,
                       bootstrap=0, evidence_hook=captured.append)
        return captured[0]

    captured_a = run(shared_matcher, as_of_a)  # warms the shared caches
    captured_b = run(shared_matcher, as_of_b)  # must not reuse A's rows/edges

    for label, captured, edges, other_edges in (
        ("A", captured_a, edges_a, edges_b), ("B", captured_b, edges_b, edges_a),
    ):
        causal_values = _values_by_source_index(captured["causal"])
        wrong = []
        for source_index, values in causal_values.items():
            ratio = values["implied_ratio"]
            expected = _bucket([ratio], edges, ("low", "mid", "high"))[0]
            if values["implied_tercile"] != expected:
                wrong.append((source_index, values["implied_tercile"], expected))
        assert not wrong, (
            f"candidate {label}: {len(wrong)} rows do not match candidate "
            f"{label}'s own causal edges {edges} — e.g. {wrong[0]}"
        )
