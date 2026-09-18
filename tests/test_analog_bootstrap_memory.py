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
