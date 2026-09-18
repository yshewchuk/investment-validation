"""Regression coverage for the strict-capture forward-pass RSS climb.

Root cause (measured on a real `capture_tier0_corpus.py --strict-phase4-trace`
run, 40 forward events, all strategies; see the commit this file ships with):
at the time of this fix, `AnalogMatcher` shared its DOCUMENTED causal block by
reference across every candidate that matches the same (strategy, alpha,
as_of) bucket (`_documented_causal`; that whole-block cache was itself removed
2026-09-18 as a further, separate duplication fix — see
`test_analog_bootstrap_memory.py` — `AnalogMatcher.match` now rebuilds
`population`/`causal` fresh per candidate from a persistent ROW-level cache
instead, unrelated to what this file tests). But
`Phase4TraceCollector.capture_analog_inputs` (the ONLY consumer of that
block's raw rows) did not reuse that sharing: EVERY scored candidate that
reaches `_score_analogs` re-walked the FULL causal population -- not just the
matched subset -- into its own fresh `source_rows`/`population_hash`, then
retained that fresh copy for the rest of the run via
`self._source_bundle["native_recipes"]["analogs"]`. A 40-event forward pass
therefore built and kept up to 40 (or more, across boundary/pinned/strike
passes) independent full-population projections even when they all shared one
causal identity -- unbounded in practice, and the measured driver of the
~0.5 GB/min RSS growth.

The fix threads an optional `recipe_cache` (in production,
`scorer.matcher.phase4_recipe_cache`, a new `AnalogMatcher` cache keyed and
evicted in lockstep with its existing `_causal_row_caches`/`_causal_pools`)
through `capture_analog_inputs`, so candidates sharing a (strategy, alpha,
as_of) key share one documented rows structure and one `population_hash` by
reference, `_document`ed exactly ONCE per key (not once per candidate) so its
NaN/Inf/numpy-scalar sanitizing pass still runs -- byte-identical output,
bounded retention. `recipe_cache=None` (the default) is untouched: no caller
outside `capture_tier0_corpus.py` opts in, and every existing test exercises
that default path unchanged. This cache is unaffected by the 2026-09-18
`_documented_causal` removal: it was already a separate, correctly-shared
cache, not one of the duplicates removed.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from engine.analogs import AnalogMatcher
from engine.score import Phase4TraceCollector, _Predocumented
from engine.v2.foundation import content_hash


def _row(row_id: str, *, mcap_bucket="large", moneyness_band="atm",
         dte_band="short", implied_tercile="mid", ret: float = 0.1) -> dict:
    return {
        "row_id": row_id,
        "source_index": row_id,
        "values": {
            "mcap_bucket": mcap_bucket,
            "moneyness_band": moneyness_band,
            "dte_band": dte_band,
            "implied_tercile": implied_tercile,
            "ret": ret,
        },
    }


def _evidence(rows: list[dict], *, strategy="STR-THRU", alpha=0.5,
              cutoff="2024-01-01T00:00:00", snapshot="snap",
              request_key="req") -> dict:
    return {
        "strategy": strategy,
        "alpha": alpha,
        "snapshot": snapshot,
        "cutoff": cutoff,
        "request_key": request_key,
        "bucket_query": {
            "mcap_bucket": "large", "moneyness_band": "atm",
            "dte_band": "short", "implied_tercile": "mid",
        },
        "causal": {"rows": rows},
    }


def _analogs_doc(collector: Phase4TraceCollector) -> Mapping[str, Any]:
    return collector._source_bundle["native_recipes"]["analogs"]


def _big_row_set(n: int, *, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        _row(f"row-{i}", ret=float(rng.normal(0, 0.3)))
        for i in range(n)
    ]


# -- content correctness: cache must not change WHAT is captured -----------


def test_no_cache_reproduces_content_of_two_independent_rebuilds():
    """`recipe_cache=None` (every caller today except capture_tier0_corpus.py)
    must behave exactly as before: two independent collectors scoring the
    SAME evidence get equal content, never a shared object.
    """
    rows = _big_row_set(50, seed=1)
    evidence = _evidence(rows)

    a = Phase4TraceCollector(content_hasher=content_hash)
    a.capture_analog_inputs(evidence)
    b = Phase4TraceCollector(content_hasher=content_hash)
    b.capture_analog_inputs(evidence)

    doc_a, doc_b = _analogs_doc(a), _analogs_doc(b)
    assert doc_a["recipe"]["population_hash"] == doc_b["recipe"]["population_hash"]
    assert isinstance(doc_a["source_rows"], _Predocumented)
    assert doc_a["source_rows"].value == doc_b["source_rows"].value
    assert doc_a["source_rows"].value is not doc_b["source_rows"].value


def test_cache_output_content_equals_no_cache_output_content():
    """The cache must be purely a retention/CPU optimization: with it wired
    up, the captured content for a fresh key is identical to the no-cache
    path's content.
    """
    rows = _big_row_set(40, seed=2)
    evidence = _evidence(rows)

    uncached = Phase4TraceCollector(content_hasher=content_hash)
    uncached.capture_analog_inputs(evidence)

    cache: dict = {}
    cached = Phase4TraceCollector(content_hasher=content_hash)
    cached.capture_analog_inputs(evidence, recipe_cache=cache)

    doc_u, doc_c = _analogs_doc(uncached), _analogs_doc(cached)
    assert doc_u["recipe"] == doc_c["recipe"]
    assert doc_u["source_rows"].value == doc_c["source_rows"].value
    assert doc_u["query_features"] == doc_c["query_features"]


def test_cache_sanitizes_non_finite_values_exactly_like_no_cache_path():
    """`_document` (via `json_safe`) turns a non-finite realized_return into
    `None`. The cache stores the `_document`-ed rows, computed once -- this
    proves that sanitizing pass still runs when the cache is populated, not
    skipped in exchange for the memory win.
    """
    rows = [_row("row-0", ret=float("inf")), _row("row-1", ret=-3.0)]
    evidence = _evidence(rows)

    uncached = Phase4TraceCollector(content_hasher=content_hash)
    uncached.capture_analog_inputs(evidence)
    cache: dict = {}
    cached = Phase4TraceCollector(content_hasher=content_hash)
    cached.capture_analog_inputs(evidence, recipe_cache=cache)

    uncached_rows = _analogs_doc(uncached)["source_rows"].value
    cached_rows = _analogs_doc(cached)["source_rows"].value
    assert uncached_rows == cached_rows
    inf_row = next(r for r in uncached_rows if r["row_id"] == "row-0")
    assert inf_row["realized_return"] is None


# -- retention: sharing across candidates on the same key -------------------


def test_second_candidate_sharing_key_reuses_documented_rows_by_reference():
    rows = _big_row_set(200, seed=3)
    cache: dict = {}

    first = Phase4TraceCollector(content_hasher=content_hash)
    first.capture_analog_inputs(_evidence(rows), recipe_cache=cache)
    second = Phase4TraceCollector(content_hasher=content_hash)
    second.capture_analog_inputs(_evidence(rows), recipe_cache=cache)

    doc_first, doc_second = _analogs_doc(first), _analogs_doc(second)
    assert doc_first["source_rows"].value is doc_second["source_rows"].value
    assert doc_first["recipe"]["population_hash"] == doc_second["recipe"]["population_hash"]
    assert len(cache) == 1


def test_different_cutoff_or_alpha_gets_its_own_cache_entry():
    rows = _big_row_set(60, seed=4)
    cache: dict = {}

    same_key_a = Phase4TraceCollector(content_hasher=content_hash)
    same_key_a.capture_analog_inputs(_evidence(rows, cutoff="2024-01-01T00:00:00"),
                                      recipe_cache=cache)
    diff_cutoff = Phase4TraceCollector(content_hasher=content_hash)
    diff_cutoff.capture_analog_inputs(_evidence(rows, cutoff="2024-02-01T00:00:00"),
                                       recipe_cache=cache)
    diff_alpha = Phase4TraceCollector(content_hasher=content_hash)
    diff_alpha.capture_analog_inputs(_evidence(rows, alpha=0.75),
                                      recipe_cache=cache)

    assert len(cache) == 3
    a_rows = _analogs_doc(same_key_a)["source_rows"].value
    c_rows = _analogs_doc(diff_cutoff)["source_rows"].value
    al_rows = _analogs_doc(diff_alpha)["source_rows"].value
    # Content happens to be equal here (same underlying rows) but the cache
    # must not have collapsed them into one shared object.
    assert a_rows == c_rows == al_rows
    assert _analogs_doc(same_key_a)["source_rows"].value is not (
        _analogs_doc(diff_cutoff)["source_rows"].value
    )


def test_empty_source_rows_cached_and_still_skips_native_recipes():
    """A causal block with no usable rows must be a no-op for BOTH the first
    and a subsequent cache-hit candidate -- matching what two independent
    (uncached) calls would each have done on their own.
    """
    malformed = [{"row_id": "x"}]  # no "values" -> filtered out entirely
    cache: dict = {}

    first = Phase4TraceCollector(content_hasher=content_hash)
    first.capture_analog_inputs(_evidence(malformed), recipe_cache=cache)
    second = Phase4TraceCollector(content_hasher=content_hash)
    second.capture_analog_inputs(_evidence(malformed), recipe_cache=cache)

    assert "analogs" not in first._source_bundle["native_recipes"]
    assert "analogs" not in second._source_bundle["native_recipes"]
    assert len(cache) == 1


def test_full_checkpoint_is_still_json_dumpable_and_content_hash_stable():
    """End-to-end through `capture_source_bundle`/`diagnostic_checkpoint` --
    the `_Predocumented` wrapper must not leak into the final structure.
    """
    rows = _big_row_set(30, seed=5)
    cache: dict = {}

    collectors = []
    for _ in range(3):
        collector = Phase4TraceCollector(content_hasher=content_hash)
        collector.capture_analog_inputs(_evidence(rows), recipe_cache=cache)
        collector.capture_source_bundle(features={"spot": 1.0})
        collectors.append(collector)

    checkpoints = [c.diagnostic_checkpoint() for c in collectors]
    encoded = [json.dumps(cp) for cp in checkpoints]
    assert encoded[0] == encoded[1] == encoded[2]
    for cp in checkpoints:
        value = cp["checkpoints"]["source_inputs"]["value"]
        assert content_hash(value) == cp["checkpoints"]["source_inputs"]["content_hash"]
        source_rows = value["native_recipes"]["analogs"]["source_rows"]
        assert isinstance(source_rows, list)
        assert all(isinstance(row, dict) for row in source_rows)


# -- integration: real AnalogMatcher, eviction lockstep, memory bound -------


def _trades(n: int, *, seed: int = 0, n_extra_cols: int = 10) -> pd.DataFrame:
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
            rng.integers(0, 3000, n), unit="D"),
        "exit_date": pd.Timestamp("2015-01-05") + pd.to_timedelta(
            rng.integers(0, 3000, n), unit="D"),
    })
    for c in range(n_extra_cols):
        frame[f"extra_{c}"] = rng.normal(0, 1, n)
    return frame


def test_matcher_phase4_recipe_cache_evicted_in_lockstep_with_causal_pools():
    """Visiting more distinct as_of dates than `MAX_CAUSAL_CACHE` must evict
    `phase4_recipe_cache` entries too -- otherwise it grows unbounded exactly
    like the bug this file exists to catch, just one layer further down.
    """
    trades = _trades(500, seed=6)
    matcher = AnalogMatcher(trades.copy(), snapshot="synthetic")
    matcher.MAX_CAUSAL_CACHE = 5
    buckets = matcher.buckets_for(mcap_usd=5e9, dte=5, moneyness_pct=0.0,
                                   implied_ratio=1.0)

    for month in range(1, 13):
        as_of = f"2024-{month:02d}-01"
        captured = []
        matcher.match("STR-THRU", buckets, alpha=0.5, as_of=as_of,
                       min_analogs=1, bootstrap=0, request_key=f"req-{month}",
                       evidence_hook=captured.append)
        collector = Phase4TraceCollector(content_hasher=content_hash)
        collector.capture_analog_inputs(
            captured[0], recipe_cache=matcher.phase4_recipe_cache,
        )

    assert len(matcher._causal_pools) <= 5
    assert len(matcher.phase4_recipe_cache) <= 5


def test_forward_pass_shaped_retention_stays_bounded_across_many_candidates():
    """Mirrors `test_analog_bootstrap_memory
    .test_retention_across_20_same_bucket_candidates_stays_bounded`, one
    layer further down the stack: 40 candidates sharing ONE persistent
    matcher AND recipe cache (how `Scorer.matcher` and
    `scorer.matcher.phase4_recipe_cache` are actually used across a capture
    run) must retain exactly ``n`` documented row objects TOTAL, not
    ``40 * n`` — the fix's whole point. Counted by object identity rather
    than `tracemalloc` bytes: this fixture's trades all land in one bucket
    (no widening), so `_AnalogMatchEvidence`'s own (already-shared, already
    tested) `selected`/`contributing` blocks are large enough on both sides
    of this comparison to swamp a byte-level peak measurement — the object
    count isolates exactly what this fix changes.
    """
    n = 1200
    trades = _trades(n, seed=7, n_extra_cols=6)
    buckets = AnalogMatcher(trades.copy()).buckets_for(
        mcap_usd=5e9, dte=5, moneyness_pct=0.0, implied_ratio=1.0)

    def run(matcher: AnalogMatcher, cache: dict | None):
        kept = []
        for i in range(40):
            captured = []
            matcher.match(
                "STR-THRU", buckets, alpha=0.5, as_of="2024-06-01",
                min_analogs=1, bootstrap=0, request_key=f"req-{i}",
                evidence_hook=captured.append,
            )
            collector = Phase4TraceCollector(content_hasher=content_hash)
            collector.capture_analog_inputs(captured[0], recipe_cache=cache)
            kept.append(collector)
        return kept

    def distinct_row_object_count(kept: list[Phase4TraceCollector]) -> int:
        return len({
            id(row) for c in kept for row in _analogs_doc(c)["source_rows"].value
        })

    no_cache_kept = run(AnalogMatcher(trades.copy()), None)
    cached_matcher = AnalogMatcher(trades.copy())
    cached_kept = run(cached_matcher, cached_matcher.phase4_recipe_cache)

    row_count = len(_analogs_doc(cached_kept[0])["source_rows"].value)
    assert row_count > 0
    assert len(cached_matcher.phase4_recipe_cache) == 1

    # No cache: every candidate independently rebuilt its own N row dicts.
    assert distinct_row_object_count(no_cache_kept) == 40 * row_count
    # Cache: every candidate shares the SAME N row dicts by reference.
    assert distinct_row_object_count(cached_kept) == row_count

    first_rows = _analogs_doc(cached_kept[0])["source_rows"].value
    assert all(
        _analogs_doc(c)["source_rows"].value is first_rows for c in cached_kept
    )
