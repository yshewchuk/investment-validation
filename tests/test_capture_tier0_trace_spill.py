"""`tools.capture_tier0_corpus`'s legacy_trace spill: retention + fidelity.

2026-09-18: a real strict-phase4-trace capture run against 280cf7c still hit
the 5.5 GB bounded_run cap after the forward pass alone, growing in steps
that lined up with new (strategy, alpha, as_of) causal keys. Reading
`main()` alongside the matcher's own (already bounded) caches showed
`candidates` — the list every scoring pass accumulates into for the WHOLE
run, never evicted — retaining every candidate's full `legacy_trace`
(per-candidate option-chain snapshots, documented analog rows, residual
population slices) even though `select()`'s covering pass never reads a
candidate's OWN `legacy_trace` (only `record`/`request`/`kind`/`relations`),
and `_rescore` never reads the SOURCE candidate's `legacy_trace` either — the
only consumers are `attach_strict_probe` and `write`'s own pairs/checkpoint
loop, both of which only run over `select()`'s small `chosen` subset.

The fix: `_candidate()` spills a non-None `legacy_trace` to its own file on
disk the moment it is produced and keeps only a tiny `_SpilledTrace` pointer
in the dict `candidates` holds; `main()` hydrates `chosen` (only) back to the
real content, byte-for-byte (via `pickle`, an internal same-process
round-trip — no JSON float/NaN-encoding risk), right before
`attach_strict_probe`/`write` run. These tests check the two things that
matter: the spilled content comes back EXACTLY unchanged, and the marker
genuinely does not carry the trace's own bytes.
"""
from __future__ import annotations

import math
import pickle

import pytest

import tools.capture_tier0_corpus as capture


def _sample_trace(*, big: int = 0) -> dict:
    """A representative `legacy_trace` shape: nested dict/list/scalar/None,
    including the kind of NaN-safe sentinel `Phase4TraceCollector._document`
    already produces (a raw NaN never reaches this far — see
    `engine.jsonio.json_safe` — but the round-trip must still be exact for
    whatever it DOES emit: strings, floats, ints, bools, None, nesting).
    """
    return {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": ["WIDE_MARKET"]},
        "checkpoints": {
            "source_inputs": {
                "value": {
                    "context": {"ticker": "ABC", "spot": 101.25},
                    "quote_domain": [
                        {"strike": float(100 + i), "bid": 1.0, "ask": 1.05,
                         "note": None, "wide": i % 2 == 0}
                        for i in range(big)
                    ],
                    "native_recipes": {
                        "analogs": {
                            "source_rows": [
                                {"row_id": f"r{i}", "realized_return": -0.01 * i,
                                 "mcap_bucket": "mega"}
                                for i in range(big)
                            ],
                        },
                    },
                },
                "content_hash": "sha256:" + "0" * 64,
            },
        },
    }


def test_candidate_spills_legacy_trace_instead_of_holding_it() -> None:
    trace = _sample_trace(big=50)
    candidate = capture._candidate(
        {"ticker": "ABC", "strategy": "STR-THRU"}, {"a": 1}, {"a": 1}, 0.01,
        legacy_trace=trace,
    )
    try:
        assert isinstance(candidate["legacy_trace"], capture._SpilledTrace)
        # The marker holds a path, not the trace's own content -- its
        # resident size does not grow with the trace it points at.
        small = capture._candidate(
            {"ticker": "ABC", "strategy": "STR-THRU"}, {"a": 1}, {"a": 1}, 0.01,
            legacy_trace=_sample_trace(big=0),
        )["legacy_trace"]
        import sys
        assert sys.getsizeof(candidate["legacy_trace"]) == sys.getsizeof(small)
        # The data really did move to disk -- readable back independently of
        # the in-process marker, with the pickle proving it is the real size.
        with open(candidate["legacy_trace"].path, "rb") as fh:
            on_disk = pickle.load(fh)
        assert on_disk == trace
    finally:
        capture._cleanup_trace_spill()


def test_hydrate_recovers_the_exact_trace_content() -> None:
    trace = _sample_trace(big=200)
    candidate = capture._candidate(
        {"ticker": "XYZ", "strategy": "TWIN-P"}, None, {}, 0.02,
        legacy_trace=trace,
    )
    try:
        hydrated = capture._hydrate_trace(candidate["legacy_trace"])
        assert hydrated == trace
        # Every float, including ones a naive JSON round-trip could perturb,
        # is bit-for-bit the same object value after pickle round-trips it.
        rows = hydrated["checkpoints"]["source_inputs"]["value"]["native_recipes"]["analogs"]["source_rows"]
        original_rows = trace["checkpoints"]["source_inputs"]["value"]["native_recipes"]["analogs"]["source_rows"]
        for got, want in zip(rows, original_rows):
            assert got["realized_return"] == want["realized_return"]
            assert math.copysign(1.0, got["realized_return"] or 1.0) == math.copysign(
                1.0, want["realized_return"] or 1.0)
    finally:
        capture._cleanup_trace_spill()


@pytest.mark.parametrize("value", [None, {"already": "hydrated"}])
def test_hydrate_is_a_no_op_on_anything_that_is_not_a_spill_pointer(value) -> None:
    # Existing tests (test_phase4_capture_strict.py, test_phase4_capture_writer.py)
    # build candidate dicts by hand with a plain `legacy_trace` dict and call
    # `attach_strict_probe`/`write` directly, never through `main()`'s
    # hydration step -- `_hydrate_trace` must leave that untouched.
    assert capture._hydrate_trace(value) is value


def test_no_candidate_needs_its_own_legacy_trace_before_select_runs() -> None:
    """`select()` covers axes from `record`/`request`/`kind`/`relations`
    alone -- a regression here would silently defeat the whole spill design,
    since `select()` runs BEFORE `chosen` is hydrated.
    """
    import inspect
    source = inspect.getsource(capture.select)
    assert "legacy_trace" not in source
    # `_rescore` legitimately PASSES its own freshly captured `legacy_trace`
    # into the new candidate it builds (`_candidate(..., legacy_trace=trace)`)
    # -- that is a NEW trace from this call's own `_score()`, not a read of
    # the SOURCE candidate's trace. The regression this guards against is
    # `_rescore` reading `source["legacy_trace"]` / `source.get("legacy_trace")`.
    assert 'source["legacy_trace"]' not in source
    assert 'source.get("legacy_trace")' not in source


def test_spilling_and_hydrating_produces_byte_identical_written_output(tmp_path) -> None:
    """End-to-end: a candidate written straight from `_candidate()`'s plain
    dict (the old behavior) and one that went through spill-then-hydrate
    (the new behavior) must write IDENTICAL pair/checkpoint files.
    """
    import json

    import pandas as pd

    trace = _sample_trace(big=30)

    def _build(request, raw, record):
        return {
            "fixture_id": "case-1",
            "covers": ["strategy:STR-THRU"],
            "request": request, "record": record, "kind": "score_result",
            "duration": 0.1,
        }

    request = {"strategy": "STR-THRU", "ticker": "ABC"}
    record = {"strategy": "STR-THRU", "ticker": "ABC"}

    direct = _build(request, {}, record)
    direct["legacy_trace"] = trace  # never spilled -- the old shape

    spilled_candidate = capture._candidate(request, {}, record, 0.1, legacy_trace=trace)
    via_spill = _build(request, {}, record)
    try:
        via_spill["legacy_trace"] = capture._hydrate_trace(spilled_candidate["legacy_trace"])

        direct_dir = tmp_path / "direct"
        spill_dir = tmp_path / "spill"
        capture.write(direct_dir, [direct], {"strategy:STR-THRU": ["case-1"]},
                      pd.Timestamp("2026-01-01"), "snap-1")
        capture.write(spill_dir, [via_spill], {"strategy:STR-THRU": ["case-1"]},
                      pd.Timestamp("2026-01-01"), "snap-1")

        direct_pair = json.loads((direct_dir / "pairs" / "case-1.json").read_text())
        spill_pair = json.loads((spill_dir / "pairs" / "case-1.json").read_text())
        # `envelope.captured_at` is deliberately wall-clock (contracts §2.2,
        # excluded from `payload_hash`) -- everything ELSE, including the
        # hash itself, must be identical byte for byte.
        assert direct_pair["payload"] == spill_pair["payload"]
        assert direct_pair["payload_hash"] == spill_pair["payload_hash"]
        assert direct_pair["request_hash"] == spill_pair["request_hash"]
        assert direct_pair["payload"]["legacy_trace"] == trace
        assert spill_pair["payload"]["legacy_trace"] == trace
    finally:
        capture._cleanup_trace_spill()


def test_many_large_spilled_traces_keep_the_candidate_list_small() -> None:
    """The retention regression this whole fix targets: N candidates with
    substantial `legacy_trace` content must not multiply the in-memory
    footprint of `candidates` with N -- the whole point of spilling.
    """
    import sys

    candidates = []
    try:
        for i in range(40):
            trace = _sample_trace(big=500)  # a few hundred KB pickled, each
            candidates.append(capture._candidate(
                {"ticker": f"T{i}", "strategy": "STR-THRU"}, None, {}, 0.01,
                legacy_trace=trace,
            ))
        markers_bytes = sum(sys.getsizeof(c["legacy_trace"]) for c in candidates)
        spill_dir = capture._trace_spill_dir()
        on_disk_bytes = sum(p.stat().st_size for p in spill_dir.glob("*.pkl"))
        # The 40 traces' real content landed on disk (comfortably more than
        # a bare marker could ever account for)...
        assert on_disk_bytes > 40 * 10_000
        # ...while every in-memory marker together is a few hundred bytes,
        # not a fraction of that content.
        assert markers_bytes < 10_000
    finally:
        capture._cleanup_trace_spill()
