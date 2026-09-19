from __future__ import annotations

import json

import pandas as pd

from tools.capture_tier0_corpus import write


def test_writer_persists_bounded_checkpoint_manifest(tmp_path) -> None:
    checkpoint = {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": []},
        "checkpoints": {
            "features": {
                "value": {
                    "feature_vector": {"spot": 100.0},
                    "missing_mask": {"spot": False},
                    "model_identity": {"model_id": "m1"},
                },
                "content_hash": "sha256:" + "0" * 64,
            },
        },
    }
    candidate = {
        "fixture_id": "case-1",
        "covers": ["strategy:STR-THRU"],
        "request": {"strategy": "STR-THRU", "ticker": "ABC"},
        "record": {"strategy": "STR-THRU", "ticker": "ABC"},
        "kind": "score_result",
        "duration": 0.1,
        "legacy_trace": checkpoint,
    }

    release = tmp_path / "release"
    write(
        release,
        [candidate],
        {"strategy:STR-THRU": ["case-1"]},
        pd.Timestamp("2026-01-01"),
        "snapshot-1",
    )

    manifest = json.loads(
        (release / "checkpoints" / "manifest.json").read_text()
    )
    case = json.loads(
        (release / "checkpoints" / "cases" / "case-1.json").read_text()
    )

    assert manifest["cases"] == [{
        "case_id": "case-1",
        "sha256": manifest["cases"][0]["sha256"],
    }]
    assert case["checkpoint"] == checkpoint
    assert "record" not in case
    assert json.loads((release / "INDEX.json").read_text())[
        "diagnostic_checkpoint_manifest"
    ] == "checkpoints/manifest.json"


def test_writer_keeps_infinite_residuals_in_the_canonical_form(tmp_path) -> None:
    """Legacy ResidualPool keeps a +/-inf residual (R4-20 gap 1), so the
    captured residual_population must carry it. It is written as the repo's
    canonical {"__nonfinite__": ...} tag (engine.v2.foundation.canonical),
    under the content hash already recorded, never as null or bare Infinity."""
    from engine.v2.foundation import content_hash
    from engine.v2.foundation.canonical import untag_nonfinite

    population = [
        {"event_date": "2025-01-02", "pred_abs_move": 4.0,
         "err_move": float("inf"), "err_crush": -1.0},
        {"event_date": "2025-01-03", "pred_abs_move": 5.0,
         "err_move": 0.5, "err_crush": float("-inf")},
    ]
    value = {"residual_population": population, "draw_count": 4}
    checkpoint = {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": []},
        "checkpoints": {"simulation": {"value": value,
                                       "content_hash": content_hash(value)}},
    }
    candidate = {
        "fixture_id": "case-inf", "covers": ["strategy:STR-THRU"],
        "request": {"strategy": "STR-THRU", "ticker": "ABC"},
        "record": {"strategy": "STR-THRU", "ticker": "ABC"},
        "kind": "score_result", "duration": 0.1, "legacy_trace": checkpoint,
    }
    release = tmp_path / "release"
    write(release, [candidate], {"strategy:STR-THRU": ["case-inf"]},
          pd.Timestamp("2026-01-01"), "snapshot-1")

    def reject(constant):
        raise AssertionError(f"bare {constant} written")

    case = json.loads((release / "checkpoints" / "cases" / "case-inf.json").read_text(),
                      parse_constant=reject)
    pair = json.loads((release / "pairs" / "case-inf.json").read_text(),
                      parse_constant=reject)
    for trace in (case["checkpoint"], pair["payload"]["legacy_trace"]):
        row = trace["checkpoints"]["simulation"]
        written = row["value"]["residual_population"]
        assert written[0]["err_move"] == {"__nonfinite__": "inf"}
        assert written[1]["err_crush"] == {"__nonfinite__": "-inf"}
        assert row["content_hash"] == content_hash(row["value"]) == content_hash(value)
        decoded = untag_nonfinite(written)
        assert decoded[0]["err_move"] == float("inf")
        assert decoded[1]["err_crush"] == float("-inf")
        assert decoded[0]["err_crush"] == -1.0


# -- the frozen pools/states are hashed and spilled once, not per candidate ----------


def _shared_trace(pool, state):
    """One candidate's collector with a fold pool and a model state recorded
    through ``capture_frozen``, as ``engine.score`` records them."""
    import tools.capture_tier0_corpus as capture
    from engine.score import Phase4TraceCollector, _Predocumented

    trace = Phase4TraceCollector(retain_full_trace=False,
                                 content_hasher=capture._SHARED_TRACE_DOCUMENTS)
    trace.capture_source_bundle(context={"ticker": "AAA"},
                                quote_domain=[{"strike": 100.0, "bid": 1.0}])
    trace.capture_frozen(
        fold_pools={"pred_abs_move": pool}, states={"payoff:x": state},
        declarations={"payoff": {"state": "payoff:x", "seed": 3}},
        inputs={"fold:size@sizing": {"iv30": 55.0}})
    assert isinstance(pool, _Predocumented)
    return trace


def _pool_and_state():
    from engine.score import _Predocumented

    pool = _Predocumented({"predictions": [0.5 * i for i in range(200)],
                           "residuals": [float("inf"), -0.0, 1e-7, 1e21, *range(196)],
                           "interval_floor": None})
    state = _Predocumented({"kind": "line", "residuals": [0.25, -1.5], "n": 2})
    return pool, state


def test_shared_fragments_hash_exactly_like_the_plain_canonical_form() -> None:
    from engine.v2.foundation.canonical import content_hash

    import tools.capture_tier0_corpus as capture

    pool, state = _pool_and_state()
    registry = capture._SharedTraceDocuments(text_budget=1)
    registry.register_shared([pool.value, state.value])
    document = {"a": pool.value, 2: [state.value, (1, 2.5)], "b": {"c": pool.value},
                "nan": float("nan"), "s": "é"}
    for _ in range(2):  # rendered, then served from the memo (or evicted)
        assert registry(document) == content_hash(document)
    assert capture._SharedTraceDocuments()(document) == content_hash(document)


def test_candidates_hash_and_spill_the_shared_pool_once_and_hydrate_it_shared() -> None:
    from engine.v2.foundation.canonical import content_hash

    import tools.capture_tier0_corpus as capture

    pool, state = _pool_and_state()
    try:
        traces = [_shared_trace(pool, state) for _ in range(3)]
        assert traces[0].shared_documents() == (pool.value, state.value)
        checkpoints = [trace.diagnostic_checkpoint() for trace in traces]
        for checkpoint in checkpoints:
            row = checkpoint["checkpoints"]["source_inputs"]
            assert row["content_hash"] == content_hash(row["value"])
            assert row["value"]["frozen"]["fold_pools"]["pred_abs_move"] is pool.value
        spilled = [capture._spill_trace(checkpoint) for checkpoint in checkpoints]
        # The spill holds a reference, not ~the pool: every file is small.
        assert max(pointer.path.stat().st_size for pointer in spilled) < 2000
        hydrated = [capture._hydrate_trace(pointer) for pointer in spilled]
        assert hydrated == checkpoints
        for trace in hydrated:
            frozen = trace["checkpoints"]["source_inputs"]["value"]["frozen"]
            assert frozen["fold_pools"]["pred_abs_move"] is pool.value
            assert frozen["states"]["payoff:x"] is state.value
    finally:
        capture._cleanup_trace_spill()
    # After the run's registry is reset, a spill of a shared value cannot be
    # hydrated into something else silently.
    assert not capture._SHARED_TRACE_DOCUMENTS.shared(pool.value)


def test_frozen_recording_redocuments_only_its_section() -> None:
    from engine.score import Phase4TraceCollector

    import tools.capture_tier0_corpus as capture

    pool, state = _pool_and_state()
    try:
        trace = _shared_trace(pool, state)
        trace.capture_frozen(declarations={"gate": {"binding": "gate", "output": "g"}})
        group = trace._checkpoint_groups["source_inputs"]
        assert group == Phase4TraceCollector._document(trace._source_bundle)
    finally:
        capture._cleanup_trace_spill()


def test_shared_texts_do_not_thrash_when_a_candidate_outgrows_the_budget() -> None:
    """A plain LRU smaller than one candidate's shared working set misses on
    every access of the cyclic candidate pattern (re-rendering everything per
    candidate); texts used by the current or previous hash are kept."""
    from engine.v2.foundation.canonical import content_hash

    import tools.capture_tier0_corpus as capture

    pools = [[float(i + j) for i in range(50)] for j in range(4)]
    registry = capture._SharedTraceDocuments(text_budget=100)
    registry.register_shared(pools)
    for candidate in range(6):
        document = {"frozen": {"pools": pools, "candidate": candidate}}
        assert registry(document) == content_hash(document)
    assert registry.renders == len(pools)
    # Texts nobody has used for two hashes are dropped back under budget.
    for _ in range(3):
        registry({"frozen": {"pools": pools[:1]}})
    assert registry.renders == len(pools)
    assert registry._text_size <= len(registry._texts[id(pools[0])][0])
