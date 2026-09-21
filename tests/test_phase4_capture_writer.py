from __future__ import annotations

import json

import pandas as pd

from checks.phase4_checkpoints import (
    CheckpointError,
    _case_from_pointer,
    validate_bundle,
)
from engine.v2.foundation import content_hash
from tools.capture_tier0_corpus import write


def _verify_case(release, case_id):
    """Validate one on-disk case against the REAL per-case validator,
    ``checks/phase4_checkpoints.py``'s own ``_case_from_pointer`` (which
    hash-verifies the file, then runs the full ``_case`` schema check --
    the same function ``validate_bundle`` calls per case). Bundle-level
    coverage (every strategy/branch axis represented across the WHOLE
    corpus) is a separate, corpus-wide property of ``select()`` and is not
    what this fix -- one candidate's case shape -- is responsible for."""
    release_root = release / "checkpoints"
    manifest = json.loads((release_root / "manifest.json").read_text())
    pointer = next(row for row in manifest["cases"] if row["case_id"] == case_id)
    return _case_from_pointer(release_root, pointer, 0, set())


def _hashed(value):
    return {"value": value, "content_hash": content_hash(value)}


def _full_legacy_trace(*, residual_population=None):
    """A REAL-shaped ``Phase4TraceCollector.diagnostic_checkpoint()`` --
    all four required groups, each hashed the way ``capture_*`` records
    them, not a partial trace. ``checks/phase4_checkpoints.py`` requires
    all four on every case, so a partial trace (the old fixture here) can
    never become a valid case -- see ``_phase4_case_document``."""
    simulation_value = {
        "horizon": {"exit_date": "2026-01-10", "expiry": "2026-01-16", "dte_exit": 6.0},
        "capital_denominator": 2.5,
        "residual_population_identity": {"cutoff": "2026-01-01", "fallback_used": False},
        "draw_count": 1000,
        "seed": 7,
    }
    if residual_population is not None:
        # Only ``Phase4TraceCollector.capture_simulation``'s OWN stored shape
        # carries this extra field; the phase4 case contract does not (see
        # ``_SIMULATION_DROPPED_FIELDS``).
        simulation_value = {**simulation_value, "residual_population": residual_population}
    return {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": []},
        "checkpoints": {
            "features": _hashed({
                "feature_vector": {"driver": {"spot": 100.0}},
                "missing_mask": {"driver": {"spot": False}},
                "model_identity": {"driver": {"model_id": "m1"}},
            }),
            "selection_pricing": _hashed({
                "selected_legs": [{"right": "call", "side": "long", "quantity": 1,
                                   "expiry": "2026-01-16"}],
                "entry_cost": 2.5,
            }),
            "simulation": _hashed(simulation_value),
            "gate_inputs": _hashed({"kind": "entry_rule", "facts": {"iv30": 55.0}}),
        },
    }


def _candidate(fixture_id="case-1", *, residual_population=None):
    return {
        "fixture_id": fixture_id,
        "covers": ["strategy:STR-THRU"],
        "request": {"strategy": "STR-THRU", "ticker": "ABC"},
        "record": {"strategy": "STR-THRU", "ticker": "ABC"},
        "kind": "score_result",
        "duration": 0.1,
        "legacy_trace": _full_legacy_trace(residual_population=residual_population),
    }


def test_writer_persists_a_case_the_real_phase4_validator_accepts(tmp_path) -> None:
    """Producer and validator wired together (not two independent
    assertions of the same literal): the case ``write()`` actually put on
    disk is loaded and checked by ``checks/phase4_checkpoints.py``'s own
    ``load_bundle``/``validate_bundle`` -- the real spec, unmodified.

    Before the shape fix, ``write()`` emitted ``{case_id, request,
    strategy, covers, record_kind, checkpoint}``: none of ``request_hash``,
    ``branches``, ``resource_refs``, ``executable_inputs``, ``disposition``,
    ``checkpoints`` (plural) or ``case_hash``. ``validate_bundle`` rejects
    that shape outright (``unexpected or missing fields``) -- see
    ``test_the_old_pre_fix_case_shape_is_rejected_by_the_real_validator``
    below, which reproduces it verbatim and proves the rejection.
    """
    candidate = _candidate()
    release = tmp_path / "release"
    write(
        release, [candidate], {"strategy:STR-THRU": ["case-1"]},
        pd.Timestamp("2026-01-01"), "snapshot-1",
    )

    manifest_path = release / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    case = json.loads((release / "checkpoints" / "cases" / "case-1.json").read_text())

    assert manifest["cases"] == [{
        "case_id": "case-1", "sha256": manifest["cases"][0]["sha256"],
    }]
    assert set(case) == {
        "case_id", "request", "request_hash", "strategy", "branches",
        "resource_refs", "executable_inputs", "disposition", "checkpoints",
        "case_hash",
    }
    assert case["strategy"] == "STR-THRU"
    assert case["branches"]  # nonempty, real: entry_cost=2.5 -> debit_credit
    assert set(case["checkpoints"]) == {
        "features", "selection_pricing", "simulation", "gate_inputs",
    }
    assert json.loads((release / "INDEX.json").read_text())[
        "diagnostic_checkpoint_manifest"
    ] == "checkpoints/manifest.json"

    # The real per-case validator, wired to the real producer output on disk.
    case_id, strategy, branches = _verify_case(release, "case-1")
    assert case_id == "case-1"
    assert strategy == "STR-THRU"
    assert "debit_credit" in branches


def test_the_old_pre_fix_case_shape_is_rejected_by_the_real_validator(tmp_path) -> None:
    """Reproduces, verbatim, the shape ``write_case()`` emitted before this
    fix (``tools/capture_tier0_corpus.py`` write(): ``{case_id, request,
    strategy, covers, record_kind, checkpoint}``) and proves the real
    ``checks/phase4_checkpoints.py`` validator refuses it. This is the
    defect this fix closes -- the old producer's own output, checked
    against the real spec, not a duplicated literal of what the spec says.
    """
    old_shape_case = {
        "case_id": "case-1",
        "request": {"strategy": "STR-THRU", "ticker": "ABC"},
        "strategy": "STR-THRU",
        "covers": ["strategy:STR-THRU"],
        "record_kind": "score_result",
        "checkpoint": _full_legacy_trace(),
    }
    import hashlib

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    data = json.dumps(old_shape_case, sort_keys=True).encode()
    (cases_dir / "case-1.json").write_bytes(data)
    pointer = {"case_id": "case-1", "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
    bundle = {
        "schema_version": "phase4_diagnostic_checkpoints.v1.0",
        "release_id": "r", "metadata": {"status": "diagnostic_only"},
        "resources": [], "coverage": {"strategies": {}, "branches": {}},
        "cases": [pointer],
    }
    bundle["manifest_hash"] = content_hash(
        {k: v for k, v in bundle.items() if k != "manifest_hash"})
    try:
        validate_bundle(bundle, tmp_path)
    except CheckpointError as exc:
        assert "unexpected or missing fields" in str(exc)
    else:
        raise AssertionError("the pre-fix case shape must be rejected")


def test_writer_keeps_infinite_residuals_in_the_canonical_form(tmp_path) -> None:
    """Legacy ResidualPool keeps a +/-inf residual (R4-20 gap 1), so the
    CORPUS PAIR's captured residual_population must carry it, unchanged --
    written as the repo's canonical {"__nonfinite__": ...} tag
    (engine.v2.foundation.canonical), under the content hash already
    recorded, never as null or bare Infinity.

    The phase4 CHECKPOINT CASE is different on purpose: its ``simulation``
    checkpoint is not allowed to carry ``residual_population`` at all (see
    ``_SIMULATION_DROPPED_FIELDS`` -- ``checks/phase4_checkpoints.py``'s own
    schema has no such field; only ``residual_population_identity``, whose
    hashes already stand in for the population). This checks that drop is a
    real, hashed-after-mutation shape correction: the case's stored
    ``content_hash`` matches the STRIPPED value, and the untouched full
    value is still recoverable from the pair file's own copy of the trace.
    """
    from engine.v2.foundation import content_hash
    from engine.v2.foundation.canonical import untag_nonfinite

    population = [
        {"event_date": "2025-01-02", "pred_abs_move": 4.0,
         "err_move": float("inf"), "err_crush": -1.0},
        {"event_date": "2025-01-03", "pred_abs_move": 5.0,
         "err_move": 0.5, "err_crush": float("-inf")},
    ]
    candidate = _candidate("case-inf", residual_population=population)
    release = tmp_path / "release"
    write(release, [candidate], {"strategy:STR-THRU": ["case-inf"]},
          pd.Timestamp("2026-01-01"), "snapshot-1")

    def reject(constant):
        raise AssertionError(f"bare {constant} written")

    case = json.loads((release / "checkpoints" / "cases" / "case-inf.json").read_text(),
                      parse_constant=reject)
    pair = json.loads((release / "pairs" / "case-inf.json").read_text(),
                      parse_constant=reject)

    # Pair side: the full raw trace, residual_population and all, untouched.
    pair_row = pair["payload"]["legacy_trace"]["checkpoints"]["simulation"]
    written = pair_row["value"]["residual_population"]
    assert written[0]["err_move"] == {"__nonfinite__": "inf"}
    assert written[1]["err_crush"] == {"__nonfinite__": "-inf"}
    decoded = untag_nonfinite(written)
    assert decoded[0]["err_move"] == float("inf")
    assert decoded[1]["err_crush"] == float("-inf")
    assert decoded[0]["err_crush"] == -1.0

    # Case side: no residual_population at all, and the stored hash is over
    # the value actually written -- not a stale hash of the superset.
    case_row = case["checkpoints"]["simulation"]
    assert "residual_population" not in case_row["value"]
    assert case_row["value"]["draw_count"] == 1000
    assert case_row["content_hash"] == content_hash(case_row["value"])
    case_id, _, _ = _verify_case(release, "case-inf")
    assert case_id == "case-inf"


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
