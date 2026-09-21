from __future__ import annotations

import copy
import json

import pandas as pd

from checks.phase4_checkpoints import (
    CheckpointError,
    _case_from_pointer,
    validate_bundle,
)
from engine.v2.foundation import content_hash
import tools.capture_tier0_corpus as capture
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
    them, not a partial trace. ``_phase4_case_document`` now WRITES a
    partial trace too (as an honest ``incomparable``/``refused_as_expected``
    case with only the groups actually reached -- see its docstring), but
    ``checks/phase4_checkpoints.py`` still requires all four unconditionally
    today, so only a FULL trace like this one can pass its validator; that
    contract gap is flagged, not fixed, in this producer."""
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
        "first_gap", "case_hash",
    }
    assert case["strategy"] == "STR-THRU"
    assert case["branches"]  # nonempty, real: entry_cost=2.5 -> debit_credit
    assert set(case["checkpoints"]) == {
        "features", "selection_pricing", "simulation", "gate_inputs",
    }
    # All four required groups present -> nothing to explain.
    assert case["first_gap"] is None
    assert json.loads((release / "INDEX.json").read_text())[
        "diagnostic_checkpoint_manifest"
    ] == "checkpoints/manifest.json"

    # The real per-case validator, wired to the real producer output on disk.
    case_id, strategy, branches = _verify_case(release, "case-1")
    assert case_id == "case-1"
    assert strategy == "STR-THRU"
    assert "debit_credit" in branches


def test_an_early_refusal_is_written_honestly_not_skipped(tmp_path) -> None:
    """2026-09-21 (coordinator): a candidate whose real trace refused
    BEFORE capturing any checkpoint group must still be WRITTEN, with an
    honest ``disposition``/``executable_inputs`` -- never silently
    skipped (that would quietly shrink the corpus a completeness gate
    reads) and never fabricated (no empty ``features``/``selection_pricing``/
    ``simulation``/``gate_inputs`` groups invented to satisfy the schema).
    The legacy_trace here is exactly what a real SUPERSEDED refusal
    produces (``Phase4TraceCollector.finish`` + the new ``first_gap``,
    see test_phase4_trace_collector.py's own wired test of that path).
    """
    candidate = {
        "fixture_id": "case-superseded",
        "covers": ["strategy:TWIN-P"],
        "request": {"strategy": "TWIN-P", "ticker": "ABC"},
        "record": {"strategy": "TWIN-P", "ticker": "ABC"},
        "kind": "score_result",
        "duration": 0.05,
        "legacy_trace": {
            "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
            "disposition": {
                "status": "refused", "flags": ["SUPERSEDED"],
                "detail": "TWIN-P is superseded by TWIN-P5",
                "first_gap": {"stage": "resolve_context",
                              "reason": "superseded strategy"},
            },
            "checkpoints": {},
        },
    }
    release = tmp_path / "release"
    write(release, [candidate], {"strategy:TWIN-P": ["case-superseded"]},
          pd.Timestamp("2026-01-01"), "snapshot-1")

    case = json.loads(
        (release / "checkpoints" / "cases" / "case-superseded.json").read_text())
    assert case["checkpoints"] == {}
    assert case["executable_inputs"] == "missing"
    assert case["disposition"] == "incomparable"
    assert "missing_inputs" in case["branches"]
    # No checkpoint group at all (not even `source_inputs`) means this row
    # truly could not be evaluated -- `executable_inputs: missing` forces
    # `incomparable` (never `refused_as_expected`, which is reserved for a
    # row that DID resolve real inputs before its deliberate early exit).
    assert case["first_gap"] == {
        "stage": "resolve_context", "reason": "superseded strategy",
    }
    assert case["case_hash"] == content_hash(
        {k: v for k, v in case.items() if k != "case_hash"})

    # 2026-09-21 contract change: a non-`compared` case with a strict
    # subset of the four required groups now validates against the REAL
    # spec, PROVIDED it carries `first_gap` -- which this one does.
    case_id, strategy, branches = _verify_case(release, "case-superseded")
    assert case_id == "case-superseded"
    assert strategy == "TWIN-P"
    assert "missing_inputs" in branches


def test_a_row_that_resolved_its_inputs_before_refusing_is_refused_as_expected(tmp_path) -> None:
    """2026-09-21 (coordinator, second fix): the real corpus proves
    ``refused_as_expected`` was unreachable -- every ``SUPERSEDED``/
    ``UNVALIDATED_STRUCTURE`` refusal ``capture_request_only_bundle``
    records leaves ``source_inputs`` populated (the request's own facts:
    ticker, strategy, structure) even though NONE of the four required
    groups are ever reached. The old rule read ONLY those four groups to
    decide ``executable_inputs``, so a row that plainly had real,
    resolved inputs still came out ``missing`` -> forced ``incomparable``
    by the validator's own rule -- collapsing "the system correctly
    refused this row" into the same bucket as "this row could not be
    evaluated at all". Fixed: ``executable_inputs`` now reads ANY
    hash-verified group, including ``source_inputs``.
    """
    candidate = {
        "fixture_id": "case-unvalidated",
        "covers": ["strategy:CAL-P"],
        "request": {"strategy": "CAL-P", "ticker": "CODA"},
        "record": {"strategy": "CAL-P", "ticker": "CODA"},
        "kind": "score_result",
        "duration": 0.05,
        "legacy_trace": {
            "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
            "disposition": {
                "status": "refused", "flags": ["UNVALIDATED_STRUCTURE"],
                "detail": "CAL-P is not scored",
                "first_gap": {"stage": "resolve_context",
                              "reason": "disabled strategy"},
            },
            "checkpoints": {
                "source_inputs": _hashed({
                    "context": {"ticker": "CODA", "strategy": "CAL-P"},
                    "quote_status": "not_reached", "scope": "request_only",
                }),
            },
        },
    }
    release = tmp_path / "release"
    write(release, [candidate], {"strategy:CAL-P": ["case-unvalidated"]},
          pd.Timestamp("2026-01-01"), "snapshot-1")

    case = json.loads(
        (release / "checkpoints" / "cases" / "case-unvalidated.json").read_text())
    # `source_inputs` itself is never part of `checkpoints` (not in the
    # schema's REQUIRED/OPTIONAL vocabulary) -- only its EVIDENCE that
    # real inputs resolved changes `executable_inputs`.
    assert case["checkpoints"] == {}
    assert case["executable_inputs"] == "available"
    assert case["disposition"] == "refused_as_expected"
    assert case["first_gap"] == {
        "stage": "resolve_context", "reason": "disabled strategy",
    }

    case_id, strategy, branches = _verify_case(release, "case-unvalidated")
    assert case_id == "case-unvalidated"
    assert strategy == "CAL-P"
    assert "missing_inputs" in branches


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


# --------------------------------------------------------------------------
# branch labelling: fallback, multi_expiry, and the research-replay trace
# --------------------------------------------------------------------------


def _written_branches(tmp_path, trace, *, record=None, kind="score_result",
                      case_id="case-1"):
    """Run the REAL producer and read the branches off the case the REAL
    per-case validator accepted -- not off ``_phase4_case_branches``
    directly, so a label that the writer drops on the way to disk cannot
    pass (the `_clean_row`-versus-artifact trap, AGENTS.md)."""
    candidate = {
        "fixture_id": case_id,
        "covers": ["strategy:STR-THRU"],
        "request": {"strategy": "STR-THRU", "ticker": "ABC"},
        "record": record if record is not None else {"strategy": "STR-THRU",
                                                     "ticker": "ABC"},
        "kind": kind,
        "duration": 0.1,
        "legacy_trace": trace,
    }
    release = tmp_path / "release"
    write(release, [candidate], {"strategy:STR-THRU": [case_id]},
          pd.Timestamp("2026-01-01"), "snapshot-1")
    return release, _verify_case(release, case_id)


def test_a_gate_that_fell_back_to_the_arithmetic_entry_rule_is_labelled_fallback(
        tmp_path) -> None:
    """``gate_inputs.kind == "entry_rule"`` is legacy's own record that the
    registry had NO gate model for this strategy and the arithmetic rule
    stood in (``Scorer._score_gate`` calls ``_apply_entry_rule`` only on
    ``loaded is None``). Without this rule the ``fallback`` branch is
    unreachable corpus-wide: the only other fallback the checkpoints carry
    is ``residual_population_identity.fallback_used``, which needs a causal
    residual pool under ~2,500 rows and therefore cannot fire on any event
    the capture can reach."""
    _release, (_case_id, _strategy, branches) = _written_branches(
        tmp_path, _full_legacy_trace())
    assert "fallback" in branches


def test_a_gate_scored_by_a_registered_model_is_not_labelled_fallback(
        tmp_path) -> None:
    """The negative control. Both arms occur in the real corpus, so
    ``fallback`` must discriminate; a rule that fired on every gated row
    would be a constant wearing a branch label."""
    trace = copy.deepcopy(_full_legacy_trace())
    trace["checkpoints"]["gate_inputs"] = _hashed({
        "kind": "model", "model_identity": "gate-v1",
        "feature_vector": {"spot": 100.0}, "threshold": 0.5,
    })
    _release, (_case_id, _strategy, branches) = _written_branches(tmp_path, trace)
    assert "fallback" not in branches
    assert branches  # still a real case, still labelled


def test_legs_spanning_two_expiries_are_labelled_multi_expiry(tmp_path) -> None:
    """``multi_expiry`` reads the per-leg ``expiry`` legacy priced, and
    only a calendar can differ across legs. Without the two-expiry arm the
    branch never fires; without the single-expiry arm below the rule could
    be labelling every priced row."""
    trace = copy.deepcopy(_full_legacy_trace())
    trace["checkpoints"]["selection_pricing"] = _hashed({
        "selected_legs": [
            {"right": "put", "side": "short", "quantity": 1, "expiry": "2026-01-16"},
            {"right": "put", "side": "long", "quantity": 1, "expiry": "2026-02-20"},
        ],
        "entry_cost": 1.25,
    })
    _release, (_case_id, _strategy, branches) = _written_branches(tmp_path, trace)
    assert "multi_expiry" in branches

    _release2, (_id2, _s2, single) = _written_branches(
        tmp_path / "single", _full_legacy_trace(), case_id="case-2")
    assert "multi_expiry" not in single


def _replay_rows():
    """Two fill-alpha rows shaped like ``engine/replay.py``'s own
    ``include_legs=True`` output. The legs are IDENTICAL across alphas
    because ``replay_one`` pins the contracts from the first alpha; only
    the cost moves."""
    legs = [
        {"name": "front", "right": "put", "side": "short", "qty": 1,
         "strike": 100.0, "expiry": "2026-01-16", "bid": 1.0, "ask": 1.2,
         "price": 1.1},
        {"name": "back", "right": "put", "side": "long", "qty": 1,
         "strike": 100.0, "expiry": "2026-02-20", "bid": 3.0, "ask": 3.4,
         "price": 3.2},
    ]
    return [
        {"ticker": "ABC", "fill_alpha": 0.5, "entry_cost": 2.1,
         "entry_legs": copy.deepcopy(legs)},
        {"ticker": "ABC", "fill_alpha": 1.0, "entry_cost": 2.4,
         "entry_legs": copy.deepcopy(legs)},
    ]


def test_a_research_replay_result_becomes_a_case_the_real_validator_accepts(
        tmp_path) -> None:
    """CAL-P is the only structure whose legs span two expiries and
    ``Scorer.score`` refuses it at ``resolve_context``, so
    ``engine.replay.replay_one`` is the ONLY legacy path that ever prices
    a multi-expiry structure. Before this, ``research_replay_pass`` built
    its candidate with no ``legacy_trace``, so ``write()``'s
    ``raw_checkpoint is not None`` guard dropped every one of them and the
    bundle carried no priced CAL-P leg at all."""
    rows = _replay_rows()
    trace = capture._research_replay_trace(rows)
    assert set(trace) == {"schema_version", "disposition", "checkpoints"}
    # Only what legacy actually produced -- no fabricated empty groups.
    assert set(trace["checkpoints"]) == {"selection_pricing"}
    priced = trace["checkpoints"]["selection_pricing"]["value"]
    assert priced["selected_legs"] == rows[0]["entry_legs"]
    assert priced["entry_cost"] == rows[0]["entry_cost"]
    assert trace["disposition"]["first_gap"] == {
        "stage": "features", "reason": capture._RESEARCH_REPLAY_GAP_REASON,
    }

    release, (case_id, strategy, branches) = _written_branches(
        tmp_path, trace, kind="research_replay",
        record={"strategy": "CAL-P", "ticker": "ABC"}, case_id="calp-1")
    assert case_id == "calp-1" and strategy == "CAL-P"
    assert "multi_expiry" in branches and "debit_credit" in branches
    case = json.loads(
        (release / "checkpoints" / "cases" / "calp-1.json").read_text())
    assert case["disposition"] == "incomparable"
    assert case["first_gap"] == {
        "stage": "features", "reason": capture._RESEARCH_REPLAY_GAP_REASON,
    }


def test_research_replay_trace_refuses_to_invent_a_checkpoint() -> None:
    """No priced row, no legs or no cost means legacy computed nothing to
    record; an empty ``selection_pricing`` group would be a fabrication."""
    assert capture._research_replay_trace([]) is None
    assert capture._research_replay_trace(
        [{"ticker": "ABC", "entry_cost": 2.1}]) is None
    assert capture._research_replay_trace(
        [{"ticker": "ABC", "entry_legs": [{"expiry": "2026-01-16"}],
          "entry_cost": None}]) is None


def test_the_tie_audit_reads_the_choosers_own_published_margin() -> None:
    """The choice-level evidence needed to settle "did a tie occur" is
    already in the corpus: ``dynamic_short_vol`` publishes
    ``chosen_margin`` on the row it returns and ``dyn_sv_pass`` keeps that
    row verbatim. ``None`` means no runner-up existed, so the tie path was
    never reachable for that choice and it must not be counted as an
    examination."""
    candidates = [
        {"kind": "dyn_sv_choice", "record": {"chosen_margin": 0.42}},
        {"kind": "dyn_sv_choice", "record": {"chosen_margin": -0.017}},
        {"kind": "dyn_sv_choice", "record": {"chosen_margin": None}},
        {"kind": "score_result", "record": {"chosen_margin": 0.0}},
    ]
    assert capture.tie_audit(candidates) == {
        "examined": 2, "exercised": 0, "closest": 0.017,
    }
    tied = capture.tie_audit(
        [{"kind": "dyn_sv_choice", "record": {"chosen_margin": 0.0}}])
    assert tied == {"examined": 1, "exercised": 1, "closest": 0.0}
    assert capture.tie_audit([]) == {
        "examined": 0, "exercised": 0, "closest": None,
    }
