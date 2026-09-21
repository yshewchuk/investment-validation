from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import engine.score as score_module
from engine.pnl_sim import DRAWS, ResidualPool
from engine.score import (
    Phase4TraceCollector,
    ScoreRequest,
    ScoreResult,
    Scorer,
    _Predocumented,
    _size_feature_capture_value,
)
from engine.v2.foundation import content_hash


def test_collector_is_opt_in_and_default_finish_does_not_mutate_result() -> None:
    scorer = Scorer.__new__(Scorer)
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
    )

    finished = scorer._finish_phase4_trace(None, result)

    assert finished is result
    assert not hasattr(result, "_phase4_trace")


def test_early_refusal_records_the_first_gap_stage_and_reason() -> None:
    """The REAL early-return path (`Scorer._finish_phase4_trace`, used
    verbatim by the ``SUPERSEDED``/disabled-strategy returns in
    ``Scorer.score``) marks every stage ``not_reached``; the FIRST one and
    its reason must survive into ``diagnostic_checkpoint()['disposition']``
    even with ``retain_full_trace=False`` (the capture tool's own setting),
    so a capture producer can write an honest early-refusal disposition
    instead of treating "no checkpoint groups" as unexplained."""
    scorer = Scorer.__new__(Scorer)
    collector = Phase4TraceCollector(retain_full_trace=False, content_hasher=content_hash)
    result = ScoreResult(ticker="ABC", strategy="STR-THRU", as_of=pd.Timestamp("2026-01-02"))
    result.flag("SUPERSEDED")

    finished = scorer._finish_phase4_trace(collector, result, "superseded strategy")

    assert finished is result
    checkpoint = collector.diagnostic_checkpoint()
    assert checkpoint["disposition"]["status"] == "refused"
    assert checkpoint["disposition"]["flags"] == ["SUPERSEDED"]
    assert checkpoint["disposition"]["first_gap"] == {
        "stage": "resolve_context", "reason": "superseded strategy",
    }
    assert checkpoint["checkpoints"] == {}


def test_collector_records_json_safe_stage_documents_and_refusal_identity() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.begin(ScoreRequest(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
        event_date=pd.Timestamp("2026-01-08"),
    ))
    collector.record(
        "features",
        {"frame": pd.DataFrame({"x": [1.0], "missing": [float("nan")]})},
        {"as_of": pd.Timestamp("2026-01-02")},
    )
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
    )
    result.flag("NO_CHAIN")
    collector.finish(result)

    document = collector.document()

    assert document["schema_version"] == "phase4_legacy_trace.v1.0"
    assert document["status"] == "refused"
    assert document["request"]["event_date"] == "2026-01-08"
    assert document["stages"]["features"]["input"]["frame"][0] == {
        "x": 1.0,
        "missing": None,
    }
    assert document["stages"]["serialization"]["status"] == "refused"


def test_compact_collector_does_not_retain_broad_stage_values() -> None:
    collector = Phase4TraceCollector(retain_full_trace=False)
    collector.begin(ScoreRequest(ticker="ABC", strategy="STR-THRU"))
    collector.record("features", {"large": ["discard-me"]}, {"value": 1.0})

    document = collector.document()

    assert document["stages"]["features"] == {"status": "completed"}


def test_corpus_capture_requests_only_bounded_checkpoint(monkeypatch) -> None:
    import tools.capture_tier0_corpus as capture

    class StubTrace:
        def __init__(self, *, retain_full_trace: bool, content_hasher) -> None:
            assert retain_full_trace is False
            assert content_hasher is not None

        def diagnostic_checkpoint(self) -> dict:
            return {"bounded": True}

        def document(self) -> dict:
            raise AssertionError("capture requested the broad trace")

    class StubResult:
        def as_dict(self) -> dict:
            return {"strategy": "STR-THRU"}

    class StubScorer:
        snapshot = "snapshot"

        def score(self, request, **kwargs):
            return StubResult()

    monkeypatch.setattr(capture.score_mod, "Phase4TraceCollector", StubTrace)
    request = ScoreRequest(ticker="ABC", strategy="STR-THRU")

    _, record, _, checkpoint = capture._score(StubScorer(), request)

    assert record == {"strategy": "STR-THRU"}
    assert checkpoint == {"bounded": True}


def _checkpoint_value(collector: Phase4TraceCollector, name: str) -> dict:
    row = collector.diagnostic_checkpoint()["checkpoints"][name]
    assert row["content_hash"] == content_hash(row["value"])
    return row["value"]


def test_model_boundary_captures_exact_vector_mask_and_identity() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    request = ScoreRequest(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
        event_date=pd.Timestamp("2026-01-08"),
    )
    collector.begin(request)
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
        event_date=pd.Timestamp("2026-01-08"),
    )
    result._phase4_checkpoint_collector = collector
    artifact = SimpleNamespace(features=("exact", "missing"))
    entry = SimpleNamespace(id="size-v7")
    scorer = Scorer.__new__(Scorer)
    scorer.model = lambda role, *args, **kwargs: (
        (entry, artifact) if role == "size" else None
    )

    scorer._score_model(
        request,
        result,
        pd.DataFrame({"exact": [1.25], "missing": [float("nan")]}),
    )
    collector.finish(result)

    features = _checkpoint_value(collector, "features")
    assert features == {
        "feature_vector": {
            "abs_move": {"exact": 1.25, "missing": None},
        },
        "missing_mask": {
            "abs_move": {"exact": False, "missing": True},
        },
        "model_identity": {
            "abs_move": {
                "model_id": "size-v7",
                "role": "abs_move",
                "input_as_of": "2026-01-02",
            },
        },
    }
    checkpoint = collector.diagnostic_checkpoint()
    assert checkpoint["disposition"]["status"] == "refused"
    assert checkpoint["disposition"]["flags"] == ["MISSING_FEATURES"]
    assert "simulation" not in checkpoint["checkpoints"]
    assert "gate_inputs" not in checkpoint["checkpoints"]


def test_size_feature_capture_value_distinguishes_absent_from_nonfinite() -> None:
    features = pd.DataFrame({"present_finite": [5.0], "present_nan": [float("nan")]})

    assert _size_feature_capture_value(features, "present_finite") == 5.0
    tagged = _size_feature_capture_value(features, "present_nan")
    assert tagged == {"__nonfinite__": "nan"}
    assert _size_feature_capture_value(features, "absent_column") is None


def test_capture_features_marks_a_nonfinite_tag_as_missing() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_features(
        {"quoted": 5.0, "unquoted": {"__nonfinite__": "nan"}, "absent": None},
        {"model_id": "m", "role": "size"},
        role="size",
    )
    features = _checkpoint_value(collector, "features")
    mask = features["missing_mask"]["size"]
    assert mask == {"quoted": False, "unquoted": True, "absent": True}


def test_source_bundle_is_bounded_and_rejects_scoring_answers() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_source_bundle(
        context={"ticker": "ABC", "strategy": "STR-THRU"},
        quote_domain=[{"right": "call", "strike": 100.0, "bid": 1.0, "ask": 2.0}],
        features={"iv30": 0.25},
        model_bindings=({"role": "size", "model_id": "size-v7"},),
    )
    source = _checkpoint_value(collector, "source_inputs")

    assert source["context"] == {"ticker": "ABC", "strategy": "STR-THRU"}
    assert source["quote_domain"][0]["strike"] == 100.0
    assert source["model_bindings"][0]["role"] == "size"

    with pytest.raises(ValueError, match="scoring answers"):
        collector.capture_source_bundle(features={"entry_cost": 4.0})


@pytest.mark.parametrize(
    ("legacy_role", "output_name", "canonical_role"),
    (
        ("abs_move", "driver_prediction", "driver"),
        ("forecast_sizing", "forecast_abs_move", "size"),
    ),
)
def test_source_bundle_declares_canonical_forecast_roles(
    legacy_role: str,
    output_name: str,
    canonical_role: str,
) -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_source_bundle(model_bindings=({
        "role": legacy_role,
        "model_id": "model-v1",
        "output_names": (output_name,),
    },))

    source = _checkpoint_value(collector, "source_inputs")

    assert source["native_recipes"]["forecast"]["required_roles"] == [
        canonical_role,
    ]


def test_source_bundle_unions_forecast_roles_across_repeated_calls() -> None:
    """`capture_source_bundle` is called once per role over the life of a
    row's scoring pass (driver, then size, then implied_t1, ...). Each call
    must ADD its role to `required_roles`, not replace the roles earlier
    calls contributed -- otherwise only the last call's role survives."""
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_source_bundle(model_bindings=({
        "role": "abs_move", "model_id": "driver-v1",
    },))
    collector.capture_source_bundle(model_bindings=({
        "role": "runup_move", "model_id": "runup-v1",
    },))
    collector.capture_source_bundle(model_bindings=({
        "role": "implied_t1", "model_id": "implied-v1",
    },))

    source = _checkpoint_value(collector, "source_inputs")

    # First-seen order across ALL calls, not just the last one.
    assert source["native_recipes"]["forecast"]["required_roles"] == [
        "driver", "runup_move", "implied_t1",
    ]


def test_simulation_checkpoint_keeps_causal_residual_population() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_simulation(
        horizon={"event_date": "2026-01-02", "dte_exit": 5.0},
        capital_denominator=4.25,
        evidence={
            "draw_count": 4,
            "seed": 17,
            "residual_draw": {
                "cutoff": "2026-01-02",
                "cutoff_index": 2,
                "bucket_count": 10,
                "bucket_index": 4,
                "eligible_indices": [0, 1],
                "fallback_used": False,
            },
            "residual_rows": [{"event_date": "2025-12-01", "err_move": 1.0}],
            "residual_population": [
                {"event_date": "2025-11-01", "pred_abs_move": 4.0,
                 "err_move": 0.5, "err_crush": -2.0},
                {"event_date": "2025-12-01", "pred_abs_move": 5.0,
                 "err_move": 1.0, "err_crush": 3.0},
            ],
        },
    )

    simulation = _checkpoint_value(collector, "simulation")
    assert simulation["residual_population"] == [
        {"event_date": "2025-11-01", "pred_abs_move": 4.0,
         "err_move": 0.5, "err_crush": -2.0},
        {"event_date": "2025-12-01", "pred_abs_move": 5.0,
         "err_move": 1.0, "err_crush": 3.0},
    ]


def test_predocumented_population_is_not_recopied_and_stays_shared() -> None:
    """The retention fix: `capture_simulation` (via `_checkpoint`) and
    `capture_source_bundle` each walk their WHOLE argument through
    `_document` unconditionally. A residual population shared by reference
    across many rescored candidates of the same boundary event must reach
    BOTH checkpoint groups as the SAME object, not a fresh copy per group --
    otherwise the sharing a caller set up (`ResidualPool.documented_population`)
    is silently undone one layer in, and the retention it exists to avoid
    comes right back.
    """
    shared_population = [
        {"event_date": "2025-11-01", "pred_abs_move": 4.0,
         "err_move": 0.5, "err_crush": -2.0},
    ]
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_simulation(
        horizon={"event_date": "2026-01-02", "dte_exit": 5.0},
        capital_denominator=4.25,
        evidence={
            "draw_count": 4,
            "seed": 17,
            "residual_draw": {
                "cutoff": "2026-01-02", "cutoff_index": 1, "bucket_count": 10,
                "bucket_index": 4, "eligible_indices": [0], "fallback_used": False,
            },
            "residual_rows": [],
            "residual_population": shared_population,
        },
    )
    collector.capture_source_bundle(features={"spot": 1.0})

    simulation = _checkpoint_value(collector, "simulation")
    source = _checkpoint_value(collector, "source_inputs")

    assert simulation["residual_population"] is shared_population
    assert source["native_recipes"]["simulation"]["residuals"] is shared_population
    # `_document` did not copy it, but the checkpoint's own value is still
    # exactly what a plain (unwrapped) list would have produced -- the hash
    # in `_checkpoint_value` already proves it round-trips through
    # `content_hash`; this proves the VALUE is unchanged from the source.
    assert simulation["residual_population"] == shared_population


def test_score_captures_boundary_legs_and_cost_before_later_mutation(
        monkeypatch) -> None:
    monkeypatch.setattr(score_module, "assert_decision_causal", lambda *args, **kwargs: None)
    scorer = Scorer.__new__(Scorer)
    scorer.snapshot = "snap-test"
    window = SimpleNamespace(
        entry_date=pd.Timestamp("2026-01-06"),
        exit_date=pd.Timestamp("2026-01-09"),
        decision_date=pd.Timestamp("2026-01-05"),
    )
    scorer.calendar = SimpleNamespace(
        resolve_offsets=lambda *args, **kwargs: window,
        is_projected=lambda value: False,
    )
    scorer._resolve_event = lambda request: (
        pd.Timestamp("2026-01-08"), "AMC",
    )
    scorer._structure = lambda request: SimpleNamespace(
        entry_offset=0,
        exit_offset=1,
        decision_offset=None,
        decided_early=False,
    )
    expected_legs = [{
        "name": "long_put",
        "side": "buy",
        "right": "put",
        "qty": 1.0,
        "strike": 100.0,
        "expiry": "2026-01-16",
    }]

    def price_entry(request, structure, result, chain_index) -> None:
        result.legs = expected_legs
        result.entry_cost = 4.25
        result.spot = 100.0
        result.quote_date = result.entry_date

    scorer._price_entry = price_entry
    scorer._features = lambda request, result: pd.DataFrame({"x": [1.0]})
    scorer._quote_today = lambda ticker, as_of: None

    def score_model(request, result, features) -> None:
        result.exp_pnl_model = 0.1

    def mutate_after_pricing(request, result, features) -> None:
        result.entry_cost = 999.0
        result.legs = [{"name": "contradictory-final-record"}]

    scorer._score_model = score_model
    scorer._score_analogs = lambda request, result, features: None
    scorer._score_gate = mutate_after_pricing
    scorer._compare_layers = lambda result: None
    collector = Phase4TraceCollector(content_hasher=content_hash)

    scorer.score(
        ScoreRequest(
            ticker="ABC",
            strategy="STR-THRU",
            as_of=pd.Timestamp("2026-01-05"),
            event_date=pd.Timestamp("2026-01-08"),
            session="AMC",
        ),
        trace=collector,
    )

    selection = _checkpoint_value(collector, "selection_pricing")
    assert selection == {
        "selected_legs": expected_legs,
        "entry_cost": 4.25,
    }


def test_forecast_sized_entry_rule_run_ends_with_analogs_and_simulation_frozen(
        monkeypatch) -> None:
    """Regression for the staleness bug: for a FORECAST_SIZED,
    entry-rule-gated strategy (TWIN-P5 here; the family also covers CND-PS,
    TWIN-P, BFLY-P, BFLY-P5, RAMP7, CTR5), the LAST ``capture_source_bundle``
    call in a real run is the model layer's driver-model binding
    (``_score_model`` -> ``_phase4_capture_served_model``), which happens
    BEFORE the analog and simulation layers run. Those layers mutate
    ``native_recipes`` directly (``capture_analog_inputs``,
    ``capture_simulation``) and the gate is an arithmetic entry rule, not a
    model, so no LATER ``capture_source_bundle`` call ever re-snapshots.
    Before the fix, the frozen ``source_inputs`` checkpoint therefore ended
    the run missing ``native_recipes.analogs`` and ``.simulation`` even
    though the live bundle had both. Asserting on the FROZEN checkpoint
    (never the live ``_source_bundle``) is the point: the live bundle was
    always right.
    """
    monkeypatch.setattr(score_module, "assert_decision_causal", lambda *args, **kwargs: None)
    scorer = Scorer.__new__(Scorer)
    scorer.snapshot = "snap-test"
    scorer.matcher = SimpleNamespace(phase4_recipe_cache={})
    window = SimpleNamespace(
        entry_date=pd.Timestamp("2026-01-06"),
        exit_date=pd.Timestamp("2026-01-09"),
        decision_date=pd.Timestamp("2026-01-05"),
    )
    scorer.calendar = SimpleNamespace(
        resolve_offsets=lambda *args, **kwargs: window,
        is_projected=lambda value: False,
    )
    scorer._resolve_event = lambda request: (pd.Timestamp("2026-01-08"), "AMC")
    scorer._structure = lambda request: SimpleNamespace(
        entry_offset=0, exit_offset=1, decision_offset=None, decided_early=False,
    )
    # Already "sized": skip the real Tier-4 fold-serving call this makes in
    # production. Its own `capture_source_bundle` (model_bindings, role
    # "forecast_sizing") is not what this test pins -- the model layer's is,
    # and it is not the last call on this strategy's real path either way.
    scorer._size_from_forecast = lambda request, result, structure, size=True: (
        request, structure
    )

    def price_entry(request, structure, result, chain_index) -> None:
        result.legs = [{"name": "leg", "side": "buy", "right": "call",
                        "qty": 1.0, "strike": 100.0, "expiry": "2026-01-16"}]
        result.entry_cost = 4.25
        result.spot = 100.0
        result.quote_date = result.entry_date

    scorer._price_entry = price_entry
    scorer._features = lambda request, result: pd.DataFrame({"x": [1.0]})
    scorer._quote_today = lambda ticker, as_of: None

    def score_model(request, result, features) -> None:
        # The real `_score_model` -> `_phase4_capture_served_model` call for
        # the driver model: the LAST `capture_source_bundle` call on this
        # strategy's real path.
        collector = result._phase4_checkpoint_collector
        collector.capture_source_bundle(model_bindings=({
            "model_id": "abs-move-v3", "role": "abs_move",
            "feature_order": ("x",), "artifact": "models/abs_move.joblib",
            "artifact_sha256": "a" * 64, "adapter": "joblib-estimator.v1",
            "output_names": ("driver_prediction",), "strategy": "TWIN-P5",
            "decision_offset": None, "input_as_of": "2026-01-05",
        },))
        result.exp_pnl_model = 0.1

    def score_analogs(request, result, features) -> None:
        # The real `_score_analogs` only sets this; `score()` itself calls
        # `capture_analog_inputs` from it (see the call right after
        # `_score_analogs` in `Scorer.score`).
        result._phase4_analog_evidence = {
            "strategy": "TWIN-P5", "alpha": 0.5, "snapshot": "snap-test",
            "cutoff": "2024-01-01T00:00:00", "request_key": "req-1",
            "bucket_query": {"mcap_bucket": "large", "moneyness_band": "atm",
                            "dte_band": "short", "implied_tercile": "mid"},
            "causal": {"rows": [{
                "row_id": "r1", "source_index": "r1",
                "values": {"mcap_bucket": "large", "moneyness_band": "atm",
                          "dte_band": "short", "implied_tercile": "mid",
                          "ret": 0.1},
            }]},
        }

    def score_gate(request, result, features) -> None:
        # The real `_apply_entry_rule` -> `_simulated_pnl` -> `_expectation`
        # call for an entry-rule-gated strategy: it mutates
        # `_source_bundle["native_recipes"]["simulation"]` and records the
        # arithmetic verdict, with NO further `capture_source_bundle` call
        # on this path (the gate is a rule, not a model).
        collector = result._phase4_checkpoint_collector
        collector.capture_simulation(
            horizon={"event_date": "2026-01-08", "dte_exit": 3.0},
            capital_denominator=4.25,
            evidence={
                "draw_count": 2000, "seed": 17,
                "residual_draw": {
                    "cutoff": "2026-01-08", "cutoff_index": 1, "bucket_count": 2,
                    "bucket_index": 0, "eligible_indices": [0], "fallback_used": False,
                },
                "residual_rows": [],
                "residual_population": [{"event_date": "2025-12-01",
                                        "pred_abs_move": 4.0, "err_move": 0.5,
                                        "err_crush": -2.0}],
            },
        )
        collector.capture_gate_inputs({
            "kind": "entry_rule", "rule_identity": "entry-rule:TWIN-P5",
            "facts": {"exp_pnl_sim": 0.05}, "terms": [],
        })
        result.gate_pass = True

    scorer._score_model = score_model
    scorer._score_analogs = score_analogs
    scorer._score_gate = score_gate
    scorer._score_chooser = lambda request, result, features: None
    scorer._compare_layers = lambda result: None
    collector = Phase4TraceCollector(content_hasher=content_hash)

    scorer.score(
        ScoreRequest(
            ticker="ABC", strategy="TWIN-P5",
            as_of=pd.Timestamp("2026-01-05"),
            event_date=pd.Timestamp("2026-01-08"), session="AMC",
        ),
        trace=collector,
    )

    recipes = _checkpoint_value(collector, "source_inputs")["native_recipes"]
    assert recipes["forecast"]["required_roles"] == ["driver"]
    assert "analogs" in recipes and recipes["analogs"]["recipe"]["alpha"] == 0.5
    assert "simulation" in recipes and recipes["simulation"]["mode"] == "planned_exit"


def test_analogs_that_genuinely_did_not_match_stay_absent_from_the_frozen_checkpoint() -> None:
    """The other half of the ordering fix: a stage that finds nothing must
    stay absent from the FROZEN checkpoint, not just the live bundle --
    `_refresh_source_inputs` must never fabricate a recipe for a stage that
    did not produce one. `capture_analog_inputs` returns early (no
    ``source_rows``) when the causal population is empty, so it never
    reaches the `native_recipes["analogs"]` assignment or the refresh call
    after it; this pins that the frozen checkpoint reflects that absence."""
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_source_bundle(context={"ticker": "ABC"})
    collector.capture_analog_inputs({
        "strategy": "STR-THRU", "alpha": 0.5, "snapshot": "snap",
        "cutoff": "2024-01-01T00:00:00", "request_key": "req",
        "bucket_query": {"mcap_bucket": "large", "moneyness_band": "atm",
                        "dte_band": "short", "implied_tercile": "mid"},
        "causal": {"rows": []},
    })

    recipes = _checkpoint_value(collector, "source_inputs")["native_recipes"]
    assert "analogs" not in recipes


def test_analog_recipe_seed_matches_legacy_matchers_own_seed_call() -> None:
    """``capture_analog_inputs`` must derive ``bootstrap_seed`` from exactly
    the buckets dict ``AnalogMatcher._summarize`` actually hashed in
    ``_seed`` -- the CAUSALLY re-bucketed ``effective_bucket_query`` (which
    also still carries the raw ``implied_ratio`` key ``_seed`` hashes,
    alongside the four bucket labels), not the pre-causal ``bucket_query``
    snapshot restricted to the four bucket-dimension names.

    Before the fix this read ``bucket_query`` and dropped ``implied_ratio``
    from the hashed payload, so the captured seed differed from legacy's own
    -- same matched population (bucket labels only ever come from the four
    dimension names, unaffected), same mean/win/n, but an unrelated bootstrap
    draw and therefore a different ci_low/ci_high than the legacy record
    this capture exists to let native reproduce.
    """
    from engine.v2.scoring.native_analog import legacy_bucket_bootstrap_seed

    bucket_query = {
        "mcap_bucket": "large", "moneyness_band": "atm", "dte_band": "short",
        "implied_tercile": "mid", "implied_ratio": 1.05,
    }
    # The CAUSAL re-bucketing: same raw ratio, a DIFFERENT tercile label
    # than the population-edge one above -- exactly what `match()` produces
    # when `as_of` causal edges disagree with the population edges.
    effective_bucket_query = {
        **bucket_query, "implied_tercile": "high",
    }
    evidence = {
        "strategy": "STR-THRU", "alpha": 0.5, "snapshot": "snap-test",
        "cutoff": "2024-01-01T00:00:00", "request_key": "req-1",
        "bucket_query": bucket_query,
        "effective_bucket_query": effective_bucket_query,
        "causal": {"rows": [{
            "row_id": "r1", "source_index": "r1",
            "values": {"mcap_bucket": "large", "moneyness_band": "atm",
                      "dte_band": "short", "implied_tercile": "high",
                      "ret": 0.1},
        }]},
    }
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_source_bundle(context={"ticker": "ABC"})
    collector.capture_analog_inputs(evidence)

    analogs_block = _checkpoint_value(collector, "source_inputs")["native_recipes"]["analogs"]
    recipe = analogs_block["recipe"]

    # What `AnalogMatcher._summarize`'s own `_seed()` call actually hashes:
    # every key of the buckets object `match()` passed it, which is
    # `effective_bucket_query` here (causal tercile + implied_ratio both
    # present).
    expected_seed = legacy_bucket_bootstrap_seed(
        snapshot="snap-test", strategy="STR-THRU", alpha=0.5,
        buckets=effective_bucket_query, request_key="req-1",
    )
    assert recipe["bootstrap_seed"] == expected_seed

    # The old, buggy derivation: `bucket_query` (pre-causal), restricted to
    # only the four bucket-dimension names (no `implied_ratio`). Pins that
    # the fix actually changed the derivation, not just its inputs.
    stale_and_restricted = {
        k: bucket_query[k]
        for k in ("mcap_bucket", "moneyness_band", "dte_band", "implied_tercile")
    }
    old_buggy_seed = legacy_bucket_bootstrap_seed(
        snapshot="snap-test", strategy="STR-THRU", alpha=0.5,
        buckets=stale_and_restricted, request_key="req-1",
    )
    assert recipe["bootstrap_seed"] != old_buggy_seed

    # The causal tercile also reaches `query_features` (used for MATCHING in
    # native replay), not the stale population-edge one.
    assert analogs_block["query_features"]["implied_tercile"] == "high"


def _residual_pool() -> ResidualPool:
    index = np.arange(600)
    return ResidualPool(pd.DataFrame({
        "event_date": pd.date_range("2020-01-01", periods=600, freq="D"),
        "pred_abs_move": 2.0 + index / 100.0,
        "err_move": index / 200.0 - 1.5,
        "err_crush": 4.0 - index / 100.0,
    }), buckets=2)


def test_simulation_boundary_is_bounded_and_records_reproducibility() -> None:
    scorer = Scorer.__new__(Scorer)
    scorer._residual_pool = _residual_pool
    scorer._pre_print_iv = lambda request, result, features: 42.0
    scorer._crush_forecast = lambda request, result, features: -18.0
    collector = Phase4TraceCollector(content_hasher=content_hash)
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2022-01-01"),
        event_date=pd.Timestamp("2022-01-01"),
        exit_date=pd.Timestamp("2022-01-02"),
        expiry=pd.Timestamp("2022-01-10"),
        spot=100.0,
        entry_cost=4.25,
        forecast_abs_move=3.0,
    )
    result._priced_legs = (
        SimpleNamespace(strike=100.0, qty=1.0, side="buy"),
    )
    result._phase4_trace_requested = True
    result._phase4_checkpoint_collector = collector

    simulated = scorer._expectation(
        ScoreRequest(
            ticker="ABC",
            strategy="STR-THRU",
            event_date=pd.Timestamp("2022-01-01"),
        ),
        result,
    )

    assert simulated is not None
    simulation = _checkpoint_value(collector, "simulation")
    assert simulation["horizon"] == {
        "exit_date": "2022-01-02",
        "expiry": "2022-01-10",
        "dte_exit": 8.0,
    }
    assert simulation["capital_denominator"] == 4.25
    assert simulation["draw_count"] == DRAWS
    assert type(simulation["seed"]) is int
    identity = simulation["residual_population_identity"]
    assert identity["cutoff_index"] == 600
    assert identity["population_hash"].startswith("sha256:")
    assert identity["selected_rows_hash"].startswith("sha256:")
    encoded = json.dumps(collector.diagnostic_checkpoint())
    assert "residual_rows" not in encoded
    assert "selected_indices" not in encoded


def test_model_gate_and_dyn_sv_capture_actual_consumed_vectors() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
    )
    result._phase4_checkpoint_collector = collector
    gate_entry = SimpleNamespace(id="gate-v3", threshold=0.6)
    gate_artifact = SimpleNamespace(
        features=("gate_x",),
        predict=lambda values: np.asarray([0.75]),
    )
    scorer = Scorer.__new__(Scorer)
    scorer.model = lambda role, *args, **kwargs: (
        (gate_entry, gate_artifact) if role == "gate" else None
    )
    scorer._gate_in_domain = lambda request, features: True
    scorer._gate_feature_frame = (
        lambda request, scored, features, wanted: features
    )

    scorer._score_gate(
        ScoreRequest(ticker="ABC", strategy="STR-THRU"),
        result,
        pd.DataFrame({"gate_x": [2.5]}),
    )

    assert _checkpoint_value(collector, "gate_inputs") == {
        "kind": "model",
        "model_identity": "gate-v3",
        "feature_vector": {"gate_x": 2.5},
        "threshold": 0.6,
    }

    chooser = Phase4TraceCollector(content_hasher=content_hash)
    chooser_result = ScoreResult(
        ticker="ABC",
        strategy="TWIN-P5",
        as_of=pd.Timestamp("2026-01-02"),
    )
    chooser_result._phase4_checkpoint_collector = chooser
    chooser_entry = SimpleNamespace(id="chooser-v2")
    chooser_artifact = SimpleNamespace(
        features=("rank_x",),
        model=SimpleNamespace(predict=lambda values: np.asarray([0.9])),
    )
    scorer.model = lambda role, *args, **kwargs: (
        (chooser_entry, chooser_artifact) if role == "chooser" else None
    )
    scorer._chooser_frame = (
        lambda request, scored, features, wanted: {"rank_x": 7.0}
    )

    scorer._score_chooser(
        ScoreRequest(ticker="ABC", strategy="TWIN-P5"),
        chooser_result,
        pd.DataFrame(),
    )

    assert _checkpoint_value(chooser, "dyn_sv") == {
        "eligibility": {
            "eligible": True,
            "candidate_strategy": "TWIN-P5",
            "missing_features": [],
        },
        "ranking": {
            "model_identity": "chooser-v2",
            "feature_vector": {"rank_x": 7.0},
            "score": 0.9,
        },
    }


def test_refusal_omits_unexecuted_groups_and_default_path_is_invariant() -> None:
    scorer = Scorer.__new__(Scorer)
    scorer.snapshot = "snap-test"
    request = ScoreRequest(
        ticker="ABC",
        strategy="CAL-P",
        as_of=pd.Timestamp("2026-01-02"),
        event_date=pd.Timestamp("2026-01-08"),
        session="AMC",
    )

    default = scorer.score(request)
    collector = Phase4TraceCollector(content_hasher=content_hash)
    traced = scorer.score(request, trace=collector)

    assert default.as_dict() == traced.as_dict()
    assert not hasattr(default, "_phase4_trace")
    checkpoint = collector.diagnostic_checkpoint()
    assert checkpoint["schema_version"] == (
        "phase4_legacy_diagnostic_checkpoint.v1.0"
    )
    assert checkpoint["disposition"]["status"] == "refused"
    assert checkpoint["disposition"]["flags"] == ["UNVALIDATED_STRUCTURE"]
    # Tier-0 gaps 005/009: nothing ran, so the only group is the request-only
    # source bundle native refuses from.
    assert set(checkpoint["checkpoints"]) == {"source_inputs"}
    source = _checkpoint_value(collector, "source_inputs")
    assert source["scope"] == "request_only"
    assert source["context"] == {"ticker": "ABC", "strategy": "CAL-P",
                                 "event_date": "2026-01-08", "session": "AMC"}
    assert (source["quote_domain"], source["quote_status"]) == ([], "not_reached")
    assert not source["model_bindings"] and not source["native_recipes"]


# -- R4-18/R4-19: source_inputs.frozen ---------------------------------------------


def test_capture_frozen_records_once_shares_pools_and_refuses_conflicts() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    pool = _Predocumented({"predictions": [1.0, 2.0], "residuals": [0.5, -0.5],
                           "interval_floor": None})
    binding = {"role": "size", "model_id": "m", "inputs": {"x": 1.0, "y": None}}
    # Before any source bundle exists, nothing is checkpointed yet.
    collector.capture_frozen(bindings={"fold:size": binding},
                             fold_pools={"pred_abs_move": pool},
                             declarations={"gate": {"binding": "gate"}})
    assert "source_inputs" not in collector.diagnostic_checkpoint()["checkpoints"]

    collector.capture_source_bundle(context={"ticker": "ABC"})
    # The same values again (another call site of the same request): no-op.
    collector.capture_frozen(bindings={"fold:size": dict(binding)},
                             fold_pools={"pred_abs_move": pool},
                             declarations={"gate": {"binding": "gate"}})
    frozen = _checkpoint_value(collector, "source_inputs")["frozen"]
    assert frozen["bindings"]["fold:size"] == binding
    assert frozen["fold_pools"]["pred_abs_move"] is pool.value  # shared, not copied
    assert frozen["declarations"] == {"gate": {"binding": "gate"}}

    # A different second value never raises inside legacy scoring (the row
    # would be lost): the first value stays and the conflict is recorded.
    collector.capture_frozen(bindings={"fold:size": {**binding, "model_id": "other"}})
    collector.capture_frozen(fold_pools={"pred_abs_move": _Predocumented(
        {"predictions": [9.0], "residuals": [0.0], "interval_floor": None})})
    frozen = _checkpoint_value(collector, "source_inputs")["frozen"]
    assert frozen["bindings"]["fold:size"]["model_id"] == "m"
    assert frozen["conflicts"] == ["bindings.fold:size", "fold_pools.pred_abs_move"]
    # A later frozen record refreshes the checkpoint group.
    collector.capture_frozen(declarations={"recalibration": {"fitted": False}})
    frozen = _checkpoint_value(collector, "source_inputs")["frozen"]
    assert frozen["declarations"]["recalibration"] == {"fitted": False}


class _Served:
    """A served Tier-4 fold with a real cache file behind ``artifact_ref``."""

    def __init__(self, path, features=("a", "b"), floor=0.0):
        self.model_id, self.fold_start, self.tier3_snapshot = "fold_m", pd.Timestamp(
            "2026-09-01"), "snap"
        self.features, self.interval_floor = features, floor
        self.pool_pred, self.pool_res = np.array([1.0, 2.0]), np.array([0.25, -0.5])
        self.path = path
        self.calls = 0

    def artifact_ref(self):
        self.calls += 1
        return self.path, "ab" * 32

    def predict(self, features):
        return np.array([0.3])


def test_crush_capture_names_the_source_legacy_used(tmp_path) -> None:
    scorer = Scorer.__new__(Scorer)
    served = _Served(tmp_path / "crush.joblib")
    scorer._serving = lambda fold, produces="pred_abs_move": served
    scorer._phase4_tier4_sha = "digest-1"
    request = ScoreRequest(ticker="ABC", strategy="TWIN-P",
                           as_of=pd.Timestamp("2026-09-02"),
                           event_date=pd.Timestamp("2026-09-10"))
    features = pd.DataFrame({"a": [1.0], "b": [float("nan")]})

    def run(stored):
        scorer._crush = stored
        collector = Phase4TraceCollector(content_hasher=content_hash)
        collector.capture_source_bundle(context={"ticker": "ABC"})
        result = ScoreResult(ticker="ABC", strategy="TWIN-P", as_of=request.as_of,
                             event_date=request.event_date)
        result._phase4_checkpoint_collector = collector
        value = scorer._crush_forecast(request, result, features)
        return value, _checkpoint_value(collector, "source_inputs")["frozen"]

    value, frozen = run({("ABC", pd.Timestamp("2026-09-10")): -12.5})
    assert value == -12.5
    stored = frozen["declarations"]["forecast:pred_iv_crush_30"]
    assert stored == {"source": "stored_tier4", "value": -12.5, "row": {
        "table": "tier4_forecasts", "table_sha256": "digest-1", "ticker": "ABC",
        "event_date": "2026-09-10"}}
    assert frozen["bindings"] == {}

    value, frozen = run({})
    assert value == pytest.approx(0.3)
    binding = frozen["bindings"]["fold:iv_crush"]
    assert (binding["role"], binding["adapter"], binding["output_names"]) == (
        "iv_crush", "tier4-serving-fold.v1", ["pred_iv_crush_30"])
    assert frozen["inputs"]["fold:iv_crush@crush"] == {"a": 1.0, "b": None}
    assert frozen["declarations"]["forecast:pred_iv_crush_30"] == {
        "source": "served_fold", "binding": "fold:iv_crush", "output": "pred_iv_crush_30",
        "site": "crush"}
    assert frozen["fold_pools"] == {}  # the crush band is never read
    # The file digest is taken once per served fold, not per candidate.
    run({})
    assert served.calls == 1


def test_fold_recording_is_inert_without_a_collector(tmp_path) -> None:
    scorer = Scorer.__new__(Scorer)
    served = _Served(tmp_path / "crush.joblib")
    result = ScoreResult(ticker="ABC", strategy="TWIN-P", as_of=pd.Timestamp("2026-09-02"))
    recorded = scorer._phase4_record_fold(
        SimpleNamespace(strategy="TWIN-P", decision_offset=None), result, served,
        pd.DataFrame({"a": [1.0]}), site="sizing")
    assert recorded is False and served.calls == 0
    assert not hasattr(scorer, "_phase4_fold_records")


# -- Tier-0 vetting gaps 001/012/003/004 (2026-09-19) --------------------------
# The resolved context is recorded as soon as resolve_context has it, the chain
# lookup says how far it got, and a BAD_QUOTE row records the models legacy
# WOULD have served. None of it changes the legacy record.

_WINDOW = SimpleNamespace(
    entry_date=pd.Timestamp("2026-01-06"),
    exit_date=pd.Timestamp("2026-01-09"),
    decision_date=pd.Timestamp("2026-01-05"),
)
_EXPECTED_CONTEXT = {
    "ticker": "ABC", "event_date": "2026-01-08", "session": "AMC",
    "entry_date": "2026-01-06", "exit_date": "2026-01-09", "as_of": "2026-01-05",
    "quote_date": "2026-01-06",
}


def _early_scorer(monkeypatch, strategy: str = "STR-THRU"):
    from engine.structures import STRUCTURES

    monkeypatch.setattr(score_module, "assert_decision_causal",
                        lambda *args, **kwargs: None)
    scorer = Scorer.__new__(Scorer)
    scorer.snapshot = "snap-test"
    scorer.calendar = SimpleNamespace(
        resolve_offsets=lambda *args, **kwargs: _WINDOW,
        is_projected=lambda value: False,
    )
    scorer._resolve_event = lambda request: (pd.Timestamp("2026-01-08"), "AMC")
    scorer._structure = lambda request: STRUCTURES[strategy]()
    scorer._features = lambda request, result: pd.DataFrame({"x": [1.0]})
    scorer._quote_today = lambda ticker, as_of: None
    scorer._note_chain_age = lambda request, result: None
    scorer._score_analogs = lambda request, result, features: None
    scorer._score_gate = lambda request, result, features: None
    scorer._compare_layers = lambda result: None
    return scorer


def _request(strategy: str = "STR-THRU") -> ScoreRequest:
    return ScoreRequest(
        ticker="ABC", strategy=strategy, as_of=pd.Timestamp("2026-01-05"),
        event_date=pd.Timestamp("2026-01-08"), session="AMC",
    )


def _source(collector) -> dict:
    return _checkpoint_value(collector, "source_inputs")


def test_no_chain_row_records_its_context_and_an_explicitly_empty_lookup(
        monkeypatch) -> None:
    """Gap 001: the NO_CHAIN row used to carry context={} and quote_domain=[]
    with nothing saying the lookup ran and came back empty."""
    scorer = _early_scorer(monkeypatch)
    scorer._score_model = lambda request, result, features: None
    empty_index = SimpleNamespace(get=lambda ticker, date: None)
    collector = Phase4TraceCollector(content_hasher=content_hash)

    traced = scorer.score(_request(), chain_index=empty_index, trace=collector)
    default = scorer.score(_request(), chain_index=empty_index)

    assert "NO_CHAIN" in traced.flags
    source = _source(collector)
    assert {key: source["context"][key] for key in _EXPECTED_CONTEXT} == _EXPECTED_CONTEXT
    assert source["context"]["strategy"] == "STR-THRU"
    assert (source["quote_domain"], source["quote_status"]) == ([], "empty")
    assert traced.as_dict() == default.as_dict()


def test_row_that_never_reaches_pricing_records_context_and_not_reached(
        monkeypatch) -> None:
    """Gap 012: a NO_FORECAST sizing refusal returns before _price_entry."""
    scorer = _early_scorer(monkeypatch, strategy="RAMP7")
    monkeypatch.setattr(score_module, "FORECAST_SIZED", {"RAMP7"})

    def decline(request, result, structure, *, size):
        result.flag("NO_FORECAST")
        return request, None

    scorer._size_from_forecast = decline

    def unreachable(*args, **kwargs):
        raise AssertionError("pricing must not run")

    scorer._price_entry = unreachable
    collector = Phase4TraceCollector(content_hasher=content_hash)

    result = scorer.score(_request("RAMP7"), trace=collector)

    assert result.flags == ["NO_FORECAST"]
    source = _source(collector)
    assert {key: source["context"][key] for key in _EXPECTED_CONTEXT} == _EXPECTED_CONTEXT
    assert (source["quote_domain"], source["quote_status"]) == ([], "not_reached")


def test_row_whose_pricer_raises_keeps_the_resolved_exit_date(monkeypatch) -> None:
    """Gap 003: COARSE_LADDER raised inside price_structure, so the post-pricing
    context capture (entry/exit/as_of) never ran."""
    scorer = _early_scorer(monkeypatch)
    scorer._score_model = lambda request, result, features: None

    def coarse(request, structure, result, chain_index):
        result.quote_date = result.entry_date
        result._phase4_checkpoint_collector.capture_source_bundle(
            quote_domain=[{"right": "C", "strike": 100.0, "expiry": "2026-01-16",
                           "bid": 1.0, "ask": 2.0}],
            quote_status="recorded",
        )
        result.flag("COARSE_LADDER")

    scorer._price_entry = coarse
    collector = Phase4TraceCollector(content_hasher=content_hash)

    scorer.score(_request(), trace=collector)

    source = _source(collector)
    assert source["context"]["exit_date"] == "2026-01-09"
    assert source["quote_status"] == "recorded"
    assert "spot" not in source["context"]


def test_bad_quote_row_records_the_models_legacy_would_serve(monkeypatch) -> None:
    """Gap 004: BAD_QUOTE leaves before _score_model, so the capture had no
    bindings, no forecast recipe and no feature vector. They are recorded as
    what _score_model would serve, and the legacy record does not move."""
    scorer = _early_scorer(monkeypatch)
    artifact = SimpleNamespace(features=("x", "y"))
    entry = SimpleNamespace(id="size-v7", path="models/size.joblib",
                            artifact_sha256="ab" * 32, decision_offset=None)
    scorer.model = lambda role, *args, **kwargs: (
        (entry, artifact) if role == "size" else None
    )

    def priced_bad(request, structure, result, chain_index):
        result.legs = [{"name": "call", "strike": 100.0}]
        result.entry_cost, result.spot = 40.0, 100.0
        result.quote_date = result.entry_date
        result.flag("BAD_QUOTE")

    scorer._price_entry = priced_bad

    def must_not_score(*args, **kwargs):
        raise AssertionError("a BAD_QUOTE row is never scored")

    scorer._score_model = must_not_score
    collector = Phase4TraceCollector(content_hasher=content_hash)

    traced = scorer.score(_request(), trace=collector)
    default = scorer.score(_request())

    assert traced.as_dict() == default.as_dict()
    assert traced.model_versions == {} and traced.model_inputs == {}
    features = _checkpoint_value(collector, "features")
    assert features["feature_vector"] == {"abs_move": {"x": 1.0, "y": None}}
    source = _source(collector)
    assert [b["model_id"] for b in source["model_bindings"]] == ["size-v7"]
    assert source["native_recipes"]["forecast"] == {"required_roles": ["driver"]}
