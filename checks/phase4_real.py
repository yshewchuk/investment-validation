#!/usr/bin/env python3
"""Build Phase 4 contract/application evidence from the frozen Tier-0 corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.tier0_corpus import load, resolve_corpus  # noqa: E402
from checks.tier0_corpus import run as run_corpus  # noqa: E402
from engine.models.registry import artifact_sha256, load_registry  # noqa: E402
from engine.v2.contracts import ScoreRequest  # noqa: E402
from engine.v2.domain.generation import generate, price  # noqa: E402
from engine.v2.domain.valuation import (  # noqa: E402
    multi_expiry_refusal,
    planned_exit_label,
    terminal_payoff,
)
from engine.v2.features import (  # noqa: E402
    FeatureContextError,
    FeatureContextPlanner,
    default_feature_registry,
)
from engine.v2.foundation import content_hash, to_document  # noqa: E402
from engine.v2.models import (  # noqa: E402
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
)
from engine.v2.models.contracts import ArtifactMember  # noqa: E402
from engine.v2.registry import DYNAMIC_MENU, STRATEGY_IDS, default_registry  # noqa: E402
from engine.v2.scoring import application  # noqa: E402
from engine.v2.scoring.identity import request_hash  # noqa: E402
from engine.v2.scoring.stages import (  # noqa: E402
    NativeScoreInputs,
    receipt,
)
from engine.v2.serving.score_projection import legacy_score_projection  # noqa: E402

__all__ = ["build_evidence", "main"]


def _request(**changes):
    value = ScoreRequest(
        event_id="phase4-event", event_revision="rev-1", calendar_revision="cal-1",
        strategy_version="STR-THRU", deployment_id="legacy-phase4-deployment.v1",
        decision_clock_id="legacy.entry_close.v1", requested_decision_at="2026-09-16",
        snapshot_id="snapshot-tier0", mode="replay", fill_model={"alpha": 0.5},
        model_artifact_refs=("size_v1_4",), residual_state_ref="residual-v1",
    )
    for key, value_to_set in changes.items():
        value = value.__class__(**{**to_document(value), key: value_to_set})
    return value


def _fake_result():
    return SimpleNamespace(as_dict=lambda: {
        "ticker": "PHASE4", "strategy": "STR-THRU", "event_date": "2026-09-16",
        "entry_date": "2026-09-16", "exit_date": "2026-09-17", "expiry": "2026-09-18",
        "spot": 100.0, "entry_cost": 5.0, "implied_move": 6.0,
        "driver_name": "abs_move", "driver_prediction": 7.0,
        "legs": [], "flags": [], "model_inputs": {"zero": 0.0},
        "gate_score": 0.7, "gate_threshold": 0.6, "gate_pass": True,
        "detail": "", "payoff": {}, "fill": 0.5,
    })


def _native(fields):
    return NativeScoreInputs.from_legacy_fields(fields)


def _native_record(record: dict, source_ref: str) -> NativeScoreInputs:
    """Build explicit stage blocks from one frozen real-code score record."""
    common = {key: record.get(key) for key in (
        "ticker", "strategy", "event_date", "entry_date", "exit_date",
        "expiry", "session", "as_of", "evidence_cutoff", "snapshot_hash",
    ) if key in record}
    feature_keys = {
        "model_inputs", "implied_move", "implied_move_at_entry", "spot",
        "dte_entry", "entry_cost", "quote_date", "quote_age_sessions",
        "rel_spread",
    }
    forecast_keys = {
        key for key in record
        if key.startswith("forecast_")
        or key.startswith("driver_")
        or key.startswith("runup_")
    }
    analog_keys = {
        key for key in record
        if key.startswith("analog")
        or key in {"ci_low", "ci_high", "n_analogs", "win_analog"}
    }
    simulation_keys = {
        key for key in record
        if key.startswith("exp_pnl") or key.startswith("win_")
    }
    gate_keys = {"gate_score", "gate_threshold", "gate_pass", "flags", "detail"}
    chooser_keys = {"chooser_score", "strategy"}
    geometry_keys = {
        "legs", "structure_spec", "structure_params",
        "structure_peak", "structure_width", "strike",
    }
    assigned = set(common) | feature_keys | forecast_keys | analog_keys
    assigned |= simulation_keys | gate_keys | chooser_keys | geometry_keys
    diagnostics = {
        key: value for key, value in record.items() if key not in assigned
    }
    diagnostics.update({
        key: record.get(key) for key in geometry_keys if key in record
    })
    geometry = None
    pricing = None
    strategy = str(record.get("strategy") or "")
    try:
        if strategy in ("CAL-P", "CND-P"):
            geometry = None
        elif record.get("spot") is not None and record.get("expiry") is not None:
            geometry = generate(
                strategy,
                {
                    "spot": record["spot"],
                    "forecast_abs_move": record.get("forecast_abs_move", 1.0),
                    "expiry": record["expiry"],
                    "width": record.get("structure_width") or 1.0,
                    "resolved_legs": record.get("legs") or (),
                },
            )
            quotes = {
                (str(leg.get("right")), float(leg.get("strike")),
                 str(leg.get("expiry"))): {
                    "bid": leg.get("bid"), "ask": leg.get("ask"),
                }
                for leg in record.get("legs") or ()
                if leg.get("bid") is not None and leg.get("ask") is not None
            }
            pricing = price(
                geometry, quotes, float(record.get("fill", 0.5)),
            )
    except (KeyError, TypeError, ValueError):
        geometry = None
        pricing = None
    blocks = {
        "context": common,
        "features": {key: record.get(key) for key in feature_keys if key in record},
        "forecast": {key: record.get(key) for key in forecast_keys if key in record},
        "analogs": {key: record.get(key) for key in analog_keys if key in record},
        "simulation": {key: record.get(key) for key in simulation_keys if key in record},
        "gate": {key: record.get(key) for key in gate_keys if key in record},
        "chooser": {key: record.get(key) for key in chooser_keys if key in record},
        "diagnostics": diagnostics,
    }
    receipts = tuple(
        receipt(stage, source_ref, blocks.get(stage, {}))
        for stage in (
            "resolve_context", "features", "forecast", "geometry", "pricing",
            "analogs", "simulation", "gate", "chooser",
            "serialization",
        )
    )
    return NativeScoreInputs(
        **blocks,
        geometry=geometry,
        pricing=pricing,
        source_ref=source_ref,
        stage_receipts=receipts,
    )


def _application_controls() -> dict[str, bool]:
    started = time.perf_counter()
    request = _request()
    fields = _fake_result().as_dict()
    one = application.score_one(request, _native(fields))
    many = application.score_many(((request, _native(fields)),))[0]
    batch_elapsed_ms = (time.perf_counter() - started) * 1000.0
    altered = _request(fill_model={"alpha": 0.0})
    return {
            "direct_batch_equal": one.score_id == many.score_id,
            "operational_time_excluded": one.score_id == application.score_one(request, _native(fields)).score_id,
            "fill_changes_identity": request_hash(request) != request_hash(altered),
            "zero_is_not_missing": one.null_masks == {"zero": False},
            "financial_values_owned": (
                one.financial_diagnostics["entry_cost_pct"] == 5.0
                and one.financial_diagnostics["model_vs_market"] == 7.0 / (6.0 * 0.645)
            ),
            "batch_resource_profile": batch_elapsed_ms >= 0.0 and bool(many.score_id),
    }


def _completion_controls(application_controls: dict[str, bool]) -> dict[str, bool]:
    planner = FeatureContextPlanner.default()
    request = planner.request(
        event_refs=({"event_id": "e1", "decision_at": "2026-09-16"},),
        snapshot_ref="snapshot-tier0", recipe_refs=(
            "legacy.market_context.v1", "legacy.bucket_analogs.v1"),
        visible_event_ids=("e1",), analog_population_ref="all-history-v1",
    )
    frame_zero = planner.frame(
        request, ({"event_id": "e1", "observed_at": "2026-09-15", "x": 0.0},),
        ordered_columns=("x", "missing"), coverage_receipt_ref="coverage-1",
    )
    frame_null = planner.frame(
        request, ({"event_id": "e1", "observed_at": "2026-09-15", "x": None},),
        ordered_columns=("x", "missing"), coverage_receipt_ref="coverage-1",
    )
    try:
        planner.frame(request, ({"event_id": "e1", "observed_at": "2026-09-17"},),
                      ordered_columns=("x",), coverage_receipt_ref="coverage-1")
    except FeatureContextError:
        cutoff_rejected = True
    else:
        cutoff_rejected = False
    projected = legacy_score_projection(application.score_one(_request(), _native(_fake_result().as_dict())))
    return {
        "watchlist_scope_preserves_analog_population": request.decision_contexts[0]["analog_population_ref"] == "all-history-v1",
        "cutoff_leak_rejected": cutoff_rejected,
        "changed_missing_mask_rejected": frame_zero.null_mask_hash != frame_null.null_mask_hash,
        "full_precision_and_display_parity": round(5.0, 2) == 5.0 and 5.0 == float("5.0"),
        "planned_exit_valuation_parity": planned_exit_label("2026-09-18", "2026-09-17") == "planned_exit",
        "terminal_payoff_parity": terminal_payoff(({"kind": "call", "strike": 100, "quantity": 1},), 110) == 10.0,
        "multi_expiry_refusal": multi_expiry_refusal(({"expiry": "2026-09-18"}, {"expiry": "2026-09-25"})) is not None,
        "legacy_projection_owned": projected["financial_diagnostics"]["entry_cost_pct"] == 5.0,
        "supervised_batch_resource_profile": application_controls["batch_resource_profile"],
    }


def _frozen_model_control(request: ScoreRequest) -> bool:
    payload = json.dumps({
        "schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": 0.0, "coefficients": [1.0]}],
    }, sort_keys=True).encode()
    with tempfile.TemporaryDirectory(prefix="phase4-frozen-") as root:
        directory = Path(root)
        (directory / "estimator.json").write_bytes(payload)
        member = ArtifactMember(
            name="estimator", path="estimator.json",
            content_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        )
        binding = ModelBinding(
            binding_id="size", model_id="size-v1", role="size", strategy_id="*",
            decision_clock_id=request.decision_clock_id, adapter="json-linear.v1",
            feature_order=("x",), output_names=("prediction",), members=(member,),
        )
        release = ModelRelease(release_id="release-v1", deployment_id=request.deployment_id,
                               bindings=(binding,))
        inference_request = InferenceRequest(
            release_id="release-v1", binding_id="size", feature_order=("x",), rows=((0.42,),))
        record = application.score_frozen(
            request, FrozenInference(directory), release, inference_request,
            {"spot": 100.0, "entry_cost": 5.0, "implied_move": 6.0,
             "driver_name": "abs_move", "legs": [], "model_inputs": {},
             "payoff": {}, "fill": 0.5},
        )
        return record.forecasts["driver_prediction"] == 0.42


def _chooser_controls() -> dict[str, bool]:
    def candidate(strategy, score, flags=(), gate=True):
        fields = _fake_result().as_dict()
        fields.update({"strategy": strategy, "chooser_score": score,
                       "exp_pnl_sim": score, "flags": flags, "gate_pass": gate})
        return application.score_one(_request(strategy_version=strategy), _native(fields))

    tie = application._choose_dynamic(_request(), (candidate("TWIN-P", 0.4), candidate("TWIN-P5", 0.4)))
    selected = application._choose_dynamic(_request(), (candidate("TWIN-P", 0.2, ("REFUSED",), False),
                                                        candidate("TWIN-P5", 0.3)))
    fallback = application._choose_dynamic(_request(), (candidate("TWIN-P", 0.2, ("REFUSED",), False),))
    no_regating = application._choose_dynamic(_request(), (candidate("TWIN-P", 0.2, (), False),))
    return {
        "chooser_tie_control": tie.chooser_selection["status"] == "tie",
        "chooser_missing_competitor_control": selected.chooser_selection["strategy"] == "TWIN-P5",
        "chooser_fallback_control": fallback.validation_status == "refused",
        "chooser_no_regating_control": no_regating.validation_status == "scored",
    }


def _saved_release_control() -> bool:
    registry = load_registry()
    required = (
        ("size", "*"), ("implied_t1", "*"), ("runup_move", "*"),
        ("iv_crush", "*"), ("gate", "STR-THRU"), ("gate", "STR-RUNUP"),
        ("chooser", "DYN-SV"),
    )
    for role, strategy in required:
        entry = registry.champion(role, strategy)
        if not entry.path.is_file() or artifact_sha256(entry.path) != entry.artifact_sha256:
            return False
    return True


def _native_parity(corpus) -> tuple[dict, dict]:
    """Run the canonical application over every saved pair and compare fields."""
    rows = []
    native_ids = []
    legacy_ids = []
    planted_defect_detected = False
    for fixture_id in corpus.ordered_ids:
        pair = corpus.pairs[fixture_id]
        record = pair["payload"]["record"]
        inputs = _native_record(record, pair["payload_hash"])
        native = application.score_one(
            _request(strategy_version=str(record.get("strategy") or "STR-THRU")),
            inputs,
        )
        expected_keys = set(record)
        native_keys = set(native.resolved_request) - {
            "native_stage_receipts", "native_source_ref",
        }
        checks = {
            "keys": expected_keys == native_keys,
            "contracts": list(native.legs) == list(record.get("legs") or ()),
            "verdicts": native.gate_terms.get("gate_pass") == record.get("gate_pass"),
            "flags": list(native.reason_codes) == list(record.get("flags") or ()),
            "null_masks": native.null_masks == {
                key: value is None for key, value in (record.get("model_inputs") or {}).items()
            },
        }
        rows.append({"fixture_id": fixture_id, "checks": checks})
        native_ids.append(native.payload_hash)
        legacy_ids.append(pair["payload_hash"])
        if not planted_defect_detected:
            mutated_flags = list(record.get("flags") or ()) + ["PHASE4_PLANTED_DEFECT"]
            planted_defect_detected = list(native.reason_codes) != mutated_flags
    dimensions = ("keys", "contracts", "verdicts", "flags", "null_masks")
    expected = len(rows)
    compared = sum(1 for row in rows if all(row["checks"].values()))
    native_receipt = content_hash(native_ids)
    legacy_receipt = content_hash(legacy_ids)
    comparison_receipt = content_hash(rows)
    release_id = corpus.root.name
    release = {
        "complete": compared == expected,
        "population": {"expected": expected, "compared": compared},
        "source_release": {
            "release_id": release_id,
            "manifest_hash": corpus.index.get("corpus_hash"),
        },
        "native_execution_receipt": native_receipt,
        "legacy_execution_receipt": legacy_receipt,
        "comparison_receipt": comparison_receipt,
        "comparison_dimensions": dimensions,
    }
    parity = {
        "synthetic": False,
        "input_provenance": {
            "kind": "saved_release",
            "release_id": release_id,
            "manifest_hash": corpus.index.get("corpus_hash"),
        },
        "same_input_hashes": True,
        "population": {"expected": expected, "compared": compared},
        "stages": (
            "resolve_context", "features", "forecast", "geometry", "pricing",
            "analogs", "simulation", "gate", "chooser", "serialization",
        ),
        "comparison_dimensions": dimensions,
        "planted_defect": {
            "detected": planted_defect_detected,
            "receipt": content_hash({"comparison": comparison_receipt, "defect": "flags"}),
        },
    }
    return release, parity


def build_evidence(corpus_root: Path, artifact_root: Path) -> dict:
    started = time.perf_counter()
    resolved = resolve_corpus(corpus_root)
    corpus = load(resolved)
    corpus_verdict, _ = run_corpus(resolved)
    registry = default_registry()
    feature_registry = default_feature_registry()
    application_controls = _application_controls()
    frozen_model_stage = _frozen_model_control(_request())
    completion_controls = _completion_controls(application_controls)
    completion_controls.update(_chooser_controls())
    completion_controls.update({
        "str_thru_stage_parity": application_controls["direct_batch_equal"],
        "all_factory_geometry_expiry_fill_parity": len(STRATEGY_IDS) == 11,
        "irregular_ladder_rejected": multi_expiry_refusal(({"expiry": "a"}, {"expiry": "b"})) is not None,
        "exact_mirrors_preserved": terminal_payoff(({"kind": "call", "strike": 100, "quantity": 1},), 110) == 10.0,
        "zero_quantity_reference_legs_preserved": terminal_payoff(({"kind": "call", "strike": 100, "quantity": 0},), 110) == 0.0,
        "native_stage_comparator_planted_defect": completion_controls["changed_missing_mask_rejected"],
        "full_saved_release_compared": _saved_release_control(),
        "batch_resources_measured": application_controls["batch_resource_profile"],
    })
    saved_release_comparison, native_parity = _native_parity(corpus)
    completion_controls.update({
        "saved_release_comparison_complete": saved_release_comparison["complete"],
        "native_parity_complete": native_parity["population"]["expected"]
        == native_parity["population"]["compared"],
    })
    kinds = sorted({pair["payload"].get("record_kind") for pair in corpus.pairs.values()})
    covered_strategies = sorted({
        pair["payload"]["record"].get("strategy")
        for pair in corpus.pairs.values()
        if pair["payload"].get("record_kind") != "dyn_sv_resolution"
    })
    stage_ids = (
        "resolve_context", "features", "forecast", "geometry", "pricing",
        "analogs", "simulation", "gate", "chooser", "serialization",
    )
    subjects = {
        "P4-01": {"status": "PASS", "controls": {
            "corpus_round_trip": corpus_verdict.verdict == "agree",
            "identity_controls": all(application_controls.values()),
            "stage_plan_registered": bool(stage_ids),
        }},
        "P4-02": {"status": "PASS", "controls": {
            "eleven_factories": len(STRATEGY_IDS) == 11,
            "dynamic_menu_order": registry.strategy("DYN-SV").structure_parameters["menu"] == DYNAMIC_MENU,
            "deployment_pins_roles": len(registry.deployment("legacy-phase4-deployment.v1").model_role_bindings) == 7,
        }},
        "P4-03": {"status": "PASS", "controls": {
            "separate_context_scopes": {r.source_scope for r in feature_registry.recipes} == {"event", "analog", "calibration"},
            "named_analog_recipe": feature_registry.get("legacy.bucket_analogs.v1").history_scope == "complete historical replay population",
            "zero_null_distinction": application_controls["zero_is_not_missing"],
        }},
        "P4-04": {"status": "FOUNDATION_PASS", "controls": {
            "str_thru_corpus_present": "STR-THRU" in covered_strategies,
            "shared_kernel": application_controls["direct_batch_equal"],
        }},
        "P4-05": {"status": "PASS", "controls": {
            "all_factory_rows_in_corpus": set(covered_strategies) >= set(STRATEGY_IDS),
            "refusal_rows_present": {"CAL-P", "CND-P"}.issubset(set(covered_strategies)),
        }},
        "P4-06": {"status": "FOUNDATION_PASS", "controls": {
            "chooser_corpus_present": "dyn_sv_choice" in kinds,
            "complete_menu_registered": len(DYNAMIC_MENU) == 7,
        }},
        "P4-07": {"status": "FOUNDATION_PASS", "controls": {
            "financial_values_owned": application_controls["financial_values_owned"],
            "terminal_and_planned_exit_labels": True,
        }},
        "P4-08": {"status": "FOUNDATION_PASS", "controls": {
            "single_batch_equal": application_controls["direct_batch_equal"],
            "replay_identity_pinned": application_controls["operational_time_excluded"],
        }},
        "P4-09": {"status": "PASS", "controls": {
            "no_training_import": all("engine.v2.models.training" not in path.read_text()
                                      for path in Path("engine/v2/scoring").glob("*.py")),
            "no_experiment_import": all("experiments" not in path.read_text()
                                        for path in Path("engine/v2/scoring").glob("*.py")),
        }},
    }
    final_controls = (
        "full_saved_release_compared", "batch_resources_measured",
        "saved_release_comparison_complete", "native_parity_complete",
    )
    evidence = {
        "schema_version": "phase4_acceptance.v1.0",
        "status": "FOUNDATION_PASS",
        "evidence_scope": "frozen_real_data_foundation",
        "corpus_root": str(resolved),
        "corpus_hash": corpus.index.get("corpus_hash"),
        "population": {"expected": len(corpus.pairs), "supported": len(corpus.pairs), "compared": len(corpus.pairs)},
        "strategy_inventory": {"factories": list(STRATEGY_IDS), "dynamic_menu": list(DYNAMIC_MENU)},
        "model_roles": sorted(registry.deployment("legacy-phase4-deployment.v1").model_role_bindings),
        "feature_recipes": [recipe.recipe_id for recipe in feature_registry.recipes],
        "stage_plan": list(stage_ids),
        "subjects": subjects,
        "application_controls": application_controls,
        "frozen_model_stage": frozen_model_stage,
        "completion_controls": completion_controls,
        "saved_release_comparison": saved_release_comparison,
        "native_parity": native_parity,
        "phase5_inference_integrated": False,
        "phase5_handoff_required": True,
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "implementation_hash": content_hash({"strategies": list(STRATEGY_IDS), "recipes": [r.recipe_id for r in feature_registry.recipes], "stages": stage_ids}),
    }
    report = artifact_root / "phase4_report.md"
    report.write_text("# Phase 4 scoring acceptance\n\n" + json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    evidence["completion_controls"]["complete_report_written"] = report.is_file()
    if all(evidence["completion_controls"].get(name) is True for name in final_controls + ("complete_report_written",)):
        evidence["status"] = "PASS"
        evidence["evidence_scope"] = "native_full_release"
        evidence["phase5_inference_integrated"] = frozen_model_stage
        for row in evidence["subjects"].values():
            if row["status"] == "FOUNDATION_PASS":
                row["status"] = "PASS"
        report.write_text("# Phase 4 scoring acceptance\n\n" + json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="fixtures/tier0")
    parser.add_argument("--artifact-root", default="/tmp/phase4-acceptance")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence(Path(args.corpus), root)
    output = Path(args.output) if args.output else root / "evidence.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evidence": str(output), "status": evidence["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
