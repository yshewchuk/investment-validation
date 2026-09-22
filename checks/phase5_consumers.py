"""P5-6 score-consumer probes over a staged release.

Each probe drives one real v2 score consumer with members resolved from the
staged release (never from ``data/``) and reports, per member object,
whether the consumer resolved it and whether it refused ``MODEL_NOT_READY``
once the member was taken away. Probe contexts are fixed and synthetic: they
prove resolution and refusal, not parity (that is the Phase 4 replay).

A probe row is ``{"consumer", "member_id", "resolved", "refused_when_missing",
"detail"}``, or ``{"consumer", "member_id", "blocked": True}`` when a member
the consumer needs is not staged and verified. ``CONSUMERS`` maps consumer id
to probe; ``None`` means no v2 consumer or probe exists yet (PENDING).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

from checks.phase5_release import deployment_root
from engine.v2.models.contracts import ModelRelease

__all__ = ["CONSUMERS", "ReleaseContext"]


@dataclasses.dataclass
class ReleaseContext:
    """What the consumer probes read: the resolved release, never data/."""

    release_root: Path
    model_release: ModelRelease
    states: dict[str, dict]



def _probe_frozen_executor(ctx: ReleaseContext) -> list[dict]:
    """``FrozenStageExecutor`` over every staged model binding."""
    from engine.v2.models.loader import FrozenInference
    from engine.v2.scoring.frozen_executor import FrozenStageExecutor, FrozenStageRefusal

    rows = []
    root = deployment_root(ctx.release_root)
    for binding in ctx.model_release.bindings:
        features = {name: 0.0 for name in binding.feature_order}
        resolved, detail = False, ""
        try:
            result = FrozenStageExecutor(inference=FrozenInference(root),
                                         release=ctx.model_release,
                                         binding_id=binding.binding_id).execute(features)
            staged = tuple(member.content_hash for member in binding.members)
            resolved = tuple(result.artifact_hashes) == staged
            detail = "" if resolved else "artifact hashes differ from the staged binding"
        except FrozenStageRefusal as exc:
            detail = exc.code + ":" + ",".join(exc.reason_codes)
        broken = dataclasses.replace(binding, members=tuple(
            dataclasses.replace(member, path=f"objects/absent-{member.name}")
            for member in binding.members))
        stripped = dataclasses.replace(ctx.model_release, bindings=tuple(
            broken if item.binding_id == binding.binding_id else item
            for item in ctx.model_release.bindings))
        refused = False
        try:
            FrozenStageExecutor(inference=FrozenInference(root), release=stripped,
                                binding_id=binding.binding_id).execute(features)
        except FrozenStageRefusal as exc:
            refused = exc.code == "MODEL_NOT_READY"
        rows.append({"consumer": "frozen_stage_executor",
                     "member_id": f"model:{binding.role}:{binding.strategy_id}",
                     "resolved": resolved, "refused_when_missing": refused, "detail": detail})
    return rows


def _probe_request(strategy: str, alpha: float):
    from engine.v2.contracts import ScoreRequest

    return ScoreRequest(
        event_id="p5-6-probe", calendar_revision="probe", strategy_version=strategy,
        deployment_id="p5-6-probe", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="probe", mode="replay",
        fill_model={"alpha": alpha},
    )


_ZERO_POOL = ({"prediction": 1.0, "residual": 0.0},)
_PROBE_QUOTES = {("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0},
                 ("P", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0}}


def _probe_bundle(strategy: str, artifact, cutoff, **overrides):
    """A source-only model-stage probe bundle; ``overrides`` replace fields."""
    from engine.v2.scoring.source_inputs import SourceBundle

    context = {"ticker": "P5PROBE", "event_date": "2026-09-16", "entry_date": "2026-09-16",
               "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0}
    recipes = {"driver_prediction": {"intercept": 1.0, "coefficients": {}}}
    refs = {"driver_prediction": "sha256:probe-driver"}
    fields = {}
    if strategy == "STR-RUNUP":
        context.update({"strike": 100.0, "days_before_print": 7.0})
        recipes["runup_move_prediction"] = {"intercept": 0.0, "coefficients": {}}
        refs["runup_move_prediction"] = "sha256:probe-move"
        fields["runup_move_residual_rows"] = _ZERO_POOL
    recipe = {"seed": 1, "draw_count": 16}
    if cutoff is not None:
        recipe["before"] = cutoff
    fields.update(
        source_ref="p5-6-probe", context=context, raw_quotes=_PROBE_QUOTES,
        feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "probe"}},
        forecast_recipes=recipes, model_artifact_refs=refs,
        residual_recipe={"terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
                         "capital_at_risk": 1.0},
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        strategy=strategy, payoff_artifact_recipe=recipe, payoff_artifact=artifact,
        model_residual_rows=_ZERO_POOL,
    )
    fields.update(overrides)
    return SourceBundle(**fields)


def _score(strategy: str, alpha: float, bundle):
    from engine.v2.scoring import application
    from engine.v2.scoring.source_inputs import build_native_score_inputs

    return application.score_one(_probe_request(strategy, alpha),
                                  build_native_score_inputs(bundle))


def _score_flags(strategy: str, artifact, alpha: float, cutoff,
                 **overrides) -> tuple[str, ...]:
    bundle = _probe_bundle(strategy, artifact, cutoff, **overrides)
    return tuple(_score(strategy, alpha, bundle).reason_codes)


def _row(consumer: str, member_id: str, flags, missing_flags) -> dict:
    resolved = not ({"MODEL_NOT_READY", "NO_PAYOFF_MAP"} & set(flags))
    return {"consumer": consumer, "member_id": member_id, "resolved": resolved,
            "refused_when_missing": "MODEL_NOT_READY" in missing_flags,
            "detail": "" if resolved else ",".join(flags)}


def _chooser_row(consumer: str, member_id: str, flags, missing_flags) -> dict:
    """Like ``_row``, but for TWIN-P chooser probes only: NO_PAYOFF_MAP is
    excluded from the unresolved set. TWIN-P is not in PAYOFF_DRIVER
    (engine/payoff.py), so legacy's model stage flags NO_PAYOFF_MAP on
    EVERY TWIN-P row unconditionally (engine/score.py:2799) -- a permanent,
    benign fact about the strategy, unrelated to whether the chooser stage
    this probe exercises resolved its declared state. Only MODEL_NOT_READY
    (a declared-but-absent chooser state) is evidence of an unresolved
    consumer here."""
    resolved = "MODEL_NOT_READY" not in set(flags)
    return {"consumer": consumer, "member_id": member_id, "resolved": resolved,
            "refused_when_missing": "MODEL_NOT_READY" in missing_flags,
            "detail": "" if resolved else ",".join(flags)}


def _payoff_probe(member_id: str, strategy: str, consumer: str):
    def probe(ctx: ReleaseContext) -> list[dict]:
        state = ctx.states.get(member_id)
        if state is None:
            return [{"consumer": consumer, "member_id": member_id, "blocked": True}]
        rows = []
        for artifact in state["artifacts"]:
            flags = _score_flags(strategy, artifact, artifact.alpha, artifact.cutoff)
            resolved = not ({"MODEL_NOT_READY", "NO_PAYOFF_MAP"} & set(flags))
            missing = _score_flags(strategy, None, artifact.alpha, artifact.cutoff)
            rows.append({"consumer": consumer, "member_id": member_id, "resolved": resolved,
                         "refused_when_missing": "MODEL_NOT_READY" in missing,
                         "detail": "" if resolved else ",".join(flags)})
        return rows
    return probe


def _probe_recalibration(ctx: ReleaseContext) -> list[dict]:
    """The STR-THRU model stage applying a frozen recalibration map.

    Legacy applies the map on the payoff path with the same
    ``(strategy, alpha, cutoff)`` key, so each staged map is scored together
    with the staged payoff line of that key. A map with no such line is a
    release defect (the stage could never reach it), reported as unresolved.
    """
    consumer, member_id = "model_stage.recalibration", "recalibration_map:STR-THRU"
    state = ctx.states.get(member_id)
    lines = ctx.states.get("payoff_line:STR-THRU")
    if state is None or lines is None:
        return [{"consumer": consumer, "member_id": member_id, "blocked": True}]
    by_key = {line.key: line for line in lines["artifacts"]}
    rows = []
    for recal in state["artifacts"]:
        line = by_key.get(recal.key)
        if line is None:
            rows.append({"consumer": consumer, "member_id": member_id, "resolved": False,
                         "refused_when_missing": True,
                         "detail": "no staged payoff_line:STR-THRU with the same key"})
            continue
        flags = _score_flags("STR-THRU", line, line.alpha, line.cutoff,
                             recalibration_artifact=recal)
        missing = _score_flags("STR-THRU", line, line.alpha, line.cutoff,
                               recalibration_declared=True)
        resolved = not ({"MODEL_NOT_READY", "NO_PAYOFF_MAP"} & set(flags))
        rows.append({"consumer": consumer, "member_id": member_id, "resolved": resolved,
                     "refused_when_missing": "MODEL_NOT_READY" in missing,
                     "detail": "" if resolved else ",".join(flags)})
    return rows


def _blocked(consumer: str, member_id: str) -> list[dict]:
    return [{"consumer": consumer, "member_id": member_id, "blocked": True}]


def _first(ctx: ReleaseContext, member_id: str):
    state = ctx.states.get(member_id)
    return state["artifacts"][0] if state and state["artifacts"] else None


def _driver_key(pool) -> dict:
    return {"role": pool.role, "model_id": pool.model_id, "fold": pool.fold,
            "content_hash": pool.content_hash}


def _driver_case(ctx: ReleaseContext, role: str):
    """``(strategy, slot, payoff artifact, partner slots)`` for one driver role.

    Legacy's model stage reads the ``size`` pool for STR-THRU (driver
    ``abs_move``) and the ``implied_t1`` plus ``runup_move`` pools for
    STR-RUNUP (engine/score.py ``_score_model``/``_score_runup_model``).
    """
    if role == "size":
        return "STR-THRU", "driver", _first(ctx, "payoff_line:STR-THRU"), {}
    partner_role = "runup_move" if role == "implied_t1" else "implied_t1"
    partner = _first(ctx, f"driver_residual_pool:{partner_role}")
    slot = "driver" if role == "implied_t1" else "runup_move"
    partner_slot = "runup_move" if slot == "driver" else "driver"
    partners = {} if partner is None else {partner_slot: partner}
    return "STR-RUNUP", slot, _first(ctx, "payoff_surface:STR-RUNUP"), partners


def _probe_driver_pools(ctx: ReleaseContext) -> list[dict]:
    """The model stage reading frozen driver residual pools by causal key.

    Each staged pool is scored with the staged payoff artifact of its
    strategy (inline fitting is forbidden under the guards) and, for
    STR-RUNUP, the other driver's first staged pool.
    """
    consumer, rows = "model_stage.driver_residual_pool", []
    for role in ("size", "implied_t1", "runup_move"):
        member_id = f"driver_residual_pool:{role}"
        state = ctx.states.get(member_id)
        strategy, slot, payoff, partners = _driver_case(ctx, role)
        if state is None or payoff is None or (strategy == "STR-RUNUP" and not partners):
            rows += _blocked(consumer, member_id)
            continue
        for pool in state["artifacts"]:
            slots = {slot: pool, **partners}
            common = dict(model_residual_rows=(), runup_move_residual_rows=(),
                          model_residual_artifact_recipe={
                              name: _driver_key(item) for name, item in slots.items()})
            flags = _score_flags(strategy, payoff, payoff.alpha, payoff.cutoff,
                                 model_residual_artifacts=slots, **common)
            missing = _score_flags(strategy, payoff, payoff.alpha, payoff.cutoff,
                                   model_residual_artifacts={**slots, slot: None}, **common)
            rows.append(_row(consumer, member_id, flags, missing))
    return rows


def _paired_bundle(pool, *, artifact):
    """The planned-exit STR-THRU shape of tests/test_v2_scoring_frozen_residuals."""
    from engine.v2.scoring.source_inputs import SourceBundle

    key = {"move_model_id": pool.move_model_id, "crush_model_id": pool.crush_model_id,
           "cutoff": pool.cutoff, "content_hash": pool.content_hash}
    return SourceBundle(
        source_ref="p5-6-paired-probe",
        context={"ticker": "P5PROBE", "event_date": "2026-09-16", "entry_date": "2026-09-16",
                 "exit_date": "2026-09-09", "expiry": "2026-09-18", "spot": 100.0,
                 "pre_iv30": 40.0},
        raw_quotes={("C", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
                    ("P", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05}},
        feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "probe"}},
        forecast_recipes={"driver_prediction": {"intercept": 7.0, "coefficients": {}},
                          "forecast_abs_move": {"intercept": 7.0, "coefficients": {}},
                          "pred_iv_crush": {"intercept": -20.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:p1", "forecast_abs_move": "sha256:p2",
                             "pred_iv_crush": "sha256:p3"},
        residual_recipe={}, analog_recipe={},
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {"exp_pnl_sim": 1.0}},
                     "threshold": 0.0},
        paired_residual_recipe=key, paired_residual_artifact=artifact,
    )


def _probe_paired_pool(ctx: ReleaseContext) -> list[dict]:
    """The planned-exit simulation reading the frozen paired residual pool."""
    consumer, member_id = "simulation.paired_residual_pool", "paired_residual_pool"
    state = ctx.states.get(member_id)
    if state is None:
        return _blocked(consumer, member_id)
    rows = []
    for pool in state["artifacts"]:
        record = _score("STR-THRU", 0.5, _paired_bundle(pool, artifact=pool))
        missing = _score("STR-THRU", 0.5, _paired_bundle(pool, artifact=None))
        resolved = (record.resolved_request.get("exp_pnl_sim") is not None
                    and "MODEL_NOT_READY" not in record.reason_codes)
        rows.append({"consumer": consumer, "member_id": member_id, "resolved": resolved,
                     "refused_when_missing": "MODEL_NOT_READY" in missing.reason_codes,
                     "detail": "" if resolved else ",".join(record.reason_codes)})
    return rows


_ANALOG_QUERY = {"mcap_bucket": "1-10B", "dte_band": "1-3", "moneyness_band": "ATM",
                 "implied_ratio": 1.0}


def _analog_recipe(pool, **overrides) -> dict:
    recipe = {"cutoff": pool.cutoff, "content_hash": pool.content_hash, "min_analogs": 30,
              "bootstrap_draws": 0, "ci_quantiles": (0.05, 0.95),
              "seed_snapshot": "p5-6-probe", "request_key": "p5-6-probe"}
    recipe.update(overrides)
    return recipe


def _analog_bundle(pool, *, artifact):
    """A source-only STR-THRU bundle whose only declared population is the
    frozen board analog pool (the shape of tests/test_v2_models_board_analog)."""
    from engine.v2.scoring.source_inputs import SourceBundle

    return SourceBundle(
        source_ref="p5-6-analog-probe",
        context={"ticker": "P5PROBE", "event_date": "2026-09-16", "entry_date": "2026-09-16",
                 "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0},
        raw_quotes=_PROBE_QUOTES, feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "probe"}},
        forecast_recipes={"driver_prediction": {"intercept": 1.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:probe-driver"},
        residual_recipe={}, analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        analog_query=dict(_ANALOG_QUERY), analog_artifact_recipe=_analog_recipe(pool),
        analog_artifact=artifact,
    )


def _analog_direct(pool, artifact) -> tuple[bool, bool]:
    """The analog stage's own reader, for a strategy the probe bundle cannot
    score end to end: ``(resolved, refused_when_missing)``."""
    from engine.v2.scoring.native_analog import evaluate_frozen_analogs

    def run(item):
        return evaluate_frozen_analogs(artifact=item, recipe=_analog_recipe(pool),
                                       query_features=_ANALOG_QUERY, strategy=pool.strategy,
                                       alpha=pool.alpha)

    result, flag = run(artifact)
    _, missing = run(None)
    return result is not None and flag is None, missing == "MODEL_NOT_READY"


def _probe_board_analog(ctx: ReleaseContext) -> list[dict]:
    """The native analog stage reading the frozen board analog matcher.

    Every staged ``(strategy, alpha, cutoff)`` pool is read under its full
    causal key and release pin: STR-THRU end to end through ``score_one``
    (resolved = the analog stage ran with no MODEL_NOT_READY), other
    strategies through the stage's own reader. Removing the artifact must
    give MODEL_NOT_READY.
    """
    consumer, member_id = "analogs.board_analog_matcher", "board_analog_matcher"
    state = ctx.states.get(member_id)
    if state is None:
        return _blocked(consumer, member_id)
    rows = []
    for pool in state["artifacts"]:
        if pool.strategy == "STR-THRU":
            record = _score("STR-THRU", pool.alpha, _analog_bundle(pool, artifact=pool))
            missing = _score("STR-THRU", pool.alpha, _analog_bundle(pool, artifact=None))
            refused = "MODEL_NOT_READY" in missing.reason_codes
            resolved = ("MODEL_NOT_READY" not in record.reason_codes
                        and "n_analogs" in record.resolved_request)
            detail = "" if resolved else ",".join(record.reason_codes)
        else:
            resolved, refused = _analog_direct(pool, pool)
            detail = "" if resolved else "frozen analog reader did not resolve"
        rows.append({"consumer": consumer, "member_id": member_id, "resolved": resolved,
                     "refused_when_missing": refused, "detail": detail})
    return rows


_CHOOSER_QUOTES = {("P", float(strike), "2026-09-18"): {"bid": 1.0, "ask": 1.2}
                   for strike in range(80, 121, 2)}


def _chooser_bundle(ctx: ReleaseContext, binding_id: str, recipe: dict, **state):
    """A TWIN-P menu candidate whose chooser declares one frozen state."""
    from engine.v2.models.loader import FrozenInference
    from engine.v2.scoring.source_inputs import SourceBundle

    return SourceBundle(
        source_ref="p5-6-chooser-probe", strategy="TWIN-P",
        context={"ticker": "P5PROBE", "event_date": "2026-09-16",
                 "entry_date": "2026-09-16", "exit_date": "2026-09-09",
                 "expiry": "2026-09-18", "spot": 100.0},
        raw_quotes=_CHOOSER_QUOTES, feature_vector={}, feature_missing_mask={},
        model_identity={"size": {"model_id": "probe"}},
        forecast_recipes={"forecast_abs_move": {"intercept": 6.0, "coefficients": {}}},
        model_artifact_refs={"forecast_abs_move": "sha256:probe-size"},
        residual_recipe={}, analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        chooser_recipe={"binding_id": binding_id, **recipe},
        frozen_inference=FrozenInference(deployment_root(ctx.release_root)),
        model_release=ctx.model_release, **state,
    )


def _chooser_probe(member_id: str, consumer: str, field: str, key_of):
    """The chooser stage reading one frozen state by its declared key; the
    staged chooser binding scores (or declines on the probe's absent
    features), and a declared-but-absent state is MODEL_NOT_READY."""
    def probe(ctx: ReleaseContext) -> list[dict]:
        state = ctx.states.get(member_id)
        chooser = [b for b in ctx.model_release.bindings if b.role == "chooser"]
        if state is None or len(chooser) != 1:
            return _blocked(consumer, member_id)
        rows = []
        for artifact in state["artifacts"]:
            recipe = {field.removeprefix("chooser_"): key_of(artifact)}
            flags = _score("TWIN-P", 0.5, _chooser_bundle(
                ctx, chooser[0].binding_id, recipe, **{field: artifact})).reason_codes
            missing = _score("TWIN-P", 0.5, _chooser_bundle(
                ctx, chooser[0].binding_id, recipe)).reason_codes
            rows.append(_chooser_row(consumer, member_id, flags, missing))
        return rows
    return probe


def _entry_rule_inputs(event_date: str):
    """Native inputs of a priced TWIN-P row with a simulated expectation,
    whose gate is then declared as its entry rule."""
    from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs

    return build_native_score_inputs(SourceBundle(
        source_ref="p5-6-entry-rule-probe", strategy="TWIN-P",
        context={"ticker": "P5PROBE", "event_date": event_date, "entry_date": "2026-09-16",
                 "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0},
        raw_quotes=_CHOOSER_QUOTES, feature_vector={}, feature_missing_mask={},
        model_identity={"size": {"model_id": "probe"}},
        forecast_recipes={"forecast_abs_move": {"intercept": 6.0, "coefficients": {}}},
        model_artifact_refs={"forecast_abs_move": "sha256:probe-size"},
        residual_recipe={"terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
                         "capital_at_risk": 1.0},
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
    ))


def _probe_trailing_cutoff(ctx: ReleaseContext) -> list[dict]:
    """The native entry-rule gate reading each staged trailing cutoff.

    A TWIN-P row dated in the artifact's month is scored end to end with the
    gate declared as its entry rule over the staged artifact, pinned to its
    hash. Resolved: no MODEL_NOT_READY, and a verdict exactly when the
    artifact holds a bar (market cap, spread and expectation are all known,
    so only a bar-less month leaves the rule undetermined). Declaring the
    same key with the artifact taken away must give MODEL_NOT_READY.
    """
    from engine.v2.scoring.native_entry_rule import entry_rule_block

    consumer, member_id = "gate.trailing_cutoff", "trailing_pnl_cutoff"
    state = ctx.states.get(member_id)
    if state is None:
        return _blocked(consumer, member_id)
    rows = []
    for cutoff in state["artifacts"]:
        inputs = _entry_rule_inputs(cutoff.month[:8] + "15")
        record = _score_inputs(inputs, entry_rule_block(
            "TWIN-P", mcap_usd=5e10, cutoff=cutoff))
        missing = _score_inputs(inputs, entry_rule_block(
            "TWIN-P", mcap_usd=5e10, cutoff=None, month=cutoff.month))
        verdict = record.gate_terms.get("gate_pass")
        resolved = ("MODEL_NOT_READY" not in record.reason_codes
                    and (verdict is not None) == (cutoff.cutoff is not None))
        rows.append({"consumer": consumer, "member_id": member_id, "resolved": resolved,
                     "refused_when_missing": "MODEL_NOT_READY" in missing.reason_codes,
                     "detail": "" if resolved else ",".join(record.reason_codes)})
    return rows


def _score_inputs(inputs, gate: dict):
    from dataclasses import replace

    from engine.v2.scoring import application

    return application.score_one(_probe_request("TWIN-P", 0.5), replace(inputs, gate=gate))


_TIER4_OUTPUTS = {"size": "pred_abs_move", "implied_t1": "pred_implied_t1",
                  "iv_crush": "pred_iv_crush_30", "runup_move": "pred_runup_abs_move_d14"}


def _fold_features(root: Path, obj: dict) -> tuple[str, ...]:
    """The feature order a staged fold was trained on (its own ``features``)."""
    import io

    import joblib

    stored = joblib.load(io.BytesIO((root / obj["path"]).read_bytes()))
    return tuple(str(name) for name in stored.get("features", ()))


def _fold_release(role: str, obj: dict, features: tuple[str, ...], path: str):
    from engine.v2.models.contracts import ArtifactMember, ModelBinding

    binding = ModelBinding(
        binding_id=f"p5-probe-{role}-{obj['name']}", model_id=f"tier4:{role}", role=role,
        strategy_id="*", decision_clock_id="legacy.entry_close.v1",
        adapter="tier4-serving-fold.v1", feature_order=features,
        output_names=(_TIER4_OUTPUTS[role],),
        members=(ArtifactMember(name="estimator", path=path,
                                content_hash=obj["content_hash"]),))
    return ModelRelease(release_id="p5-probe", deployment_id="p5-probe",
                        bindings=(binding,)), binding.binding_id


def _run_fold(root: Path, role: str, obj: dict, features, path: str):
    from engine.v2.models.loader import FrozenInference
    from engine.v2.scoring.frozen_executor import FrozenStageExecutor

    release, binding_id = _fold_release(role, obj, features, path)
    return FrozenStageExecutor(inference=FrozenInference(root), release=release,
                               binding_id=binding_id).execute(
        {name: 0.0 for name in features})


def _fold_row(root: Path, member_id: str, obj: dict) -> dict:
    from engine.v2.scoring.frozen_executor import FrozenStageRefusal

    role = member_id.split(":", 1)[1]
    resolved, refused, detail = False, False, ""
    try:
        features = _fold_features(root, obj)
        result = _run_fold(root, role, obj, features, obj["path"])
        resolved = tuple(result.artifact_hashes) == (obj["content_hash"],)
        detail = "" if resolved else "artifact hashes differ from the staged fold"
        _run_fold(root, role, obj, features, f"objects/absent-{obj['name']}")
    except FrozenStageRefusal as exc:
        refused = resolved and exc.code == "MODEL_NOT_READY"
        detail = detail if resolved else exc.code + ":" + ",".join(exc.reason_codes)
    except Exception as exc:  # noqa: BLE001 -- an undecodable fold is unresolved
        detail = type(exc).__name__
    return {"consumer": "features.tier4_serving_folds", "member_id": member_id,
            "resolved": resolved, "refused_when_missing": refused, "detail": detail}


def _probe_tier4_folds(ctx: ReleaseContext) -> list[dict]:
    """The Tier-4 serving-fold adapter over every staged fold, through
    ``FrozenStageExecutor``: each fold serves its own feature order from the
    staged store, and the same binding pointed at an absent object refuses
    ``MODEL_NOT_READY``. Folds are raw joblib members (no typed loader), so
    the probe reads the staged manifest rows, never ``data/models/tier4``."""
    root = deployment_root(ctx.release_root)
    rows = []
    for member_id in sorted(m for m in ctx.states if m.startswith("tier4_folds:")):
        for obj in sorted(ctx.states[member_id]["row"]["objects"], key=lambda o: o["name"]):
            rows.append(_fold_row(root, member_id, obj))
    return rows or _blocked("features.tier4_serving_folds", "tier4_folds")


#: consumer id -> probe. ``None`` means no probe exists yet: PENDING.
CONSUMERS: dict[str, Callable[[ReleaseContext], list[dict]] | None] = {
    "frozen_stage_executor": _probe_frozen_executor,
    "model_stage.payoff_line": _payoff_probe(
        "payoff_line:STR-THRU", "STR-THRU", "model_stage.payoff_line"),
    "model_stage.payoff_surface": _payoff_probe(
        "payoff_surface:STR-RUNUP", "STR-RUNUP", "model_stage.payoff_surface"),
    "model_stage.driver_residual_pool": _probe_driver_pools,
    "simulation.paired_residual_pool": _probe_paired_pool,
    "model_stage.recalibration": _probe_recalibration,
    "chooser.admissible_table": _chooser_probe(
        "admissible_table:dyn_sv", "chooser.admissible_table", "chooser_admissible_table",
        lambda table: {"table_id": table.table_id, "version": table.version,
                       "content_hash": table.content_hash}),
    "gate.trailing_cutoff": _probe_trailing_cutoff,
    "chooser.analog_pool": _chooser_probe(
        "chooser_analog_pool", "chooser.analog_pool", "chooser_analog_pool",
        lambda pool: {"pool_id": pool.pool_id, "cutoff": pool.cutoff,
                      "content_hash": pool.content_hash}),
    "analogs.board_analog_matcher": _probe_board_analog,
    "features.tier4_serving_folds": _probe_tier4_folds,
}
