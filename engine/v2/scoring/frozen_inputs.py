"""Production frozen inference input builder (Phase 6, P6-2).

Turns one record's captured feature vectors into the ``InferenceRequest``
sequence a ``FrozenInference`` runs -- the pure feature-vector-to-request half
of what ``checks/phase4_frozen_bridge.py`` used to own privately. Replay (the
Phase 4 bridge, now a delegating caller) and future native workers (e.g. the
nightly frozen batch path over ``frozen_batch.score_frozen_batch``) share this
one implementation, so capture-side and replay-side rows can never drift
apart again; the bridge and ``tools/capture_tier0_corpus.py`` keep importing
``binding_feature_row`` under its established name.

The answer-free validation gate moved here too as the Phase 6 worker
prerequisite: a native worker must refuse a record whose captured blocks
smuggle stage outputs back in as inputs exactly as replay does, so
``validate_answer_free`` (with the ``_ANSWER_FIELDS`` map the bridge used to
own privately) is production policy and the bridge's ``_answer_free`` is now a
delegating compatibility wrapper over it.

Verification of the release sidecar, resource hashes and receipts stays on the
bridge side: this module reads no files and trusts its caller to hand it a
verified release and a trace-derived feature mapping. Refusals here raise
``FrozenInputsError`` (a ``ValueError``); the bridge converts them to its
``FrozenBridgeError`` with an identical message.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping, Sequence

from engine.v2.foundation import untag_nonfinite
from engine.v2.models.contracts import InferenceRequest, ModelBinding

from .native_gate_features import GATE_ANALOG_COLUMNS, GATE_FORECAST_COLUMNS
from .stages import NativeScoreInputs

__all__ = [
    "FrozenInputsError",
    "binding_feature_row",
    "build_inference_requests",
    "validate_answer_free",
]

#: Roles whose feature vector is their own, not the forecast-family merge.
_ROLE_PRIVATE_VECTORS = frozenset({"gate", "chooser"})

#: ``NativeScoreInputs`` blocks that may carry a calculated answer, and the
#: key names in each that identify one: an output of the system under test
#: (a model prediction, a simulation or analog summary, a gate or chooser
#: decision, a diagnostic, a selected contract set or entry cost) must never
#: ride in as this run's input, or the frozen comparison is circular. The
#: map's insertion order is the block scan order ``validate_answer_free``
#: refuses in.
_ANSWER_FIELDS = {
    "context": frozenset({"legs", "selected_contracts", "entry_cost", "gate_pass"}),
    "features": frozenset({
        "driver_prediction", "forecast_abs_move", "runup_move_prediction",
        "exp_pnl_sim", "win_sim", "gate_score", "gate_pass",
    }),
    "forecast": frozenset({
        "frozen_outputs", "driver_prediction", "forecast_abs_move",
        "runup_move_prediction", "pred_iv_crush", "pred_iv_crush_30",
        "model_fair_pct",
    }),
    "analogs": frozenset({"exp_pnl_analog", "win_analog", "ci_low", "ci_high"}),
    "simulation": frozenset({"exp_pnl_sim", "win_sim", "sim_p10", "sim_p90"}),
    "gate": frozenset({"frozen_score", "gate_score", "gate_pass", "gate_decision"}),
    "chooser": frozenset({"chosen_strategy", "chooser_selection"}),
    "diagnostics": frozenset({
        "financial_diagnostics", "fair_premium_pct", "premium_vs_fair",
        "cost_over_width",
    }),
}


class FrozenInputsError(ValueError):
    """The captured feature vectors cannot construct this binding's row."""


def _gate_deferral_applies(binding: ModelBinding, missing: list[str]) -> bool:
    """True when ``missing`` is exactly a legal gate derived-column omission.

    Classifies EVERY absent name before any present cell is touched: an
    unknown missing base feature is a hard error no matter what the captured
    cells happen to hold, so it is rejected here -- not shadowed by an early
    return the moment a present cell reads non-finite. The only legal
    omission is a GATE-role row whose absent names are ALL the derived
    forecast/analog columns (``native_gate_features.GATE_FORECAST_COLUMNS |
    GATE_ANALOG_COLUMNS``): a legacy base frame that predates the gate
    feature extension, which the native gate stage reconstructs at its own
    executor. It defers eager inference rather than returning from the
    caller, so the present cells are still validated -- a legal (or illegal)
    omission must never hide a malformed captured string.
    """
    base_role = str(binding.role).split(":", 1)[0]
    derived = set(GATE_FORECAST_COLUMNS) | set(GATE_ANALOG_COLUMNS)
    if base_role == "gate" and set(missing) <= derived:
        return True
    raise FrozenInputsError(
        f"binding {binding.binding_id}: missing feature {missing[0]}"
    )


def _decode_present_cells(
    binding: ModelBinding, feature_order: tuple[str, ...], vector: Mapping[str, Any],
) -> tuple[dict[str, float], bool]:
    """Validate every cell the vector DID capture: ``(finite values, nonfinite)``.

    A malformed (non-numeric, untagged) value is a hard error on either side,
    never an omission. A genuinely non-finite captured value omits the
    binding exactly as a complete row would -- but the flag is ACCUMULATED
    rather than returned early, so an earlier NaN cannot mask a later
    malformed cell.
    """
    values: dict[str, float] = {}
    nonfinite = False
    for name in feature_order:
        if name not in vector:
            continue
        raw = vector[name]
        decoded = untag_nonfinite(raw) if isinstance(raw, Mapping) else raw
        try:
            value = float(decoded)
        except (TypeError, ValueError) as exc:
            raise FrozenInputsError(
                f"binding {binding.binding_id}: nonnumeric feature {name}"
            ) from exc
        if not isfinite(value):
            nonfinite = True
            continue
        values[name] = value
    return values, nonfinite


def binding_feature_row(
    binding: ModelBinding, vector: Mapping[str, Any],
) -> tuple[float, ...] | None:
    """One inference row for ``binding`` from its captured feature ``vector``,
    or ``None`` when a required feature came back non-finite.

    Absent names are classified first (``_gate_deferral_applies``: an
    unknown missing feature raises, a GATE row missing ONLY the derived
    forecast/analog columns defers eager inference -- the native gate stage
    derives those columns at its own executor); every present cell is then
    validated (``_decode_present_cells``: a malformed value raises, a
    genuinely non-finite one omits). Only when the row is complete and
    finite does it come back as values in ``feature_order``.

    This is the ONE predicate that decides whether a binding is included,
    shared by ``build_inference_requests`` below (the replay path, via
    ``checks/phase4_frozen_bridge.py``) and ``_frozen_runtime`` in
    ``tools/capture_tier0_corpus.py`` (capture, building the
    ``resolve_context`` receipt from the same recorded feature vector). The
    two sides used to duplicate this rule -- capture's copy checked only for
    a structurally missing feature, never for a non-finite one -- so a trace
    could assert a binding both included (capture's receipt, which fed the
    model unconditionally) and omitted (replay's re-derivation from the
    identical ``role_model_inputs`` the trace itself recorded), a real
    self-contradiction the runtime-vs-captured-receipt check exists to
    catch. Reading both sides from this one function is what keeps them
    from drifting apart again.

    A captured feature that was genuinely non-finite at capture time is
    tagged ``{"__nonfinite__": repr(value)}`` (contracts §2.1; see
    ``engine.v2.foundation.canonical.tag_nonfinite``/``untag_nonfinite``) --
    a real sourced NaN (e.g. no ORATS quote that day), not a dropped column.
    Decoding it (rather than letting ``float()`` raise on the dict itself)
    is what tells a genuinely non-finite captured value apart from an
    actually-malformed one.

    A binding whose row comes back non-finite yields ``None`` here -- the
    caller must omit it, never raise. This mirrors
    ``engine/v2/scoring/frozen_executor.py``'s ``FrozenStageExecutor._row``
    ("a non-finite value is a missing feature, not an invalid one") and
    legacy itself: ``engine.score.Scorer._score_model`` flags
    MISSING_FEATURES and never calls ``.predict`` on a non-finite row, and
    ``engine.data.features.tier4.ServingModel.predict`` masks an incomplete
    row to NaN without calling its estimator either way -- neither legacy
    path ever asks a model to score an incomplete feature vector. On the
    replay side, omitting the request means ``prepare_frozen_replay`` hands
    the omission's binding fewer requests than bindings, which
    ``engine.v2.scoring.application.score_frozen`` already handles (its
    ``bindings``/``results`` are built FROM ``requests``, not from
    ``release.bindings``), so the role simply produces no frozen output --
    read as the native record's own MISSING_FORECAST_OUTPUT for that role,
    comparable against legacy's own decline, instead of a hard refusal that
    excludes the whole record from the population before any comparison is
    even attempted.

    A structurally missing feature name (the key never captured at all) or
    a genuinely non-numeric value (a string, not a nonfinite tag) both
    raise ``FrozenInputsError`` -- those are not an omission either side
    should make silently.
    """
    feature_order = tuple(binding.feature_order)
    missing = [name for name in feature_order if name not in vector]
    defer = _gate_deferral_applies(binding, missing) if missing else False
    values, nonfinite = _decode_present_cells(binding, feature_order, vector)
    if defer or nonfinite:
        return None
    return tuple(values[name] for name in feature_order)


def build_inference_requests(
    inputs: NativeScoreInputs,
    bindings: Sequence[ModelBinding],
    release_id: str,
) -> tuple[InferenceRequest, ...]:
    """One inference row per binding, in the binding's own feature order.

    A strict capture records the row each binding was fed per role
    (``features.role_model_inputs``): the gate model's vector lives only in
    its own ``gate_inputs`` checkpoint, never in the merged forecast-family
    ``model_inputs``, and a same-named column may hold another value there.
    When the trace carries per-role rows, each binding reads ONLY its own
    role's row. A trace without them can still feed the forecast-family
    roles from ``model_inputs`` (the merge refuses a cross-role conflict), but
    never a gate or chooser binding.

    Per-binding row construction (nonfinite decoding, omission on a
    non-finite feature) is ``binding_feature_row`` -- see its docstring.
    """
    merged = inputs.features.get("model_inputs")
    if not isinstance(merged, Mapping):
        raise FrozenInputsError("native inputs require features.model_inputs")
    role_rows = inputs.features.get("role_model_inputs")
    if role_rows is not None and not isinstance(role_rows, Mapping):
        raise FrozenInputsError("features.role_model_inputs: expected object")
    requests = []
    for binding in bindings:
        role = binding.role.split(":", 1)[0]
        if role_rows is not None:
            vector = role_rows.get(binding.role, role_rows.get(role))
            if not isinstance(vector, Mapping):
                raise FrozenInputsError(
                    f"binding {binding.binding_id}: no captured row for role {binding.role}"
                )
        elif role in _ROLE_PRIVATE_VECTORS:
            raise FrozenInputsError(
                f"binding {binding.binding_id}: role {role} needs its own captured "
                "row (features.role_model_inputs); the merged model_inputs is "
                "not its feature vector"
            )
        else:
            vector = merged
        row = binding_feature_row(binding, vector)
        if row is None:
            continue
        requests.append(InferenceRequest(
            release_id=release_id,
            binding_id=binding.binding_id,
            feature_order=binding.feature_order,
            rows=(row,),
        ))
    return tuple(requests)


def validate_answer_free(inputs: NativeScoreInputs) -> None:
    """Refuse ``NativeScoreInputs`` that smuggle a calculated answer in.

    A prebuilt geometry or pricing object is refused first -- selected legs,
    entry cost and the payoff arithmetic they imply are exactly what the
    geometry and pricing stages exist to derive. Then every block named in
    ``_ANSWER_FIELDS`` is scanned in that map's insertion order for any key
    matching a forbidden output name, and the first dirty block is refused by
    name with its offending keys sorted; a clean sourced-only record passes.
    Key matching is exact (on ``str(key)``), so a recipe, address or raw
    market fact carrying a similar name is never caught here -- the
    ``source_inputs`` builder's own answer boundary owns how those blocks get
    filled; this gate is what replay (``checks/phase4_frozen_bridge.py``, via
    its ``_answer_free`` compatibility wrapper) and future native workers
    share so no execution path scores against its own answers.
    """
    if inputs.geometry is not None or inputs.pricing is not None:
        raise FrozenInputsError("native inputs contain calculated geometry or pricing")
    for block_name, forbidden in _ANSWER_FIELDS.items():
        block = getattr(inputs, block_name)
        found = sorted(str(key) for key in block if str(key) in forbidden)
        if found:
            raise FrozenInputsError(
                f"native inputs {block_name} contain calculated answers: {found}"
            )
