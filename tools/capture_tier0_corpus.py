#!/usr/bin/env python3
"""Capture the tier-0 corpus: frozen ``(request, record)`` pairs (phase 0 step 4).

    python3 tools/capture_tier0_corpus.py                 # fixtures/tier0/<version>/
    python3 tools/capture_tier0_corpus.py --out /tmp/c1
    python3 tools/capture_tier0_corpus.py --forward-days 35 --max-events 60

This is the slow half of the loop and it runs once: it builds a real
:class:`engine.score.Scorer` (which loads the panel and half a million replayed
trades), scores real events through the real public entry points, and writes
the answers down. Everything after it — ``checks/tier0_corpus.py`` — runs in
seconds against what this wrote, with no panel, no network and no fitting.

Four capture rules, each of them a defect this program has already paid for:

* **Full precision.** Replay inputs are serialized unrounded. ``b33036c`` and
  ``6b9d5cf`` are exactly this: ``json_safe`` rounded ``structure_params`` to
  six places and ``_write_pair`` re-rounded it after the exemption. A corpus
  written through the board's display path would freeze the bug as the
  baseline, so nothing here goes near ``round_to``.
* **Deterministic payload, separate envelope.** Wall-clock time, worker id and
  duration live outside the hashed payload (contracts §2.2), so a replay
  reproduces the payload without reproducing the elapsed time.
* **Real public entry points.** ``engine.score.Scorer.score`` for scores,
  ``engine.score.dynamic_short_vol`` for the chooser, ``engine.replay.replay_one``
  for a disabled structure priced under research. §3.2: do not invent a column
  such as ``event_id`` in a fixture if the current serving row does not carry
  one.
* **Private.** The fixtures carry real quotes. ``checks/repo_hygiene.py`` blocks
  ``fixtures/`` from the public repo.

**Coverage is reported, never faked.** The §7.1 table is a set of axes, and
what each axis MEANS is :func:`checks.tier0_corpus.derive_covers` — one
definition, used here to select and there to re-derive. The capture scores a
wide window and then selects the covering subset from what the store actually
produced. An axis nothing covered is written into ``INDEX.json`` as a named
gap. A fixture invented to fill a row of a table proves nothing about the
engine.

**Relations are frozen, not implied.** A pinned fixture records which
selector-resolved pair it was pinned FROM, and that pair is kept, so the
`e845f3e` regression can be checked on real data. A DYN-SV fixture freezes the
exact rows, in order, the chooser ranked, so tier 1 can re-score them and
re-run the choice; tie-breaking depends on that order.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.tier0_corpus import derive_covers, priced  # noqa: E402
from engine import replay as replay_mod  # noqa: E402
from engine import score as score_mod  # noqa: E402
from engine.data import store  # noqa: E402
from engine.fills import MID  # noqa: E402
from engine.structures import STRUCTURES  # noqa: E402
from engine.v2.contracts import ScoreRequest as V2ScoreRequest  # noqa: E402
from engine.v2.diagnosis import content_hash  # noqa: E402
from engine.v2.foundation import to_document  # noqa: E402
from engine.v2.models import FrozenInference, InferenceRequest, ModelBinding, ModelRelease  # noqa: E402
from engine.v2.models.contracts import ArtifactMember  # noqa: E402
from engine.v2.scoring import application as v2_application  # noqa: E402
from engine.v2.scoring.stages import (  # noqa: E402
    NativeScoreInputs, StageObservation, receipt,
)
from tools.phase4_checkpoint_sink import DiskCheckpointSink  # noqa: E402
from tools.phase4_release_assembler import assemble_input_trace  # noqa: E402
from tools.phase4_frozen_resources import FrozenResourcePackage, package_frozen_resources  # noqa: E402

SCHEMA_VERSION = "tier0_pair.v1.1"
INDEX_VERSION = "tier0_corpus.v1.1"
DEFAULT_OUT = ROOT / "fixtures" / "tier0"

#: How a NaN is frozen. Not ``null``: contracts §2.1 forbids sending a missing
#: value as NaN, and collapsing the two here would lose the distinction between
#: "the engine produced NaN" and "the engine produced nothing" — which is half
#: of what the null-mask comparison exists to catch.
NONFINITE = "__nonfinite__"

#: The refusal codes of §7.1, and the flag the current engine emits for each.
#: Six, not seven: ``BAD_QUOTE_COST_PCT`` is the 30% threshold constant in
#: ``engine.fills`` behind the single ``BAD_QUOTE`` flag (``engine/score.py``
#: emits ``BAD_QUOTE`` in exactly one place, on that bar), not a separate
#: refusal. The baseline package exports the constant.
REFUSAL_CODES = {code: code for code in (
    "UNVALIDATED_STRUCTURE", "OUT_OF_DOMAIN", "NO_CHAIN", "BAD_QUOTE",
    "COARSE_LADDER", "NO_FORECAST",
)}

MODEL_ROLES = ("size", "implied_t1", "runup_move", "iv_crush", "gate", "chooser")

#: ``ScoreRequest`` fields serialized as dates.
_DATE_FIELDS = frozenset({"as_of", "event_date", "expiry", "chain_as_of"})

#: Strategies whose native scoring path (``engine.v2.scoring.stages``'
#: ``_STRATEGY_FORECAST_ROLES`` and ``engine.v2.scoring.source_inputs``'
#: ``_SUPPORTED_STRATEGIES``) can reconstruct an executable recipe from
#: source-owned captured inputs alone: STR-THRU, STR-RUNUP, and the seven
#: DYN-SV menu strategies. This used to be narrower (STR-THRU/STR-RUNUP only)
#: when this capture tool was first written; R4-6 extended native scoring's
#: bucket-analog recipe to the menu strategies, but this constant was never
#: widened to match. Disabled strategies (CAL-P, CND-P — research-only, no
#: production gate) are excluded: their rows are captured as a different
#: fixture kind (``research_replay``), never ``score_result``, so they never
#: reach strict probing in the first place. The DYN-SV chooser meta-strategy
#: is excluded the same way (``dyn_sv_choice`` kind).
STRICT_TRACE_SUPPORTED_STRATEGIES = (
    frozenset(STRUCTURES) - frozenset(score_mod.DISABLED_STRATEGIES)
)


class StrictTraceCaptureError(ValueError):
    """A legacy capture lacks source-owned inputs needed for native replay."""


def parse_strategies(values: Iterable[str] | None) -> tuple[str, ...] | None:
    """Normalize an optional CLI strategy filter; None preserves all."""
    if values is None:
        return None
    requested = tuple(dict.fromkeys(
        item.strip()
        for value in values
        for item in str(value).split(",")
        if item.strip()
    ))
    allowed = set(STRUCTURES) | {score_mod.DYNAMIC_STRATEGY}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise StrictTraceCaptureError(f"unknown strategies: {unknown}")
    if not requested:
        raise StrictTraceCaptureError("--strategies requires at least one strategy")
    return requested


def _score_strategies(strategies: tuple[str, ...] | None) -> tuple[str, ...]:
    selected = set(STRUCTURES) if strategies is None else set(strategies)
    return tuple(name for name in STRUCTURES if name in selected)


# --------------------------------------------------------------------------
# full-precision serialization
# --------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """Convert to JSON-writable form **without rounding anything, ever**."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return {NONFINITE: repr(out)} if not math.isfinite(out) else out
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int)):
        return value
    if hasattr(value, "_asdict"):
        return jsonable(value._asdict())
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: jsonable(getattr(value, f.name))
                for f in dataclass_fields(value)}
    return str(value)


def request_to_dict(request: score_mod.ScoreRequest) -> dict:
    """The exact replay request, at full precision.

    Per contracts §9.5 this is persisted independently of any display
    projection: a client replays from the saved request or the score id, never
    from rounded values copied out of a table.
    """
    out = {f.name: jsonable(getattr(request, f.name))
           for f in dataclass_fields(request) if f.name != "fill"}
    out["fill"] = {"policy_id": "legacy.fill_alpha.v1",
                   "alpha": float(request.fill.alpha)}
    out["identity_key"] = request.key()
    return out


def request_from_dict(data: dict) -> score_mod.ScoreRequest:
    """Inverse of :func:`request_to_dict`, field by field."""
    kwargs = {}
    for f in dataclass_fields(score_mod.ScoreRequest):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name == "fill":
            value = score_mod.FillModel(alpha=float(value["alpha"]))
        elif f.name in _DATE_FIELDS and isinstance(value, str):
            value = pd.Timestamp(value)
        kwargs[f.name] = value
    return score_mod.ScoreRequest(**kwargs)


def canonical_v2_request(candidate: Mapping[str, Any], snapshot: str) -> V2ScoreRequest:
    """Translate source-owned legacy request identity into a canonical command."""
    event_id = candidate.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise StrictTraceCaptureError("strict trace requires the captured event_id")
    raw = candidate.get("request")
    if not isinstance(raw, Mapping):
        raw = request_to_dict(raw)
    legacy = request_from_dict(dict(raw))
    if legacy.strategy not in STRICT_TRACE_SUPPORTED_STRATEGIES:
        raise StrictTraceCaptureError(
            f"strict probe does not support {legacy.strategy} "
            f"(supported: {sorted(STRICT_TRACE_SUPPORTED_STRATEGIES)})"
        )
    decision = legacy.as_of if legacy.as_of is not None else legacy.chain_as_of
    if decision is None:
        raise StrictTraceCaptureError("strict trace requires a decision date")
    event_date = legacy.event_date
    if event_date is None:
        raise StrictTraceCaptureError("strict trace requires an event date")
    event_identity = {
        "event_id": event_id,
        "ticker": legacy.ticker,
        "event_date": str(pd.Timestamp(event_date).date()),
        "session": legacy.session,
    }
    geometry_override = dict(legacy.structure_params or {})
    if legacy.strike is not None:
        geometry_override["strike"] = float(legacy.strike)
    if legacy.expiry is not None:
        geometry_override["expiry"] = str(pd.Timestamp(legacy.expiry).date())
    return V2ScoreRequest(
        event_id=event_id,
        event_revision="event:" + content_hash(event_identity),
        calendar_revision="calendar:" + content_hash({
            "decision": str(pd.Timestamp(decision).date()),
            "event": event_identity,
        }),
        strategy_version=legacy.strategy,
        deployment_id=f"legacy-capture:{snapshot}",
        decision_clock_id=(
            "legacy.decision_offset."
            + str(legacy.decision_offset if legacy.decision_offset is not None else 0)
        ),
        requested_decision_at=str(pd.Timestamp(decision).date()),
        snapshot_id=str(snapshot),
        mode="replay",
        fill_model={
            "policy_id": "legacy.fill_alpha.v1",
            "alpha": float(legacy.fill.alpha),
        },
        geometry_override=geometry_override or None,
    )


def _checkpoint_value(candidate: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    trace = candidate.get("legacy_trace")
    checkpoints = trace.get("checkpoints") if isinstance(trace, Mapping) else None
    row = checkpoints.get(name) if isinstance(checkpoints, Mapping) else None
    if not isinstance(row, Mapping) or not isinstance(row.get("value"), Mapping):
        raise StrictTraceCaptureError(f"legacy checkpoint missing {name}")
    value = row["value"]
    if row.get("content_hash") != content_hash(value):
        raise StrictTraceCaptureError(f"legacy checkpoint hash mismatch: {name}")
    return value


def _checkpoint_value_optional(candidate: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """Like :func:`_checkpoint_value`, but returns ``None`` if the group was
    never recorded (e.g. an entry-rule gate never writes a model feature
    vector). A recorded-but-malformed or hash-mismatched group still raises."""
    trace = candidate.get("legacy_trace")
    checkpoints = trace.get("checkpoints") if isinstance(trace, Mapping) else None
    if not isinstance(checkpoints, Mapping) or name not in checkpoints:
        return None
    return _checkpoint_value(candidate, name)


def _quote_map(rows: Any) -> dict[str, dict[str, float]]:
    if not isinstance(rows, list) or not rows:
        raise StrictTraceCaptureError("source_inputs.quote_domain is empty")
    quotes: dict[str, dict[str, float]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StrictTraceCaptureError(f"quote_domain[{index}] is not an object")
        try:
            right = str(row["right"]).upper()
            right = {"CALL": "C", "PUT": "P"}.get(right, right)
            strike = float(row["strike"])
            expiry = str(pd.Timestamp(row["expiry"]).date())
            bid = float(row["bid"])
            ask = float(row["ask"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StrictTraceCaptureError(
                f"quote_domain[{index}] lacks a complete contract quote"
            ) from exc
        if right not in {"C", "P"} or not all(
            math.isfinite(value) for value in (strike, bid, ask)
        ) or bid < 0.0 or ask < bid:
            raise StrictTraceCaptureError(f"quote_domain[{index}] is invalid")
        key = f"{right}:{strike}:{expiry}"
        quote = {"bid": bid, "ask": ask}
        if key in quotes and quotes[key] != quote:
            raise StrictTraceCaptureError(f"conflicting source quote: {key}")
        quotes[key] = quote
    return quotes


#: Legacy checkpoint role strings that ``tools/phase4_frozen_resources.py``
#: (``_normalized_binding``) canonicalizes before a binding reaches
#: ``ModelBinding.role``. Kept in lockstep with that mapping so a per-role
#: captured vector can be looked up by the SAME role a frozen binding carries.
_LEGACY_ROLE_ALIASES = {
    "abs_move": "driver",
    "forecast_sizing": "size",
}


def _coerce_feature_value(role: str, name: str, raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise StrictTraceCaptureError(
            f"feature {role}.{name} is missing or nonnumeric"
        ) from exc
    if not math.isfinite(value):
        raise StrictTraceCaptureError(f"feature {role}.{name} is nonfinite")
    return value


def _merged_model_inputs(candidate: Mapping[str, Any]) -> dict[str, float]:
    features = _checkpoint_value(candidate, "features")
    vectors = features.get("feature_vector")
    if not isinstance(vectors, Mapping) or not vectors:
        raise StrictTraceCaptureError("features.feature_vector is empty")
    merged: dict[str, float] = {}
    for role, vector in vectors.items():
        if not isinstance(vector, Mapping):
            raise StrictTraceCaptureError(f"feature vector {role} is malformed")
        for name, raw in vector.items():
            value = _coerce_feature_value(str(role), str(name), raw)
            if name in merged and merged[name] != value:
                raise StrictTraceCaptureError(
                    f"feature {name} differs across model roles"
                )
            merged[str(name)] = value
    return merged


def _role_feature_vectors(candidate: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Per-role captured feature vectors, keyed by canonical binding role.

    Unlike :func:`_merged_model_inputs` — which unions every captured role's
    vector into one flat dict for the forecast-facing ``model_inputs`` block,
    and refuses a genuine cross-role value conflict — this keeps each role's
    vector separate. A binding whose ``feature_order`` is private to its own
    role (the gate model's own feature vector is captured into the
    ``gate_inputs`` checkpoint, not the shared ``features`` checkpoint used by
    the forecast-family roles) can still be resolved here, and frozen
    inference row assembly (:func:`_frozen_runtime`) must read from here, not
    from the merged dict.
    """
    vectors: dict[str, dict[str, float]] = {}

    def _add(role: str, raw_vector: Mapping[str, Any]) -> None:
        coerced = {
            str(name): _coerce_feature_value(role, str(name), raw)
            for name, raw in raw_vector.items()
        }
        if role in vectors and vectors[role] != coerced:
            raise StrictTraceCaptureError(
                f"feature role {role} captured twice with different values"
            )
        vectors[role] = coerced

    features = _checkpoint_value(candidate, "features")
    role_vectors = features.get("feature_vector")
    if not isinstance(role_vectors, Mapping) or not role_vectors:
        raise StrictTraceCaptureError("features.feature_vector is empty")
    for raw_role, vector in role_vectors.items():
        if not isinstance(vector, Mapping):
            raise StrictTraceCaptureError(f"feature vector {raw_role} is malformed")
        role = _LEGACY_ROLE_ALIASES.get(str(raw_role), str(raw_role))
        _add(role, vector)

    gate = _checkpoint_value_optional(candidate, "gate_inputs")
    if isinstance(gate, Mapping) and gate.get("kind") == "model":
        gate_vector = gate.get("feature_vector")
        if isinstance(gate_vector, Mapping) and gate_vector:
            _add("gate", gate_vector)

    dyn_sv = _checkpoint_value_optional(candidate, "dyn_sv")
    if isinstance(dyn_sv, Mapping):
        ranking = dyn_sv.get("ranking")
        chooser_vector = (
            ranking.get("feature_vector") if isinstance(ranking, Mapping) else None
        )
        if isinstance(chooser_vector, Mapping) and chooser_vector:
            _add("chooser", chooser_vector)

    return vectors


def native_inputs_from_capture(
    candidate: Mapping[str, Any],
    request: V2ScoreRequest,
) -> tuple[NativeScoreInputs, dict[str, Any]]:
    """Build executable native inputs only from source-owned captured material."""
    source = _checkpoint_value(candidate, "source_inputs")
    recipes = source.get("native_recipes")
    if not isinstance(recipes, Mapping):
        raise StrictTraceCaptureError(
            "source_inputs lacks executable native_recipes "
            "(forecast, analogs, simulation, and gate)"
        )
    required_recipes = {"forecast"}
    missing_recipes = sorted(required_recipes - set(recipes))
    if missing_recipes:
        raise StrictTraceCaptureError(
            f"source_inputs.native_recipes missing {missing_recipes}"
        )
    context = dict(source.get("context") or {})
    source_features = dict(source.get("features") or {})
    for key in ("entry_date", "exit_date", "expiry", "spot", "as_of"):
        if key not in context and source_features.get(key) is not None:
            context[key] = source_features[key]
    context["strategy"] = request.strategy_version
    context["quotes"] = _quote_map(source.get("quote_domain"))
    missing_context = sorted(
        key for key in ("ticker", "event_date", "entry_date", "exit_date", "spot")
        if context.get(key) is None
    )
    if missing_context:
        raise StrictTraceCaptureError(
            f"source_inputs context missing {missing_context}"
        )
    recipes = dict(recipes)
    recipes.setdefault("analogs", {"mode": "not_applicable"})
    recipes.setdefault("simulation", {"mode": "not_applicable"})
    recipes.setdefault("gate", {"mode": "not_applicable"})
    blocks = {
        "context": context,
        "features": {
            "model_inputs": _merged_model_inputs(candidate),
            "source_features": source_features,
        },
        "forecast": dict(recipes["forecast"]),
        "geometry": None,
        "pricing": None,
        "analogs": dict(recipes["analogs"]),
        "simulation": dict(recipes["simulation"]),
        "gate": dict(recipes["gate"]),
        "chooser": dict(recipes.get("chooser") or {}),
        "diagnostics": dict(recipes.get("diagnostics") or {}),
    }
    request_doc = to_document(request)
    shared_inputs = {"request": request_doc, "native_inputs": blocks}
    source_ref = content_hash(shared_inputs)
    declarations = tuple(
        receipt(stage, {"source_ref": source_ref}, {"execution": "native-runtime"})
        for stage in (
            "resolve_context", "features", "forecast", "geometry", "pricing",
            "analogs", "simulation", "gate", "chooser", "serialization",
        )
    )
    inputs = NativeScoreInputs(
        **blocks, source_ref=source_ref, stage_receipts=declarations,
    )
    return inputs, shared_inputs


def package_strict_trace(
    request: V2ScoreRequest,
    inputs: NativeScoreInputs,
    shared_inputs: Mapping[str, Any],
    *,
    resources: list[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    frozen_runtime: tuple[FrozenInference, ModelRelease, tuple[InferenceRequest, ...]] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Execute native scoring and package the observer output for verification."""
    observations = []
    if frozen_runtime is None:
        native = v2_application.score_one(request, inputs, observer=observations.append)
    else:
        inference, release, inference_requests = frozen_runtime
        native = v2_application.score_frozen(
            request,
            inference,
            release,
            inference_requests,
            {"_native_inputs": inputs},
            observer=observations.append,
        )
    observations = [
        StageObservation(
            input_document=to_document(item.input_document),
            output_document=to_document(item.output_document),
            receipt=receipt(
                item.receipt.stage,
                to_document(item.input_document),
                to_document(item.output_document),
            ),
        )
        for item in observations
    ]
    native_document = {
        "context": dict(inputs.context),
        "features": dict(inputs.features),
        "forecast": dict(inputs.forecast),
        "geometry": None if inputs.geometry is None else to_document(inputs.geometry),
        "pricing": None if inputs.pricing is None else to_document(inputs.pricing),
        "analogs": dict(inputs.analogs),
        "simulation": dict(inputs.simulation),
        "gate": dict(inputs.gate),
        "chooser": dict(inputs.chooser),
        "diagnostics": dict(inputs.diagnostics),
        "source_ref": inputs.source_ref,
    }
    trace = assemble_input_trace(
        request=to_document(request),
        shared_inputs=shared_inputs,
        native_inputs=native_document,
        observations=observations,
        resources=list(resources or ()),
        metadata=metadata,
    )
    return trace, native


def _frozen_runtime(
    package: FrozenResourcePackage,
    release_root: Path,
    request: V2ScoreRequest,
    inputs: NativeScoreInputs,
    candidate: Mapping[str, Any],
) -> tuple[FrozenInference, ModelRelease, tuple[InferenceRequest, ...]]:
    resources = {row["resource_id"]: row for row in package.resource_rows}
    bindings = []
    for raw in package.sidecar_document["bindings"]:
        members = tuple(
            ArtifactMember(
                name=member["name"],
                path=resources[member["resource_id"]]["path"],
                content_hash=resources[member["resource_id"]]["sha256"],
            )
            for member in raw["members"]
        )
        bindings.append(ModelBinding(
            binding_id=raw["binding_id"],
            model_id=raw["model_id"],
            role=raw["role"],
            strategy_id=raw["strategy_id"],
            decision_clock_id=raw["decision_clock_id"],
            adapter=raw["adapter"],
            feature_order=tuple(raw["feature_order"]),
            output_names=tuple(raw["output_names"]),
            members=members,
        ))
    release = ModelRelease(
        release_id=package.sidecar_document["release_id"],
        deployment_id=request.deployment_id,
        bindings=tuple(bindings),
    )
    # Each binding's row must come from THAT binding's own captured per-role
    # feature vector, never the cross-role merged dict
    # (`inputs.features["model_inputs"]`): a binding's feature_order can name
    # features private to its own role (e.g. the gate model's own vector),
    # which the merge — built only from the forecast-family `features`
    # checkpoint — never carries. See `_role_feature_vectors`.
    role_vectors = _role_feature_vectors(candidate)
    inference_requests = []
    for binding in bindings:
        vector = role_vectors.get(binding.role)
        if vector is None:
            raise StrictTraceCaptureError(
                f"frozen runtime binding {binding.binding_id} (role={binding.role}): "
                "no captured per-role feature vector"
            )
        missing = [name for name in binding.feature_order if name not in vector]
        if missing:
            raise StrictTraceCaptureError(
                f"frozen runtime binding {binding.binding_id} (role={binding.role}): "
                f"missing feature(s) {missing}"
            )
        inference_requests.append(InferenceRequest(
            release_id=release.release_id,
            binding_id=binding.binding_id,
            feature_order=binding.feature_order,
            rows=(tuple(vector[name] for name in binding.feature_order),),
        ))
    return FrozenInference(release_root), release, tuple(inference_requests)


# --------------------------------------------------------------------------
# one pair
# --------------------------------------------------------------------------


def make_pair(fixture_id: str, covers: list[str], request: dict, record: dict,
              *, record_kind: str, duration: float,
              legacy_trace: dict | None = None,
              input_trace: dict | None = None,
              legacy_input_hash: str | None = None,
              strict_trace_gap: str | None = None,
              relations: dict | None = None, notes: str = "") -> dict:
    payload: dict[str, Any] = {"request": request, "record": record,
                               "record_kind": record_kind}
    if legacy_trace is not None:
        # This is the source execution trace, not a Phase 4 acceptance bundle.
        # Acceptance requires a typed request, sidecars, and native receipts;
        # publication records the current disposition explicitly so an
        # incomplete trace cannot be mistaken for a completed one.
        payload["legacy_trace"] = legacy_trace
        payload["trace_disposition"] = "incomplete"
    if strict_trace_gap is not None:
        if input_trace is not None:
            raise StrictTraceCaptureError(
                "a pair cannot carry both a verified input_trace and a "
                "strict_trace_gap"
            )
        # Strict tracing was attempted for this row and could not produce an
        # honest trace. The typed reason is recorded verbatim, never
        # replaced by a fabricated trace and never silently dropped.
        payload["trace_disposition"] = "gap"
        payload["strict_trace_gap"] = strict_trace_gap
    if input_trace is not None:
        if legacy_input_hash != input_trace.get("shared_input_hash"):
            raise StrictTraceCaptureError("strict trace legacy input hash mismatch")
        payload["input_trace"] = input_trace
        payload["input_trace_hash"] = input_trace["trace_hash"]
        payload["legacy_input_hash"] = legacy_input_hash
        payload["trace_disposition"] = "complete"
    if relations:
        payload["relations"] = relations
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "covers": sorted(set(covers)),
        "notes": notes,
        "payload": payload,
        "payload_hash": content_hash(payload),
        "request_hash": content_hash(request),
        # contracts §2.2: the envelope is excluded from the payload hash, so a
        # replay reproduces the payload without reproducing the elapsed time.
        "envelope": {
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "worker_ref": f"{platform.node()}:{os.getpid()}",
            "duration_seconds": duration,
        },
    }


# --------------------------------------------------------------------------
# required coverage (§7.1)
# --------------------------------------------------------------------------


def axis_inputs() -> dict:
    """Everything :func:`derive_covers` needs, frozen into the index."""
    return {
        "structures": sorted(STRUCTURES),
        "dynamic_strategy": score_mod.DYNAMIC_STRATEGY,
        "menu": list(score_mod.DYNAMIC_MENU),
        "disabled": list(score_mod.DISABLED_STRATEGIES),
        "model_roles": list(MODEL_ROLES),
        "refusal_code_mapping": dict(REFUSAL_CODES),
    }


def required_axes() -> list[str]:
    axes = [f"strategy:{name}" for name in STRUCTURES]
    axes.append(f"strategy:{score_mod.DYNAMIC_STRATEGY}")
    # Every SERVED strategy must also appear PRICED — legs and an entry cost,
    # not a refusal. The disabled pair is exempt: production refuses them by
    # design, and their priced behaviour is the research_replay axis instead.
    axes += [f"priced:{name}" for name in STRUCTURES
             if name not in score_mod.DISABLED_STRATEGIES]
    axes.append(f"priced:{score_mod.DYNAMIC_STRATEGY}")
    axes += [f"model_role:{r}" for r in MODEL_ROLES]
    axes += [f"refusal:{c}" for c in REFUSAL_CODES]
    axes += ["session:BMO", "session:AMC", "boundary:year", "boundary:month"]
    axes += ["geometry:pinned", "geometry:selector", "geometry:computed_width",
             "geometry:round_listed_strike", "geometry:coarse_ladder",
             "geometry:exact_mirror"]
    # `dyn_sv:tie` is deliberately NOT required (decision 2026-09-12). No
    # genuine tie between two different structures exists in the store or in
    # the prediction ledger — chooser scores are continuous — and the corpus
    # may not invent one. The tie RULE (input-row order breaks a tie, on both
    # ranking paths) is guarded instead by the frozen definition in
    # `definitions/dyn_sv.json` and by
    # `tests/test_baseline_export.py::test_the_exported_tie_rule_is_the_measured_behaviour`,
    # which runs the real `dynamic_short_vol` on tied rows in both orders.
    # `derive_covers` still reports the axis if a real tie is ever captured.
    axes += ["dyn_sv:full_menu", "dyn_sv:partial_menu", "dyn_sv:fallback"]
    for name in score_mod.DISABLED_STRATEGIES:
        axes += [f"disabled:{name}:refused", f"disabled:{name}:research_replay"]
    return sorted(set(axes))


# --------------------------------------------------------------------------
# scoring passes
# --------------------------------------------------------------------------


def _events(as_of: pd.Timestamp, forward_days: int, max_events: int) -> pd.DataFrame:
    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"]
    )
    horizon = as_of + pd.Timedelta(days=forward_days)
    forward = events[(events["event_date"] >= as_of)
                     & (events["event_date"] <= horizon)
                     & events["session"].notna()]
    forward = forward.sort_values(["event_date", "ticker"]).head(max_events)
    return forward.reset_index(drop=True)


def _with_chains(candidates: pd.DataFrame, calendar, per_kind: int,
                 structure: str = "STR-THRU") -> pd.DataFrame:
    """Keep only events whose entry and exit chains are both in the store.

    Without this the boundary fixtures come back as NO_CHAIN placeholders, which
    carry no entry or exit date and therefore cannot demonstrate a boundary at
    all — a fixture that covers the axis in name only.

    ``structure`` is the structure whose plan defines the window. The year kind
    checks STR-RUNUP rather than STR-THRU because STR-THRU enters on the last
    pre-print session and exits on the first post-print one — a one-session
    window that cannot cross a year boundary for ANY event.
    """
    if candidates.empty:
        return candidates
    available = replay_mod.available_chain_keys()
    plan = replay_mod.plan_events(STRUCTURES[structure](), candidates,
                                  calendar=calendar)
    keep = []
    for row in plan.frame.to_dict("records"):
        entry = (row["ticker"], pd.Timestamp(row["entry_date"]).normalize())
        exit_ = (row["ticker"], pd.Timestamp(row["exit_date"]).normalize())
        if entry in available and exit_ in available:
            keep.append(row["event_id"])
        if len(keep) >= per_kind:
            break
    return candidates[candidates["event_id"].isin(keep)]


def _boundary_events(as_of: pd.Timestamp, per_kind: int, calendar) -> pd.DataFrame:
    """Past events whose trade window crosses a month or a year boundary.

    The year boundary needs a print early enough in January that a d-14 entry
    lands in December, on a name the chain store carries in December, and a
    structure that actually enters pre-print — so the year candidates ride on
    STR-RUNUP.
    """
    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"]
    )
    past = events[(events["event_date"] < as_of)
                  & (events["event_date"] >= as_of - pd.Timedelta(days=2500))
                  & events["session"].notna()].copy()
    past["day"] = past["event_date"].dt.day
    past["month"] = past["event_date"].dt.month
    year = _with_chains(
        past[(past["month"] == 1) & (past["day"] <= 12)].sort_values(
            "event_date", ascending=False),
        calendar, per_kind, structure="STR-RUNUP")
    month = _with_chains(
        past[past["day"] <= 2].sort_values("event_date", ascending=False),
        calendar, per_kind)
    return pd.concat([year, month]).drop_duplicates("event_id").reset_index(drop=True)


# --------------------------------------------------------------------------
# legacy_trace spill: keep every candidate's Phase 4 checkpoint content on
# disk, not resident, for the run's whole life
# --------------------------------------------------------------------------
#
# `main()` scores every forward/boundary/pinned/strike/coarse/research-replay
# candidate into ONE list (`candidates`) before `select()` ever runs, and
# nothing between capture and `select()` reads a candidate's OWN
# `legacy_trace` again: `select()` covers axes from `record`/`request`/`kind`/
# `relations` alone (see its docstring), and `_rescore` builds each pinned/
# strike/coarse variant from `source["request"]`, never `source["legacy_
# trace"]`. The ONLY code that ever reads a candidate's `legacy_trace`
# content is `attach_strict_probe` and `write`'s own pairs/checkpoint loop —
# both of which only run over `chosen`, `select()`'s "minimal covering
# subset" (a small fraction of everything scored).
#
# Measured 2026-09-18 (diag_capture_retention.py, run 4 against 280cf7c):
# `harness.all_candidates`'s own "UNSAMPLED" deep-size print undercounted
# `out["candidates"]`'s true content by ~14x at n=445 (2.78 MB reported vs.
# 38.28 MB in `legacy_trace.analogs.source_rows` ALONE, measured by a direct,
# non-recursive walk to that one field) because the sizer's nested-level cap
# stops 3 levels down and `legacy_trace["checkpoints"]["source_inputs"]
# ["value"]["native_recipes"]["analogs"]["source_rows"]` sits 7 levels below
# `all_candidates` itself. `matcher._causal_pools`/`_causal_row_caches`/
# `phase4_recipe_cache` (bounded at MAX_CAUSAL_CACHE=64, needed for scoring
# itself) only accounted for 38.5% of the RSS climb between two checkpoints
# in that run; the remainder tracks `out["candidates"]` growing by keeping
# every candidate's checkpoint content (chain snapshots, documented analog
# rows, residual population slices — several MB each for a strategy with a
# large matched population) resident for the rest of the run, for EVERY
# candidate ever scored, not just the ones `select()` eventually keeps.
#
# Spilling removes that: `_candidate()` writes a non-None `legacy_trace` to
# its own file the moment it is produced and keeps only a small `_SpilledTrace`
# pointer in the in-memory dict `candidates` holds. `select()`'s covering pass
# never looks at that pointer. Right after `select()` returns, `chosen` (only)
# is hydrated back to the real dict before `attach_strict_probe`/`write` run —
# the exact same content, read back byte-for-byte (`pickle`, not `json`, so no
# float-precision/NaN-encoding round-trip risk for an internal, same-process,
# same-Python-version spill), so every downstream consumer sees the identical
# object it always did and every written file is unchanged.


class _SpilledTrace:
    """A pointer to one candidate's ``legacy_trace``, held on disk instead of
    resident in the ``candidates`` list. See the module note above.
    """

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        self.path = path


_TRACE_SPILL_DIR: Path | None = None
_TRACE_SPILL_COUNTER = itertools.count()


def _trace_spill_dir() -> Path:
    """The run's spill directory, created on first use and removed by
    ``main`` when the run ends (success or failure).
    """
    global _TRACE_SPILL_DIR
    if _TRACE_SPILL_DIR is None:
        _TRACE_SPILL_DIR = Path(
            tempfile.mkdtemp(prefix="capture_tier0_trace_spill_")
        )
    return _TRACE_SPILL_DIR


def _spill_trace(trace: dict) -> _SpilledTrace:
    path = _trace_spill_dir() / f"{next(_TRACE_SPILL_COUNTER)}.pkl"
    with path.open("wb") as fh:
        pickle.dump(trace, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return _SpilledTrace(path)


def _hydrate_trace(value: Any) -> Any:
    """Read a spilled ``legacy_trace`` back, unchanged, byte-for-byte.

    A no-op on anything that is not a spill pointer (``None``, or an
    already-hydrated dict — idempotent, so a caller never needs to know
    whether an earlier step already hydrated this candidate).
    """
    if isinstance(value, _SpilledTrace):
        with value.path.open("rb") as fh:
            return pickle.load(fh)
    return value


def _cleanup_trace_spill() -> None:
    global _TRACE_SPILL_DIR
    if _TRACE_SPILL_DIR is not None:
        shutil.rmtree(_TRACE_SPILL_DIR, ignore_errors=True)
        _TRACE_SPILL_DIR = None


# `engine.score.Phase4TraceCollector` shares two families of content BY
# REFERENCE across every candidate that hits the same underlying cache --
# `_Predocumented` (see its docstring) exists to stop the collector's own
# sanitizing passes from re-copying:
#   * the residual population before a fixed cutoff (`ResidualPool.
#     documented_population`) -- identical across every pinned/strike/
#     coarse-ladder rescore of the SAME boundary event, since they all share
#     its date;
#   * the analog "recipe" source rows (`phase4_recipe_cache`, see
#     `capture_analog_inputs`'s docstring) -- identical across every
#     candidate sharing a (strategy, alpha, as_of) causal key, which
#     `select()` commonly keeps MORE than one of: a chosen pinned/strike
#     candidate's source is explicitly re-added to `chosen` alongside it,
#     and `_rescore` always inherits the source's `as_of`/`strategy`.
#
# `_document` unwraps `_Predocumented` to the plain shared object itself
# (`value.value`), so by the time a candidate's `legacy_trace` reaches
# `_candidate()` the sharing is invisible except as the SAME object embedded
# at more than one path. Spilling each candidate independently (separate
# `pickle.dump()` calls) does not break sharing WITHIN one candidate's own
# trace -- pickle's per-call memo still reconstructs one object for both
# paths on load -- but it CANNOT see across candidates: two independent
# `pickle.dump()` calls share no memo, so hydrating two `chosen` candidates
# that shared one object in memory before spilling reads back two separate
# full copies. `_reconcile_shared_trace_content` restores the original
# sharing after hydration, keyed on content (not `id()`: `pickle.load()`
# builds fresh objects every time, so no pre-spill identity survives to key
# on) using the same `content_hash` the trace machinery already hashes this
# exact content with elsewhere in this file.
_SHARED_CONTENT_GROUPS: tuple[tuple[tuple[str, ...], ...], ...] = (
    (
        ("checkpoints", "simulation", "value", "residual_population"),
        ("checkpoints", "source_inputs", "value", "native_recipes",
         "simulation", "residuals"),
    ),
    (
        ("checkpoints", "source_inputs", "value", "native_recipes",
         "analogs", "source_rows"),
    ),
)

_MISSING = object()


def _get_in(root: Any, path: tuple[str, ...]) -> Any:
    node = root
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _set_in(root: Any, path: tuple[str, ...], value: Any) -> None:
    node = root
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value


def _reconcile_shared_trace_content(
    trace: dict, cache: dict[str, Any],
) -> None:
    """Re-share content across hydrated candidates that shared it pre-spill.

    ``cache`` is a single dict the caller keeps across every candidate it
    hydrates this run, keyed on ``content_hash`` of the shared value. Content
    that resolves to a hash already in ``cache`` is REPLACED in place with
    the earlier candidate's object, so every hydrated candidate that shared
    an object before spilling shares ONE object again after hydration --
    the written output is unaffected either way (`write` only ever reads
    values, never object identity), only the retained memory is.
    """
    if not isinstance(trace, dict):
        return
    for group in _SHARED_CONTENT_GROUPS:
        present = [path for path in group if _get_in(trace, path) is not _MISSING]
        if not present:
            continue
        value = _get_in(trace, present[0])
        digest = content_hash(value)
        cached = cache.get(digest)
        resolved = cached if cached is not None else value
        for path in present:
            _set_in(trace, path, resolved)
        if cached is None:
            cache[digest] = value


def _score(scorer, request, *, index=None) -> tuple[dict, dict, float, dict]:
    """``(raw as_dict, jsonable record, seconds)`` through ``Scorer.score``.

    The raw row is kept for the chooser frame: ``dynamic_short_vol`` reads the
    board's own rows, with NaN where the engine produced NaN, not the frozen
    ``__nonfinite__`` markers.
    """
    started = time.monotonic()
    as_of = request.as_of if request.as_of is not None else request.chain_as_of
    try:
        trace = score_mod.Phase4TraceCollector(
            retain_full_trace=False, content_hasher=content_hash,
        )
        result = (scorer.score(request, chain_index=index, trace=trace) if index is not None
                  else scorer.score(request, trace=trace))
    except score_mod.UNSCORABLE as exc:
        result = score_mod.unscorable_result(
            request, as_of=as_of, snapshot=scorer.snapshot, exc=exc
        )
        trace.finish(result)
    raw = result.as_dict()
    return raw, jsonable(raw), time.monotonic() - started, trace.diagnostic_checkpoint()


def _candidate(request, raw: dict | None, record: dict, took: float, *,
               kind: str = "score_result", frame: str | None = None,
               relations: dict | None = None, legacy_trace: dict | None = None) -> dict:
    # Spilled immediately, not held: see the module note above `_score` for
    # why nothing between here and `select()` needs this candidate's OWN
    # `legacy_trace` content, and `chosen` (only) is hydrated back after
    # `select()` runs.
    stored_trace = _spill_trace(legacy_trace) if legacy_trace is not None else None
    return {"request": request if isinstance(request, dict) else request_to_dict(request),
            "raw": raw, "record": record, "duration": took, "kind": kind,
            "frame": frame, "relations": relations or {},
            "legacy_trace": stored_trace}


def forward_pass(scorer, events: pd.DataFrame, as_of: pd.Timestamp,
                 quote_max_age: int,
                 strategies: tuple[str, ...] | None = None) -> list[dict]:
    """Every strategy on every forward event, as the board's scoring loop does."""
    strategy_names = _score_strategies(strategies)
    keys: set[tuple[str, pd.Timestamp]] = set()
    for strategy in strategy_names:
        if strategy in score_mod.DISABLED_STRATEGIES:
            continue
        plan = replay_mod.plan_events(
            STRUCTURES[strategy](), events, calendar=scorer.calendar
        )
        keys |= plan.chain_keys
    for ticker in events["ticker"].astype(str).unique():
        try:
            newest = replay_mod.latest_chain_date(ticker, as_of)
        except Exception:  # pragma: no cover - store-dependent
            newest = None
        if newest is not None:
            keys.add((ticker, pd.Timestamp(newest).normalize()))
    index = replay_mod.load_chain_index(keys, progress_every=0) if keys else None

    out: list[dict] = []
    for row in events.itertuples(index=False):
        for strategy in strategy_names:
            request = score_mod.ScoreRequest(
                ticker=str(row.ticker), strategy=strategy, as_of=None,
                event_date=pd.Timestamp(row.event_date), session=str(row.session),
                fill=MID, quote_max_age_sessions=quote_max_age, chain_as_of=as_of,
            )
            raw, record, took, trace = _score(scorer, request, index=index)
            candidate = _candidate(
                request, raw, record, took, frame="forward", legacy_trace=trace,
            )
            candidate["event_id"] = str(row.event_id)
            out.append(candidate)
    return out


def boundary_pass(scorer, events: pd.DataFrame,
                  strategies: tuple[str, ...] | None = None) -> list[dict]:
    """Past events, scored at their own decision close, for the two boundaries.

    ``as_of`` is the structure's DECISION date, resolved through the calendar,
    not the print date. Scoring a BMO print as of the print itself is a leak —
    ``engine.audit`` refuses it. These are also where the priced:S axes are
    won: in the forward window the forecast-sized families come back
    NO_FORECAST with empty legs.
    """
    out: list[dict] = []
    for strategy in _score_strategies(strategies):
        structure = STRUCTURES[strategy]()
        plan = replay_mod.plan_events(structure, events, calendar=scorer.calendar)
        for row in plan.frame.to_dict("records"):
            as_of = pd.Timestamp(row["decision_date"])
            request = score_mod.ScoreRequest(
                ticker=str(row["ticker"]), strategy=strategy, as_of=as_of,
                event_date=pd.Timestamp(row["event_date"]),
                session=str(row["session"]), fill=MID, chain_as_of=as_of,
            )
            try:
                raw, record, took, trace = _score(scorer, request)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(f"[corpus]   skipped {row['ticker']} {strategy}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            candidate = _candidate(
                request, raw, record, took, frame="boundary", legacy_trace=trace,
            )
            candidate["event_id"] = str(row["event_id"])
            out.append(candidate)
    return out


def _rescore(scorer, source: dict, label: str, **changes) -> dict | None:
    """Re-score a captured request with some fields changed — same clock.

    The changed request inherits the source's ``as_of``, ``chain_as_of`` and
    quote-age policy, so a pinned or strike variant of a historical row is
    scored at that row's decision date rather than today's.
    """
    request = replace(request_from_dict(source["request"]), **changes)
    try:
        raw, record, took, trace = _score(scorer, request)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        print(f"[corpus]   {label} skip {request.ticker} {request.strategy}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None
    candidate = _candidate(request, raw, record, took, legacy_trace=trace)
    if source.get("event_id") is not None:
        candidate["event_id"] = source["event_id"]
    return candidate


def _anchor_strike(record: dict) -> float | None:
    """A strike the row's legs actually resolved to — a LISTED strike."""
    legs = [leg for leg in record.get("legs") or [] if isinstance(leg, dict)]
    anchor = next((leg for leg in legs if leg.get("name") == "atm"), legs[0] if legs else None)
    strike = (anchor or {}).get("strike")
    return float(strike) if isinstance(strike, (int, float)) else None


def pinned_and_strike_pass(scorer, scored: list[dict], limit: int = 8) -> list[dict]:
    """Re-score priced rows with their geometry pinned, and at a listed strike.

    The pinned pair is the `e845f3e` regression case made permanent: a replay
    that pins the shape must still record the forecast that chose it. It is
    evidence only beside the selector-resolved row it was pinned FROM, so the
    relation is recorded and :func:`select` keeps the source.

    The strike pair asks for a strike the source's legs resolved to — a real
    listed strike, not a computed moneyness the chain would snap away from.
    """
    out: list[dict] = []
    for source in scored:
        record = source["record"]
        params = record.get("structure_params")
        if not priced(record) or not isinstance(params, dict) or not params:
            continue
        pinned = _rescore(scorer, source, "pinned",
                          structure_params={k: v for k, v in params.items() if v is not None})
        if pinned is not None:
            pinned["relations"] = {"pinned_from": content_hash(source["request"])}
            out.append(pinned)
        listed = _anchor_strike(record)
        at_strike = (_rescore(scorer, source, "strike", strike=listed)
                     if listed is not None else None)
        if at_strike is not None:
            out.append(at_strike)
        if len(out) >= limit:
            break
    return out


#: Widths tried, narrowest first, when hunting a COARSE_LADDER refusal.
_COARSE_WIDTHS = (0.002, 0.004, 0.006, 0.01)


def coarse_ladder_pass(scorer, scored: list[dict]) -> list[dict]:
    """Ask for a width the ticker's listed ladder cannot carry.

    §7.1 requires a coarse ladder and it cannot be waited for, so it is
    *requested* — a real ``structure_params`` width, through the real entry
    point, narrow enough that two legs resolve onto one contract. Asking for a
    refusal is not the same as inventing one.
    """
    for source in scored:
        record = source["record"]
        if record.get("spot") is None or record.get("strategy") not in ("TWIN-P5", "TWIN-P"):
            continue
        for width in _COARSE_WIDTHS:
            got = _rescore(scorer, source, "coarse",
                           structure_params={"width_moneyness": width})
            if got is None:
                break
            if "COARSE_LADDER" in (got["record"].get("flags") or []):
                return [got]
    return []


def _event_key(record: dict) -> tuple:
    return (record.get("ticker"), record.get("event_date"))


def dyn_sv_pass(candidates: list[dict]) -> list[dict]:
    """The chooser, run per event over a BOARD-SHAPED frame.

    Only rows the board's scoring loop produces — one per structure per event,
    at the ATM pass — enter the frame. The first corpus fed the chooser every
    candidate, pinned copies included, and its only "tie" was BFLY-P tying with
    its own pinned re-score. The event's rows are frozen in frame order inside
    the request: ``dynamic_short_vol`` breaks a tie by that order, so a replay
    that did not reproduce it would not reproduce the choice.
    """
    out: list[dict] = []
    for frame_name in ("forward", "boundary"):
        members = [c for c in candidates if c.get("frame") == frame_name]
        events: dict[tuple, list[dict]] = {}
        for cand in members:
            events.setdefault(_event_key(cand["record"]), []).append(cand)
        for key, siblings in events.items():
            frame = pd.DataFrame([c["raw"] | {"strike_offset": None} for c in siblings])
            chosen = score_mod.dynamic_short_vol(frame)
            if chosen.empty:
                continue
            request = {
                "kind": "dyn_sv_resolution",
                "entry_point": "engine.score.dynamic_short_vol",
                "menu": list(score_mod.DYNAMIC_MENU),
                "frame": frame_name,
                "frame_rows": [{"request": c["request"], "record": c["record"]}
                               for c in siblings],
            }
            record = jsonable(chosen.iloc[0].to_dict())
            out.append(_candidate(request, None, record, 0.0, kind="dyn_sv_choice"))
    return out


def research_replay_pass(scorer, events: pd.DataFrame, limit: int = 2,
                         strategies: tuple[str, ...] | None = None) -> list[dict]:
    """Price CAL-P and CND-P under research, where the scorer refuses them.

    §7.1 requires both to appear as *refusals* on the production path and to
    *replay* under research. That is a different entry point with a different
    record, and conflating the two would lose the distinction.
    """
    out: list[dict] = []
    for strategy in score_mod.DISABLED_STRATEGIES:
        if strategies is not None and strategy not in strategies:
            continue
        structure = STRUCTURES[strategy]()
        plan = replay_mod.plan_events(structure, events, calendar=scorer.calendar)
        index = replay_mod.load_chain_index(plan.chain_keys, progress_every=0)
        taken = 0
        for row in plan.frame.to_dict("records"):
            started = time.monotonic()
            rows, skip = replay_mod.replay_one(structure, row, index,
                                               include_legs=True)
            if not rows:
                continue
            request = {
                "kind": "research_replay",
                "entry_point": "engine.replay.replay_one",
                "strategy": strategy,
                "structure": structure.to_dict(),
                "plan_row": jsonable(row),
            }
            record = {"rows": jsonable(rows), "skip_reason": skip, "strategy": strategy}
            out.append(_candidate(request, None, record, time.monotonic() - started,
                                  kind="research_replay"))
            taken += 1
            if taken >= limit:
                break
    return out


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def select(candidates: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """A minimal covering subset, greedily, plus the axis -> fixtures index.

    Greedy rather than exhaustive: what matters is that every axis is covered
    by *some* frozen pair, not that the subset is provably the smallest. After
    the greedy pass every chosen pinned fixture pulls in the pair it was
    pinned from — without it the pinned pair demonstrates nothing.
    """
    inputs = axis_inputs()
    for cand in candidates:
        cand["covers"] = derive_covers(cand["record"], cand["request"], cand["kind"],
                                       inputs, cand.get("relations"))

    wanted = set(required_axes())
    chosen: list[dict] = []
    remaining = list(candidates)
    while wanted and remaining:
        remaining.sort(key=lambda c: -len(wanted & set(c["covers"])))
        best = remaining.pop(0)
        gain = wanted & set(best["covers"])
        if not gain:
            break
        chosen.append(best)
        wanted -= gain

    by_request = {content_hash(c["request"]): c for c in candidates}
    chosen_hashes = {content_hash(c["request"]) for c in chosen}
    for cand in list(chosen):
        source = (cand.get("relations") or {}).get("pinned_from")
        if source and source not in chosen_hashes and source in by_request:
            chosen.append(by_request[source])
            chosen_hashes.add(source)

    index: dict[str, list[str]] = {}
    for i, cand in enumerate(chosen):
        cand["fixture_id"] = _fixture_id(cand, i)
        for axis in cand["covers"]:
            index.setdefault(axis, []).append(cand["fixture_id"])
    return chosen, index


def _fixture_id(cand: dict, i: int) -> str:
    record = cand["record"]
    stem = "-".join(str(x) for x in (
        record.get("strategy", cand["kind"]),
        record.get("ticker", ""),
        record.get("event_date", ""),
    ) if x)
    digest = content_hash(cand["request"])[7:15]
    return f"{i:03d}_{stem}_{digest}".replace("/", "-").replace(" ", "")


def attach_strict_probe(
    chosen: list[dict], snapshot: str, release_root: Path,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Attach a strict native trace to every eligible selected score row.

    Every ``score_result`` candidate is executed independently, row by row.
    A row that cannot produce an honest trace (unsupported strategy, a
    missing or hash-mismatched checkpoint, an unresolvable feature or
    binding, ...) is never fabricated and never allowed to abort rows that
    DID assemble cleanly: its typed reason is recorded in the returned
    ``gaps`` map (fixture_id -> reason) instead, and the caller persists it
    on that pair as ``trace_disposition: "gap"``. The whole capture only
    refuses when NOT ONE row produced a verified trace — that is a wiring
    failure (nothing works at all), not a per-case gap.
    """
    gaps: dict[str, str] = {}
    attached = []
    for candidate in chosen:
        if candidate.get("kind") != "score_result":
            continue
        fixture_id = str(candidate.get("fixture_id", "candidate"))
        try:
            request = canonical_v2_request(candidate, snapshot)
            source = _checkpoint_value(candidate, "source_inputs")
            bindings = source.get("model_bindings") or ()
            package = None
            if bindings:
                package = package_frozen_resources(
                    model_bindings=bindings,
                    deployment_id=request.deployment_id,
                    release_root=release_root,
                    source_root=ROOT,
                )
                request = replace(request, model_artifact_refs=package.request_refs)
            inputs, shared_inputs = native_inputs_from_capture(candidate, request)
            runtime = (
                _frozen_runtime(package, release_root, request, inputs, candidate)
                if package else None
            )
            resources = list(package.resource_rows) if package else []
            if package:
                resources.extend({
                    "resource_id": binding["binding_id"],
                    "ref": binding["request_ref"],
                    "kind": "sidecar",
                    "document": binding,
                    "content_hash": content_hash(binding),
                } for binding in package.sidecar_document["bindings"])
            trace, native = package_strict_trace(
                request, inputs, shared_inputs,
                resources=resources,
                metadata={
                    "capture_mode": "bounded-strict-probe",
                    "legacy_checkpoint_hash": content_hash(
                        candidate["legacy_trace"]
                    ),
                    "frozen_inference": package.trace_declaration if package else None,
                },
                frozen_runtime=runtime,
            )
        except (StrictTraceCaptureError, TypeError, ValueError) as exc:
            gaps[fixture_id] = str(exc)
            continue
        candidate["request"] = to_document(request)
        candidate["input_trace"] = trace
        candidate["legacy_input_hash"] = trace["shared_input_hash"]
        candidate["native_score_id"] = native.score_id
        attached.append(fixture_id)
    if not attached:
        detail = "; ".join(f"{k}: {v}" for k, v in list(gaps.items())[:3])
        detail = detail or "no score_result candidate was selected"
        raise StrictTraceCaptureError(
            f"no honest strict trace could be assembled for any candidate: {detail}"
        )
    return tuple(attached), gaps


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _publish_current(root: Path, version: str) -> None:
    """Point ``CURRENT`` at a version directory, atomically."""
    tmp = root / "CURRENT.tmp"
    tmp.write_text(json.dumps({"version": version}, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, root / "CURRENT")


def write(out_dir: Path, chosen: list[dict], index: dict[str, list[str]],
          as_of: pd.Timestamp, snapshot: str, *, replace_existing: bool = False,
          strict_trace: bool = False) -> dict:
    """Publish one immutable version directory, atomically.

    The version is built under a temporary sibling and published with one
    rename; an existing non-empty version directory refuses without an
    explicit ``--replace``.
    """
    if out_dir.exists() and any(out_dir.iterdir()) and not replace_existing:
        raise SystemExit(
            f"{out_dir} already exists and is not empty. A frozen corpus is "
            "never overwritten in place: capture a NEW version directory, or "
            "re-run with --replace to authorize replacing this one explicitly.")
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    pairs_dir = tmp / "pairs"
    pairs_dir.mkdir(parents=True)
    strict_gaps: dict[str, str] = {}
    if strict_trace:
        strict_ids, strict_gaps = attach_strict_probe(chosen, snapshot, tmp)
        print(f"[corpus] strict Phase 4 traces: {len(strict_ids)} attached "
              f"({', '.join(strict_ids)})", flush=True)
        if strict_gaps:
            print(f"[corpus] strict Phase 4 trace gaps (recorded, never "
                  f"faked): {len(strict_gaps)}", flush=True)
            for fixture_id, reason in strict_gaps.items():
                print(f"    {fixture_id}: {reason}", flush=True)
    checkpoint_sink = DiskCheckpointSink(tmp / "checkpoints")

    manifest_pairs = {}
    for cand in chosen:
        pair = make_pair(
            cand["fixture_id"], cand["covers"], cand["request"], cand["record"],
            record_kind=cand["kind"], duration=cand["duration"],
            legacy_trace=cand.get("legacy_trace"),
            input_trace=cand.get("input_trace"),
            legacy_input_hash=cand.get("legacy_input_hash"),
            strict_trace_gap=strict_gaps.get(str(cand["fixture_id"])),
            relations=cand.get("relations"),
        )
        text = json.dumps(pair, indent=2, sort_keys=True) + "\n"
        (pairs_dir / f"{cand['fixture_id']}.json").write_text(text)
        manifest_pairs[cand["fixture_id"]] = {
            "payload_hash": pair["payload_hash"],
            "request_hash": pair["request_hash"],
            "record_kind": cand["kind"],
            "covers": pair["covers"],
            "trace_disposition": pair["payload"].get("trace_disposition", "absent"),
        }
        checkpoint = cand.get("legacy_trace")
        if checkpoint is not None:
            checkpoint_sink.write_case(
                cand["fixture_id"],
                {
                    "case_id": cand["fixture_id"],
                    "request": pair["payload"]["request"],
                    "strategy": pair["payload"]["record"].get("strategy"),
                    "covers": pair["covers"],
                    "record_kind": cand["kind"],
                    "checkpoint": checkpoint,
                },
            )

    missing = sorted(set(required_axes()) - set(index))
    doc = {
        "schema_version": INDEX_VERSION,
        "as_of": str(as_of.date()),
        "snapshot": snapshot,
        "tier": 0,
        "stage_plan_ref": "scorer.v1",
        "tolerance_policy_ref": "score_record.exact.v1",
        "refusal_code_mapping": REFUSAL_CODES,
        # Everything checks/tier0_corpus.py needs to RE-DERIVE coverage from
        # the surviving records alone, with no engine import.
        "axis_inputs": axis_inputs(),
        "pairs": manifest_pairs,
        "coverage": {axis: sorted(ids) for axis, ids in sorted(index.items())},
        "required_axes": required_axes(),
        "uncovered_axes": missing,
        "corpus_hash": content_hash(
            {k: v["payload_hash"] for k, v in sorted(manifest_pairs.items())}
        ),
    }
    checkpoint_sink.finalize({
        "release_id": out_dir.name,
        "source_snapshot": snapshot,
        "coverage": doc["coverage"],
        "status": "diagnostic_only",
    })
    doc["diagnostic_checkpoint_manifest"] = "checkpoints/manifest.json"
    (tmp / "INDEX.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.rename(tmp, out_dir)
    return doc


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None,
                    help="write exactly here (skips the versioned layout)")
    ap.add_argument("--version", default=None,
                    help="version directory name under fixtures/tier0 "
                         "(default: UTC timestamp)")
    ap.add_argument("--replace", action="store_true",
                    help="authorize replacing an existing non-empty version")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--forward-days", type=int, default=35)
    ap.add_argument("--max-events", type=int, default=40)
    ap.add_argument("--boundary-events", type=int, default=4)
    ap.add_argument("--quote-max-age", type=int, default=5)
    ap.add_argument(
        "--strategies", nargs="+", default=None,
        help="capture only these strategies (space- or comma-separated); "
             "default preserves the current all-strategy capture",
    )
    ap.add_argument(
        "--strict-phase4-trace", action="store_true",
        help="attach a verified native input_trace to every captured "
             f"score_result row whose strategy currently supports one "
             f"({', '.join(sorted(STRICT_TRACE_SUPPORTED_STRATEGIES))}); a "
             "row that cannot is recorded with a typed gap, never faked",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    try:
        strategies = parse_strategies(args.strategies)
    except StrictTraceCaptureError as exc:
        ap.error(str(exc))
    as_of = (pd.Timestamp(args.as_of).normalize() if args.as_of
             else pd.Timestamp.today().normalize())
    started = time.time()
    print("[corpus] building the scorer (panel + replayed trades)...", flush=True)
    scorer = score_mod.Scorer()
    print(f"[corpus] scorer ready in {time.time()-started:.0f}s", flush=True)

    # `_candidate()` spills every candidate's `legacy_trace` to disk the
    # moment it is produced (see the module note above `_score`) rather than
    # holding it in `candidates` for the rest of this function; the `finally`
    # below removes that spill directory whether the run finishes or raises.
    try:
        forward = _events(as_of, args.forward_days, args.max_events)
        print(f"[corpus] forward events: {len(forward)}", flush=True)
        candidates = forward_pass(
            scorer, forward, as_of, args.quote_max_age, strategies,
        )
        print(f"[corpus] forward scores: {len(candidates)}", flush=True)

        boundaries = _boundary_events(as_of, args.boundary_events, scorer.calendar)
        print(f"[corpus] boundary events: {len(boundaries)}", flush=True)
        candidates += boundary_pass(scorer, boundaries, strategies)

        candidates += pinned_and_strike_pass(scorer, candidates)
        candidates += coarse_ladder_pass(scorer, candidates)
        if strategies is None or score_mod.DYNAMIC_STRATEGY in strategies:
            candidates += dyn_sv_pass(candidates)
        candidates += research_replay_pass(
            scorer, boundaries, strategies=strategies,
        )
        print(f"[corpus] candidates: {len(candidates)}", flush=True)

        chosen, index = select(candidates)
        # Only `chosen` -- `select()`'s small covering subset, never
        # `candidates` itself -- needs its real `legacy_trace` content back;
        # everything from here on (`attach_strict_probe`, `write`) reads it
        # directly off `cand`. Every OTHER candidate's checkpoint content
        # stays on disk, unread, for the rest of the run.
        _shared_hydration_cache: dict[str, Any] = {}
        for cand in chosen:
            trace = _hydrate_trace(cand.get("legacy_trace"))
            _reconcile_shared_trace_content(trace, _shared_hydration_cache)
            cand["legacy_trace"] = trace
        if args.out:
            out_dir = Path(args.out)
        else:
            version = args.version or datetime.now(
                timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_dir = DEFAULT_OUT / version
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        doc = write(
            out_dir, chosen, index, as_of, scorer.snapshot,
            replace_existing=args.replace,
            strict_trace=args.strict_phase4_trace,
        )
    finally:
        _cleanup_trace_spill()
    if not args.out and out_dir.parent == DEFAULT_OUT:
        _publish_current(DEFAULT_OUT, out_dir.name)
        print(f"[corpus] CURRENT -> {out_dir.name}")

    print(f"[corpus] wrote {len(chosen)} pairs to {out_dir}")
    print(f"[corpus] corpus hash {doc['corpus_hash']}")
    total = len(doc["required_axes"])
    print(f"[corpus] coverage {total - len(doc['uncovered_axes'])}/{total} required axes")
    if doc["uncovered_axes"]:
        print("[corpus] UNCOVERED (recorded as gaps, not faked):")
        for axis in doc["uncovered_axes"]:
            print(f"    {axis}")
    print(f"[corpus] total {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
