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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: The CODE checkout (imports, default output). Model artifacts live under
#: the DATA root instead (``_artifact_source_root``); from a worktree the two
#: differ.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _artifact_source_root() -> Path:
    """Where captured model artifacts are resolved: ``engine.paths.ROOT``
    (honours ``INVESTING_PLAN_ROOT``), the root the registry and the Tier-4
    serving caches build their paths from -- not this file's checkout."""
    from engine import paths

    return Path(paths.ROOT)

from checks.phase4_frozen_bridge import (  # noqa: E402
    FROZEN_CHOOSER_FIELD,
    FROZEN_CHOOSER_SCHEMA,
    prepare_frozen_chooser,
    with_frozen_chooser,
)
from checks.tier0_corpus import derive_covers, priced  # noqa: E402
from engine import replay as replay_mod  # noqa: E402
from engine import score as score_mod  # noqa: E402
from engine.data import store  # noqa: E402
from engine.fills import MID  # noqa: E402
from engine.structures import STRUCTURES  # noqa: E402
from engine.v2.contracts import ScoreRequest as V2ScoreRequest  # noqa: E402
from engine.v2.diagnosis import content_hash  # noqa: E402
from engine.v2.foundation import to_document  # noqa: E402
from engine.v2.foundation.canonical import tag_nonfinite  # noqa: E402
from engine.v2.models import (  # noqa: E402
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
)
from engine.v2.models.contracts import ArtifactMember  # noqa: E402
from engine.v2.scoring import application as v2_application  # noqa: E402
from engine.v2.scoring.stages import (  # noqa: E402
    NativeScoreInputs,
    StageObservation,
    receipt,
)
from tools.phase4_checkpoint_sink import DiskCheckpointSink  # noqa: E402
from tools.phase4_frozen_resources import (  # noqa: E402
    FrozenResourcePackage,
    package_frozen_resources,
)
from tools.phase4_release_assembler import assemble_input_trace  # noqa: E402
from tools.phase4_request_translation import (  # noqa: E402
    LegacyRequestTranslationError,
    canonical_request_from_legacy,
)

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

#: The chooser's 17 primitive inputs (legacy ``Scorer._chooser_frame``).
CHOOSER_PRIMITIVE_COLUMNS = tuple(score_mod.Scorer._CHOOSER_PRIMITIVES)

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
#: production gate) are excluded here: legacy refuses them before any stage,
#: so the only strict trace a disabled ``score_result`` row can carry is the
#: minimal one built from its request-only source bundle
#: (:data:`DISABLED_REQUEST_ONLY_STRATEGIES`). The DYN-SV chooser
#: meta-strategy is captured as its own ``dyn_sv_choice`` kind.
STRICT_TRACE_SUPPORTED_STRATEGIES = (
    frozenset(STRUCTURES) - frozenset(score_mod.DISABLED_STRATEGIES)
)
#: Disabled strategies: traced only from a request-only source bundle
#: (``engine.score.Phase4TraceCollector.capture_request_only_bundle``), so
#: native's own refusal is compared with legacy's UNVALIDATED_STRUCTURE.
DISABLED_REQUEST_ONLY_STRATEGIES = frozenset(score_mod.DISABLED_STRATEGIES)


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
    """Translate source-owned legacy request identity into a canonical command.

    The translation itself lives in :mod:`tools.phase4_request_translation`,
    shared with ``checks/phase4_real.py``, which re-derives it from the saved
    legacy request to bind the traced V2 request to its pair.
    """
    event_id = candidate.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise StrictTraceCaptureError("strict trace requires the captured event_id")
    raw = candidate.get("request")
    if not isinstance(raw, Mapping):
        raw = request_to_dict(raw)
    legacy = request_to_dict(request_from_dict(dict(raw)))
    request_only = (legacy["strategy"] in DISABLED_REQUEST_ONLY_STRATEGIES
                    and _request_only_source(candidate) is not None)
    if legacy["strategy"] not in STRICT_TRACE_SUPPORTED_STRATEGIES and not request_only:
        raise StrictTraceCaptureError(
            f"strict probe does not support {legacy['strategy']} "
            f"(supported: {sorted(STRICT_TRACE_SUPPORTED_STRATEGIES)})"
        )
    try:
        return canonical_request_from_legacy(legacy, event_id=event_id, snapshot=snapshot)
    except LegacyRequestTranslationError as exc:
        raise StrictTraceCaptureError(str(exc)) from exc


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


#: ``source_inputs.quote_status`` values under which legacy recorded NO quote
#: domain on purpose (``engine.score.Phase4TraceCollector.QUOTE_STATUSES``).
_EMPTY_QUOTE_STATUSES = frozenset({"empty", "not_reached"})


def _request_only_source(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The request-only source bundle of a row legacy refused before any
    stage, or ``None``. It must be the trace's ONLY checkpoint group, carry
    no quotes and no recipes, and say the lookup was never reached;
    anything else is not a request-only row and is refused, not repaired."""
    trace = candidate.get("legacy_trace")
    checkpoints = trace.get("checkpoints") if isinstance(trace, Mapping) else None
    if not isinstance(checkpoints, Mapping) or "source_inputs" not in checkpoints:
        return None
    source = _checkpoint_value(candidate, "source_inputs")
    if source.get("scope") != "request_only":
        return None
    if set(checkpoints) != {"source_inputs"}:
        raise StrictTraceCaptureError(
            "request-only source bundle alongside other checkpoint groups "
            f"{sorted(set(checkpoints) - {'source_inputs'})}"
        )
    if (source.get("quote_status") != "not_reached" or source.get("quote_domain")
            or source.get("model_bindings") or source.get("native_recipes")
            or source.get("features")):
        raise StrictTraceCaptureError(
            "request-only source bundle carries more than the request"
        )
    return source


def _request_only_inputs(
    source: Mapping[str, Any], request: V2ScoreRequest,
) -> dict[str, Any]:
    """Native blocks for a request-only row: the request's own facts, no
    quotes, no model inputs, no recipes. Native refuses from these alone."""
    context = dict(source.get("context") or {})
    missing = sorted(key for key in ("ticker", "strategy") if not context.get(key))
    if missing:
        raise StrictTraceCaptureError(f"source_inputs context missing {missing}")
    if context["strategy"] != request.strategy_version:
        raise StrictTraceCaptureError("request-only bundle names another strategy")
    context["quotes"] = {}
    return {
        "context": context,
        "features": {"model_inputs": {}, "source_features": {}},
        "forecast": {},
        "geometry": None,
        "pricing": None,
        "analogs": {"mode": "not_applicable"},
        "simulation": {"mode": "not_applicable"},
        "gate": {"mode": "not_applicable"},
        "chooser": {},
        "diagnostics": {},
    }


def _quote_map(rows: Any, quote_status: Any = None) -> dict[str, dict[str, float]]:
    """The native quote map. An empty domain is accepted only when legacy
    recorded WHY it is empty (lookup found no chain, or the row never reached
    pricing); an empty domain with no recorded status was never captured."""
    if quote_status in _EMPTY_QUOTE_STATUSES:
        if rows != []:
            raise StrictTraceCaptureError(
                f"quote_status {quote_status} but quote_domain is not empty"
            )
        return {}
    if quote_status not in (None, "recorded", "priced"):
        raise StrictTraceCaptureError(f"unknown quote_status {quote_status!r}")
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
            # Only the 17 primitive columns (R4-20 gap (a)): native derives
            # the other 50, and a declared derived column would override the
            # native derivation (the compatibility path). The regime values
            # are the ones `_chooser_frame` filled through `_regime_extra`.
            # A missing (None/non-finite) primitive is left out: no binding
            # is fed this vector, and native declines the chooser on a
            # missing column exactly as legacy declines on a NaN one.
            _add("chooser", {
                name: chooser_vector[name]
                for name in CHOOSER_PRIMITIVE_COLUMNS
                if name in chooser_vector
                and _finite_or_none(chooser_vector[name]) is not None
            })

    return vectors


def _finite_or_none(raw: Any) -> float | None:
    """``raw`` as a finite float, ``None`` when missing or non-finite."""
    from engine.v2.foundation.canonical import untag_nonfinite

    raw = untag_nonfinite(raw)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise StrictTraceCaptureError(f"chooser input {raw!r} is nonnumeric") from exc
    return value if math.isfinite(value) else None


def _frozen_block(candidate: Mapping[str, Any]) -> dict[str, Any]:
    from engine.v2.foundation.canonical import untag_nonfinite

    source = _checkpoint_value(candidate, "source_inputs")
    return untag_nonfinite(dict(source.get("frozen") or {}))


def _chooser_consumed_rows(candidate: Mapping[str, Any]) -> dict[str, float | None]:
    """Every column the legacy chooser consumed: its 17 primitives and the
    rows it fed the Tier-4 producer folds (``<slot>@chooser``). ``None`` is a
    missing value. A column two of those recorded differently is refused, as
    ``frozen_source_declarations`` refuses it."""
    frozen = _frozen_block(candidate)
    if frozen.get("conflicts"):
        raise StrictTraceCaptureError(
            f"legacy recorded different values for {sorted(frozen['conflicts'])}")
    declared = dict(frozen.get("declarations") or {})
    inputs = dict(frozen.get("inputs") or {})
    rows: dict[str, float | None] = {}

    def feature(name: str, raw: Any, where: str) -> None:
        value = _finite_or_none(raw)
        if name in rows and rows[name] != value:
            raise StrictTraceCaptureError(f"chooser feature {name} differs at {where}")
        rows[name] = value

    for key in sorted(declared):
        entry = declared[key]
        if not key.startswith("chooser_fold:") or not isinstance(entry, Mapping):
            continue
        if "site" not in entry:
            continue
        slot = f"{entry['binding']}@{entry['site']}"
        if slot not in inputs:
            raise StrictTraceCaptureError(f"no recorded input row {slot}")
        for name, raw in sorted(inputs[slot].items()):
            feature(str(name), raw, slot)
    for name, raw in (declared.get("chooser_primitives") or {}).items():
        feature(str(name), raw, "chooser_primitives")
    return rows


def _merge_chooser_rows(model_inputs: dict[str, float],
                        rows: Mapping[str, float | None]) -> None:
    """Add the chooser's consumed columns to ``model_inputs`` (the native
    chooser reads its features from there, as a ``SourceBundle``'s
    ``feature_vector``). A missing column stays absent; a column the forecast
    roles hold at another value (or hold where the chooser saw none) is one
    feature vector legacy never had, and is refused."""
    for name, value in rows.items():
        held = model_inputs.get(name)
        if name in model_inputs and held != value:
            raise StrictTraceCaptureError(
                f"feature {name} differs between the model roles and the chooser")
        if value is not None:
            model_inputs[name] = value


def native_inputs_from_capture(
    candidate: Mapping[str, Any],
    request: V2ScoreRequest,
    *,
    frozen_chooser: Mapping[str, Any] | None = None,
) -> tuple[NativeScoreInputs, dict[str, Any]]:
    """Build executable native inputs only from source-owned captured material.

    ``frozen_chooser``: the row's frozen chooser declaration
    (:func:`frozen_chooser_declaration`); it becomes the chooser block, and the
    rows the chooser consumed join ``model_inputs``.
    """
    request_only = _request_only_source(candidate)
    if request_only is not None:
        if request.strategy_version not in DISABLED_REQUEST_ONLY_STRATEGIES:
            raise StrictTraceCaptureError(
                f"request-only source bundle for enabled {request.strategy_version}"
            )
        blocks = _request_only_inputs(request_only, request)
    else:
        blocks = _captured_blocks(candidate, request, frozen_chooser)
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


def entry_rule_gate_block(candidate: Mapping[str, Any], strategy: str,
                          event_date: Any) -> dict[str, Any] | None:
    """The native entry-rule gate block for a row legacy gated by a rule.

    Legacy ``Scorer._apply_entry_rule`` records ``gate_inputs`` with
    ``kind == "entry_rule"`` and the facts it evaluated. Two of them are
    declared here, neither an answer: ``mcap_usd`` (market state; native has
    no other source for it) and the trailing ``pnl_cutoff`` bar legacy served,
    frozen as a :class:`TrailingCutoffArtifact` for the event's month, the key
    legacy's ``pnl_sim.trailing_cutoff(history, event_date)`` computes it on
    (no bar recorded = ``cutoff=None``, as legacy served none). The other
    facts are derived natively: ``exp_pnl_sim`` by the simulation stage and
    ``rel_spread`` from the priced legs; ``cost``/``w``/``peak`` are read by
    no live rule. ``None`` when the row reached no entry rule.
    """
    from engine.v2.foundation.canonical import untag_nonfinite
    from engine.v2.models.trailing_cutoff_artifact import (
        cutoff_month,
        make_trailing_cutoff_artifact,
    )
    from engine.v2.models.training.trailing_cutoff import trailing_cutoff_lineage
    from engine.v2.scoring.native_entry_rule import entry_rule_block

    gate = _checkpoint_value_optional(candidate, "gate_inputs")
    if not isinstance(gate, Mapping) or gate.get("kind") != "entry_rule":
        return None
    if gate.get("rule_identity") != f"entry-rule:{strategy}":
        raise StrictTraceCaptureError(
            f"entry rule {gate.get('rule_identity')!r} recorded for {strategy}")
    facts = untag_nonfinite(dict(gate.get("facts") or {}))
    if event_date is None:
        raise StrictTraceCaptureError("entry rule row has no event_date")
    month = cutoff_month(event_date)
    cutoff = make_trailing_cutoff_artifact(
        month=month, cutoff=_finite_or_none(facts.get("pnl_cutoff")),
        lineage=trailing_cutoff_lineage(month))
    return entry_rule_block(strategy, mcap_usd=_finite_or_none(facts.get("mcap_usd")),
                            cutoff=cutoff)


def _captured_blocks(candidate: Mapping[str, Any],
                     request: V2ScoreRequest,
                     frozen_chooser: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Native blocks from a scored row's captured source bundle."""
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
    quote_status = source.get("quote_status")
    context["quotes"] = _quote_map(source.get("quote_domain"), quote_status)
    # A spot exists only once legacy priced the structure. A row whose chain
    # lookup came back empty, that never reached pricing, or whose pricer
    # raised has none, and native refuses it with its own code; a capture
    # that predates quote_status keeps the old requirement.
    required = ["ticker", "event_date", "entry_date", "exit_date"]
    if quote_status in (None, "priced"):
        required.append("spot")
    missing_context = sorted(key for key in required if context.get(key) is None)
    if missing_context:
        raise StrictTraceCaptureError(
            f"source_inputs context missing {missing_context}"
        )
    recipes = dict(recipes)
    entry_rule = entry_rule_gate_block(candidate, request.strategy_version,
                                       context.get("event_date"))
    if entry_rule is not None:
        if "gate" in recipes:
            raise StrictTraceCaptureError(
                "gate_inputs recorded an entry rule and a model gate recipe")
        recipes["gate"] = entry_rule
    recipes.setdefault("analogs", {"mode": "not_applicable"})
    recipes.setdefault("simulation", {"mode": "not_applicable"})
    recipes.setdefault("gate", {"mode": "not_applicable"})
    model_inputs = _merged_model_inputs(candidate)
    if frozen_chooser is not None:
        _merge_chooser_rows(model_inputs, _chooser_consumed_rows(candidate))
    features = {
        "model_inputs": model_inputs,
        "source_features": source_features,
    }
    if source.get("model_bindings"):
        # The row each frozen binding is fed, per role (the gate's own
        # vector is not in the merged ``model_inputs``). Kept in the
        # source-bound features block so the replay
        # (checks/phase4_frozen_bridge.py) feeds every binding this row.
        features["role_model_inputs"] = _role_feature_vectors(candidate)
    return {
        "context": context,
        "features": features,
        "forecast": dict(recipes["forecast"]),
        "geometry": None,
        "pricing": None,
        "analogs": dict(recipes["analogs"]),
        "simulation": dict(recipes["simulation"]),
        "gate": dict(recipes["gate"]),
        "chooser": (
            {FROZEN_CHOOSER_FIELD: dict(frozen_chooser)} if frozen_chooser is not None
            else dict(recipes.get("chooser") or {})
        ),
        "diagnostics": dict(recipes.get("diagnostics") or {}),
    }


def package_strict_trace(
    request: V2ScoreRequest,
    inputs: NativeScoreInputs,
    shared_inputs: Mapping[str, Any],
    *,
    resources: list[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    frozen_runtime: tuple[FrozenInference, ModelRelease, tuple[InferenceRequest, ...]] | None = None,
    execution_inputs: NativeScoreInputs | None = None,
) -> tuple[dict[str, Any], Any]:
    """Execute native scoring and package the observer output for verification.

    ``execution_inputs``: what actually runs when it differs from the
    declared ``inputs`` the trace stores (a frozen chooser declaration
    resolved into its executable block, ``with_frozen_chooser``).
    """
    observations = []
    declared_inputs = inputs
    if execution_inputs is not None:
        inputs = execution_inputs
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
    inputs = declared_inputs
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
    release = _package_release(package, request.deployment_id)
    bindings = release.bindings
    # Each binding's row must come from THAT binding's own captured per-role
    # feature vector, never the cross-role merged dict
    # (`inputs.features["model_inputs"]`): a binding's feature_order can name
    # features private to its own role (e.g. the gate model's own vector),
    # which the merge — built only from the forecast-family `features`
    # checkpoint — never carries. See `_role_feature_vectors`.
    role_vectors = inputs.features.get("role_model_inputs")
    if not isinstance(role_vectors, Mapping):
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


def _package_release(package: FrozenResourcePackage, deployment_id: str) -> ModelRelease:
    """The ``ModelRelease`` a frozen resource package's sidecar describes."""
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
    return ModelRelease(
        release_id=package.sidecar_document["release_id"],
        deployment_id=deployment_id,
        bindings=tuple(bindings),
    )


def frozen_source_declarations(
    candidate: Mapping[str, Any],
    *,
    release_root: Path,
    deployment_id: str,
    source_root: Path | None = None,
    chooser_analog_pool: Any = None,
    release_states: Sequence[Any] = (),
) -> dict[str, Any]:
    """``SourceBundle`` keyword arguments from a capture's ``source_inputs.frozen``.

    R4-18/R4-19: turns what the legacy scorer recorded (the bindings that
    served each forecast, the gate and the chooser; the served folds' pools;
    the payoff fit, the champion residual pools, the recalibration map and
    the chooser keys) into the frozen declarations native scoring reads.
    Nothing here recomputes a value: bindings are copied by digest into
    ``release_root`` (``package_frozen_resources``, the same identity-derived
    binding ids the strict probe uses), and every pool, fit and map is the
    recorded one.

    ``release_states``: frozen artifacts from the release (payoff line/surface,
    driver residual pools, recalibration maps). One whose key AND content
    equal what legacy used is declared instead of the inline copy, pinned to
    its content hash; one with the key but other content is refused (the
    release is not what legacy served). Without one, the recorded state is
    frozen inline (pure wrappers, no fit).

    Feature rows: each consumer's declaration names the call site whose
    input row it used. A column two consumed sites recorded differently
    cannot be one ``feature_vector`` and is refused, as is any conflict the
    collector noted (``frozen.conflicts``).

    Returned keys: ``frozen_inference``, ``model_release``,
    ``forecast_recipes`` (binding-named recipes only), ``stored_forecasts``,
    ``gate_recipe``/``gate_forecast_pool``, ``chooser_recipe``/
    ``chooser_fold_pools``/``chooser_admissible_table``/
    ``chooser_analog_pool``, ``payoff_artifact_recipe``/``payoff_artifact``,
    ``model_residual_artifact_recipe``/``model_residual_artifacts``,
    ``recalibration_declared``/``recalibration_artifact``,
    ``feature_vector``/``feature_missing_mask``. The chooser's k-NN pool is
    only KEYED from the capture; the artifact comes from the release
    (``chooser_analog_pool``), whose key must match.
    """
    from engine.v2.foundation.canonical import untag_nonfinite
    from engine.v2.models.lineage import DataDependency, Lineage
    from engine.v2.scoring.source_inputs import stored_forecast_row_hash

    source = _checkpoint_value(candidate, "source_inputs")
    frozen = untag_nonfinite(dict(source.get("frozen") or {}))
    bindings = dict(frozen.get("bindings") or {})
    pools = dict(frozen.get("fold_pools") or {})
    states = dict(frozen.get("states") or {})
    inputs = dict(frozen.get("inputs") or {})
    declared = dict(frozen.get("declarations") or {})
    conflicts = list(frozen.get("conflicts") or ())
    if conflicts:
        raise StrictTraceCaptureError(
            f"legacy recorded different values for {sorted(conflicts)}")
    out: dict[str, Any] = {}
    if not bindings and not declared:
        return out

    ids: dict[str, str] = {}
    if bindings:
        slots = sorted(bindings)
        package = package_frozen_resources(
            model_bindings=[bindings[slot] for slot in slots],
            deployment_id=deployment_id,
            release_root=Path(release_root),
            source_root=(_artifact_source_root() if source_root is None
                         else Path(source_root)),
        )
        by_role = {row["role"]: row["binding_id"]
                   for row in package.sidecar_document["bindings"]}
        ids = {slot: by_role[_LEGACY_ROLE_ALIASES.get(bindings[slot]["role"],
                                                      bindings[slot]["role"])]
               for slot in slots}
        out["frozen_inference"] = FrozenInference(Path(release_root))
        out["model_release"] = _package_release(package, deployment_id)

    def ref(entry: Mapping[str, Any]) -> dict[str, str]:
        slot = str(entry.get("binding"))
        if slot not in ids:
            raise StrictTraceCaptureError(f"declared binding {slot} was not recorded")
        return {"binding_id": ids[slot], "output": str(entry["output"])}

    def pool(name: Any) -> dict[str, Any]:
        if name not in pools:
            raise StrictTraceCaptureError(f"declared fold pool {name} was not recorded")
        return dict(pools[name])

    def state(name: Any) -> dict[str, Any]:
        if name not in states:
            raise StrictTraceCaptureError(f"declared state {name} was not recorded")
        return dict(states[name])

    # -- feature rows, per consumed call site ---------------------------------
    vector: dict[str, float] = {}
    missing: dict[str, bool] = {}
    origin: dict[str, str] = {}

    def feature(name: str, value: Any, where: str) -> None:
        number = None if value is None else float(value)
        if name in origin:
            held = None if missing.get(name) else vector.get(name)
            if held != number:
                raise StrictTraceCaptureError(
                    f"feature {name} differs between {origin[name]} and {where}")
            return
        origin[name] = where
        if number is None:
            missing[name] = True
        else:
            vector[name], missing[name] = number, False

    def consume(entry: Mapping[str, Any]) -> None:
        if "site" not in entry:
            return
        key = f"{entry['binding']}@{entry.get('site')}"
        if key not in inputs:
            raise StrictTraceCaptureError(f"no recorded input row {key}")
        for name, value in sorted(inputs[key].items()):
            feature(str(name), value, key)

    for key in sorted(declared):
        entry = declared[key]
        if isinstance(entry, Mapping) and key != "chooser_primitives":
            consume(entry)
    for name, value in (declared.get("chooser_primitives") or {}).items():
        feature(str(name), value, "chooser_primitives")
    out["feature_vector"] = {name: vector[name] for name in sorted(vector)}
    out["feature_missing_mask"] = dict(sorted(missing.items()))

    # -- forecasts ------------------------------------------------------------
    forecasts, stored = {}, {}
    for key, entry in sorted(declared.items()):
        if not key.startswith("forecast:"):
            continue
        target = key.split(":", 1)[1]
        if entry.get("source") == "stored_tier4":
            row = dict(entry["row"])
            stored[target] = {"value": float(entry["value"]), "row": row,
                              "row_hash": stored_forecast_row_hash(row, entry["value"])}
            continue
        forecasts[target] = ref(entry)
    if forecasts:
        out["forecast_recipes"] = forecasts
    if stored:
        out["stored_forecasts"] = stored

    # -- gate -------------------------------------------------------------------
    gate = declared.get("gate")
    if gate is not None:
        recipe = {**ref(gate), "threshold": gate.get("threshold")}
        forecast = declared.get("gate_forecast")
        if forecast is not None:
            recipe["forecast"] = ref(forecast)
            out["gate_forecast_pool"] = pool(forecast["pool"])
        out["gate_recipe"] = recipe

    # -- chooser ----------------------------------------------------------------
    chooser = declared.get("chooser")
    if chooser is not None:
        from engine.v2.models.admissible_table import legacy_n_admissible_table

        recipe: dict[str, Any] = ref(chooser)
        producers, fold_pools = {}, {}
        for key, entry in sorted(declared.items()):
            if not key.startswith("chooser_fold:"):
                continue
            output = key.split(":", 1)[1]
            fold_pools[output] = pool(entry["pool"])
            if "binding" in entry:
                producers[output] = ref(entry)
        if producers:
            recipe["producers"] = producers
        out["chooser_fold_pools"] = fold_pools
        table_record = declared.get("chooser_admissible_table")
        if table_record is not None:
            table = legacy_n_admissible_table()
            recorded = tuple(tuple(float(v) for v in pair)
                             for pair in table_record["breakpoints"])
            if (recorded != tuple(table.breakpoints)
                    or float(table_record["fallback"]) != float(table.fallback)):
                raise StrictTraceCaptureError(
                    "legacy n_admissible table differs from the frozen v1 table")
            recipe["admissible_table"] = {"table_id": table.table_id,
                                          "version": table.version,
                                          "content_hash": table.content_hash}
            out["chooser_admissible_table"] = table
        pool_record = declared.get("chooser_analog_pool")
        if pool_record is not None:
            from tools.phase5_prepare_release import CHOOSER_POOL_ID

            key = {"pool_id": CHOOSER_POOL_ID, "cutoff": pool_record["cutoff"]}
            recipe["analog_pool"] = key
            if chooser_analog_pool is not None:
                if (chooser_analog_pool.pool_id, chooser_analog_pool.cutoff) != (
                        key["pool_id"], key["cutoff"]):
                    raise StrictTraceCaptureError(
                        "supplied chooser analog pool carries another key")
                out["chooser_analog_pool"] = chooser_analog_pool
        out["chooser_recipe"] = recipe

    # -- the model layer (payoff, driver pools, recalibration) ---------------------
    lineage = Lineage(data=(DataDependency(table="phase4.capture.legacy_served"),))
    payoff = declared.get("payoff")
    if payoff is not None:
        fit = state(payoff["state"])
        inline = _inline_payoff_artifact(fit)
        artifact = _release_match(release_states, inline, _payoff_content)
        out["payoff_artifact_recipe"] = {
            "before": payoff["before"], "seed": payoff["seed"],
            "draw_count": payoff["draw_count"]}
        out["payoff_artifact"] = artifact
        residual_recipe, residual_artifacts = {}, {}
        for key, entry in sorted(declared.items()):
            if not key.startswith("model_residual:"):
                continue
            slot = key.split(":", 1)[1]
            recorded = state(entry["state"])
            inline = _inline_driver_pool(recorded, lineage)
            chosen = _release_match(release_states, inline, _driver_pool_content)
            residual_artifacts[slot] = chosen
            residual_recipe[slot] = {"role": chosen.role, "model_id": chosen.model_id,
                                     "fold": chosen.fold,
                                     "content_hash": chosen.content_hash}
        out["model_residual_artifact_recipe"] = residual_recipe
        out["model_residual_artifacts"] = residual_artifacts

    recal = declared.get("recalibration")
    if recal is not None:
        from engine.v2.models.recalibration_artifact import make_recalibration_map_artifact

        fit = None if not recal["fitted"] else {
            "n": recal["n"], "base_rate": recal["base_rate"],
            "x_thresholds": recal["x_thresholds"], "y_thresholds": recal["y_thresholds"],
        }
        inline = make_recalibration_map_artifact(
            fit, strategy=recal["strategy"], alpha=recal["alpha"],
            cutoff=recal["cutoff"], min_pairs=recal["min_pairs"])
        out["recalibration_declared"] = True
        out["recalibration_artifact"] = _release_match(
            release_states, inline, _recalibration_content)
    return out


def _payoff_content(artifact: Any) -> Any:
    common = (artifact.n, artifact.resid_sd, artifact.r, tuple(artifact.residuals))
    if hasattr(artifact, "coefficients"):
        return ("surface", tuple(artifact.coefficients), *common)
    return ("line", artifact.driver, artifact.intercept, artifact.slope, *common)


def _driver_pool_content(artifact: Any) -> Any:
    return (artifact.min_pool, tuple(artifact.flat_residuals), artifact.bucket_edges,
            artifact.bucket_pools)


def _recalibration_content(artifact: Any) -> Any:
    return (artifact.fitted, artifact.n, artifact.base_rate, tuple(artifact.x_thresholds),
            tuple(artifact.y_thresholds), artifact.min_pairs)


def _release_match(release_states: Sequence[Any], inline: Any, content) -> Any:
    """The release's artifact for ``inline``'s key when its content equals
    what legacy used; ``inline`` when the release holds none. A release
    artifact with the key but other content is refused."""
    for candidate in release_states:
        if type(candidate) is not type(inline) or candidate.key != inline.key:
            continue
        if content(candidate) != content(inline):
            raise StrictTraceCaptureError(
                f"release {type(inline).__name__} {inline.key} is not what legacy used")
        return candidate
    return inline


def _inline_payoff_artifact(fit: Mapping[str, Any]) -> Any:
    from engine.v2.models.payoff_artifact import (
        make_payoff_line_artifact,
        make_payoff_surface_artifact,
    )

    common = {"n": fit["n"], "resid_sd": fit["resid_sd"], "r": fit["r"],
              "residuals": fit["residuals"]}
    if fit["kind"] == "surface":
        return make_payoff_surface_artifact(
            {**common, "coefficients": fit["coefficients"]},
            strategy=fit["strategy"], alpha=fit["alpha"], cutoff=fit["cutoff"])
    return make_payoff_line_artifact(
        {**common, "intercept": fit["intercept"], "slope": fit["slope"]},
        strategy=fit["strategy"], driver=fit["driver"], alpha=fit["alpha"],
        cutoff=fit["cutoff"])


def _inline_driver_pool(recorded: Mapping[str, Any], lineage: Any) -> Any:
    from engine.v2.models.residual_artifact import make_driver_residual_pool_artifact
    from engine.v2.scoring import native_payoff

    buckets = recorded.get("buckets")
    return make_driver_residual_pool_artifact(
        role=recorded["role"], model_id=recorded["model_id"], fold=None,
        flat_residuals=recorded["flat_residuals"],
        buckets=None if not buckets else {"edges": buckets["edges"],
                                          "pools": buckets["pools"]},
        deciles=native_payoff.DECILES,
        min_pool=(int(buckets["min_pool"]) if buckets else native_payoff.MIN_POOL),
        lineage=lineage)


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


class _SharedTraceDocuments:
    """The frozen pools and model states (R4-18/R4-19) many candidates share.

    A served Tier-4 fold's pool (~50k predictions and residuals), a
    champion's residual pool and a payoff fit are documented ONCE per
    (model, fold) by ``engine.score`` and embedded by reference in every
    candidate's trace (``Phase4TraceCollector.shared_documents``). Per
    candidate they used to be canonical-hashed twice (the RFC 8785 number
    formatting of ~300k floats: >95% of the recording's CPU) and pickled
    into the spill, then unpickled as separate copies. Registered here:

    * hashing (``__call__``, the collector's ``content_hasher``) reuses each
      shared value's canonical text, so it is rendered once per value, not
      per candidate; the hash is byte-identical (``canonical_json``);
    * the spill writes a shared value as a reference (pickle persistent id)
      and hydration returns the one registered object, so ``chosen``
      candidates share it again with no copy.

    Values are held until ``reset`` (the run's end); ``engine.score`` caches
    the same objects for the Scorer's life, so this adds no copies. Rendered
    texts are kept least-recently-used first, bounded by total size except
    for what the last two hashes used (see ``canonical``). ``renders``
    counts the texts rendered (a test and benchmark handle).
    """

    def __init__(self, text_budget: int = 16_000_000) -> None:
        self._held: dict[int, Any] = {}
        self._texts: dict[int, tuple[str, int]] = {}
        self._text_size = 0
        self._text_budget = int(text_budget)
        self._generation = 0
        self.renders = 0

    def register_shared(self, values: Iterable[Any]) -> None:
        for value in values:
            self._held.setdefault(id(value), value)

    def shared(self, value: Any) -> bool:
        return self._held.get(id(value)) is value

    def canonical(self, node: Any, render) -> str | None:
        key = id(node)
        if self._held.get(key) is not node:
            return None
        entry = self._texts.pop(key, None)
        if entry is None:
            self.renders += 1
            entry = (render(node), self._generation)
            self._text_size += len(entry[0])
        self._texts[key] = (entry[0], self._generation)  # most recent last
        # Over budget: drop least recently used texts, but never one the
        # current or previous hash used. A plain LRU smaller than one
        # candidate's working set (a DYN-SV row touches three folds and two
        # champions' pools) misses on EVERY access of a cyclic pattern, i.e.
        # re-renders everything per candidate, which is the cost removed here.
        while self._text_size > self._text_budget:
            oldest = next(iter(self._texts))
            text, used = self._texts[oldest]
            if used >= self._generation - 1:
                break
            del self._texts[oldest]
            self._text_size -= len(text)
        return entry[0]

    def __call__(self, value: Any) -> str:
        self._generation += 1
        # Shared values sit only in the source_inputs group's `frozen`
        # section (or are hashed directly, by reconcile); every other group
        # takes the plain path, which is faster on content with none.
        scoped = bool(self._held) and (
            self.shared(value) or (isinstance(value, dict) and "frozen" in value))
        return content_hash(value, fragments=self if scoped else None)

    def reset(self) -> None:
        self._held.clear()
        self._texts.clear()
        self._text_size = 0


_SHARED_TRACE_DOCUMENTS = _SharedTraceDocuments()


class _SpillPickler(pickle.Pickler):
    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, (dict, list)) and _SHARED_TRACE_DOCUMENTS.shared(obj):
            return ("phase4-shared", id(obj))
        return None


class _SpillUnpickler(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> Any:
        tag, key = pid
        if tag != "phase4-shared" or key not in _SHARED_TRACE_DOCUMENTS._held:
            raise pickle.UnpicklingError(f"unknown shared trace document {pid!r}")
        return _SHARED_TRACE_DOCUMENTS._held[key]


def _spill_trace(trace: dict) -> _SpilledTrace:
    path = _trace_spill_dir() / f"{next(_TRACE_SPILL_COUNTER)}.pkl"
    with path.open("wb") as fh:
        _SpillPickler(fh, protocol=pickle.HIGHEST_PROTOCOL).dump(trace)
    return _SpilledTrace(path)


def _hydrate_trace(value: Any) -> Any:
    """Read a spilled ``legacy_trace`` back, unchanged, byte-for-byte.

    A no-op on anything that is not a spill pointer (``None``, or an
    already-hydrated dict — idempotent, so a caller never needs to know
    whether an earlier step already hydrated this candidate). A shared
    frozen pool/state comes back as the one registered object.
    """
    if isinstance(value, _SpilledTrace):
        with value.path.open("rb") as fh:
            return _SpillUnpickler(fh).load()
    return value


def _cleanup_trace_spill() -> None:
    global _TRACE_SPILL_DIR
    if _TRACE_SPILL_DIR is not None:
        shutil.rmtree(_TRACE_SPILL_DIR, ignore_errors=True)
        _TRACE_SPILL_DIR = None
    _SHARED_TRACE_DOCUMENTS.reset()


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
    # The served Tier-4 folds' pools (R4-19), one list per fold shared by
    # every candidate the fold served.
    *(
        (("checkpoints", "source_inputs", "value", "frozen", "fold_pools", output),)
        for output in ("pred_abs_move", "pred_im_t1_d14", "pred_runup_abs_move_d14")
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
        digest = _SHARED_TRACE_DOCUMENTS(value)
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
            retain_full_trace=False, content_hasher=_SHARED_TRACE_DOCUMENTS,
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


def _forward_chain_keys(
    scorer, events: pd.DataFrame, as_of: pd.Timestamp,
    strategies: tuple[str, ...] | None = None,
) -> set[tuple[str, pd.Timestamp]]:
    """Every ``(ticker, date)`` :func:`forward_pass`'s own scoring will look
    up — computed before any chain I/O, exactly as :class:`engine.replay.
    ChainIndex`'s own docstring asks: the plan first, then one filtered read.

    A disabled strategy (``score_mod.DISABLED_STRATEGIES``) is skipped here:
    ``Scorer.score`` returns for it before ever reaching ``_price_entry``
    (the ``UNVALIDATED_STRUCTURE``/superseded early-return), so planning its
    chain keys would load data no request ever reads.
    """
    keys: set[tuple[str, pd.Timestamp]] = set()
    for strategy in _score_strategies(strategies):
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
    return keys


def forward_pass(scorer, events: pd.DataFrame, as_of: pd.Timestamp,
                 quote_max_age: int,
                 strategies: tuple[str, ...] | None = None,
                 index: "replay_mod.ChainIndex | None" = None) -> list[dict]:
    """Every strategy on every forward event, as the board's scoring loop does.

    ``index`` lets a caller supply one ``ChainIndex`` covering more than just
    this pass (see ``main()``, which unions this with :func:`boundary_pass`'s
    and the rescore passes' own keys into a single up-front load) — a strict
    superset of what this function would build for itself resolves every key
    it queries identically, so passing one in never changes a scored row.
    """
    strategy_names = _score_strategies(strategies)
    if index is None:
        keys = _forward_chain_keys(scorer, events, as_of, strategies)
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


def _boundary_chain_keys(
    scorer, events: pd.DataFrame, strategies: tuple[str, ...] | None = None,
) -> set[tuple[str, pd.Timestamp]]:
    """Every ``(ticker, date)`` :func:`boundary_pass`'s own scoring will look
    up — the same pattern as :func:`_forward_chain_keys`, over boundary
    events instead of forward ones. No ``latest_chain_date`` fallback here:
    unlike forward requests, a boundary ``ScoreRequest`` never sets
    ``quote_max_age_sessions`` (it is a PAST event, scored at its own
    decision close, never an upcoming one needing a stale-quote substitute),
    so ``_price_entry``'s fallback branch is unreachable for it regardless.
    """
    keys: set[tuple[str, pd.Timestamp]] = set()
    for strategy in _score_strategies(strategies):
        if strategy in score_mod.DISABLED_STRATEGIES:
            continue
        plan = replay_mod.plan_events(
            STRUCTURES[strategy](), events, calendar=scorer.calendar
        )
        keys |= plan.chain_keys
    return keys


def boundary_pass(scorer, events: pd.DataFrame,
                  strategies: tuple[str, ...] | None = None,
                  index: "replay_mod.ChainIndex | None" = None) -> list[dict]:
    """Past events, scored at their own decision close, for the two boundaries.

    ``as_of`` is the structure's DECISION date, resolved through the calendar,
    not the print date. Scoring a BMO print as of the print itself is a leak —
    ``engine.audit`` refuses it. These are also where the priced:S axes are
    won: in the forward window the forecast-sized families come back
    NO_FORECAST with empty legs.

    2026-09-18: an instrumented capture run measured this pass's own
    ``_price_entry`` calls building a FRESH ``ChainIndex`` per boundary
    request (transients up to +673 MB/call, the net climbing in steps) —
    every request reached ``Scorer.score`` with ``chain_index=None``, so
    ``_price_entry`` streamed its own 1-2-key ``load_chain_index`` call
    against the whole ``option_chains`` table every time. ``index`` (built
    once, up front — see ``_boundary_chain_keys`` and ``main()``) is exactly
    :func:`forward_pass`'s own fix for the identical shape of problem,
    applied here: a superset index resolves every key this pass ever queries
    identically to a fresh per-call one, so output is unchanged.
    """
    if index is None:
        keys = _boundary_chain_keys(scorer, events, strategies)
        index = replay_mod.load_chain_index(keys, progress_every=0) if keys else None

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
                raw, record, took, trace = _score(scorer, request, index=index)
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


def _rescore(scorer, source: dict, label: str,
            index: "replay_mod.ChainIndex | None" = None, **changes) -> dict | None:
    """Re-score a captured request with some fields changed — same clock.

    The changed request inherits the source's ``as_of``, ``chain_as_of`` and
    quote-age policy, so a pinned or strike variant of a historical row is
    scored at that row's decision date rather than today's.

    ``changes`` only ever sets ``structure_params``/``strike`` (never
    ``event_date``/``as_of``/``chain_as_of`` — see ``phase4_required_folds``'s
    docstring in ``tools/prepare_phase4_tier4_caches.py`` and this module's
    own test coverage for that invariant), and ``_price_entry``'s chain-date
    resolution (``quote_date``/``exit_date``) depends only on ``event_date``/
    ``session``/the structure's own offsets — never on ``structure_params``/
    ``strike``. So a rescored request always needs EXACTLY the same chain
    keys its source did, already resolved once when the source itself was
    scored: an ``index`` built from the source passes' own combined keys (see
    ``main()``) covers every rescore here, with no per-call chain load and no
    change to which quote a row prices against.
    """
    request = replace(request_from_dict(source["request"]), **changes)
    try:
        raw, record, took, trace = _score(scorer, request, index=index)
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


def pinned_and_strike_pass(scorer, scored: list[dict], limit: int = 8,
                           index: "replay_mod.ChainIndex | None" = None) -> list[dict]:
    """Re-score priced rows with their geometry pinned, and at a listed strike.

    The pinned pair is the `e845f3e` regression case made permanent: a replay
    that pins the shape must still record the forecast that chose it. It is
    evidence only beside the selector-resolved row it was pinned FROM, so the
    relation is recorded and :func:`select` keeps the source.

    The strike pair asks for a strike the source's legs resolved to — a real
    listed strike, not a computed moneyness the chain would snap away from.

    ``index``: see ``_rescore``'s docstring — every rescore here needs exactly
    the chain keys its source already resolved, so ``main()``'s combined
    forward+boundary index covers this pass with no per-call chain load.
    """
    out: list[dict] = []
    for source in scored:
        record = source["record"]
        params = record.get("structure_params")
        if not priced(record) or not isinstance(params, dict) or not params:
            continue
        pinned = _rescore(scorer, source, "pinned", index=index,
                          structure_params={k: v for k, v in params.items() if v is not None})
        if pinned is not None:
            pinned["relations"] = {"pinned_from": content_hash(source["request"])}
            out.append(pinned)
        listed = _anchor_strike(record)
        at_strike = (_rescore(scorer, source, "strike", index=index, strike=listed)
                     if listed is not None else None)
        if at_strike is not None:
            out.append(at_strike)
        if len(out) >= limit:
            break
    return out


#: Widths tried, narrowest first, when hunting a COARSE_LADDER refusal.
_COARSE_WIDTHS = (0.002, 0.004, 0.006, 0.01)


def coarse_ladder_pass(scorer, scored: list[dict],
                       index: "replay_mod.ChainIndex | None" = None) -> list[dict]:
    """Ask for a width the ticker's listed ladder cannot carry.

    §7.1 requires a coarse ladder and it cannot be waited for, so it is
    *requested* — a real ``structure_params`` width, through the real entry
    point, narrow enough that two legs resolve onto one contract. Asking for a
    refusal is not the same as inventing one.

    ``index``: see ``pinned_and_strike_pass``'s docstring.
    """
    for source in scored:
        record = source["record"]
        if record.get("spot") is None or record.get("strategy") not in ("TWIN-P5", "TWIN-P"):
            continue
        for width in _COARSE_WIDTHS:
            got = _rescore(scorer, source, "coarse", index=index,
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
            choice = _candidate(request, None, record, 0.0, kind="dyn_sv_choice")
            # Every ranked member's own checkpoint (still spilled), so the
            # strict probe can trace the choice member by member. Never
            # written to the pair: the pair carries the members' traces.
            choice["members"] = [
                {"request": c["request"], "record": c["record"],
                 "legacy_trace": c["legacy_trace"], "event_id": c.get("event_id"),
                 "kind": "score_result"}
                for c in siblings
            ]
            out.append(choice)
    return out


def _research_replay_chain_keys(
    scorer, events: pd.DataFrame, strategies: tuple[str, ...] | None = None,
) -> set[tuple[str, pd.Timestamp]]:
    """Every ``(ticker, date)`` :func:`research_replay_pass`'s own
    ``replay_one`` calls will look up, for CAL-P and CND-P specifically.

    ``_forward_chain_keys``/``_boundary_chain_keys`` deliberately SKIP
    ``score_mod.DISABLED_STRATEGIES`` — for THOSE passes that is correct,
    because a disabled strategy never reaches ``_price_entry`` through
    ``Scorer.score`` (it returns early). ``research_replay_pass`` is a
    different entry point: it calls ``replay_mod.replay_one`` directly, with
    no scorer short-circuit, so it genuinely needs CAL-P's/CND-P's own chain
    keys — this function is the disabled-strategy counterpart the other two
    do not, and must not, provide.
    """
    keys: set[tuple[str, pd.Timestamp]] = set()
    for strategy in score_mod.DISABLED_STRATEGIES:
        if strategies is not None and strategy not in strategies:
            continue
        plan = replay_mod.plan_events(
            STRUCTURES[strategy](), events, calendar=scorer.calendar
        )
        keys |= plan.chain_keys
    return keys


def research_replay_pass(scorer, events: pd.DataFrame, limit: int = 2,
                         strategies: tuple[str, ...] | None = None,
                         index: "replay_mod.ChainIndex | None" = None) -> list[dict]:
    """Price CAL-P and CND-P under research, where the scorer refuses them.

    §7.1 requires both to appear as *refusals* on the production path and to
    *replay* under research. That is a different entry point with a different
    record, and conflating the two would lose the distinction.

    2026-09-19: this pass used to build its own fresh ``ChainIndex`` per
    ``DISABLED_STRATEGIES`` entry (up to two full streamed table scans, late
    in the pipeline, on top of whatever forward/boundary/rescore had already
    accumulated) -- the same "fresh index per pass" shape already fixed for
    ``boundary_pass``/``_rescore``. ``index`` (a superset built once in
    ``main()`` via ``_research_replay_chain_keys``, unioned into the same
    combined ``ChainIndex`` the other passes share) resolves every key this
    pass ever queries identically to a fresh per-strategy one, so a caller
    that supplies it changes nothing about which rows come back.
    """
    out: list[dict] = []
    for strategy in score_mod.DISABLED_STRATEGIES:
        if strategies is not None and strategy not in strategies:
            continue
        structure = STRUCTURES[strategy]()
        plan = replay_mod.plan_events(structure, events, calendar=scorer.calendar)
        pass_index = (
            index if index is not None
            else replay_mod.load_chain_index(plan.chain_keys, progress_every=0)
        )
        taken = 0
        for row in plan.frame.to_dict("records"):
            started = time.monotonic()
            rows, skip = replay_mod.replay_one(structure, row, pass_index,
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


#: One built chooser analog pool per source file: ``(path, sha256)`` ->
#: ``(serialized state, content_hash, pool_id, cutoff)``. Every DYN-SV menu
#: row of a capture reads the same file, so it is frozen once.
_CHOOSER_POOL_STATES: dict[tuple[str, str], tuple[bytes, str, str, str]] = {}


def _chooser_pool_state(record: Mapping[str, Any]) -> tuple[bytes, str, str, str]:
    """The k-NN pool legacy loaded, frozen as the release freezes it.

    ``record`` is what legacy recorded (``path``, ``sha256``, ``cutoff``).
    The file must still hash to what legacy read, and the frozen state is
    built by the release preparer's own code (``chooser_pool_payloads``: the
    legacy filter, source order), keyed by the same cutoff rule.
    """
    import hashlib

    from engine.v2.models.frozen_documents import frozen_state_from_document
    from tools.phase5_prepare_release import chooser_pool_payloads

    path, expected = Path(str(record["path"])), str(record["sha256"])
    key = (str(path), expected)
    if key not in _CHOOSER_POOL_STATES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if "sha256:" + digest.hexdigest() != expected:
            raise StrictTraceCaptureError(
                "chooser analog pool file differs from the one legacy loaded")
        payloads = chooser_pool_payloads(path).get("chooser_analog_pool") or {}
        if len(payloads) != 1:
            raise StrictTraceCaptureError("chooser analog pool could not be frozen")
        ((pool_key, raw),) = payloads.items()
        pool_id, cutoff = pool_key.split("|", 1)
        state = frozen_state_from_document(json.loads(raw))
        _CHOOSER_POOL_STATES[key] = (raw, state.content_hash, pool_id, cutoff)
    raw, state_hash, pool_id, cutoff = _CHOOSER_POOL_STATES[key]
    if cutoff != str(record["cutoff"]):
        raise StrictTraceCaptureError(
            "chooser analog pool cutoff differs from the one legacy recorded")
    return raw, state_hash, pool_id, cutoff


def frozen_chooser_declaration(
    candidate: Mapping[str, Any], *, deployment_id: str, release_root: Path,
) -> tuple[dict[str, Any], FrozenResourcePackage, list[dict[str, Any]]] | None:
    """The trace's frozen chooser for a DYN-SV menu row whose legacy scorer
    ran the chooser champion (R4-18/R4-19 recording), or ``None``.

    Returns ``(declaration, package, extra resource rows)``. The champion and
    the producer folds are packaged as their own release (their roles are
    not the forecast bindings' and their rows are derived natively, never
    fed as inference requests); the fold pools are the recorded ones; the
    admissible table must be the frozen v1 table; the k-NN pool is written
    once under ``release_root`` as a content-addressed artifact.
    """
    from engine.v2.models.admissible_table import legacy_n_admissible_table

    frozen = _frozen_block(candidate)
    declared = dict(frozen.get("declarations") or {})
    chooser = declared.get("chooser")
    if chooser is None:
        return None
    if frozen.get("conflicts"):
        raise StrictTraceCaptureError(
            f"legacy recorded different values for {sorted(frozen['conflicts'])}")
    bindings = dict(frozen.get("bindings") or {})
    pools = dict(frozen.get("fold_pools") or {})
    folds = {key.split(":", 1)[1]: entry for key, entry in sorted(declared.items())
             if key.startswith("chooser_fold:")}
    slots = [str(chooser["binding"])] + [
        str(entry["binding"]) for entry in folds.values() if "binding" in entry]
    missing = sorted(slot for slot in slots if slot not in bindings)
    if missing:
        raise StrictTraceCaptureError(f"chooser binding(s) {missing} were not recorded")
    package = package_frozen_resources(
        model_bindings=[bindings[slot] for slot in slots],
        deployment_id=deployment_id,
        release_root=release_root,
        source_root=_artifact_source_root(),
    )
    by_role = {row["role"]: row["binding_id"] for row in package.sidecar_document["bindings"]}

    def ref(entry: Mapping[str, Any]) -> dict[str, str]:
        role = bindings[str(entry["binding"])]["role"]
        return {"binding_id": by_role[_LEGACY_ROLE_ALIASES.get(role, role)],
                "output": str(entry["output"])}

    recipe: dict[str, Any] = ref(chooser)
    producers = {output: ref(entry) for output, entry in folds.items() if "binding" in entry}
    if producers:
        recipe["producers"] = producers
    fold_pools = {}
    for output, entry in folds.items():
        if entry.get("pool") not in pools:
            raise StrictTraceCaptureError(f"declared fold pool {entry.get('pool')} was not recorded")
        fold_pools[output] = dict(pools[entry["pool"]])
    table_key = None
    table_record = declared.get("chooser_admissible_table")
    if table_record is not None:
        table = legacy_n_admissible_table()
        recorded = tuple(tuple(float(v) for v in pair) for pair in table_record["breakpoints"])
        if (recorded != tuple(table.breakpoints)
                or float(table_record["fallback"]) != float(table.fallback)):
            raise StrictTraceCaptureError(
                "legacy n_admissible table differs from the frozen v1 table")
        table_key = {"table_id": table.table_id, "version": table.version,
                     "content_hash": table.content_hash}
        recipe["admissible_table"] = table_key
    extra: list[dict[str, Any]] = []
    pool_resource_id = None
    pool_record = declared.get("chooser_analog_pool")
    if pool_record is not None:
        raw, state_hash, pool_id, cutoff = _chooser_pool_state(pool_record)
        recipe["analog_pool"] = {"pool_id": pool_id, "cutoff": cutoff,
                                 "content_hash": state_hash}
        hex_digest = _bytes_sha256(raw)[len("sha256:"):]
        relative = f"resources/states/{hex_digest}.json"
        destination = Path(release_root) / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, destination)
        pool_resource_id = f"phase4-state-{hex_digest}"
        extra.append({
            "resource_id": pool_resource_id, "ref": f"state:{hex_digest}",
            "kind": "artifact", "path": relative, "sha256": _bytes_sha256(raw),
        })
    declaration = {
        "schema_version": FROZEN_CHOOSER_SCHEMA,
        "release_resource_id": package.trace_declaration["release_resource_id"],
        "binding_ids": list(package.trace_declaration["binding_ids"]),
        "recipe": recipe,
        "fold_pools": fold_pools,
        "admissible_table": table_key,
        "analog_pool_resource_id": pool_resource_id,
    }
    return declaration, package, extra


def _bytes_sha256(raw: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _package_resources(package: FrozenResourcePackage) -> list[dict[str, Any]]:
    """A package's rows plus one inline sidecar per binding (its ``ref`` is
    the binding's request ref, which the request names)."""
    rows = [dict(row) for row in package.resource_rows]
    rows.extend({
        "resource_id": binding["binding_id"],
        "ref": binding["request_ref"],
        "kind": "sidecar",
        "document": binding,
        "content_hash": content_hash(binding),
    } for binding in package.sidecar_document["bindings"])
    return rows


def _merged_resources(*groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Resource rows of several packages: a row two packages share (the same
    artifact) appears once; the same id for different content is refused."""
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for row in group:
            held = merged.get(row["resource_id"])
            if held is not None and held != dict(row):
                raise StrictTraceCaptureError(
                    f"resource {row['resource_id']} declared twice with different content")
            merged.setdefault(row["resource_id"], dict(row))
    return list(merged.values())


def strict_trace_one(
    candidate: Mapping[str, Any], snapshot: str, release_root: Path,
) -> tuple[dict[str, Any], Any]:
    """One verified strict trace for one scored row: ``(trace, native record)``.

    Raises :class:`StrictTraceCaptureError` (or ``TypeError``/``ValueError``)
    when the row cannot be traced honestly.
    """
    request = canonical_v2_request(candidate, snapshot)
    source = (None if _request_only_source(candidate) is not None
              else _checkpoint_value(candidate, "source_inputs"))
    bindings = (source or {}).get("model_bindings") or ()
    package = None
    if bindings:
        package = package_frozen_resources(
            model_bindings=bindings,
            deployment_id=request.deployment_id,
            release_root=release_root,
            source_root=_artifact_source_root(),
        )
    chooser = (frozen_chooser_declaration(
        candidate, deployment_id=request.deployment_id, release_root=release_root)
        if source is not None else None)
    refs = (*(package.request_refs if package else ()),
            *(chooser[1].request_refs if chooser else ()))
    if refs:
        request = replace(request, model_artifact_refs=tuple(dict.fromkeys(refs)))
    inputs, shared_inputs = native_inputs_from_capture(
        candidate, request, frozen_chooser=chooser[0] if chooser else None,
    )
    resources = _merged_resources(
        _package_resources(package) if package else (),
        _package_resources(chooser[1]) if chooser else (),
        chooser[2] if chooser else (),
    )
    execution_inputs = None
    if chooser is not None:
        documents = {chooser[0]["release_resource_id"]: chooser[1].sidecar_document}
        execution_inputs = with_frozen_chooser(inputs, prepare_frozen_chooser(
            release_root=Path(release_root), resource_rows=resources,
            verified_documents=documents, request=request, inputs=inputs,
        ))
    runtime = (
        _frozen_runtime(package, release_root, request, inputs, candidate)
        if package else None
    )
    return package_strict_trace(
        request, inputs, shared_inputs,
        resources=resources,
        metadata={
            "capture_mode": "bounded-strict-probe",
            "legacy_checkpoint_hash": content_hash(candidate["legacy_trace"]),
            "frozen_inference": package.trace_declaration if package else None,
        },
        frozen_runtime=runtime,
        execution_inputs=execution_inputs,
    )


#: ``payload.input_trace`` of a ``dyn_sv_choice`` pair: one strict trace per
#: menu member the legacy chooser ranked, in frame order.
CHOOSER_TRACE_SCHEMA = "phase4_chooser_trace.v1.0"


def chooser_trace(
    candidate: Mapping[str, Any], snapshot: str, release_root: Path,
) -> dict[str, Any]:
    """The multi-member trace of a ``dyn_sv_choice`` pair.

    ``dynamic_short_vol`` chooses among the event's menu rows, so the choice
    can only be replayed natively from every member it ranked. Each member
    is traced exactly as a scored row is (:func:`strict_trace_one`, from its
    own legacy checkpoint) and bound to its legacy request in
    ``request.frame_rows`` by hash. One member that cannot be traced makes
    the whole pair a gap: a choice replayed over part of the menu is not the
    legacy choice.
    """
    request = candidate.get("request") or {}
    frame_rows = request.get("frame_rows")
    members = candidate.get("members")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise StrictTraceCaptureError("dyn_sv_choice request has no frame_rows")
    if not isinstance(members, list) or len(members) != len(frame_rows):
        raise StrictTraceCaptureError(
            "dyn_sv_choice candidate does not carry every ranked member's checkpoint")
    traced = []
    for index, (row, member) in enumerate(zip(frame_rows, members, strict=True)):
        if content_hash(member.get("request")) != content_hash(row.get("request")):
            raise StrictTraceCaptureError(f"member {index}: not the frame row's request")
        try:
            trace, native = strict_trace_one(member, snapshot, release_root)
        except (StrictTraceCaptureError, TypeError, ValueError) as exc:
            raise StrictTraceCaptureError(f"member {index}: {exc}") from exc
        traced.append({
            "member_index": index,
            "request_hash": content_hash(row["request"]),
            "legacy_input_hash": trace["shared_input_hash"],
            "native_score_id": native.score_id,
            "input_trace": trace,
        })
    body = {
        "schema_version": CHOOSER_TRACE_SCHEMA,
        "members": traced,
        "shared_input_hash": content_hash(
            [item["legacy_input_hash"] for item in traced]),
    }
    return {**body, "trace_hash": content_hash(body)}


def attach_strict_probe(
    chosen: list[dict], snapshot: str, release_root: Path,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Attach a strict native trace to every eligible selected score row.

    Every ``score_result`` candidate is executed independently, row by row,
    and so is every ``dyn_sv_choice`` candidate (one trace per ranked menu
    member, :func:`chooser_trace`). A row that cannot produce an honest trace
    (unsupported strategy, a missing or hash-mismatched checkpoint, an
    unresolvable feature or binding, ...) is never fabricated and never
    allowed to abort rows that DID assemble cleanly: its typed reason is
    recorded in the returned ``gaps`` map (fixture_id -> reason) instead, and
    the caller persists it on that pair as ``trace_disposition: "gap"``. The
    whole capture only refuses when NOT ONE row produced a verified trace —
    that is a wiring failure (nothing works at all), not a per-case gap.
    """
    gaps: dict[str, str] = {}
    attached = []
    for candidate in chosen:
        kind = candidate.get("kind")
        if kind not in ("score_result", "dyn_sv_choice"):
            continue
        fixture_id = str(candidate.get("fixture_id", "candidate"))
        try:
            if kind == "dyn_sv_choice":
                trace = chooser_trace(candidate, snapshot, release_root)
                native_score_id = None
            else:
                trace, native = strict_trace_one(candidate, snapshot, release_root)
                native_score_id = native.score_id
        except (StrictTraceCaptureError, TypeError, ValueError) as exc:
            gaps[fixture_id] = str(exc)
            continue
        # ``candidate["request"]`` stays the LEGACY request: coverage, pinned
        # links, seeded controls and tier-1 replay all read it. The canonical
        # V2 request lives only in ``input_trace.request`` (and its
        # ``shared_inputs``); phase4_real re-derives it from the legacy one.
        candidate["input_trace"] = trace
        candidate["legacy_input_hash"] = trace["shared_input_hash"]
        if native_score_id is not None:
            candidate["native_score_id"] = native_score_id
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
        # A +/-inf residual (legacy ResidualPool keeps it, R4-20 gap 1) is
        # written in the repo's canonical non-finite form,
        # {"__nonfinite__": "inf"} (engine.v2.foundation.canonical), never
        # dropped to null and never as a bare Infinity. content_hash already
        # hashes that same form, so every recorded hash is unchanged; readers
        # decode it with untag_nonfinite.
        checkpoint = cand.get("legacy_trace")
        if checkpoint is not None:
            checkpoint = tag_nonfinite(checkpoint)
        pair = make_pair(
            cand["fixture_id"], cand["covers"], cand["request"], cand["record"],
            record_kind=cand["kind"], duration=cand["duration"],
            legacy_trace=checkpoint,
            input_trace=cand.get("input_trace"),
            legacy_input_hash=cand.get("legacy_input_hash"),
            strict_trace_gap=strict_gaps.get(str(cand["fixture_id"])),
            relations=cand.get("relations"),
        )
        text = json.dumps(tag_nonfinite(pair), indent=2, sort_keys=True,
                          allow_nan=False) + "\n"
        (pairs_dir / f"{cand['fixture_id']}.json").write_text(text)
        manifest_pairs[cand["fixture_id"]] = {
            "payload_hash": pair["payload_hash"],
            "request_hash": pair["request_hash"],
            "record_kind": cand["kind"],
            "covers": pair["covers"],
            "trace_disposition": pair["payload"].get("trace_disposition", "absent"),
        }
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
        boundaries = _boundary_events(as_of, args.boundary_events, scorer.calendar)
        print(f"[corpus] boundary events: {len(boundaries)}", flush=True)

        # One ChainIndex, built up front, for every pass that prices a
        # request: forward, boundary, the two rescore passes (pinned/strike,
        # coarse ladder), and research_replay_pass's own disabled-strategy
        # replay. 2026-09-18: an instrumented run measured boundary_pass
        # alone streaming its own fresh ChainIndex per request (transients up
        # to +673 MB/call) because it reached Scorer.score with
        # chain_index=None every time; the rescore passes had the same shape
        # of gap. 2026-09-19: research_replay_pass had the identical gap one
        # level down -- it calls replay_mod.replay_one directly (never
        # Scorer.score), so _forward_chain_keys/_boundary_chain_keys's own
        # DISABLED_STRATEGIES skip does not cover it; it needs its own
        # _research_replay_chain_keys union, added here rather than left as
        # a second, separately-timed load late in the pipeline. See
        # boundary_pass's/_rescore's/research_replay_pass's own docstrings
        # for why a combined, superset index resolves every key identically
        # to a fresh per-call one -- this changes nothing about which quote
        # a row prices against, only when and how many times it is loaded.
        chain_keys = _forward_chain_keys(scorer, forward, as_of, strategies)
        chain_keys |= _boundary_chain_keys(scorer, boundaries, strategies)
        chain_keys |= _research_replay_chain_keys(scorer, boundaries, strategies)
        chain_index = (
            replay_mod.load_chain_index(chain_keys, progress_every=0)
            if chain_keys else None
        )

        candidates = forward_pass(
            scorer, forward, as_of, args.quote_max_age, strategies,
            index=chain_index,
        )
        print(f"[corpus] forward scores: {len(candidates)}", flush=True)

        candidates += boundary_pass(scorer, boundaries, strategies, index=chain_index)

        candidates += pinned_and_strike_pass(scorer, candidates, index=chain_index)
        candidates += coarse_ladder_pass(scorer, candidates, index=chain_index)
        if strategies is None or score_mod.DYNAMIC_STRATEGY in strategies:
            candidates += dyn_sv_pass(candidates)
        candidates += research_replay_pass(
            scorer, boundaries, strategies=strategies, index=chain_index,
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
            for holder in (cand, *(cand.get("members") or ())):
                trace = _hydrate_trace(holder.get("legacy_trace"))
                _reconcile_shared_trace_content(trace, _shared_hydration_cache)
                holder["legacy_trace"] = trace
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
