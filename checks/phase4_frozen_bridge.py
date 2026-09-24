"""Strict bridge from saved Phase 4 resources to verified frozen inference.

The trace describes model bindings with resource identifiers. It never
supplies filesystem paths to an inference adapter and never carries prediction
rows. This module resolves each member through the already hash-verified trace
resource table, derives inference rows from native model inputs, and delegates
all artifact decoding to FrozenInference.

The trace metadata declaration contains only release_resource_id and ordered
binding_ids. The referenced sidecar maps each binding member to an artifact
resource ID, so paths and hashes come from the verified resource table rather
than from executable inputs. The current 20260912T233551Z corpus cannot use
this bridge because its pair payloads contain no input_trace; the capture lane
must emit those traces and sidecars before full-release parity can compare any
records.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from engine.v2.contracts import ScoreRequest
from engine.v2.foundation import content_hash
from engine.v2.models import (
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
)
from engine.v2.models.contracts import ArtifactMember
from engine.v2.scoring.stages import NativeScoreInputs

FROZEN_TRACE_SCHEMA = "phase4_frozen_inference.v1.0"
FROZEN_RELEASE_SCHEMA = "phase4_frozen_release.v1.0"

_TRACE_KEYS = frozenset({"schema_version", "release_resource_id", "binding_ids"})
_RELEASE_KEYS = frozenset({"schema_version", "release_id", "deployment_id", "bindings"})
_BINDING_KEYS = frozenset({
    "binding_id", "model_id", "request_ref", "role", "strategy_id",
    "decision_clock_id", "adapter", "feature_order", "output_names", "members",
})
_MEMBER_KEYS = frozenset({"name", "resource_id"})
#: Roles whose feature vector is their own, not the forecast-family merge.
_ROLE_PRIVATE_VECTORS = frozenset({"gate", "chooser"})
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


class FrozenBridgeError(ValueError):
    """The saved release cannot safely construct a frozen native execution."""


@dataclass(frozen=True)
class FrozenReplayPlan:
    inference: FrozenInference
    release: ModelRelease
    requests: tuple[InferenceRequest, ...]
    receipt: str


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrozenBridgeError(f"{label}: expected object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise FrozenBridgeError(
            f"{label}: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrozenBridgeError(f"{label}: expected nonempty string")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise FrozenBridgeError(f"{label}: expected nonempty list")
    items = tuple(_nonempty(item, f"{label}[]") for item in value)
    if len(items) != len(set(items)):
        raise FrozenBridgeError(f"{label}: duplicate values")
    return items


def _resource_index(rows: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise FrozenBridgeError("resources: expected list")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _object(raw, f"resources[{index}]")
        resource_id = _nonempty(row.get("resource_id"), f"resources[{index}].resource_id")
        if resource_id in result:
            raise FrozenBridgeError(f"resources[{index}].resource_id: duplicate")
        result[resource_id] = row
    return result


def _verified_artifact(
    release_root: Path,
    resource_id: str,
    resources: Mapping[str, Mapping[str, Any]],
) -> ArtifactMember:
    row = resources.get(resource_id)
    if row is None:
        raise FrozenBridgeError(f"model member resource missing: {resource_id}")
    if row.get("kind") != "artifact":
        raise FrozenBridgeError(f"model member is not an artifact: {resource_id}")
    relative = _nonempty(row.get("path"), f"resource {resource_id}.path")
    path = (release_root / relative).resolve()
    try:
        path.relative_to(release_root.resolve())
    except ValueError as exc:
        raise FrozenBridgeError(f"resource {resource_id}.path: escapes release root") from exc
    if not path.is_file():
        raise FrozenBridgeError(f"resource {resource_id}.path: missing")
    digest = _nonempty(row.get("sha256"), f"resource {resource_id}.sha256")
    actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != actual:
        raise FrozenBridgeError(f"resource {resource_id}.sha256: mismatch")
    return ArtifactMember(name="pending", path=relative, content_hash=digest)


def _answer_free(inputs: NativeScoreInputs) -> None:
    if inputs.geometry is not None or inputs.pricing is not None:
        raise FrozenBridgeError("native inputs contain calculated geometry or pricing")
    for block_name, forbidden in _ANSWER_FIELDS.items():
        block = getattr(inputs, block_name)
        found = sorted(str(key) for key in block if str(key) in forbidden)
        if found:
            raise FrozenBridgeError(
                f"native inputs {block_name} contain calculated answers: {found}"
            )


def binding_feature_row(
    binding: ModelBinding, vector: Mapping[str, Any],
) -> tuple[float, ...] | None:
    """One inference row for ``binding`` from its captured feature ``vector``,
    or ``None`` when a required feature came back non-finite.

    This is the ONE predicate that decides whether a binding is included,
    shared by ``_feature_rows`` below (replay, via
    ``checks/phase4_real.py``) and ``_frozen_runtime`` in
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
    Decoding it here (rather than letting ``float()`` raise on the dict
    itself) is what tells a genuinely non-finite captured value apart from
    an actually-malformed one.

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
    a genuinely non-numeric value (a string, not a nonfinite tag) both still
    raise ``FrozenBridgeError`` -- those are not an omission either side
    should make silently. The one exception: a GATE-role binding whose ONLY
    absent names are the derived forecast/analog columns (``native_gate_
    features.GATE_FORECAST_COLUMNS | GATE_ANALOG_COLUMNS``) is a legacy row
    whose base frame predates the gate feature extension -- defer eager
    inference (``None``) rather than raise; the native gate stage derives
    those columns at its own executor.
    """
    from engine.v2.foundation import untag_nonfinite
    from engine.v2.scoring.native_gate_features import (
        GATE_ANALOG_COLUMNS, GATE_FORECAST_COLUMNS,
    )

    feature_order = tuple(binding.feature_order)
    missing = [name for name in feature_order if name not in vector]

    # Classify EVERY absent name before touching a single present cell: an
    # unknown missing base feature is a hard error no matter what the captured
    # cells happen to hold, so it must be rejected here -- not shadowed by an
    # early return below the moment a present cell reads non-finite. The only
    # legal omission is a GATE-role row whose absent names are ALL the derived
    # forecast/analog columns: a legacy base frame that predates the gate
    # feature extension, which the native gate stage reconstructs at its own
    # executor. Mark it for deferral rather than returning, so the present
    # cells are still validated -- a legal (or illegal) omission must never
    # hide a malformed captured string.
    defer = False
    if missing:
        base_role = str(binding.role).split(":", 1)[0]
        derived = set(GATE_FORECAST_COLUMNS) | set(GATE_ANALOG_COLUMNS)
        if base_role == "gate" and set(missing) <= derived:
            defer = True
        else:
            raise FrozenBridgeError(
                f"binding {binding.binding_id}: missing feature {missing[0]}"
            )

    # Validate every cell the vector DID capture. A malformed (non-numeric,
    # untagged) value is a hard error on either side, never an omission. A
    # genuinely non-finite captured value omits the binding exactly as a
    # complete row would -- but accumulate that flag rather than returning
    # early, so an earlier NaN cannot mask a later malformed cell.
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
            raise FrozenBridgeError(
                f"binding {binding.binding_id}: nonnumeric feature {name}"
            ) from exc
        if not isfinite(value):
            nonfinite = True
            continue
        values[name] = value

    if defer or nonfinite:
        return None
    return tuple(values[name] for name in feature_order)


def _feature_rows(
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
        raise FrozenBridgeError("native inputs require features.model_inputs")
    role_rows = inputs.features.get("role_model_inputs")
    if role_rows is not None and not isinstance(role_rows, Mapping):
        raise FrozenBridgeError("features.role_model_inputs: expected object")
    requests = []
    for binding in bindings:
        role = binding.role.split(":", 1)[0]
        if role_rows is not None:
            vector = role_rows.get(binding.role, role_rows.get(role))
            if not isinstance(vector, Mapping):
                raise FrozenBridgeError(
                    f"binding {binding.binding_id}: no captured row for role {binding.role}"
                )
        elif role in _ROLE_PRIVATE_VECTORS:
            raise FrozenBridgeError(
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


def _verified_release(
    *,
    release_root: Path,
    resources: Mapping[str, Mapping[str, Any]],
    verified_documents: Mapping[str, Any],
    release_resource_id: Any,
    binding_ids: Any,
    request: ScoreRequest,
    label: str,
) -> tuple[ModelRelease, set[str], Mapping[str, Any], tuple[str, ...]]:
    """The selected bindings of one verified release sidecar, as a release.

    Returns ``(release, request refs, sidecar document, selected ids)``.
    """
    release_resource_id = _nonempty(release_resource_id, f"{label}.release_resource_id")
    release_row = resources.get(release_resource_id)
    if release_row is None or release_row.get("kind") != "sidecar":
        raise FrozenBridgeError("frozen release resource must be a verified sidecar")
    release_document = verified_documents.get(release_resource_id)
    release_document = _object(release_document, "frozen release")
    _exact_keys(release_document, _RELEASE_KEYS, "frozen release")
    if release_document["schema_version"] != FROZEN_RELEASE_SCHEMA:
        raise FrozenBridgeError("frozen release.schema_version: unsupported")
    deployment_id = _nonempty(release_document["deployment_id"], "frozen release.deployment_id")
    if deployment_id != request.deployment_id:
        raise FrozenBridgeError("frozen release.deployment_id: request mismatch")
    release_id = _nonempty(release_document["release_id"], "frozen release.release_id")

    raw_bindings = release_document["bindings"]
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise FrozenBridgeError("frozen release.bindings: expected nonempty list")
    selected_ids = _string_tuple(binding_ids, f"{label}.binding_ids")
    selected = set(selected_ids)
    bindings: dict[str, ModelBinding] = {}
    request_refs: set[str] = set()
    roles: set[str] = set()
    for index, raw_binding in enumerate(raw_bindings):
        item = _object(raw_binding, f"frozen release.bindings[{index}]")
        _exact_keys(item, _BINDING_KEYS, f"frozen release.bindings[{index}]")
        binding_id = _nonempty(item["binding_id"], f"binding[{index}].binding_id")
        if binding_id in bindings:
            raise FrozenBridgeError(f"binding {binding_id}: duplicate")
        if binding_id not in selected:
            continue
        strategy_id = _nonempty(item["strategy_id"], f"binding {binding_id}.strategy_id")
        if strategy_id not in {request.strategy_version, "*"}:
            raise FrozenBridgeError(f"binding {binding_id}: strategy mismatch")
        clock = _nonempty(item["decision_clock_id"], f"binding {binding_id}.decision_clock_id")
        if clock != request.decision_clock_id:
            raise FrozenBridgeError(f"binding {binding_id}: decision clock mismatch")
        role = _nonempty(item["role"], f"binding {binding_id}.role")
        base_role = role.split(":", 1)[0]
        if base_role in roles:
            raise FrozenBridgeError(f"binding {binding_id}: duplicate role {base_role}")
        roles.add(base_role)
        request_ref = _nonempty(item["request_ref"], f"binding {binding_id}.request_ref")
        request_refs.add(request_ref)
        members_raw = item["members"]
        if not isinstance(members_raw, list) or not members_raw:
            raise FrozenBridgeError(f"binding {binding_id}.members: expected nonempty list")
        members = []
        names = set()
        for member_index, raw_member in enumerate(members_raw):
            member = _object(raw_member, f"binding {binding_id}.members[{member_index}]")
            _exact_keys(member, _MEMBER_KEYS, f"binding {binding_id}.members[{member_index}]")
            name = _nonempty(member["name"], f"binding {binding_id}.members[{member_index}].name")
            if name in names:
                raise FrozenBridgeError(f"binding {binding_id}: duplicate member name {name}")
            names.add(name)
            verified = _verified_artifact(
                release_root,
                _nonempty(member["resource_id"], f"binding {binding_id}.members[{member_index}].resource_id"),
                resources,
            )
            members.append(ArtifactMember(
                name=name,
                path=verified.path,
                content_hash=verified.content_hash,
            ))
        bindings[binding_id] = ModelBinding(
            binding_id=binding_id,
            model_id=_nonempty(item["model_id"], f"binding {binding_id}.model_id"),
            role=role,
            strategy_id=strategy_id,
            decision_clock_id=clock,
            adapter=_nonempty(item["adapter"], f"binding {binding_id}.adapter"),
            feature_order=_string_tuple(item["feature_order"], f"binding {binding_id}.feature_order"),
            output_names=_string_tuple(item["output_names"], f"binding {binding_id}.output_names"),
            members=tuple(members),
        )
    if set(bindings) != selected:
        raise FrozenBridgeError(
            f"frozen release: missing selected bindings {sorted(selected - set(bindings))}"
        )
    release = ModelRelease(
        release_id=release_id,
        deployment_id=deployment_id,
        bindings=tuple(bindings[binding_id] for binding_id in selected_ids),
    )
    return release, request_refs, release_document, selected_ids


def prepare_frozen_replay(
    *,
    release_root: Path,
    resource_rows: Any,
    verified_documents: Mapping[str, Any],
    metadata: Any,
    request: ScoreRequest,
    inputs: NativeScoreInputs,
    extra_refs: Iterable[str] = (),
) -> FrozenReplayPlan | None:
    """Build a frozen execution plan from a hash-verified trace.

    ``extra_refs``: request model refs another verified declaration of the
    same trace owns (the frozen chooser's bindings); together with this
    release's refs they must be exactly the request's.
    """
    if metadata is None:
        return None
    metadata = _object(metadata, "input_trace.metadata")
    declaration = metadata.get("frozen_inference")
    if declaration is None:
        return None
    declaration = _object(declaration, "input_trace.metadata.frozen_inference")
    _exact_keys(declaration, _TRACE_KEYS, "frozen_inference")
    if declaration["schema_version"] != FROZEN_TRACE_SCHEMA:
        raise FrozenBridgeError("frozen_inference.schema_version: unsupported")

    _answer_free(inputs)
    release, request_refs, release_document, selected_ids = _verified_release(
        release_root=release_root,
        resources=_resource_index(resource_rows),
        verified_documents=verified_documents,
        release_resource_id=declaration["release_resource_id"],
        binding_ids=declaration["binding_ids"],
        request=request,
        label="frozen_inference",
    )
    if request_refs | set(extra_refs) != set(request.model_artifact_refs):
        raise FrozenBridgeError("frozen release request refs do not exactly match request")

    ordered_bindings = release.bindings
    requests = _feature_rows(inputs, ordered_bindings, release.release_id)
    receipt = content_hash({
        "release": release_document,
        "binding_ids": selected_ids,
        "requests": tuple({
            "binding_id": item.binding_id,
            "feature_order": item.feature_order,
            "rows": item.rows,
        } for item in requests),
        "artifact_hashes": tuple(
            member.content_hash for binding in ordered_bindings for member in binding.members
        ),
    })
    return FrozenReplayPlan(
        inference=FrozenInference(release_root),
        release=release,
        requests=requests,
        receipt=receipt,
    )


# -- the frozen DYN-SV chooser ------------------------------------------------

#: ``native_inputs.chooser`` of a DYN-SV menu candidate whose legacy row ran
#: the chooser champion: a JSON declaration (never executors) that
#: :func:`prepare_frozen_chooser` turns into the executable chooser block the
#: native chooser stage runs (``engine.v2.scoring.chooser_inputs``).
FROZEN_CHOOSER_SCHEMA = "phase4_frozen_chooser.v1.0"
FROZEN_CHOOSER_FIELD = "frozen_chooser"
_CHOOSER_KEYS = frozenset({
    "schema_version", "release_resource_id", "binding_ids", "recipe",
    "fold_pools", "admissible_table", "analog_pool_resource_id",
})


@dataclass(frozen=True)
class FrozenChooserPlan:
    """The executable chooser block plus the declared parts it was built from
    (so a consumer can rebuild it over another store, e.g. the P5-6 replay
    serving the same bytes from a staged release)."""

    release: ModelRelease
    request_refs: frozenset[str]
    block: Mapping[str, Any]
    recipe: Mapping[str, Any] = field(default_factory=dict)
    fold_pools: Mapping[str, Any] = field(default_factory=dict)
    analog_pool: Any = None
    admissible_table: Any = None


def prepare_frozen_chooser(
    *,
    release_root: Path,
    resource_rows: Any,
    verified_documents: Mapping[str, Any],
    request: ScoreRequest,
    inputs: NativeScoreInputs,
) -> FrozenChooserPlan | None:
    """The executable chooser block a trace's frozen chooser declares.

    The chooser champion and the Tier-4 producer folds are bindings of their
    own verified release sidecar; the fold pools are the recorded ones; the
    n_admissible table is the frozen v1 table (its key is checked by the
    native stage); the k-NN pool is a verified artifact resource holding the
    serialized ``ChooserAnalogPoolArtifact``. Nothing is fitted here.
    """
    declaration = inputs.chooser.get(FROZEN_CHOOSER_FIELD)
    if declaration is None:
        return None
    if set(inputs.chooser) != {FROZEN_CHOOSER_FIELD}:
        raise FrozenBridgeError("native_inputs.chooser: frozen_chooser must stand alone")
    declaration = _object(declaration, "frozen_chooser")
    _exact_keys(declaration, _CHOOSER_KEYS, "frozen_chooser")
    if declaration["schema_version"] != FROZEN_CHOOSER_SCHEMA:
        raise FrozenBridgeError("frozen_chooser.schema_version: unsupported")
    resources = _resource_index(resource_rows)
    release, request_refs, _document, _ids = _verified_release(
        release_root=release_root,
        resources=resources,
        verified_documents=verified_documents,
        release_resource_id=declaration["release_resource_id"],
        binding_ids=declaration["binding_ids"],
        request=request,
        label="frozen_chooser",
    )
    table = None
    if declaration["admissible_table"] is not None:
        from engine.v2.models.admissible_table import legacy_n_admissible_table

        table = legacy_n_admissible_table()
    pool = None
    if declaration["analog_pool_resource_id"] is not None:
        from engine.v2.models.chooser_analog_pool import ChooserAnalogPoolArtifact
        from engine.v2.models.frozen_documents import frozen_state_from_document

        member = _verified_artifact(
            release_root,
            _nonempty(declaration["analog_pool_resource_id"],
                      "frozen_chooser.analog_pool_resource_id"),
            resources,
        )
        document = json.loads((release_root / member.path).read_bytes())
        pool = frozen_state_from_document(document)
        if not isinstance(pool, ChooserAnalogPoolArtifact):
            raise FrozenBridgeError("frozen_chooser analog pool: not a chooser analog pool")
    from engine.v2.scoring.chooser_inputs import frozen_chooser_block

    try:
        block = frozen_chooser_block(
            strategy=request.strategy_version,
            recipe=_object(declaration["recipe"], "frozen_chooser.recipe"),
            fold_pools=_object(declaration["fold_pools"], "frozen_chooser.fold_pools"),
            analog_pool=pool,
            admissible_table=table,
            inference=FrozenInference(release_root),
            release=release,
        )
    except ValueError as exc:
        raise FrozenBridgeError(f"frozen_chooser: {exc}") from exc
    return FrozenChooserPlan(
        release=release, request_refs=frozenset(request_refs), block=block,
        recipe=declaration["recipe"], fold_pools=declaration["fold_pools"],
        analog_pool=pool, admissible_table=table,
    )


def with_frozen_chooser(
    inputs: NativeScoreInputs, plan: FrozenChooserPlan | None,
) -> NativeScoreInputs:
    """``inputs`` with its chooser declaration replaced by the executable block."""
    if plan is None:
        return inputs
    return replace(inputs, chooser=dict(plan.block))


def with_frozen_gate_forecast(
    inputs: NativeScoreInputs, *, release_root: Path, release: ModelRelease,
) -> NativeScoreInputs:
    """The gate's derived-forecast executor, resolved from the verified release.

    A packaged declared gate (``tools.capture_tier0_corpus.frozen_gate_packaging``)
    carries its forecast producer fold as a REFERENCE -- ``forecast`` naming the
    size-fold binding of THIS release plus ``forecast_pool``, exactly the shape
    ``engine.v2.scoring.source_inputs._gate_block`` pops into a
    ``forecast_executor`` on the SourceBundle path. Strict traces hold JSON, so
    the resolution happens at execution instead: capture
    (``strict_trace_one``) and replay (``checks/phase4_real.py``) call this over
    the same hash-verified release, so both sides run the fold the declaration
    names and neither can substitute another model. A block without a declared
    forecast is returned unchanged; a reference that is not exactly one binding
    of this release (or names an output it does not produce) is refused.
    """
    block = inputs.gate
    forecast = block.get("forecast") if isinstance(block, Mapping) else None
    if forecast is None or block.get("forecast_executor") is not None:
        return inputs
    from engine.v2.scoring.frozen_executor import (
        FrozenRecipeExecutor,
        FrozenStageExecutor,
    )

    forecast = _object(forecast, "gate.forecast")
    binding_id = _nonempty(forecast.get("binding_id"), "gate.forecast.binding_id")
    output = _nonempty(forecast.get("output"), "gate.forecast.output")
    matches = [b for b in release.bindings if b.binding_id == binding_id]
    if len(matches) != 1:
        raise FrozenBridgeError(
            f"gate.forecast binding {binding_id} is not exactly one binding "
            "of the verified release")
    binding = matches[0]
    if str(binding.role).split(":", 1)[0] != "size":
        raise FrozenBridgeError(
            f"gate.forecast binding {binding_id} serves role {binding.role!r}, not size")
    if output not in tuple(binding.output_names):
        raise FrozenBridgeError(
            f"gate.forecast output {output} is not an output of binding {binding_id}")
    executor = FrozenRecipeExecutor(
        target="pred_abs_move", binding_id=binding_id, source=output,
        executor=FrozenStageExecutor(
            inference=FrozenInference(Path(release_root)), release=release,
            binding_id=binding_id),
        binding=binding,
    )
    return replace(inputs, gate={
        **block, "forecast_executor": executor, "forecast_recipe": str(executor),
    })


__all__ = [
    "FROZEN_CHOOSER_FIELD",
    "FROZEN_CHOOSER_SCHEMA",
    "FROZEN_RELEASE_SCHEMA",
    "FROZEN_TRACE_SCHEMA",
    "FrozenBridgeError",
    "FrozenChooserPlan",
    "FrozenReplayPlan",
    "prepare_frozen_chooser",
    "prepare_frozen_replay",
    "with_frozen_chooser",
    "with_frozen_gate_forecast",
]
