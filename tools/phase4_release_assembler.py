"""Assemble hash-bound Phase 4 comparison traces from captured sources."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from engine.v2.contracts import ScoreRequest
from engine.v2.foundation import NONFINITE_KEY, content_hash, from_document
from engine.v2.scoring.identity import request_hash
from engine.v2.scoring.source_inputs import (
    _ANSWER_FIELDS,
    STORED_REF_FIELDS,
    STORED_ROW_FIELDS,
)

TRACE_SCHEMA = "phase4_input_trace.v1.0"
TRANSLATION_SCHEMA = "phase4_input_translation.v1.0"
REQUIRED_STAGES = (
    "resolve_context", "features", "forecast", "geometry", "pricing",
    "analogs", "simulation", "gate", "chooser", "serialization",
)
#: The payoff-calibration/model stage (exp_pnl_model, win_model) is executed
#: by every native scoring call made after it was added, so its observation
#: is always present in a fresh capture and must be a *known*, real-evidence
#: stage rather than an unknown one. It is intentionally NOT in
#: REQUIRED_STAGES: bundles captured before the stage existed never recorded
#: it, the trace schema carries no version field to key a hard requirement
#: on, and this file must not manufacture one. When present, it is validated
#: with the exact same hash/owner checks as every other stage below.
OPTIONAL_STAGES = ("model",)
NATIVE_INPUT_KEYS = frozenset({
    "context", "features", "forecast", "geometry", "pricing", "analogs",
    "simulation", "gate", "chooser", "diagnostics", "source_ref",
})
#: Containers under ``native_inputs.forecast`` whose KEYS are output names,
#: exempt from :func:`_reject_answers` for that reason and no other.
#:
#: The rule this exempts from: an output of the system under test may not be
#: supplied back to it as an input, or the comparison is circular -- native
#: would "agree" with legacy even if the path that produces the value were
#: entirely broken. ``_ANSWER_FIELDS`` names those outputs, and the walk
#: refuses any dict key that matches one.
#:
#: A container keyed BY output name trips that walk on its keys while
#: carrying none of the values. What each one is allowed to carry, and why
#: none of it is an answer:
#:
#: * ``models``      -- per output, the recipe (intercept + coefficients) that
#:                      native must EXECUTE to produce the output. Inputs to
#:                      a calculation, not its result.
#: * ``model_artifact_refs``
#:                   -- per output, the identifier of the frozen artifact
#:                      native must load and run. An address, not a result.
#: * ``stored_refs`` -- per output, the ADDRESS of a cell in a stored table
#:                      (which table, which vintage of it, which column, which
#:                      key, which producer model wrote it) plus a one-way
#:                      content hash of the row and value together. Native has
#:                      to read the table to obtain the value; the hash only
#:                      lets it check what it read. Added for the stored
#:                      ``pred_iv_crush_30`` crush forecast, which legacy reads
#:                      from the Tier-4 table for any event that has already
#:                      printed -- every HISTORICAL row of the seven planned-exit
#:                      DYN-SV strategies. Carrying that cell's value here is
#:                      what made those rows incomparable; carrying its address
#:                      makes native exercise its own retrieval path.
#:
#: The exemption is one level deep by construction (it tests ``path[-2] ==
#: "forecast"``), so an answer nested any further inside is still refused, and
#: :func:`_reject_stored_refs` separately pins ``stored_refs`` to exactly its
#: address fields so a raw value cannot ride in beside the address under a
#: name ``_ANSWER_FIELDS`` does not happen to list.
_FORECAST_ROLE_CONTAINERS = frozenset({
    "models", "model_artifact_refs", "stored_refs",
})


class ReleaseAssemblyError(ValueError):
    """Captured material cannot form a strict comparison trace."""


def _leaves(value: Any, path: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], Any]:
    if isinstance(value, Mapping):
        # This walk currently only ever sees the PRE-tag document (this
        # module runs before tools/capture_tier0_corpus.py's
        # _prepare_normalized_shared tags a nonfinite float as
        # {NONFINITE_KEY: repr(value)} for the stored JSON), so a raw
        # NaN/Inf is already an atomic leaf below and this branch is a
        # no-op today. It exists so this function keeps agreeing with
        # checks/phase4_real.py's _leaf_values -- the reader-side walk over
        # the on-disk (already-tagged) form, which MUST stop here (see that
        # module for why) -- under one definition of "leaf" instead of two
        # that only coincidentally agree because of call order. If a future
        # change ever has this run on an already-tagged document, the
        # recorded path stays correct instead of landing one segment too
        # deep with a mismatched value_hash.
        if set(value) == {NONFINITE_KEY} and isinstance(value[NONFINITE_KEY], str):
            return {path: value}
        if not value:
            return {path: {}}
        result = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ReleaseAssemblyError(
                    "input translation object keys must be strings"
                )
            result.update(_leaves(value[key], path + (key,)))
        return result
    if isinstance(value, (list, tuple)):
        if not value:
            return {path: []}
        result = {}
        for index, item in enumerate(value):
            result.update(_leaves(item, path + (index,)))
        return result
    return {path: value}


def _request_hash(request: Mapping[str, Any]) -> str:
    try:
        typed = from_document(ScoreRequest, dict(request))
    except (TypeError, ValueError) as exc:
        raise ReleaseAssemblyError("request is not canonical v2") from exc
    return request_hash(typed)


def _reject_answers(
    value: Any,
    path: tuple[str, ...] = ("native_inputs",),
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            role_declaration = (
                len(path) >= 2
                and path[-2] == "forecast"
                and path[-1] in _FORECAST_ROLE_CONTAINERS
            )
            # ``features.role_model_inputs`` is keyed role -> feature name ->
            # value: the input ROW each frozen binding is fed. A feature that
            # shares a name with an output (a downstream model consuming an
            # upstream forecast, e.g. ``forecast_abs_move`` as a gate feature)
            # is that binding's input, not this row's answer -- native still
            # has to run the binding to get an answer out of it. The depth
            # test keeps the exemption to the feature-name level.
            role_feature_vector = (
                len(path) >= 3
                and path[-3] == "features"
                and path[-2] == "role_model_inputs"
            )
            if name in _ANSWER_FIELDS and not (role_declaration or role_feature_vector):
                location = ".".join((*path, name))
                raise ReleaseAssemblyError(f"{location}: calculated answer")
            _reject_answers(child, (*path, name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_answers(child, (*path, str(index)))


def _reject_stored_refs(native_inputs: Any) -> None:
    """Pin ``forecast.stored_refs`` to an address, and refuse a stored VALUE.

    :data:`_FORECAST_ROLE_CONTAINERS` exempts ``stored_refs`` from the
    answer walk because its keys are output names. That exemption must not
    turn into a hole: :func:`_reject_answers` only knows the names in
    ``_ANSWER_FIELDS``, and a captured cell smuggled in as ``"value"`` --
    or as ``"v"``, or as an extra row column -- is not one of them. So the
    shape is pinned here instead of enumerated: exactly
    ``{"row", "row_hash"}``, and the row exactly its address fields. Any
    other member refuses, whatever it is called.

    A ``forecast.stored`` block is refused outright: that is the RESOLVED
    block, which only ``checks/phase4_stored_forecasts.py`` may produce, at
    replay, from a table it read itself. Captured material never holds one.
    """
    if not isinstance(native_inputs, Mapping):
        return
    forecast = native_inputs.get("forecast")
    if not isinstance(forecast, Mapping):
        return
    if "stored" in forecast:
        raise ReleaseAssemblyError(
            "native_inputs.forecast.stored: resolved stored forecasts are "
            "produced natively at replay, never captured"
        )
    refs = forecast.get("stored_refs")
    if refs is None:
        return
    if not isinstance(refs, Mapping):
        raise ReleaseAssemblyError("native_inputs.forecast.stored_refs: expected object")
    for output, entry in refs.items():
        location = f"native_inputs.forecast.stored_refs.{output}"
        if not isinstance(entry, Mapping):
            raise ReleaseAssemblyError(f"{location}: expected object")
        extra = sorted(set(map(str, entry)) - STORED_REF_FIELDS)
        if extra:
            raise ReleaseAssemblyError(
                f"{location}: stored forecast reference carries {extra}; it may "
                f"carry only {sorted(STORED_REF_FIELDS)}"
            )
        if set(map(str, entry)) != STORED_REF_FIELDS:
            raise ReleaseAssemblyError(
                f"{location}: must name {sorted(STORED_REF_FIELDS)}"
            )
        row = entry["row"]
        if not isinstance(row, Mapping):
            raise ReleaseAssemblyError(f"{location}.row: expected object")
        row_extra = sorted(set(map(str, row)) - STORED_ROW_FIELDS)
        if row_extra:
            raise ReleaseAssemblyError(
                f"{location}.row: stored forecast address carries {row_extra}; it "
                f"may carry only {sorted(STORED_ROW_FIELDS)}"
            )
        if set(map(str, row)) != STORED_ROW_FIELDS:
            raise ReleaseAssemblyError(
                f"{location}.row: must name {sorted(STORED_ROW_FIELDS)}"
            )


def _path(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ReleaseAssemblyError(f"{label}: expected nonempty path")
    result: list[Any] = []
    for segment in value:
        if isinstance(segment, str):
            if not segment:
                raise ReleaseAssemblyError(f"{label}: empty path segment")
        elif type(segment) is int:
            if segment < 0:
                raise ReleaseAssemblyError(f"{label}: negative list index")
        else:
            raise ReleaseAssemblyError(f"{label}: invalid path segment")
        result.append(segment)
    return tuple(result)


def _translation(
    shared: Mapping[str, Any],
    native: Mapping[str, Any],
    mappings: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    native_doc = {
        "request": shared["request"],
        "native_inputs": {
            key: value for key, value in native.items()
            if key != "source_ref"
        },
    }
    shared_leaves = _leaves(shared)
    native_leaves = _leaves(native_doc)
    if mappings is None:
        if set(shared_leaves) != set(native_leaves):
            raise ReleaseAssemblyError("translation leaf coverage differs")
        mapping_paths = tuple((path, path) for path in sorted(
            shared_leaves, key=repr,
        ))
    else:
        paths = []
        for index, row in enumerate(mappings):
            label = f"mappings[{index}]"
            if not isinstance(row, Mapping) or set(row) - {
                "shared_path", "native_path", "value_hash",
            }:
                raise ReleaseAssemblyError(f"{label}: malformed")
            if "shared_path" not in row or "native_path" not in row:
                raise ReleaseAssemblyError(f"{label}: malformed")
            paths.append((
                _path(row["shared_path"], f"{label}.shared_path"),
                _path(row["native_path"], f"{label}.native_path"),
            ))
        mapping_paths = tuple(paths)

    rows = []
    shared_paths: set[tuple[Any, ...]] = set()
    native_paths: set[tuple[Any, ...]] = set()
    for index, (shared_path, native_path) in enumerate(mapping_paths):
        label = f"mappings[{index}]"
        if shared_path in shared_paths or native_path in native_paths:
            raise ReleaseAssemblyError(f"{label}: duplicate path")
        if shared_path not in shared_leaves or native_path not in native_leaves:
            raise ReleaseAssemblyError(f"{label}: path is not a document leaf")
        shared_value = shared_leaves[shared_path]
        if content_hash(shared_value) != content_hash(native_leaves[native_path]):
            raise ReleaseAssemblyError(f"{label}: translated values differ")
        rows.append({
            "shared_path": list(shared_path),
            "native_path": list(native_path),
            "value_hash": content_hash(shared_value),
        })
        shared_paths.add(shared_path)
        native_paths.add(native_path)
    if shared_paths != set(shared_leaves) or native_paths != set(native_leaves):
        raise ReleaseAssemblyError("translation leaf coverage mismatch")

    body = {
        "schema_version": TRANSLATION_SCHEMA,
        "shared_input_hash": content_hash(shared),
        "native_input_hash": content_hash(native),
        "mappings": rows,
        "derived": [{
            "native_path": ["native_inputs", "source_ref"],
            "operation": "shared_input_hash",
            "value_hash": content_hash(content_hash(shared)),
        }],
    }
    body["translation_hash"] = content_hash(body)
    return body


def assemble_input_trace(*, request: Mapping[str, Any], shared_inputs: Mapping[str, Any],
                         native_inputs: Mapping[str, Any], observations: Sequence[Any],
                         resources: Sequence[Mapping[str, Any]],
                         metadata: Mapping[str, Any] | None = None,
                         mappings: Sequence[Mapping[str, Any]] | None = None,
                         shared_documents: Sequence[Any] = ()) -> dict[str, Any]:
    """Build the strict trace consumed by ``checks.phase4_real``."""
    request_hash = _request_hash(request)
    if shared_inputs.get("request") != request:
        raise ReleaseAssemblyError("shared_inputs.request differs from request")
    unknown_inputs = set(native_inputs) - NATIVE_INPUT_KEYS
    missing_inputs = NATIVE_INPUT_KEYS - set(native_inputs)
    if unknown_inputs or missing_inputs:
        raise ReleaseAssemblyError(
            "native_inputs keys differ: "
            f"unknown={sorted(unknown_inputs)}, missing={sorted(missing_inputs)}"
        )
    _reject_answers(native_inputs)
    _reject_stored_refs(native_inputs)
    shared_hash = content_hash(shared_inputs)
    if native_inputs.get("source_ref") != shared_hash:
        raise ReleaseAssemblyError(
            "native_inputs.source_ref must equal shared input hash"
        )

    by_stage = {}
    allowed_stages = set(REQUIRED_STAGES) | set(OPTIONAL_STAGES) | {"diagnostics"}
    for item in observations:
        try:
            stage = item.receipt.stage
            input_hash = item.receipt.input_hash
            output_hash = item.receipt.output_hash
            owner = item.receipt.owner
            input_document = item.input_document
            output_document = item.output_document
        except AttributeError as exc:
            raise ReleaseAssemblyError("observation is malformed") from exc
        if stage not in allowed_stages:
            raise ReleaseAssemblyError(f"unknown native observation stage: {stage}")
        if stage in by_stage:
            raise ReleaseAssemblyError(f"duplicate native observation stage: {stage}")
        if not isinstance(owner, str) or not owner.strip():
            raise ReleaseAssemblyError(f"native observation {stage}: owner missing")
        if input_hash != content_hash(input_document):
            raise ReleaseAssemblyError(
                f"native observation {stage}: input hash mismatch"
            )
        if output_hash != content_hash(output_document):
            raise ReleaseAssemblyError(
                f"native observation {stage}: output hash mismatch"
            )
        by_stage[stage] = item
    missing = sorted(set(REQUIRED_STAGES) - set(by_stage))
    if missing:
        raise ReleaseAssemblyError(f"native observations missing stages: {missing}")
    stages = {}
    for stage in (*REQUIRED_STAGES, *OPTIONAL_STAGES):
        if stage not in by_stage:
            continue
        item = by_stage[stage]
        stages[stage] = {
            "input": item.input_document,
            "output": item.output_document,
            "input_hash": item.receipt.input_hash,
            "output_hash": item.receipt.output_hash,
            "owner": item.receipt.owner,
        }
    native_hash = content_hash(native_inputs)
    # `memo`: pre-seeded with every object `shared_documents` names, so
    # `deepcopy` below reuses THOSE specific objects by reference (its own
    # `id(x) in memo` fast path) instead of copying them -- everything else
    # still gets an independent deep copy, unchanged. Values and hashes are
    # identical either way; only which Python object holds a shared value
    # changes. See `tools/capture_tier0_corpus.py`'s `_untag_nonfinite_
    # shared`/`_SharedTraceDocuments` for why this matters for a DYN-SV
    # chooser trace, whose members can share large served Tier-4 fold pools.
    memo = {id(value): value for value in shared_documents}
    body = {
        "schema_version": TRACE_SCHEMA,
        "request": deepcopy(dict(request), memo),
        "request_hash": request_hash,
        "shared_inputs": deepcopy(dict(shared_inputs), memo),
        "shared_input_hash": shared_hash,
        "native_input_hash": native_hash,
        "native_inputs": deepcopy(dict(native_inputs), memo),
        "native_inputs_hash": native_hash,
        "input_translation": _translation(shared_inputs, native_inputs, mappings),
        "stages": stages,
        "resources": deepcopy(list(resources), memo),
        "metadata": dict(metadata or {}),
    }
    body["trace_hash"] = content_hash(body)
    return body


__all__ = [
    "NATIVE_INPUT_KEYS", "OPTIONAL_STAGES", "REQUIRED_STAGES",
    "ReleaseAssemblyError", "assemble_input_trace",
]
