"""Native scoring as the v2 shadow-serving row source (spec_ns_b, G1/G5).

``build_native_bundle_rows`` runs the real
:func:`engine.v2.scoring.application.score_one` kernel once per declared
``(ScoreRequest, NativeScoreInputs)`` pair -- each call with its own fresh
observer list, so one row's analog
:class:`~engine.v2.scoring.stages.StageObservation` channel can never leak
into another row's
:func:`~engine.v2.serving.native_render.native_display_row` call -- and keys
each display row by the real population identity
:func:`~engine.v2.serving.native_render.native_row_key` derives.

``shadow_serving_row_source`` is the one seam a composing caller switches on:
``"legacy"`` returns the legacy ``bundle_rows_by_ticker`` unchanged, and
``"native"`` regroups the keyed native display rows into that same
``{ticker: [display_row, ...]}`` shape, because that is what
:func:`engine.v2.serving.projections.build_candidate` (through
``bridge.build_bridges``) already consumes.  The native display value is
rounded to the renderer's own ``BUNDLE_PRECISION`` (6), exact display fields
excepted, because the bridge compares ``round6(engine_record) ==
display_record``: a native display row left at full precision would refuse
every candidate whose value carries more than six decimals.  The plan's own
``shadow_serving_scorer`` string is the only input; nothing here is an
environment variable or a CLI-only flag (G5).

G1: neither function writes anything.  They return dicts, and the write
boundary stays exactly where it already is -- ``build_candidate``'s own
``store``/``conn`` parameters.  This module names no
``engine.v2.ops.decision_commit``, ``...ledger_history_import``,
``...legacy_actions`` or ``engine.dashboard.nightly`` symbol (the legacy
board's own write path); ``tests/test_v2_ops_native_shadow_render.py`` walks
this module's AST and refuses each one statically.

A native scoring failure propagates uncaught: there is deliberately no
``except Exception`` and no fallback to the legacy rows, so a broken native
path fails the shadow render loudly instead of serving stale/legacy rows
under a "native" label.  ``engine.v2.ops.native_shadow_render`` is this
seam's ops half; the two packages are layer-7 peers
(``checks/import_layers.py``), so both read the allowed values and the
default from ``engine.v2.contracts.serving`` and each raises its own typed
``INVALID_REQUEST`` refusal.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from engine.v2.contracts import ScoreRequest
from engine.v2.foundation.typed import SHADOW_SERVING_SCORERS, shadow_serving_scorer
from engine.v2.scoring.application import score_one
from engine.v2.scoring.stages import NativeScoreInputs, analog_display_fields

from .bridge import LEGACY_DISPLAY_MAPPING_V1
from .native_render import native_display_row, native_row_key

__all__ = [
    "NativeShadowConfigError",
    "SHADOW_SERVING_SCORERS",
    "build_native_bundle_rows",
    "shadow_serving_row_source",
]


#: The renderer's own precision rule, read off the bridge's mapping: a display
#: row is rounded to ``dashboard/render.py``'s ``BUNDLE_PRECISION`` (6) except
#: for the fields the renderer keeps exact.  ``bridge._compare_display``
#: compares ``round6(engine_record) == display_record``, so a native display
#: row carrying full-precision floats would make ``build_candidate`` refuse
#: every row whose value has more than six decimals; rounding the display side
#: (never the engine side) is what the real renderer does.
_EXACT_DISPLAY_FIELDS = frozenset(
    spec.field for spec in LEGACY_DISPLAY_MAPPING_V1 if spec.exact)

#: Identity the serving bridge join (``bridge._join_key``) needs but a
#: :class:`~engine.v2.contracts.ScoreRecord` does not carry.  Present values
#: are copied from the native inputs context -- the answer-free source the
#: score was computed from -- never invented for a missing one.
_EVENT_FIELDS = ("ticker", "event_date")
_BRIDGE_FIELDS = ("expiry", "strike", "strike_offset", "as_of", "session")


class NativeShadowConfigError(Exception):
    """The plan's ``shadow_serving_scorer`` is not one this seam implements.

    ``code`` mirrors ``engine.v2.ops.errors.fail``'s typed envelope from the
    serving side of the layer-7 peer split: the two modules cannot import
    each other, so each validates the same two strings and reports the same
    ``INVALID_REQUEST`` code rather than a bare ``ValueError``.
    """

    code = "INVALID_REQUEST"


def _shadow_serving_mode(plan: Mapping[str, Any]) -> str:
    mode = shadow_serving_scorer(plan)
    if mode is None:
        raise NativeShadowConfigError(
            "shadow_serving_scorer must be legacy or native")
    return mode


def _identity_value(name: str, record, inputs: NativeScoreInputs) -> Any:
    """One identity fact: the native inputs context first, the record's own
    resolved request second, for an input that does not repeat the fact."""
    value = inputs.context.get(name)
    if value is None:
        value = record.resolved_request.get(name)
    return value


def _event_ref(record, inputs: NativeScoreInputs) -> dict[str, Any]:
    """``record.event_ref`` plus the event pair ``score_one`` does not stamp.

    ``ScoreRecord.event_ref`` carries only ``event_id``/``event_revision``
    (``application._record_payload``), while ``native_row_key`` -- the twin of
    the legacy population key -- reads ``ticker``/``event_date``.  The pair
    comes from the real inputs context; a context missing either leaves
    ``native_row_key`` to raise its own named ``KeyError`` rather than
    inventing a placeholder that would collide two events.
    """
    identity = {name: _identity_value(name, record, inputs) for name in _EVENT_FIELDS}
    return {**record.event_ref,
            **{name: value for name, value in identity.items() if value is not None}}


def _render_value(value: Any) -> Any:
    """One display value at the renderer's precision (``bridge._round6``)."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_render_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item) for key, item in value.items()}
    return value


def _rendered_row(row: dict) -> dict:
    """A display row rounded exactly as ``engine.dashboard.render`` rounds it."""
    return {name: (value if name in _EXACT_DISPLAY_FIELDS else _render_value(value))
            for name, value in row.items()}


def _display_row(record, inputs: NativeScoreInputs, observations: list) -> dict:
    row = native_display_row(record, analog_fields=analog_display_fields(observations))
    for name in (*_EVENT_FIELDS, *_BRIDGE_FIELDS):
        value = _identity_value(name, record, inputs)
        if value is not None:
            row[name] = value
    return _rendered_row(row)


def build_native_bundle_rows(
    score_document: Mapping[str, Any],
    requests_by_key: Mapping[str, tuple[ScoreRequest, NativeScoreInputs]],
) -> dict[str, dict]:
    """One real ``score_one`` call per pair, keyed by population identity.

    ``score_document`` is the caller's shadow score document; it is accepted
    (and threaded by ``shadow_serving_row_source``) so the seam keeps one call
    shape, but the native rows themselves come only from the real
    ``score_one`` records -- never from a document value.

    Each call gets a fresh observer list; ``analog_display_fields`` reads the
    analog stage's display-only row ids from that one call's observations.
    The returned display rows carry the renderer's precision (see
    ``_rendered_row``), so the bridge's ``round6`` comparison accepts the
    candidate when ``score_document`` supplies the full-precision native
    record.  Every ``score_one`` exception propagates uncaught: no fallback
    to legacy rows, no swallowed failure.
    """
    rows: dict[str, dict] = {}
    for request, inputs in requests_by_key.values():
        observations: list = []
        record = score_one(request, inputs, observer=observations.append)
        key = native_row_key(replace(record, event_ref=_event_ref(record, inputs)))
        rows[key] = _display_row(record, inputs, observations)
    return rows


def _rows_by_ticker(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict]]:
    """Regroup keyed native rows into ``bundle_rows_by_ticker``'s real shape."""
    grouped: dict[str, list[dict]] = {}
    for row in rows.values():
        grouped.setdefault(str(row["ticker"]), []).append(dict(row))
    return grouped


def shadow_serving_row_source(
    plan: Mapping[str, Any],
    score_document: Mapping[str, Any],
    bundle_rows_by_ticker: dict[str, list[dict]],
    requests_by_key: Mapping[str, tuple[ScoreRequest, NativeScoreInputs]],
) -> dict[str, list[dict]]:
    """The one ``if`` the whole G5 seam turns on.

    A future authority switch (Phase 7) only ever changes what this function
    returns; ``build_candidate``, ``bridge.py`` and ``projections.py`` never
    need to know which scorer produced the rows.  ``"legacy"`` returns the
    caller's own bundle rows by identity; ``"native"`` builds fresh rows from
    the real scoring kernel and regroups them into the same
    ``{ticker: [row, ...]}`` shape.
    """
    if _shadow_serving_mode(plan) == "legacy":
        return bundle_rows_by_ticker
    return _rows_by_ticker(build_native_bundle_rows(score_document, requests_by_key))
