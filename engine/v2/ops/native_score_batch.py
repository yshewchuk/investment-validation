"""Cutover PR-3 batch assembler.

Turns one ``ScoringReleaseBinding`` (PR-1, ``engine/v2/scoring/
release_bindings.py``) plus per-event staged inputs (PR-2, ``engine/v2/
scoring/nightly_source_bundle.py``) into ``{BoardRequest: (ScoreRequest,
NativeScoreInputs)}``, bounded to ``STR-THRU`` only: batch-level malformed
input raises, every per-row gap is a collected
:class:`NativeScoreBatchRowRefusal`, and ``run_native_score_batch_worker``
scores the assembled batch under ``no_fit_guard``. There is no production
caller yet -- see ``engine/v2/ops/ARCHITECTURE.md``'s
``native_score_batch.py`` section for the full design.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from engine.v2.contracts import ScoreBatch, ScoreRequest
from engine.v2.foundation import content_hash, to_document
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.scoring.application import score_batch
from engine.v2.scoring.identity import request_hash
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal,
    assemble_nightly_source_bundle,
    validated_as_of,
)
from engine.v2.scoring.release_bindings import ScoringReleaseBinding
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs
from engine.v2.scoring.stages import NativeScoreInputs

__all__ = [
    "NativeScoreBatchRowRefusal",
    "NightlyEventInputs",
    "assemble_score_batch_inputs",
    "run_native_score_batch_worker",
]

#: This bounded batch assembler's one supported strategy (matches
#: nightly_source_bundle.py's own bounded-builder scope).
_SUPPORTED_STRATEGY = "STR-THRU"
_SHADOW_FILL_ALPHA = 0.5  # engine.fills.MID.alpha; do not import engine.fills
                          # (legacy) from this v2 module -- this is a shadow
                          # default, documented in ARCHITECTURE.md, not a
                          # legacy-derived fact.


@dataclass(frozen=True, slots=True)
class NightlyEventInputs:
    """One (ticker, event, strategy)'s already-staged inputs for batch assembly."""

    key: BoardRequest
    calendar_row: Mapping[str, Any]
    panel_row: Mapping[str, Any]
    #: Issue #53's fix (PR #67): the caller-declared upper bound on when
    #: every one of panel_row's market-state feature values was actually
    #: observed -- FeatureVector.as_of for a forward event, panel_row["date"]
    #: for an already-realized one. Passed straight through to
    #: assemble_nightly_source_bundle's own required panel_anchor parameter;
    #: this module derives nothing about it itself.
    panel_anchor: Any
    tier4_row: Mapping[str, Any]
    quote_rows: Sequence[Mapping[str, Any]]
    quote_status: Any = None


class NativeScoreBatchRowRefusal(ValueError):
    """One row's typed refusal. Collected, never raised, by
    :func:`assemble_score_batch_inputs` -- one bad row never sinks the batch.
    """

    def __init__(self, key: BoardRequest, code: str, detail: str) -> None:
        self.key = key
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {key!r}: {detail}")

    def as_document(self) -> dict[str, Any]:
        return {
            "key": {
                "ticker": self.key.ticker,
                "strategy": self.key.strategy,
                "event_date": _event_date_identity(self.key.event_date),
                "session": self.key.session,
            },
            "code": self.code,
            "detail": self.detail,
        }


def _board_request_key(key: BoardRequest) -> str:
    """The canonical, JSON-object-key-safe string identity for one row:
    ``f"{ticker}|{strategy}|{event_date_identity}|{session}"``. The
    ``event_date_identity`` component (the strict :func:`
    _event_date_identity` form -- ``YYYY-MM-DD`` for a midnight/day event,
    canonical naive ISO datetime for an intraday one) can never contain the
    ``"|"`` join delimiter, but ``ticker``/``strategy``/``session`` are
    free-text-shaped inputs this module does not control at the source.
    Raises a :class:`NativeScoreBatchRowRefusal` (code
    ``INVALID_KEY_FIELD``) the moment ``"|"`` appears in any of those three
    fields, BEFORE the row is ever encoded into ``records.json``'s/
    ``refusals.json``'s own keys -- this makes the encoding a true
    bijection for every row that does get a canonical key.
    """
    for field_name, value in (
        ("ticker", key.ticker), ("strategy", key.strategy), ("session", key.session),
    ):
        if "|" in value:
            raise NativeScoreBatchRowRefusal(
                key, "INVALID_KEY_FIELD",
                f"{field_name} contains the canonical key delimiter")
    return f"{key.ticker}|{key.strategy}|{_event_date_identity(key.event_date)}|{key.session}"


def _iso(value: Any) -> str | None:
    """One date-shaped value as an ISO date string, ``None`` passed through."""
    if value is None:
        return None
    return str(pd.Timestamp(value).date())


#: The one fixed message every strict event-date identity rejection carries
#: -- never an echo of the rejected value (``refusals.json`` and
#: ``producer_refusals.json`` decode errors surface on published-output
#: paths; CWE-209 discipline, as everywhere else in this module).
_EVENT_DATE_IDENTITY_ERROR = (
    "event_date must be a naive calendar date/day value or a naive "
    "datetime, encoded as YYYY-MM-DD (midnight) or a canonical ISO datetime")


def _event_date_identity(value: Any) -> str:
    """One ``BoardRequest.event_date`` as its canonical identity component.

    Issue #356: a midnight/day value keeps the legacy ``YYYY-MM-DD`` form
    (every existing wire/key identity is byte-identical), while a non-
    midnight naive datetime encodes as its full canonical ISO datetime
    (``YYYY-MM-DDTHH:MM:SS``, microseconds appended only when set) -- the
    instant's stated naive wall-clock fields are preserved, never
    normalized to UTC or truncated to a calendar day, so midnight and
    intraday events on the same day keep DISTINCT identities and equal
    instants keep ONE identity. Strings must already be in canonical form
    (parse, then re-encode, then round-trip against the input), which
    rejects relative strings like ``"today"``/``"now"`` (they would resolve
    to wall-clock time at gate-run time instead of from the document's
    content), offset/Z values, and every malformed or non-canonical shape.
    Timezone-aware values and non-date-shaped values raise a fixed
    ``ValueError``.
    """
    if isinstance(value, str):
        canonical_input: str | None = value
    else:
        if not isinstance(value, (date, datetime)) \
                or getattr(value, "tzinfo", None) is not None:
            raise ValueError(_EVENT_DATE_IDENTITY_ERROR)
        canonical_input = None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise ValueError(_EVENT_DATE_IDENTITY_ERROR) from None
    if pd.isna(timestamp) or getattr(timestamp, "tz", None) is not None:
        raise ValueError(_EVENT_DATE_IDENTITY_ERROR)
    encoded = (timestamp.date().isoformat()
               if timestamp.normalize() == timestamp else timestamp.isoformat())
    if canonical_input is not None and encoded != canonical_input:
        raise ValueError(_EVENT_DATE_IDENTITY_ERROR)
    return encoded


def _matched_decision_clock(
    _key: BoardRequest,
    model_identity: Mapping[str, Any],
    strategy: str,
) -> tuple[str | None, str | None, tuple[Any, Any] | None]:
    """The one decision clock this assembler assumes per strategy, or a refusal.

    Returns a tagged 3-tuple: ``(code, detail, None)`` on failure (missing
    ``driver``/``gate`` role, or the two roles disagreeing on
    ``decision_clock_id``), ``(None, None, (driver_identity, gate_identity))``
    on success. The row's ``key`` is passed in for the caller's error
    context; the refusal codes/details are row-scoped by that caller.
    """
    driver_identity = model_identity.get(f"driver:{strategy}")
    gate_identity = model_identity.get(f"gate:{strategy}")
    if driver_identity is None:
        return ("RELEASE_MISSING_ROLE", f"driver:{strategy}", None)
    if gate_identity is None:
        return ("RELEASE_MISSING_ROLE", f"gate:{strategy}", None)
    if driver_identity.decision_clock_id != gate_identity.decision_clock_id:
        return ("AMBIGUOUS_DECISION_CLOCK",
                f"driver:{strategy}={driver_identity.decision_clock_id!r} "
                f"gate:{strategy}={gate_identity.decision_clock_id!r}", None)
    return (None, None, (driver_identity, gate_identity))


def _identity_context(as_of: Any, snapshot_id: str,
                       calendar_row: Mapping[str, Any]) -> dict[str, Any]:
    """The MC-seed identity fields ``stages._model_seed`` needs, merged into
    ``bundle.context`` before ``build_native_score_inputs`` -- see
    ARCHITECTURE.md's MC-seed note for the shadow defaults' provenance."""
    return {
        "snapshot": snapshot_id,
        "requested_as_of": _iso(as_of),
        "requested_event_date": _iso(calendar_row["event_date"]),
        "requested_strike": None,
        "requested_expiry": _iso(calendar_row["expiry"]),
        "fill_alpha": _SHADOW_FILL_ALPHA,
        "variant": None,
        "decision_offset": None,
        "quote_max_age_sessions": None,
        "chain_as_of": _iso(as_of),
    }


def _calendar_row_problem(key: BoardRequest, calendar_row: Any) -> tuple[str, str] | None:
    """``None`` if ``calendar_row`` is a well-formed mapping whose own
    ``ticker``/``event_date``/``expiry`` are parseable and match ``key``,
    else ``(code, fixed_detail)`` for a per-row refusal. Checked first,
    before every other per-row check (CodeRabbit rounds 2-5, PR #66):

    - Not a mapping at all (e.g. a null ``calendar_row`` in ``events.json``)
      -- ``CALENDAR_ROW_INVALID``. Nothing upstream of this module checks
      this; calling ``.get`` on a non-mapping would otherwise raise
      ``AttributeError`` and abort the whole batch before
      ``assemble_nightly_source_bundle`` gets a chance to refuse it.
    - An unparseable ``event_date`` or ``expiry`` -- ``CALENDAR_ROW_INVALID``.
      ``assemble_nightly_source_bundle`` never parses either itself (it
      copies both straight into ``context``), and this module's own
      ``_identity_context`` parses ``expiry`` later, outside every
      try/except in :func:`_assemble_one_event` -- an unparseable value
      must be caught here, before that point, not there.
    - ``ticker``/``event_date`` not matching ``key`` -- ``CALENDAR_ROW_KEY_
      MISMATCH``. Neither this module nor ``assemble_nightly_source_bundle``
      (which only checks ``panel_row`` against ``calendar_row``, never
      against the caller's ``BoardRequest``) verifies this elsewhere, and
      ``ScoreRequest`` carries no ticker/event_date of its own -- a
      caller-side pairing bug would otherwise pass silently and could
      produce a duplicate ``request_hash`` (see
      :func:`assemble_score_batch_inputs`) that corrupts a different row.

    Every detail here is a FIXED string, never the raw staged value
    (CodeRabbit round 4, CWE-209) -- ``refusals.json`` is a published
    output of a successful attempt.
    """
    if not isinstance(calendar_row, Mapping):
        return "CALENDAR_ROW_INVALID", "calendar row is missing or not an object"
    try:
        calendar_event_date = _iso(calendar_row.get("event_date"))
        _iso(calendar_row.get("expiry"))
    except (TypeError, ValueError):
        return "CALENDAR_ROW_INVALID", "calendar row has an unparseable date field"
    if calendar_row.get("ticker") != key.ticker or calendar_event_date != _iso(key.event_date):
        return "CALENDAR_ROW_KEY_MISMATCH", "calendar row does not match the request key"
    return None


def _bundle_or_refusal(
    key: BoardRequest,
    event: NightlyEventInputs,
    *,
    strategy: str,
    driver_identity: Any,
    gate_identity: Any,
    gate_policy: Mapping[str, Mapping[str, Any]],
    as_of: Any,
    feature_names: Sequence[str],
    driver_name: str,
) -> SourceBundle | NativeScoreBatchRowRefusal:
    """``assemble_nightly_source_bundle``'s result, or a re-wrapped refusal.

    Split out of :func:`_assemble_one_event` purely to keep it under its
    line budget. ``exc.detail`` is never published (CodeRabbit round 5,
    CWE-209): it can itself echo staged input (e.g. ``quote_domain_map``
    embeds an invalid ``quote_status`` string into some refusals) --
    ``exc.code`` is a closed, module-controlled vocabulary and is safe to
    publish unchanged; the free-text detail is not.
    """
    try:
        return assemble_nightly_source_bundle(
            source_ref=f"native-score-batch:{key.ticker}:{_iso(key.event_date)}:{strategy}",
            strategy=strategy, as_of=as_of,
            calendar_row=event.calendar_row, panel_row=event.panel_row,
            panel_anchor=event.panel_anchor,
            tier4_row=event.tier4_row, quote_rows=event.quote_rows,
            quote_status=event.quote_status, feature_names=feature_names,
            driver_name=driver_name,
            model_identity={f"driver:{strategy}": to_document(driver_identity),
                            f"gate:{strategy}": to_document(gate_identity)},
            model_artifact_refs={"driver_prediction": driver_identity.artifact_hash},
            forecast_recipes={"driver_prediction": {"binding_id": driver_identity.binding_id}},
            gate_recipe={**gate_policy[strategy], "binding_id": gate_identity.binding_id},
        )
    except NightlySourceBundleRefusal as exc:
        return NativeScoreBatchRowRefusal(
            key, exc.code, f"nightly_source_bundle refused: {exc.code}")


def _assemble_one_event(
    event: NightlyEventInputs,
    *,
    binding: ScoringReleaseBinding,
    gate_policy: Mapping[str, Mapping[str, Any]],
    as_of: Any,
    snapshot_id: str,
    calendar_revision: str,
    feature_names: Sequence[str],
    driver_name: str,
) -> tuple[ScoreRequest, NativeScoreInputs] | NativeScoreBatchRowRefusal:
    """One event's ``(ScoreRequest, NativeScoreInputs)`` pair, or its refusal.

    Split out of :func:`assemble_score_batch_inputs` (which owns the batch-
    level checks and the loop) purely to keep each function under its line/
    complexity budget -- see that function's docstring for the refusal codes.
    """
    key = event.key
    try:
        _board_request_key(key)
    except NativeScoreBatchRowRefusal as exc:
        return exc
    strategy = key.strategy
    problem = _calendar_row_problem(key, event.calendar_row)
    if problem is not None:
        return NativeScoreBatchRowRefusal(key, *problem)
    if strategy != _SUPPORTED_STRATEGY:
        return NativeScoreBatchRowRefusal(
            key, "UNSUPPORTED_STRATEGY",
            f"only {_SUPPORTED_STRATEGY} is supported, got {strategy!r}")
    code, detail, pair = _matched_decision_clock(key, binding.model_identity, strategy)
    if code is not None:
        return NativeScoreBatchRowRefusal(key, code, detail)
    driver_identity, gate_identity = pair
    if strategy not in gate_policy:
        return NativeScoreBatchRowRefusal(
            key, "GATE_POLICY_NOT_STAGED", f"no gate policy staged for {strategy!r}")
    bundle = _bundle_or_refusal(
        key, event, strategy=strategy, driver_identity=driver_identity,
        gate_identity=gate_identity, gate_policy=gate_policy, as_of=as_of,
        feature_names=feature_names, driver_name=driver_name)
    if isinstance(bundle, NativeScoreBatchRowRefusal):
        return bundle
    event_id = event.calendar_row.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        return NativeScoreBatchRowRefusal(
            key, "MISSING_STAGED_INPUT", "calendar_row missing event_id")
    bundle = replace(
        bundle, context={**bundle.context,
                         **_identity_context(as_of, snapshot_id, event.calendar_row)},
        model_release=binding.model_release, frozen_inference=binding.frozen_inference,
    )
    try:
        native_inputs = build_native_score_inputs(bundle)
    except ValueError:
        # CodeRabbit round 4 (PR #66, CWE-209): a fixed message, not str(exc)
        # -- build_native_score_inputs' own ValueError text can name staged
        # recipe/field shapes, and refusals.json is a published output.
        return NativeScoreBatchRowRefusal(
            key, "NATIVE_INPUT_BUILD_FAILED", "native input build failed")
    request = ScoreRequest(
        event_id=str(event_id), calendar_revision=str(calendar_revision),
        strategy_version=strategy, deployment_id=binding.model_release.deployment_id,
        decision_clock_id=driver_identity.decision_clock_id,
        requested_decision_at=_iso(as_of) or "", snapshot_id=str(snapshot_id),
        mode="shadow", fill_model={"alpha": _SHADOW_FILL_ALPHA},
        model_artifact_refs=tuple(dict.fromkeys(
            (driver_identity.artifact_hash, gate_identity.artifact_hash))),
    )
    return (request, native_inputs)


def _checked_batch_arguments(
    *,
    binding: Any,
    snapshot_id: Any,
    calendar_revision: Any,
    as_of: Any,
    events: Sequence[Any],
) -> tuple[NightlyEventInputs, ...]:
    """Validate every batch-level (shared-by-every-row) argument to
    :func:`assemble_score_batch_inputs`, or raise.

    Split out purely to keep :func:`assemble_score_batch_inputs` under its
    line budget -- see that function's own docstring for why each of these
    checks exists and raises rather than becoming a per-row refusal.
    """
    if not isinstance(binding, ScoringReleaseBinding):
        raise TypeError("binding must be a ScoringReleaseBinding")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("snapshot_id must be a non-empty string")
    if not isinstance(calendar_revision, str) or not calendar_revision:
        raise ValueError("calendar_revision must be a non-empty string")
    try:
        validated_as_of(as_of, label="as_of")
    except NightlySourceBundleRefusal as exc:
        raise ValueError(f"invalid as_of: {exc.code}: {exc.detail}") from exc
    events = tuple(events)
    if any(not isinstance(event, NightlyEventInputs) for event in events):
        raise TypeError("events must be a sequence of NightlyEventInputs")
    keys = [event.key for event in events]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate BoardRequest key in events")
    return events


def assemble_score_batch_inputs(
    *,
    as_of: Any,
    snapshot_id: str,
    calendar_revision: str,
    binding: ScoringReleaseBinding,
    events: Sequence[NightlyEventInputs],
    feature_names: Sequence[str],
    gate_policy: Mapping[str, Mapping[str, Any]] | None = None,
    driver_name: str = "abs_move",
) -> tuple[dict[BoardRequest, tuple[ScoreRequest, NativeScoreInputs]],
           tuple[NativeScoreBatchRowRefusal, ...]]:
    """Assemble ``{BoardRequest: (ScoreRequest, NativeScoreInputs)}`` for a batch.

    Batch-level malformed input -- a non-``ScoringReleaseBinding`` ``binding``,
    a non-``NightlyEventInputs`` item in ``events``, or two events sharing one
    ``BoardRequest`` key -- raises (the whole call is meaningless); every
    per-row gap is a collected :class:`NativeScoreBatchRowRefusal` instead, so
    one bad row never sinks the batch: ``UNSUPPORTED_STRATEGY`` (not
    ``STR-THRU``), ``RELEASE_MISSING_ROLE``/``AMBIGUOUS_DECISION_CLOCK`` (the
    release's driver/gate identities for the strategy), ``GATE_POLICY_NOT_
    STAGED`` (no caller-supplied threshold for the strategy -- gate thresholds
    are not part of the release binding), any re-wrapped
    ``NightlySourceBundleRefusal`` from the per-event bundle assembly,
    ``MISSING_STAGED_INPUT`` (no ``event_id`` in the calendar row),
    ``CALENDAR_ROW_INVALID`` (``calendar_row`` is not a mapping, or its
    ``event_date``/``expiry`` do not parse), ``CALENDAR_ROW_KEY_MISMATCH``
    (``calendar_row``'s own ``ticker``/``event_date`` does not match its
    ``NightlyEventInputs.key``) and ``NATIVE_INPUT_BUILD_FAILED`` (a
    ``ValueError`` from ``build_native_score_inputs``). Per-row assembly
    itself lives in :func:`_assemble_one_event`.

    A colliding ``request_hash`` across two DIFFERENT ``BoardRequest`` keys
    (CodeRabbit round 2, PR #66) also raises ``ValueError`` rather than
    becoming a per-row refusal: ``ScoreRequest`` carries no ticker/event_date
    of its own, so two distinct rows whose ``ScoreRequest`` fields coincide
    (most plausibly a duplicated ``calendar_row["event_id"]``, since that is
    the one genuinely per-event field) would otherwise silently collide in
    ``run_native_score_batch_worker``'s ``fields_by_request`` map -- one row
    clobbering the other's inputs with no visible refusal for either.
    Nothing here can say which of the two rows is "the bad one", so this is
    batch-level, not a row-level refusal.

    ``as_of``/``snapshot_id``/``calendar_revision`` are batch-level, shared-
    by-every-row arguments validated once, here, before any row is
    attempted (Opus gate, PR #66; see :func:`_checked_batch_arguments`): a
    bad ``as_of`` would otherwise only surface once ``assemble_nightly_
    source_bundle`` re-validates it inside each row, refusing every row
    individually while the attempt still reports success, and a bad
    ``snapshot_id``/``calendar_revision`` would otherwise flow straight
    into every ``ScoreRequest`` via ``str()``.
    """
    events = _checked_batch_arguments(
        binding=binding, snapshot_id=snapshot_id, calendar_revision=calendar_revision,
        as_of=as_of, events=events)
    gate_policy = dict(gate_policy or {})

    results: dict[BoardRequest, tuple[ScoreRequest, NativeScoreInputs]] = {}
    refusals: list[NativeScoreBatchRowRefusal] = []
    seen_hashes: dict[str, BoardRequest] = {}
    for event in events:
        outcome = _assemble_one_event(
            event, binding=binding, gate_policy=gate_policy, as_of=as_of,
            snapshot_id=snapshot_id, calendar_revision=calendar_revision,
            feature_names=feature_names, driver_name=driver_name,
        )
        if isinstance(outcome, NativeScoreBatchRowRefusal):
            refusals.append(outcome)
            continue
        request, _ = outcome
        row_hash = request_hash(request)
        if row_hash in seen_hashes:
            raise ValueError(
                f"duplicate request_hash {row_hash} for {event.key!r} and "
                f"{seen_hashes[row_hash]!r}")
        seen_hashes[row_hash] = event.key
        results[event.key] = outcome
    return results, tuple(refusals)


def _event_inputs_from_document(doc: Mapping[str, Any]) -> NightlyEventInputs:
    """Decode one ``events.json`` array item into :class:`NightlyEventInputs`.

    Date-shaped values are passed straight through as plain ISO strings --
    ``assemble_nightly_source_bundle`` already accepts them (exactly as
    ``tests/test_v2_scoring_nightly_source_bundle.py``'s own fixtures do).
    """
    key = BoardRequest(
        ticker=str(doc["key"]["ticker"]), strategy=str(doc["key"]["strategy"]),
        event_date=pd.Timestamp(doc["key"]["event_date"]),
        session=str(doc["key"]["session"]))
    return NightlyEventInputs(
        key=key, calendar_row=doc["calendar_row"], panel_row=doc["panel_row"],
        panel_anchor=doc["panel_anchor"],
        tier4_row=doc["tier4_row"], quote_rows=doc["quote_rows"],
        quote_status=doc.get("quote_status"))


def _keyed_by_board_request(items: Any) -> dict[str, Any]:
    """Key an iterable of ``(BoardRequest, value)`` pairs by
    :func:`_board_request_key`, raising ``ValueError`` on a duplicate
    canonical identity instead of silently letting the later pair overwrite
    the earlier one.

    ``_event_date_identity`` preserves the instant's naive wall-clock
    fields (midnight/day keeps the legacy ``YYYY-MM-DD`` form, intraday
    keeps its full datetime form, issue #356), so two rows that differ only
    by time of day have DISTINCT identities here; what this guard still
    catches is two rows sharing ONE identity (equal instants, however
    expressed) -- a plain ``ValueError`` batch-level failure, matching this
    module's existing ``duplicate request_hash`` batch-level check in
    :func:`assemble_score_batch_inputs`, never a per-row refusal: nothing
    here can say which of the two colliding rows is "the bad one".
    """
    keyed: dict[str, Any] = {}
    for key, value in items:
        canonical_key = _board_request_key(key)
        if canonical_key in keyed:
            raise ValueError(f"duplicate canonical BoardRequest key: {canonical_key}")
        keyed[canonical_key] = value
    return keyed


#: A producer refusal's detail is upstream, pipeline-generated text (the
#: same trust level as events.json, not external/adversarial input), but
#: refusals.json is a published artifact -- cap it defensively so one
#: malformed producer detail can never dump an unbounded blob into it.
_MAX_PRODUCER_DETAIL_LENGTH = 500

#: The only ``event_date`` wire shapes ``producer_refusals.json`` may carry
#: -- exactly what this module's own ``as_document()``/``_board_request_key``
#: write through :func:`_event_date_identity`: the legacy ``YYYY-MM-DD`` day
#: form for a midnight/day event, and the full canonical naive ISO datetime
#: form for an intraday one. Relative strings like "now"/"today" parse fine
#: under ``pd.Timestamp`` but resolve to wall-clock time, which would make a
#: merged refusal's identity depend on when the gate runs instead of on the
#: document's content; offset/Z strings would normalize an instant onto a
#: different naive wall clock. Both are rejected (issue #356) -- a decoder
#: identity must equal the producer's own, never a truncation of it.


def _validated_producer_refusal_fields(index: int, item: Any) -> tuple[Mapping[str, Any], str, str]:
    """Validate one ``producer_refusals.json`` item's shape and field types
    -- presence, then type/value -- and return ``(raw_key, code, detail)``
    ready for ``BoardRequest`` construction. Raises ``ValueError`` (a
    fixed, index-naming message) for any problem; this is the stage split
    out of :func:`_decode_producer_refusals` to keep that function's own
    complexity within budget.
    """
    if not isinstance(item, Mapping):
        raise ValueError(
            f"producer_refusals.json refusal at index {index} must be an object")
    raw_key = item.get("key")
    if not isinstance(raw_key, Mapping):
        raise ValueError(
            f"producer_refusals.json refusal at index {index} has no \"key\" object")
    for field_name in ("ticker", "strategy", "event_date", "session"):
        if field_name not in raw_key:
            raise ValueError(
                f"producer_refusals.json refusal at index {index} key is "
                f"missing {field_name!r}")
    if "code" not in item:
        raise ValueError(
            f"producer_refusals.json refusal at index {index} is missing \"code\"")
    if "detail" not in item:
        raise ValueError(
            f"producer_refusals.json refusal at index {index} is missing \"detail\"")
    for field_name in ("ticker", "strategy", "event_date", "session"):
        if not isinstance(raw_key[field_name], str) or not raw_key[field_name]:
            raise ValueError(
                f"producer_refusals.json refusal at index {index} key "
                f"{field_name!r} must be a non-empty string")
    if not isinstance(item["code"], str) or not item["code"]:
        raise ValueError(
            f"producer_refusals.json refusal at index {index} \"code\" "
            f"must be a non-empty string")
    if not isinstance(item["detail"], str):
        raise ValueError(
            f"producer_refusals.json refusal at index {index} \"detail\" "
            f"must be a string")
    event_date_error = (
        f"producer_refusals.json refusal at index {index} key "
        f"\"event_date\" must be a canonical YYYY-MM-DD date or canonical "
        f"naive ISO datetime string")
    try:
        # Strict round trip through the SAME formatter this module's own
        # keys/documents encode with: only strings that are already their
        # own canonical identity survive (issue #356).
        _event_date_identity(raw_key["event_date"])
    except ValueError:
        raise ValueError(event_date_error) from None
    return raw_key, str(item["code"]), str(item["detail"])


def _decode_producer_refusals(doc: Mapping[str, Any]) -> tuple[NativeScoreBatchRowRefusal, ...]:
    """Decode ``producer_refusals.json``'s v1.0 document into the same typed
    per-row refusal shape ``run_native_score_batch_worker``'s own per-row
    loop already produces, so merging it reuses ``_native_score_batch_documents``'s
    existing records/refusals collision check UNCHANGED -- see that
    function's own docstring. A malformed document (wrong/missing
    ``schema_version``, a non-list ``"refusals"``, or a malformed item) is a
    whole-call ``ValueError``, matching this module's existing
    "events.json must be a JSON array" discipline for malformed caller
    input -- never a silently-dropped item.
    """
    if (not isinstance(doc, Mapping)
            or doc.get("schema_version") != "native_score_batch_producer_refusals.v1.0"):
        raise ValueError(
            "producer_refusals.json must be a "
            "native_score_batch_producer_refusals.v1.0 document")
    items = doc.get("refusals")
    if not isinstance(items, list):
        raise ValueError("producer_refusals.json's \"refusals\" must be a list")
    decoded = []
    for index, item in enumerate(items):
        raw_key, code, detail = _validated_producer_refusal_fields(index, item)
        key = BoardRequest(
            ticker=str(raw_key["ticker"]), strategy=str(raw_key["strategy"]),
            event_date=pd.Timestamp(raw_key["event_date"]), session=str(raw_key["session"]))
        if code == "INVALID_KEY_FIELD" and not any(
                "|" in value for value in (key.ticker, key.strategy, key.session)):
            raise ValueError(
                "producer refusal claims INVALID_KEY_FIELD for an encodable key")
        detail = detail[:_MAX_PRODUCER_DETAIL_LENGTH]
        decoded.append(NativeScoreBatchRowRefusal(key, code, detail))
    return tuple(decoded)


def _native_score_batch_documents(
    keys_in_order: Sequence[BoardRequest],
    records: Sequence[Any],
    refusals: Sequence[NativeScoreBatchRowRefusal],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the v2.0 ``records.json``/``refusals.json`` documents.

    Split out of :func:`run_native_score_batch_worker` purely to keep that
    function under its line budget -- see that function's docstring for the
    full contract.
    """
    # score_batch/score_many preserve request order and identity (see their
    # own docstrings), and the caller builds batch.requests from this SAME
    # assembled.values() iteration that keys_in_order comes from -- so
    # zipping keys_in_order (assembled's own dict-order keys) against
    # records is a safe, exact positional pairing within this one call.
    # This is NOT the events.json-vs-records.json positional pairing the
    # PR-7a design names as broken (that hazard is about reconstructing
    # order from the ORIGINAL per-event array after refusals have been
    # dropped; here nothing has been dropped or reordered between the two
    # zipped sequences).
    #
    # Issue #53 (the post-as_of panel-feature anchor gap) is fixed
    # upstream (#67): assemble_nightly_source_bundle's own required
    # panel_anchor parameter refuses POST_AS_OF_ROW per-row before a
    # bundle is ever built here. Nothing populates known_gaps today;
    # the key stays in the schema for a future gap this module might
    # need to flag.
    records_by_key = _keyed_by_board_request(
        (key, to_document(record))
        for key, record in zip(keys_in_order, records, strict=True)
    )
    unkeyable_refusals: list[dict[str, Any]] = [
        # No safe canonical string exists for these rows (that is what
        # "INVALID_KEY_FIELD" means) -- keep their raw structured key,
        # never a joined string that could collide or fail to re-parse.
        refusal.as_document() for refusal in refusals
        if refusal.code == "INVALID_KEY_FIELD"
    ]
    keyed_refusals = _keyed_by_board_request(
        (refusal.key, {"code": refusal.code, "detail": refusal.detail})
        for refusal in refusals if refusal.code != "INVALID_KEY_FIELD"
    )
    overlap = set(records_by_key) & set(keyed_refusals)
    if overlap:
        # The same duplicate-identity collision _keyed_by_board_request
        # already catches WITHIN one dict can also happen ACROSS the two:
        # one row with one identity succeeded (a record) while another row
        # with that SAME identity failed (a refusal), and neither dict's own
        # internal check can see the other dict at all.
        raise ValueError(
            "canonical BoardRequest key used by both a record and a "
            f"refusal: {sorted(overlap)!r}")
    records_document = {
        "schema_version": "native_score_batch_records.v2.0",
        "authoritative": False,
        "known_gaps": [],
        "records": records_by_key,
    }
    refusals_document = {
        "schema_version": "native_score_batch_refusals.v2.0",
        "refusals": keyed_refusals,
        "unkeyable_refusals": unkeyable_refusals,
    }
    return records_document, refusals_document


def run_native_score_batch_worker(parameters: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The ``native_score_batch`` job kind's worker entrypoint.

    Resolves the release binding once from ``parameters["release_root"]``,
    decodes the staged ``events.json`` per-event input array, assembles the
    batch with :func:`assemble_score_batch_inputs`, scores it under
    ``engine.v2.models.no_fit.no_fit_guard``, and writes ``records.json`` /
    ``refusals.json`` into ``root`` -- see ARCHITECTURE.md's
    ``native_score_batch.py`` section for the full contract.
    """
    from engine.v2.models.no_fit import no_fit_guard
    from engine.v2.scoring.release_bindings import resolve_release_binding

    events_doc = json.loads((root / "events.json").read_text())
    if not isinstance(events_doc, list):
        raise ValueError("events.json must be a JSON array")
    events = tuple(_event_inputs_from_document(item) for item in events_doc)
    binding = resolve_release_binding(parameters["release_root"])
    as_of = parameters["as_of"]
    snapshot_id = parameters["snapshot_id"]
    calendar_revision = parameters["calendar_revision"]
    assembled, refusals = assemble_score_batch_inputs(
        as_of=as_of, snapshot_id=snapshot_id,
        calendar_revision=calendar_revision, binding=binding,
        events=events, feature_names=tuple(parameters["feature_names"]),
        gate_policy=parameters.get("gate_policy") or {},
    )
    producer_refusals_path = root / "producer_refusals.json"
    if producer_refusals_path.exists():
        producer_doc = json.loads(producer_refusals_path.read_text(encoding="utf-8"))
        refusals = refusals + _decode_producer_refusals(producer_doc)
    seen_unkeyable_identities: set[tuple[str, str, str, str]] = set()
    for refusal in refusals:
        if refusal.code != "INVALID_KEY_FIELD":
            continue
        identity = (refusal.key.ticker, refusal.key.strategy,
                    _event_date_identity(refusal.key.event_date), refusal.key.session)
        if identity in seen_unkeyable_identities:
            raise ValueError(
                f"duplicate unkeyable refusal identity: {identity!r}")
        seen_unkeyable_identities.add(identity)
    fields_by_request = {request_hash(request): inputs
                         for request, inputs in assembled.values()}
    batch_id = content_hash({
        "as_of": as_of, "snapshot_id": snapshot_id,
        "calendar_revision": calendar_revision,
    })
    keys_in_order = tuple(assembled.keys())
    batch = ScoreBatch(batch_id=batch_id, requests=tuple(r for r, _ in assembled.values()),
                       population_ref=snapshot_id)
    with no_fit_guard():
        records = score_batch(batch, fields_by_request)
    records_document, refusals_document = _native_score_batch_documents(
        keys_in_order, records, refusals)
    (root / "records.json").write_text(json.dumps(records_document, sort_keys=True,
                                                  separators=(",", ":")))
    (root / "refusals.json").write_text(json.dumps(
        refusals_document, sort_keys=True, separators=(",", ":")))
    return {
        "outputs": [
            {"name": "records", "path": "records.json",
             "schema": "native_score_batch_records.v2.0"},
            {"name": "refusals", "path": "refusals.json",
             "schema": "native_score_batch_refusals.v2.0"},
        ],
        "completed_ids": list(parameters["expected_ids"]),
        "no_work": not parameters["expected_ids"],
        "refused": [refusal.code for refusal in refusals],
    }
