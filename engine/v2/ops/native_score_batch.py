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
                "event_date": pd.Timestamp(self.key.event_date).date().isoformat(),
                "session": self.key.session,
            },
            "code": self.code,
            "detail": self.detail,
        }


def _iso(value: Any) -> str | None:
    """One date-shaped value as an ISO date string, ``None`` passed through."""
    if value is None:
        return None
    return str(pd.Timestamp(value).date())


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
    fields_by_request = {request_hash(request): inputs
                         for request, inputs in assembled.values()}
    batch_id = content_hash({
        "as_of": as_of, "snapshot_id": snapshot_id,
        "calendar_revision": calendar_revision,
    })
    batch = ScoreBatch(batch_id=batch_id, requests=tuple(r for r, _ in assembled.values()),
                       population_ref=snapshot_id)
    with no_fit_guard():
        records = score_batch(batch, fields_by_request)
    records_document = {
        "schema_version": "native_score_batch_records.v1.0",
        "authoritative": False,
        # Issue #53 (the post-as_of panel-feature anchor gap) is fixed
        # upstream (#67): assemble_nightly_source_bundle's own required
        # panel_anchor parameter refuses POST_AS_OF_ROW per-row before a
        # bundle is ever built here. Nothing populates known_gaps today;
        # the key stays in the schema for a future gap this module might
        # need to flag.
        "known_gaps": [],
        "records": [to_document(record) for record in records],
    }
    (root / "records.json").write_text(json.dumps(records_document, sort_keys=True,
                                                  separators=(",", ":")))
    (root / "refusals.json").write_text(json.dumps(
        [refusal.as_document() for refusal in refusals], sort_keys=True,
        separators=(",", ":")))
    return {
        "outputs": [
            {"name": "records", "path": "records.json",
             "schema": "native_score_batch_records.v1.0"},
            {"name": "refusals", "path": "refusals.json",
             "schema": "native_score_batch_refusals.v1.0"},
        ],
        "completed_ids": list(parameters["expected_ids"]),
        "no_work": not parameters["expected_ids"],
        "refused": [refusal.code for refusal in refusals],
    }
