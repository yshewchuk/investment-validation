"""Preview release, event-page and legacy-score-bridge contracts.

Rearchitecture phase-3 guide §5.1-§5.3 and §6 (P3-1a). Schemas only: frozen
kw_only dataclasses and closed vocabularies, exactly like every other module
in this package. Nothing here reads a clock, hashes a document or joins a
row to another — ``engine/v2/serving/bridge.py`` does that, over these
shapes.

Three families here are new *content* identity, not envelopes:

* ``PreviewInput`` (§5.1) is the one private document a projection command
  accepts — pinned refs to already-verified Phase 2 artifacts, never a
  directory to glob.
* ``PreviewRelease`` (§5.2) and ``LegacyScoreBridge`` (§5.2) separate engine
  evidence from display values. ``LegacyScoreBridge`` is explicitly **not**
  the future Phase 4 ``ScoreRecord``: it advertises its own
  ``score_format`` rather than guessing at fields that schema will need.
* ``EventPage``/``EventPageItem`` (§6) is the bounded read-API page shape;
  ``ProjectionFindings`` (§5.3 point 7) is the one-receipt mismatch list the
  bridge must emit, with the funnel counts a zero-population release can
  never hide behind.

``operational timestamps live in envelopes, outside deterministic score/
projection identity`` (§5.2): nothing in ``PreviewRelease``/
``LegacyScoreBridge`` is a wall-clock write time; ``requested_as_of``/
``resolved_as_of`` are the pinned session identifiers the projection was
built for, not when it happened to run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from engine.v2.contracts.data import EventRef, ObjectRef

__all__ = [
    "DEFAULT_SHADOW_SERVING_SCORER",
    "SHADOW_SERVING_SCORERS",
    "EVENT_PAGE_ITEM_V1",
    "EVENT_PAGE_V1",
    "EVENT_SCORE_SUMMARY_V1",
    "FINDING_V1",
    "LEGACY_SCORE_BRIDGE_V1",
    "PREVIEW_INPUT_V1",
    "PREVIEW_RELEASE_V1",
    "PROJECTION_FINDINGS_V1",
    "ROW_IDENTITY_V1",
    "EventPage",
    "EventPageItem",
    "EventScoreSummary",
    "Finding",
    "FindingCategory",
    "LegacyScoreBridge",
    "PreviewCapabilities",
    "PreviewInput",
    "PreviewRelease",
    "ProjectionFindings",
    "RowIdentity",
]

PREVIEW_INPUT_V1 = "preview_input.v1.0"
PREVIEW_RELEASE_V1 = "preview_release.v1.0"
LEGACY_SCORE_BRIDGE_V1 = "legacy_score_bridge.v1.0"
EVENT_PAGE_V1 = "event_page.v1.0"
EVENT_PAGE_ITEM_V1 = "event_page_item.v1.0"
EVENT_SCORE_SUMMARY_V1 = "event_score_summary.v1.2"
PROJECTION_FINDINGS_V1 = "projection_findings.v1.0"
ROW_IDENTITY_V1 = "row_identity.v1.0"
FINDING_V1 = "finding.v1.0"

#: §5.3 point 7: what one mismatch in a comparison receipt can be about.
#: ``unresolved_event`` covers point 3's refusal (ambiguous/unmapped
#: ticker+event_date), kept distinct from ``identity`` (a join-key collision
#: or a scorer/renderer row-id disagreement on an otherwise-mapped event).
FindingCategory = Literal[
    "missing", "unplanned", "duplicate", "identity",
    "null_mask", "value", "model_ref", "unresolved_event",
]


@dataclass(frozen=True, kw_only=True)
class PreviewInput:
    """§5.1: the one private document a projection command accepts.

    Every ref must resolve to a verified retained object; none may be null
    or a fabricated ID except ``model_evidence_ref``, whose absence needs a
    reason recorded elsewhere and must never be silently backfilled from a
    file currently on disk.
    """

    source_release_id: str
    source_release_manifest_ref: str
    snapshot_ref: str
    legacy_snapshot_object_ref: ObjectRef
    score_batch_ref: str
    score_job_input_refs: tuple[str, ...]
    bundle_manifest_ref: str
    model_registry_artifact_refs: tuple[str, ...]
    model_evidence_ref: str | None = None
    finality_ref: str
    expected_population_ref: str
    score_comparison_receipt_ref: str
    render_comparison_receipt_ref: str
    source_code_hash: str
    source_environment_hash: str
    schema_version: str = PREVIEW_INPUT_V1


@dataclass(frozen=True, kw_only=True)
class PreviewCapabilities:
    """What a shadow ``PreviewRelease`` actually lets a client do (§5.2)."""

    read: bool
    submit_jobs: bool
    collect_live: bool


@dataclass(frozen=True, kw_only=True)
class PreviewRelease:
    """§5.2: engine evidence kept apart from the display values it produced.

    ``score_format``/``producer`` are advertised rather than assumed, so a
    client can support the native Phase 4 schema alongside this one without
    this release pretending to be it.
    """

    release_id: str
    source_release_id: str
    projection_manifest_ref: str
    snapshot_ref: str
    score_batch_ref: str
    bundle_manifest_ref: str
    model_registry_artifact_refs: tuple[str, ...]
    model_evidence_ref: str | None = None
    comparison_receipt_refs: tuple[str, ...]
    source_code_hash: str
    projection_code_hash: str
    requested_as_of: str
    resolved_as_of: str
    clock_ids: tuple[str, ...]
    coverage_summary: dict[str, float] = field(default_factory=dict)
    stale_or_degraded_reasons: tuple[str, ...] = ()
    score_format: Literal["legacy_score_bridge.v1.0"] = LEGACY_SCORE_BRIDGE_V1
    producer: Literal["legacy_via_v2"] = "legacy_via_v2"
    #: A plain (frozen, hashable) instance default, not a ``field(default_
    #: factory=...)`` lambda: this package defines no function anywhere,
    #: including a lambda (§4, checked by ``test_contracts_define_no_functions``).
    capabilities: PreviewCapabilities = PreviewCapabilities(
        read=True, submit_jobs=False, collect_live=False)
    schema_version: str = PREVIEW_RELEASE_V1


@dataclass(frozen=True, kw_only=True)
class LegacyScoreBridge:
    """§5.2: one joined source/display row pair, not the Phase 4 ``ScoreRecord``.

    ``engine_record`` retains saved precision, null/absent distinctions and
    ordered geometry exactly as the score job wrote them. ``display_record``
    retains the renderer's own values and units, unchanged. Neither is
    recomputed here, and neither the display digest nor a rounded value may
    become an inference/replay input (§5.2).
    """

    score_id: str
    event_ref: EventRef
    clock_id: str
    legacy_row_id: str
    score_batch_ref: str
    source_row_key: str
    source_record_hash: str
    request_provenance_refs: tuple[str, ...]
    snapshot_ref: str
    model_registry_artifact_refs: tuple[str, ...]
    engine_record: dict[str, Any]
    display_record: dict[str, Any]
    detail_refs: tuple[str, ...] = ()
    unavailable_detail_reasons: tuple[str, ...] = ()
    schema_version: str = LEGACY_SCORE_BRIDGE_V1


@dataclass(frozen=True, kw_only=True)
class EventScoreSummary:
    """One main-board strategy summary carried on an event page row (§6).

    The exact fields §7's board table names: verdict/refusal, driver
    forecast, market implied move, entry premium, "available expected-return
    fields" (§7's own plural — the legacy board carries three: model,
    analog, sim; a v1.1 addition, nullable, so an older reader still decodes
    a v1.0 document) and the DYN-SV choice. Copied from an already-built
    ``LegacyScoreBridge.display_record`` by the read API layer; nothing here
    computes any of them.

    ``expected_return`` is never populated (v1.1, review fix): the rendered
    row carries no single merged headline field for it — ``compact_row``
    emits ``exp_pnl_model``/``exp_pnl_analog``/``exp_pnl_sim`` separately,
    and the board's own "pick model, else sim, never analog" headline
    (``dashboard/static/assets/app.js`` ``pnlCell``) is JS logic over two of
    those three raw fields, not a value ``compact_row`` itself ever writes.
    Choosing between them here would be exactly the "recompute or choose
    between rendered values" the guide (§5.2/§5.3) refuses. The field stays
    for a future reader that reproduces ``pnlCell`` faithfully; until then
    ``expected_return_model``/``_analog``/``_sim`` are the honest read.

    ``flags`` (v1.2 addition): the rendered row's own exact flag codes, copied verbatim.
    """

    score_id: str
    strategy: str
    verdict: str | None = None
    refusal_reason: str | None = None
    driver_forecast: float | None = None
    market_implied_move: float | None = None
    entry_premium: float | None = None
    expected_return: float | None = None
    expected_return_model: float | None = None
    expected_return_analog: float | None = None
    expected_return_sim: float | None = None
    chosen_strategy: str | None = None
    chosen_margin: float | None = None
    menu_size: int | None = None
    flags: tuple[str, ...] = ()
    schema_version: str = EVENT_SCORE_SUMMARY_V1


@dataclass(frozen=True, kw_only=True)
class EventPageItem:
    """One row of a bounded event page (§6): identity plus board summaries.

    Omits engine records, full ladders, history, analogs and model
    evidence — those stay lazy detail, fetched only on selection.
    """

    event_ref: EventRef
    ticker: str
    event_date: str
    session: str | None = None
    clock_id: str
    readiness: str
    scores: tuple[EventScoreSummary, ...] = ()
    schema_version: str = EVENT_PAGE_ITEM_V1


@dataclass(frozen=True, kw_only=True)
class EventPage:
    """§6: ``schema_version, release_id, query_hash, items, next_cursor, total_matching``.

    ``total_matching`` covers the complete filtered population, never only
    the rows on this page (§6 rules); ``next_cursor`` is null on the last
    page.
    """

    release_id: str
    query_hash: str
    items: tuple[EventPageItem, ...]
    next_cursor: str | None = None
    total_matching: int
    schema_version: str = EVENT_PAGE_V1


@dataclass(frozen=True, kw_only=True)
class RowIdentity:
    """What one finding is about: the join-key components §5.3 point 4 names.

    ``strike_key``/``discriminator`` are strings, never floats (component
    contracts §2.1): a binary float would give the same row two different
    identities depending on encoding, exactly the reason
    ``contracts.data.KeyPredicate`` excludes float membership values.
    """

    ticker: str
    event_date: str
    strategy: str
    expiry: str | None = None
    strike_key: str | None = None
    discriminator: str | None = None
    schema_version: str = ROW_IDENTITY_V1


@dataclass(frozen=True, kw_only=True)
class Finding:
    """One independent mismatch (§5.3 point 7). Never stops at the first."""

    code: str
    category: FindingCategory
    message: str
    row: RowIdentity | None = None
    field_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    schema_version: str = FINDING_V1


@dataclass(frozen=True, kw_only=True)
class ProjectionFindings:
    """§5.3 point 7: the complete mismatch list plus funnel counts.

    ``ok`` is set by the bridge builder from these same counts and
    ``findings`` — never recomputed by a reader — and is always False when
    ``compared_population`` is zero or any finding was emitted (§5.3: "A
    zero compared population cannot pass real launch.").
    """

    planned_population: int
    rendered_main_population: int
    rendered_ladder_population: int
    matched_population: int
    compared_population: int
    findings: tuple[Finding, ...] = ()
    ok: bool
    schema_version: str = PROJECTION_FINDINGS_V1


#: The only values a plan's ``shadow_serving_scorer`` may carry (spec_ns_b G5).
#: The ONE source of truth for both halves of the shadow-serving seam:
#: ``engine.v2.ops.native_shadow_render`` and
#: ``engine.v2.serving.native_shadow_render`` are layer-7 peers and cannot
#: import each other, so both read the allowed values and the default here.
SHADOW_SERVING_SCORERS = ("native", "legacy")

#: The shadow board's default scorer when a plan carries no explicit value.
DEFAULT_SHADOW_SERVING_SCORER = "native"

