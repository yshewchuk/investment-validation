"""Plan-time snapshot pinning for ``ops plan nightly --input-mode snapshot`` — §8.1, §9.3.

:func:`pin_snapshot_inputs` calls ``snapshots.resolve_snapshot_head`` exactly
once, reads the published ``SnapshotRef`` back, builds one
``LegacyMaterializationRequest`` on that ref and publishes it. The plan records
both artifact IDs; ``nightly.build_legacy_job_requests`` binds them into the
job graph. A retry of the same plan therefore reuses the same refs, and a new
head can only enter through a new plan (and a new job identity).

Scopes: evidence = the full planned ticker set over the planned years; direct
= the planned score population (tickers and event years of every
``ticker|strategy|event_date`` key). ``read_plan_complete`` must hold at plan
time, so a population outside the evidence years is refused here, not at launch.

The non-dataset inputs a request pins (legacy SNAPSHOT object, registry/model
refs, calendar refs) come from the data catalog: the reference inputs recorded
by the most recent committed import receipt in ``scope`` whose resulting
snapshot is the pinned one (``engine.v2.data.reference_catalog``). No such
receipt, or one that pinned no reference inputs, refuses the plan
(``INPUT_CHANGED`` with ``data_code`` ``SNAPSHOT_NOT_READY``).

That receipt's own id is resolved exactly once here and returned as
``snapshot_generation_receipt_id`` (external review #5, 2026-09-14): a later
reference-only reimport can commit a NEW receipt against the SAME
``result_snapshot_id`` with different pinned model/reference files, so
"newest committed receipt for this snapshot id" is only a correct answer at
THIS moment, plan time. ``nightly._stage_parameters`` stamps the returned
receipt id onto every stage in the plan graph, and
``engine.v2.ops.generation_binding`` reads that pinned id at launch time
rather than re-resolving "latest" — see its module docstring.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.v2.contracts import SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.data.legacy_materialization import (
    SCORE_READ_PLAN_TABLES,
    build_materialization_request,
    parse_pinned_ref,
    read_plan_complete,
)
from engine.v2.data.reference_catalog import (
    committed_receipt_for_snapshot,
    pinned_materialization_refs,
    reference_inputs_for_snapshot,
)
from engine.v2.data.repository import Repository
from engine.v2.foundation import from_document, to_document
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.snapshots import resolve_snapshot_head

__all__ = ["REQUEST_SCHEMA_REF", "direct_scope_for", "pin_snapshot_inputs", "scratch_estimate"]

REQUEST_SCHEMA_REF = "legacy_materialization_request.v1.0"


def direct_scope_for(expected_population) -> dict:
    tickers, years = set(), set()
    for key in expected_population:
        parts = str(key).split("|")
        if len(parts) != 3 or not parts[0] or not parts[2][:4].isdigit():
            raise fail("INVALID_REQUEST", "planned population key is not ticker|strategy|event_date",
                       details={"key": str(key)})
        tickers.add(parts[0])
        years.add(int(parts[2][:4]))
    return {"tickers": sorted(tickers), "years": sorted(years)}


def scratch_estimate(repository, store, request) -> int:
    """The request's pinned byte total: every read-plan fragment plus pinned files."""
    total = int(request.legacy_snapshot_object_ref.byte_size)
    for name in SCORE_READ_PLAN_TABLES:
        total += sum(int(record.object_ref.byte_size)
                     for record in repository.fragment_records(request.snapshot_ref, name))
    for ref in (*request.registry_and_model_refs, *request.calendar_refs):
        digest = parse_pinned_ref(ref)[1].removeprefix("sha256:")
        total += (Path(store.root) / "objects" / digest[:2] / digest).stat().st_size
    return total


def _request(repository, store, snapshot, pinned, scopes, ceilings, observation_ceiling):
    return build_materialization_request(
        repository, store, snapshot, pinned["legacy_snapshot_object_ref"],
        direct_scope=scopes["direct"], evidence_scope=scopes["evidence"],
        registry_and_model_refs=pinned["registry_and_model_refs"],
        calendar_refs=pinned["calendar_refs"], expected_population=ceilings,
        observation_ceiling=observation_ceiling)


def _build(repository, store, snapshot, pinned, scopes, observation_ceiling):
    """Two passes: the per-table row ceilings come from the first pass's own queries."""
    first = _request(repository, store, snapshot, pinned, scopes, {}, observation_ceiling)
    ceilings = {name: query.max_result_rows for name, query in first.table_queries.items()}
    request = _request(repository, store, snapshot, pinned, scopes, ceilings, observation_ceiling)
    if not read_plan_complete(request, repository):
        raise fail("INVALID_REQUEST", "snapshot read plan is not complete for the planned scope")
    return request


def pin_snapshot_inputs(conn, store, scope, *, tickers, year_start, year_end,
                        expected_population, clock, session: str) -> dict:
    """Resolve ``scope``'s head once and publish the request built on that ref,
    with the reference inputs the catalog recorded for that exact snapshot.

    ``tickers`` (P2-C04) is the historical EVIDENCE universe (the caller's
    ``context_tickers``, not the narrower direct watchlist) — the evidence
    scope is built from it directly; the direct scope is derived independently
    from ``expected_population``'s own tickers. A watchlist not fully covered
    by the evidence universe is refused: scoring can never need analog/feature
    context for a ticker it has no evidence plan for.

    ``session`` (SEND-BACK 2026-09-14 item 2) is the nightly plan's own
    session date (``args.as_of`` at the CLI) -- this job's decision cutoff.
    Pinned into the request as ``observation_ceiling =
    f"{session}T23:59:59.000000Z"`` (RFC 3339 UTC with microseconds, per
    ``engine.v2.foundation.clock.parse_timestamp``) so materialization can
    never see a ``price_history`` retrieval made after this job's own
    cutoff, whatever a later capture adds to the pinned snapshot version.
    """
    if not tickers or not expected_population:
        raise fail("INVALID_REQUEST", "snapshot input mode needs planned tickers and population")
    if not session:
        raise fail("INVALID_REQUEST", "snapshot input mode needs the plan's session date")
    observation_ceiling = f"{session}T23:59:59.000000Z"
    # P2-C03: the resolved session is unknown at plan time — a walk-back
    # (``engine.data.finality.resolve_final_session``, up to 15 trading
    # sessions back) can cross a year boundary a requested-year-only range
    # would miss (requested Jan 2, resolved Dec 31); ``year_start - 1`` gives
    # the same one-year margin ``engine.data.finality._coverage_frame``
    # already reads for the identical reason. Cheap and unconditional, like
    # that margin, rather than trying to predict which requests are close
    # enough to January to need it.
    scopes = {"direct": direct_scope_for(expected_population),
              "evidence": {"tickers": sorted(set(tickers)),
                           "years": list(range(int(year_start) - 1, int(year_end) + 1))}}
    if not set(scopes["direct"]["tickers"]) <= set(scopes["evidence"]["tickers"]):
        raise fail("INVALID_REQUEST",
                   "direct scope tickers are not covered by the evidence scope tickers")
    try:
        head = resolve_snapshot_head(conn, store, scope, clock=clock)
        snapshot = from_document(SnapshotRef, json.loads(store.read_verified(head)))
        # External review #5: resolve "latest committed receipt for this
        # snapshot" exactly once, here, and carry the receipt_id itself
        # forward in the returned dict -- nightly._stage_parameters stamps
        # it onto every stage, so a launch-time generation check reads this
        # SAME receipt rather than re-resolving "latest" against whatever a
        # later reference-only reimport has since committed.
        receipt_id = committed_receipt_for_snapshot(conn, scope=scope, snapshot_id=snapshot.snapshot_id)
        pinned = pinned_materialization_refs(
            reference_inputs_for_snapshot(conn, scope=scope, snapshot_id=snapshot.snapshot_id))
        repository = Repository(conn, store)
        request = _build(repository, store, snapshot, pinned, scopes, observation_ceiling)
        estimate = scratch_estimate(repository, store, request)
    except DataError as exc:
        raise fail("INPUT_CHANGED", "snapshot inputs cannot be pinned",
                   details={"data_code": exc.code}) from None
    except OpsError:
        raise
    ref = store.publish_bytes(json.dumps(to_document(request), sort_keys=True).encode(),
                              schema_ref=REQUEST_SCHEMA_REF)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return {"scope": scope, "snapshot_ref_artifact_id": head.artifact_id,
            "snapshot_id": snapshot.snapshot_id, "snapshot_manifest_hash": snapshot.manifest_hash,
            "snapshot_generation_receipt_id": receipt_id or "",
            "materialization_request_ref": ref.artifact_id,
            "materialization_request_hash": request.request_hash,
            "scratch_estimate_bytes": estimate}
