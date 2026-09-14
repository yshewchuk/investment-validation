"""Pure derivation of ``decision_plan.v1.0`` / ``decision_evidence.v1.0``.

P2-5/B1c wires a ``decision_evidence`` stage into the nightly job DAG so
``legacy_decisions`` finally has a plan and evidence document to bind (B1a
already taught the commit path to read them; nothing produced them).  This
module is the one place that KNOWS how those two documents are built from
already-committed artifacts. It is deliberately pure: no clock, no catalog,
no filesystem — every value it needs (the parsed score/finality/replay
documents, ``session``/``deployment``/``decision_clock``, and refs for the
score/finality artifacts) is handed in by the caller.

Two callers use it, independently, over the same inputs:

* the ``decision_evidence`` WORKER (``engine/v2/ops/worker.py``), which reads
  its bound ``score.json``/``finality.json``/``replay.json``/
  ``finality_coverage.json`` off disk and recomputes
  ``score_ref``/``finality_ref`` locally with
  :func:`engine.v2.foundation.artifact_reference` (the artifact store's own
  identity function, over the bytes it just read);
* the COORDINATOR (``engine/v2/ops/supervisor.py``), which already has the
  launch-time resolved bindings and re-derives from those to check the
  worker's two published outputs byte-for-byte.

Both use the same pure function, so "the worker computed it" and "the
coordinator re-derived it" either agree exactly or the job is refused —
never a probabilistic or partial check.

``engine/v2/ops/decision_validation.py::validate`` is never imported here and
is never changed by this module; this only produces the two documents it
reads. See that module's docstring for the exact contract every field below
satisfies.
"""
from __future__ import annotations

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.foundation.artifacts import artifact_reference
from engine.v2.ops.decision_replay import (
    compare_rows,
    decision_population,
    population_key,
    score_row_id,
)
from engine.v2.ops.errors import fail
from engine.v2.ops.session_resolution import resolve_effective_session

__all__ = ["derive"]


def _document_bytes(document):
    return canonical_json(document).encode("utf-8")


def _score_rows(score_doc):
    rows = score_doc.get("rows") if isinstance(score_doc, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise fail("VALIDATION_FAILED", "score document has no row list")
    return rows


def _common_bindings(score_ref, finality_ref, plan_ref, *, session, deployment,
                     decision_clock, expected):
    """The ten fields every receipt binds identically (validator §"Per-receipt")."""
    return {
        "score_artifact_id": score_ref.artifact_id, "score_content_hash": score_ref.content_hash,
        "finality_artifact_id": finality_ref.artifact_id,
        "finality_content_hash": finality_ref.content_hash,
        "plan_artifact_id": plan_ref.artifact_id, "plan_content_hash": plan_ref.content_hash,
        "session": session, "deployment": deployment, "decision_clock": decision_clock,
        "expected_population": expected,
    }


def _receipt(kind, common, **fields):
    return dict(common, schema_version="decision_receipt.v1.0", kind=kind, **fields)


def _replay_receipt(common, population, replay_doc):
    """Recompute agreement ourselves; ``replay_doc["findings"]`` is never trusted.

    ``replay_doc["source_rows"]`` must be exactly the eligible population (the
    replay stage's own contract). ``replayed_rows`` is rebuilt with a
    freshly-computed ``row_id`` — never copied from the source — before
    comparing; only when that comparison finds nothing does the receipt claim
    agreement, and in that case it emits the SOURCE rows (the only rows the
    validator will accept as "verified"). Any finding makes the receipt
    honestly carry the rebuilt (disagreeing) rows instead, which the
    validator then refuses — see ``decision_validation._validate_replay``.
    """
    if not isinstance(replay_doc, dict):
        raise fail("VALIDATION_FAILED", "replay document is malformed")
    if canonical_json(replay_doc.get("source_rows")) != canonical_json(population):
        raise fail("VALIDATION_FAILED", "replay source rows differ from the eligible population")
    raw_replayed = replay_doc.get("replayed_rows")
    if not isinstance(raw_replayed, list) or not all(isinstance(row, dict) for row in raw_replayed):
        raise fail("VALIDATION_FAILED", "replay document has no replayed row list")
    rebuilt = [dict(row, row_id=score_row_id(row)) for row in raw_replayed]
    findings = compare_rows(population, rebuilt)
    agreed = population if not findings else rebuilt
    return _receipt("replay", common, source_rows=population, replayed_rows=agreed,
                    source_rows_hash=content_hash(population),
                    replayed_rows_hash=content_hash(agreed), findings=findings)


def _finality_receipt(common, finality_doc, coverage_doc, session):
    """``covered_tickers`` comes from ``finality_coverage.json`` alone.

    ``engine/data/finality.py``'s ``SessionFinality`` (what ``finality.json``
    itself holds) has no per-ticker coverage field — only the aggregate
    ``tickers``/``covered`` counts — so ``finality.json`` is never touched for
    this (and never gains a key: it is embedded verbatim into ledger rows, a
    v1 parity requirement). ``legacy_adapter._action_finality`` now writes a
    SECOND output, ``finality_coverage.json``, from
    ``engine.data.finality.covered_tickers`` — a real per-ticker finality
    test, not a stand-in derived from what happened to get scored.
    """
    if not isinstance(coverage_doc, dict) or coverage_doc.get(
            "schema_version") != "finality_coverage.v1.0":
        raise fail("VALIDATION_FAILED", "finality coverage document is malformed")
    if coverage_doc.get("date") != session:
        raise fail("VALIDATION_FAILED", "finality coverage document is for a different session")
    covered = coverage_doc.get("covered_tickers")
    if not isinstance(covered, list) or not all(isinstance(item, str) for item in covered):
        raise fail("VALIDATION_FAILED", "finality coverage document has no ticker list")
    return _receipt("finality", common, observed_finality_hash=content_hash(finality_doc),
                    covered_tickers=sorted(covered))


def derive(score_doc, score_ref, finality_doc, finality_ref, replay_doc, coverage_doc, *,
          requested_session, deployment, decision_clock, scope="shadow"):
    """Derive ``(plan_bytes, evidence_bytes)`` for one nightly session.

    ``requested_session`` is the date the nightly PLAN carries (job identity
    never depends on data not yet read); the plan's own ``session`` is
    ``finality_doc``'s resolved date (P2-C03 — refused via
    :func:`engine.v2.ops.session_resolution.resolve_effective_session` unless
    it is a genuine walk-back at or before ``requested_session``), recorded
    alongside ``requested_session`` so both survive into every receipt.

    ``scope`` (P2-C04) is the job's own effect scope
    (``nightly.effect_scope_for`` — ``"shadow"`` for a full-universe run,
    ``"shadow:<hash>"`` for a ticker subset). It is stamped onto the plan so
    ``decision_validation.validate`` and ``decision_commit`` bind every
    downstream watermark/export/release to THIS run's own scope, never the
    global authority namespace. It plays no part in any receipt hash or
    identity check — only the plan document carries it — so it defaults to
    the pre-existing constant for a caller that predates this parameter.

    ``score_ref``/``finality_ref`` need only ``artifact_id``/``content_hash``
    attributes (an ``ArtifactRef``, a recorded ``ResolvedBinding``, or a
    locally recomputed :func:`artifact_reference` all satisfy this).
    ``coverage_doc`` is the parsed ``finality_coverage.json`` document (see
    :func:`_finality_receipt`); its ``date`` must equal the resolved session.

    On a night with no decision-eligible rows, ``expected_population`` is
    simply empty; this function does not special-case it. Whether that plan
    is committable with zero decisions is ``decision_validation.validate``'s
    call, not this function's.
    """
    session = resolve_effective_session(finality_doc, requested_session)
    score_rows = _score_rows(score_doc)
    population = decision_population(score_doc, session)
    expected = [population_key(row) for row in population]

    plan = {"schema_version": "decision_plan.v1.0", "session": session,
            "requested_session": requested_session,
            "deployment": deployment, "decision_clock": decision_clock,
            "expected_population": expected, "scope": scope}
    plan_bytes = _document_bytes(plan)
    plan_ref = artifact_reference(plan_bytes, "decision_plan.v1.0")

    common = _common_bindings(score_ref, finality_ref, plan_ref, session=session,
                              deployment=deployment, decision_clock=decision_clock,
                              expected=expected)
    cutoffs = {population_key(row): row.get("evidence_cutoff") for row in score_rows}
    receipts = {
        "causality": _receipt("causality", common, observed_cutoffs=cutoffs),
        "coverage": _receipt("coverage", common, observed_population=expected),
        "finality": _finality_receipt(common, finality_doc, coverage_doc, session),
        "selection": _receipt("selection", common, eligible_candidate_keys=expected),
        "replay": _replay_receipt(common, population, replay_doc),
    }
    evidence_bytes = _document_bytes(
        {"schema_version": "decision_evidence.v1.0", "receipts": receipts})
    return plan_bytes, evidence_bytes
