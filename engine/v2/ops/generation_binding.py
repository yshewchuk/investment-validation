"""Bind manifest-backed barrier stages to the accepted data/model generation —
P2-C02 (Phase 2 review closeout, §12.2): "Bind remaining manifest-backed
stages to the same accepted data/model generation as snapshot scoring."

A barrier-only kind (``legacy_finality``/``legacy_model_evidence``/
``legacy_selfcheck``, ``stages.BARRIER_ONLY_REASONS``) has no declared read
plan, so it can never run snapshot-backed: it always launches through
``supervisor.Service._pin_read_set``, reading a pinned ``LegacyInputManifest``
copied from the LIVE legacy store at that moment. Nothing before this task
ever compared that live read against the generation a snapshot import
actually committed, so a barrier stage could silently describe a newer (or
older) legacy tree than the one snapshot-backed scoring just ran against.

**Gated on snapshot input mode (review fix).** The check must never fire for
a default ``--input-mode legacy`` nightly: the legacy store moves ahead of
whatever a shadow snapshot happened to pin, so comparing against "the newest
committed import in scope" refused finality/model_evidence/selfcheck on
every ordinary data refresh the moment any shadow snapshot existed —
unrelated to whether THIS run reads snapshot-backed at all. A barrier-only
kind can never declare ``input_mode="snapshot"`` itself (``stages.
input_mode_problems`` refuses it — no declared read plan for that kind), so
it carries no marker of its own; ``nightly._stage_parameters`` now stamps
``snapshot_generation_id``/``snapshot_generation_scope`` (the plan's own
pinned ``snapshot_id``/``scope`` from ``snapshot_planning.pin_snapshot_inputs``)
onto EVERY stage in a snapshot-mode plan graph, barrier stages included. The
caller (``supervisor.Service._pin_read_set``) only calls into this module
when ``snapshot_generation_id`` is non-empty — a legacy-mode job's
parameters leave it ``""`` and the check never runs.

:func:`accepted_generation_refs` reads ONE SPECIFIC committed import receipt,
by its own ``receipt_id`` — never "whatever is newest for this snapshot_id
right now" (external review finding #5, 2026-09-14: the data snapshot id
alone does not identify a generation, because a reference-only reimport can
commit a NEW receipt against the SAME ``result_snapshot_id`` with different
pinned model/reference files; resolving "latest committed receipt for this
snapshot_id" at launch time — as this module did before the fix — let a
later import silently move what an already-planned job validated against,
either into a false pass or a false ``INPUT_CHANGED`` on a job that never
itself changed). ``pin_snapshot_inputs`` resolves "latest" exactly once, at
plan time, and stamps the resulting ``receipt_id`` onto every stage in the
plan graph (``nightly._stage_parameters`` ->
``LegacyParameters.snapshot_generation_receipt_id``); that is the only place
"latest" is allowed to mean anything. Every launch-time caller here reads
the plan's own pinned receipt id and never re-resolves.

``accepted_generation_refs`` returns the pinned ``LegacyInputManifest`` of
that one receipt, the same document ``snapshot_import_effect`` verified
against ``SnapshotImportRequest.source_manifest_hash`` before ever
committing, via the plain ``attempt_input_bindings`` row the import job's own
launch already recorded for ``legacy_manifest.json``
(``input_bindings.record_resolved_bindings``) — OR, when that receipt's own
attempt carries no such binding, by following ``data_receipt_lineage`` (v8)
to its base receipt and repeating there (see "Receipt lineage" below). No
new binding, table (beyond v8) or coordinator flow.

**Receipt lineage (v8).** A ``price_history`` capture
(``engine.v2.ops.price_history_store``) commits its generation under its own
honest, non-scheduler ``attempt_id`` — never a real job attempt — so it never
gains an ``attempt_input_bindings`` row for ``legacy_manifest.json``. A
capture never changes the legacy input manifest: it only adds
``price_history`` and carries every other table's dataset version forward
unchanged, so its accepted legacy read-set is exactly its BASE receipt's
(the committed receipt of the head it captured onto), recorded once, in the
capture's own commit transaction, as a ``data_receipt_lineage`` row
(:func:`record_price_history_lineage`). :func:`accepted_generation_refs`
walks that row when a receipt's own attempt has no manifest binding: resolve
``base_receipt_id``, check IT for a binding, and if it also has none, follow
ITS lineage row, and so on — first ancestor with its own binding wins. The
walk is bounded (64 hops) and cycle-safe (a repeated receipt_id stops it
early); a chain that ends without ever finding a binding, or whose next hop
does not resolve to a currently-committed receipt (the same
``_COMMITTED_RECEIPT_ATTEMPT`` query every hop re-runs), returns ``None`` —
:func:`refuse_generation_mismatch` turns that into the same typed
``INPUT_CHANGED``/``generation_receipt_missing`` refusal an unresolvable
receipt_id has always produced. A receipt with its own binding is returned
directly, exactly as before v8 — the lineage table is never consulted for
one.

:func:`refuse_generation_mismatch` is the launch-time check: any path both
manifests pin, pinned to a DIFFERENT content hash, refuses ``INPUT_CHANGED``
with ``details.reason: "generation_mismatch"`` and the differing paths only
(never a byte, never a hash — paths are enough to act on). A path only one
side pins (a barrier stage's own finality-coverage frames, or a file the
import never touched) is not a disagreement and is silently allowed, exactly
as the task brief specifies. An empty ``receipt_id`` is a deliberate no-op
for isolated callers (nothing was ever pinned to check against); a
NON-empty ``receipt_id`` that does not resolve to a committed receipt with a
pinned manifest — deleted, tampered, or simply never committed — refuses
``INPUT_CHANGED``/``generation_receipt_missing`` rather than silently
proceeding, because by construction a real plan never stamps a receipt id
that was not committed at plan time. ``Service._pin_read_set`` is the one
caller that must never pass an empty ``receipt_id`` for a snapshot-mode job:
it refuses ``generation_not_pinned`` itself first (see its docstring) for a
job planned before this fix, which has no receipt id to pass at all.
"""
from __future__ import annotations

import json

from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import fail

__all__ = ["accepted_generation_refs", "record_price_history_lineage", "refuse_generation_mismatch"]

_COMMITTED_RECEIPT_ATTEMPT = """
SELECT attempt_id FROM data_import_receipts
WHERE receipt_id = ? AND status = 'committed'
"""

_MANIFEST_BINDING = """
SELECT artifact_id FROM attempt_input_bindings
WHERE attempt_id = ? AND name = 'legacy_manifest.json'
"""

_LINEAGE_BASE = """
SELECT base_receipt_id FROM data_receipt_lineage WHERE receipt_id = ?
"""

#: "Bound the walk (depth <= 64, cycle-safe)" — task brief. A real chain is
#: one or two hops (price_history_store never captures onto another
#: capture's OWN receipt more than the task's "two successive captures"
#: scenario describes); 64 is headroom, not an expected depth.
_MAX_LINEAGE_DEPTH = 64

_PRICE_HISTORY_CAPTURE_LINEAGE = "price_history_capture"


def accepted_generation_refs(conn, store, *, receipt_id: str) -> dict[str, str] | None:
    """``{path: content_hash}`` of ONE SPECIFIC committed import receipt's own
    pinned ``LegacyInputManifest`` — never "whatever is newest right now" —
    following ``data_receipt_lineage`` to a base receipt when this one's own
    attempt carries no manifest binding (see the module docstring's "Receipt
    lineage" section). ``None`` when ``receipt_id`` is empty, the chain
    starting there ends without ever finding a committed receipt with its
    own pinned manifest, a hop resolves to a receipt that is no longer
    committed, or the walk exceeds :data:`_MAX_LINEAGE_DEPTH` — nothing
    accepted to bind a barrier stage to."""
    if not receipt_id:
        return None
    current = receipt_id
    seen: set[str] = set()
    for _ in range(_MAX_LINEAGE_DEPTH):
        if current in seen:
            return None  # cycle
        seen.add(current)
        receipt = conn.execute(_COMMITTED_RECEIPT_ATTEMPT, (current,)).fetchone()
        if receipt is None:
            return None
        binding = conn.execute(_MANIFEST_BINDING, (receipt[0],)).fetchone()
        if binding is not None:
            document = json.loads(store.read_verified(artifact(conn, store, binding[0])))
            return {ref["path"]: ref["content_hash"] for ref in document.get("file_refs", [])}
        lineage = conn.execute(_LINEAGE_BASE, (current,)).fetchone()
        if lineage is None:
            return None
        current = lineage[0]
    return None  # depth exceeded


def record_price_history_lineage(conn, *, receipt_id: str, base_receipt_id: str) -> None:
    """Record that ``receipt_id`` (a price_history-only capture generation's
    own receipt) inherits its accepted legacy read-set from
    ``base_receipt_id`` (the committed receipt of the head it captured
    onto) — see the module docstring's "Receipt lineage" section. Called
    from inside the capture's own commit transaction
    (``price_history_store._commit_generation``'s ``record_references``
    callback), so the row is all-or-nothing with the receipt it names."""
    conn.execute(
        "INSERT INTO data_receipt_lineage (receipt_id, base_receipt_id, kind) VALUES (?, ?, ?)",
        (receipt_id, base_receipt_id, _PRICE_HISTORY_CAPTURE_LINEAGE))


def refuse_generation_mismatch(conn, store, *, receipt_id: str, barrier_manifest: dict) -> None:
    """Refuse ``INPUT_CHANGED`` when ``barrier_manifest`` (the raw
    ``LegacyInputManifest`` document a barrier job just pinned) disagrees
    with the plan's pinned generation receipt (``generation_mismatch``), or
    when that exact receipt no longer resolves at all
    (``generation_receipt_missing``). An empty ``receipt_id`` is a
    deliberate no-op for isolated callers — see the module docstring."""
    if not receipt_id:
        return
    accepted = accepted_generation_refs(conn, store, receipt_id=receipt_id)
    if accepted is None:
        raise fail("INPUT_CHANGED",
                   "the plan's pinned generation receipt is missing, was never committed, or "
                   "pinned no reference manifest",
                   details={"reason": "generation_receipt_missing", "receipt_id": receipt_id})
    barrier_refs = {ref["path"]: ref["content_hash"] for ref in barrier_manifest.get("file_refs", [])}
    differing = sorted(path for path, digest in barrier_refs.items()
                       if path in accepted and accepted[path] != digest)
    if differing:
        raise fail("INPUT_CHANGED",
                   "barrier stage's legacy read set differs from the accepted data/model "
                   "generation the plan's own snapshot pinned",
                   details={"reason": "generation_mismatch", "paths": differing})
