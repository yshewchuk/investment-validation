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

:func:`accepted_generation_refs` reads the SPECIFIC committed import that
produced the plan's own pinned ``snapshot_id`` in its own ``scope`` — never
"whatever is newest right now", which could have moved since the plan was
built — and returns its pinned ``LegacyInputManifest``, the same document
``snapshot_import_effect`` verified against ``SnapshotImportRequest.
source_manifest_hash`` before ever committing, via the plain
``attempt_input_bindings`` row the import job's own launch already recorded
for ``legacy_manifest.json`` (``input_bindings.record_resolved_bindings``);
no new binding, table or coordinator flow.

:func:`refuse_generation_mismatch` is the launch-time check: any path both
manifests pin, pinned to a DIFFERENT content hash, refuses ``INPUT_CHANGED``
with ``details.reason: "generation_mismatch"`` and the differing paths only
(never a byte, never a hash — paths are enough to act on). A path only one
side pins (a barrier stage's own finality-coverage frames, or a file the
import never touched) is not a disagreement and is silently allowed, exactly
as the task brief specifies. No committed import for the pinned snapshot_id
is not a mismatch either — there is nothing accepted to bind to.
"""
from __future__ import annotations

import json

from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import fail

__all__ = ["accepted_generation_refs", "refuse_generation_mismatch"]

_COMMITTED_IMPORT_FOR_SNAPSHOT = """
SELECT attempt_id FROM data_import_receipts
WHERE scope = ? AND status = 'committed' AND result_snapshot_id = ?
ORDER BY registered_at DESC, rowid DESC LIMIT 1
"""

_MANIFEST_BINDING = """
SELECT artifact_id FROM attempt_input_bindings
WHERE attempt_id = ? AND name = 'legacy_manifest.json'
"""


def accepted_generation_refs(conn, store, *, scope: str, snapshot_id: str) -> dict[str, str] | None:
    """``{path: content_hash}`` of the committed import's own pinned
    ``LegacyInputManifest`` that produced ``snapshot_id`` in ``scope`` — the
    EXACT snapshot a plan pinned, never "whatever is newest in scope right
    now". ``None`` when no committed import produced it (or pinned no
    manifest) — nothing accepted to bind a barrier stage to."""
    receipt = conn.execute(_COMMITTED_IMPORT_FOR_SNAPSHOT, (scope, snapshot_id)).fetchone()
    if receipt is None:
        return None
    binding = conn.execute(_MANIFEST_BINDING, (receipt[0],)).fetchone()
    if binding is None:
        return None
    document = json.loads(store.read_verified(artifact(conn, store, binding[0])))
    return {ref["path"]: ref["content_hash"] for ref in document.get("file_refs", [])}


def refuse_generation_mismatch(conn, store, *, scope: str, snapshot_id: str,
                               barrier_manifest: dict) -> None:
    """Refuse ``INPUT_CHANGED``/``generation_mismatch`` when ``barrier_manifest``
    (the raw ``LegacyInputManifest`` document a barrier job just pinned) names
    a path the plan's own pinned generation also pins, at a different content
    hash. The caller is expected to skip this entirely for a legacy-mode job
    (``snapshot_id`` empty) — see the module docstring."""
    accepted = accepted_generation_refs(conn, store, scope=scope, snapshot_id=snapshot_id)
    if not accepted:
        return
    barrier_refs = {ref["path"]: ref["content_hash"] for ref in barrier_manifest.get("file_refs", [])}
    differing = sorted(path for path, digest in barrier_refs.items()
                       if path in accepted and accepted[path] != digest)
    if differing:
        raise fail("INPUT_CHANGED",
                   "barrier stage's legacy read set differs from the accepted data/model "
                   "generation the plan's own snapshot pinned",
                   details={"reason": "generation_mismatch", "paths": differing})
