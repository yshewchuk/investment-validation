"""Phase 3 guide §5.5 item 1 ("make nightly effects generation-aware").

The real 2026-09-14 failure: a re-planned run of an already-run session
(``run_sha256:4541ca27acdbeb04d``, session 2026-09-10) submitted 14 fresh job
ids cleanly, then ``engineering_gate`` FAILED with ``IDEMPOTENCY_CONFLICT``
("completed occurrence has another receipt") -- ``outbox.watermark`` keyed
engineering_gate/ledger_export/backup/publication completion on
``(scope, session)`` alone, so an earlier generation's already-completed
receipt collided with this generation's, even though its content (a
different ``code_hash``) genuinely differed.

This file proves the fix at the level ``tests/test_v2_ops_effects_graph.py``
already exercises each effect: two DISTINCT generations of ONE session both
complete engineering_gate/ledger_export/backup, each recording its OWN
watermark receipt without touching the other's; an identical retry of either
generation stays idempotent; and a same-generation content change still
conflicts (the invariant this fix must not weaken). It also proves the
health/streak hook (§5.5 item 2, not implemented) cannot double-count a
night just because two generations both observed it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.v2.ops.catalog import transaction
from engine.v2.ops.effects_graph import backup_effect, engineering_gate_effect, ledger_export_effect
from engine.v2.ops.errors import OpsError
from engine.v2.ops.outbox import watermark, watermark_would_conflict
from tests.ops_support import catalog
from tests.test_v2_ops_effects_graph import _commit, _open, _params, _row, _seed_decisions
from tests.test_v2_ops_effects_graph import _submit_and_claim as _claim

REPO = Path(__file__).resolve().parents[1]
SESSION = "2026-09-12"


def _gen_params(kind, session, scope, deployment, decision_clock, **extra):
    return _params(kind, session, scope, deployment=deployment, decision_clock=decision_clock,
                   input_bindings={"legacy_manifest.json": "art_" + deployment}, **extra)


def _watermarks(conn, scope, stage):
    return {row["generation"]: (row["occurrence"], row["receipt_ref"]) for row in conn.execute(
        "SELECT generation, occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' "
        "AND scope=? AND stage=?", (scope, stage))}


# --------------------------------------------------------------------------
# engineering_gate: two generations, each with their own receipt; retry and
# same-generation-conflict behaviour.
# --------------------------------------------------------------------------


def test_engineering_gate_two_generations_each_get_their_own_receipt(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        gen1 = _claim(conn, clock, supervisor, kind="engineering_gate", key="gate-gen1",
                     parameters=_gen_params("engineering_gate", SESSION, scope,
                                            "shadow:impl-1", "2026-09-12T01:00:00.000000Z"))
        result1 = engineering_gate_effect(conn, store, gen1, REPO, clock=clock)
        _commit(conn, clock, gen1, result1)

        gen2 = _claim(conn, clock, supervisor, kind="engineering_gate", key="gate-gen2",
                     parameters=_gen_params("engineering_gate", SESSION, scope,
                                            "shadow:impl-2", "2026-09-12T05:00:00.000000Z"))
        result2 = engineering_gate_effect(conn, store, gen2, REPO, clock=clock)
        _commit(conn, clock, gen2, result2)

        marks = _watermarks(conn, scope, "engineering_gate")
        assert len(marks) == 2
        # Both name the SAME occurrence (one night); neither rewrote the
        # other's receipt (their generation keys differ, per _generation_ref).
        assert {occurrence for occurrence, _ in marks.values()} == {SESSION}
        gen1_ref, gen2_ref = marks.keys()
        assert gen1_ref != gen2_ref

        # An identical retry of generation 1 (fresh job id, SAME plan
        # identity) is a no-op: still exactly two rows, generation 1's
        # unchanged.
        retry = _claim(conn, clock, supervisor, kind="engineering_gate", key="gate-gen1-retry",
                       parameters=_gen_params("engineering_gate", SESSION, scope,
                                              "shadow:impl-1", "2026-09-12T01:00:00.000000Z"))
        result_retry = engineering_gate_effect(conn, store, retry, REPO, clock=clock)
        _commit(conn, clock, retry, result_retry)
        marks_after_retry = _watermarks(conn, scope, "engineering_gate")
        assert marks_after_retry == marks
    finally:
        conn.close()


def test_watermark_same_generation_different_content_still_conflicts(tmp_path):
    """The invariant every generation-aware effect still relies on: WITHIN
    one generation, a completed occurrence with a different receipt is
    refused -- unchanged from the pre-generation contract, just scoped one
    level deeper. Exercised directly against ``outbox.watermark`` since all
    four effects (engineering_gate/ledger_export/backup/publication) funnel
    through this exact same function."""
    conn, clock, _ = catalog(tmp_path)
    with transaction(conn):
        watermark(conn, "nightly", "shadow", "engineering_gate", SESSION, "receipt-a",
                 clock=clock, generation="gen-1")
        # A different generation for the same occurrence is NOT a conflict.
        watermark(conn, "nightly", "shadow", "engineering_gate", SESSION, "receipt-b",
                 clock=clock, generation="gen-2")
    assert watermark_would_conflict(conn, "nightly", "shadow", "engineering_gate", SESSION,
                                    "receipt-a", generation="gen-1") is False
    assert watermark_would_conflict(conn, "nightly", "shadow", "engineering_gate", SESSION,
                                    "different-receipt", generation="gen-1") is True
    with pytest.raises(OpsError, match="IDEMPOTENCY_CONFLICT"):
        with transaction(conn):
            watermark(conn, "nightly", "shadow", "engineering_gate", SESSION, "different-receipt",
                     clock=clock, generation="gen-1")
    # generation 2's own row is untouched by the refused write against gen-1.
    assert conn.execute(
        "SELECT receipt_ref FROM watermarks WHERE pipeline='nightly' AND scope='shadow' "
        "AND stage='engineering_gate' AND generation='gen-2'").fetchone()["receipt_ref"] == "receipt-b"
    conn.close()


# --------------------------------------------------------------------------
# ledger_export: two generations both complete, sharing the same underlying
# decisions content (generation-independent) but each with its own receipt.
# --------------------------------------------------------------------------


def test_ledger_export_two_generations_each_get_their_own_receipt(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, predictions=[_row("evt-1", "pred")])

        gen1 = _claim(conn, clock, supervisor, kind="ledger_export", key="export-gen1",
                     parameters=_gen_params("ledger_export", SESSION, scope,
                                            "shadow:impl-1", "2026-09-12T01:00:00.000000Z"))
        result1 = ledger_export_effect(conn, store, gen1, root, REPO, clock=clock)
        _commit(conn, clock, gen1, result1)

        gen2 = _claim(conn, clock, supervisor, kind="ledger_export", key="export-gen2",
                     parameters=_gen_params("ledger_export", SESSION, scope,
                                            "shadow:impl-2", "2026-09-12T05:00:00.000000Z"))
        result2 = ledger_export_effect(conn, store, gen2, root, REPO, clock=clock)
        _commit(conn, clock, gen2, result2)

        marks = _watermarks(conn, scope, "export")
        assert len(marks) == 2
        # Same underlying decisions content -> same tar content -> same
        # receipt_ref for both, but each generation still owns its own row.
        receipts = {receipt for _, receipt in marks.values()}
        assert receipts == {result1[1][0][1].content_hash}
    finally:
        conn.close()


# --------------------------------------------------------------------------
# backup: two generations both complete -- the effect's own enqueue key
# (not only its watermark) had to become generation-scoped too, or the
# second generation's ``run_backup`` would find nothing pending and refuse
# with STALE_EXPECTATION once generation 1's backup already delivered.
# --------------------------------------------------------------------------


def test_backup_two_generations_each_deliver_their_own_effect(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, predictions=[_row("evt-1", "pred")])

        gen1 = _claim(conn, clock, supervisor, kind="backup", key="backup-gen1",
                     parameters=_gen_params("backup", SESSION, scope,
                                            "shadow:impl-1", "2026-09-12T01:00:00.000000Z"))
        result1 = backup_effect(conn, store, gen1, root, clock=clock)
        _commit(conn, clock, gen1, result1)

        gen2 = _claim(conn, clock, supervisor, kind="backup", key="backup-gen2",
                     parameters=_gen_params("backup", SESSION, scope,
                                            "shadow:impl-2", "2026-09-12T05:00:00.000000Z"))
        # Before the fix this raised STALE_EXPECTATION: generation 1's
        # backup outbox row (same scope/session, generation-blind key) was
        # already 'delivered', so ``run_backup``'s own claim() found nothing
        # pending for generation 2 to complete.
        result2 = backup_effect(conn, store, gen2, root, clock=clock)
        _commit(conn, clock, gen2, result2)

        marks = _watermarks(conn, scope, "backup")
        assert len(marks) == 2
        assert {occurrence for occurrence, _ in marks.values()} == {SESSION}
        rows = conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchall()
        assert [r["state"] for r in rows] == ["delivered", "delivered"]
    finally:
        conn.close()


# --------------------------------------------------------------------------
# health/streak hook: §5.5 item 2 is not implemented (engineering_gate_effect
# does not call health.record_check yet), but the table it will write into
# already counts by occurrence, never by generation or retry.
# --------------------------------------------------------------------------


def test_health_streak_counts_one_occurrence_per_night_not_per_generation(tmp_path):
    """``health_observations`` is keyed ``PRIMARY KEY(occurrence, kind)`` --
    recording an engineering observation for the SAME night under two
    different generations still counts as one scheduled occurrence, never
    two. This is the hook guide §5.5 item 2 ("populate live engineering
    health from real observations") writes into; item 2 itself stays out of
    scope here."""
    from engine.v2.ops.health import budget_streak, record_check

    conn, clock, _ = catalog(tmp_path)
    record_check(conn, SESSION, "engineering", True, {"generation": "gen-1"})
    record_check(conn, SESSION, "engineering", True, {"generation": "gen-2"})
    assert conn.execute("SELECT COUNT(*) FROM health_observations").fetchone()[0] == 1
    streak = budget_streak(conn)
    assert streak["ok"] is True
    assert streak["latest"] == {"generation": "gen-2"}
    conn.close()
