"""#1060: a ``commit_snapshot`` replay applies the still-pending head CAS.

A prior receipt short-circuits the second call; before the fix that shortcut
returned the receipt (claiming the head moved) without ever advancing a head
that still sat at the caller's expected parent. Synthetic fragments/snapshots
are built the same way ``tests/v2/data/test_v2_data_import.py`` and its
siblings build them for a real ``catalog.commit_snapshot`` (publish, inspect,
dataset manifest, snapshot ref — see ``tests.data_scan_support``).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine.v2.data import catalog, manifests  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    RECEIPT,
    catalog_and_store,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)

SCOPE = "shadow"
_SEC = contract_for("securities")
_SEC_REF = contract_ref_for(_SEC)


def _row(ticker: str, year: int) -> dict:
    return dict(ticker=ticker, year=year, first_date=None, last_date=None, mcap_usd=1.5e9,
                mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
                n_obs=250, src="orats")


def _build(store, ticker: str, year: int):
    record = publish_and_inspect(store, _SEC, _SEC_REF, [_row(ticker, year)], partition_key=str(year))
    manifest = manifests.dataset_manifest(
        _SEC_REF, [record], knowledge_mode="reconstructed", coverage_receipt_refs=(RECEIPT,),
        availability_evidence_refs=())
    snap = manifests.snapshot_ref(
        {"securities": manifest}, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=(RECEIPT,))
    return record, manifest, snap


def _commit(conn, clock, store, *, record, manifest, snap, receipt_id,
            expected_head, expected_generation):
    return catalog.commit_snapshot(
        conn, scope=SCOPE, request_hash=fake_hash(receipt_id), contracts=[_SEC],
        objects=[record.object_ref], records=[record], manifests=[manifest], snapshot=snap,
        expected_head_snapshot_id=expected_head, expected_head_generation=expected_generation,
        receipt_id=receipt_id, attempt_id="att-" + receipt_id, fence=1,
        fence_check=lambda _c: None, clock=clock, store=store)


def _head(conn):
    row = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope=?",
                       (SCOPE,)).fetchone()
    return None if row is None else (row["snapshot_id"], row["generation"])


def _seed_h0_and_s1(conn, clock, store):
    """Head H0 (gen 1), then S1 committed at H0 with receipt ``r-s1`` (gen 2)."""
    rec_h, man_h, snap_h = _build(store, "AAA", 2024)
    _commit(conn, clock, store, record=rec_h, manifest=man_h, snap=snap_h, receipt_id="r-h0",
            expected_head=None, expected_generation=0)
    rec_s, man_s, snap_s = _build(store, "BBB", 2024)
    receipt = _commit(conn, clock, store, record=rec_s, manifest=man_s, snap=snap_s,
                      receipt_id="r-s1", expected_head=snap_h.snapshot_id,
                      expected_generation=1)
    assert _head(conn) == (snap_s.snapshot_id, 2)
    s1_args = dict(record=rec_s, manifest=man_s, snap=snap_s, receipt_id="r-s1",
                   expected_head=snap_h.snapshot_id, expected_generation=1)
    return snap_h.snapshot_id, snap_s.snapshot_id, receipt, s1_args


def test_replay_applies_cas_when_head_sits_at_expected_parent(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    try:
        h0, s1, original, s1_args = _seed_h0_and_s1(conn, clock, store)
        assert original.resulting_head_snapshot_id == s1

        # A later import re-committed the bare parent, advancing generation.
        conn.execute("UPDATE data_snapshot_heads SET snapshot_id=?, generation=? WHERE scope=?",
                     (h0, 3, SCOPE))
        assert _head(conn) == (h0, 3)

        replay = _commit(conn, clock, store, **{**s1_args, "expected_generation": 3})
        assert _head(conn) == (s1, 4)
        assert replay.resulting_head_snapshot_id == s1
        assert replay.resulting_head_generation == 4
    finally:
        conn.close()


def test_replay_leaves_head_when_already_advanced(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    try:
        h0, s1, _, s1_args = _seed_h0_and_s1(conn, clock, store)
        assert _head(conn) == (s1, 2)

        replay = _commit(conn, clock, store, **s1_args)
        assert _head(conn) == (s1, 2)
        assert replay.resulting_head_snapshot_id == s1
        assert replay.resulting_head_generation == 2
        assert h0 != s1
    finally:
        conn.close()


def test_replay_leaves_head_when_moved_to_a_third_snapshot(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    try:
        _h0, s1, _, s1_args = _seed_h0_and_s1(conn, clock, store)
        rec_x, man_x, snap_x = _build(store, "CCC", 2025)
        _commit(conn, clock, store, record=rec_x, manifest=man_x, snap=snap_x, receipt_id="r-x",
                expected_head=s1, expected_generation=2)
        x = snap_x.snapshot_id
        assert x != s1 and _head(conn) == (x, 3)

        replay = _commit(conn, clock, store, **s1_args)
        assert _head(conn) == (x, 3)
        assert replay.resulting_head_snapshot_id == s1
        assert replay.resulting_head_generation == 2
    finally:
        conn.close()


def test_replay_with_stale_expected_generation_leaves_head(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    try:
        h0, s1, _, s1_args = _seed_h0_and_s1(conn, clock, store)
        conn.execute("UPDATE data_snapshot_heads SET snapshot_id=?, generation=? WHERE scope=?",
                     (h0, 3, SCOPE))
        assert _head(conn) == (h0, 3)

        replay = _commit(conn, clock, store, **{**s1_args, "expected_generation": 1})
        assert _head(conn) == (h0, 3)
        assert replay.resulting_head_snapshot_id == s1
    finally:
        conn.close()
