"""Receipt lineage (schema v8): a ``price_history`` capture generation's
accepted legacy read-set is inherited from its base receipt.

Closes the integration gap ``engine.v2.ops.price_history_store``/
``engine.v2.ops.generation_binding`` documented (their module docstrings,
"Receipt lineage" / "Honest attempt identity" sections): a capture commits
its generation under its own honest, non-scheduler ``attempt_id``, which
never gains an ``attempt_input_bindings`` row for ``legacy_manifest.json``
-- so without lineage, a snapshot-mode plan pinned to a capture-only head
could never launch a barrier-only stage (``legacy_finality``/
``legacy_model_evidence``/``legacy_selfcheck``).

Reuses the real SQLite catalog + ``ArtifactStore`` + ``Service`` fixtures
``tests.test_v2_ops_snapshot_stages.Case``/``case`` and the
generation-binding test helpers already established in
``tests.test_v2_ops_tier4_coverage`` (``_accept_generation``, ``_claim``,
``_publish_manifest``, ``_service``) rather than re-deriving them -- the
same cross-file reuse pattern that file itself already uses for ``Case``.
"""
from __future__ import annotations

import pytest

from engine.v2.data import reference_catalog
from engine.v2.ops import price_history_store as ph_store
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.generation_binding import (
    accepted_generation_refs,
    record_price_history_lineage,
    refuse_generation_mismatch,
)
from engine.v2.ops.price_history_store import capture as ph_capture
from engine.v2.ops.snapshot_planning import pin_snapshot_inputs
from engine.v2.ops.stages import BARRIER_ONLY_REASONS
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.test_v2_ops_snapshot_stages import SESSION, Case, case  # noqa: F401
from tests.test_v2_ops_tier4_coverage import _accept_generation, _claim, _publish_manifest, _service

_HASH_A = "sha256:" + "1" * 64
_HASH_MISMATCH = "sha256:" + "2" * 64


def _insert_receipt(conn, *, receipt_id, snapshot_id, status="committed"):
    """A bare ``data_import_receipts`` row with a made-up, non-FK'd
    ``attempt_id`` -- exactly the shape ``price_history_store.
    _commit_generation`` mints (its module docstring, "Honest attempt
    identity"): no real ``attempts`` row, so it can never gain its own
    ``attempt_input_bindings`` entry. Stands in for a real capture receipt
    throughout this file's chain-walk tests, without needing a full
    ``capture()`` run for each one."""
    result_snapshot_id = snapshot_id if status == "committed" else None
    with transaction(conn):
        conn.execute(
            "INSERT INTO data_import_receipts (receipt_id, attempt_id, fence, source_manifest_hash, "
            "result_snapshot_id, status, registered_at, scope) VALUES (?, ?, 1, ?, ?, ?, "
            "'2026-01-01T00:00:00.000000Z', 'shadow')",
            (receipt_id, "attempt_ph_" + receipt_id, "sha256:" + "0" * 64, result_snapshot_id, status))


def _link(conn, *, receipt_id, base_receipt_id):
    with transaction(conn):
        record_price_history_lineage(conn, receipt_id=receipt_id, base_receipt_id=base_receipt_id)


# --------------------------------------------------------------------------
# the lineage walk itself -- synthetic receipts, no real capture() run
# --------------------------------------------------------------------------


def test_capture_receipt_resolves_through_lineage_to_base_manifest(case):
    """import A -> capture: a new plan pins the capture receipt -> barrier-
    stage launch validation accepts against A's manifest."""
    _accept_generation(case, [("data/x.csv", _HASH_A)], receipt_id="lin-A")
    _insert_receipt(case.conn, receipt_id="lin-cap", snapshot_id=case.snap.snapshot_id)
    _link(case.conn, receipt_id="lin-cap", base_receipt_id="lin-A")

    assert (accepted_generation_refs(case.conn, case.store, receipt_id="lin-cap")
           == accepted_generation_refs(case.conn, case.store, receipt_id="lin-A")
           == {"data/x.csv": _HASH_A})
    barrier = {"file_refs": [{"path": "data/x.csv", "content_hash": _HASH_A}]}
    refuse_generation_mismatch(case.conn, case.store, receipt_id="lin-cap",
                               barrier_manifest=barrier)  # does not raise


def test_capture_receipt_with_differing_barrier_manifest_still_refuses(case):
    """A barrier manifest that differs from A's still refuses
    ``generation_mismatch`` -- lineage changes WHICH receipt is read, never
    the comparison itself."""
    _accept_generation(case, [("data/x.csv", _HASH_A)], receipt_id="lin-mismatch-A")
    _insert_receipt(case.conn, receipt_id="lin-mismatch-cap", snapshot_id=case.snap.snapshot_id)
    _link(case.conn, receipt_id="lin-mismatch-cap", base_receipt_id="lin-mismatch-A")

    barrier = {"file_refs": [{"path": "data/x.csv", "content_hash": _HASH_MISMATCH}]}
    with pytest.raises(OpsError) as err:
        refuse_generation_mismatch(case.conn, case.store, receipt_id="lin-mismatch-cap",
                                   barrier_manifest=barrier)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_mismatch"
    assert err.value.problem.details["paths"] == ["data/x.csv"]


def test_two_successive_captures_still_resolve_to_the_original_base(case):
    """Two-level chain: cap2 -> cap1 -> A, where cap1 itself has no binding
    either -- both hops must be walked."""
    _accept_generation(case, [("data/x.csv", _HASH_A)], receipt_id="lin-chain-A")
    _insert_receipt(case.conn, receipt_id="lin-chain-cap1", snapshot_id=case.snap.snapshot_id)
    _link(case.conn, receipt_id="lin-chain-cap1", base_receipt_id="lin-chain-A")
    _insert_receipt(case.conn, receipt_id="lin-chain-cap2", snapshot_id=case.snap.snapshot_id)
    _link(case.conn, receipt_id="lin-chain-cap2", base_receipt_id="lin-chain-cap1")

    assert (accepted_generation_refs(case.conn, case.store, receipt_id="lin-chain-cap2")
           == accepted_generation_refs(case.conn, case.store, receipt_id="lin-chain-A")
           == {"data/x.csv": _HASH_A})


def test_capture_receipt_with_no_lineage_and_no_binding_refuses(case):
    """A missing chain: no lineage row at all, and no manifest binding --
    refuses exactly as an unresolvable receipt_id always has."""
    _insert_receipt(case.conn, receipt_id="lin-orphan", snapshot_id=case.snap.snapshot_id)
    with pytest.raises(OpsError) as err:
        refuse_generation_mismatch(case.conn, case.store, receipt_id="lin-orphan",
                                   barrier_manifest={"file_refs": []})
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_receipt_missing"


def test_lineage_row_pointing_to_a_non_committed_receipt_refuses(case):
    """A broken chain: the lineage row's own base receipt exists but was
    never committed (a failed import, say) -- refused, never silently
    treated as accepting nothing versus a deliberate empty-receipt no-op."""
    _insert_receipt(case.conn, receipt_id="lin-broken-base", snapshot_id=case.snap.snapshot_id,
                    status="failed")
    _insert_receipt(case.conn, receipt_id="lin-broken-cap", snapshot_id=case.snap.snapshot_id)
    _link(case.conn, receipt_id="lin-broken-cap", base_receipt_id="lin-broken-base")
    with pytest.raises(OpsError) as err:
        refuse_generation_mismatch(case.conn, case.store, receipt_id="lin-broken-cap",
                                   barrier_manifest={"file_refs": []})
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_receipt_missing"


def test_lineage_cycle_does_not_hang_and_refuses(case):
    """Cycle-safety: two capture-shaped receipts pointing at each other,
    neither with a binding of its own -- the walk terminates (never loops
    forever) and refuses, since no ancestor in the cycle ever has one."""
    _insert_receipt(case.conn, receipt_id="lin-cyc-1", snapshot_id=case.snap.snapshot_id)
    _insert_receipt(case.conn, receipt_id="lin-cyc-2", snapshot_id=case.snap.snapshot_id)
    _link(case.conn, receipt_id="lin-cyc-1", base_receipt_id="lin-cyc-2")
    _link(case.conn, receipt_id="lin-cyc-2", base_receipt_id="lin-cyc-1")
    assert accepted_generation_refs(case.conn, case.store, receipt_id="lin-cyc-1") is None


def test_full_import_receipt_still_resolves_via_its_own_binding(case):
    """A full import receipt (a real attempt with its own manifest binding,
    no lineage row at all) is returned directly -- the lineage table is
    never consulted for one."""
    _accept_generation(case, [("data/x.csv", _HASH_A)], receipt_id="lin-full")
    refs = accepted_generation_refs(case.conn, case.store, receipt_id="lin-full")
    assert refs == {"data/x.csv": _HASH_A}


# --------------------------------------------------------------------------
# atomicity: the lineage row is absent when a capture's commit fails
# --------------------------------------------------------------------------


def test_lineage_row_absent_when_captures_commit_fails(tmp_path, monkeypatch):
    sec = contract_for("securities")
    sec_ref = contract_ref_for(sec)
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(
        store, sec, sec_ref,
        [dict(ticker="AAPL", year=2024, first_date=None, last_date=None, mcap_usd=1.5e9,
             mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
             n_obs=250, src="orats")],
        "2024")
    commit_tables(conn, clock, {"securities": [record]}, {"securities": sec}, scope="shadow",
                  receipt_id="atomic-base")
    ref = reference_catalog.ReferenceInput(kind="calendar", legacy_path="calendar.csv",
                                           object_id="art_cal", content_hash="sha256:" + "cd" * 32,
                                           byte_size=5)
    with transaction(conn):
        reference_catalog.insert_reference_inputs(conn, "atomic-base", [ref])

    px_dir = tmp_path / "legacy" / "earnings_predictions" / "data" / "raw" / "yfinance"
    px_dir.mkdir(parents=True)
    (px_dir / "px_AAPL.csv").write_text(
        "date,close_adj,close_raw,high_raw\n2024-01-01,100.0,100.0,100.0\n")

    real_record = ph_store.record_price_history_lineage

    def _boom(c, **kwargs):
        real_record(c, **kwargs)  # the row IS staged, inside the same open transaction
        raise RuntimeError("forced failure after the lineage row is staged")

    monkeypatch.setattr(ph_store, "record_price_history_lineage", _boom)
    with pytest.raises(RuntimeError):
        ph_store.capture(conn, store, tmp_path / "legacy", root=tmp_path, scope="shadow", clock=clock)

    assert conn.execute("SELECT COUNT(*) FROM data_receipt_lineage").fetchone()[0] == 0
    # The receipt itself never committed either -- one transaction, one rollback.
    assert conn.execute(
        "SELECT COUNT(*) FROM data_import_receipts WHERE receipt_id != 'atomic-base'"
    ).fetchone()[0] == 0
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'").fetchone()
    assert head["snapshot_id"] is not None  # the base head is untouched


# --------------------------------------------------------------------------
# end-to-end: a real capture, a real plan, a real (launch-only) barrier pin
# --------------------------------------------------------------------------


def test_end_to_end_capture_then_plan_then_barrier_launch_pins_through_lineage(case, monkeypatch, tmp_path):
    """The full integration this task closes: a real ``price_history``
    capture commits a generation with no manifest binding of its own; a
    fresh snapshot-mode plan built AFTER it (``pin_snapshot_inputs``, the
    real planner) resolves the capture's own receipt; and each barrier-only
    stage's pin step (``Service._pin_read_set`` -- launch only, no real
    worker runs) accepts it against the BASE receipt's real manifest
    binding, via the lineage walk."""
    import engine.v2.ops.supervisor as supervisor_mod

    file_refs = [("data/x.csv", _HASH_A)]
    _accept_generation(case, file_refs, receipt_id="e2e-base")
    # e2e-base needs the same reference inputs case.request was built from,
    # so both pin_snapshot_inputs and capture's own reference-input copy
    # (``price_history_store._copy_reference_inputs``) find something to read.
    existing = reference_catalog.reference_inputs_for_receipt(case.conn, receipt_id="r1-references")
    with transaction(case.conn):
        reference_catalog.insert_reference_inputs(case.conn, "e2e-base", existing)

    px_dir = tmp_path / "e2e_legacy" / "earnings_predictions" / "data" / "raw" / "yfinance"
    px_dir.mkdir(parents=True)
    (px_dir / "px_AAA.csv").write_text(
        "date,close_adj,close_raw,high_raw\n2020-01-02,10.0,10.0,10.0\n")
    report = ph_capture(case.conn, case.store, tmp_path / "e2e_legacy", root=case.root, scope="shadow",
                        clock=case.clock)

    plan = pin_snapshot_inputs(case.conn, case.store, "shadow", tickers=("AAA", "BBB"),
                               year_start=2020, year_end=2021,
                               expected_population=("AAA|S1|2020-01-15",), clock=case.clock,
                               session=SESSION)
    assert plan["snapshot_generation_receipt_id"] == report["receipt_id"]
    assert plan["snapshot_id"] == report["result_snapshot_id"]

    manifest_ref, document = _publish_manifest(case, file_refs)
    monkeypatch.setattr(supervisor_mod, "pin_read_set", lambda *a, **k: None)
    for kind in BARRIER_ONLY_REASONS:
        parameters = {"input_bindings": {"legacy_manifest.json": manifest_ref.artifact_id},
                      "snapshot_generation_id": plan["snapshot_id"],
                      "snapshot_generation_scope": "shadow",
                      "snapshot_generation_receipt_id": plan["snapshot_generation_receipt_id"]}
        claim = _claim(kind, parameters=parameters, attempt_id="att_" + kind)
        result = _service(case)._pin_read_set(claim)  # does not raise
        assert result == document
