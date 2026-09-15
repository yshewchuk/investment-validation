"""2026-09-15: the outcome-session backfill (``engine.v2.ops.session_backfill``)
must actually run in production, not just in tests that call it directly.
This file pins the wiring: ``engine.v2.ops.cli.main`` -- the single
production entry point every real writer of a ``kind="outcome"`` decision
goes through (``ops serve``'s nightly ``legacy_settlement`` action and
``ops ledger import-history`` both dispatch from inside it, right after
``open_catalog``) -- runs the backfill automatically, exactly once per
catalog, and that ``ops doctor``/``ops health`` surface what is left NULL.
"""
from __future__ import annotations

import base64
import json

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore
from engine.v2.ledger.decisions import import_lines, rows, set_authority
from engine.v2.ops import cli, session_backfill
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.decision_commit import import_settlement_candidates_in_transaction
from engine.v2.ops.catalog import transaction
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit
from tests.ops_support import DEFAULT_POLICY, FakeClock, sample

STAMP = "2026-09-13T12:00:00.000000Z"
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _catalog_at(root):
    """Same shape as ``tests.ops_support.catalog``, but at the exact path
    ``engine.v2.ops.cli.main`` itself opens (``<root>/catalog.sqlite``) --
    needed so a later ``cli.main(["--root", str(root), ...])`` call in the
    same test sees the state seeded here."""
    clock = FakeClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    return conn, clock, Supervisor(epoch, "boot")


def _seed_prediction(conn, row_id, event_date, *, ticker="FAKE", strategy="TWIN-P"):
    from engine.v2.ledger.decisions import insert
    payload = {"row_id": row_id, "ticker": ticker, "strategy": strategy,
              "event_date": event_date, "settlement": {"policy": "fixed"}}
    with transaction(conn):
        insert(conn, logical_key=row_id, decision_id="prediction:" + row_id, payload=payload,
              purpose="shadow", kind="prediction", validations={}, created_at=STAMP)


def _raw(row):
    return json.dumps(row).encode()


def _settlement_claim(conn, clock, supervisor, *, session="2026-09-13"):
    job = JobSpec(kind="legacy_settlement", implementation_ref="code", spec_hash=None,
                  environment_ref="env", parameters={
                      "expected_ids": ("legacy_settlement",), "session": session},
                  output_namespace="shadow", resource_class="legacy_rebuild",
                  retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace="shadow", idempotency_key="settle",
                                                    principal="operator", job=job), clock=clock)
    return claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                      clock=clock, registry=registry())


def _register_nightly_settlement_artifact(conn, store, clock, claim, document):
    ref = store.publish_bytes(json.dumps(document, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, claim.attempt_id, clock)
        conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                     (claim.attempt_id, "legacy_settlement", ref.artifact_id))
    return ref.content_hash


# --------------------------------------------------------------------------
# the hook runs exactly once per catalog, from the production entry point
# --------------------------------------------------------------------------


def test_first_production_open_backfills_once_second_open_does_not_rescan(tmp_path, monkeypatch):
    root = tmp_path
    conn, clock, _ = _catalog_at(root)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    _seed_prediction(conn, "prediction-1", "2026-09-09")
    # A pre-fix ``ops ledger import-history``-shaped row: no generation_ref.
    old_row = {"row_id": "prediction-1", "ticker": "FAKE", "strategy": "TWIN-P",
              "event_date": "2026-09-09", "settlement": {"policy": "fixed"},
              "status": "unresolvable", "resolved_at": "2026-09-01T23:00:00+00:00"}
    with transaction(conn):
        import_lines(conn, "legacy_file_2026-09-01", [_raw(old_row)], kind="outcome",
                    created_at=STAMP)
    assert rows(conn, kind="outcome")[0]["generation_ref"] is None
    conn.close()

    calls = []
    real_map = session_backfill._nightly_session_map

    def _counting_map(conn, store):
        calls.append(1)
        return real_map(conn, store)

    monkeypatch.setattr(session_backfill, "_nightly_session_map", _counting_map)

    code = cli.main(["--root", str(root), "health"])
    assert code == 0
    assert len(calls) == 1

    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        [backfilled] = rows(conn, kind="outcome")
        assert backfilled["generation_ref"] == "2026-09-01"
    finally:
        conn.close()

    # A second production-path open of the SAME catalog must not rescan --
    # the one-shot schema_versions marker short-circuits before any artifact
    # or row is touched.
    code = cli.main(["--root", str(root), "health"])
    assert code == 0
    assert len(calls) == 1


def test_init_marks_a_fresh_catalog_applied_immediately(tmp_path, monkeypatch):
    """``ops init`` also dispatches from inside ``main()``'s catalog-opening
    branch -- a brand-new (empty) catalog costs one scan of zero rows and is
    marked applied right away, so the very first real command afterwards
    (``ops health``, ``ops serve``, ``ops ledger import-history``) costs only
    the marker's SELECT."""
    root = tmp_path
    calls = []
    real_map = session_backfill._nightly_session_map

    def _counting_map(conn, store):
        calls.append(1)
        return real_map(conn, store)

    monkeypatch.setattr(session_backfill, "_nightly_session_map", _counting_map)

    assert cli.main(["--root", str(root), "init"]) == 0
    assert len(calls) == 1

    assert cli.main(["--root", str(root), "health"]) == 0
    assert len(calls) == 1


# --------------------------------------------------------------------------
# doctor/health surface what is left undetermined -- never hidden
# --------------------------------------------------------------------------


def test_doctor_and_health_report_the_undetermined_count(tmp_path, capsys):
    root = tmp_path
    conn, clock, _ = _catalog_at(root)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    _seed_prediction(conn, "prediction-good", "2026-09-09")
    _seed_prediction(conn, "prediction-bad", "2026-09-01")
    good_row = {"row_id": "prediction-good", "ticker": "FAKE", "strategy": "TWIN-P",
               "event_date": "2026-09-09", "settlement": {"policy": "fixed"},
               "status": "unresolvable", "resolved_at": "2026-09-01T23:00:00+00:00"}
    bad_row = {"row_id": "prediction-bad", "ticker": "FAKE", "strategy": "TWIN-P",
              "event_date": "2026-09-01", "settlement": {"policy": "fixed"},
              "status": "unresolvable", "resolved_at": "not-a-real-timestamp"}
    with transaction(conn):
        import_lines(conn, "legacy_good", [_raw(good_row)], kind="outcome", created_at=STAMP)
        import_lines(conn, "legacy_bad", [_raw(bad_row)], kind="outcome", created_at=STAMP)
    conn.close()

    # Before any write-opening command has run against this root, doctor
    # (a read-only connection, never triggers the backfill itself) reports
    # the pre-fix state plainly.
    before = cli.doctor(root, clock)
    assert before["outcome_sessions"]["backfill_applied"] is False
    assert before["outcome_sessions"]["undetermined"] == 2

    assert cli.main(["--root", str(root), "health"]) == 0
    capsys.readouterr()  # discard this call's stdout; only the next is asserted below

    after = cli.doctor(root, clock)
    assert after["outcome_sessions"]["backfill_applied"] is True
    assert after["outcome_sessions"]["undetermined"] == 1  # bad_row stays NULL

    code = cli.main(["--root", str(root), "health"])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["undetermined_outcome_sessions"] == 1


# --------------------------------------------------------------------------
# end to end: a same-session rerun commits 0 after the automatic backfill
# --------------------------------------------------------------------------


def test_same_session_rerun_after_automatic_backfill_commits_zero(tmp_path):
    root = tmp_path
    conn, clock, supervisor = _catalog_at(root)
    store = ArtifactStore(root)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-attempt13-1"
    _seed_prediction(conn, row_id, "2026-09-09")
    claim = _settlement_claim(conn, clock, supervisor)  # requested session "2026-09-13"

    old_row = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-09-09",
              "settlement": {"policy": "fixed"}, "status": "unresolvable",
              "resolved_at": "2026-09-10T01:00:00+00:00"}
    document = {"session": "2026-09-10", "requested_session": "2026-09-13", "rows": [old_row]}
    source_hash = _register_nightly_settlement_artifact(conn, store, clock, claim, document)
    with transaction(conn):
        import_lines(conn, source_hash, [_raw(old_row)], kind="outcome", created_at=STAMP)
    assert rows(conn, kind="outcome")[0]["generation_ref"] is None
    conn.close()

    # The real production entry point -- not a direct call to
    # backfill_outcome_sessions -- recovers the session automatically.
    assert cli.main(["--root", str(root), "health"]) == 0

    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        assert rows(conn, kind="outcome")[0]["generation_ref"] == "2026-09-10"

        new_row = dict(old_row, resolved_at="2026-09-10T09:00:00+00:00")
        candidate = [{"row": new_row,
                     "original_b64": base64.b64encode(_raw(new_row)).decode("ascii")}]
        ref = store.publish_bytes(json.dumps({"rows": candidate}, sort_keys=True).encode(),
                                  schema_ref="legacy_action.v1.0")
        skipped = []
        with transaction(conn):
            repeated = import_settlement_candidates_in_transaction(
                conn, claim, ref, candidate, clock=clock, session="2026-09-10",
                on_skip=lambda r, reason: skipped.append((r, reason)))
        assert repeated == []
        assert skipped == [(row_id, "already_observed_this_session")]
        assert len(rows(conn, kind="outcome")) == 1
        assert conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0] == 0
    finally:
        conn.close()
