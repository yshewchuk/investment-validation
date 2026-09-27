"""S4C Part 4 (revised after Opus BLOCK(3) on dc7f9360): computed_moves_refresh
is no longer submitted by ``build_legacy_job_requests`` in any mode -- doing
that pinned the same shadow head the REQUIRED "refresh" stage was about to
advance, so whichever native job committed second failed
(``_check_head_expectation``), and ``submit_graph``'s all-or-nothing insert
let a build-time problem in the optional stage refuse "refresh" with it. The
only submitter now is ``nightly.submit_computed_moves_refresh_if_ready``,
called by ``supervisor.Service``'s own tick loop AFTER a native "refresh" job
has already succeeded, resolving the shadow head fresh at that point.

``run_computed_moves_refresh`` itself (the worker) is already covered end to
end by ``tests/test_v2_ops_computed_moves_store.py``; these tests prove only
nightly.py's/supervisor.py's own new code, never re-testing the worker or the
store's own target-selection logic (monkeypatched here to a fixed list).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.v2.contracts import SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.data.repository import Repository
from engine.v2.ops import computed_moves_store, incremental_data
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.calendar_moves_jobs import (
    COMPUTED_MOVES_REFRESH_ACTION,
    CalendarMovesParameters,
    calendar_moves_job_spec,
)
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.incremental_data import AcquisitionOutcome
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.nightly import (
    NATIVE_COMPUTED_MOVES_ACCOUNT,
    NATIVE_REFRESH_ACTION,
    _NATIVE_ACTION_STAGES,
    _build_native_computed_moves_plan,
    _computed_moves_identity,
    _computed_moves_refresh_key,
    _session_from_refresh_key,
    _stage_sequence,
    build_legacy_job_requests,
    build_nightly_plan,
    submit_computed_moves_refresh_if_ready,
)
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit
from engine.v2.ops.supervisor import Service
from tests.data_scan_support import commit_tables, contract_for
from tests.ops_support import TEST_POLICY, AdmissionWatch, catalog, request

ROOT = Path(__file__).resolve().parents[1]
_EVENTS = contract_for("earnings_events")
_DAILY = contract_for("daily_market")
_POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _commit_parent(conn, clock, store, *, tables=("earnings_events", "daily_market")):
    contracts = {name: (_EVENTS if name == "earnings_events" else _DAILY) for name in tables}
    commit_tables(conn, clock, {name: [] for name in tables}, contracts, store=store)
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()


def _advance_shadow_head(conn, clock, store, prior_head):
    """Simulate "refresh"'s own worker committing new rows AFTER "refresh"
    was already marked succeeded against ``prior_head``.

    ``commit_tables`` always assumes an empty catalog
    (``expected_head_snapshot_id=None``/``expected_head_generation=0``), and
    a real advance would need the PRIOR snapshot's own earnings_events
    manifest carried over unchanged alongside a new daily_market one --
    machinery this test has no need to re-prove (the data layer's own
    commit path is exhaustively covered elsewhere). Instead: commit a
    second, fully valid, resolvable snapshot with BOTH tables under a
    DIFFERENT scope ("smoke", so ``_check_head_expectation`` never sees
    it), then point "shadow"'s own head row at it directly -- the exact
    shape ``submit_computed_moves_refresh_if_ready`` reads
    (``SELECT snapshot_id, generation FROM data_snapshot_heads``), with a
    generation strictly greater than ``prior_head``'s."""
    other = commit_tables(conn, clock, {"earnings_events": [], "daily_market": []},
                          {"earnings_events": _EVENTS, "daily_market": _DAILY},
                          scope="smoke", receipt_id="advance-r1", attempt_id="advance-att-1",
                          fence=2, store=store)
    generation = prior_head["generation"] + 1
    conn.execute("UPDATE data_snapshot_heads SET snapshot_id = ?, generation = ? "
                "WHERE scope = 'shadow'", (other.snapshot_id, generation))
    conn.commit()
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()


def _mark_refresh_succeeded(conn, clock, store, tmp_path, *, session, head, scope_hash="scopehash"):
    """A minimal, REAL ``incremental_refresh`` job row -- built with the same
    ``incremental_data.plan_refresh``/``refresh_job_spec`` pair
    ``_build_native_refresh_plan``/``_native_refresh_request`` use, so
    ``spec_json``/policy checks are genuine -- submitted under exactly the
    idempotency-key SHAPE ``build_legacy_job_requests`` gives a native
    "refresh" job (``"nightly:<session>:<scope_hash>:refresh"``), then its
    ``state`` set directly to ``succeeded``: the only way to put a row at a
    specific logical time, mirroring
    ``test_v2_ops_engineering_history.py``'s own ``_insert_publication_job``.
    """
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])
    unit = incremental_data.RefreshUnit(
        request_id="daily_market:FAKE", table_name="daily_market",
        partition_key="FAKE", expected_keys=("FAKE",))
    plan = incremental_data.plan_refresh(
        snapshot, (unit,),
        cached_outcomes={unit.request_id: AcquisitionOutcome(
            request_id=unit.request_id, kind="complete", requested_keys=unit.expected_keys,
            returned_keys=unit.expected_keys, receipt_ref="sha256:" + "0" * 64)},
        provider_account=None, expected_head_generation=head["generation"])
    job = incremental_data.refresh_job_spec(
        plan, implementation_ref="test-impl", environment_ref="test-env",
        output_namespace="shadow", catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path))
    key = f"nightly:{session}:{scope_hash}:refresh"
    submit(conn, registry(), _POLICY,
          SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=job),
          clock=clock)
    job_id = job_id_for("shadow", key)
    conn.execute("UPDATE jobs SET state = 'succeeded', updated_at = ? WHERE job_id = ?",
                (clock.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"), job_id))
    conn.commit()
    return job_id


# --------------------------------------------------------------------------
# _stage_sequence / _NATIVE_ACTION_STAGES: computed_moves_refresh builds no
# kind, in any mode, any more
# --------------------------------------------------------------------------


def test_native_action_stages_maps_only_refresh():
    assert _NATIVE_ACTION_STAGES == {"refresh": NATIVE_REFRESH_ACTION}


def test_stage_sequence_native_mode_prepends_only_refresh():
    plan = build_nightly_plan(ROOT, "2026-09-18")
    stages = _stage_sequence(plan, False, None, "native")
    assert stages[0] == "refresh"
    assert COMPUTED_MOVES_REFRESH_ACTION not in stages


def test_stage_sequence_legacy_prerequisite_walk_filters_computed_moves_refresh():
    """computed_moves_refresh is a real GRAPH node, so a prerequisite-inclusive
    plan["order"] names it -- but no mode has a kind to build for it any
    more, so _stage_sequence must filter it out unconditionally. "refresh"
    itself stays in a legacy walk (the pre-existing behaviour, unchanged)."""
    plan = build_nightly_plan(ROOT, "2026-09-18")
    assert COMPUTED_MOVES_REFRESH_ACTION in plan["order"]
    stages = _stage_sequence(plan, True, None, "legacy")
    assert COMPUTED_MOVES_REFRESH_ACTION not in stages
    assert "refresh" in stages


@pytest.mark.parametrize("refresh_mode", ["legacy", "native"])
def test_build_legacy_job_requests_never_submits_computed_moves_refresh(refresh_mode, tmp_path):
    """True in EITHER mode now -- the first cut of Part 4 only proved this
    for legacy mode, because native mode used to submit it right here."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, clock, store)
    plan = build_nightly_plan(ROOT, "2026-09-18")
    requests = build_legacy_job_requests(
        plan, tickers=("FAKE",), year_start=2025, year_end=2026, refresh_mode=refresh_mode,
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
        conn=conn, store=store, clock=clock)
    kinds = [request.job.kind for request in requests]
    assert COMPUTED_MOVES_REFRESH_ACTION not in kinds
    assert "legacy_computed_moves_refresh" not in kinds


# --------------------------------------------------------------------------
# _computed_moves_refresh_key / _session_from_refresh_key
# --------------------------------------------------------------------------


def test_computed_moves_refresh_key_is_session_only_no_scope_hash():
    assert _computed_moves_refresh_key("2026-09-18") == "nightly:2026-09-18:computed_moves_refresh"


@pytest.mark.parametrize("key,expected", [
    ("nightly:2026-09-18:abc123:refresh", "2026-09-18"),
    ("nightly:2026-09-18:abc123:computed_moves_refresh", None),
    ("nightly:2026-09-18:refresh", None),
    ("not-a-nightly-key", None),
])
def test_session_from_refresh_key(key, expected):
    assert _session_from_refresh_key(key) == expected


# --------------------------------------------------------------------------
# _build_native_computed_moves_plan: as_of directly, graceful degradation
# --------------------------------------------------------------------------


def test_no_open_catalog_returns_none():
    assert _build_native_computed_moves_plan(
        "2026-09-18", catalog_path=None, objects_root=None,
        conn=None, store=None, clock=None) is None


def test_no_shadow_head_returns_none(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    assert _build_native_computed_moves_plan(
        "2026-09-18", catalog_path=None, objects_root=None,
        conn=conn, store=store, clock=clock) is None


def test_head_missing_daily_market_returns_none(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, clock, store, tables=("earnings_events",))
    assert _build_native_computed_moves_plan(
        "2026-09-18", catalog_path=None, objects_root=None,
        conn=conn, store=store, clock=clock) is None


def test_empty_target_list_returns_none(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, clock, store)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: ([], {}))
    assert _build_native_computed_moves_plan(
        "2026-09-18", catalog_path=None, objects_root=None,
        conn=conn, store=store, clock=clock) is None


def test_real_targets_build_a_refresh_plan_with_matching_expected_ids(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAPL", "MSFT"], {}))
    built = _build_native_computed_moves_plan(
        "2026-09-18", catalog_path=None, objects_root=None,
        conn=conn, store=store, clock=clock)
    assert built is not None
    refresh_plan, expected_ids = built
    assert expected_ids == ("AAPL", "MSFT")
    assert refresh_plan.parent_snapshot_id == head["snapshot_id"]
    assert refresh_plan.provider_account == NATIVE_COMPUTED_MOVES_ACCOUNT


def test_fully_cached_plan_omits_the_provider_budget_ref(tmp_path):
    """#57 point 1, verified against the current code: plan_refresh already
    nulls provider_account out whenever provider_calls == 0, and
    calendar_moves_job_spec propagates that None straight through to
    JobSpec.provider_budget_ref -- so a fully-cached plan is NOT rejected by
    _refresh_budget_problems's "provider_calls == 0 but provider_budget_ref
    is set" check. This is the proving test decision 2 asked for; no
    production fix was needed for this specific point."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])
    units = computed_moves_store.computed_moves_units(["AAPL"], as_of="2026-09-18")
    cached = {unit.request_id: AcquisitionOutcome(
        request_id=unit.request_id, kind="complete", requested_keys=unit.expected_keys,
        returned_keys=unit.expected_keys, receipt_ref="sha256:" + "0" * 64) for unit in units}
    plan = incremental_data.plan_refresh(
        snapshot, units, cached_outcomes=cached, provider_account=NATIVE_COMPUTED_MOVES_ACCOUNT,
        expected_head_generation=head["generation"])
    assert plan.provider_calls == 0
    assert plan.provider_account is None

    parameters = CalendarMovesParameters(expected_ids=("AAPL",), as_of="2026-09-18")
    job = calendar_moves_job_spec(
        COMPUTED_MOVES_REFRESH_ACTION, plan, parameters, implementation_ref="test-impl",
        environment_ref="test-env", output_namespace="shadow",
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path))
    assert job.provider_budget_ref is None

    request = SubmitRequest(namespace="shadow", idempotency_key=_computed_moves_refresh_key(
        "2026-09-18"), principal="operator", job=job)
    receipt = submit(conn, registry(), _POLICY, request, clock=clock)
    assert receipt.job_id == job_id_for("shadow", _computed_moves_refresh_key("2026-09-18"))


# --------------------------------------------------------------------------
# submit_computed_moves_refresh_if_ready: the only submitter
# --------------------------------------------------------------------------


def test_no_conn_returns_none():
    assert submit_computed_moves_refresh_if_ready(
        None, registry(), _POLICY, None, catalog_path=None, objects_root=None,
        code_source=ROOT, clock=SystemClock()) is None


def test_no_succeeded_refresh_yet_returns_none(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, clock, store)
    assert submit_computed_moves_refresh_if_ready(
        conn, registry(), _POLICY, store, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path), code_source=ROOT, clock=clock) is None


def test_submits_after_refresh_succeeds_and_is_idempotent_on_a_same_session_rerun(
        tmp_path, monkeypatch):
    """Decision 2's "rerun in the same session" test: the SECOND call is a
    pure skip -- the existence check runs before any rebuild, so it can
    never reach submission a second time, let alone hit
    IDEMPOTENCY_CONFLICT."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAPL"], {}))

    receipt = submit_computed_moves_refresh_if_ready(
        conn, registry(), _POLICY, store, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path), code_source=ROOT, clock=clock)
    assert receipt is not None
    assert receipt.job_id == job_id_for("shadow", _computed_moves_refresh_key("2026-09-18"))
    row = conn.execute("SELECT kind FROM jobs WHERE job_id = ?", (receipt.job_id,)).fetchone()
    assert row["kind"] == COMPUTED_MOVES_REFRESH_ACTION

    again = submit_computed_moves_refresh_if_ready(
        conn, registry(), _POLICY, store, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path), code_source=ROOT, clock=clock)
    assert again is None
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE kind = ?", (COMPUTED_MOVES_REFRESH_ACTION,)
    ).fetchone()["n"]
    assert count == 1


def test_service_reconcile_swallows_a_planted_builder_exception_and_refresh_stays_succeeded(
        tmp_path, monkeypatch):
    """Decision 2's "planted builder exception with refresh still
    committing" test: a broken computed-moves build must never crash
    tick() and must never touch the already-succeeded "refresh" row."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    refresh_job_id = _mark_refresh_succeeded(
        conn, clock, store, tmp_path, session="2026-09-18", head=head)

    def _boom(*a, **k):
        raise ValueError("planted computed-moves builder failure")

    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot", _boom)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    service._reconcile_computed_moves_refresh()  # must not raise
    refresh_row = conn.execute("SELECT state FROM jobs WHERE job_id = ?",
                               (refresh_job_id,)).fetchone()
    assert refresh_row["state"] == "succeeded"
    assert conn.execute(
        "SELECT 1 FROM jobs WHERE job_id = ?",
        (job_id_for("shadow", _computed_moves_refresh_key("2026-09-18")),)).fetchone() is None


def test_resolves_the_head_refresh_just_committed_not_a_stale_one(tmp_path, monkeypatch):
    """Decision 2's "refresh committing first with computed_moves then
    succeeding on the new head" test: commit an initial head, mark a
    "refresh" job succeeded against it, THEN advance the head again
    (simulating "refresh"'s own worker attempt committing new daily_market
    rows) -- the submitted computed_moves_refresh job must pin the ADVANCED
    head, never the one that existed when "refresh" was first marked
    succeeded."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    initial_head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=initial_head)
    advanced_head = _advance_shadow_head(conn, clock, store, initial_head)
    assert advanced_head["generation"] > initial_head["generation"]
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAPL"], {}))

    receipt = submit_computed_moves_refresh_if_ready(
        conn, registry(), _POLICY, store, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path), code_source=ROOT, clock=clock)
    assert receipt is not None
    row = conn.execute("SELECT spec_json FROM jobs WHERE job_id = ?", (receipt.job_id,)).fetchone()
    params = json.loads(row["spec_json"])["parameters"]
    assert params["parent_snapshot_id"] == advanced_head["snapshot_id"]
    assert params["expected_head_generation"] == advanced_head["generation"]


def test_identity_picks_max_session_not_latest_updated_row(tmp_path):
    """Opus re-gate finding (non-blocking, my call): ``_computed_moves_identity``
    must pick the session by MAX(session), never by the latest-UPDATED "refresh"
    row. Mark the NEWER session succeeded first, then touch an OLDER session's
    "refresh" row again later (a backfill/re-verify) -- ``updated_at DESC``
    would then wrongly select the older session."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head,
                            scope_hash="newer")
    clock.advance(120)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-10", head=head,
                            scope_hash="older")

    identity = _computed_moves_identity(conn)

    assert identity is not None
    assert identity[0] == "2026-09-18"


def test_no_rescan_on_repeated_ticks_for_the_same_key(tmp_path, monkeypatch):
    """Opus re-gate blocking finding: the expensive pandas scan
    (``target_tickers_from_snapshot``) must be memoized per (session, head)
    at the ``Service`` layer -- repeated ticks against the same identity must
    not re-scan while a build attempt is backing off."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head)
    calls = []

    def _counting_scan(*a, **k):
        calls.append(1)
        return ([], {})

    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot", _counting_scan)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    for _ in range(5):
        service._reconcile_computed_moves_refresh()

    assert len(calls) == 1


def test_reset_on_new_head_resets_attempts(tmp_path, monkeypatch):
    """Opus re-gate blocking finding: a new identity (here, a new shadow head)
    must reset the memo's attempt count/backoff, allowing an immediate rescan
    even while the prior identity was still backing off."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head)
    calls = []

    def _counting_scan(*a, **k):
        calls.append(1)
        return ([], {})

    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot", _counting_scan)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    service._reconcile_computed_moves_refresh()
    assert len(calls) == 1
    service._reconcile_computed_moves_refresh()  # same identity, still backing off
    assert len(calls) == 1

    _advance_shadow_head(conn, clock, store, head)  # new head -> new identity

    service._reconcile_computed_moves_refresh()
    assert len(calls) == 2


def test_backoff_after_an_exception_prevents_an_immediate_rescan(tmp_path, monkeypatch):
    """Opus re-gate: the ``except`` branch of ``_reconcile_computed_moves_refresh``
    must schedule a backoff exactly like the "nothing to submit" branch does
    -- a raising scan, not merely an empty-target ``None`` back, must also
    stop the very next tick from rescanning with no clock advance. (Verified
    during development: deleting the ``self._computed_moves_backoff(memo, now)``
    call from the ``except`` branch makes this test fail with 2 calls.)"""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head)
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise ValueError("planted computed-moves scan failure")

    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot", _boom)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)

    service._reconcile_computed_moves_refresh()
    assert len(calls) == 1

    service._reconcile_computed_moves_refresh()  # no clock advance -- must not rescan
    assert len(calls) == 1


def test_backoff_expires_after_its_window_and_retries_exactly_once(tmp_path, monkeypatch):
    """Once the current backoff window has fully elapsed, the very next tick
    retries -- but only once; a second tick with no further clock advance
    must back off again under the NEXT window."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head)
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise ValueError("planted computed-moves scan failure")

    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot", _boom)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)

    service._reconcile_computed_moves_refresh()
    assert len(calls) == 1

    clock.advance(Service._COMPUTED_MOVES_BACKOFF_SECONDS[0])
    service._reconcile_computed_moves_refresh()
    assert len(calls) == 2  # exactly one retry once the first window elapsed

    service._reconcile_computed_moves_refresh()  # same clock -- backing off again
    assert len(calls) == 2


def test_five_failed_attempts_stop_retrying_that_session(tmp_path, monkeypatch):
    """After ``_COMPUTED_MOVES_MAX_ATTEMPTS`` (5) failed attempts against the
    same identity, no further attempt is made for that session even long
    after every backoff window has elapsed."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head)
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise ValueError("planted computed-moves scan failure")

    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot", _boom)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)

    for seconds in Service._COMPUTED_MOVES_BACKOFF_SECONDS:
        service._reconcile_computed_moves_refresh()
        clock.advance(seconds)
    assert len(calls) == Service._COMPUTED_MOVES_MAX_ATTEMPTS == 5

    clock.advance(10 * Service._COMPUTED_MOVES_BACKOFF_SECONDS[-1])
    service._reconcile_computed_moves_refresh()
    assert len(calls) == 5  # capped -- no 6th attempt for this identity


def test_a_new_session_resets_the_attempt_count_even_past_the_cap(tmp_path, monkeypatch):
    """A new session (not merely a new head, already covered by
    ``test_reset_on_new_head_resets_attempts``) is also a new identity, and
    must reset the memo -- even once the PRIOR session had already spent
    every attempt."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head)
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise ValueError("planted computed-moves scan failure")

    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot", _boom)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)

    for seconds in Service._COMPUTED_MOVES_BACKOFF_SECONDS:
        service._reconcile_computed_moves_refresh()
        clock.advance(seconds)
    assert len(calls) == Service._COMPUTED_MOVES_MAX_ATTEMPTS == 5

    service._reconcile_computed_moves_refresh()  # still capped for "2026-09-18"
    assert len(calls) == 5

    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-19", head=head,
                            scope_hash="secondsession")
    service._reconcile_computed_moves_refresh()
    assert len(calls) == 6


def test_tick_survives_an_identity_lookup_exception_and_legacy_dispatch_still_runs(tmp_path, monkeypatch):
    """Opus re-gate, thread 3 (comment on supervisor.py:380): the identity
    lookup and the job-exists SELECT now run inside the SAME try block as
    submit_computed_moves_refresh_if_ready, matching
    _reconcile_publication_status -- a raise from either must not escape
    _reconcile_computed_moves_refresh, must not stop tick() from returning
    normally, and must not stop the SAME tick's legacy dispatch
    (claim_next/_launch) from running. Proven with a real "artifact_check"
    job -- the same minimal always-registered kind and real-dispatch
    fixture shape tests/test_v2_ops_worker_typed_failures.py's own
    real-Service tests use (SystemClock + open_catalog, not the FakeClock
    ``catalog()`` helper the rest of this file uses)."""
    root = tmp_path / "svc"
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)

    def _boom(conn):
        raise ValueError("planted identity lookup failure")

    monkeypatch.setattr("engine.v2.ops.nightly._computed_moves_identity", _boom)

    profile = profile_named(TEST_POLICY, "delivery")
    job = submit(conn, registry(), _POLICY, request(
        "dispatch-check", kind="artifact_check", checkpoint_contract_ref="receipt.v1.0",
        parameters={"expected_ids": ["a"], "input_bindings": None},
        implementation_ref=content_hash(worker_source_manifest(ROOT)),
        environment_ref=content_hash(
            environment_identity(profile.thread_count or profile.cpu_count))), clock=clock)

    service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                      code_source=ROOT, store_root=store_root)
    try:
        service.start()
        if service.tick() is not True:  # must still claim+launch despite the sidecar's own raise
            AdmissionWatch(conn, job.job_id).check(final=True)  # RESOURCE WAIT, if that is why
            pytest.fail("tick did not claim/launch the legacy job")

        state = conn.execute("SELECT state FROM jobs WHERE job_id = ?",
                             (job.job_id,)).fetchone()["state"]
        assert state != "queued"  # legacy dispatch happened despite the sidecar's own exception

        assert service._computed_moves_memo is not None
        assert service._computed_moves_memo["attempts"] == 1  # logged and backed off, not silent
    finally:
        service.close()
        conn.close()


def test_reconcile_submits_through_the_real_production_service_construction(tmp_path, monkeypatch):
    """Opus re-gate, confirmed real (issue #63): Service is constructed
    with DEFAULT_POLICY in production -- engine/v2/ops/cli.py:574's
    ``serve`` command and engine/v2/ops/nightly_trigger.py:559 both do
    exactly this, never a NamespacePolicy stand-in. DEFAULT_POLICY is a
    bare ResourcePolicy (.profiles, for claim_next) with no .allows() --
    the method submission.submit's own admission check
    (validate_request) calls. Before the fix, threading self.policy
    straight into submit_computed_moves_refresh_if_ready made every real
    submission attempt raise AttributeError (caught by this method's own
    try/except, so no job was ever actually queued). This test builds
    Service exactly as those two production entrypoints do -- not a fake
    NamespacePolicy -- and proves a real submission goes through."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAPL"], {}))

    service = Service(conn, tmp_path, registry(), DEFAULT_POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    service._reconcile_computed_moves_refresh()

    job_id = job_id_for("shadow", _computed_moves_refresh_key("2026-09-18"))
    row = conn.execute("SELECT state FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    assert row is not None, "submission never reached the catalog -- see issue #63"
    assert row["state"] == "queued"
    assert service._computed_moves_memo is None  # cleared: a job now exists


def test_implementation_ref_uses_the_callers_code_source_not_this_module(tmp_path, monkeypatch):
    """CodeRabbit finding on the Opus re-gate: implementation_ref must
    come from the CALLER's own code_source (Service.code_source in
    production -- the exact root Service._launch validates
    implementation_ref against), never a root nightly.py derives from its
    own file position, which risks disagreeing with Service.code_source
    under a pinned/worktree execution model. Proven by spying on
    worker_source_manifest and asserting it is called with exactly the
    code_source this function was given, never anything self-derived."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    _mark_refresh_succeeded(conn, clock, store, tmp_path, session="2026-09-18", head=head)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAPL"], {}))
    seen = []

    def _spy(root):
        seen.append(root)
        return {"marker": str(root)}

    monkeypatch.setattr("engine.v2.ops.fingerprints.worker_source_manifest", _spy)
    other_root = tmp_path / "other-checkout"

    receipt = submit_computed_moves_refresh_if_ready(
        conn, registry(), _POLICY, store, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path), code_source=other_root, clock=clock)

    assert receipt is not None
    assert seen == [other_root]  # exactly the caller's code_source, called exactly once
