"""P2-C04: watchlist scope vs historical evidence scope, and a subset run's
effect scope carried through every producer (decision_evidence, decision
commit, settlement, export, publication).

The review's proof: "A current one-ticker request can retain older analogs
from other tickers without widening its score population. A real subset
commit followed by export succeeds under its subset scope and advances no
global watermark." Tests here go through the real producers -- never a
pre-seeded scoped watermark -- mirroring the patterns in
``tests/test_v2_ops_nightly_completion.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engine.v2.contracts import JobSpec, LegacyFileRef, SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ledger.decisions import set_authority
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.effects_graph import ledger_export_effect
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import environment_identity, file_hash, worker_source_manifest
from engine.v2.ops.legacy_adapter import _action_decision_replay, _action_score
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan, effect_scope_for
from engine.v2.ops.plans import nightly_plan
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit, submit_graph
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, sample
from tests.test_v2_ops_nightly_completion import (
    _finality_coverage,
    _manifest_ref,
    _publish,
    _run_until_terminal,
    _score_and_finality,
    _succeed_finality_parent,
    _succeed_parent,
)
from tests.test_v2_ops_snapshot_stages import case  # noqa: F401 -- pytest fixture

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
SESSION = "2026-09-12"


# --------------------------------------------------------------------------
# effect scope: context_tickers drives full-vs-subset, not the bare watchlist
# --------------------------------------------------------------------------


def test_build_legacy_job_requests_context_tickers_are_recorded_per_stage():
    """``context_tickers`` is recorded on every stage's parameters (the
    historical evidence universe), independent of the effect scope --
    see ``test_full_run_must_be_declared_explicitly`` below for the scope."""
    plan = build_nightly_plan(str(REPO), SESSION)
    subset = build_legacy_job_requests(plan, tickers=("A",), context_tickers=("A", "B"),
                                       year_start=2025, year_end=2026)
    by_kind_subset = {r.job.kind: r for r in subset}
    assert by_kind_subset["legacy_score"].job.parameters["context_tickers"] == ("A", "B")
    no_context = build_legacy_job_requests(plan, tickers=("A",), year_start=2025, year_end=2026)
    assert {r.job.kind: r for r in no_context}["legacy_score"].job.parameters[
        "context_tickers"] == ("A",)


def test_full_run_must_be_declared_explicitly():
    """P2-5 effect-scope decision: writing the global "shadow" scope
    requires an explicit ``full_universe`` (``--full-run``) declaration.
    A watchlist that merely equals its context, with no such declaration,
    still gets a subset scope -- equality alone is never inferred as a full
    run (this is the exact bug: a small debugging run with no context would
    otherwise silently advance the global watermark)."""
    plan = build_nightly_plan(str(REPO), SESSION)
    common = dict(context_tickers=("A", "B"), year_start=2025, year_end=2026)
    declared_full = build_legacy_job_requests(plan, tickers=("A", "B"), full_universe=("A", "B"),
                                              **common)
    equal_no_declaration = build_legacy_job_requests(plan, tickers=("A", "B"), **common)
    subset = build_legacy_job_requests(plan, tickers=("A",), **common)

    def scope(requests):
        return {r.job.kind: r for r in requests}["legacy_score"].job.parameters["effect_scope"]

    assert scope(declared_full) == "shadow"
    equal_scope = scope(equal_no_declaration)
    subset_scope = scope(subset)
    assert equal_scope.startswith("shadow:") and equal_scope != "shadow"
    assert subset_scope.startswith("shadow:") and subset_scope != "shadow"
    assert equal_scope != subset_scope  # the hash still covers the actual watchlist


def test_build_legacy_job_requests_refuses_watchlist_outside_context():
    plan = build_nightly_plan(str(REPO), SESSION)
    with pytest.raises(OpsError) as err:
        build_legacy_job_requests(plan, tickers=("A", "Z"), context_tickers=("A", "B"),
                                  year_start=2025, year_end=2026)
    assert err.value.problem.code == "INVALID_REQUEST"


def test_build_legacy_job_requests_full_run_refuses_narrower_watchlist():
    """A full run must score its whole context (the plan carries this as
    ``full_universe``, only ever set from a ``--full-run`` plan's own
    ``context_tickers``); refused here too, not only in ``nightly_plan``,
    as defense in depth for a caller that builds requests directly."""
    plan = build_nightly_plan(str(REPO), SESSION)
    with pytest.raises(OpsError) as err:
        build_legacy_job_requests(plan, tickers=("A",), context_tickers=("A", "B"),
                                  full_universe=("A", "B"), year_start=2025, year_end=2026)
    assert err.value.problem.code == "INVALID_REQUEST"


def test_nightly_plan_refuses_watchlist_outside_context():
    with pytest.raises(OpsError) as err:
        nightly_plan(str(REPO), SESSION, manifest_ref="art_x", tickers=("A", "Z"),
                    context_tickers=("A", "B"), expected_population=("A|S1|2026-09-12",))
    assert err.value.problem.code == "INVALID_REQUEST"
    plan = nightly_plan(str(REPO), SESSION, manifest_ref="art_x", tickers=("A",),
                        context_tickers=("A", "B"), expected_population=("A|S1|2026-09-12",))
    assert plan["tickers"] == ["A"] and plan["context_tickers"] == ["A", "B"]
    assert plan["full_run"] is False


def test_nightly_plan_full_run_refuses_a_narrower_watchlist():
    with pytest.raises(OpsError) as err:
        nightly_plan(str(REPO), SESSION, manifest_ref="art_x", tickers=("A",),
                    context_tickers=("A", "B"), expected_population=("A|S1|2026-09-12",),
                    full_run=True)
    assert err.value.problem.code == "INVALID_REQUEST"
    plan = nightly_plan(str(REPO), SESSION, manifest_ref="art_x", tickers=("A", "B"),
                        context_tickers=("A", "B"), expected_population=("A|S1|2026-09-12",),
                        full_run=True)
    assert plan["full_run"] is True


# --------------------------------------------------------------------------
# pin_snapshot_inputs: evidence scope widens without widening direct scope
# --------------------------------------------------------------------------


def test_pin_snapshot_inputs_refuses_a_watchlist_outside_the_context():
    from engine.v2.ops.snapshot_planning import pin_snapshot_inputs

    # The subset check fires before ``conn``/``store`` are ever touched, so a
    # placeholder connection/store is enough to prove the refusal is real.
    with pytest.raises(OpsError) as err:
        pin_snapshot_inputs(None, None, "shadow", tickers=("AAA", "BBB"),
                            year_start=2020, year_end=2021,
                            expected_population=("ZZZ|S1|2020-01-15",), clock=None)
    assert err.value.problem.code == "INVALID_REQUEST"


def test_pin_snapshot_inputs_widens_evidence_scope_not_direct_scope(case):
    from engine.v2.ops.snapshot_planning import pin_snapshot_inputs

    context_tickers = ("AAA", "BBB", "CCC", "DDD", "EEE")
    result = pin_snapshot_inputs(case.conn, case.store, "shadow", tickers=context_tickers,
                                 year_start=2020, year_end=2021,
                                 expected_population=("AAA|S1|2020-01-15",), clock=case.clock)
    ref = artifact(case.conn, case.store, result["materialization_request_ref"])
    request_doc = json.loads(case.store.read_verified(ref))
    assert sorted(request_doc["evidence_scope"]["tickers"]) == sorted(context_tickers)
    assert request_doc["direct_scope"]["tickers"] == ["AAA"]


# --------------------------------------------------------------------------
# scoring/replay: context_tickers loads FeatureContext, tickers scores
# --------------------------------------------------------------------------


def _stub_scoring(monkeypatch, rows):
    import engine.dashboard.nightly as nightly_module
    import engine.features as features_module
    import engine.score as score_module

    calls = {}

    def fake_load(tickers, years):
        calls["context_tickers"] = sorted(tickers)
        return object()

    class _FakeScorer:
        def __init__(self, context):
            self.analog_entry_coverage = 1.0

    def fake_score_calendar(as_of, *, tickers, **kwargs):
        calls["score_tickers"] = sorted(tickers)
        return pd.DataFrame(rows)

    monkeypatch.setattr(features_module.FeatureContext, "load", staticmethod(fake_load))
    monkeypatch.setattr(score_module, "Scorer", _FakeScorer)
    monkeypatch.setattr(score_module, "score_calendar", fake_score_calendar)
    monkeypatch.setattr(nightly_module, "strike_ladder", lambda *a, **k: [])
    return calls


def test_action_score_loads_context_tickers_and_scores_only_the_watchlist(monkeypatch, tmp_path):
    rows = [{"ticker": "AAA", "strategy": "TWIN-P", "event_date": SESSION,
            "strike": 100.0, "expiry": "2026-10-16"}]
    calls = _stub_scoring(monkeypatch, rows)
    (tmp_path / "finality.json").write_text(json.dumps({"date": SESSION, "is_final": True}))

    parameters = {"tickers": ["AAA"], "context_tickers": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                 "year_start": 2024, "year_end": 2026, "session": SESSION, "horizon_days": 35,
                 "alt_strikes": 1, "expected_population": ("AAA|TWIN-P|" + SESSION,)}
    result = _action_score(parameters, tmp_path)
    assert result["hash"]

    # A one-ticker watchlist retains the wider 5-ticker analog/feature
    # context; the actual scored population stays the narrow watchlist.
    assert calls["context_tickers"] == ["AAA", "BBB", "CCC", "DDD", "EEE"]
    assert calls["score_tickers"] == ["AAA"]

    document = json.loads((tmp_path / "score.json").read_text())
    assert document["tickers"] == ["AAA"]
    assert document["context_tickers"] == ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_action_score_context_tickers_defaults_to_the_watchlist(monkeypatch, tmp_path):
    """A caller that predates ``context_tickers`` (or omits it) is unchanged:
    the loaded context is exactly the watchlist, as before this fix."""
    rows = [{"ticker": "AAA", "strategy": "TWIN-P", "event_date": SESSION,
            "strike": 100.0, "expiry": "2026-10-16"}]
    calls = _stub_scoring(monkeypatch, rows)
    (tmp_path / "finality.json").write_text(json.dumps({"date": SESSION, "is_final": True}))

    parameters = {"tickers": ["AAA"], "year_start": 2024, "year_end": 2026, "session": SESSION,
                 "horizon_days": 35, "alt_strikes": 1,
                 "expected_population": ("AAA|TWIN-P|" + SESSION,)}
    _action_score(parameters, tmp_path)
    assert calls["context_tickers"] == ["AAA"]
    assert calls["score_tickers"] == ["AAA"]


def test_action_decision_replay_loads_context_tickers_not_the_watchlist(monkeypatch, tmp_path):
    row_aaa = {"ticker": "AAA", "strategy": "TWIN-P", "event_date": SESSION,
              "as_of": SESSION, "session": "AMC", "fill": 0.5, "strike_offset": None,
              "exp_pnl_model": 0.1}
    (tmp_path / "score.json").write_text(json.dumps({"rows": [row_aaa]}))
    (tmp_path / "finality.json").write_text(json.dumps({"date": SESSION, "is_final": True}))
    calls = _stub_scoring(monkeypatch, [row_aaa])

    result = _action_decision_replay(
        {"session": SESSION, "tickers": ("AAA",),
         "context_tickers": ("AAA", "BBB", "CCC", "DDD", "EEE"),
         "year_start": 2024, "year_end": 2026}, tmp_path)
    assert result["hash"]
    assert calls["context_tickers"] == ["AAA", "BBB", "CCC", "DDD", "EEE"]
    assert calls["score_tickers"] == ["AAA"]


# --------------------------------------------------------------------------
# real producers: decision_evidence -> legacy_decisions -> ledger_export
# --------------------------------------------------------------------------


def _scoped_request(kind, *, key, parameters, input_refs, dependency_job_ids, resource_class,
                    checkpoint_contract_ref):
    profile = profile_named(DEFAULT_POLICY, resource_class)
    job = JobSpec(
        kind=kind, implementation_ref=content_hash(worker_source_manifest(REPO)), spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters=parameters, input_refs=tuple(input_refs),
        dependency_job_ids=tuple(dependency_job_ids), output_namespace="shadow",
        resource_class=resource_class, retry_policy_ref="bounded",
        checkpoint_contract_ref=checkpoint_contract_ref)
    return SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=job)


def _decision_evidence_request(*, key, score_job, finality_job, replay_job, deployment,
                               decision_clock, effect_scope):
    bindings = {"score.json": score_job + "#legacy_score",
                "finality.json": finality_job + "#legacy_finality",
                "replay.json": replay_job + "#legacy_decision_replay",
                "finality_coverage.json": finality_job + "#legacy_finality_coverage"}
    parameters = {"expected_ids": ("decision_evidence",), "session": SESSION,
                 "tickers": (), "year_start": 2024, "year_end": 2026,
                 "deployment": deployment, "decision_clock": decision_clock,
                 "effect_scope": effect_scope, "input_bindings": bindings}
    return _scoped_request("decision_evidence", key=key, parameters=parameters, input_refs=(),
                           dependency_job_ids=(score_job, finality_job, replay_job),
                           resource_class="validation",
                           checkpoint_contract_ref="decision_evidence_pair.v1.0")


def _legacy_decisions_request(*, key, manifest_ref, score_job, finality_job, evidence_job,
                              effect_scope):
    bindings = {"legacy_manifest.json": manifest_ref.artifact_id,
                "score.json": score_job + "#legacy_score",
                "finality.json": finality_job + "#legacy_finality",
                "decision_plan.json": evidence_job + "#decision_plan",
                "decision_evidence.json": evidence_job + "#decision_evidence"}
    parameters = {"expected_ids": ("legacy_decisions",), "session": SESSION,
                 "tickers": (), "year_start": 2024, "year_end": 2026,
                 "effect_scope": effect_scope, "input_bindings": bindings}
    return _scoped_request("legacy_decisions", key=key, parameters=parameters,
                           input_refs=(manifest_ref.artifact_id,),
                           dependency_job_ids=(evidence_job, score_job, finality_job),
                           resource_class="validation",
                           checkpoint_contract_ref="legacy_action.v1.0")


def _run_ledger_export(conn, store, setup, clock, *, key, ops_root, effect_scope):
    """Submit, claim and run ``ledger_export`` for real: the same
    ``ledger_export_effect`` call ``supervisor.Service._coordinator_effect``
    makes, driven directly against a real claim -- the established pattern
    ``tests/test_v2_ops_effects_graph.py`` and
    ``tests/test_v2_ops_resolved_session.py`` use for these coordinator-only
    "pure worker" kinds (no subprocess is spun up for the trivial receipt
    writer itself)."""
    job = JobSpec(kind="ledger_export", implementation_ref="test-impl", spec_hash=None,
                 environment_ref="test-env",
                 parameters={"expected_ids": ("ledger_export",), "session": SESSION,
                             "effect_scope": effect_scope, "input_bindings": {}},
                 output_namespace="shadow", resource_class="delivery", retry_policy_ref="bounded",
                 checkpoint_contract_ref="effect_receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace="shadow", idempotency_key=key,
          principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=setup,
                       clock=clock, registry=registry())
    assert claim is not None, "ledger_export job did not claim"
    effect, extra_refs = ledger_export_effect(conn, store, claim, ops_root, REPO, clock=clock)

    def commit(inner_conn):
        for name, ref in extra_refs:
            register_artifact(inner_conn, ref, claim.attempt_id, clock)
            inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                               (claim.attempt_id, name, ref.artifact_id))
        effect(inner_conn)

    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=commit)
    return job_id_for("shadow", key)


def _run_chain(root, store_root, *, effect_scope, tag):
    """Seed synthetic finality/score/replay parents, then run
    decision_evidence -> legacy_decisions for real through the Service
    (real subprocess workers) and ledger_export for real through its
    coordinator function, all under ``effect_scope``.
    """
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    store = ArtifactStore(root)
    setup_epoch = begin_epoch(conn, clock=clock, boot_id="setup-" + tag, pid=1)
    setup = Supervisor(setup_epoch, "setup-" + tag)

    fixture = store_root / "unused.txt"
    if not fixture.exists():
        fixture.write_bytes(b"a legacy read-set member decisions never opens")
    manifest_ref = _manifest_ref(store, conn, clock, (LegacyFileRef(
        path="unused.txt", content_hash=file_hash(fixture), byte_size=fixture.stat().st_size),))

    score, finality = _score_and_finality()
    key = "FAKE|TWIN-P|" + SESSION
    score_ref = _publish(store, conn, clock, {"rows": [score]}, "legacy_action.v1.0")
    finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
    coverage_ref = _publish(store, conn, clock, _finality_coverage(), "finality_coverage.v1.0")
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                 "population": [key], "source_rows": [score], "replayed_rows": [score],
                 "source_rows_hash": content_hash([score]),
                 "replayed_rows_hash": content_hash([score]), "findings": []}
    replay_ref = _publish(store, conn, clock, replay_doc, "legacy_action.v1.0")

    score_job = _succeed_parent(conn, clock, setup, key=tag + "-score",
                                output_name="legacy_score", ref=score_ref)
    finality_job = _succeed_finality_parent(conn, clock, setup, key=tag + "-finality",
                                            finality_ref=finality_ref, coverage_ref=coverage_ref)
    replay_job = _succeed_parent(conn, clock, setup, key=tag + "-replay",
                                 output_name="legacy_decision_replay", ref=replay_ref)

    current_owner = conn.execute(
        "SELECT owner FROM decision_authority WHERE singleton=1").fetchone()
    if current_owner is None or current_owner[0] != "catalog":
        with transaction(conn):
            set_authority(conn, current_owner[0] if current_owner else None, "catalog",
                          SESSION + "T20:00:00.000000Z")

    deployment, decision_clock = "shadow:test-impl", SESSION + "T21:00:00+00:00"
    evidence_request = _decision_evidence_request(
        key=tag + "-evidence", score_job=score_job, finality_job=finality_job,
        replay_job=replay_job, deployment=deployment, decision_clock=decision_clock,
        effect_scope=effect_scope)
    evidence_job_id = job_id_for("shadow", tag + "-evidence")
    decisions_request = _legacy_decisions_request(
        key=tag + "-decisions", manifest_ref=manifest_ref, score_job=score_job,
        finality_job=finality_job, evidence_job=evidence_job_id, effect_scope=effect_scope)
    decisions_job_id = job_id_for("shadow", tag + "-decisions")

    receipts = submit_graph(conn, registry(), POLICY,
                            [evidence_request, decisions_request], clock=clock)
    assert len(receipts) == 2

    service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=REPO,
                      store_root=store_root)
    try:
        service.start()
        states = {
            "decision_evidence": _run_until_terminal(service, conn, evidence_job_id),
            "legacy_decisions": _run_until_terminal(service, conn, decisions_job_id),
        }
    finally:
        service.close()
    for job_id, name in ((evidence_job_id, "decision_evidence"),
                         (decisions_job_id, "legacy_decisions")):
        row = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        assert states[name] == "succeeded", (name, row["failure_json"] if row else None)

    _run_ledger_export(conn, store, setup, clock, key=tag + "-export", ops_root=root,
                      effect_scope=effect_scope)
    return conn


def test_subset_run_commits_and_exports_under_its_own_scope_through_real_producers(tmp_path):
    """P2-C04 proof: decision_evidence -> legacy_decisions -> ledger_export
    for a 2-of-5-ticker plan commit and export under their OWN
    "shadow:<hash>" scope, and write NOTHING under the global "shadow" scope
    -- never a pre-seeded scoped watermark."""
    root = tmp_path / "subset"
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    watchlist, context = ("AAA", "BBB"), ("AAA", "BBB", "CCC", "DDD", "EEE")
    effect_scope = effect_scope_for(watchlist, context)
    assert effect_scope.startswith("shadow:") and effect_scope != "shadow"

    conn = _run_chain(root, store_root, effect_scope=effect_scope, tag="sub")
    try:
        decisions_wm = conn.execute(
            "SELECT scope, occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' "
            "AND stage='decisions'").fetchall()
        export_wm = conn.execute(
            "SELECT scope, occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' "
            "AND stage='export'").fetchall()
        assert [tuple(row)[:2] for row in decisions_wm] == [(effect_scope, SESSION)]
        assert [tuple(row)[:2] for row in export_wm] == [(effect_scope, SESSION)]

        # Zero rows -- decisions, export, release_intent -- under the global scope.
        assert conn.execute(
            "SELECT COUNT(*) FROM watermarks WHERE scope='shadow'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1

        # Release-intent and export outbox keys are the subset scope's own
        # release_key (the watermark's own receipt_ref), never a global one.
        release_key = decisions_wm[0]["receipt_ref"]
        outbox_kinds = sorted(r[0] for r in conn.execute(
            "SELECT kind FROM outbox WHERE logical_key=?", (release_key,)))
        assert outbox_kinds == ["export", "release_intent"]
        global_release_key = content_hash(["shadow", SESSION, content_hash(
            {"candidate": None, "plan": None, "evidence": None})])
        assert conn.execute("SELECT COUNT(*) FROM outbox WHERE logical_key=?",
                            (global_release_key,)).fetchone()[0] == 0
    finally:
        conn.close()


def test_mixed_subset_and_full_runs_keep_independent_watermarks(tmp_path):
    """A subset run followed by a full-universe run: each scope's decisions/
    export watermarks stay independent -- neither overwrites the other."""
    root = tmp_path / "mixed"
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    watchlist, context = ("AAA", "BBB"), ("AAA", "BBB", "CCC", "DDD", "EEE")
    subset_scope = effect_scope_for(watchlist, context)
    full_scope = effect_scope_for(context, context)
    assert full_scope == "shadow" and subset_scope != "shadow"

    conn = _run_chain(root, store_root, effect_scope=subset_scope, tag="mix-sub")
    conn.close()
    conn = _run_chain(root, store_root, effect_scope=full_scope, tag="mix-full")
    try:
        rows = conn.execute(
            "SELECT scope, occurrence FROM watermarks WHERE pipeline='nightly' "
            "AND stage='decisions' ORDER BY scope").fetchall()
        assert [tuple(row) for row in rows] == [
            ("shadow", SESSION), (subset_scope, SESSION)]
        # The two runs commit the SAME synthetic candidate (identical
        # decision_id/payload), so the decisions ledger dedupes it to one
        # row (engine.v2.ledger.decisions.insert's own idempotent-content
        # rule) -- the two scopes' WATERMARKS are what must stay
        # independent, which the assertion above already proves.
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
        export_rows = conn.execute(
            "SELECT scope, occurrence FROM watermarks WHERE pipeline='nightly' "
            "AND stage='export' ORDER BY scope").fetchall()
        assert [tuple(row) for row in export_rows] == [
            ("shadow", SESSION), (subset_scope, SESSION)]
    finally:
        conn.close()
