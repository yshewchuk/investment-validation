"""P2-C03: propagate the finality-resolved session through the supervised
nightly (review closeout).

v1 (``engine/dashboard/nightly.py:1223-1247``) walks a requested ``as_of``
back to the newest FINAL session and uses the resolved date for scoring,
decisions, settlement and render, while preserving both the requested and
resolved dates in the render clock. This file proves the v2 supervised
graph does the same: score/replay/settlement/decisions/render all act on
the RESOLVED session (never the requested one held in job parameters), the
strict-date refusal remains explicit, and export/publication watermarks and
metadata carry the resolved session too.

Tier 0: no private data, no real scoring/legacy rebuild. Pure helpers and
in-process legacy actions are exercised with synthetic docs and
monkeypatched ``Scorer``/``FeatureContext.load``/``score_calendar``/
``score_outcomes``; the decision-commit chain uses the real ``Service``
and ``TEST_POLICY``, the same pattern ``tests/test_v2_ops_nightly_completion.py``
and ``tests/test_v2_ops_effects_graph.py`` use.
"""
from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path

import pandas as pd
import pytest

from engine.v2.contracts import JobSpec, LegacyFileRef, SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ledger.decisions import set_authority
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.decision_evidence import derive
from engine.v2.ops.effects_graph import ledger_export_effect, publication_effect
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import environment_identity, file_hash, worker_source_manifest
from engine.v2.ops import legacy_adapter
from engine.v2.ops.legacy_adapter import (
    _action_decision_replay,
    _action_decisions,
    _action_finality,
    _action_render,
    _action_score,
    _action_settlement,
)
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.session_resolution import resolve_effective_session, walk_back_flag
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, sample
from tests.test_v2_ops_effects_graph import (
    FAKE_STORE_ROOT,
    _open as _effects_open,
    _params as _effects_params,
    _publication_setup,
    _seed_decisions,
    _submit_and_claim as _effects_submit_and_claim,
)
from tests.test_v2_ops_nightly_completion import (
    _manifest_ref,
    _publish,
    _run_until_terminal,
    _succeed_evidence_parent,
    _succeed_finality_parent,
    _succeed_parent,
)

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
REQUESTED = "2026-09-14"
RESOLVED = "2026-09-11"


def _finality(date=RESOLVED, is_final=True, detail="chains 70% < 80%"):
    return {"date": date, "is_final": is_final, "market_wide": True,
            "daily_share": 1.0, "chain_share": 1.0, "covered": 1, "detail": detail}


# --------------------------------------------------------------------------
# pure: resolve_effective_session / walk_back_flag
# --------------------------------------------------------------------------


def test_resolve_effective_session_walks_back_and_passes_through_unchanged():
    assert resolve_effective_session(_finality(), REQUESTED) == RESOLVED
    assert resolve_effective_session(_finality(date=REQUESTED), REQUESTED) == REQUESTED


def test_resolve_effective_session_refuses_a_resolution_after_the_request():
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        resolve_effective_session(_finality(date="2026-09-15"), REQUESTED)


def test_resolve_effective_session_refuses_when_not_final():
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        resolve_effective_session(_finality(is_final=False), REQUESTED)


def test_walk_back_flag_matches_v1_kind_and_detail_and_is_none_without_one():
    finality = _finality()
    flag = walk_back_flag(REQUESTED, RESOLVED, finality)
    assert flag == {"kind": "as_of_resolved",
                    "detail": f"requested {REQUESTED} resolved to final {RESOLVED}: "
                              f"{finality['detail']}"}
    assert walk_back_flag(REQUESTED, REQUESTED, finality) is None


# --------------------------------------------------------------------------
# strict refusal: no final session -> SOURCE_NOT_FINAL, never a fallback
# --------------------------------------------------------------------------


def test_action_finality_raises_source_not_final_when_no_session_qualifies(monkeypatch, tmp_path):
    import engine.calendar as calendar_module
    import engine.data.finality as finality_module

    def boom(requested, tickers, *, calendar, max_sessions=15):
        raise RuntimeError(f"no final session at or before {requested} "
                           f"within {max_sessions} trading sessions")

    monkeypatch.setattr(finality_module, "resolve_final_session", boom)
    monkeypatch.setattr(calendar_module, "trading_calendar", lambda extend_days=400: object())

    with pytest.raises(OpsError) as err:
        _action_finality({"session": REQUESTED, "tickers": ("FAKE",)}, tmp_path)
    assert err.value.problem.code == "SOURCE_NOT_FINAL"
    # nothing was written -- a strict refusal never leaves a partial finality.json
    assert not (tmp_path / "finality.json").is_file()


# --------------------------------------------------------------------------
# last read-set gap fix (2026-09-15): legacy_finality's content cross-check
# against this run's own materialization (task decision (b))
# --------------------------------------------------------------------------


class _MarketWideEntry:
    """``engine.data.fetch.CachedEntry``'s two fields ``_market_wide_complete``
    reads, matching ``tests/test_finality.py``'s own fixture shape."""

    def __init__(self, endpoint, day):
        self.endpoint = endpoint
        self.params = {"tradeDate": day}
        self.meta = {"status": 200}


def _write_curated_parquet(root, table, column, year, ticker_dates, extra_columns=None):
    frame = pd.DataFrame({"ticker": [t for t, _ in ticker_dates],
                          column: [d for _, d in ticker_dates],
                          **(extra_columns or {})})
    directory = root / "data" / "curated" / table / f"year={year}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "part-0000.parquet")


def _patch_resolved_session(monkeypatch, date, tickers):
    """Pin the BARRIER's own resolved session (``resolve_final_session``) to
    a fixed, already-final result, so only the cross-check's own logic is
    under test -- same monkeypatch shape
    ``test_action_finality_raises_source_not_final_when_no_session_qualifies``
    already uses."""
    import engine.calendar as calendar_module
    import engine.data.finality as finality_module
    from engine.data.finality import SessionFinality

    resolved = SessionFinality(date=date, market_wide=True, daily_share=1.0, chain_share=1.0,
                               is_final=True, detail="final", tickers=len(tickers),
                               covered=len(tickers))
    monkeypatch.setattr(finality_module, "resolve_final_session",
                        lambda requested, tickers, *, calendar, max_sessions=15: resolved)
    monkeypatch.setattr(finality_module, "covered_tickers", lambda date, tickers: list(tickers))
    monkeypatch.setattr(calendar_module, "trading_calendar", lambda extend_days=400: object())


def _install_market_wide(monkeypatch, day):
    import engine.data.finality as finality_module

    monkeypatch.setattr(finality_module.fetch, "iter_cached",
                        lambda source: [_MarketWideEntry("hist/summaries", day),
                                        _MarketWideEntry("hist/cores", day)])


def test_finality_cross_check_passes_when_materialization_independently_confirms(
        monkeypatch, tmp_path):
    date = "2026-09-11"
    tickers = ("AAA", "BBB")
    _patch_resolved_session(monkeypatch, date, tickers)
    _install_market_wide(monkeypatch, date)

    root = tmp_path / "materialization"
    _write_curated_parquet(root, "daily_market", "date", 2026, [("AAA", date), ("BBB", date)])
    _write_curated_parquet(root, "option_chains", "obs_date", 2026, [("AAA", date), ("BBB", date)])

    staging = tmp_path / "staging"
    staging.mkdir()
    _action_finality({"session": REQUESTED, "tickers": tickers}, staging,
                     cross_check={"materialization_root": str(root)})
    assert (staging / "finality.json").is_file()
    assert json.loads((staging / "finality.json").read_text())["date"] == date


def test_finality_cross_check_refuses_when_materialization_lacks_the_resolved_session(
        monkeypatch, tmp_path):
    """The real risk the task names: the live tree grows daily, so the
    barrier can resolve a session (``date``) the pinned materialization has
    no rows for -- here it only carries an older session, ``2026-09-08``."""
    date = "2026-09-11"
    tickers = ("AAA", "BBB")
    _patch_resolved_session(monkeypatch, date, tickers)
    _install_market_wide(monkeypatch, date)

    root = tmp_path / "materialization"
    _write_curated_parquet(root, "daily_market", "date", 2026,
                           [("AAA", "2026-09-08"), ("BBB", "2026-09-08")])
    _write_curated_parquet(root, "option_chains", "obs_date", 2026,
                           [("AAA", "2026-09-08"), ("BBB", "2026-09-08")])

    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(OpsError) as err:
        _action_finality({"session": REQUESTED, "tickers": tickers}, staging,
                         cross_check={"materialization_root": str(root)})
    assert err.value.problem.code == "SOURCE_NOT_FINAL"
    assert err.value.problem.details["reason"] == "finality_snapshot_drift"
    assert not (staging / "finality.json").is_file()


def test_finality_cross_check_is_skipped_without_a_materialization_root(monkeypatch, tmp_path):
    """A plain legacy-mode nightly's ``cross_check`` is always ``None`` --
    unaffected, exactly as before this fix: no materialization root is ever
    consulted and nothing new can refuse it."""
    date = "2026-09-11"
    tickers = ("AAA",)
    _patch_resolved_session(monkeypatch, date, tickers)

    staging = tmp_path / "staging"
    staging.mkdir()
    _action_finality({"session": REQUESTED, "tickers": tickers}, staging)
    assert (staging / "finality.json").is_file()


def test_finality_cross_check_ignores_a_projection_only_difference(monkeypatch, tmp_path):
    """Task fact: the 3 differing real-world ``option_chains`` parquet files
    are a projection artifact (extra columns the materialization drops), not
    drift. A column finality's own projection never reads (here, an extra
    ``close`` column on ``daily_market``) must never trip the check -- only
    ``ticker``+``date``/``obs_date`` are compared."""
    date = "2026-09-11"
    tickers = ("AAA",)
    _patch_resolved_session(monkeypatch, date, tickers)
    _install_market_wide(monkeypatch, date)

    root = tmp_path / "materialization"
    _write_curated_parquet(root, "daily_market", "date", 2026, [("AAA", date)],
                           extra_columns={"close": [123.45]})
    _write_curated_parquet(root, "option_chains", "obs_date", 2026, [("AAA", date)],
                           extra_columns={"bid": [1.1], "ask": [1.3]})

    staging = tmp_path / "staging"
    staging.mkdir()
    _action_finality({"session": REQUESTED, "tickers": tickers}, staging,
                     cross_check={"materialization_root": str(root)})
    assert (staging / "finality.json").is_file()


# --------------------------------------------------------------------------
# derive(): resolved session, requested_session, and the same refusals
# --------------------------------------------------------------------------


def _synthetic_chain(*, requested=REQUESTED, resolved=RESOLVED):
    score = {"ticker": "FAKE", "event_id": "event-1", "event_date": resolved,
             "as_of": resolved, "entry_date": resolved, "evidence_cutoff": resolved,
             "strategy": "TWIN-P", "strike": 100.0, "expiry": "2026-10-16",
             "session": "AMC", "snapshot_hash": "sha256:" + "a" * 64, "strike_offset": None}
    score_doc = {"rows": [score]}
    finality = _finality(date=resolved)
    key = "FAKE|TWIN-P|" + resolved
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": resolved,
                  "requested_session": requested, "population": [key],
                  "source_rows": [score], "replayed_rows": [score],
                  "source_rows_hash": content_hash([score]),
                  "replayed_rows_hash": content_hash([score]), "findings": []}
    coverage_doc = {"schema_version": "finality_coverage.v1.0", "date": resolved,
                    "covered_tickers": ["FAKE"]}
    return score_doc, finality, replay_doc, coverage_doc


def test_derive_resolves_the_session_and_keeps_the_requested_one():
    from engine.v2.foundation import artifact_reference

    score_doc, finality, replay_doc, coverage_doc = _synthetic_chain()
    score_ref = artifact_reference(b"synthetic-score", "legacy_action.v1.0")
    finality_ref = artifact_reference(b"synthetic-finality", "legacy_action.v1.0")
    plan_bytes, evidence_bytes = derive(
        score_doc, score_ref, finality, finality_ref, replay_doc, coverage_doc,
        requested_session=REQUESTED, deployment="shadow:test-impl",
        decision_clock=RESOLVED + "T21:00:00+00:00")
    plan = json.loads(plan_bytes)
    assert plan["session"] == RESOLVED
    assert plan["requested_session"] == REQUESTED
    assert plan["expected_population"] == ["FAKE|TWIN-P|" + RESOLVED]


def test_derive_refuses_the_same_bad_finality_shapes():
    from engine.v2.foundation import artifact_reference

    score_doc, _, replay_doc, coverage_doc = _synthetic_chain()
    score_ref = artifact_reference(b"synthetic-score", "legacy_action.v1.0")
    finality_ref = artifact_reference(b"synthetic-finality", "legacy_action.v1.0")
    for bad in (_finality(date="2026-09-15"), _finality(is_final=False)):
        with pytest.raises(OpsError, match="VALIDATION_FAILED"):
            derive(score_doc, score_ref, bad, finality_ref, replay_doc, coverage_doc,
                  requested_session=REQUESTED, deployment="shadow:test-impl",
                  decision_clock=RESOLVED + "T21:00:00+00:00")


# --------------------------------------------------------------------------
# score / replay / settlement: in-process, resolved date reaches the call
# --------------------------------------------------------------------------


def _stub_scoring(monkeypatch, rows):
    import engine.dashboard.nightly as nightly_module
    import engine.features as features_module
    import engine.score as score_module

    calls = {}

    class _FakeScorer:
        def __init__(self, context):
            self.analog_entry_coverage = 1.0

    def fake_score_calendar(as_of, *, horizon_days, alt_strikes, scorer, tickers, **kwargs):
        calls["as_of"] = as_of
        calls["tickers"] = sorted(tickers)
        return pd.DataFrame(rows)

    monkeypatch.setattr(score_module, "Scorer", _FakeScorer)
    monkeypatch.setattr(features_module.FeatureContext, "load",
                        staticmethod(lambda tickers, years: object()))
    monkeypatch.setattr(score_module, "score_calendar", fake_score_calendar)
    monkeypatch.setattr(nightly_module, "strike_ladder", lambda *a, **k: [])
    return calls


def test_action_score_scores_at_the_resolved_session_on_walk_back(monkeypatch, tmp_path):
    (tmp_path / "finality.json").write_text(json.dumps(_finality()))
    # P6-2: _action_score now refuses without a matching features receipt.
    (tmp_path / "features.json").write_text(json.dumps(
        {"panel_sha256": "panel-sha", "tier4_sha256": "tier4-sha"}))
    monkeypatch.setattr(legacy_adapter, "_current_features_hashes",
                        lambda: {"panel_sha256": "panel-sha", "tier4_sha256": "tier4-sha"})
    rows = [{"ticker": "FAKE", "strategy": "TWIN-P", "event_date": RESOLVED,
            "strike": 100.0, "expiry": "2026-10-16"}]
    calls = _stub_scoring(monkeypatch, rows)

    result = _action_score({"tickers": ["FAKE"], "year_start": 2024, "year_end": 2026,
                            "session": REQUESTED, "horizon_days": 35, "alt_strikes": 1,
                            "expected_population": ("FAKE|TWIN-P|" + RESOLVED,)}, tmp_path)
    assert result["hash"]
    assert calls["as_of"] == pd.Timestamp(RESOLVED)
    document = json.loads((tmp_path / "score.json").read_text())
    assert document["session"] == RESOLVED
    assert document["requested_session"] == REQUESTED


def test_action_decision_replay_rescoring_uses_the_resolved_session(monkeypatch, tmp_path):
    row = {"ticker": "FAKE", "strategy": "TWIN-P", "event_date": RESOLVED,
          "as_of": RESOLVED, "session": "AMC", "fill": 0.5, "strike_offset": None}
    (tmp_path / "score.json").write_text(json.dumps({"rows": [row]}))
    (tmp_path / "finality.json").write_text(json.dumps(_finality()))
    calls = _stub_scoring(monkeypatch, [row])

    result = _action_decision_replay(
        {"session": REQUESTED, "tickers": ("FAKE",), "year_start": 2024, "year_end": 2026},
        tmp_path)
    assert result["hash"]
    assert calls["as_of"] == pd.Timestamp(RESOLVED)
    document = json.loads((tmp_path / "replay.json").read_text())
    assert document["session"] == RESOLVED
    assert document["requested_session"] == REQUESTED
    assert document["population"] == ["FAKE|TWIN-P|" + RESOLVED]


def test_action_settlement_scores_outcomes_through_the_resolved_session(monkeypatch, tmp_path):
    import engine.ledger as legacy_ledger

    (tmp_path / "finality.json").write_text(json.dumps(_finality()))
    calls = {}

    def fake_score_outcomes(*, through):
        calls["through"] = through
        return {"resolved": 0}

    monkeypatch.setattr(legacy_ledger, "score_outcomes", fake_score_outcomes)
    result = _action_settlement({"session": REQUESTED}, tmp_path)
    assert result["hash"]
    assert calls["through"] == RESOLVED
    document = json.loads((tmp_path / "settlement.json").read_text())
    assert document["session"] == RESOLVED
    assert document["requested_session"] == REQUESTED


def test_action_decisions_uses_the_plan_resolved_session_for_as_of(monkeypatch, tmp_path):
    frame = pd.DataFrame([{"ticker": "FAKE", "strategy": "TWIN-P", "event_date": RESOLVED,
                          "as_of": RESOLVED, "entry_date": RESOLVED, "evidence_cutoff": RESOLVED,
                          "event_id": "event-1", "strike_offset": None}])
    (tmp_path / "score.json").write_text(json.dumps({"rows": frame.to_dict(orient="records")}))
    (tmp_path / "finality.json").write_text(json.dumps(_finality()))
    (tmp_path / "decision_plan.json").write_text(json.dumps({
        "schema_version": "decision_plan.v1.0", "session": RESOLVED, "requested_session": REQUESTED,
        "deployment": "shadow:test-impl", "decision_clock": RESOLVED + "T21:00:00+00:00",
        "expected_population": ["FAKE|TWIN-P|" + RESOLVED]}))

    calls = {}
    import engine.ledger as legacy_ledger
    real_build = legacy_ledger.build_prediction_rows

    def spy(frame, *, as_of, decision_ts=None, finality=None, entry_dated_only=False):
        calls["as_of"] = as_of
        return real_build(frame, as_of=as_of, decision_ts=decision_ts, finality=finality,
                          entry_dated_only=entry_dated_only)

    # ``_action_decisions`` does ``from engine.ledger import
    # build_prediction_rows`` inside its own body, so patching the module
    # attribute here is picked up at call time.
    monkeypatch.setattr(legacy_ledger, "build_prediction_rows", spy)

    _action_decisions({"session": REQUESTED, "tickers": [], "year_start": 2024, "year_end": 2026},
                      tmp_path)
    assert calls["as_of"] == RESOLVED
    document = json.loads((tmp_path / "decisions.json").read_text())
    assert document["rows"][0]["as_of"] == RESOLVED


# --------------------------------------------------------------------------
# render: v1's execution_clock shape and walk-back flag
# --------------------------------------------------------------------------


def test_action_render_execution_clock_and_walk_back_flag_match_v1(monkeypatch, tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    score_document = {"rows": [{"ticker": "FAKE", "strategy": "TWIN-P", "event_date": RESOLVED,
                               "as_of": RESOLVED, "fill": 0.5}], "ladder": []}
    (root / "score.json").write_text(json.dumps(score_document))
    finality = _finality()
    (root / "finality.json").write_text(json.dumps(finality))
    (root / "model_evidence.json").write_text(json.dumps({"models": {}}))
    with tarfile.open(root / "ledger_generation.tar", "w"):
        pass

    import engine.dashboard.render as render_module
    import engine.features as features_module
    import engine.score as score_module

    captured = {}

    def fake_build_meta(scores, *, as_of, horizon_days, fill_alpha, alt_strikes, freshness, quota,
                        registry):
        captured["build_meta_as_of"] = as_of
        return {}

    def fake_build_health(*, as_of, size_mae, selfcheck_report=None):
        return {}

    def fake_render_bundle(scores, out, *, as_of, horizon_days, fill_alpha, alt_strikes, panel,
                           trades, meta, health, flags, registry):
        captured["meta"] = meta
        captured["flags"] = flags
        Path(out).mkdir(parents=True, exist_ok=True)
        return {"ok": True}

    monkeypatch.setattr(render_module, "build_meta", fake_build_meta)
    monkeypatch.setattr(render_module, "build_health", fake_build_health)
    monkeypatch.setattr(render_module, "render_bundle", fake_render_bundle)
    monkeypatch.setattr(render_module, "freshness_summary", lambda as_of: {})
    monkeypatch.setattr(render_module, "quota_state", lambda: {})
    monkeypatch.setattr(render_module, "size_model_mae_from_ledger", lambda panel: {})

    class _FakeContext:
        panel = None

    class _FakeScorer:
        def __init__(self, context):
            self.context = context
            self.registry = None
            self.trades = None

    monkeypatch.setattr(score_module, "Scorer", _FakeScorer)
    monkeypatch.setattr(features_module.FeatureContext, "load",
                        staticmethod(lambda tickers, years: _FakeContext()))

    result = _action_render({"session": REQUESTED, "tickers": ["FAKE"], "context_tickers": ["FAKE"],
                             "year_start": 2024, "year_end": 2026}, root)
    assert result["path"] == "bundle.tar"
    assert captured["build_meta_as_of"] == RESOLVED
    clock = captured["meta"]["execution_clock"]
    assert clock == {"requested_as_of": REQUESTED, "resolved_as_of": RESOLVED, "finality": finality}
    flag = {"kind": "as_of_resolved",
           "detail": f"requested {REQUESTED} resolved to final {RESOLVED}: {finality['detail']}"}
    assert flag in captured["flags"]

    # no-walk-back regression: identical requested/resolved carries no flag
    # and both execution_clock dates equal the (single) session.
    (root / "finality.json").write_text(json.dumps(_finality(date=REQUESTED)))
    captured.clear()
    _action_render({"session": REQUESTED, "tickers": ["FAKE"], "context_tickers": ["FAKE"],
                    "year_start": 2024, "year_end": 2026}, root)
    assert captured["meta"]["execution_clock"] == {
        "requested_as_of": REQUESTED, "resolved_as_of": REQUESTED, "finality": _finality(date=REQUESTED)}
    assert all(f.get("kind") != "as_of_resolved" for f in captured["flags"])


# --------------------------------------------------------------------------
# DAG: the new finality bindings are declared dependencies
# --------------------------------------------------------------------------


def test_score_replay_settlement_and_publication_bind_and_depend_on_finality():
    plan = build_nightly_plan(str(REPO), REQUESTED)
    requests = build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026)
    by_kind = {r.job.kind: r for r in requests}
    finality_id = job_id_for("shadow", by_kind["legacy_finality"].idempotency_key)
    for kind in ("legacy_score", "legacy_decision_replay", "legacy_settlement", "publication"):
        request = by_kind[kind]
        assert "finality.json" in request.job.parameters["input_bindings"]
        assert request.job.parameters["input_bindings"]["finality.json"] == (
            finality_id + "#legacy_finality")
        assert finality_id in request.job.dependency_job_ids


# --------------------------------------------------------------------------
# end to end: score through decision-commit, walk-back and the regression
# --------------------------------------------------------------------------


def _run_decision_commit(tmp_path, tag, *, requested, resolved):
    """Derive plan/evidence for ``(requested, resolved)``, seed succeeded
    score/finality/evidence parents, submit ``legacy_decisions`` for real
    (job parameters carry the REQUESTED session, exactly as
    ``build_legacy_job_requests`` does), run it to completion, and return
    ``(state, failure_json, decisions_row, decisions_watermark_occurrence)``.
    """
    score_doc, finality, replay_doc, coverage_doc = _synthetic_chain(
        requested=requested, resolved=resolved)
    decision_clock = resolved + "T21:00:00+00:00"

    root = tmp_path / tag
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    fixture = store_root / "unused.txt"
    fixture.write_bytes(b"a legacy read-set member decisions never opens")

    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    store = ArtifactStore(root)
    try:
        setup_epoch = begin_epoch(conn, clock=clock, boot_id="setup", pid=1)
        setup = Supervisor(setup_epoch, "setup")

        manifest_ref = _manifest_ref(store, conn, clock, (LegacyFileRef(
            path="unused.txt", content_hash=file_hash(fixture), byte_size=fixture.stat().st_size),))
        score_ref = _publish(store, conn, clock, score_doc, "legacy_action.v1.0")
        finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
        coverage_ref = _publish(store, conn, clock, coverage_doc, "finality_coverage.v1.0")

        plan_bytes, evidence_bytes = derive(
            score_doc, score_ref, finality, finality_ref, replay_doc, coverage_doc,
            requested_session=requested, deployment="shadow:test-impl",
            decision_clock=decision_clock)
        plan_ref = store.publish_bytes(plan_bytes, schema_ref="decision_plan.v1.0")
        evidence_ref = store.publish_bytes(evidence_bytes, schema_ref="decision_evidence.v1.0")

        score_job = _succeed_parent(conn, clock, setup, key=tag + "-score",
                                    output_name="legacy_score", ref=score_ref)
        finality_job = _succeed_finality_parent(conn, clock, setup, key=tag + "-finality",
                                                finality_ref=finality_ref, coverage_ref=coverage_ref)
        evidence_job = _succeed_evidence_parent(conn, clock, setup, key=tag + "-evidence",
                                                plan_ref=plan_ref, evidence_ref=evidence_ref)

        with transaction(conn):
            set_authority(conn, None, "catalog", resolved + "T20:00:00.000000Z")

        bindings = {"legacy_manifest.json": manifest_ref.artifact_id,
                   "score.json": score_job + "#legacy_score",
                   "finality.json": finality_job + "#legacy_finality",
                   "decision_plan.json": evidence_job + "#decision_plan",
                   "decision_evidence.json": evidence_job + "#decision_evidence"}
        profile = profile_named(DEFAULT_POLICY, "validation")
        job = JobSpec(
            kind="legacy_decisions",
            implementation_ref=content_hash(worker_source_manifest(REPO)),
            spec_hash=None,
            environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
            # P2-C03: job parameters keep ``session`` REQUESTED -- job
            # identity must never depend on data (finality) not yet read.
            parameters={"expected_ids": ("legacy_decisions",), "session": requested,
                        "tickers": (), "year_start": 2024, "year_end": 2026,
                        "input_bindings": bindings},
            input_refs=(manifest_ref.artifact_id,),
            dependency_job_ids=(evidence_job, score_job, finality_job),
            output_namespace="shadow", resource_class="validation", retry_policy_ref="bounded",
            checkpoint_contract_ref="legacy_action.v1.0")
        receipt = submit(conn, registry(), POLICY, SubmitRequest(
            namespace="shadow", idempotency_key=tag + "-decisions", principal="operator", job=job),
            clock=clock)

        service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            state = _run_until_terminal(service, conn, receipt.job_id)
        finally:
            service.close()

        row = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        decision_row = conn.execute(
            "SELECT payload_json FROM decisions WHERE kind='prediction'").fetchone()
        watermark_row = conn.execute(
            "SELECT occurrence FROM watermarks WHERE pipeline='nightly' AND scope='shadow' "
            "AND stage='decisions'").fetchone()
        return (state, row["failure_json"],
               json.loads(decision_row["payload_json"]) if decision_row else None,
               watermark_row["occurrence"] if watermark_row else None)
    finally:
        conn.close()


def test_walk_back_commits_with_resolved_as_of_and_advances_the_resolved_watermark(tmp_path):
    state, failure, decision, occurrence = _run_decision_commit(
        tmp_path, "walkback", requested=REQUESTED, resolved=RESOLVED)
    assert state == "succeeded", failure
    assert decision is not None
    assert decision["as_of"] == RESOLVED
    assert occurrence == RESOLVED


def test_no_walk_back_regression_commits_exactly_as_before(tmp_path):
    state, failure, decision, occurrence = _run_decision_commit(
        tmp_path, "nowalkback", requested=REQUESTED, resolved=REQUESTED)
    assert state == "succeeded", failure
    assert decision is not None
    assert decision["as_of"] == REQUESTED
    assert occurrence == REQUESTED


# --------------------------------------------------------------------------
# export / publication: watermark occurrence and receipt carry the resolved
# session, never the requested one, once decisions have committed at it
# --------------------------------------------------------------------------


def test_ledger_export_watermark_and_receipt_use_the_resolved_session(tmp_path):
    conn, clock, supervisor, store, root = _effects_open(tmp_path)
    try:
        scope = "shadow"
        # decision_commit writes the DECISIONS watermark at the RESOLVED
        # session -- seed it that way and give the job the REQUESTED one.
        _seed_decisions(conn, clock, scope, RESOLVED,
                        predictions=[{"row_id": "evt-1-pred", "event_id": "evt-1", "ticker": "FAKE",
                                     "strategy": "TWIN-P", "event_date": RESOLVED}])
        claim = _effects_submit_and_claim(
            conn, clock, supervisor, kind="ledger_export", key="export-1",
            parameters=_effects_params("ledger_export", REQUESTED, scope))
        effect, extra_refs = ledger_export_effect(conn, store, claim, root, REPO, clock=clock)

        def commit(inner_conn):
            for name, ref in extra_refs:
                from engine.v2.ops.checkpoints import register_artifact
                register_artifact(inner_conn, ref, claim.attempt_id, clock)
                inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                                   (claim.attempt_id, name, ref.artifact_id))
            effect(inner_conn)

        from engine.v2.ops.lifecycle import Outcome, commit_attempt
        commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                       clock=clock, effects=commit)

        row = conn.execute("SELECT receipt_json FROM outbox WHERE kind='export'").fetchone()
        receipt = json.loads(row["receipt_json"])
        assert receipt["session"] == RESOLVED
        assert receipt["requested_session"] == REQUESTED
        wm = conn.execute("SELECT occurrence FROM watermarks WHERE pipeline='nightly' AND scope=? "
                          "AND stage='export'", (scope,)).fetchone()
        assert wm["occurrence"] == RESOLVED
    finally:
        conn.close()


def test_publication_uses_the_resolved_session_on_walk_back(tmp_path):
    conn, clock, supervisor, store, root = _effects_open(tmp_path)
    try:
        scope = "shadow"
        claim = _publication_setup(conn, clock, supervisor, store, scope=scope, session=REQUESTED,
                                   decisions_session=RESOLVED, finality_session=RESOLVED)
        from engine.v2.ops.publication import current as release_current
        publication_effect(conn, store, claim, root, REPO, clock=clock, store_root=FAKE_STORE_ROOT)
        assert release_current(root / "releases" / scope) is not None
        wm = conn.execute("SELECT occurrence FROM watermarks WHERE pipeline='nightly' AND scope=? "
                          "AND stage='publication'", (scope,)).fetchone()
        assert wm["occurrence"] == RESOLVED
    finally:
        conn.close()


def test_publication_still_refuses_a_decisions_watermark_that_disagrees_with_finality(tmp_path):
    """The finality binding is the independent anchor (guide: never trust
    the decisions watermark alone) -- a decisions watermark for neither the
    requested nor the resolved session is still refused."""
    conn, clock, supervisor, store, root = _effects_open(tmp_path)
    try:
        scope = "shadow"
        claim = _publication_setup(conn, clock, supervisor, store, scope=scope, session=REQUESTED,
                                   decisions_session="2026-09-10", finality_session=RESOLVED)
        with pytest.raises(OpsError, match="PUBLICATION_REFUSED"):
            publication_effect(conn, store, claim, root, REPO, clock=clock)
    finally:
        conn.close()
