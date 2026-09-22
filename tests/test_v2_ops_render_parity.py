"""P2-5 render parity (D19) -- tier 0, synthetic, no private data.

``_action_render`` is compared against ``render_bundle`` invoked the legacy
way (as ``engine/dashboard/nightly.py`` calls it) on the SAME scores, ladder,
model evidence, ledger generation and finality. Both paths independently call
``build_meta``/``build_health``, so their ``generated_at`` (and the other
:data:`EXECUTION_METADATA_FIELDS`) genuinely differ; every other serialized
byte must agree.

``Scorer``/``FeatureContext.load`` are monkeypatched to a small synthetic
panel/trades/registry -- the same pattern
``test_decision_replay_action_scopes_context_full_and_scoring_eligible`` uses
in ``tests/test_v2_ops_nightly_completion.py``. No real scoring, no real
store, no network.
"""
from __future__ import annotations

import datetime as datetime_module
import json
import sqlite3
import tarfile
from pathlib import Path

import pandas as pd
import pytest

import engine.dashboard.render as render_module
import engine.features as features_module
import engine.score as score_module
from checks.phase2_render_oracle import legacy_way_bundle, v1_flags_for_scenario
from engine.score import ScoreResult
from engine.v2.ledger.export import export_generation
from engine.v2.ops.errors import OpsError
from engine.v2.ops.legacy_adapter import _action_render
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.submission import job_id_for
from engine.v2.ops.render_inputs import (
    ABSENT_STAGES,
    absent_stage_flags,
    assemble_scores,
    bundle_content_hash,
    diff_bundles,
    model_evidence_stale_flag,
    stage_model_evidence,
    unknown_selfcheck_report,
)

AS_OF = pd.Timestamp("2026-08-10")
EVENT = pd.Timestamp("2026-08-12")
TICKER = "ZZTEST"
ROW_ID = "2026-08-10|ZZTEST|STR-THRU|100.0000|2026-08-13"


class _FrozenDatetime(datetime_module.datetime):
    """Stands in for ``engine.dashboard.render``'s module-level ``datetime``.

    ``build_meta``/``build_health`` call ``datetime.now(timezone.utc)`` (the
    only two ``now()``s in that module -- see
    ``grep -n "now()" engine/dashboard/render.py``). Real ``datetime`` can't
    be monkeypatched directly (it's a builtin extension type), so the
    module's bound name is replaced with this subclass instead.
    """

    _value = datetime_module.datetime(2020, 1, 1, tzinfo=datetime_module.timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls._value.astimezone(tz) if tz is not None else cls._value


def _freeze_render_clock(monkeypatch, when: datetime_module.datetime) -> None:
    monkeypatch.setattr(render_module, "datetime", _FrozenDatetime)
    _FrozenDatetime._value = when


@pytest.fixture(autouse=True)
def _pointed_paths(tmp_path, monkeypatch):
    """Point ``engine.paths`` at ``<tmp_path>/job/legacy`` for this test.

    ``_action_render``'s legacy calls (``size_model_mae_from_ledger``,
    ``freshness_summary``, the book view) read through modules that already
    imported ``engine.paths`` before this test ran, so ``paths.ROOT`` /
    ``paths.LEDGER`` are frozen at first import (see ``engine/paths.py``).
    ``_rooted_import`` only sets the env var; it never reloads that frozen
    module. This mirrors ``tests/conftest.py``'s own ``tmp_root`` fixture,
    pinned to the exact path ``_stage`` below uses.
    """
    import importlib
    import os

    from engine import paths

    # Restore whatever INVESTING_PLAN_ROOT this process actually started
    # with, not unconditionally unset -- an unconditional delenv assumes the
    # process began with no override, which is false whenever the harness
    # exports the real checkout's root before running pytest from a
    # worktree. Getting this wrong leaves paths.ROOT re-derived from the
    # file-default after teardown while any module that read paths.ROOT at
    # its own import time (under the original env) keeps the stale value.
    original_root_env = os.environ.get("INVESTING_PLAN_ROOT")
    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(tmp_path / "job" / "legacy"))
    importlib.reload(paths)
    yield
    if original_root_env is None:
        monkeypatch.delenv("INVESTING_PLAN_ROOT", raising=False)
    else:
        monkeypatch.setenv("INVESTING_PLAN_ROOT", original_root_env)
    importlib.reload(paths)


# --------------------------------------------------------------------------
# synthetic inputs
# --------------------------------------------------------------------------


def _result(strike=100.0, strike_offset=None, **kwargs) -> dict:
    base = dict(
        as_of=AS_OF, event_date=EVENT, session="AMC", exp_pnl_model=0.031,
        win_model=0.56, model_p10=-0.22, model_p90=0.41, exp_pnl_analog=0.028,
        win_analog=0.54, ci_low=-0.01, ci_high=0.07, n_analogs=120,
        gate_score=0.71, gate_threshold=0.6, gate_pass=True, entry_date=EVENT,
        exit_date=EVENT + pd.Timedelta(days=1), strike=strike, spot=100.0,
        expiry=EVENT + pd.Timedelta(days=2), entry_cost=6.0, dte_entry=2,
        payoff={"intercept": 0.01, "slope": 0.006, "n": 400},
        driver_name="abs_move", driver_prediction=7.5,
        model_versions={"size": "size_v13"}, snapshot_hash="snap-test",
    )
    base.update(kwargs)
    row = ScoreResult(ticker=TICKER, strategy="STR-THRU", **base).as_dict()
    row["strike_offset"] = strike_offset
    return row


def _score_document() -> dict:
    board = [_result()]
    ladder = [_result(strike=95.0, strike_offset=-5.0)]
    for row in board + ladder:
        row["row_id"] = "|".join(str(row.get(k, "")) for k in
                                 ("ticker", "strategy", "event_date", "strike", "expiry"))
    return {"rows": board, "ladder": ladder, "analog_entry_coverage": 1.0,
            "expected_population": [f"{TICKER}|STR-THRU|{EVENT.date()}"],
            "observed_population": [f"{TICKER}|STR-THRU|{EVENT.date()}"],
            "tickers": [TICKER]}


def _finality_doc() -> dict:
    return {"date": str(AS_OF.date()), "market_wide": True, "daily_share": 1.0,
            "chain_share": 1.0, "is_final": True, "detail": "synthetic", "tickers": 1,
            "covered": 1}


def _model_evidence_doc() -> dict:
    return {"generated_at": "2026-08-01T00:00:00+00:00", "elapsed_s": 4.2,
            "models": {"size": {"inputs": {}}}}


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [TICKER, TICKER], "date": pd.to_datetime(["2026-05-11", "2026-08-12"]),
        "k": [30, 31], "implied_move": [6.0, 7.2], "or_implied": [6.1, 7.2],
        "move": [-4.0, 7.9], "abs_move": [4.0, 7.9]})


def _trades() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [TICKER], "strategy": ["STR-THRU"], "event_date": [EVENT],
        "entry_date": [EVENT], "exit_date": [EVENT + pd.Timedelta(days=1)],
        "fill_alpha": [0.5], "entry_cost": [6.0], "exit_value": [6.6], "ret": [0.10]})


class _FakeRegistry:
    def champion(self, role, strategy=None):
        raise KeyError(role)


class _FakeScorer:
    """Stands in for ``engine.score.Scorer``: holds context/trades/registry only."""

    def __init__(self, *, context):
        self.context = context
        self.registry = _FakeRegistry()
        self.trades = _trades()


def _patch_scorer(monkeypatch, panel):
    context = features_module.FeatureContext(panel=panel, daily=None, calendar=None)
    monkeypatch.setattr(features_module.FeatureContext, "load",
                        staticmethod(lambda *a, **k: context))
    monkeypatch.setattr(score_module, "Scorer", _FakeScorer)


def _prediction_row() -> dict:
    score = _result()
    return {
        "schema_version": 3, "row_id": ROW_ID, "written_at": "2026-08-10T21:30:00+00:00",
        "as_of": str(AS_OF.date()), "decision_ts": "2026-08-10T21:30:00+00:00",
        "ticker": TICKER, "event_id": None, "event_date": str(EVENT.date()), "session": "AMC",
        "strategy": "STR-THRU",
        "settlement": {"policy": "close", "spec_version": 1, "structure_spec": None,
                       "structure_params": None, "variant": None},
        "structure": {"strike": 100.0, "expiry": str((EVENT + pd.Timedelta(days=2)).date()),
                      "legs": [], "decision_date": str(AS_OF.date()),
                      "entry_date": str(EVENT.date()),
                      "exit_date": str((EVENT + pd.Timedelta(days=1)).date()), "dte_entry": 2},
        "intended_prices": {"alpha": 0.5, "quote_date": str(AS_OF.date()), "quoted_cost": 6.0,
                            "entry_cost": 6.0, "spot": 100.0},
        "finality": {}, "score": score, "model_versions": {}, "snapshot_hash": "snap-test",
        "audit_receipt": None, "supersedes": None,
    }


def _outcome_row() -> dict:
    return {"row_id": ROW_ID, "status": "resolved", "event_date": str(EVENT.date()),
            "resolved_at": "2026-08-15T21:00:00+00:00", "realized_pnl": 0.02,
            "realized_entry_cost": 6.1, "realized_exit_value": 8.0, "reason": None,
            "exit_finality": {"is_final": True}}


def _build_ledger_generation_tar(tmp_path: Path) -> Path:
    """A tiny catalog -> ``export_generation`` -> one generation -> a tar of it."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE decisions (sequence INTEGER, kind TEXT, payload_json TEXT)")
    conn.execute("INSERT INTO decisions VALUES (1, 'prediction', ?)",
                (json.dumps(_prediction_row()),))
    conn.execute("INSERT INTO decisions VALUES (2, 'outcome', ?)",
                (json.dumps(_outcome_row()),))
    conn.commit()
    ledger_root = tmp_path / "ledger_root"
    generation = export_generation(conn, ledger_root, generation="gen1")
    tar_path = tmp_path / "ledger_generation.tar"
    with tarfile.open(tar_path, "w") as archive:
        for child in sorted(generation.iterdir()):
            archive.add(child, arcname=child.name)
    conn.close()
    return tar_path


def _stage(tmp_path: Path, *, with_model_evidence=True, with_ledger=True) -> Path:
    root = tmp_path / "job"
    root.mkdir()
    (root / "score.json").write_text(json.dumps(_score_document()))
    (root / "finality.json").write_text(json.dumps(_finality_doc()))
    if with_model_evidence:
        (root / "model_evidence.json").write_text(json.dumps(_model_evidence_doc()))
    if with_ledger:
        tar_path = _build_ledger_generation_tar(tmp_path)
        (root / "ledger_generation.tar").write_bytes(tar_path.read_bytes())
    return root


_PARAMETERS = {"session": str(AS_OF.date()), "tickers": [TICKER], "context_tickers": [TICKER],
               "year_start": 2025, "year_end": 2026, "horizon_days": 35, "alt_strikes": 1}


# --------------------------------------------------------------------------
# D19
# --------------------------------------------------------------------------


def test_d19_v2_render_matches_render_bundle_the_legacy_way(monkeypatch, tmp_path):
    """The two renders run under clocks more than a day apart (fix 2): a
    same-second comparison can't prove EXECUTION_METADATA_FIELDS covers
    every wall-clock value in the bundle, and can flake at a second
    boundary. A day-plus gap makes any undeclared timestamp field show up
    as a real diff below, not a coincidence of timing.
    """
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    root = _stage(tmp_path)

    _freeze_render_clock(monkeypatch, datetime_module.datetime(
        2020, 1, 1, tzinfo=datetime_module.timezone.utc))
    result = _action_render(dict(_PARAMETERS), root)
    bundle_v2 = root / "bundle"
    assert bundle_v2.is_dir()
    assert result["path"] == "bundle.tar"
    render_doc = json.loads((root / "render.json").read_text())
    assert render_doc["bundle_content_hash"] == bundle_content_hash(bundle_v2)
    assert render_doc["absent_stages"] == list(ABSENT_STAGES)

    _freeze_render_clock(monkeypatch, datetime_module.datetime(
        2031, 6, 15, tzinfo=datetime_module.timezone.utc))
    scores = assemble_scores(_score_document())
    finality = _finality_doc()
    bundle_legacy = tmp_path / "bundle_legacy"
    legacy_way_bundle(bundle_legacy, scores=scores, panel=panel, trades=_trades(),
                      registry=_FakeRegistry(), finality=finality,
                      requested_as_of=AS_OF, resolved_as_of=AS_OF, tickers=(TICKER,),
                      horizon_days=35, evidence=_model_evidence_doc())

    diffs = diff_bundles(bundle_v2, bundle_legacy)
    assert diffs == []
    assert bundle_content_hash(bundle_v2) == bundle_content_hash(bundle_legacy)

    meta_v2 = json.loads((bundle_v2 / "data" / "meta.json").read_text())
    meta_legacy = json.loads((bundle_legacy / "data" / "meta.json").read_text())
    # Both paths independently called build_meta over a decade apart -- this
    # is not a same-second coincidence, which is exactly why
    # EXECUTION_METADATA_FIELDS must exclude it.
    assert meta_v2["generated_at"] != meta_legacy["generated_at"]
    assert meta_v2["generated_at"].startswith("2020-01-01")
    assert meta_legacy["generated_at"].startswith("2031-06-15")


def test_metadata_only_change_does_not_move_the_content_hash(monkeypatch, tmp_path):
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    root = _stage(tmp_path)
    _action_render(dict(_PARAMETERS), root)
    bundle = root / "bundle"
    before = bundle_content_hash(bundle)

    meta_path = bundle / "data" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["freshness"] = {"as_of": "1999-01-01", "daily_market_last_date": None}
    meta["quota"] = {"remaining": 999}
    meta["cron"] = {"entry": "changed"}
    meta_path.write_text(json.dumps(meta, sort_keys=True) + "\n")

    health_path = bundle / "data" / "health.json"
    health = json.loads(health_path.read_text())
    health["generated_at"] = "1999-01-01T00:00:00+00:00"
    health_path.write_text(json.dumps(health, sort_keys=True) + "\n")

    assert bundle_content_hash(bundle) == before


def test_a_planted_score_field_change_is_caught_and_named(monkeypatch, tmp_path):
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    root = _stage(tmp_path)
    _action_render(dict(_PARAMETERS), root)
    bundle_a = root / "bundle"

    bundle_b = tmp_path / "bundle_b"
    import shutil

    shutil.copytree(bundle_a, bundle_b)
    board_path = bundle_b / "data" / "board.json"
    board = json.loads(board_path.read_text())
    board["rows"][0]["exp_pnl_model"] = 999.0
    board_path.write_text(json.dumps(board, sort_keys=True) + "\n")

    diffs = diff_bundles(bundle_a, bundle_b)
    assert any("board.json" in d for d in diffs)
    assert bundle_content_hash(bundle_a) != bundle_content_hash(bundle_b)


def test_a_planted_ledger_prediction_row_change_is_caught(monkeypatch, tmp_path):
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    root = _stage(tmp_path)
    _action_render(dict(_PARAMETERS), root)
    bundle_a = root / "bundle"

    bundle_b = tmp_path / "bundle_b"
    import shutil

    shutil.copytree(bundle_a, bundle_b)
    health_path = bundle_b / "data" / "health.json"
    health = json.loads(health_path.read_text())
    health["size_model"] = {"available": True, "n": 1, "model_mae_pp": 12345.0}
    health_path.write_text(json.dumps(health, sort_keys=True) + "\n")

    diffs = diff_bundles(bundle_a, bundle_b)
    assert any(d.startswith("data/health.json") for d in diffs)
    assert bundle_content_hash(bundle_a) != bundle_content_hash(bundle_b)


# --------------------------------------------------------------------------
# book view: the generation's predictions, not a staged mutable ledger
# --------------------------------------------------------------------------


def test_book_reflects_the_bound_generation_and_ignores_a_planted_mutable_ledger(
        monkeypatch, tmp_path):
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    root = _stage(tmp_path)
    # A mutable-ledger file planted in staging before the action runs -- it
    # must be removed/replaced, never read.
    poisoned = root / "legacy" / "ledger" / "predictions"
    poisoned.mkdir(parents=True)
    (poisoned / "9999-01-01.jsonl").write_text(json.dumps(
        {"row_id": "poison", "ticker": "POISON", "event_date": "9999-01-01",
         "as_of": "9999-01-01", "strategy": "STR-THRU", "score": {}}) + "\n")

    _action_render(dict(_PARAMETERS), root)

    predictions_dir = root / "legacy" / "ledger" / "predictions"
    files = sorted(p.name for p in predictions_dir.glob("*.jsonl"))
    assert "9999-01-01.jsonl" not in files
    all_rows = []
    for path in predictions_dir.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
    assert {row["row_id"] for row in all_rows} == {ROW_ID}

    book = json.loads((root / "bundle" / "data" / "book.json").read_text())
    assert book["available"] is True
    assert any(row.get("ticker") == TICKER for row in book["rows"])


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_missing_ledger_binding_refuses(monkeypatch, tmp_path):
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    root = _stage(tmp_path, with_ledger=False)
    with pytest.raises(OpsError, match="ledger generation not bound"):
        _action_render(dict(_PARAMETERS), root)


def test_missing_model_evidence_refuses(monkeypatch, tmp_path):
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    root = _stage(tmp_path, with_model_evidence=False)
    with pytest.raises(OpsError, match="model evidence artifact is missing"):
        _action_render(dict(_PARAMETERS), root)


# --------------------------------------------------------------------------
# absent stages
# --------------------------------------------------------------------------


def test_absent_stage_flags_list_all_five_stages():
    flags = absent_stage_flags()
    assert len(ABSENT_STAGES) == 5
    assert {f["stage"] for f in flags} == set(ABSENT_STAGES)
    assert {f["kind"] for f in flags} == {"shadow_stage_absent"}


# --------------------------------------------------------------------------
# P2-C08: the full render flag inventory and health's explicit unknown state
# --------------------------------------------------------------------------


class _FakeCalendar:
    def shift(self, date, n):
        return pd.Timestamp(date).normalize() + pd.Timedelta(days=int(n))


def _conflict_events(event_date) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": ["evt1"], "ticker": [TICKER],
        "event_date": [pd.Timestamp(event_date)], "session": ["AMC"],
        "date_conflict": [True], "src_orats": [False],
    })


def _stale_panel() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(["2026-08-01", "2026-08-03"])})


def _patch_flag_sources(monkeypatch):
    """Everything :func:`legacy_adapter._panel_lag_flags`/
    ``_calendar_conflict_flags`` read that this test's synthetic root does
    not otherwise provide: a lagging panel, a deterministic calendar, and one
    conflicted earnings_events row."""
    import engine.calendar as calendar_module
    import engine.data.store as store_module

    monkeypatch.setattr(features_module, "load_panel", lambda *a, **k: _stale_panel())
    monkeypatch.setattr(calendar_module, "trading_calendar", lambda *a, **k: _FakeCalendar())
    monkeypatch.setattr(
        store_module, "read_table",
        lambda name, **k: _conflict_events(AS_OF) if name == "earnings_events" else pd.DataFrame())


def test_render_flags_cover_every_class_with_nonempty_scenario(monkeypatch, tmp_path):
    """P2-C08 acceptance: a scenario carrying a walk-back, a lagging panel, a
    calendar-conflict row and stale model evidence produces every class (a)
    flag at v1's own kind/detail, the class (b) absent-stage disclosures, and
    the class (c) explicit unknowns -- none silently dropped, and the
    resulting v2 flag list equals one built independently of the adapter
    (:func:`v1_flags_for_scenario`).
    """
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    _patch_flag_sources(monkeypatch)

    root = _stage(tmp_path)
    evidence = dict(_model_evidence_doc())
    evidence["degraded"] = True
    evidence["degraded_reason"] = "RuntimeError: rebuild boom"
    (root / "model_evidence.json").write_text(json.dumps(evidence))

    requested = AS_OF + pd.Timedelta(days=1)
    params = dict(_PARAMETERS)
    params["session"] = str(requested.date())

    _action_render(params, root)
    flags = json.loads((root / "bundle" / "data" / "flags.json").read_text())["flags"]

    finality = _finality_doc()
    expected = v1_flags_for_scenario(
        requested_as_of=str(requested.date()), resolved_as_of=str(AS_OF.date()),
        finality=finality, tickers=(TICKER,), horizon_days=35, evidence=evidence)
    assert flags == expected

    kinds = [f["kind"] for f in flags]
    assert kinds.count("as_of_resolved") == 1
    assert kinds.count("panel_stale") == 1
    assert kinds.count("calendar_date_conflict") == 1
    assert kinds.count("model_evidence_stale") == 1
    assert {f["stage"] for f in flags if f["kind"] == "shadow_stage_absent"} == set(ABSENT_STAGES)
    assert {"quota_unknown", "freshness_unknown", "prior_run_state_unknown"} <= set(kinds)
    conflict = next(f for f in flags if f["kind"] == "calendar_date_conflict")
    assert conflict["tickers"] == {TICKER: [str(AS_OF.date())]}
    stale = next(f for f in flags if f["kind"] == "model_evidence_stale")
    assert "rebuild boom" in stale["detail"]


def test_health_selfcheck_is_explicit_unknown_when_nothing_bound(monkeypatch, tmp_path):
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    root = _stage(tmp_path)
    _action_render(dict(_PARAMETERS), root)
    health = json.loads((root / "bundle" / "data" / "health.json").read_text())
    assert health["last_selfcheck"] == unknown_selfcheck_report()
    assert health["last_selfcheck"]["ok"] is None
    assert health["last_selfcheck"]["known"] is False


def test_health_reflects_a_bound_prior_selfcheck(monkeypatch, tmp_path):
    panel = _panel()
    _patch_scorer(monkeypatch, panel)
    root = _stage(tmp_path)
    prior = {"ok": True, "n_checked": 20, "n_board_rows": 20, "mismatches": [],
             "snapshot_ok": True, "as_of": str(AS_OF.date()), "seed": 0,
             "elapsed_s": 1.2, "detail": ""}
    (root / "prior_selfcheck.json").write_text(json.dumps(prior))
    _action_render(dict(_PARAMETERS), root)
    health = json.loads((root / "bundle" / "data" / "health.json").read_text())
    assert health["last_selfcheck"] == prior


def test_model_evidence_action_preserves_degraded_state_for_render(monkeypatch, tmp_path):
    """Decision 3: a failed rebuild must degrade to the cached table AS DATA
    on the artifact, not as a live exception -- render runs in a separate
    process/job and can only see what the artifact carries.
    """
    import engine.dashboard.model_evidence as model_evidence_module
    from engine.v2.ops.legacy_adapter import _action_model_evidence

    def _boom(**kwargs):
        raise RuntimeError("rebuild boom")

    monkeypatch.setattr(model_evidence_module, "build_model_evidence", _boom)
    monkeypatch.setattr(model_evidence_module, "load_model_evidence",
                        lambda: {"models": {}, "generated_at": "cached-2026-08-01"})
    root = tmp_path / "job"
    root.mkdir()
    _action_model_evidence({}, root)
    doc = json.loads((root / "model_evidence.json").read_text())
    assert doc["degraded"] is True
    assert "rebuild boom" in doc["degraded_reason"]
    assert doc["generated_at"] == "cached-2026-08-01"

    flag = model_evidence_stale_flag(doc)
    assert flag["kind"] == "model_evidence_stale"
    assert "rebuild boom" in flag["detail"]


def test_model_evidence_action_marks_a_clean_rebuild_not_degraded(monkeypatch, tmp_path):
    import engine.dashboard.model_evidence as model_evidence_module
    from engine.v2.ops.legacy_adapter import _action_model_evidence

    monkeypatch.setattr(model_evidence_module, "build_model_evidence",
                        lambda **k: {"generated_at": "fresh", "models": {}})
    root = tmp_path / "job"
    root.mkdir()
    _action_model_evidence({}, root)
    doc = json.loads((root / "model_evidence.json").read_text())
    assert doc["degraded"] is False
    assert model_evidence_stale_flag(doc) is None


def test_model_evidence_action_writes_a_real_nan_as_valid_json_and_stages_it_for_legacy(
        monkeypatch, tmp_path):
    """Real shadow nightly attempt 14: a Spearman correlation on a constant
    input (``engine/dashboard/model_evidence.py:104``) can be a genuine
    NaN. ``json.dumps(..., allow_nan=False)`` used to raise ValueError on it
    (``legacy_adapter._write_action``, ``json.dumps`` has no way to write a
    Python NaN as strict JSON) -- fixed by tagging it
    (``engine.v2.foundation.tag_nonfinite``) before the dump. The artifact on
    disk must now be valid strict JSON, AND ``stage_model_evidence`` must
    still hand the REAL (unmodified) ``engine.dashboard.model_evidence.
    load_model_evidence`` -- real legacy code, plain ``json.load`` -- back
    a real ``float('nan')``, not the tag: legacy's own
    ``abs(s.get("magnitude_spearman") or 0.0)`` would TypeError on a dict.
    """
    import math

    import engine.dashboard.model_evidence as model_evidence_module
    from engine.v2.ops.legacy_adapter import _action_model_evidence

    monkeypatch.setattr(model_evidence_module, "build_model_evidence", lambda **k: {
        "generated_at": "fresh",
        "models": {"dyn_sv_chooser_v1_1": {"inputs": [
            {"name": "magnitude_spearman", "n": 12, "coverage": 1.0,
             "usable": True, "magnitude_spearman": float("nan")},
        ]}},
    })
    root = tmp_path / "job"
    root.mkdir()
    _action_model_evidence({}, root)

    raw_text = (root / "model_evidence.json").read_text()
    # Valid strict JSON: no bare NaN/Infinity literal, and json.loads (no
    # parse_constant override) does not silently accept one either.
    parsed = json.loads(raw_text)
    magnitude = parsed["models"]["dyn_sv_chooser_v1_1"]["inputs"][0]["magnitude_spearman"]
    assert magnitude == {"__nonfinite__": "nan"}

    legacy_root = tmp_path / "legacy"
    destination = stage_model_evidence(root / "model_evidence.json", legacy_root)
    staged = json.loads(destination.read_text())
    staged_value = staged["models"]["dyn_sv_chooser_v1_1"]["inputs"][0]["magnitude_spearman"]
    assert isinstance(staged_value, float) and math.isnan(staged_value)
    # Legacy's own reader sees exactly what its own writer would have
    # produced for this value -- a bare NaN token, not a tagged object.
    assert "NaN" in destination.read_text()
    assert "__nonfinite__" not in destination.read_text()


# --------------------------------------------------------------------------
# DAG
# --------------------------------------------------------------------------


def test_render_job_binds_finality_score_and_model_evidence_as_job_outputs():
    plan = build_nightly_plan(str(Path(__file__).resolve().parents[1]), str(AS_OF.date()))
    requests = build_legacy_job_requests(plan, tickers=(TICKER,), year_start=2025, year_end=2026)
    render_request = next(r for r in requests if r.job.kind == "legacy_render")
    bindings = render_request.job.parameters["input_bindings"]
    # P2-5/Task5: the export stage is now wired between decision commit and
    # projection (guide §9.4 item 3), so render's ledger generation is the
    # verified ``ledger_export`` output, never a staged mutable ledger copy.
    assert set(bindings) == {"score.json", "model_evidence.json", "finality.json",
                             "ledger_generation.tar"}
    assert bindings["finality.json"].endswith("#legacy_finality")
    assert bindings["score.json"].endswith("#legacy_score")
    assert bindings["model_evidence.json"].endswith("#legacy_model_evidence")
    assert bindings["ledger_generation.tar"].endswith("#ledger_export")
    assert bindings["ledger_generation.tar"].split("#")[0] in render_request.job.dependency_job_ids


def test_every_job_binding_names_a_declared_dependency():
    """A ``job_…#output`` binding whose job isn't in ``dependency_job_ids``
    is refused at launch by ``input_bindings._resolve_job_binding`` with
    ``INPUT_CHANGED``. Covers every stage ``build_legacy_job_requests``
    produces, not only render, so this class of bug can't recur silently.
    """
    plan = build_nightly_plan(str(Path(__file__).resolve().parents[1]), str(AS_OF.date()))
    requests = build_legacy_job_requests(plan, tickers=(TICKER,), year_start=2025, year_end=2026)
    violations = []
    for request in requests:
        bindings = request.job.parameters.get("input_bindings") or {}
        declared = set(request.job.dependency_job_ids)
        for name, binding in bindings.items():
            binding = str(binding)
            if binding.startswith("job_"):
                dependency = binding.split("#", 1)[0]
                if dependency not in declared:
                    violations.append((request.job.kind, name, dependency))
    assert violations == []


#: The output names each producer kind actually registers in
#: ``attempt_outputs``, mirroring ``worker.py``'s ``dispatch``/
#: ``_dispatch_effect_receipt`` (worker-side) plus
#: ``effects_graph.py``'s per-kind ``extra_refs`` (coordinator-side).
#: ``ledger_export``/``engineering_gate`` carry both their receipt AND
#: their coordinator-published artifact under the bare kind name; a worker
#: output and a coordinator ``extra_refs`` output sharing one name would
#: collide in ``attempt_outputs`` (P2-5 collision fix) -- kept explicit here
#: so a future rename drifting out of sync with a binding is caught, not
#: silently resolved to the wrong artifact.
_DECLARED_OUTPUT_NAMES = {
    "legacy_finality": frozenset({"legacy_finality", "legacy_finality_coverage"}),
    "legacy_features": frozenset({"legacy_features"}),
    "legacy_score": frozenset({"legacy_score"}),
    "legacy_decisions": frozenset({"legacy_decisions"}),
    "legacy_settlement": frozenset({"legacy_settlement"}),
    "legacy_model_evidence": frozenset({"legacy_model_evidence"}),
    "legacy_render": frozenset({"legacy_render"}),
    "legacy_selfcheck": frozenset({"legacy_selfcheck"}),
    "legacy_decision_replay": frozenset({"legacy_decision_replay"}),
    "legacy_materialize": frozenset({"materialization_manifest"}),
    "decision_evidence": frozenset({"decision_plan", "decision_evidence"}),
    "ledger_export": frozenset({"ledger_export_receipt", "ledger_export"}),
    "engineering_gate": frozenset({"engineering_gate_receipt", "engineering_gate"}),
    "publication": frozenset({"publication_receipt"}),
    "backup": frozenset({"backup_receipt"}),
}


def test_every_output_binding_names_an_output_its_producer_actually_registers():
    """P2-5 collision-fix companion to
    ``test_every_job_binding_names_a_declared_dependency``: that test proves
    every ``job_<id>#name`` binding's *job* is a declared dependency; this
    proves the ``#name`` half is one the producer's *kind* actually
    registers, so a stale or colliding name is caught here instead of
    resolving to whatever else happens to share the row at runtime."""
    plan = build_nightly_plan(str(Path(__file__).resolve().parents[1]), str(AS_OF.date()))
    requests = build_legacy_job_requests(plan, tickers=(TICKER,), year_start=2025, year_end=2026,
                                         full_universe=(TICKER,))
    kind_by_job_id = {job_id_for("shadow", r.idempotency_key): r.job.kind for r in requests}
    violations = []
    for request in requests:
        bindings = request.job.parameters.get("input_bindings") or {}
        for name, binding in bindings.items():
            binding = str(binding)
            if not binding.startswith("job_"):
                continue
            dependency, _, output_name = binding.partition("#")
            producer_kind = kind_by_job_id.get(dependency)
            declared = _DECLARED_OUTPUT_NAMES.get(producer_kind, frozenset())
            if output_name not in declared:
                violations.append((request.job.kind, name, producer_kind, output_name))
    assert violations == []


def test_prior_selfcheck_ref_binds_only_when_the_caller_supplies_one():
    """P2-C08 decision 2: the optional ``prior_selfcheck.json`` binding is
    off by default (already proved by
    ``test_render_job_binds_finality_score_and_model_evidence_as_job_outputs``'s
    exact 4-key set) and, when a caller has a previous run's committed
    selfcheck artifact, is bound as a direct artifact ref admitted into the
    render job's own ``input_refs`` -- not a ``job_<id>#output`` reference,
    since no job in this plan produces it.
    """
    plan = build_nightly_plan(str(Path(__file__).resolve().parents[1]), str(AS_OF.date()))
    prior_ref = "art_prior_selfcheck_test"
    requests = build_legacy_job_requests(plan, tickers=(TICKER,), year_start=2025, year_end=2026,
                                         prior_selfcheck_ref=prior_ref)
    render_request = next(r for r in requests if r.job.kind == "legacy_render")
    bindings = render_request.job.parameters["input_bindings"]
    assert bindings["prior_selfcheck.json"] == prior_ref
    assert prior_ref in render_request.job.input_refs
    # every other binding on this job is unaffected
    assert set(bindings) == {"score.json", "model_evidence.json", "finality.json",
                             "ledger_generation.tar", "prior_selfcheck.json"}
