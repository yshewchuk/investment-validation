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
from engine.dashboard.render import (
    build_health,
    build_meta,
    freshness_summary,
    quota_state,
    render_bundle,
    size_model_mae_from_ledger,
)
from engine.score import ScoreResult
from engine.v2.ledger.export import export_generation
from engine.v2.ops.errors import OpsError
from engine.v2.ops.legacy_adapter import _action_render
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.render_inputs import (
    ABSENT_STAGES,
    absent_stage_flags,
    assemble_scores,
    bundle_content_hash,
    diff_bundles,
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

    from engine import paths

    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(tmp_path / "job" / "legacy"))
    importlib.reload(paths)
    yield
    monkeypatch.delenv("INVESTING_PLAN_ROOT", raising=False)
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


_PARAMETERS = {"session": str(AS_OF.date()), "tickers": [TICKER], "year_start": 2025,
               "year_end": 2026, "horizon_days": 35, "alt_strikes": 1}


def _legacy_way_bundle(out_dir, *, scores, panel, trades, registry, finality):
    """``engine/dashboard/nightly.py:1580-1622``, called directly on the same
    scores/panel/trades/registry the v2 action used, and the same staged
    legacy tree (INVESTING_PLAN_ROOT), so it reads the same ledger generation.
    """
    meta = build_meta(scores, as_of=AS_OF, horizon_days=35, fill_alpha=0.5, alt_strikes=1,
                      freshness=freshness_summary(AS_OF), quota=quota_state(), registry=registry)
    meta["execution_clock"] = {"requested_as_of": str(AS_OF.date()),
                               "resolved_as_of": str(AS_OF.date()), "finality": finality}
    health = build_health(as_of=AS_OF, size_mae=size_model_mae_from_ledger(panel=panel))
    return render_bundle(scores, out_dir, as_of=AS_OF, horizon_days=35, fill_alpha=0.5,
                         alt_strikes=1, panel=panel, trades=trades, meta=meta, health=health,
                         flags=absent_stage_flags(), registry=registry)


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
    _legacy_way_bundle(bundle_legacy, scores=scores, panel=panel, trades=_trades(),
                       registry=_FakeRegistry(), finality=finality)

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
# DAG
# --------------------------------------------------------------------------


def test_render_job_binds_finality_score_and_model_evidence_as_job_outputs():
    plan = build_nightly_plan("/root/investing-plan", str(AS_OF.date()))
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
    plan = build_nightly_plan("/root/investing-plan", str(AS_OF.date()))
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
