"""P6-2: the Tier-3 panel / Tier-4 forecast rebuild gets a real identity.

Covers ``tools/phase6_capabilities.toml`` row ``nightly-features``:
  * ``_action_features`` writes a ``features.json`` receipt binding the
    panel/tier4 files it just built.
  * ``_check_features_current`` -- the guard ``_action_score`` now calls
    unconditionally -- refuses with a DISTINCT code for each of "no receipt"
    (``FEATURES_MISSING``) and "receipt does not match disk"
    (``FEATURES_STALE``), and passes when they match.
  * The guard cannot be silently deleted: a monkeypatched sentinel proves it
    actually fires on the real ``_action_score`` call path, not merely when
    called directly.
  * The DAG wiring: "features" is a real stage between finality and score,
    with its own job kind, and ``legacy_score``'s own parameters bind the
    features job's output the way they already bind finality's.

No real Tier-3/Tier-4 rebuild runs anywhere in this file: ``rebuild.py``'s
builders are monkeypatched to write tiny synthetic files.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.ops import legacy_adapter
from engine.v2.ops.errors import OpsError
from engine.v2.ops.legacy_adapter import (
    _action_features,
    _action_score,
    _check_features_current,
    _load_features,
)
from engine.v2.ops.nightly import (
    _DAG_PARENTS,
    _DAG_STAGES,
    _action_for,
    _legacy_action,
    build_legacy_job_requests,
    build_nightly_plan,
)
from engine.v2.ops.stages import registry

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# _action_features: builds panel+tier4 (stubbed), writes the receipt.
# --------------------------------------------------------------------------


def _stub_rebuild(monkeypatch, panel_bytes=b"synthetic-panel", tier4_bytes=b"synthetic-tier4"):
    """Fakes ``engine.data.rebuild``'s two builders: each just writes a tiny
    synthetic file at the (also faked) ``engine.paths.PANEL``/``TIER4``
    location, instead of touching real data or fitting anything."""
    import engine.data.rebuild as rebuild_module
    from engine import paths as paths_module

    def fake_build_panel_table():
        paths_module.PANEL.parent.mkdir(parents=True, exist_ok=True)
        paths_module.PANEL.write_bytes(panel_bytes)
        return {"rows": 3}

    def fake_build_tier4_table(since=None):
        paths_module.TIER4.parent.mkdir(parents=True, exist_ok=True)
        paths_module.TIER4.write_bytes(tier4_bytes)
        return {"rows": 3}

    monkeypatch.setattr(rebuild_module, "build_panel_table", fake_build_panel_table)
    monkeypatch.setattr(rebuild_module, "build_tier4_table", fake_build_tier4_table)


def _fake_paths(tmp_path, monkeypatch):
    from engine import paths as paths_module

    monkeypatch.setattr(paths_module, "PANEL", tmp_path / "panel.parquet")
    monkeypatch.setattr(paths_module, "TIER4", tmp_path / "tier4.parquet")
    return paths_module


def test_action_features_writes_a_receipt_that_matches_the_files_it_built(monkeypatch, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    data_dir = tmp_path / "data"
    _fake_paths(data_dir, monkeypatch)
    _stub_rebuild(monkeypatch)

    from engine.data import store

    result = _action_features({}, staging)
    assert result["path"] == "features.json"
    document = json.loads((staging / "features.json").read_text())
    assert document["schema_version"] == "features.v1.0"
    assert document["panel_sha256"] == store.file_sha256(data_dir / "panel.parquet")
    assert document["tier4_sha256"] == store.file_sha256(data_dir / "tier4.parquet")
    assert document["panel_rows"] == 3
    assert document["tier4_rows"] == 3


# --------------------------------------------------------------------------
# _check_features_current: all three states, distinct codes, specific fields.
# --------------------------------------------------------------------------


def test_check_features_current_missing_receipt_refuses_with_a_distinct_code(tmp_path):
    with pytest.raises(OpsError) as excinfo:
        _check_features_current(tmp_path)
    assert excinfo.value.code == "FEATURES_MISSING"
    # And the raw loader used underneath raises the identical code, not the
    # generic INPUT_CHANGED ``_load_finality`` uses for its own missing case.
    with pytest.raises(OpsError) as excinfo2:
        _load_features(tmp_path)
    assert excinfo2.value.code == "FEATURES_MISSING"


def test_check_features_current_matching_receipt_proceeds(monkeypatch, tmp_path):
    (tmp_path / "features.json").write_text(json.dumps(
        {"panel_sha256": "panel-abc", "tier4_sha256": "tier4-abc"}))
    monkeypatch.setattr(legacy_adapter, "_current_features_hashes",
                        lambda: {"panel_sha256": "panel-abc", "tier4_sha256": "tier4-abc"})
    assert _check_features_current(tmp_path) is None  # does not raise


def test_check_features_current_stale_receipt_refuses_and_names_the_mismatch(monkeypatch, tmp_path):
    (tmp_path / "features.json").write_text(json.dumps(
        {"panel_sha256": "panel-abc", "tier4_sha256": "tier4-abc"}))
    # Tier 4 was rebuilt (a champion promotion) after the receipt was written;
    # the panel did not change.
    monkeypatch.setattr(legacy_adapter, "_current_features_hashes",
                        lambda: {"panel_sha256": "panel-abc", "tier4_sha256": "tier4-NEW"})
    with pytest.raises(OpsError) as excinfo:
        _check_features_current(tmp_path)
    assert excinfo.value.code == "FEATURES_STALE"
    mismatches = excinfo.value.problem.details["mismatches"]
    assert set(mismatches) == {"tier4_sha256"}  # panel matched; only tier4 is named
    assert mismatches["tier4_sha256"] == {"receipt": "tier4-abc", "current": "tier4-NEW"}


# --------------------------------------------------------------------------
# The guard cannot be silently deleted: prove it fires on the REAL
# _action_score call path, before anything else runs (no finality.json, no
# Scorer stub needed -- if the call site were removed, this would instead
# fail on the missing finality.json with INPUT_CHANGED, not the sentinel).
# --------------------------------------------------------------------------


class _Sentinel(Exception):
    pass


def test_action_score_calls_the_features_guard_before_anything_else(monkeypatch, tmp_path):
    def _raise_sentinel(root):
        raise _Sentinel("features guard reached")

    monkeypatch.setattr(legacy_adapter, "_check_features_current", _raise_sentinel)
    with pytest.raises(_Sentinel):
        _action_score({"tickers": ["FAKE"], "year_start": 2024, "year_end": 2026,
                       "session": "2026-09-12", "expected_population": ()}, tmp_path)


def test_action_score_refuses_features_missing_with_no_receipt(tmp_path):
    """End-to-end through the real (non-stubbed) guard: an empty staging root
    refuses FEATURES_MISSING, never reaching finality/scorer code at all."""
    with pytest.raises(OpsError) as excinfo:
        _action_score({"tickers": ["FAKE"], "year_start": 2024, "year_end": 2026,
                       "session": "2026-09-12", "expected_population": ()}, tmp_path)
    assert excinfo.value.code == "FEATURES_MISSING"


# --------------------------------------------------------------------------
# DAG wiring: features is a real stage, between finality and score, with its
# own job kind and a bound artifact on legacy_score.
# --------------------------------------------------------------------------


def test_features_stage_is_wired_into_the_dag():
    assert "features" in _DAG_STAGES
    assert _DAG_STAGES.index("finality") < _DAG_STAGES.index("features") < _DAG_STAGES.index("score")
    assert _DAG_PARENTS["features"] == ("finality",)
    assert "features" in _DAG_PARENTS["score"]
    assert _legacy_action("features") == "legacy_features"
    assert _action_for("features") == "legacy_features"


def test_legacy_features_is_a_registered_job_kind():
    kind = registry().get("legacy_features")
    assert kind.resource_classes == frozenset({"legacy_rebuild"})
    assert kind.checkpoint_contract == "legacy_action.v1.0"


def test_legacy_score_binds_the_features_job_output():
    from engine.v2.ops.submission import job_id_for

    plan = build_nightly_plan(str(REPO_ROOT), "2026-09-12")
    requests = build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026)
    by_kind = {r.job.kind: r for r in requests}
    assert "legacy_features" in by_kind
    score = by_kind["legacy_score"]
    features_key = by_kind["legacy_features"].idempotency_key
    features_job_id = job_id_for("shadow", features_key)
    assert features_job_id in score.job.dependency_job_ids
    assert score.job.parameters["input_bindings"]["features.json"] == \
        features_job_id + "#legacy_features"
