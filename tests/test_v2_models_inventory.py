"""P5-1: the generated model-release inventory (engine.v2.models.inventory).

Tier 0 style: seconds, frozen/synthetic fixtures for the negative controls
(no data/ dependency), plus a handful of assertions against the REAL,
git-tracked ``engine/models/registry.json`` for role coverage — that file is
metadata (ids, roles, feature names, hashes), never a data value, and it is
present in every worktree unlike ``data/models/*.joblib``.
"""
from __future__ import annotations

from dataclasses import replace

import joblib
import numpy as np
import pytest

from engine.models import registry as legacy_registry
from engine.v2.models import ModelReleaseRefusal, release_issues, require_complete_release
from engine.v2.models.inventory import (
    current_release_inventory,
    non_model_state_inventory,
    served_roles,
    tier4_fold_coverage,
)


class _DummyModel:
    def predict(self, X):
        return [0.0 for _ in X]


def _save_artifact(path, *, role="size", features=("f1", "f2"), residuals=(),
                    residual_buckets=None, target="t"):
    artifact = legacy_registry.ModelArtifact(
        model=_DummyModel(), role=role, features=features,
        residuals=np.asarray(residuals, dtype=float), target=target,
        residual_buckets=residual_buckets,
    )
    return artifact.save(path)


def _entry(*, id_, role, path, sha, features, threshold=None, strategy="*"):
    return legacy_registry.RegistryEntry(
        id=id_, role=role, strategy=strategy, artifact=str(path),
        artifact_sha256=sha, features=list(features), target="t",
        train_window="w", champion=True, threshold=threshold,
    )


def _one_entry_release(tmp_path, **kwargs):
    path = tmp_path / f"{kwargs.get('id_', 'm')}.joblib"
    sha = _save_artifact(
        path, role=kwargs.get("role", "size"), features=kwargs.get("features", ("f1",)),
        residuals=kwargs.get("residuals", (1.0, 2.0, 3.0)),
    )
    entry = _entry(
        id_=kwargs.get("id_", "m"), role=kwargs.get("role", "size"), path=path, sha=sha,
        features=kwargs.get("features", ("f1",)), threshold=kwargs.get("threshold"),
    )
    reg = legacy_registry.Registry(entries=[entry])
    return current_release_inventory(reg)


# --------------------------------------------------------------------------
# negative controls — existing refusal types, exercised through this module
# --------------------------------------------------------------------------


def test_missing_residual_member_refuses(tmp_path):
    release, _ = _one_entry_release(tmp_path, id_="size_a", role="size", residuals=())
    codes = {issue.code for issue in release_issues(release)}
    assert "MISSING_RESIDUAL_MEMBER" in codes
    with pytest.raises(ModelReleaseRefusal) as error:
        require_complete_release(release)
    assert error.value.code == "MODEL_RELEASE_INCOMPLETE"


def test_missing_transform_member_refuses_if_a_recipe_required_one(tmp_path):
    """No current recipe separates a transform artifact (see the module's
    ``_required_kinds`` docstring) — a target/preprocessing transform is
    embedded in the estimator's own pickled class. This proves the SAME
    existing ``MISSING_TRANSFORM_MEMBER`` refusal still fires through a
    release this module built, the moment a binding declares one required.
    """
    release, _ = _one_entry_release(tmp_path, id_="size_b", role="size")
    binding = replace(release.bindings[0],
                       required_member_kinds=(*release.bindings[0].required_member_kinds, "transform"))
    release = replace(release, bindings=(binding,))
    codes = {issue.code for issue in release_issues(release)}
    assert "MISSING_TRANSFORM_MEMBER" in codes


def test_incompatible_feature_order_refuses(tmp_path):
    release, _ = _one_entry_release(tmp_path, id_="size_c", role="size", features=("f1", "f2"))
    mutated = replace(release.bindings[0], ordered_features=("f2", "f1"))
    release = replace(release, bindings=(mutated,))
    codes = {issue.code for issue in release_issues(release)}
    assert "FEATURE_ORDER_MISMATCH" in codes


def test_unknown_clock_refuses(tmp_path):
    release, _ = _one_entry_release(tmp_path, id_="size_d", role="size")
    binding = replace(release.bindings[0], clock_id="unknown-clock")
    requirement = replace(release.requirements[0], clock_id="unknown-clock")
    release = replace(release, bindings=(binding,), requirements=(requirement,))
    codes = {issue.code for issue in release_issues(release)}
    assert "UNKNOWN_CLOCK" in codes


def test_artifact_hash_drift_is_detected(tmp_path):
    path = tmp_path / "size_e.joblib"
    sha = _save_artifact(path, role="size", features=("f1",), residuals=(1.0, 2.0))
    entry = _entry(id_="size_e", role="size", path=path, sha="sha256:" + "0" * 64, features=("f1",))
    reg = legacy_registry.Registry(entries=[entry])
    _, issues = current_release_inventory(reg)
    assert any(issue.code == "ARTIFACT_HASH_DRIFT" for issue in issues)


def test_missing_artifact_file_is_detected(tmp_path):
    entry = _entry(id_="size_f", role="size", path=tmp_path / "absent.joblib",
                    sha="sha256:" + "0" * 64, features=("f1",))
    reg = legacy_registry.Registry(entries=[entry])
    release, issues = current_release_inventory(reg)
    assert any(issue.code == "MISSING_ARTIFACT_FILE" for issue in issues)
    assert release.artifacts[0].members == ()


def test_gate_threshold_member_present_when_registered(tmp_path):
    release, _ = _one_entry_release(tmp_path, id_="gate_a", role="gate", threshold=0.05)
    kinds = {m.kind for m in release.artifacts[0].members}
    assert "threshold" in kinds


def test_chooser_role_requires_only_estimator(tmp_path):
    release, _ = _one_entry_release(tmp_path, id_="chooser_a", role="chooser")
    assert release.bindings[0].required_member_kinds == ("estimator",)
    assert release_issues(release) == ()


# --------------------------------------------------------------------------
# every model role the legacy scorer serves — derived from code
# --------------------------------------------------------------------------


def test_served_roles_matches_registry_roles():
    assert set(served_roles()) == set(legacy_registry.ROLES)


def test_served_roles_is_derived_from_source_not_hand_typed():
    fake_source = 'x = "size"\ny = "gate"\n'
    assert served_roles(fake_source) == ("size", "gate")


def test_every_served_role_is_bound_in_the_real_registry():
    """Uses the real, git-tracked registry.json (metadata, no data/ needed)."""
    reg = legacy_registry.load_registry()
    release, _ = current_release_inventory(reg)
    bound_roles = {b.role for b in release.bindings}
    assert bound_roles == set(legacy_registry.ROLES) == set(served_roles())


# --------------------------------------------------------------------------
# Tier-4 monthly fold coverage
# --------------------------------------------------------------------------


def test_tier4_fold_coverage_flags_present_and_incompatible(tmp_path):
    good = {
        "estimator": _DummyModel(), "model_id": "size_v1_4", "fold_start": "2026-09-01",
        "tier3_snapshot": "a" * 40, "features": ["f1", "f2"],
        "pool_pred": np.array([1.0, 2.0]), "pool_res": np.array([0.1, 0.2]),
    }
    joblib.dump(good, tmp_path / "size_v1_4_202609_a1b2c3d4e5f6.joblib")
    bad = {"estimator": _DummyModel(), "model_id": "size_v1_4", "fold_start": "2026-08-01",
           "tier3_snapshot": "b" * 40, "features": ["f1"]}
    joblib.dump(bad, tmp_path / "size_v1_4_202608_b1b2c3d4e5f6.joblib")

    entries = {e.fold_start: e for e in tier4_fold_coverage("size_v1_4", directory=tmp_path)}
    assert entries["202609"].status == "present"
    assert entries["202609"].model_id == "size_v1_4"
    assert entries["202609"].snapshot12 == "a1b2c3d4e5f6"
    assert entries["202608"].status == "incompatible"
    assert "pool_pred" in entries["202608"].reason or "pool_res" in entries["202608"].reason


def test_tier4_fold_coverage_empty_directory_is_empty(tmp_path):
    assert tier4_fold_coverage("size_v1_4", directory=tmp_path / "missing") == ()


# --------------------------------------------------------------------------
# non-model serving state (payoff, recalibration, pools) — no artifact today
# --------------------------------------------------------------------------


def test_non_model_state_reports_presence_against_a_root(tmp_path):
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)
    (features_dir / "chooser_analog_pool.parquet").write_bytes(b"x")
    (features_dir / "pnl_sim_history.parquet").write_bytes(b"y")

    entries = {e.name: e for e in non_model_state_inventory(root=tmp_path)}
    assert entries["chooser_analog_pool"].status == "present_unversioned"
    assert entries["trailing_pnl_cutoff"].status == "source_present"
    assert entries["payoff_line:STR-THRU"].status == "not_persisted"
    assert entries["payoff_line:STR-THRU"].backing_file is None
    assert entries["paired_residual_pool"].status == "missing_source"
    assert entries["recalibration_map"].status == "missing_source"
