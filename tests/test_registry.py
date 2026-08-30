"""The model registry.

Every check here corresponds to a way a model silently becomes the wrong model:
retrained in place while its metrics stay put, promoted twice, scored on a
permuted feature vector, or trained on a feature no live event can supply.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from engine.models import registry as reg


class ConstantModel:
    def __init__(self, value: float = 1.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def artifact(**kwargs) -> reg.ModelArtifact:
    base = dict(
        model=ConstantModel(),
        role="size",
        features=("a", "b"),
        residuals=np.array([-1.0, 0.0, 1.0, 2.0]),
        target="abs_move",
        seed=1,
    )
    base.update(kwargs)
    return reg.ModelArtifact(**base)


def entry(**kwargs) -> reg.RegistryEntry:
    base = dict(
        id="m1",
        role="size",
        strategy="*",
        artifact="data/models/m1.joblib",
        artifact_sha256="deadbeef",
        features=["a", "b"],
        target="abs_move",
        train_window="<=2022",
    )
    base.update(kwargs)
    return reg.RegistryEntry(**base)


class TestModelArtifact:
    def test_rejects_an_unknown_role(self):
        with pytest.raises(ValueError, match="unknown role"):
            artifact(role="oracle")

    def test_drops_non_finite_residuals(self):
        art = artifact(residuals=np.array([1.0, np.nan, np.inf, 2.0]))
        assert art.residuals.tolist() == [1.0, 2.0]

    def test_residual_draws_come_from_the_stored_distribution(self):
        art = artifact()
        draws = art.residual_draws(500, np.random.default_rng(0))
        assert draws.shape == (500,)
        assert set(np.unique(draws)).issubset(set(art.residuals.tolist()))

    def test_draws_are_reproducible_for_a_seed(self):
        art = artifact()
        a = art.residual_draws(50, np.random.default_rng(7))
        b = art.residual_draws(50, np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_no_residuals_is_an_error_not_a_zero(self):
        art = artifact(residuals=np.array([]))
        with pytest.raises(reg.RegistryError, match="no residuals"):
            art.residual_draws(10, np.random.default_rng(0))

    def test_roundtrips_through_joblib(self, tmp_path):
        path = tmp_path / "models" / "m.joblib"
        digest = artifact().save(path)
        assert path.exists() and len(digest) == 64
        loaded = reg.load_artifact(path)
        assert loaded.features == ("a", "b")
        assert loaded.predict(np.zeros((3, 2))).tolist() == [1.0, 1.0, 1.0]

    def test_refuses_to_load_something_that_is_not_an_artifact(self, tmp_path):
        import joblib

        path = tmp_path / "junk.joblib"
        joblib.dump({"not": "an artifact"}, path)
        with pytest.raises(reg.RegistryError, match="does not hold"):
            reg.load_artifact(path)


class TestRegistryInvariants:
    def test_rejects_duplicate_ids(self):
        with pytest.raises(reg.RegistryError, match="duplicate registry id"):
            reg.Registry(entries=[entry(), entry()])

    def test_rejects_two_champions_for_one_role(self):
        with pytest.raises(reg.RegistryError, match="more than one champion"):
            reg.Registry(
                entries=[
                    entry(id="a", champion=True),
                    entry(id="b", champion=True),
                ]
            )

    def test_allows_one_champion_per_strategy(self):
        registry = reg.Registry(
            entries=[
                entry(id="g1", role="gate", strategy="STR-THRU", champion=True),
                entry(id="g2", role="gate", strategy="STR-RUNUP", champion=True),
            ]
        )
        assert registry.champion("gate", "STR-THRU").id == "g1"

    def test_rejects_an_empty_feature_list(self):
        with pytest.raises(reg.RegistryError, match="empty feature list"):
            entry(features=[])

    def test_rejects_an_unknown_role(self):
        with pytest.raises(reg.RegistryError, match="unknown role"):
            entry(role="oracle")


class TestChampionLookup:
    def test_strategy_specific_beats_the_wildcard(self):
        registry = reg.Registry(
            entries=[
                entry(id="shared", role="gate", strategy="*", champion=True),
                entry(id="thru", role="gate", strategy="STR-THRU", champion=True),
            ]
        )
        assert registry.champion("gate", "STR-THRU").id == "thru"
        assert registry.champion("gate", "STR-RUNUP").id == "shared"

    def test_missing_champion_says_how_to_make_one(self):
        with pytest.raises(reg.RegistryError, match="train one with"):
            reg.Registry(entries=[]).champion("size")

    def test_has_champion_does_not_raise(self):
        assert reg.Registry(entries=[]).has_champion("size") is False

    def test_a_non_champion_entry_is_not_returned(self):
        registry = reg.Registry(entries=[entry(champion=False)])
        assert registry.has_champion("size") is False


class TestIntegrityChecks:
    @pytest.fixture
    def built(self, tmp_path):
        path = tmp_path / "models" / "m1.joblib"
        digest = artifact().save(path)
        e = entry(artifact=str(path), artifact_sha256=digest, champion=True)
        return reg.Registry(entries=[e]), e, path

    def test_loads_a_consistent_artifact(self, built):
        registry, e, _ = built
        assert registry.load(e).role == "size"

    def test_refuses_a_retrained_artifact_the_manifest_misdescribes(self, built):
        """The expensive kind of stale: metrics that no longer describe the file."""
        registry, e, path = built
        artifact(model=ConstantModel(99.0)).save(path)  # retrain in place
        with pytest.raises(reg.RegistryError, match="hash mismatch"):
            registry.load(e)

    def test_refuses_a_feature_list_that_disagrees(self, built):
        registry, e, path = built
        e.features = ["b", "a"]  # permuted — same names, different meaning
        with pytest.raises(reg.RegistryError, match="feature list disagrees"):
            registry.load(e)

    def test_refuses_a_role_that_disagrees(self, built):
        registry, e, _ = built
        e.role = "gate"
        with pytest.raises(reg.RegistryError, match="role"):
            registry.load(e)

    def test_missing_artifact_says_how_to_rebuild(self, built):
        registry, e, path = built
        path.unlink()
        with pytest.raises(reg.RegistryError, match="train_all"):
            registry.load(e)

    def test_verify_false_skips_the_hash_check(self, built):
        registry, e, path = built
        artifact(model=ConstantModel(99.0)).save(path)
        assert registry.load(e, verify=False).role == "size"


class TestValidate:
    def test_flags_an_unservable_feature(self):
        registry = reg.Registry(
            entries=[entry(features=["ema12r_abs", "implied_move"], champion=True)]
        )
        problems = registry.validate(check_artifacts=False)
        assert any("could never be served live" in p for p in problems)

    def test_flags_a_champion_gate_without_a_threshold(self):
        registry = reg.Registry(
            entries=[entry(id="g", role="gate", strategy="STR-THRU", champion=True)]
        )
        problems = registry.validate(check_artifacts=False)
        assert any("no threshold" in p for p in problems)

    def test_a_clean_registry_reports_nothing(self):
        registry = reg.Registry(
            entries=[entry(features=["ema12r_abs", "or_implied"], champion=True)]
        )
        assert registry.validate(check_artifacts=False) == []


class TestPersistence:
    def test_roundtrips_through_json(self, tmp_path):
        path = tmp_path / "registry.json"
        reg.Registry(entries=[entry(champion=True)], path=path).save()
        loaded = reg.load_registry(path)
        assert loaded.champion("size").id == "m1"

    def test_missing_registry_is_empty_not_fatal(self, tmp_path):
        assert reg.load_registry(tmp_path / "absent.json").entries == []

    def test_missing_registry_can_be_made_fatal(self, tmp_path):
        with pytest.raises(reg.RegistryError):
            reg.load_registry(tmp_path / "absent.json", missing_ok=False)

    def test_registering_a_champion_demotes_the_incumbent_atomically(self, tmp_path):
        path = tmp_path / "registry.json"
        reg.Registry(entries=[entry(id="old", champion=True)], path=path).save()
        reg.register(entry(id="new", champion=True), path=path)
        loaded = reg.load_registry(path)
        assert loaded.champion("size").id == "new"
        assert loaded.get("old").champion is False
        # The invariant held on disk at every point, never two champions.
        doc = json.loads(path.read_text())
        assert sum(m["champion"] for m in doc["models"]) == 1

    def test_registering_replaces_an_entry_of_the_same_id(self, tmp_path):
        path = tmp_path / "registry.json"
        reg.Registry(entries=[entry(id="m1", train_window="old")], path=path).save()
        reg.register(entry(id="m1", train_window="new"), path=path)
        loaded = reg.load_registry(path)
        assert len(loaded.entries) == 1
        assert loaded.get("m1").train_window == "new"
