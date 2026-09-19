from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from engine.data.features import tier4
from tools import prepare_phase4_tier4_caches as prepare

MODEL = tier4.FeatureModel(
    model_id="implied-test",
    produces="pred_im_t1_d14",
    features=("x",),
    target="y",
    fit=lambda *_args: None,
    prepare=lambda panel: panel,
)
#: A second, non-implied producer shaped like the real `iv_crush`
#: (tier4.py:401 iv_crush_feature_model): a different model_id, a different
#: feature list, a SIGNED target with `interval_floor=None`. Exercises the
#: preparer's generic model handling rather than only the implied_t1 shape.
OTHER_MODEL = tier4.FeatureModel(
    model_id="iv-crush-test",
    produces="pred_iv_crush_30",
    features=("z", "w"),
    target="crush",
    fit=lambda *_args: None,
    prepare=lambda panel: panel,
    interval_floor=None,
)
SNAPSHOT = "a" * 64
FOLD = pd.Timestamp("2026-09-01")


def _cache(
    directory: Path,
    *,
    model: tier4.FeatureModel = MODEL,
    pools: bool = False,
    partial: bool = False,
    bad_features: bool = False,
) -> Path:
    path = directory / tier4._serving_path(model.model_id, FOLD, SNAPSHOT).name
    stored = {
        "estimator": {"weights": [1.0]},
        "model_id": model.model_id,
        "fold_start": str(FOLD.date()),
        "tier3_snapshot": SNAPSHOT,
        "features": list(MODEL.features if bad_features else model.features),
    }
    if pools:
        stored["pool_pred"] = np.array([1.0, 2.0])
        stored["pool_res"] = np.array([0.1, -0.2])
    elif partial:
        stored["pool_pred"] = np.array([1.0])
    joblib.dump(stored, path)
    return path


def test_discovery_selects_only_existing_old_matching_cache(tmp_path, monkeypatch):
    path = _cache(tmp_path)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    targets, missing = prepare.discover_targets(
        [FOLD, pd.Timestamp("2026-10-01")],
        cache_dir=tmp_path,
        model=MODEL,
        snapshot=SNAPSHOT,
    )

    assert targets == [prepare.CacheTarget(path, FOLD, prepare._sha256(path))]
    assert missing == [pd.Timestamp("2026-10-01")]


def test_discovery_skips_already_upgraded_cache(tmp_path, monkeypatch):
    _cache(tmp_path, pools=True)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    targets, missing = prepare.discover_targets(
        [FOLD], cache_dir=tmp_path, model=MODEL, snapshot=SNAPSHOT
    )

    assert targets == []
    assert missing == []


def test_discovery_refuses_partial_pool(tmp_path, monkeypatch):
    _cache(tmp_path, partial=True)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    with pytest.raises(prepare.CachePreparationError, match="partial residual pool"):
        prepare.discover_targets([FOLD], cache_dir=tmp_path, model=MODEL, snapshot=SNAPSHOT)


def test_upgrade_is_atomic_and_preserves_estimator_and_metadata(tmp_path, monkeypatch):
    path = _cache(tmp_path)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    original = joblib.load(path)
    target = prepare.CacheTarget(path, FOLD, prepare._sha256(path))

    count = prepare.upgrade_one(
        target,
        model=MODEL,
        snapshot=SNAPSHOT,
        panel_loader=lambda: pd.DataFrame({"x": [1.0], "y": [2.0]}),
        pool_builder=lambda *_args: (
            np.array([2.0, 3.0]),
            np.array([-0.25, 0.5]),
        ),
    )

    assert count == 2
    upgraded = joblib.load(path)
    assert joblib.hash(upgraded["estimator"]) == joblib.hash(original["estimator"])
    assert upgraded["model_id"] == original["model_id"]
    assert upgraded["fold_start"] == original["fold_start"]
    assert upgraded["tier3_snapshot"] == original["tier3_snapshot"]
    assert upgraded["features"] == original["features"]
    assert np.array_equal(upgraded["pool_pred"], np.array([2.0, 3.0]))
    assert np.array_equal(upgraded["pool_res"], np.array([-0.25, 0.5]))
    assert not list(tmp_path.glob(".*.tmp"))


def test_upgrade_refuses_a_race_before_replacement(tmp_path, monkeypatch):
    path = _cache(tmp_path)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    target = prepare.CacheTarget(path, FOLD, "0" * 64)

    with pytest.raises(prepare.CachePreparationError, match="changed after discovery"):
        prepare.upgrade_one(
            target,
            model=MODEL,
            snapshot=SNAPSHOT,
            panel_loader=lambda: pd.DataFrame(),
            pool_builder=lambda *_args: (np.array([]), np.array([])),
        )


def test_fold_must_be_month_boundary():
    with pytest.raises(prepare.CachePreparationError, match="fold boundary"):
        prepare._fold("2026-09-17")


def test_fold_must_not_be_missing():
    with pytest.raises(prepare.CachePreparationError, match="must not be missing"):
        prepare._fold(None)


# -- --model: mapping, default, selection -----------------------------------


def test_default_model_is_implied_t1():
    assert prepare.DEFAULT_MODEL == "implied_t1"
    assert prepare.MODEL_CHOICES[prepare.DEFAULT_MODEL] == "pred_im_t1_d14"


def test_model_choices_cover_every_tier4_producer():
    # Same producer set Scorer._serving/_crush_forecast can reach: the
    # `_chooser_frame` loop (pred_abs_move via its own forecast call,
    # pred_im_t1_d14, pred_runup_abs_move_d14) plus `_crush_forecast`
    # (pred_iv_crush_30).
    assert set(prepare.MODEL_CHOICES.values()) == set(tier4.PRODUCES)


def test_selected_models_default_dedupes_and_expands_all():
    assert prepare._selected_models([]) == [prepare.DEFAULT_MODEL]
    assert prepare._selected_models(["size", "size", "iv_crush"]) == ["size", "iv_crush"]
    assert prepare._selected_models(["size", "all"]) == list(prepare.MODEL_CHOICES)


def test_resolve_model_maps_cli_name_to_produces_like_scorer_serving(monkeypatch):
    seen = {}

    def fake_feature_model(produces, registry=None):
        seen["produces"] = produces
        return OTHER_MODEL

    monkeypatch.setattr(tier4, "feature_model", fake_feature_model)

    resolved = prepare._resolve_model("iv_crush")

    assert seen["produces"] == "pred_iv_crush_30"
    assert resolved is OTHER_MODEL


def test_resolve_model_rejects_unknown_name():
    with pytest.raises(prepare.CachePreparationError, match="not a known --model"):
        prepare._resolve_model("bogus")


def test_main_rejects_unknown_model_before_any_real_work():
    # argparse's `choices` check fires during parse_args, before main() ever
    # reaches phase4_required_folds or store.file_sha256(paths.PANEL) — so
    # this never touches real data.
    with pytest.raises(SystemExit):
        prepare.main(["--model", "bogus", "--dry-run"])


# -- generic handling of a non-implied model ---------------------------------


def test_discovery_selects_a_non_implied_old_matching_cache(tmp_path, monkeypatch):
    path = _cache(tmp_path, model=OTHER_MODEL)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    targets, missing = prepare.discover_targets(
        [FOLD], cache_dir=tmp_path, model=OTHER_MODEL, snapshot=SNAPSHOT
    )

    assert targets == [prepare.CacheTarget(path, FOLD, prepare._sha256(path))]
    assert missing == []


def test_discovery_refuses_identity_mismatch_for_non_implied_model(tmp_path, monkeypatch):
    # File lives at OTHER_MODEL's path (model_id/fold/snapshot), but its
    # stored `features` belong to a different model — exactly the corruption
    # `_validate_identity` exists to catch, now exercised on a non-implied
    # producer instead of only implied_t1.
    _cache(tmp_path, model=OTHER_MODEL, bad_features=True)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    with pytest.raises(prepare.CachePreparationError, match="identity mismatch"):
        prepare.discover_targets(
            [FOLD], cache_dir=tmp_path, model=OTHER_MODEL, snapshot=SNAPSHOT
        )


def test_upgrade_succeeds_for_a_non_implied_model_and_payload_is_unchanged(
    tmp_path, monkeypatch
):
    path = _cache(tmp_path, model=OTHER_MODEL)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    original = joblib.load(path)
    target = prepare.CacheTarget(path, FOLD, prepare._sha256(path))

    count = prepare.upgrade_one(
        target,
        model=OTHER_MODEL,
        snapshot=SNAPSHOT,
        panel_loader=lambda: pd.DataFrame({"z": [1.0], "w": [2.0], "crush": [0.1]}),
        pool_builder=lambda *_args: (np.array([5.0]), np.array([-1.0])),
    )

    assert count == 1
    upgraded = joblib.load(path)
    assert joblib.hash(upgraded["estimator"]) == joblib.hash(original["estimator"])
    assert upgraded["model_id"] == OTHER_MODEL.model_id == original["model_id"]
    assert upgraded["fold_start"] == original["fold_start"]
    assert upgraded["tier3_snapshot"] == original["tier3_snapshot"]
    assert upgraded["features"] == original["features"] == list(OTHER_MODEL.features)
    assert np.array_equal(upgraded["pool_pred"], np.array([5.0]))
    assert np.array_equal(upgraded["pool_res"], np.array([-1.0]))
    assert not list(tmp_path.glob(".*.tmp"))


# -- fold-state classification: current / old / missing ----------------------


class TestClassifyFold:
    def test_a_pool_embedded_cache_is_current(self, tmp_path, monkeypatch):
        _cache(tmp_path, pools=True)
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

        state, target = prepare._classify_fold(
            FOLD, directory=tmp_path, model=MODEL, snapshot=SNAPSHOT
        )
        assert state == "current"
        assert target is None

    def test_a_pool_less_cache_is_old(self, tmp_path, monkeypatch):
        path = _cache(tmp_path)
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

        state, target = prepare._classify_fold(
            FOLD, directory=tmp_path, model=MODEL, snapshot=SNAPSHOT
        )
        assert state == "old"
        assert target == prepare.CacheTarget(path, FOLD, prepare._sha256(path))

    def test_no_file_at_all_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

        state, target = prepare._classify_fold(
            FOLD, directory=tmp_path, model=MODEL, snapshot=SNAPSHOT
        )
        assert state == "missing"
        assert target is None


class TestReportFoldStates:
    """The listing this preparer's `--dry-run`/`--report` mode exposes: every
    requested (model, fold) pair's state, no values, no Scorer.
    """

    def test_reports_all_three_states_per_model(self, tmp_path, monkeypatch):
        current_fold = FOLD
        old_fold = pd.Timestamp("2026-10-01")
        missing_fold = pd.Timestamp("2026-11-01")
        _cache(tmp_path, pools=True)  # FOLD, current
        old_path = tmp_path / tier4._serving_path(
            MODEL.model_id, old_fold, SNAPSHOT
        ).name
        joblib.dump(
            {
                "estimator": {"weights": [1.0]},
                "model_id": MODEL.model_id,
                "fold_start": str(old_fold.date()),
                "tier3_snapshot": SNAPSHOT,
                "features": list(MODEL.features),
            },
            old_path,
        )
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
        monkeypatch.setattr(prepare, "_resolve_model", lambda name: MODEL)

        reports = prepare.report_fold_states(
            [current_fold, old_fold, missing_fold],
            model_names=["implied_t1"],
            cache_dir=tmp_path,
            snapshot=SNAPSHOT,
        )

        assert len(reports) == 1
        report = reports[0]
        assert report.current == (current_fold,)
        assert report.old == (old_fold,)
        assert report.missing == (missing_fold,)

    def test_prints_no_pool_or_estimator_values(self, tmp_path, monkeypatch, capsys):
        # A cache whose stored pool contains an easily-recognizable sentinel
        # value: if a "list" pass ever grew a stray print of stored content,
        # this sentinel appearing in stdout would catch it.
        path = tier4._serving_path(MODEL.model_id, FOLD, SNAPSHOT)
        joblib.dump(
            {
                "estimator": {"weights": [1.0]},
                "model_id": MODEL.model_id,
                "fold_start": str(FOLD.date()),
                "tier3_snapshot": SNAPSHOT,
                "features": list(MODEL.features),
                "pool_pred": np.array([424242.0]),
                "pool_res": np.array([-424242.0]),
            },
            tmp_path / path.name,
        )
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
        monkeypatch.setattr(prepare, "_resolve_model", lambda name: MODEL)

        prepare.report_fold_states(
            [FOLD], model_names=["implied_t1"], cache_dir=tmp_path, snapshot=SNAPSHOT,
        )
        assert "424242" not in capsys.readouterr().out


# -- building a genuinely missing cache ---------------------------------------


def _fake_served(model: tier4.FeatureModel, fold: pd.Timestamp) -> "tier4.ServingModel":
    return tier4.ServingModel(
        estimator={"weights": [9.0]},
        model_id=model.model_id,
        fold_start=fold,
        tier3_snapshot=SNAPSHOT,
        features=tuple(model.features),
        pool_pred=np.array([3.0, 4.0]),
        pool_res=np.array([0.5, -0.5]),
    )


class TestBuildOne:
    """`build_one` is the atomic-install counterpart of `upgrade_one`, for a
    fold with no cache file at all — see the module docstring's "missing"
    bullet for why this must fit through the SAME temp+fsync+verify+replace
    discipline rather than `serving_model`'s own direct `joblib.dump`.
    """

    def test_builds_and_installs_atomically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
        path = tmp_path / tier4._serving_path(MODEL.model_id, FOLD, SNAPSHOT).name
        assert not path.exists()

        count = prepare.build_one(
            FOLD,
            model=MODEL,
            snapshot=SNAPSHOT,
            cache_dir=tmp_path,
            panel_loader=lambda: pd.DataFrame({"x": [1.0], "y": [2.0]}),
            fit_builder=lambda fold, model, panel: _fake_served(model, fold),
        )

        assert count == 2
        installed = joblib.load(path)
        assert installed["model_id"] == MODEL.model_id
        assert installed["fold_start"] == str(FOLD.date())
        assert installed["tier3_snapshot"] == SNAPSHOT
        assert installed["features"] == list(MODEL.features)
        assert np.array_equal(installed["pool_pred"], np.array([3.0, 4.0]))
        assert np.array_equal(installed["pool_res"], np.array([0.5, -0.5]))
        assert not list(tmp_path.glob(".*.tmp"))

    def test_refuses_when_a_cache_already_exists(self, tmp_path, monkeypatch):
        _cache(tmp_path, pools=True)
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

        with pytest.raises(prepare.CachePreparationError, match="not missing"):
            prepare.build_one(
                FOLD,
                model=MODEL,
                snapshot=SNAPSHOT,
                cache_dir=tmp_path,
                panel_loader=lambda: pd.DataFrame(),
                fit_builder=lambda fold, model, panel: _fake_served(model, fold),
            )

    def test_refuses_a_fit_with_the_wrong_identity(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
        wrong_fold = pd.Timestamp("2020-01-01")

        with pytest.raises(prepare.CachePreparationError, match="identity mismatch"):
            prepare.build_one(
                FOLD,
                model=MODEL,
                snapshot=SNAPSHOT,
                cache_dir=tmp_path,
                panel_loader=lambda: pd.DataFrame(),
                # Fit result claims a DIFFERENT fold than the one requested.
                fit_builder=lambda fold, model, panel: _fake_served(model, wrong_fold),
            )
        assert not list(tmp_path.glob("*.joblib"))
        assert not list(tmp_path.glob(".*.tmp"))

    def test_writes_only_under_the_serving_directory(self, tmp_path, monkeypatch):
        # `_validate_identity` recomputes the expected path from
        # ``tier4.SERVING_DIR`` regardless of a ``cache_dir`` override, so a
        # correct caller keeps the two in sync — exactly what every real
        # invocation does, since ``build_one``'s own default IS
        # ``tier4.SERVING_DIR``. This asserts that default lands exactly
        # where ``tier4.SERVING_DIR`` points, and nowhere else (e.g. its
        # parent directory).
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setattr(tier4, "SERVING_DIR", elsewhere)

        prepare.build_one(
            FOLD,
            model=MODEL,
            snapshot=SNAPSHOT,
            panel_loader=lambda: pd.DataFrame(),
            fit_builder=lambda fold, model, panel: _fake_served(model, fold),
        )

        installed = list(elsewhere.glob("*.joblib"))
        assert len(installed) == 1
        assert not list(tmp_path.glob("*.joblib"))  # nothing landed at the parent

    def test_cache_dir_override_must_match_serving_dir_identity(self, tmp_path, monkeypatch):
        """A `cache_dir` override that disagrees with `tier4.SERVING_DIR` is
        refused rather than silently writing an identity-mismatched file —
        this is `_validate_identity`'s job, reused unchanged from
        `upgrade_one`.
        """
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
        nested = tmp_path / "nested"

        with pytest.raises(prepare.CachePreparationError, match="does not match"):
            prepare.build_one(
                FOLD,
                model=MODEL,
                snapshot=SNAPSHOT,
                cache_dir=nested,
                panel_loader=lambda: pd.DataFrame(),
                fit_builder=lambda fold, model, panel: _fake_served(model, fold),
            )


# -- every pass reuses phase4_required_folds's fold set -----------------------


class TestPhase4RequiredFoldsCoversEveryPass:
    """`phase4_required_folds` reads only forward+boundary events, so its
    docstring makes a claim: every OTHER pass either shares one of those
    folds or never touches tier4 at all. These are the static invariants
    that claim depends on — a change to any of them means the fold planner
    needs revisiting, not just this file.
    """

    def test_rescore_only_ever_changes_structure_params_or_strike(self):
        """`_rescore`'s `**changes` must never touch `as_of`/`event_date`/
        `chain_as_of` — the only fields `tier4.serving_fold` reads — or a
        pinned/strike/coarse variant could need a fold `phase4_required_
        folds` never planned for.

        `index` (added 2026-09-18 so these two passes can share ONE
        up-front `ChainIndex` — see `_rescore`'s own docstring) is a real
        DECLARED parameter of `_rescore`, not part of `**changes`: derived
        from `inspect.signature` rather than hardcoded, so this stays
        correct if `_rescore` grows another named parameter later. Only the
        VAR_KEYWORD catch-all's contents are what `serving_fold` could ever
        see, and fold-safety is exactly what this test polices.
        """
        import inspect
        import re

        from tools import capture_tier0_corpus as capture

        declared = {
            name for name, param in
            inspect.signature(capture._rescore).parameters.items()
            if param.kind != inspect.Parameter.VAR_KEYWORD
        }
        fold_safe_changes = {"structure_params", "strike"}
        allowed = declared | fold_safe_changes
        for source in (
            inspect.getsource(capture.pinned_and_strike_pass),
            inspect.getsource(capture.coarse_ladder_pass),
        ):
            calls = re.findall(r"_rescore\([^)]*\)", source, flags=re.DOTALL)
            assert calls, "expected at least one _rescore(...) call in this pass"
            for call in calls:
                kwargs = set(re.findall(r"(\w+)\s*=", call))
                changes_kwargs = kwargs - declared
                unexpected = changes_kwargs - fold_safe_changes
                assert not unexpected, (
                    f"_rescore call changes {unexpected}, not just "
                    f"{fold_safe_changes}: {call}"
                )
                assert kwargs <= allowed  # sanity: every kwarg accounted for

    def test_dyn_sv_pass_never_calls_score(self):
        """No new fold can enter through dyn_sv: it must never reach `_score`."""
        import inspect

        from tools import capture_tier0_corpus as capture

        source = inspect.getsource(capture.dyn_sv_pass)
        assert "_score(" not in source
        assert "_rescore(" not in source

    def test_research_replay_pass_never_calls_score(self):
        """Research replay prices disabled strategies through
        `engine.replay.replay_one` directly, never `Scorer.score`/`_score`
        — so it can never need a Tier-4 fold this planner does not already
        have from the STRUCTURES-keyed passes.
        """
        import inspect

        from tools import capture_tier0_corpus as capture

        source = inspect.getsource(capture.research_replay_pass)
        assert "_score(" not in source
        assert "_rescore(" not in source
        assert "replay_one(" in source


# -- main()'s --dry-run/--report mode -----------------------------------------


def test_main_report_lists_current_old_and_missing_without_touching_anything(
    tmp_path, monkeypatch, capsys
):
    current_fold = FOLD
    old_fold = pd.Timestamp("2026-10-01")
    missing_fold = pd.Timestamp("2026-11-01")
    _cache(tmp_path, pools=True)
    joblib.dump(
        {
            "estimator": {"weights": [1.0]},
            "model_id": MODEL.model_id,
            "fold_start": str(old_fold.date()),
            "tier3_snapshot": SNAPSHOT,
            "features": list(MODEL.features),
        },
        tmp_path / tier4._serving_path(MODEL.model_id, old_fold, SNAPSHOT).name,
    )
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    monkeypatch.setattr(prepare.store, "file_sha256", lambda _path: SNAPSHOT)
    monkeypatch.setattr(prepare, "_resolve_model", lambda name: MODEL)

    called = {"upgrade": False, "build": False}
    monkeypatch.setattr(
        prepare, "_run_child",
        lambda *a, **k: called.__setitem__("upgrade", True),
    )
    monkeypatch.setattr(
        prepare, "_run_build_child",
        lambda *a, **k: called.__setitem__("build", True),
    )

    code = prepare.main([
        "--dry-run",
        "--fold", str(current_fold.date()),
        "--fold", str(old_fold.date()),
        "--fold", str(missing_fold.date()),
    ])

    assert code == 0
    assert called == {"upgrade": False, "build": False}
    out = capsys.readouterr().out
    assert f"current implied_t1 {current_fold:%Y-%m}" in out
    assert f"old implied_t1 {old_fold:%Y-%m}" in out
    assert f"missing implied_t1 {missing_fold:%Y-%m}" in out
    assert "1 current, 1 old, 1 missing" in out


def test_main_report_alias_is_equivalent_to_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    monkeypatch.setattr(prepare.store, "file_sha256", lambda _path: SNAPSHOT)
    monkeypatch.setattr(prepare, "_resolve_model", lambda name: MODEL)

    code = prepare.main(["--report", "--fold", str(FOLD.date())])

    assert code == 0
    assert f"missing implied_t1 {FOLD:%Y-%m}" in capsys.readouterr().out


def test_main_builds_missing_and_upgrades_old_when_not_a_dry_run(tmp_path, monkeypatch):
    old_fold = pd.Timestamp("2026-10-01")
    missing_fold = pd.Timestamp("2026-11-01")
    joblib.dump(
        {
            "estimator": {"weights": [1.0]},
            "model_id": MODEL.model_id,
            "fold_start": str(old_fold.date()),
            "tier3_snapshot": SNAPSHOT,
            "features": list(MODEL.features),
        },
        tmp_path / tier4._serving_path(MODEL.model_id, old_fold, SNAPSHOT).name,
    )
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    monkeypatch.setattr(prepare.store, "file_sha256", lambda _path: SNAPSHOT)
    monkeypatch.setattr(prepare, "_resolve_model", lambda name: MODEL)

    seen = {"upgraded": [], "built": []}
    monkeypatch.setattr(
        prepare, "_run_child",
        lambda target, produces, snapshot: seen["upgraded"].append(target.fold),
    )
    monkeypatch.setattr(
        prepare, "_run_build_child",
        lambda fold, model, produces, snapshot: seen["built"].append(fold),
    )

    code = prepare.main([
        "--fold", str(old_fold.date()),
        "--fold", str(missing_fold.date()),
    ])

    assert code == 0
    assert seen["upgraded"] == [old_fold]
    assert seen["built"] == [missing_fold]
