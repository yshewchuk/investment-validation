"""Regression tests for two 2026-09-25 defects that let a champion evidence
reason string reach the dashboard's publish-time secret scan carrying an
absolute local path:

1. ``engine.models.training.chooser`` dynamically loads
   ``experiments/EXP-169_menu7prime_confirmation/run.py`` (and its own
   transitive dynamic-load chain), invisible to
   ``engine.v2.ops.fingerprints``'s static AST import scan. A worker's
   private code snapshot built from ``worker_source_manifest`` must declare
   all four files so the chooser's training-set rebuild can find them
   wherever the code executes.
2. ``engine.dashboard.model_evidence`` must never embed an absolute local
   path in a champion's ``reason`` string, regardless of why a rebuild
   failed -- defense in depth for (1) and for any other cause of the same
   shape.
"""
from __future__ import annotations

from pathlib import Path

from engine import paths
from engine.dashboard.model_evidence import _sanitize_reason
from engine.v2.ops.fingerprints import CODE_ASSET_FILES, worker_source_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]

_CHOOSER_EXPERIMENT_FILES = (
    "experiments/EXP-169_menu7prime_confirmation/run.py",
    "experiments/EXP-161_dynamic_short_vol_neural_structure_picker/run.py",
    "experiments/EXP-163_two_headed_level_and_deviation_dyn_sv/margin163.py",
    "experiments/EXP-134_priced_right_funded_and_held_structure_s/margin.py",
)


def test_chooser_experiment_chain_is_declared_as_a_code_asset():
    for relative in _CHOOSER_EXPERIMENT_FILES:
        assert relative in CODE_ASSET_FILES, (
            f"{relative} is dynamically loaded by the DYN-SV chooser's "
            "training-set rebuild but missing from CODE_ASSET_FILES -- a "
            "worker's private code snapshot will not carry it"
        )


def test_worker_source_manifest_carries_the_real_chooser_experiment_chain():
    manifest = worker_source_manifest(REPO_ROOT)
    for relative in _CHOOSER_EXPERIMENT_FILES:
        assert relative in manifest, f"{relative} missing from worker_source_manifest"


def test_chooser_build_dataset_resolves_the_real_experiment_from_this_checkout():
    """The plain (non-worker) path this bug ticket is actually about: a
    legacy nightly running directly against the repo checkout must resolve
    engine.models.training.chooser's experiment file to THIS repo, not
    raise FileNotFoundError."""
    from engine.models.training import chooser

    assert chooser._EXPERIMENT.is_file()
    assert chooser._EXPERIMENT == (
        REPO_ROOT / "experiments/EXP-169_menu7prime_confirmation/run.py"
    )


def test_sanitize_reason_redacts_a_worker_snapshot_path():
    leaky = (
        "rebuilding the training set raised FileNotFoundError: [Errno 2] "
        "No such file or directory: '/root/phase2-shadow-ops/code/"
        "c9260643520240ff19379c70483475b34c31953bb0be3b79946351c00e41f11d/"
        "experiments/EXP-169_menu7prime_confirmation/run.py'"
    )
    cleaned = _sanitize_reason(leaky)
    assert "/root/" not in cleaned
    assert "phase2-shadow-ops" not in cleaned
    assert "FileNotFoundError" in cleaned  # still informative


def test_sanitize_reason_relativizes_a_repo_root_path():
    leaky = f"rebuilding the training set raised RuntimeError: bad file {paths.ROOT}/experiments/x.py"
    cleaned = _sanitize_reason(leaky)
    assert "/root/" not in cleaned
    assert "<repo>/experiments/x.py" in cleaned


def test_sanitize_reason_is_a_noop_on_clean_text():
    clean = "rebuilding the training set raised ValueError: bad target column"
    assert _sanitize_reason(clean) == clean
