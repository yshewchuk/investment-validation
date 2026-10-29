"""#1102 slice C: the pool-refresh sidecar's bootstrap-cycle submission.

Synthetic only, reusing ``test_pool_nightly_refresh``'s fake db/planners: a
``NONE`` (missing-pointer) or ``NO_POOL_CATALOG`` prior runs the twelve-phase
bootstrap training, the ``mode: "bootstrap"`` stage with the four environment
inputs, and the guarded promote; a ``DAMAGED`` prior refuses. The prior is
classified only on the tick that starts a cycle, and recipe-job bookkeeping
files never reach the stage's state refs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from checks.phase5_release_assembly import (
    PRIOR_DAMAGED,
    PRIOR_NO_POOL_CATALOG,
    PRIOR_NONE,
    PRIOR_USABLE,
    PriorClass,
)
from engine.v2.ops import pool_refresh
from engine.v2.ops.errors import OpsError
from tests.v2.ops.test_pool_nightly_refresh import (
    MANIFEST,
    PRIOR,
    SESSION,
    TAG,
    _add_score,
    _add_training,
    _db,
    _submit,
    _succeeded_training,
    _wire,
    _write_pointer,
)

NONE_CLASS = PriorClass(PRIOR_NONE, "no prior release was requested")
NO_CATALOG_CLASS = PriorClass(PRIOR_NO_POOL_CATALOG, "the prior release has no catalog")
DAMAGED_CLASS = PriorClass(PRIOR_DAMAGED,
                           "the prior catalog could not be read or validated")

_ENV_VALUES = {
    "BOOTSTRAP_TIER4_DIR": "/srv/tier4",
    "BOOTSTRAP_PANEL_CATALOG": "/srv/panel/catalog.json",
    "BOOTSTRAP_PANEL_SNAPSHOT_ID": "snap-1",
    "BOOTSTRAP_PANEL_ARTIFACT_ROOT": "/srv/panel/artifacts",
}


def _env(monkeypatch):
    for name, value in _ENV_VALUES.items():
        monkeypatch.setenv(name, value)


def _bootstrap_training_succeeded(conn, *, prior="none", failed=None):
    for phase in pool_refresh.BOOTSTRAP_TRAINING_PHASES:
        if phase == failed:
            _add_training(conn, phase, prior=prior, state="failed")
        else:
            _succeeded_training(conn, phase, [f"training/{phase}.json"], prior=prior)


def _by_phase(rec):
    return {key.split(":")[3]: kwargs
            for (_plan, key), kwargs in zip(rec.requests, rec.training)}


# --------------------------------------------------------------------------
# 1. a NONE prior submits the twelve training jobs, keyed with "none"
# --------------------------------------------------------------------------


def test_none_prior_submits_the_twelve_training_jobs(monkeypatch, tmp_path, capsys):
    _env(monkeypatch)
    rec = _wire(monkeypatch, tmp_path, pointer=None, prior_class=NONE_CLASS)
    conn = _db()
    _add_score(conn)

    assert _submit(conn, tmp_path) == ["receipt"] * 12
    assert [key for _plan, key in rec.requests] == [
        pool_refresh.phase_key(SESSION, phase, "none", TAG)
        for phase in pool_refresh.BOOTSTRAP_TRAINING_PHASES]

    expected = {
        "train-paired": {"mode": "state", "state": "paired_residual_pool",
                         "cutoffs": (SESSION,)},
        "train-board": {"mode": "board_analog", "alpha": pool_refresh.FILL_ALPHA,
                        "cutoffs": (SESSION,)},
        "train-trailing": {"mode": "trailing_cutoff", "cutoffs": (SESSION,)},
        "train-driver-size": {"mode": "state", "state": "driver_residual_pool:size"},
        "train-driver-implied_t1": {"mode": "state",
                                    "state": "driver_residual_pool:implied_t1"},
        "train-driver-runup_move": {"mode": "state",
                                    "state": "driver_residual_pool:runup_move"},
        "train-chooser": {"mode": "state", "state": "chooser_analog_pool",
                          "cutoffs": (SESSION,)},
        "train-line-thru": {"mode": "recipe",
                            "recipe": "payoff_line:STR-THRU:calibration",
                            "alpha": pool_refresh.FILL_ALPHA, "cutoffs": (SESSION,)},
        "train-map-thru": {"mode": "recipe",
                           "recipe": "recalibration_map:STR-THRU:calibration",
                           "alpha": pool_refresh.FILL_ALPHA, "cutoffs": (SESSION,)},
        "train-surface-runup": {"mode": "recipe",
                                "recipe": "payoff_surface:STR-RUNUP:calibration",
                                "alpha": pool_refresh.FILL_ALPHA, "cutoffs": (SESSION,)},
        "train-line-runup": {"mode": "recipe",
                             "recipe": "payoff_line:STR-RUNUP:calibration",
                             "alpha": pool_refresh.FILL_ALPHA, "cutoffs": (SESSION,)},
        "train-map-runup": {"mode": "recipe",
                            "recipe": "recalibration_map:STR-RUNUP:calibration",
                            "alpha": pool_refresh.FILL_ALPHA, "cutoffs": (SESSION,)},
    }
    built = _by_phase(rec)
    assert set(built) == set(expected)
    for phase, kwargs in expected.items():
        for field, value in kwargs.items():
            assert built[phase].get(field, ()) == value, phase
        assert built[phase]["manifest_ref"] == MANIFEST, phase
    # driver pools take no cutoff
    assert "cutoffs" not in built["train-driver-size"]

    err = capsys.readouterr().err
    assert "prior class NONE" in err
    assert "bootstrap submitted 12/12 training jobs" in err


# --------------------------------------------------------------------------
# 2. unset/blank environment inputs refuse before any training submit
# --------------------------------------------------------------------------


def test_unset_bootstrap_inputs_refuse_with_no_submission(monkeypatch, tmp_path):
    for name in _ENV_VALUES:
        monkeypatch.delenv(name, raising=False)
    rec = _wire(monkeypatch, tmp_path, pointer=None, prior_class=NONE_CLASS)
    conn = _db()
    _add_score(conn)
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "bootstrap panel/tier4 inputs are not configured"
    assert rec.submits == [] and rec.training == []


def test_blank_bootstrap_input_refuses_with_no_submission(monkeypatch, tmp_path):
    _env(monkeypatch)
    monkeypatch.setenv("BOOTSTRAP_PANEL_SNAPSHOT_ID", "   ")
    rec = _wire(monkeypatch, tmp_path, pointer=None, prior_class=NONE_CLASS)
    conn = _db()
    _add_score(conn)
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "bootstrap panel/tier4 inputs are not configured"
    assert rec.submits == []


# --------------------------------------------------------------------------
# 3. all twelve succeeded -> the bootstrap stage
# --------------------------------------------------------------------------


def test_second_tick_submits_the_bootstrap_stage(monkeypatch, tmp_path):
    _env(monkeypatch)
    rec = _wire(monkeypatch, tmp_path, pointer=None, prior_class=NONE_CLASS)
    conn = _db()
    _add_score(conn)
    _bootstrap_training_succeeded(conn)

    assert _submit(conn, tmp_path) == ["receipt"]
    assert len(rec.stage) == 1
    stage = rec.stage[0]
    assert stage["mode"] == "bootstrap"
    assert stage["bootstrap"] == {
        "tier4_dir": "/srv/tier4",
        "panel_catalog": "/srv/panel/catalog.json",
        "panel_snapshot_id": "snap-1",
        "panel_artifact_root": "/srv/panel/artifacts",
    }
    assert stage["prior_release_id"] is None
    assert stage["release_id"] == f"pools-{SESSION}-{TAG}"
    assert stage["cutoff"] == SESSION
    assert stage["release_root"] == str(tmp_path)
    assert stage["manifest_ref"] == MANIFEST
    assert set(stage["state_refs"]) == {f"{phase}__{phase}.json"
                                        for phase in pool_refresh.BOOTSTRAP_TRAINING_PHASES}
    assert rec.requests[-1][1] == pool_refresh.phase_key(SESSION, "stage", "none", TAG)


def test_failed_bootstrap_training_blocks_later_phases(monkeypatch, tmp_path):
    _env(monkeypatch)
    rec = _wire(monkeypatch, tmp_path, pointer=None, prior_class=NONE_CLASS)
    conn = _db()
    _add_score(conn)
    _bootstrap_training_succeeded(conn, failed="train-chooser")

    assert _submit(conn, tmp_path) == []
    assert rec.stage == [] and rec.promote == [] and rec.submits == []


def test_existing_bootstrap_keys_are_not_resubmitted(monkeypatch, tmp_path):
    _env(monkeypatch)
    rec = _wire(monkeypatch, tmp_path, pointer=None, prior_class=NONE_CLASS)
    conn = _db()
    _add_score(conn)
    _add_training(conn, "train-chooser", prior="none", state="queued")

    receipts = _submit(conn, tmp_path)
    assert receipts == ["receipt"] * 11
    keys = [key for _plan, key in rec.requests]
    assert pool_refresh.phase_key(SESSION, "train-chooser", "none", TAG) not in keys
    assert len(keys) == len(pool_refresh.BOOTSTRAP_TRAINING_PHASES) - 1


# --------------------------------------------------------------------------
# 4. the guarded promote
# --------------------------------------------------------------------------


def test_bootstrap_promote_for_none_requires_the_empty_store(monkeypatch, tmp_path):
    _env(monkeypatch)
    rec = _wire(monkeypatch, tmp_path, pointer=None, prior_class=NONE_CLASS)
    conn = _db()
    _add_score(conn)
    _bootstrap_training_succeeded(conn)
    _add_training(conn, "stage", prior="none", state="succeeded")

    assert _submit(conn, tmp_path) == ["receipt"]
    assert len(rec.promote) == 1
    promote = rec.promote[0]
    assert promote["require_empty_store"] is True
    assert "expected_previous_release_id" not in promote
    assert promote["release_id"] == f"pools-{SESSION}-{TAG}"
    assert promote["release_root"] == str(Path(tmp_path) / "deployment")
    assert rec.requests[-1][1] == pool_refresh.phase_key(SESSION, "promote", "none", TAG)


def test_bootstrap_promote_for_no_pool_catalog_carries_the_expected_id(monkeypatch,
                                                                       tmp_path):
    _env(monkeypatch)
    rec = _wire(monkeypatch, tmp_path, pointer=PRIOR, prior_class=NO_CATALOG_CLASS)
    conn = _db()
    _add_score(conn)
    _bootstrap_training_succeeded(conn, prior=PRIOR)
    _add_training(conn, "stage", prior=PRIOR, state="succeeded")

    assert _submit(conn, tmp_path) == ["receipt"]
    assert len(rec.promote) == 1
    promote = rec.promote[0]
    assert promote["expected_previous_release_id"] == PRIOR
    assert "require_empty_store" not in promote
    assert rec.requests[-1][1] == pool_refresh.phase_key(SESSION, "promote", PRIOR, TAG)


# --------------------------------------------------------------------------
# 5. a DAMAGED prior submits nothing and reports PRIOR_MISMATCH
# --------------------------------------------------------------------------


def test_damaged_prior_raises_prior_mismatch_without_submitting(monkeypatch, tmp_path):
    _env(monkeypatch)
    rec = _wire(monkeypatch, tmp_path, prior_class=DAMAGED_CLASS)
    conn = _db()
    _add_score(conn)
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "prior release could not be read"
    assert err.value.problem.details == {"code": "PRIOR_MISMATCH"}
    assert rec.submits == [] and rec.training == []


# --------------------------------------------------------------------------
# 6. pool_cycle_pending across the bootstrap key set
# --------------------------------------------------------------------------


def test_pool_cycle_pending_true_while_a_bootstrap_phase_runs(tmp_path):
    conn = _db()
    _add_score(conn)
    _add_training(conn, "train-line-thru", prior="none", state="running")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True


def test_pool_cycle_pending_false_when_a_bootstrap_phase_failed(tmp_path):
    conn = _db()
    _add_score(conn)
    _add_training(conn, "train-line-thru", prior="none", state="failed")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_pool_cycle_pending_false_without_a_pointer_and_without_bootstrap_jobs(tmp_path):
    conn = _db()
    _add_score(conn)
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_pool_cycle_pending_false_for_bootstrap_refusal_gaps(tmp_path):
    conn = _db()
    _add_score(conn)
    _bootstrap_training_succeeded(conn)  # no stage row after reconciliation
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False
    _add_training(conn, "stage", prior="none", state="succeeded")  # no promote row after reconciliation
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_pool_cycle_pending_true_while_a_bootstrap_stage_runs(tmp_path):
    conn = _db()
    _add_score(conn)
    _bootstrap_training_succeeded(conn)
    _add_training(conn, "stage", prior="none", state="running")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True


def test_pool_cycle_pending_true_for_a_no_pool_catalog_bootstrap(tmp_path):
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, "oldrel")  # no catalog -> the real classifier: NO_POOL_CATALOG
    _add_training(conn, "train-line-thru", prior="oldrel", state="running")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True


# --------------------------------------------------------------------------
# 7. recipe bookkeeping files are not state refs; only the artifacts bind
# --------------------------------------------------------------------------


def test_recipe_bookkeeping_files_are_skipped_by_the_stage_plan(monkeypatch, tmp_path):
    _env(monkeypatch)
    rec = _wire(monkeypatch, tmp_path, pointer=None, prior_class=NONE_CLASS)
    conn = _db()
    _add_score(conn)
    fold = f"folds__cut-{SESSION}__"
    bookkeeping = [
        "training/job.json",
        "training/summary.json",
        f"training/{fold}COMPLETE.json",
        f"training/{fold}membership_receipt.json",
        f"training/{fold}label_availability_receipt.json",
    ]
    # every recipe job publishes the same flat fold-artifact name
    artifacts = {
        "train-line-thru": "payoff_artifact.json",
        "train-map-thru": "recalibration_artifact.json",
        "train-surface-runup": "payoff_artifact.json",
        "train-line-runup": "payoff_artifact.json",
        "train-map-runup": "recalibration_artifact.json",
    }
    jobs = {}
    for phase in pool_refresh.BOOTSTRAP_TRAINING_PHASES:
        if phase in artifacts:
            jobs[phase] = _succeeded_training(
                conn, phase, bookkeeping + [f"training/{fold}{artifacts[phase]}"],
                prior="none")
        else:
            jobs[phase] = _succeeded_training(conn, phase, [f"training/{phase}.json"],
                                              prior="none")

    assert _submit(conn, tmp_path) == ["receipt"]  # identical names, no refusal
    refs = rec.stage[0]["state_refs"]
    expected = {f"{phase}__{phase}.json": f"{jobs[phase]}#training/{phase}.json"
                for phase in pool_refresh.BOOTSTRAP_TRAINING_PHASES
                if phase not in artifacts}
    for phase, artifact in artifacts.items():
        expected[f"{phase}__{fold}{artifact}"] = (
            f"{jobs[phase]}#training/{fold}{artifact}")
    assert refs == expected
    recipe_refs = {key: value for key, value in refs.items()
                   if key.split("__", 1)[0] in artifacts}
    assert len(recipe_refs) == 5  # one distinct phase-prefixed key per recipe job
    assert all("/" not in key and key.endswith(".json") for key in refs)


# --------------------------------------------------------------------------
# 8. the prior is classified only on the tick that starts the cycle
# --------------------------------------------------------------------------


def test_bootstrap_cycle_classifies_only_on_the_start_tick(monkeypatch, tmp_path):
    _env(monkeypatch)
    _wire(monkeypatch, tmp_path, pointer=None, prior_class=NONE_CLASS)
    calls = []

    def classify(_root, _prior_id):
        calls.append(1)
        return NONE_CLASS

    monkeypatch.setattr(pool_refresh, "classify_prior", classify)
    conn = _db()
    _add_score(conn)
    assert len(_submit(conn, tmp_path)) == 12  # the start tick classified once
    assert len(calls) == 1
    for phase in pool_refresh.BOOTSTRAP_TRAINING_PHASES:
        _add_training(conn, phase, prior="none", state="running")
    assert _submit(conn, tmp_path) == []  # later ticks: keys fix the family
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True
    assert len(calls) == 1


def test_carry_cycle_classifies_only_on_the_start_tick(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    calls = []

    def classify(_root, _prior_id):
        calls.append(1)
        return PriorClass(PRIOR_USABLE, "the fake prior catalog is usable")

    monkeypatch.setattr(pool_refresh, "classify_prior", classify)
    conn = _db()
    _add_score(conn)
    assert len(_submit(conn, tmp_path)) == 3  # the start tick classified once
    assert len(calls) == 1
    for phase in pool_refresh.TRAINING_PHASES:
        _add_training(conn, phase, state="running")
    assert _submit(conn, tmp_path) == []  # later ticks: keys fix the family
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True
    assert len(calls) == 1
