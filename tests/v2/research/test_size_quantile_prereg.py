from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import yaml

from engine.v2.models.frozen_state_io import read_bounded_state
from engine.v2.ops.errors import OpsError
from experiments import lib, size_quantile_prereg
from experiments.size_quantile_prereg import check_id_free, load_spec

REAL_SPEC = size_quantile_prereg.SPEC_PATH
STAMP = "2026-11-25T12:00:00+00:00"


def _write_spec(case, *, settled=True, drop_stage_b=False, **changes):
    """Re-dump the real EXP-190 spec.yaml into ``case``: settled stage_b and a stamp unless told otherwise."""
    spec = yaml.safe_load(read_bounded_state(REAL_SPEC, 1 << 20).decode("utf-8"))
    if settled:
        spec["stage_b"] = {**spec["stage_b"], "status": "settled"}
        spec["preregistered_at"] = STAMP
    if drop_stage_b:
        spec.pop("stage_b")
    spec.update(changes)
    path = case / "spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def _digest(spec_path):
    spec = yaml.safe_load(read_bounded_state(Path(spec_path), 1 << 20).decode("utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"spec at {spec_path} did not parse to a mapping")
    return lib.spec_hash(spec)


def _planned_row(digest, **overrides):
    row = {
        "id": "EXP-190",
        "spec_hash": digest,
        "date": "2026-11-25",
        "stage": "planned",
        "oos_mean_mid": "",
        "sharpe_trade": "",
        "promoted": "False",
    }
    row.update(overrides)
    return row


def _write_ledger(path, rows):
    lib.ledger_append(rows, path=path)
    return path


def _run_log_path(case):
    return case / "results" / "run_log.jsonl"


def _check(case, spec_path, ledger_path, experiments_dir=None):
    return size_quantile_prereg.require_preregistration(
        spec_path=spec_path, ledger_path=ledger_path, run_log_path=_run_log_path(case),
        experiments_dir=experiments_dir)


def _own_experiments(case):
    """A tmp experiments dir holding exactly one EXP-190_* directory."""
    exps = case / "exps"
    (exps / "EXP-190_x").mkdir(parents=True)
    return exps


def refusal(callable_):
    try:
        callable_()
    except OpsError as exc:
        assert exc.code == "EXPERIMENT_IDENTITY_CONFLICT"
        return exc.problem.details["reason"]
    raise AssertionError("expected an OpsError refusal, got a normal return")


# --------------------------------------------------------------------------
# require_preregistration: refusals and the receipt (settled tmp spec)
# --------------------------------------------------------------------------


def test_missing_ledger_file_is_planned_row_missing(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = tmp_path / "LEDGER.csv"
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "planned_row_missing"


def test_planned_row_of_the_right_hash_gives_a_checked_receipt(tmp_path):
    spec = _write_spec(tmp_path)
    digest = _digest(spec)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(digest)])
    receipt = _check(tmp_path, spec, ledger)
    assert receipt["spec_hash_checked"] is True
    assert receipt["spec_hash"] == digest


def test_wrong_hash_is_spec_hash_mismatch(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row("0" * 64)])
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "spec_hash_mismatch"


def test_two_planned_rows_are_ambiguous(tmp_path):
    spec = _write_spec(tmp_path)
    digest = _digest(spec)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(digest), _planned_row(digest)])
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "planned_row_ambiguous"


def test_planned_row_for_another_id_only_is_planned_row_missing(tmp_path):
    spec = _write_spec(tmp_path)
    digest = _digest(spec)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(digest, id="EXP-189")])
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "planned_row_missing"


def test_declared_amendment_needs_two_planned_rows(tmp_path):
    spec = _write_spec(tmp_path, amendments=["a"])
    digest = _digest(spec)
    exps = _own_experiments(tmp_path)
    one = _write_ledger(tmp_path / "ONE.csv", [_planned_row(digest)])
    assert refusal(lambda: _check(tmp_path, spec, one, exps)) == "planned_row_ambiguous"
    two = _write_ledger(tmp_path / "TWO.csv", [_planned_row(digest), _planned_row(digest)])
    assert _check(tmp_path, spec, two, exps)["spec_hash_checked"] is True


def test_spec_id_changed_to_exp_189_is_spec_id_mismatch(tmp_path):
    spec = _write_spec(tmp_path, id="EXP-189")
    ledger = tmp_path / "LEDGER.csv"
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "spec_id_mismatch"


def test_null_preregistered_at_is_preregistered_at_missing(tmp_path):
    spec = _write_spec(tmp_path, preregistered_at=None)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    assert refusal(lambda: _check(tmp_path, spec, ledger, _own_experiments(tmp_path))) == "preregistered_at_missing"


def test_unparseable_spec_is_spec_unreadable(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text("id: [unclosed\n")
    assert refusal(lambda: _check(tmp_path, spec, tmp_path / "LEDGER.csv")) == "spec_unreadable"


def test_guard_never_writes_the_ledger_or_registers(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = tmp_path / "LEDGER.csv"
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "planned_row_missing"
    assert not ledger.exists()
    assert not hasattr(size_quantile_prereg, "register_planned")


def test_successful_check_leaves_the_ledger_bytes_unchanged(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    before = ledger.read_bytes()
    receipt = _check(tmp_path, spec, ledger, _own_experiments(tmp_path))
    assert receipt["spec_hash_checked"] is True
    assert ledger.read_bytes() == before


# --------------------------------------------------------------------------
# stage_b gate: refuses before the ledger is read
# --------------------------------------------------------------------------


def test_pending_stage_b_refuses_even_with_a_matching_planned_row(tmp_path):
    spec = _write_spec(tmp_path, settled=False)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    before = ledger.read_bytes()
    assert refusal(lambda: _check(tmp_path, spec, ledger, _own_experiments(tmp_path))) == "stage_b_pending"
    assert ledger.read_bytes() == before


def test_missing_stage_b_refuses_even_with_a_matching_planned_row(tmp_path):
    spec = _write_spec(tmp_path, drop_stage_b=True)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    assert refusal(lambda: _check(tmp_path, spec, ledger, _own_experiments(tmp_path))) == "stage_b_pending"


def test_non_dict_stage_b_refuses_even_with_a_matching_planned_row(tmp_path):
    spec = _write_spec(tmp_path, stage_b="settled")
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    assert refusal(lambda: _check(tmp_path, spec, ledger, _own_experiments(tmp_path))) == "stage_b_pending"


def test_real_spec_with_a_matching_ledger_row_refuses_stage_b_pending(tmp_path):
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(REAL_SPEC))])
    assert refusal(lambda: _check(tmp_path, REAL_SPEC, ledger, _own_experiments(tmp_path))) == "stage_b_pending"


# --------------------------------------------------------------------------
# check_id_free
# --------------------------------------------------------------------------


def _make_experiment(base, name, spec_id):
    case = base / name
    case.mkdir(parents=True)
    (case / "spec.yaml").write_text(yaml.safe_dump({"id": spec_id}))
    return case


def test_check_id_free_returns_the_single_exp_190_dir(tmp_path):
    own = _make_experiment(tmp_path, "EXP-190_x", "EXP-190")
    assert check_id_free(tmp_path) == own


def test_check_id_free_refuses_a_second_exp_190_dir(tmp_path):
    _make_experiment(tmp_path, "EXP-190_x", "EXP-190")
    (tmp_path / "EXP-190_y").mkdir()
    assert refusal(lambda: check_id_free(tmp_path)) == "experiment_id_conflict"


def test_check_id_free_refuses_another_spec_claiming_exp_190(tmp_path):
    _make_experiment(tmp_path, "EXP-190_x", "EXP-190")
    _make_experiment(tmp_path, "EXP-191_z", "EXP-190")
    assert refusal(lambda: check_id_free(tmp_path)) == "experiment_id_conflict"


def test_check_id_free_refuses_when_no_exp_190_dir_exists(tmp_path):
    assert refusal(lambda: check_id_free(tmp_path)) == "experiment_id_conflict"


def test_check_id_free_on_the_real_repo_returns_the_exp_190_dir():
    assert check_id_free().name.startswith("EXP-190_")


def test_check_id_free_refuses_malformed_sibling_that_mentions_exp_190(tmp_path):
    _make_experiment(tmp_path, "EXP-190_x", "EXP-190")
    sibling = tmp_path / "EXP-191_z"
    sibling.mkdir()
    (sibling / "spec.yaml").write_text("id: EXP-190\n: [unclosed")
    assert refusal(lambda: check_id_free(tmp_path)) == "experiment_id_conflict"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")
def test_check_id_free_refuses_a_fifo_spec_without_hanging(tmp_path):
    _make_experiment(tmp_path, "EXP-190_x", "EXP-190")
    sibling = tmp_path / "EXP-191_z"
    sibling.mkdir()
    os.mkfifo(sibling / "spec.yaml")
    assert refusal(lambda: check_id_free(tmp_path)) == "experiment_id_conflict"


# --------------------------------------------------------------------------
# the real spec
# --------------------------------------------------------------------------


def test_real_spec_has_the_design_levels_grid_and_pending_decisions():
    spec, digest = load_spec()
    assert spec["id"] == "EXP-190"
    assert digest == _digest(REAL_SPEC)
    assert spec["levels"] == [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    assert spec["grid"]["axes"] == {"max_leaf_nodes": [7, 15], "min_samples_leaf": [300, 1000]}
    assert spec["grid"]["fixed"] == {
        "learning_rate": 0.03, "l2_regularization": 10.0, "max_iter": 3000, "early_stopping": True,
        "validation_fraction": 0.1, "n_iter_no_change": 10, "tol": 1.0e-07, "random_state": 0,
    }
    product = {(a, b) for a in spec["grid"]["axes"]["max_leaf_nodes"]
               for b in spec["grid"]["axes"]["min_samples_leaf"]}
    assert len(product) == 4
    assert spec["stage_b"]["status"] == "pending_decisions"
    assert set(spec["stage_b"]["open_decisions"]) == {"tau_window_aggregation", "buildability_fallback"}
    assert spec["preregistered_at"] is None


# --------------------------------------------------------------------------
# stage_b and Stage A terms of the real spec (literals, not production modules)
# --------------------------------------------------------------------------

FVF_ID = "735f3e6cb2fa65c303599152cfa6fead2d7950e9bcb367117d50ed25faffc0b7"
MANIFEST_SHA = "17ecc75e84d2e261900eccc3ad79a1cbd065af2d58eabdfebfd49dd75001f75d"
ARMS = {"P0", "E", "W", "LW", "D", "WD", "LWD", "TF", "EQ", "N1", "N2", "ST", "ST0"}
TAUS = ["p10", "p25", "p40", "p50", "p60", "p75"]
CONTROLS = {
    "W": ["P0", "E", "LW"], "D": ["P0", "E"], "EQ": ["P0", "E"], "TF": ["P0", "E"],
    "WD": ["P0", "E", "LWD"], "N1": ["P0", "E", "N2"], "ST": ["ST0"],
}
BOTH = ["centre_peaked", "twin"]


def test_fill_model_is_pinned_to_fvf_literals():
    fill = load_spec()[0]["stage_b"]["fill_model"]
    assert fill["id"] == "F"
    assert fill["fvf_id"] == FVF_ID
    assert fill["carry_dividend_mode"] == "f7_orats_borrow_discrete_dividend"
    assert fill["reingest"]["manifest_sha256"] == MANIFEST_SHA


def test_arms_taus_and_universe_literals():
    spec = load_spec()[0]
    assert set(spec["stage_b"]["arms"]) == ARMS
    assert spec["stage_b"]["tau"]["set"] == TAUS
    assert spec["stage_b"]["universe"]["centre_peaked"] == ["CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5"]
    assert spec["stage_b"]["universe"]["twin"] == ["TWIN-P", "TWIN-P5"]


def test_success_rules_partition_the_arms():
    success = load_spec()[0]["stage_b"]["success"]
    assert success["controls"] == CONTROLS
    assert set(success["judged"]) == set(success["controls"])
    never = set(success["never_candidates"])
    assert never.isdisjoint(success["judged"])
    assert set(success["judged"]) | never == ARMS


def test_selection_and_account_settings_are_literal():
    stage_b = load_spec()[0]["stage_b"]
    assert stage_b["selection_models"]["configuration"] == {
        "loss": "squared_error", "max_leaf_nodes": 7, "min_samples_leaf": 300, "learning_rate": 0.03,
        "l2_regularization": 10.0, "max_iter": 300, "early_stopping": False, "random_state": 0, "tuning": "none",
    }
    assert stage_b["accounts"]["policy"] == {
        "start_equity": 200000, "cap": 0.66, "target_share": 0.25, "expensive_first": True,
    }


def test_spec_hash_covers_stage_b_fill_and_controls():
    spec = load_spec()[0]
    base = lib.spec_hash(spec)
    changed = copy.deepcopy(spec)
    changed["stage_b"]["fill_model"]["fvf_id"] = "0" * 64
    assert lib.spec_hash(changed) != base
    changed = copy.deepcopy(spec)
    changed["stage_b"]["success"]["controls"]["W"] = ["P0", "E"]
    assert lib.spec_hash(changed) != base


def test_stage_a_terms_are_not_changed_by_stage_b():
    spec = load_spec()[0]
    assert set(spec["stage_a"]["settlements"]) == {"status"} | {f"s{i}" for i in range(1, 10)}
    assert "publishes no receipt for that year" in spec["failure_semantics"]["fit_failed"]
    assert "Q_raw" in spec["failure_semantics"]["fit_failed"]
    assert "50-ticker smoke" in spec["real_data_order"]
    assert "admitted event keys only" in spec["real_data_order"]
    assert spec["real_data_order"].index("PLANNED row") < spec["real_data_order"].index("50-ticker smoke")
