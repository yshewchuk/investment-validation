from __future__ import annotations

import yaml

from engine.v2.ops.errors import OpsError
from experiments import lib, size_quantile_prereg
from experiments.size_quantile_prereg import check_id_free, load_spec

REAL_SPEC = size_quantile_prereg.SPEC_PATH
STAMP = "2026-11-25T12:00:00+00:00"


def _write_spec(case, *, settled=True, drop_stage_b=False, **changes):
    """Re-dump the real EXP-190 spec.yaml into ``case``: settled stage_b and a stamp unless told otherwise."""
    spec = yaml.safe_load(REAL_SPEC.read_text())
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
    return lib.spec_hash(lib.load_spec(spec_path))


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


# --------------------------------------------------------------------------
# the real spec
# --------------------------------------------------------------------------


def test_real_spec_has_the_design_levels_grid_and_pending_stage_b():
    spec, digest = load_spec()
    assert spec["id"] == "EXP-190"
    assert digest == lib.spec_hash(lib.load_spec(REAL_SPEC))
    assert spec["levels"] == [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    assert spec["grid"]["axes"] == {"max_leaf_nodes": [7, 15], "min_samples_leaf": [300, 1000]}
    assert spec["grid"]["fixed"] == {
        "learning_rate": 0.03, "l2_regularization": 10.0, "max_iter": 3000, "early_stopping": True,
        "validation_fraction": 0.1, "n_iter_no_change": 10, "tol": 1.0e-07, "random_state": 0,
    }
    product = {(a, b) for a in spec["grid"]["axes"]["max_leaf_nodes"]
               for b in spec["grid"]["axes"]["min_samples_leaf"]}
    assert len(product) == 4
    assert spec["stage_b"]["status"] == "pending_part_2"
    assert spec["preregistered_at"] is None
