from __future__ import annotations

import json
from pathlib import Path

import yaml

from engine.v2.ops.errors import OpsError
from experiments import fair_value_inputs as fvi
from experiments import lib, size_width_account, size_width_prereg, size_width_sizing

REAL_SPEC = size_width_prereg.SPEC_PATH


def _write_real_spec(case, **changes):
    """Re-dump the real EXP-189 spec.yaml into ``case``: amendments are emptied unless passed in ``changes``."""
    spec = yaml.safe_load(REAL_SPEC.read_text())
    spec["amendments"] = []
    spec.update(changes)
    path = case / "spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def _digest(spec_path):
    return lib.spec_hash(lib.load_spec(spec_path))


def _planned_row(digest, **overrides):
    row = {
        "id": "EXP-189",
        "spec_hash": digest,
        "date": "2026-11-24",
        "stage": "planned",
        "oos_mean_mid": "",
        "sharpe_trade": "",
        "promoted": "False",
    }
    row.update(overrides)
    return row


def _write_ledger(path, rows):
    """Append rows through the real ledger writer (synthetic, under tmp_path)."""
    lib.ledger_append(rows, path=path)
    return path


def _run_log_path(case):
    return case / "results" / "run_log.jsonl"


def _write_run_log(case, text):
    path = _run_log_path(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _check(case, spec_path, ledger_path):
    return size_width_prereg.require_preregistration(
        spec_path=spec_path, ledger_path=ledger_path, run_log_path=_run_log_path(case))


def refusal(callable_):
    try:
        callable_()
    except OpsError as exc:
        assert exc.code == "EXPERIMENT_IDENTITY_CONFLICT"
        return exc.problem.details["reason"]
    raise AssertionError("expected an OpsError refusal, got a normal return")


# --------------------------------------------------------------------------
# require_preregistration: refusals and the receipt
# --------------------------------------------------------------------------


def test_missing_ledger_file_is_planned_row_missing(tmp_path):
    spec = _write_real_spec(tmp_path)
    ledger = tmp_path / "LEDGER.csv"
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "planned_row_missing"


def test_planned_row_of_the_right_hash_gives_a_checked_receipt(tmp_path):
    spec = _write_real_spec(tmp_path)
    digest = _digest(spec)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(digest)])
    receipt = _check(tmp_path, spec, ledger)
    assert receipt["spec_hash_checked"] is True
    assert receipt["spec_hash"] == digest


def test_wrong_hash_is_spec_hash_mismatch(tmp_path):
    spec = _write_real_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row("0" * 64)])
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "spec_hash_mismatch"


def test_two_planned_rows_are_ambiguous(tmp_path):
    spec = _write_real_spec(tmp_path)
    digest = _digest(spec)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(digest), _planned_row(digest)])
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "planned_row_ambiguous"


def test_planned_row_for_another_id_only_is_planned_row_missing(tmp_path):
    spec = _write_real_spec(tmp_path)
    digest = _digest(spec)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(digest, id="EXP-188")])
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "planned_row_missing"


def test_spec_id_changed_to_exp_188_is_spec_id_mismatch(tmp_path):
    spec = _write_real_spec(tmp_path, id="EXP-188")
    ledger = tmp_path / "LEDGER.csv"
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "spec_id_mismatch"


def test_unparseable_spec_is_spec_unreadable(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text("id: [unclosed\n")
    assert refusal(lambda: _check(tmp_path, spec, tmp_path / "LEDGER.csv")) == "spec_unreadable"


def test_preregistered_after_first_run_is_refused(tmp_path):
    spec = _write_real_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    _write_run_log(tmp_path, json.dumps({"ts": "2020-01-01T00:00:00+00:00"}) + "\n")
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "preregistered_after_run"


def test_guard_never_writes_the_ledger(tmp_path):
    spec = _write_real_spec(tmp_path)
    ledger = tmp_path / "LEDGER.csv"
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "planned_row_missing"
    assert not ledger.exists()
    assert not hasattr(size_width_prereg, "register_planned")


# --------------------------------------------------------------------------
# check_id_free
# --------------------------------------------------------------------------


def _make_experiment(base, name, spec_id):
    case = base / name
    case.mkdir(parents=True)
    (case / "spec.yaml").write_text(yaml.safe_dump({"id": spec_id}))
    return case


def test_check_id_free_returns_the_single_exp_189_dir(tmp_path):
    own = _make_experiment(tmp_path, "EXP-189_x", "EXP-189")
    assert size_width_prereg.check_id_free(tmp_path) == own


def test_check_id_free_refuses_a_second_exp_189_dir(tmp_path):
    _make_experiment(tmp_path, "EXP-189_x", "EXP-189")
    (tmp_path / "EXP-189_y").mkdir()
    assert refusal(lambda: size_width_prereg.check_id_free(tmp_path)) == "experiment_id_conflict"


def test_check_id_free_refuses_another_spec_claiming_exp_189(tmp_path):
    _make_experiment(tmp_path, "EXP-189_x", "EXP-189")
    _make_experiment(tmp_path, "EXP-190_z", "EXP-189")
    assert refusal(lambda: size_width_prereg.check_id_free(tmp_path)) == "experiment_id_conflict"


def test_check_id_free_refuses_when_no_exp_189_dir_exists(tmp_path):
    assert refusal(lambda: size_width_prereg.check_id_free(tmp_path)) == "experiment_id_conflict"


def test_check_id_free_on_the_real_repo_returns_the_exp_189_dir():
    assert size_width_prereg.check_id_free().name.startswith("EXP-189_")


# --------------------------------------------------------------------------
# the real spec
# --------------------------------------------------------------------------


def test_real_spec_is_exp_189_with_the_design_grid_and_account_equity():
    spec, digest = size_width_prereg.load_spec()
    assert spec["id"] == "EXP-189"
    assert digest == lib.spec_hash(lib.load_spec(REAL_SPEC))
    assert list(spec["grid"]["k"]) == list(size_width_sizing.GRID)
    assert spec["account"]["start_equity"] == size_width_account.START_EQUITY


# The hash of the original pre-registered spec: the ledger's first EXP-189 PLANNED row.
ORIGINAL_HASH = "60773dc627e4b18040d02de0d0e79e29b0b5964749ecb5d52737c17f4958b41a"


def _real_amendment():
    spec, _ = size_width_prereg.load_spec()
    (amendment,) = spec["amendments"]
    return spec, amendment


def test_amendment_1_leaves_the_original_fields_untouched():
    spec, _ = _real_amendment()
    assert lib.spec_hash({**spec, "amendments": []}) == ORIGINAL_HASH
    assert spec["preregistered_at"] == "2026-11-23T12:00:00+00:00"


def test_amendment_1_selects_f_with_loadable_bindings():
    _, amendment = _real_amendment()
    assert amendment["post_hoc"] is True
    assert amendment["label"] == "POST-HOC (Amendment 1, F fills)"
    fill = amendment["fill_model"]
    assert fill["id"] == "F"
    assert fill["fvf_id"] == "735f3e6cb2fa65c303599152cfa6fead2d7950e9bcb367117d50ed25faffc0b7"
    assert fill["rate_rules"] == {"rate_out_of_range": "missing", "rate_empty_row": "rate_unknown"}
    for name, value in fill["rate_rules"].items():
        assert value in fvi.RATE_RULE_VALUES[name]
    assert (fill["vol_scale"], fill["rate_scale"]) == (1.0, 0.01)
    reingest = fill["reingest"]
    assert reingest["manifest_sha256"] == "17ecc75e84d2e261900eccc3ad79a1cbd065af2d58eabdfebfd49dd75001f75d"
    assert reingest["manifest_units"] == {
        "iRate5wk": "percent", "iRateLt": "percent", "borrow30": "percent", "borrow2yr": "percent",
        "divAmt": "dollars_per_share", "annIdiv": "dollars_per_share", "annActDiv": "dollars_per_share",
        "divDate": "date", "assetType": "code"}
    # F's own manifest check accepts the bound unit labels (it refuses input_unit otherwise).
    unread = {"unreadable": 0, "unrecognized": 0}
    fvi._check_manifest(json.dumps({"tables": {"quotes": unread, "rates": unread},
                                    "units": reingest["manifest_units"]}).encode())


def test_amended_spec_needs_the_original_row_and_a_new_one_last(tmp_path):
    _, digest = size_width_prereg.load_spec()
    exps = _own_experiments(tmp_path)

    def check(name, rows):
        return size_width_prereg.require_preregistration(
            spec_path=REAL_SPEC, ledger_path=_write_ledger(tmp_path / name, rows),
            run_log_path=_run_log_path(tmp_path), experiments_dir=exps)

    original, new = _planned_row(ORIGINAL_HASH), _planned_row(digest)
    assert refusal(lambda: check("only_original.csv", [original])) == "planned_row_ambiguous"
    assert refusal(lambda: check("wrong_order.csv", [new, original])) == "spec_hash_mismatch"
    receipt = check("registered.csv", [original, new])
    assert receipt["spec_hash"] == digest and receipt["spec_hash_checked"] is True


# --------------------------------------------------------------------------
# require_preregistration: id check, ledger bound, amendments, read-only
# --------------------------------------------------------------------------


def _own_experiments(case):
    """A tmp experiments dir holding exactly one EXP-189_* directory."""
    exps = case / "exps"
    (exps / "EXP-189_x").mkdir(parents=True)
    return exps


def test_two_exp_189_dirs_refuse_even_with_a_correct_planned_row(tmp_path):
    spec = _write_real_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    exps = tmp_path / "exps"
    (exps / "EXP-189_a").mkdir(parents=True)
    (exps / "EXP-189_b").mkdir()
    assert refusal(lambda: size_width_prereg.require_preregistration(
        spec_path=spec, ledger_path=ledger, run_log_path=_run_log_path(tmp_path),
        experiments_dir=exps)) == "experiment_id_conflict"


def test_ledger_larger_than_16_mib_is_ledger_malformed(tmp_path):
    spec = _write_real_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    with open(ledger, "r+b") as f:
        f.truncate(17 << 20)
    assert ledger.stat().st_size == 17 << 20
    exps = _own_experiments(tmp_path)
    assert refusal(lambda: size_width_prereg.require_preregistration(
        spec_path=spec, ledger_path=ledger, run_log_path=_run_log_path(tmp_path),
        experiments_dir=exps)) == "ledger_malformed"


def test_declared_amendment_with_one_planned_row_is_ambiguous(tmp_path):
    spec = _write_real_spec(tmp_path, amendments=["a"])
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    exps = _own_experiments(tmp_path)
    assert refusal(lambda: size_width_prereg.require_preregistration(
        spec_path=spec, ledger_path=ledger, run_log_path=_run_log_path(tmp_path),
        experiments_dir=exps)) == "planned_row_ambiguous"


def test_successful_check_leaves_the_ledger_bytes_unchanged(tmp_path):
    spec = _write_real_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    before = ledger.read_bytes()
    receipt = size_width_prereg.require_preregistration(
        spec_path=spec, ledger_path=ledger, run_log_path=_run_log_path(tmp_path),
        experiments_dir=_own_experiments(tmp_path))
    assert receipt["spec_hash_checked"] is True
    assert ledger.read_bytes() == before


# --------------------------------------------------------------------------
# check_id_free: a malformed sibling spec must not fail open
# --------------------------------------------------------------------------


def test_check_id_free_refuses_malformed_sibling_that_mentions_exp_189(tmp_path):
    _make_experiment(tmp_path, "EXP-189_x", "EXP-189")
    sibling = tmp_path / "EXP-190_z"
    sibling.mkdir()
    (sibling / "spec.yaml").write_text("id: EXP-189\n: [unclosed")
    assert refusal(lambda: size_width_prereg.check_id_free(tmp_path)) == "experiment_id_conflict"


def test_check_id_free_skips_malformed_sibling_that_never_mentions_exp_189(tmp_path):
    own = _make_experiment(tmp_path, "EXP-189_x", "EXP-189")
    sibling = tmp_path / "EXP-190_z"
    sibling.mkdir()
    (sibling / "spec.yaml").write_text("a: [unclosed")
    assert size_width_prereg.check_id_free(tmp_path) == own


def test_check_id_free_refuses_oversized_malformed_sibling_unread(tmp_path):
    _make_experiment(tmp_path, "EXP-189_x", "EXP-189")
    sibling = tmp_path / "EXP-190_z"
    sibling.mkdir()
    (sibling / "spec.yaml").write_text("a: [unclosed\n" + "x" * (4 << 20))
    assert refusal(lambda: size_width_prereg.check_id_free(tmp_path)) == "experiment_id_conflict"
