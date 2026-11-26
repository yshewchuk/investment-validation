"""EXP-189 Amendment 1, slice W7d re-cut: spec-driven F selection only; every F run is refused until W7d2 wires F.

Synthetic only: fixture spec dicts and tmp_path spec files, never the real spec.yaml selection, never data/."""
import json

import pytest

from experiments import fair_value_inputs as fvi
from experiments import fair_value_legs as fvl
from experiments import fair_value_run as fvr
from experiments import size_width_driver as driver
from experiments import size_width_f_select as sel
from experiments import size_width_prereg
from experiments import size_width_report
from tests.v2.research import test_size_pnl_driver as pnl
from tests.v2.research import test_size_width_prereg as ptest
from tests.v2.research.test_size_width_run import FAKE, NO_SOURCES, OK, _argv, _install, _run_main

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"


def _fill(**changes):
    fill = {"id": "F", "fvf_id": fvr.fvf_id()}
    fill.update(changes)
    return fill


def _f_receipt(fill=None):
    return {**OK(), "fill_model": _fill() if fill is None else fill}


def _spec(*amendments):
    return {"id": "EXP-189", "amendments": list(amendments)}


def _refused(reason, call):
    with pytest.raises(fvi.FairValueRefusal) as caught:
        call()
    assert caught.value.details["reason"] == reason
    assert caught.value.code == CONFLICT


# ---- 1. fill_model -------------------------------------------------------------------

def test_fill_model_is_none_without_amendments_or_without_the_key():
    assert sel.fill_model({}) is None
    assert sel.fill_model({"amendments": []}) is None
    assert sel.fill_model(_spec({"id": "A1", "note": "x"}, "plain")) is None


def test_fill_model_returns_the_one_f_entry_among_others():
    f = _fill(x=1)
    assert sel.fill_model(_spec({"id": "A0"}, {"id": "A1", "fill_model": f})) == f


def test_fill_model_with_another_id_refuses_fill_model_id():
    _refused("fill_model_id", lambda: sel.fill_model(_spec({"id": "A1", "fill_model": {"id": "G"}})))


def test_non_dict_fill_model_refuses_fill_model_invalid():
    _refused("fill_model_invalid", lambda: sel.fill_model(_spec({"id": "A1", "fill_model": "F"})))


def test_two_entries_carrying_fill_model_are_ambiguous_whatever_their_ids():
    _refused("fill_model_ambiguous", lambda: sel.fill_model(_spec({"fill_model": {"id": "F"}}, {"fill_model": {"id": "F"}})))
    _refused("fill_model_ambiguous", lambda: sel.fill_model(_spec({"fill_model": {"id": "G"}}, {"fill_model": None})))


@pytest.mark.parametrize("fvf_id", [None, ""], ids=["missing", "empty"])
def test_fill_model_without_a_non_empty_fvf_id_refuses_fill_model_invalid(fvf_id):
    fill = _fill()
    if fvf_id is None:
        del fill["fvf_id"]
    else:
        fill["fvf_id"] = fvf_id
    _refused("fill_model_invalid", lambda: sel.fill_model(_spec({"id": "A1", "fill_model": fill})))


def test_fill_model_with_a_wrong_fvf_id_refuses_fvf_id_mismatch():
    _refused("fvf_id_mismatch", lambda: sel.fill_model(_spec({"id": "A1", "fill_model": _fill(fvf_id="0" * 64)})))


# ---- 2. require_preregistration carries the selection into the receipt ------------------

def _prereg_case(tmp_path, amendments):
    spec = ptest._write_real_spec(tmp_path, amendments=amendments)
    digest = ptest._digest(spec)
    ledger = ptest._write_ledger(tmp_path / "LEDGER.csv", [ptest._planned_row(digest) for _ in range(len(amendments) + 1)])
    return spec, ledger


def _receipt(tmp_path, spec, ledger):
    return size_width_prereg.require_preregistration(
        spec_path=spec, ledger_path=ledger, run_log_path=ptest._run_log_path(tmp_path),
        experiments_dir=ptest._own_experiments(tmp_path))


def test_an_f_amendment_with_two_rows_puts_its_fill_model_in_the_receipt(tmp_path):
    fill = _fill()
    spec, ledger = _prereg_case(tmp_path, [{"id": "A1", "fill_model": fill}])
    receipt = _receipt(tmp_path, spec, ledger)
    assert receipt["fill_model"] == fill and receipt["spec_hash_checked"] is True


def test_no_fill_model_amendment_leaves_the_receipt_without_the_key(tmp_path):
    spec, ledger = _prereg_case(tmp_path, [])
    assert "fill_model" not in _receipt(tmp_path, spec, ledger)


def test_an_amendment_with_id_g_refuses_fill_model_id_through_the_guard(tmp_path):
    spec, ledger = _prereg_case(tmp_path, [{"id": "A1", "fill_model": {"id": "G"}}])
    assert ptest.refusal(lambda: _receipt(tmp_path, spec, ledger)) == "fill_model_id"


@pytest.mark.parametrize("fvf_id", [None, ""], ids=["missing", "empty"])
def test_an_f_amendment_without_a_non_empty_fvf_id_refuses_fill_model_invalid(tmp_path, fvf_id):
    fill = _fill()
    if fvf_id is None:
        del fill["fvf_id"]
    else:
        fill["fvf_id"] = fvf_id
    spec, ledger = _prereg_case(tmp_path, [{"id": "A1", "fill_model": fill}])
    assert ptest.refusal(lambda: _receipt(tmp_path, spec, ledger)) == "fill_model_invalid"


def test_an_f_amendment_with_a_wrong_fvf_id_refuses_fvf_id_mismatch(tmp_path):
    spec, ledger = _prereg_case(tmp_path, [{"id": "A1", "fill_model": _fill(fvf_id="0" * 64)}])
    assert ptest.refusal(lambda: _receipt(tmp_path, spec, ledger)) == "fvf_id_mismatch"


# ---- 3. open_run: the public run boundary --------------------------------------------

def test_open_run_without_a_selection_is_current_marks():
    assert sel.open_run(None) is None


def test_open_run_refuses_a_non_dict_fill():
    _refused("fill_model_invalid", lambda: sel.open_run("F"))


def test_open_run_with_id_g_refuses_fill_model_id():
    _refused("fill_model_id", lambda: sel.open_run(_fill(id="G")))


def test_open_run_with_no_id_refuses_fill_model_id():
    fill = _fill()
    del fill["id"]
    _refused("fill_model_id", lambda: sel.open_run(fill))


@pytest.mark.parametrize("fvf_id", [None, ""], ids=["missing", "empty"])
def test_open_run_without_a_non_empty_fvf_id_refuses_fill_model_invalid(fvf_id):
    fill = _fill()
    if fvf_id is None:
        del fill["fvf_id"]
    else:
        fill["fvf_id"] = fvf_id
    _refused("fill_model_invalid", lambda: sel.open_run(fill))


def test_open_run_with_a_wrong_fvf_id_refuses_fvf_id_mismatch():
    _refused("fvf_id_mismatch", lambda: sel.open_run(_fill(fvf_id="0" * 64)))


def test_open_run_with_a_valid_f_selection_refuses_f_run_not_wired():
    _refused("f_run_not_wired", lambda: sel.open_run(_fill()))


# ---- 4. the driver refuses an F spec before sources or writes ---------------------------

class _Boom:
    """Sources whose every attribute raises: an F spec must refuse before any of them is touched."""

    def __getattr__(self, name):
        raise AssertionError(f"sources.{name} was touched")


@pytest.mark.parametrize("stage", ["inner", "outer"])
def test_driver_refuses_a_valid_f_spec_before_sources_or_writes(tmp_path, stage):
    out = tmp_path / "out"
    _refused("f_run_not_wired", lambda: driver.run(_Boom(), stage, out, prereg=_f_receipt))
    assert not out.exists()


@pytest.mark.parametrize("stage", ["inner", "outer"])
def test_driver_refuses_an_id_g_spec_with_fill_model_id_before_sources_or_writes(tmp_path, stage):
    out = tmp_path / "out"
    _refused("fill_model_id", lambda: driver.run(_Boom(), stage, out, prereg=lambda: _f_receipt(_fill(id="G"))))
    assert not out.exists()


# ---- 5. run.py refuses an F spec at every stage, before inputs or --sources --------------

@pytest.mark.parametrize("stage", ["inner", "outer", "report"])
def test_run_py_refuses_a_valid_f_spec_for_every_stage_before_sources_or_inputs(tmp_path, monkeypatch, capsys, stage):
    run_py = _run_main()
    monkeypatch.setattr(size_width_prereg, "require_preregistration", _f_receipt)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")

    def boom(out_dir, **kwargs):
        raise AssertionError("report must not run for an F spec")

    monkeypatch.setattr(size_width_report, "assemble", boom)
    assert run_py.main(_argv(tmp_path, stage=stage, sources=NO_SOURCES)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == CONFLICT and refused["reason"] == "f_run_not_wired"
    assert not (tmp_path / "out").exists()


# ---- 6. run.py maps the fair-value refusals to a JSON refusal and exit 2 ----------------

@pytest.mark.parametrize("make, reason", [
    (lambda: fvl.FairValueRefusal("quote_repaired_invalid"), "quote_repaired_invalid"),
    (lambda: fvi.FairValueRefusal("some_reason", "msg"), "some_reason"),
], ids=["legs", "inputs"])
def test_run_py_prints_a_fair_value_refusal_from_the_driver_and_returns_2(tmp_path, monkeypatch, capsys, make, reason):
    run_py = _run_main()
    monkeypatch.setattr(size_width_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    _install(monkeypatch, lambda args: pnl.sources())

    def refuse(src, stage, out_dir, **kwargs):
        raise make()

    monkeypatch.setattr(driver, "run", refuse)
    assert run_py.main(_argv(tmp_path, sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == CONFLICT and refused["reason"] == reason
