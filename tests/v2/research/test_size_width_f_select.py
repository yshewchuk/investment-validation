"""EXP-189 Amendment 1, slice W7d2: spec-driven F selection, the F run boundary (open_run), the report check and the run.py F inputs.

Synthetic only: fixture spec dicts, tmp_path pinned tables and fake quote readers; never the real spec.yaml selection, never data/."""
import json
import logging
import sys
import types

import pytest

from engine import paths
from experiments import fair_value_inputs as fvi
from experiments import fair_value_legs as fvl
from experiments import fair_value_run as fvr
from experiments import size_pnl_receipts as receipts
from experiments import size_pnl_shortvol as shortvol
from experiments import size_width_driver as driver
from experiments import size_width_f_fills as ff
from experiments import size_width_f_select as sel
from experiments import size_width_prereg
from experiments import size_width_report
from tests.v2.research import test_size_pnl_driver as pnl
from tests.v2.research import test_size_width_f_quotes as qt
from tests.v2.research import test_size_width_prereg as ptest
from tests.v2.research.test_size_width_run import FAKE, NO_SOURCES, OK, _argv, _gates, _install, _run_main

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
BINDING_KEYS = {"fill_model_id", "fvf_id", "reingest_manifest_sha256", "every_table_file_sha256", "quote_snapshot_id", "units",
                "rate_out_of_range", "rate_empty_row", "carry_dividend_mode"}


@pytest.fixture(autouse=True)
def bounded(monkeypatch):
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")


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

def _tables(tmp_path):
    """The pinned tables under tmp_path/root (the fill's reingest path); returns their pins."""
    return qt._pins(qt._write(tmp_path, qt._base(qt._frames())))


def _full(pins, **changes):
    """A complete F fill (the spec's fill_model shape) for the pinned tables under ``root``, with ``changes`` applied."""
    fill = {"id": "F", "fvf_id": fvr.fvf_id(), "carry_dividend_mode": sel.MODE,
            "rate_rules": {"rate_out_of_range": "missing", "rate_empty_row": "rate_unknown"}, "vol_scale": 0.01, "rate_scale": 0.01,
            "reingest": {"path": "root", "tool_version": "4", "manifest_sha256": pins["manifest_sha256"], "manifest_units": qt.UNITS}}
    fill.update(changes)
    return fill


def _inputs(pins, read=None, snapshot_id="snap-1"):
    return sel.FInputs(pins, read or qt._reader(qt._base(qt._frames())), snapshot_id)


def _open(fill, inputs, tmp_path):
    return sel.open_run(fill, inputs, qt.DEFAULT_PLAN, "snap-1", root=tmp_path)


def test_open_run_without_a_selection_is_current_marks():
    assert sel.open_run(None) is None


def test_open_run_with_inputs_and_no_selection_refuses_f_inputs_unexpected():
    _refused("f_inputs_unexpected", lambda: sel.open_run(None, _inputs({"manifest_sha256": "0" * 64, "files": {"a": "0"}})))


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


def test_open_run_with_a_valid_f_selection_and_no_inputs_refuses_f_inputs_missing():
    _refused("f_inputs_missing", lambda: sel.open_run(_fill()))


@pytest.mark.parametrize("make_inputs", [
    lambda pins, calls: object(),
    lambda pins, calls: {"pins": pins, "read_quotes": qt._reader(qt._base(qt._frames()), calls), "snapshot_id": "snap-1"},
    lambda pins, calls: (pins, qt._reader(qt._base(qt._frames()), calls), "snap-1"),
], ids=["object", "dict", "plain_tuple"])
def test_open_run_with_inputs_that_are_not_f_inputs_refuses_f_inputs_invalid_before_any_read(tmp_path, make_inputs):
    pins, calls = _tables(tmp_path), []
    _refused("f_inputs_invalid", lambda: _open(_full(pins), make_inputs(pins, calls), tmp_path))
    assert calls == []


def test_open_run_with_a_valid_f_selection_returns_the_f_run(tmp_path):
    pins = _tables(tmp_path)
    run = _open(_full(pins), _inputs(pins), tmp_path)
    assert isinstance(run, sel.FRun)
    assert set(run.binding) == BINDING_KEYS
    assert run.binding["fill_model_id"] == "F" and run.binding["quote_snapshot_id"] == "snap-1"
    assert run.binding["every_table_file_sha256"] == dict(sorted(pins["files"].items()))
    assert run.binding["units"] == {"vol_scale": 0.01, "rate_scale": 0.01, "manifest_units": qt.UNITS}
    assert run.binding["rate_out_of_range"] == "missing" and run.binding["rate_empty_row"] == "rate_unknown"
    assert isinstance(run.pricing.fills, ff.FFills)


def test_open_run_reads_the_tables_root_from_paths_at_call_time(tmp_path, monkeypatch):
    pins = _tables(tmp_path)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    assert isinstance(sel.open_run(_full(pins), _inputs(pins), qt.DEFAULT_PLAN, "snap-1"), sel.FRun)


def test_a_non_callable_reader_refuses_f_quotes_invalid_before_any_read(tmp_path):
    pins = _tables(tmp_path)
    _refused("f_quotes_invalid", lambda: _open(_full(pins), _inputs(pins, read="not callable"), tmp_path))


MALFORMED = [
    pytest.param(lambda f: f.pop("reingest"), id="no-reingest"),
    pytest.param(lambda f: f.pop("vol_scale"), id="no-vol-scale"),
    pytest.param(lambda f: f.update(rate_rules={"rate_out_of_range": "bogus", "rate_empty_row": "rate_unknown"}), id="bad-rate-rule"),
    pytest.param(lambda f: f.update(rate_rules={"rate_out_of_range": "missing"}), id="short-rate-rules"),
    pytest.param(lambda f: f["reingest"].update(path="/abs/root"), id="absolute-path"),
    pytest.param(lambda f: f["reingest"].update(path="root/../root"), id="dotdot-path"),
    pytest.param(lambda f: f["reingest"].update(manifest_units=["percent"]), id="units-not-a-dict"),
]


@pytest.mark.parametrize("edit", MALFORMED)
def test_malformed_fill_refuses_fill_model_invalid_before_any_read(tmp_path, edit):
    pins, calls = _tables(tmp_path), []
    fill = _full(pins)
    edit(fill)
    _refused("fill_model_invalid", lambda: _open(fill, _inputs(pins, read=qt._reader(qt._base(qt._frames()), calls)), tmp_path))
    assert calls == []


def test_a_malformed_reingest_hash_refuses_manifest_binding_before_any_read(tmp_path):
    pins, calls = _tables(tmp_path), []
    fill = _full(pins)
    fill["reingest"]["manifest_sha256"] = "ABC"
    _refused("manifest_binding", lambda: _open(fill, _inputs(pins, read=qt._reader(qt._base(qt._frames()), calls)), tmp_path))
    assert calls == []


def test_a_carry_and_dividend_mode_other_than_the_registered_one_refuses(tmp_path):
    pins = _tables(tmp_path)
    _refused("carry_dividend_mode", lambda: _open(_full(pins, carry_dividend_mode="other"), _inputs(pins), tmp_path))


BAD_PINS = [
    pytest.param({}, id="empty"),
    pytest.param(None, id="none"),
    pytest.param({"manifest_sha256": 1, "files": {"a": "0"}}, id="hash-not-a-string"),
    pytest.param({"manifest_sha256": "0" * 64, "files": {}}, id="no-files"),
    pytest.param({"manifest_sha256": "0" * 64, "files": {"a": None}}, id="file-hash-not-a-string"),
]


@pytest.mark.parametrize("pins", BAD_PINS)
def test_malformed_pins_refuse_pins_invalid_before_any_read(tmp_path, pins):
    _tables(tmp_path)
    _refused("pins_invalid", lambda: _open(_full({"manifest_sha256": "0" * 64}), _inputs(pins), tmp_path))


def test_a_missing_manifest_refuses_manifest_binding(tmp_path):
    pins = _tables(tmp_path)
    (tmp_path / "root" / "manifest.json").unlink()
    _refused("manifest_binding", lambda: _open(_full(pins), _inputs(pins), tmp_path))


def test_manifest_units_differing_from_the_spec_refuse_manifest_binding(tmp_path):
    pins = _tables(tmp_path)
    fill = _full(pins)
    fill["reingest"]["manifest_units"] = {**qt.UNITS, "borrow30": "dollars_per_share"}
    _refused("manifest_binding", lambda: _open(fill, _inputs(pins), tmp_path))


def test_a_pins_hash_differing_from_the_fill_refuses_manifest_binding(tmp_path):
    pins = _tables(tmp_path)
    _refused("manifest_binding", lambda: _open(_full(pins), _inputs({**pins, "manifest_sha256": "1" * 64}), tmp_path))


def test_a_pinned_file_hash_differing_from_the_file_opens_and_refuses_table_hash_at_the_first_load(tmp_path):
    pins = _tables(tmp_path)
    files = {**pins["files"], "quotes/year=2024/month=02.parquet": "0" * 64}
    calls = []
    run = _open(_full(pins), _inputs({**pins, "files": files}, read=qt._reader(qt._base(qt._frames()), calls)), tmp_path)
    assert isinstance(run, sel.FRun) and calls == []
    chunk = qt._chunk(qt._frames(), qt.DEFAULT_PLAN)
    _refused("table_hash", lambda: run.pricing.fills.quotes.table(chunk))


def test_open_run_binds_and_loads_from_a_private_copy_of_the_pins(tmp_path):
    pins = _tables(tmp_path)
    original = dict(sorted(pins["files"].items()))
    run = _open(_full(pins), _inputs(pins), tmp_path)
    for key in list(pins["files"]):
        pins["files"][key] = "0" * 64
    pins["files"]["bogus/extra.parquet"] = "0" * 64
    pins["manifest_sha256"] = "x"
    assert run.binding["every_table_file_sha256"] == original
    chunk = qt._chunk(qt._frames(), qt.DEFAULT_PLAN)
    run.pricing.fills.quotes.table(chunk)  # the first load still verifies against the copy's original hashes


# ---- 4. the report check: an F report reads only inputs that bind F ------------------

def _write_inputs(out, body):
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    receipts.write_receipt(out / "receipts" / "inputs_inner.json", body)


NON_F = {"spec_hash": "SPEC", "spec_hash_checked": True}


def test_check_report_under_a_non_f_registration_passes_without_a_receipts_dir(tmp_path):
    assert sel.check_report(NON_F, tmp_path / "missing") is None


def test_check_report_under_a_non_f_registration_accepts_inputs_without_fill_model(tmp_path):
    _write_inputs(tmp_path, {"spec_hash": "SPEC"})
    assert sel.check_report(NON_F, tmp_path) is None


def test_check_report_under_a_non_f_registration_refuses_inputs_that_bind_f(tmp_path):
    _write_inputs(tmp_path, {"spec_hash": "SPEC", "fill_model": {"fill_model_id": "F"}})
    _refused("f_report_unbound", lambda: sel.check_report(NON_F, tmp_path))


def test_check_report_refuses_an_f_report_whose_inputs_are_missing(tmp_path):
    with pytest.raises(shortvol.Refusal) as caught:
        sel.check_report(_f_receipt(), tmp_path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "tuning_receipt_missing"


@pytest.mark.parametrize("body", [
    pytest.param({"spec_hash": "SPEC"}, id="non-f-inputs"),
    pytest.param({"spec_hash": "SPEC", "fill_model": {"fill_model_id": "G"}}, id="fill-id-not-f"),
    pytest.param({"spec_hash": "OTHER", "fill_model": {"fill_model_id": "F"}}, id="other-spec-hash"),
])
def test_check_report_refuses_inputs_that_do_not_bind_f_under_this_registration(tmp_path, body):
    _write_inputs(tmp_path, body)
    _refused("f_report_unbound", lambda: sel.check_report(_f_receipt(), tmp_path))


def test_check_report_accepts_inputs_bound_to_f_under_this_registration(tmp_path):
    _write_inputs(tmp_path, {"spec_hash": "SPEC", "fill_model": {"fill_model_id": "F"}})
    sel.check_report(_f_receipt(), tmp_path)


# ---- 5. the driver and run.py refuse an F spec's missing or unexpected inputs --------------

def test_driver_refuses_an_id_g_spec_before_any_write(tmp_path):
    out = tmp_path / "out"
    _refused("fill_model_id", lambda: driver.run(pnl.sources(), "inner", out, prereg=lambda: _f_receipt(_fill(id="G"))))
    assert not out.exists()


@pytest.mark.parametrize("stage", ["inner", "outer"])
def test_driver_refuses_an_f_spec_without_inputs_before_any_write(tmp_path, stage):
    out = tmp_path / "out"
    _refused("f_inputs_missing", lambda: driver.run(pnl.sources(), stage, out, prereg=_f_receipt))
    assert not out.exists()


def _pins_file(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"manifest_sha256": "0" * 64, "files": {"quotes/year=2024/month=02.parquet": "0" * 64}}), encoding="utf-8")
    return str(path)


def _quotes_factory(monkeypatch, result, name="exp189_fake_quotes"):
    module = types.ModuleType(name)
    module.make = lambda args: result
    monkeypatch.setitem(sys.modules, name, module)
    return f"{name}:make"


def _quotes_pair():
    return (qt._reader(qt._base(qt._frames())), "snap-1")


@pytest.mark.parametrize("only_pins", [True, False], ids=["pins-only", "quotes-only"])
def test_run_py_refuses_one_f_flag_without_the_other(tmp_path, monkeypatch, capsys, only_pins):
    run_py = _run_main()
    _gates(monkeypatch)
    _install(monkeypatch, lambda args: pnl.sources())
    flags = ["--f-pins", _pins_file(tmp_path)] if only_pins else ["--f-quotes", "exp189_fake_quotes:make"]
    assert run_py.main(_argv(tmp_path, *flags, sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC" and refused["reason"] == "f_inputs_incomplete"


def test_run_py_refuses_an_f_quotes_factory_that_does_not_resolve(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    _gates(monkeypatch)
    _install(monkeypatch, lambda args: pnl.sources())
    assert run_py.main(_argv(tmp_path, "--f-pins", _pins_file(tmp_path), "--f-quotes", "no_such_module_exp189_q:make",
                             sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["reason"] == "f_quotes_invalid" and refused["refused"] == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("result", [
    pytest.param("nope", id="not-a-pair"),
    pytest.param((lambda expiries: None,), id="one-item"),
    pytest.param((1, "snap-1"), id="reader-not-callable"),
])
def test_run_py_refuses_an_f_quotes_result_that_is_not_a_read_and_snapshot_pair(tmp_path, monkeypatch, capsys, result):
    run_py = _run_main()
    _gates(monkeypatch)
    _install(monkeypatch, lambda args: pnl.sources())
    quotes = _quotes_factory(monkeypatch, result)
    assert run_py.main(_argv(tmp_path, "--f-pins", _pins_file(tmp_path), "--f-quotes", quotes, sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC" and refused["reason"] == "f_quotes_invalid"


@pytest.mark.parametrize("stage", ["inner", "outer"])
def test_run_py_refuses_an_f_spec_with_no_f_flags_before_any_write(tmp_path, monkeypatch, capsys, stage):
    run_py = _run_main()
    monkeypatch.setattr(size_width_prereg, "require_preregistration", _f_receipt)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    _install(monkeypatch, lambda args: pnl.sources())
    assert run_py.main(_argv(tmp_path, sources=FAKE, stage=stage)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == CONFLICT and refused["reason"] == "f_inputs_missing"
    assert not (tmp_path / "out").exists()


def test_run_py_refuses_f_inputs_under_a_spec_that_selects_no_f(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    _gates(monkeypatch)
    _install(monkeypatch, lambda args: pnl.sources())
    quotes = _quotes_factory(monkeypatch, _quotes_pair())
    assert run_py.main(_argv(tmp_path, "--f-pins", _pins_file(tmp_path), "--f-quotes", quotes, sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == CONFLICT and refused["reason"] == "f_inputs_unexpected"


def test_run_py_report_of_an_f_spec_refuses_when_its_inputs_are_missing(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(size_width_prereg, "require_preregistration", _f_receipt)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")

    def boom(out_dir, **kwargs):
        raise AssertionError("report must not be assembled without inputs that bind F")

    monkeypatch.setattr(size_width_report, "assemble", boom)
    assert run_py.main(_argv(tmp_path, sources=NO_SOURCES, stage="report")) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == CONFLICT and refused["reason"] == "tuning_receipt_missing"


def test_run_py_report_under_a_non_f_spec_refuses_an_out_dir_whose_inputs_bind_f(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(size_width_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    _write_inputs(tmp_path / "out", {"spec_hash": "SPEC", "fill_model": {"fill_model_id": "F"}})

    def boom(out_dir, **kwargs):
        raise AssertionError("report must not be assembled over F inputs under a non-F registration")

    monkeypatch.setattr(size_width_report, "assemble", boom)
    assert run_py.main(_argv(tmp_path, sources=NO_SOURCES, stage="report")) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == CONFLICT and refused["reason"] == "f_report_unbound"


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


# ---- 7. run.py configures stderr logging; stdout stays the JSON document ----------------

def test_run_py_logs_progress_to_stderr_and_never_to_stdout(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    _gates(monkeypatch)
    monkeypatch.setattr(logging.root, "handlers", [])
    level = logging.root.level
    logging.root.setLevel(logging.WARNING)  # setLevel also clears logging's per-logger isEnabledFor cache
    try:
        ids = tmp_path / "ids.json"
        ids.write_text('{"a": 1}', encoding="utf-8")
        assert run_py.main(_argv(tmp_path, "--event-ids", str(ids))) == 2  # a quick refusal; main configures logging first
        logging.getLogger("size_width_f_quotes").info("probe-line")
        captured = capsys.readouterr()
        assert "probe-line" in captured.err and "probe-line" not in captured.out
    finally:
        for handler in list(logging.root.handlers):  # the root list is monkeypatched to [], so these are main's
            logging.root.removeHandler(handler)
            handler.close()
        logging.root.setLevel(level)
