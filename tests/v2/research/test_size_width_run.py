"""EXP-189 runner (experiments/EXP-189_short_vol_width_tuning/run.py): argv surface, the prereg-first and not-bounded gates,
input refusals, the happy path with a fake sources module, and the report stage. Synthetic; run.py is loaded by path."""
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from engine.v2.ops.errors import fail
from experiments import size_width_driver, size_width_prereg, size_width_report
from tests.v2.research import test_size_pnl_driver as pnl

RUN = Path(__file__).resolve().parents[3] / "experiments" / "EXP-189_short_vol_width_tuning" / "run.py"
CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
OK = lambda: {"spec_hash": "SPEC", "spec_hash_checked": True}  # noqa: E731
NO_SOURCES = "no_such_module_exp189:make"
FAKE = "exp189_fake_sources:make"


def _run_main():
    spec = importlib.util.spec_from_file_location("exp189_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected(tmp_path):
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(pnl.EXPECTED), encoding="utf-8")
    return str(path)


def _argv(tmp_path, *extra, stage="inner", sources=NO_SOURCES):
    return ["--stage", stage, "--out-dir", str(tmp_path / "out"), "--expected", _expected(tmp_path),
            "--sources", sources, "--no-ledger", *extra]


def _install(monkeypatch, make):
    """Make ``make`` the factory behind the ``--sources`` spec FAKE."""
    module = types.ModuleType("exp189_fake_sources")
    module.make = make
    monkeypatch.setitem(sys.modules, "exp189_fake_sources", module)


def _gates(monkeypatch):
    """Prereg passes and the run is bounded."""
    monkeypatch.setattr(size_width_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")


# ---- argv surface --------------------------------------------------------------

def test_no_ledger_is_required(tmp_path):
    argv = ["--stage", "inner", "--out-dir", str(tmp_path), "--expected", _expected(tmp_path), "--sources", NO_SOURCES]
    with pytest.raises(SystemExit) as caught:
        _run_main().main(argv)
    assert caught.value.code == 2


def test_a_typo_stage_is_a_parser_error(tmp_path):
    run_py = _run_main()
    with pytest.raises(SystemExit) as caught:
        run_py.main(_argv(tmp_path, stage="outr"))
    assert caught.value.code == 2
    parsed = run_py._parser().parse_args(["--stage", "report", "--out-dir", "o", "--expected", "e", "--sources", NO_SOURCES, "--no-ledger"])
    assert parsed.stage == "report"


# ---- gates ----------------------------------------------------------------------

def test_prereg_refusal_runs_before_the_not_bounded_check_and_before_importing_sources(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    imported, real_import = [], importlib.import_module

    def refuse():
        raise fail(CONFLICT, "no planned row", details={"reason": "planned_row_missing"})

    def spy(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(size_width_prereg, "require_preregistration", refuse)
    monkeypatch.setattr(importlib, "import_module", spy)
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)  # would be not_bounded if reached
    assert run_py.main(_argv(tmp_path, sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == CONFLICT
    assert refused["reason"] == "planned_row_missing"
    assert imported == []


def test_not_bounded_refuses_before_the_sources_factory(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    calls = []

    def make(args):
        calls.append(args)
        return pnl.sources()

    monkeypatch.setattr(size_width_prereg, "require_preregistration", OK)
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)
    _install(monkeypatch, make)
    assert run_py.main(_argv(tmp_path, sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "RESOURCE_UNAVAILABLE"
    assert refused["reason"] == "not_bounded"
    assert calls == []


# ---- input refusals -------------------------------------------------------------

@pytest.mark.parametrize("payload", ['{"a": 1}', "[1, 2]"])  # not an array, and an array of non-strings
def test_event_ids_that_are_not_an_array_of_strings_refuse(tmp_path, monkeypatch, capsys, payload):
    run_py = _run_main()
    _gates(monkeypatch)
    ids = tmp_path / "ids.json"
    ids.write_text(payload, encoding="utf-8")
    assert run_py.main(_argv(tmp_path, "--event-ids", str(ids))) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "event_ids_invalid"


def test_event_ids_with_the_report_stage_refuse(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    _gates(monkeypatch)
    ids = tmp_path / "ids.json"
    ids.write_text('["E1"]', encoding="utf-8")

    def boom(out_dir, **kwargs):
        raise AssertionError("report must not run when event ids are refused")

    monkeypatch.setattr(size_width_report, "assemble", boom)
    assert run_py.main(_argv(tmp_path, "--event-ids", str(ids), stage="report")) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "event_ids_not_supported"


# ---- happy paths ----------------------------------------------------------------

def test_inner_prints_the_driver_summary_and_passes_the_receipt(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    _gates(monkeypatch)
    _install(monkeypatch, lambda args: pnl.sources())
    seen = {}

    def fake_run(src, stage, out_dir, *, event_ids=None, prereg):
        seen.update(stage=stage, event_ids=event_ids, spec_hash=prereg()["spec_hash"])
        return {"stage": "inner"}

    monkeypatch.setattr(size_width_driver, "run", fake_run)
    assert run_py.main(_argv(tmp_path, sources=FAKE)) == 0
    assert json.loads(capsys.readouterr().out) == {"stage": "inner"}
    assert seen == {"stage": "inner", "event_ids": None, "spec_hash": "SPEC"}


def test_report_assembles_without_reading_sources_or_the_bounded_env(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    calls, seen = [], []

    def make(args):
        calls.append(args)
        return pnl.sources()

    def fake_assemble(out_dir, **kwargs):
        seen.append((out_dir, kwargs))
        return {"stage": "report"}

    monkeypatch.setattr(size_width_prereg, "require_preregistration", OK)
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)  # report is not gated by the bounded check
    _install(monkeypatch, make)
    monkeypatch.setattr(size_width_report, "assemble", fake_assemble)
    assert run_py.main(_argv(tmp_path, sources=FAKE, stage="report")) == 0
    assert json.loads(capsys.readouterr().out) == {"stage": "report"}
    assert seen == [(str(tmp_path / "out"), {"expected": pnl.EXPECTED})]  # the --expected identities reach assemble
    assert calls == []
