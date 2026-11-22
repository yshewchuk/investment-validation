"""EXP-188 S9c: run.py --stage report hands the plan frame to size_pnl_assemble.assemble (the assembler is stubbed)."""
import json

import pandas as pd

from experiments import size_pnl_assemble
from tests.v2.research.test_size_pnl_driver import (  # noqa: F401  (bounded is an autouse fixture)
    OK, PLAN_FRAME, _install_sources, _run_main, bounded, sources)
from tests.v2.research.test_size_pnl_driver_outer import _argv_for


def test_run_py_report_stage_calls_assemble_with_the_plan_frame(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    calls = []
    monkeypatch.setattr(size_pnl_assemble, "assemble",
                        lambda out_dir, plan, **kwargs: calls.append((out_dir, plan, kwargs)) or {"report": "r", "sections": 0, "rows": 0})
    _install_sources(monkeypatch, sources())
    assert run_py.main(_argv_for(tmp_path, "report")) == 0
    out_dir, plan, kwargs = calls[0]
    pd.testing.assert_frame_equal(plan, PLAN_FRAME)  # the plan frame itself: assemble derives the tickers and the binding from it
    assert kwargs == {"expected": sources().expected, "spec_hash": "SPEC", "snapshot": "snap-1"}  # the requested identity; no tickers or binding
    assert json.loads(capsys.readouterr().out)["sections"] == 0


def test_run_py_report_stage_refuses_event_ids_and_malformed_plans(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    ids = tmp_path / "ids.json"
    ids.write_text('["E1"]', encoding="utf-8")
    assert run_py.main(_argv_for(tmp_path, "report", "--event-ids", str(ids))) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "event_ids_not_supported"
    calls = []
    monkeypatch.setattr(size_pnl_assemble, "assemble", lambda *args, **kwargs: calls.append(args) or {"report": "r", "sections": 0, "rows": 0})
    # The real check_plan runs on the report path (the admission fixture is the one the scoring path accepts).
    malformed = {
        "missing_ticker_column": (PLAN_FRAME.drop(columns=["ticker"]), "plan_malformed"),
        "invalid_event_date": (PLAN_FRAME.assign(event_date="not a date"), "plan_malformed"),
        "numeric_ticker": (PLAN_FRAME.assign(ticker=1), "plan_malformed"),
        "duplicate_event_id": (pd.concat([PLAN_FRAME, PLAN_FRAME.iloc[:1]]), "duplicate_event_id"),
    }
    for name, (plan, reason) in malformed.items():
        _install_sources(monkeypatch, sources(plan_frame=plan))
        assert run_py.main(_argv_for(tmp_path, "report")) == 2, name
        printed = json.loads(capsys.readouterr().out)
        assert printed["refused"] == "EXPERIMENT_IDENTITY_CONFLICT", name
        assert printed["reason"] == reason, name
    assert calls == []  # no refused plan reaches assemble
