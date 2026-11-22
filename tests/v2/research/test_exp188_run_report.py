"""EXP-188 S9c: run.py --stage report hands the plan's ticker map to size_pnl_assemble.assemble (the assembler is stubbed)."""
import json

from experiments import size_pnl_assemble
from tests.v2.research.test_size_pnl_driver import (  # noqa: F401  (bounded is an autouse fixture)
    OK, PLAN_FRAME, _install_sources, _run_main, bounded, sources)
from tests.v2.research.test_size_pnl_driver_outer import _argv_for


def test_run_py_report_stage_calls_assemble_with_the_plan_ticker_map(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    calls = []
    monkeypatch.setattr(size_pnl_assemble, "assemble",
                        lambda out_dir, tickers: calls.append((out_dir, tickers)) or {"report": "r", "sections": 0, "rows": 0})
    _install_sources(monkeypatch, sources())
    assert run_py.main(_argv_for(tmp_path, "report")) == 0
    out_dir, tickers = calls[0]
    assert tickers == {e: "T" for e in PLAN_FRAME["event_id"].astype(str)}  # the plan's ticker for every event
    assert json.loads(capsys.readouterr().out)["sections"] == 0
