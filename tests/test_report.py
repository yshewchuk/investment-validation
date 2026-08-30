"""engine.report — the standard report: sections, checklist honesty, provenance.

The generator is the only way a result becomes a record, so the tests pin the
contract consumers rely on: the fixed section order, the auto-evaluated
accuracy checklist (computed, not taken on faith), and a provenance block
complete enough to regenerate the report.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.evaluate import evaluate
from engine.report import (
    GENERATOR_VERSION,
    Report,
    accuracy_checklist,
    _file_fingerprint,
)


def _eval_result(tmp_path: Path, price_source="ORATS chains (engine.replay pricing)"):
    rng = np.random.default_rng(1)
    rets = rng.normal(0.02, 0.1, 80)
    dates = pd.date_range("2021-01-01", periods=len(rets), freq="12D")
    frames = []
    for a in (0.0, 0.5, 1.0):
        r = rets + (a - 0.5) * 0.04
        frames.append(pd.DataFrame({
            "event_id": [f"E{i}" for i in range(len(rets))],
            "ticker": "T", "event_date": dates,
            "entry_date": dates - pd.Timedelta(days=1),
            "exit_date": dates + pd.Timedelta(days=1),
            "fill_alpha": a, "entry_cost": 1.0,
            "exit_value": 1.0 + r, "ret": r,
        }))
    trades = pd.concat(frames, ignore_index=True)
    spec = {
        "id": "EXP-TEST", "title": "report probe",
        "strategy": "STR-THRU",
        "price_source": price_source,
        "primary_spec": {"x": 1},
        "walk_forward": {"min_train_years": 1},
        "preregistered_at": "2020-01-01T00:00:00+00:00",
    }
    return evaluate(spec, trades, run_dir=tmp_path, mc_paths=50,
                    stress=False, write_report=False)


SECTION_ORDER = [
    "## 1. Headline",
    "## 2. Equity curve",
    "## 3. By year",
    "## 4. Monte Carlo",
    "## 5. Stress battery",
    "## 6. Calibration",
    "## 7. Accuracy-evidence checklist",
    "## 8. Provenance",
    "## 9. Appendix",
]


class TestSectionOrder:
    def test_fixed_section_order(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result)
        path = report.write(tmp_path / "out")
        md = path.read_text()
        positions = [md.find(s) for s in SECTION_ORDER]
        assert all(p >= 0 for p in positions), f"missing section: {positions}"
        assert positions == sorted(positions), "sections out of order"

    def test_figures_written(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result)
        path = report.write(tmp_path / "out")
        figures = list((path.parent / "figures").glob("*.png"))
        assert figures, "no figures rendered"


class TestChecklist:
    def _results(self, tmp_path):
        return _eval_result(tmp_path).results

    def test_pass_fail_na_computed_not_asserted(self, tmp_path):
        spec_ok = {"price_source": "ORATS chains"}
        results = self._results(tmp_path)
        items = accuracy_checklist(results, spec_ok, ledger_path=None)
        status = {i.name: i.status for i in items}
        assert status["Real prices only"] == "PASS"
        assert status["Headline = walk-forward OOS"] == "PASS"
        assert status["Fill sensitivity"] == "PASS"
        assert status["Multiple-testing ledger"] == "FAIL"  # none attached
        assert status["Survivorship caveat"] == "PASS"

    def test_oquants_marks_fail_item_one(self, tmp_path):
        results = self._results(tmp_path)
        items = accuracy_checklist(results, {"price_source": "oquants fitted marks"},
                                   ledger_path=None)
        assert items[0].status == "FAIL"

    def test_missing_fill_sweep_fails_item_four(self, tmp_path):
        results = self._results(tmp_path)
        results["headline"]["alpha_sweep"] = {"0.50": {"mean": 0.01}}
        results["backtest"]["alpha_sweep"] = {"0.50": {"mean": 0.01}}
        items = accuracy_checklist(results, {"price_source": "polygon"}, ledger_path=None)
        by_name = {i.name: i for i in items}
        assert by_name["Fill sensitivity"].status == "FAIL"

    def test_ledger_cited_when_present(self, tmp_path):
        ledger = tmp_path / "LEDGER.csv"
        results = self._results(tmp_path)
        sha = results["spec_hash"]
        ledger.write_text("id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted\n"
                          f"EXP-1,{sha},2026-08-30,ran,0.02,0.5,False\n")
        items = accuracy_checklist(results, {"price_source": "orats"}, ledger_path=ledger)
        by_name = {i.name: i for i in items}
        assert by_name["Multiple-testing ledger"].status == "PASS"
        assert "1 spec(s) tried" in by_name["Multiple-testing ledger"].evidence


class TestRedBanner:
    def test_banner_when_any_fail(self, tmp_path):
        from engine.report import ChecklistItem

        context = {
            "kind": "evaluation", "spec": {}, "results": {}, "headline": {},
            "backtest": {},
            "checklist": [ChecklistItem("Real prices only", "FAIL", "probe")],
            "provenance": {}, "survivorship_note": "",
        }
        report = Report(context)
        assert report.any_fail
        md = report.write(tmp_path / "out").read_text()
        assert md.startswith("> **⚠ ACCURACY CHECKLIST HAS FAILING ITEMS")

    def test_no_banner_on_clean_context(self):
        context = {"kind": "evaluation", "spec": {}, "results": {}, "headline": {},
                   "backtest": {}, "checklist": [], "provenance": {},
                   "survivorship_note": ""}
        report = Report(context)
        assert not report.any_fail


class TestProvenance:
    def test_provenance_block_contents(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result, input_files=[tmp_path / "missing.csv"])
        prov = report.context["provenance"]
        assert prov["generator_version"] == GENERATOR_VERSION
        assert prov["spec_hash"] == result.results["spec_hash"]
        assert prov["code"], "no code hashes pinned"
        assert "engine/evaluate.py" in prov["code"]
        # A missing input file is recorded as missing, never dropped silently.
        assert any(f.get("missing") for f in prov["inputs"])

    def test_fingerprint_small_and_large(self, tmp_path):
        small = tmp_path / "small.csv"
        small.write_text("a,b\n1,2\n")
        fp = _file_fingerprint(small)
        assert "sha256" in fp and "note" not in fp

        large = tmp_path / "large.parquet"
        with open(large, "wb") as fh:
            fh.truncate(101 * (1 << 20))  # sparse >100MB
        fp = _file_fingerprint(large)
        assert "first_mb_sha256" in fp and fp.get("note")
        assert "sha256" not in fp


class TestDeterminism:
    def test_markdown_stable_modulo_timestamp(self, tmp_path):
        result = _eval_result(tmp_path)
        r1 = Report.from_eval(result)
        r2 = Report.from_eval(result)
        md1 = r1.write(tmp_path / "o1").read_text()
        md2 = r2.write(tmp_path / "o2").read_text()

        def strip_ts(md: str) -> str:
            return re.sub(r"Generated [0-9T:.+\-]+ by", "Generated X by", md)

        # Figures are referenced by relative name and carry no timestamps, so
        # the two renders must match once the wall-clock line is removed.
        assert strip_ts(md1) == strip_ts(md2)


class TestStressAndAppendixSections:
    def _result_with_stress(self, tmp_path):
        result = _eval_result(tmp_path)
        result.results["stress"] = {
            "regimes": {"2018Q4": {"n": 20, "mean": -0.02, "win_rate": 0.4},
                        "2022": {"n": 40, "mean": 0.05, "win_rate": 0.6}},
            "iv_regime": {"split_by": "spy_vol20 (per trade)",
                          "high": {"n": 30, "mean": 0.04},
                          "low": {"n": 50, "mean": 0.01}},
            "tail_injection": {"available": False, "required": True,
                               "note": "short-leg spec without a tail shock"},
            "slippage": {"available": False, "note": "no repricer supplied"},
            "stale_dates": {"available": True, "n_misdated": 3, "delta_mean": -0.004},
        }
        return result

    def test_stress_section_renders(self, tmp_path):
        result = self._result_with_stress(tmp_path)
        report = Report.from_eval(result)
        md = report.write(tmp_path / "out").read_text()
        assert "| 2018Q4 | 20 |" in md
        assert "IV-regime split" in md
        assert "Tail injection" in md and "REQUIRED and missing" in md
        assert "Slippage days: N/A" in md
        assert "Stale dates (1% mis-dated)" in md
        assert (tmp_path / "out" / "figures" / "stress_grid.png").exists()

    def test_grid_results_appendix(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result)
        report.context["grid_results"] = {"back_dte=7": {"mean": 0.01}}
        md = report.write(tmp_path / "out").read_text()
        assert "Grid / secondary results" in md
        assert "back_dte=7" in md

    def test_calibration_block(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result)
        report.context["calibration"] = {"brier": 0.24}
        md = report.write(tmp_path / "out").read_text()
        assert '"brier": 0.24' in md

    def test_equity_series_from_mapping(self, tmp_path):
        result = _eval_result(tmp_path)
        # The JSON-safe series shape evaluate() stores.
        series = result.results.get("equity_curve_series")
        assert series and len(series["date"]) > 1
        report = Report.from_eval(result)
        path = report.write(tmp_path / "out")
        assert (path.parent / "figures" / "equity_drawdown.png").exists()
