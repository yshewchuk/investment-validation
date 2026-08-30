"""engine.report — the standard report: sections, checklist honesty, provenance.

The generator is the only way a result becomes a record, so the tests pin the
contract consumers rely on: the fixed section order, the auto-evaluated
accuracy checklist (computed, not taken on faith), and a provenance block
complete enough to regenerate the report.
"""
from __future__ import annotations

import json
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
    "## 0. Verdict",
    "## 1. Headline",
    "## 1.5 Sample funnel",
    "## 2. Equity curve",
    "## 3. By year",
    "## 4. Monte Carlo",
    "## 5. Stress battery",
    "## 6. Calibration",
    "## 7. Accuracy-evidence checklist",
    "## 8. Provenance",
    "## 9. Appendix",
    "## 10. Glossary",
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
        assert "| 2018Q4 | 20 | -2.00% |" in md
        assert "IV-regime split" in md
        assert "Tail injection" in md and "REQUIRED and missing" in md
        assert "**Slippage days:** N/A" in md
        assert "**Stale dates** (1% mis-dated): MEASURED" in md
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
        report.context["calibration"] = {
            "available": True, "n": 300, "base_rate": 0.4, "brier": 0.24,
            "brier_base_rate": 0.235, "brier_skill": -0.02,
            "reliability_monotonicity": 0.8,
            "deciles": [{"predicted": 0.3, "realized": 0.35, "n": 150},
                        {"predicted": 0.5, "realized": 0.45, "n": 150}],
        }
        md = report.write(tmp_path / "out").read_text()
        # A section, not a JSON dump: a verdict sentence, then a decile table.
        assert "{" not in md.split("## 6. Calibration")[1].split("## 7.")[0]
        assert "Brier skill" in md
        assert "| 1 | 30.0% | 35.0% | 150 | +5.0 pp |" in md

    def test_equity_series_from_mapping(self, tmp_path):
        result = _eval_result(tmp_path)
        # The JSON-safe series shape evaluate() stores.
        series = result.results.get("equity_curve_series")
        assert series and len(series["date"]) > 1
        report = Report.from_eval(result)
        path = report.write(tmp_path / "out")
        assert (path.parent / "figures" / "equity_drawdown.png").exists()


class TestChecklistHonestyRound2:
    def _results(self, audit):
        return {"spec_hash": "abc", "headline": {"alpha_sweep": None},
                "backtest": {"alpha_sweep": {"0.00": {"mean": -0.01},
                                              "0.50": {"mean": 0.00},
                                              "1.00": {"mean": 0.01}}},
                "headline_stage": "wf_oos",
                "walk_forward": {"audit": audit},
                "preregistration": {"valid": True}}

    def test_gateless_audit_is_na_not_pass(self, tmp_path):
        from engine.report import accuracy_checklist

        results = self._results({"years": [], "fit_years_seen": [], "leak_free": True})
        items = accuracy_checklist(results, {"price_source": "orats"}, ledger_path=None)
        by_name = {i.name: i for i in items}
        assert by_name["Leak audit ran"].status == "N/A"

    def test_fitted_audit_is_pass(self):
        from engine.report import accuracy_checklist

        results = self._results({"years": [2020, 2021], "fit_years_seen": [2020],
                                 "leak_free": True})
        items = accuracy_checklist(results, {"price_source": "orats"}, ledger_path=None)
        by_name = {i.name: i for i in items}
        assert by_name["Leak audit ran"].status == "PASS"

    def test_empty_ledger_is_na(self, tmp_path):
        from engine.report import accuracy_checklist

        ledger = tmp_path / "LEDGER.csv"
        ledger.write_text("id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted\n")
        results = self._results(None)
        items = accuracy_checklist(results, {"price_source": "orats"}, ledger_path=ledger)
        by_name = {i.name: i for i in items}
        assert by_name["Multiple-testing ledger"].status == "N/A"
        assert "no experiments tried yet" in by_name["Multiple-testing ledger"].evidence

    def test_spec_absent_from_ledger_is_fail(self, tmp_path):
        from engine.report import accuracy_checklist

        ledger = tmp_path / "LEDGER.csv"
        ledger.write_text(
            "id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted\n"
            "EXP-101,ffff,2026-08-30,ran,,,False\n")
        results = self._results(None)
        items = accuracy_checklist(results, {"price_source": "orats"}, ledger_path=ledger)
        by_name = {i.name: i for i in items}
        assert by_name["Multiple-testing ledger"].status == "FAIL"
        assert "never registered" in by_name["Multiple-testing ledger"].evidence


class TestNewFigures:
    def test_reliability_figure(self, tmp_path):
        from engine.report import fig_reliability

        cal = {"deciles": [{"predicted": 0.3, "realized": 0.28, "n": 50},
                           {"predicted": 0.6, "realized": 0.55, "n": 40}]}
        path = fig_reliability(cal, tmp_path / "rel.png", "reliability probe")
        assert path.exists()

    def test_mc_fan_paths_figure(self, tmp_path):
        from engine.report import fig_mc_fan_paths

        bands = {"p05": [1.0, 0.98, 0.95], "p50": [1.0, 1.01, 1.03],
                 "p95": [1.0, 1.04, 1.09]}
        path = fig_mc_fan_paths(bands, tmp_path / "fan.png", "fan probe")
        assert path.exists()

    def test_promotion_context_renders(self, tmp_path):
        from engine.report import Report

        context = {
            "kind": "promotion",
            "spec": {"id": "EXP-101", "title": "probe", "hypothesis": "h."},
            "headline": {"mean": 0.05, "sharpe_trade": 1.5, "n": 100},
            "results": {
                "champion": {"headline": {"mean": 0.03, "sharpe_trade": 1.2},
                             "mc": {"by_fraction": {"0.05": {"p_loss": 0.1}}}},
                "reasons": ["PASS (a1) probe"],
                "ledger_context": {"specs_tried": 3, "this_spec_rows": 2},
                "decision": "PROMOTED",
                "decided_at": "2026-08-30T00:00:00+00:00",
                "spec_hash": "abc",
            },
            "checklist": [],
            "provenance": {"generator_version": "1.0.0", "spec_hash": "abc",
                           "data_snapshot": "snap", "inputs": [], "code": {}},
            "survivorship_note": "",
        }
        path = Report(context).write(tmp_path / "out")
        md = path.read_text()
        assert "Promotion report" in md
        assert "3 spec(s) were tried" in md
        assert "| mean/trade | +5.00% | +3.00% |" in md


class TestCapacityAndDeploymentRendering:
    def _result(self):
        import numpy as np
        import pandas as pd
        from engine.evaluate import evaluate

        rng = np.random.default_rng(0)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="5D")
        rets = rng.normal(0.02, 0.1, n)
        legs = json.dumps({"entry": [{"name": "call", "bid": 1.0, "ask": 1.2}]})
        frames = []
        for a in (0.0, 0.5, 1.0):
            r = rets + (a - 0.5) * 0.06
            frames.append(pd.DataFrame({
                "event_id": [f"E{i}" for i in range(n)], "ticker": "T",
                "event_date": dates,
                "entry_date": dates - pd.Timedelta(days=1),
                "exit_date": dates + pd.Timedelta(days=1),
                "fill_alpha": a, "entry_cost": 1.0,
                "exit_value": 1.0 + r, "ret": r, "legs": legs,
            }))
        trades = pd.concat(frames, ignore_index=True)
        spec = {"id": "EXP-REND", "title": "render probe", "primary_spec": {"x": 1},
                "walk_forward": {"min_train_years": 1},
                "preregistered_at": "2020-01-01T00:00:00+00:00"}
        return evaluate(spec, trades, mc_paths=30, stress=False, write_report=False)

    def test_capacity_and_deployment_reach_the_markdown(self, tmp_path):
        from engine.report import Report

        result = self._result()
        path = Report.from_eval(result).write(tmp_path / "out")
        md = path.read_text()
        assert "**Capacity:**" in md
        assert "mean relative spread" in md
        assert "Deployment at 5% sizing" in md
        assert "UNCAPPED" in md


class TestFormattingContract:
    """One unit per number, one token for missing — the A0 contract."""

    def test_formatters(self):
        from engine.report import count, money_x, num, pct, pp, prob, ratio

        assert pct(0.0404) == "+4.04%"
        assert pct(-0.1902) == "-19.02%"
        assert prob(0.39781) == "39.8%"
        assert ratio(1.2438) == "1.24×"
        assert num(2.2336) == "2.23"
        assert count(7620) == "7,620"
        assert pp(0.021) == "+2.1 pp"
        assert money_x(341870.79) == "341,871×"
        assert money_x(4.55e16) == ">1e6×", "absurd terminals are clamped"

    def test_missing_is_a_word_not_a_dash(self):
        from engine.report import MISSING, num, pct, prob

        # A dash reads as a rendered number; "n/a" cannot be mistaken for one.
        assert pct(None) == MISSING == "n/a"
        assert prob(float("nan")) == "n/a"
        assert num(None) == "n/a"

    def test_every_metric_key_has_a_definition(self):
        from engine.evaluate import METRIC_KEYS
        from engine.report import COMPOSITE_METRIC_KEYS, METRIC_SPEC

        missing = [k for k in METRIC_KEYS
                   if k not in METRIC_SPEC and k not in COMPOSITE_METRIC_KEYS]
        assert not missing, f"metrics with no definition: {missing}"

    def test_no_bare_four_decimal_floats_in_the_body(self, tmp_path):
        result = _eval_result(tmp_path)
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        body = md.split("## 8. Provenance")[0]
        bare = re.findall(r"\|\s*-?\d+\.\d{4}\s*\|", body)
        assert not bare, f"unformatted floats in tables: {bare[:5]}"

    def test_no_bare_dash_cells(self, tmp_path):
        result = _eval_result(tmp_path)
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        assert not re.findall(r"\|\s*—\s*\|", md)


class TestVerdictBlock:
    def test_supported_when_positive_and_margin(self, tmp_path):
        result = _eval_result(tmp_path)
        v = Report.from_eval(result).verdict
        assert v.call.startswith("SUPPORTED") or v.call.startswith("MIXED")
        questions = [q for q, _a, _w in v.rows]
        assert "Does it make money at mid fills?" in questions
        assert "How much fill quality does it need?" in questions

    def test_not_supported_when_negative(self, tmp_path):
        from engine.report import verdict

        results = {"headline": {"mean": -0.02, "n": 100, "breakeven_alpha": 0.8,
                                "by_year": {}}}
        v = verdict(results, {"type": "confirmatory"})
        assert v.call.startswith("NOT SUPPORTED")

    def test_descriptive_spec_gets_no_passfail(self):
        from engine.report import verdict

        results = {"headline": {"mean": 0.02, "n": 100, "breakeven_alpha": 0.4,
                                "by_year": {}}}
        v = verdict(results, {"type": "descriptive"})
        assert v.call.startswith("DESCRIPTIVE")
        v2 = verdict(results, {"hypothesis": "Descriptive measurement, no target.",
                               "promotion_target": None})
        assert v2.call.startswith("DESCRIPTIVE"), "inferred from the hypothesis"

    def test_warnings_are_computed_not_passed(self):
        from engine.report import compute_warnings

        results = {
            "headline": {"max_concurrency": 133, "by_year": {"2020": {"n": 5, "mean": 0.01}},
                         "breakeven_alpha": 0.47},
            "mc": {"by_fraction": {"0.05": {"p_loss": 0.0}}},
            "stress": {"slippage": {"available": True,
                                    "shifts": {"-1d": {"coverage": 0.001,
                                                       "delta_mean": float("nan")}}}},
        }
        warns = compute_warnings(results, {"brier_skill": -0.084})
        joined = " ".join(warns)
        assert "Brier skill" in joined
        assert "INCONCLUSIVE" in joined
        assert "133-position overlap" in joined
        assert "n < 30" in joined
        assert "margin below mid" in joined

    def test_falsifier_quoted_verbatim(self):
        from engine.report import verdict

        hyp = "The gate lifts the mean. Falsified if the OOS lift is not positive."
        v = verdict({"headline": {"mean": 0.01, "n": 10, "breakeven_alpha": 0.3,
                                  "by_year": {}}},
                    {"type": "confirmatory", "hypothesis": hyp})
        answers = [a for _q, a, _w in v.rows]
        assert "Falsified if the OOS lift is not positive." in answers


class TestSampleFunnel:
    def test_upstream_stages_and_headline_marker(self, tmp_path):
        result = _eval_result(tmp_path)
        result.spec["universe_funnel"] = [
            {"stage": "calendar events in window", "events": 98705, "note": "2018-2026"},
            {"stage": "both legs' chains cached", "events": 17754, "note": "chain availability"},
        ]
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        section = md.split("## 1.5 Sample funnel")[1].split("## 2.")[0]
        assert "98,705" in section and "17,754" in section
        assert "← **headline**" in section

    def test_absent_funnel_says_so(self, tmp_path):
        from engine.report import Report as R

        md_lines: list[str] = []
        R({"funnel": []})._render_funnel(md_lines)
        assert "No funnel supplied" in "\n".join(md_lines)


class TestStressStates:
    def test_low_coverage_is_inconclusive(self):
        from engine.report import stress_state

        state, why = stress_state({"available": True}, coverage=0.001)
        assert state == "INCONCLUSIVE" and "has not been performed" in why

    def test_nan_statistic_is_inconclusive(self):
        from engine.report import stress_state

        state, _ = stress_state({"available": True}, coverage=0.9,
                                statistic=float("nan"))
        assert state == "INCONCLUSIVE"

    def test_measured_when_covered(self):
        from engine.report import stress_state

        assert stress_state({"available": True}, coverage=0.9, statistic=0.01)[0] == "MEASURED"

    def test_dead_stress_renders_as_dead(self, tmp_path):
        result = _eval_result(tmp_path)
        result.results["stress"] = {
            "slippage": {"available": True, "base_mean": 0.02,
                         "shifts": {"-1d": {"coverage": 0.001, "n": 3,
                                            "mean": float("nan"),
                                            "delta_mean": float("nan")}}},
            "stale_dates": {"available": True, "n_misdated": 76,
                            "coverage": 0.001, "delta_mean": float("nan")},
        }
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        assert "**Slippage -1d:** INCONCLUSIVE" in md
        assert "**Stale dates** (1% mis-dated): INCONCLUSIVE" in md
        assert "Δmean n/a" not in md, "a dead stage must not render a value at all"


class TestExtraSections:
    def test_extra_sections_render_in_order(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result, extra_sections=[
            {"title": "Max-loss distribution vs net debit",
             "note": "80 of 4,736 trades lost more than the debit.",
             "columns": ["classification", "count"], "align": ["---", "---:"],
             "rows": [["real_loss", "67"], ["data_artifact", "13"]],
             "falsifies": "any exceedance that is not a data artifact."},
        ])
        md = report.write(tmp_path / "out").read_text()
        assert "## 8.5 Additional analyses" in md
        assert "### 8.5.1 Max-loss distribution vs net debit" in md
        assert "| real_loss | 67 |" in md
        assert md.find("## 8.5") < md.find("## 9. Appendix") < md.find("## 10. Glossary")


class TestGlossary:
    def test_glossary_covers_every_rendered_metric(self, tmp_path):
        from engine.report import GLOSSARY, METRIC_SPEC

        result = _eval_result(tmp_path)
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        glossary = md.split("## 10. Glossary")[1]
        for _key, (_f, label, definition) in METRIC_SPEC.items():
            if definition:
                assert f"| {label} |" in glossary, f"{label} missing from the glossary"
        for term in GLOSSARY:
            assert term in glossary


class TestAdvisories:
    def test_advisories_flag_calibration_and_concurrency(self, tmp_path):
        result = _eval_result(tmp_path)
        result.results["headline"]["max_concurrency"] = 133
        report = Report.from_eval(result)
        report.context["calibration"] = {"available": True, "brier_skill": -0.084,
                                         "base_rate": 0.4, "brier": 0.26,
                                         "brier_base_rate": 0.24, "deciles": []}
        md = report.write(tmp_path / "out").read_text()
        advisory = md.split("**Advisory**")[1].split("## 8.")[0]
        assert "Calibration state" in advisory and "WARN" in advisory
        assert "Concurrency vs sizing" in advisory

    def test_advisories_never_block(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result)
        report.context["calibration"] = {"available": True, "brier_skill": -0.5,
                                         "deciles": []}
        before = {i.name for i in report.context["checklist"] if i.status == "FAIL"}
        report.write(tmp_path / "out")
        after = {i.name for i in report.context["checklist"] if i.status == "FAIL"}
        assert before == after, "an advisory must not turn into a checklist FAIL"


class TestPromotedVerdictRows:
    def test_extra_section_can_reach_the_verdict(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result, extra_sections=[
            {"title": "Max-loss vs debit", "body": ["…"],
             "promote_to_verdict": True,
             "verdict_row": ("Is CAL-P defined-risk?", "**No** — 80 of 4,736", "§8.5.1")},
        ])
        md = report.write(tmp_path / "out").read_text()
        verdict_block = md.split("## 0. Verdict")[1].split("## 1.")[0]
        assert "Is CAL-P defined-risk?" in verdict_block

    def test_falsifier_row_stays_last(self, tmp_path):
        result = _eval_result(tmp_path)
        result.spec["hypothesis"] = "H. Falsified if the lift is not positive."
        report = Report.from_eval(result, extra_sections=[
            {"title": "x", "promote_to_verdict": True,
             "verdict_row": ("Q?", "A", "§8.5.1")},
        ])
        md = report.write(tmp_path / "out").read_text()
        block = md.split("## 0. Verdict")[1].split("## 1.")[0]
        assert block.index("| Q? |") < block.index("What would falsify it?")


class TestPnlCalibration:
    """Win-rate calibration and P&L calibration are different claims."""

    CAL = {"available": True, "n": 210, "base_rate": 0.33, "brier": 0.226,
           "brier_base_rate": 0.221, "brier_skill": -0.025,
           "reliability_monotonicity": 0.17,
           "predicted_mean_pnl": 0.0408, "realized_mean_pnl": -0.0009,
           "deciles": [{"predicted": 0.25, "realized": 0.24, "n": 21}]}

    def test_pnl_gap_warns_even_when_brier_passes(self):
        from engine.report import compute_warnings

        warns = compute_warnings({"headline": {"by_year": {}}}, self.CAL)
        assert any("expected P&L misses realized" in w for w in warns), warns

    def test_pnl_gap_reaches_the_verdict_and_the_section(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result)
        report.context["calibration"] = self.CAL
        md = report.write(tmp_path / "out").read_text()
        assert "Does expected P&L match what happened?" in md
        assert "does NOT match" in md
        assert "Expected vs realized P&L" in md

    def test_matching_pnl_reads_ok(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result)
        report.context["calibration"] = dict(self.CAL, predicted_mean_pnl=0.021,
                                             realized_mean_pnl=0.019)
        md = report.write(tmp_path / "out").read_text()
        assert "**Expected P&L matches realized P&L:**" in md


class TestTransactionLog:
    """The chart's audit trail is announced where the chart is."""

    def test_log_line_and_reconciliation_render(self, tmp_path):
        result = _eval_result(tmp_path)
        report = Report.from_eval(result)
        md = report.write(tmp_path / "out").read_text()
        section = md.split("## 2. Equity curve")[1].split("## 3.")[0]
        assert "Transaction log" in section
        assert "*Reconciled:" in section

    def test_a_log_that_does_not_add_up_gets_a_banner(self, tmp_path):
        result = _eval_result(tmp_path)
        result.results["transaction_log"] = {
            "rows": 80, "reconciles": False, "implied_final": 1.4,
            "final_equity": 2.1, "abs_error": 0.7, "path": "results/x.csv"}
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        assert "does NOT reconcile" in md

    def test_the_rederivation_recipe_is_printed(self, tmp_path):
        # A spot-checker who marks per event instead of per date gets a deeper
        # drawdown than the chart and thinks the report is wrong; the recipe is
        # what stops that being a bug report.
        result = _eval_result(tmp_path)
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        section = md.split("## 2. Equity curve")[1].split("## 3.")[0]
        assert "exits before entries" in section
        assert "one mark per date" in section
        assert "figures/equity_drawdown.json" in section


class TestCellEscaping:
    def test_a_pipe_in_free_text_does_not_split_the_row(self, tmp_path):
        from engine.report import cell

        assert cell("max |Δ| 3.6e-15") == r"max \|Δ\| 3.6e-15"

        result = _eval_result(tmp_path)
        report = Report.from_eval(result, extra_sections=[
            {"title": "escaping", "columns": ["check", "result"],
             "rows": [["rebuilt", "max |Δ| 3.6e-15"]]},
        ])
        md = report.write(tmp_path / "out").read_text()
        row = [ln for ln in md.splitlines() if ln.startswith("| rebuilt |")][0]
        # Two columns declared, so exactly two separators inside the row.
        assert row.count("|") - row.count(r"\|") == 3, row


class TestConcentration:
    """A mean return cannot say whether ten trades carried the curve."""

    def _result(self, tmp_path, share=0.8):
        result = _eval_result(tmp_path)
        result.results["transaction_log"] = {
            "rows": 500, "reconciles": True, "implied_final": 3.0,
            "final_equity": 3.0, "abs_error": 0.0, "path": "results/x.csv",
            "concentration": {"top10_net_share": share, "top1_net_share": share / 3,
                              "trades_for_half_the_gains": 7, "n_winners": 210},
        }
        return result

    def test_concentration_line_renders(self, tmp_path):
        md = Report.from_eval(self._result(tmp_path)).write(tmp_path / "out").read_text()
        section = md.split("## 2. Equity curve")[1].split("## 3.")[0]
        assert "**Concentration:**" in section
        assert "80.0% of the net result" in section
        assert "7 of 210 winning trades" in section

    def test_heavy_concentration_warns(self, tmp_path):
        report = Report.from_eval(self._result(tmp_path))
        md = report.write(tmp_path / "out").read_text()
        assert "10 trades carry 80.0%" in md.split("## 1.")[0]

    def test_spread_out_gains_do_not_warn(self, tmp_path):
        report = Report.from_eval(self._result(tmp_path, share=0.2))
        verdict_block = report.write(tmp_path / "out").read_text().split("## 1.")[0]
        assert "trades carry" not in verdict_block


class TestCapitalWeightedAndUngatedDisclosure:
    """The two readings that separate a trade edge from a sizing artifact."""

    def test_equal_vs_capital_weighted_gap_is_called_out(self, tmp_path):
        result = _eval_result(tmp_path)
        result.results["headline"]["dollar_weighted"] = 0.005
        result.results["headline"]["mean"] = 0.035
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        assert "Equal-weighted +3.50% vs capital-weighted +0.50%" in md
        assert "the edge sits in the cheapest contracts" in md.split("## 1.")[0]

    def test_matching_readings_do_not_warn(self, tmp_path):
        result = _eval_result(tmp_path)
        result.results["headline"]["dollar_weighted"] = 0.034
        result.results["headline"]["mean"] = 0.035
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        assert "cheapest contracts" not in md.split("## 1.")[0]

    def test_ungated_share_is_disclosed_and_warns(self, tmp_path):
        result = _eval_result(tmp_path)
        result.results["headline"]["ungated_share"] = 0.709
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        assert "70.9% of these trades come from ungated years" in md
        assert "base exposure, not the gate" in md.split("## 1.")[0]

    def test_wide_market_concentration_warns(self, tmp_path):
        result = _eval_result(tmp_path)
        result.results["headline"]["capacity"] = {
            "available": True, "mean_rel_spread": 0.36, "p95_rel_spread": 1.6,
            "note": "spread-based capacity only",
            "pnl_by_spread": {"widest_quintile_share": 1.155,
                              "tightest_two_quintiles_share": -1.089,
                              "median_rel_spread_widest": 0.481,
                              "median_rel_spread_tightest": 0.042},
        }
        md = Report.from_eval(result).write(tmp_path / "out").read_text()
        assert "Where the P&L sits, by quoted width" in md
        assert "widest-quoted fifth" in md.split("## 1.")[0]
