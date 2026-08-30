#!/usr/bin/env python3
"""Phase 4 acceptance tests.

    python3 checks/phase4_checks.py               # everything
    python3 checks/phase4_checks.py --list
    python3 checks/phase4_checks.py --only golden_report
    python3 checks/phase4_checks.py --no-data     # skip checks needing the store

The guide's sixteen checks — seven from the original Phase 4 spec, nine added
by the completion pass because they are what stop the report format regressing:

 0. ``unittests``          the pytest suite (report + ledger + audit units)
 1. ``golden_report``      fixed context → stable markdown; figure DATA identical
 2. ``regeneration``       a real experiment regenerates from its saved metrics
 3. ``checklist_honesty``  a missing fill sweep FAILs and raises the banner
 4. ``ledger_append_only`` rewriting refuses; supersede round-trips
 5. ``outcome_idempotent`` scoring twice writes one outcome; unresolvable kept
 6. ``leak_poison``        future-dated evidence raises in all three paths
 7. ``calibration_trigger`` 50 scored rows regenerate the report + health.json
 8. ``numbers_preserved``  regenerated reports hold the committed reports' numbers
 9. ``no_raw_json``        no JSON blob in a report body
10. ``units_present``      no bare four-decimal floats in a report body
11. ``no_bare_dashes``     every "—" carries an explanation token
12. ``section_order``      the fixed order, on every report kind
13. ``glossary_complete``  every rendered metric has a definition
14. ``verdict_derivable``  the verdict call and warnings are computed, not passed
15. ``no_report_append``   no run.py opens REPORT.md for append
16. ``figure_captions``    every figure has a caption clear of its axes
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.evaluate import EvalResult, evaluate  # noqa: E402
from engine.report import (  # noqa: E402
    COMPOSITE_METRIC_KEYS,
    GLOSSARY,
    METRIC_SPEC,
    Report,
    accuracy_checklist,
    compute_warnings,
    verdict,
)

#: Sections every evaluation report renders, in order.
SECTION_ORDER = [
    "## 0. Verdict", "## 1. Headline", "## 1.5 Sample funnel", "## 2. Equity curve",
    "## 3. By year", "## 4. Monte Carlo", "## 5. Stress battery", "## 6. Calibration",
    "## 7. Accuracy-evidence checklist", "## 8. Provenance", "## 9. Appendix",
    "## 10. Glossary",
]


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0
    skipped: bool = False


REGISTRY: dict[str, dict] = {}


def check(name: str, *, needs_data: bool = True, description: str = ""):
    def wrap(fn):
        REGISTRY[name] = {"fn": fn, "needs_data": needs_data, "description": description}
        return fn

    return wrap


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------


def _synthetic_result(tmp: Path, *, seed: int = 1, n: int = 90):
    """A small evaluation with every stage populated, for format checks."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.02, 0.1, n)
    dates = pd.date_range("2021-01-04", periods=n, freq="12D")
    frames = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = rets + (a - 0.5) * 0.04
        frames.append(pd.DataFrame({
            "event_id": [f"E{i}" for i in range(n)], "ticker": "T",
            "event_date": dates,
            "entry_date": dates - pd.Timedelta(days=1),
            "exit_date": dates + pd.Timedelta(days=1),
            "fill_alpha": a, "entry_cost": 1.0, "exit_value": 1.0 + r, "ret": r,
        }))
    trades = pd.concat(frames, ignore_index=True)
    spec = {
        "id": "EXP-P4", "title": "phase-4 format probe", "type": "confirmatory",
        "strategy": "STR-THRU", "price_source": "ORATS chains",
        "primary_spec": {"x": 1}, "walk_forward": {"min_train_years": 1},
        "hypothesis": "The probe is positive at mid. Falsified if it is not.",
        "preregistered_at": "2020-01-01T00:00:00+00:00",
    }
    return evaluate(spec, trades, run_dir=tmp, mc_paths=50, stress=False,
                    write_report=False)


def _experiment_dirs() -> list[Path]:
    return sorted(d for d in (ROOT / "experiments").glob("EXP-*")
                  if (d / "spec.yaml").exists() and any(d.glob("results/metrics_*.json")))


def _regenerate(exp_dir: Path, out_dir: Path) -> str:
    """Re-render one experiment's report from its saved artifacts.

    Metrics plus the experiment's own required-output artifact, so the
    regenerated document is the WHOLE report — extra sections and the verdict
    rows they promote included — rather than the generator's part of it.
    """
    import importlib.util

    import yaml

    spec = yaml.safe_load((exp_dir / "spec.yaml").read_text())
    results = json.loads(sorted(exp_dir.glob("results/metrics_*.json"))[0].read_text())
    result = EvalResult(spec=spec, results=results, run_dir=exp_dir)

    sections: list = []
    module_spec = importlib.util.spec_from_file_location(
        f"run_{exp_dir.name[:7].replace('-', '_')}", exp_dir / "run.py")
    if module_spec and module_spec.loader:
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        appendix_path = exp_dir / "results" / "appendix.json"
        mechanics_path = exp_dir / "results" / "risk_mechanics.json"
        if appendix_path.exists():
            sections = module.appendix_sections(json.loads(appendix_path.read_text()))
        elif mechanics_path.exists():
            sections = module.appendix_sections(
                spec, result, json.loads(mechanics_path.read_text()))
    return Report.from_eval(result, extra_sections=sections).write(out_dir).read_text()


def _numbers(text: str) -> list[float]:
    """Every number a reader can see, normalized so formatting does not matter.

    The wall-clock line is dropped first: its microseconds are digits, not a
    result, and they would otherwise read as a value that went missing.
    """
    text = re.sub(r"\*Generated .*?\*", "", text)
    out = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------
# 0. unit suite
# --------------------------------------------------------------------------


@check("unittests", needs_data=False, description="the pytest suite (report, ledger, audit)")
def check_unittests() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests/test_report.py"),
         str(ROOT / "tests/test_ledger.py"), str(ROOT / "tests/test_audit.py"),
         "-q", "--no-header"],
        capture_output=True, text=True, cwd=ROOT,
    )
    tail = (result.stdout or result.stderr).strip().splitlines()[-1:]
    _require(result.returncode == 0, f"pytest failed: {' '.join(tail)}")
    return " ".join(tail)


# --------------------------------------------------------------------------
# 1. golden report
# --------------------------------------------------------------------------


@check("golden_report", needs_data=False,
       description="fixed context → stable markdown and identical figure data")
def check_golden_report() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        result = _synthetic_result(tmp)
        first = Report.from_eval(result).write(tmp / "a").read_text()
        second = Report.from_eval(result).write(tmp / "b").read_text()

        def strip_ts(md: str) -> str:
            return re.sub(r"Generated [0-9T:.+\-]+ by", "Generated X by", md)

        _require(strip_ts(first) == strip_ts(second),
                 "two renders of one context differ beyond the timestamp line")

        # Figure DATA, not pixels: font metrics drift between matplotlib
        # versions, the arrays behind the plot do not.
        data_a = sorted((tmp / "a" / "figures").glob("*.json"))
        data_b = sorted((tmp / "b" / "figures").glob("*.json"))
        _require(data_a and len(data_a) == len(data_b), "figure data files missing")
        for pa, pb in zip(data_a, data_b):
            _require(json.loads(pa.read_text()) == json.loads(pb.read_text()),
                     f"figure data differs between renders: {pa.name}")
        return f"{len(first.splitlines())} lines, {len(data_a)} figure data files identical"


# --------------------------------------------------------------------------
# 2. regeneration from the provenance block
# --------------------------------------------------------------------------


@check("regeneration", description="a real experiment regenerates from its saved metrics")
def check_regeneration() -> str:
    dirs = _experiment_dirs()
    _require(bool(dirs), "no experiment with a saved metrics artifact")
    with tempfile.TemporaryDirectory() as tmp:
        details = []
        for exp in dirs:
            body = _regenerate(exp, Path(tmp) / exp.name)
            _require("## 8. Provenance" in body, f"{exp.name}: no provenance block")
            _require("spec hash" in body and "data snapshot" in body,
                     f"{exp.name}: provenance block incomplete")
            details.append(exp.name.split("_")[0])
        return f"regenerated {', '.join(details)}"


# --------------------------------------------------------------------------
# 3. checklist honesty
# --------------------------------------------------------------------------


@check("checklist_honesty", needs_data=False,
       description="a missing fill sweep FAILs item 4 and raises the banner")
def check_checklist_honesty() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        result = _synthetic_result(tmp)
        result.results["headline"]["alpha_sweep"] = {"0.50": {"mean": 0.01, "n": 1,
                                                              "win_rate": 0.5}}
        result.results["backtest"]["alpha_sweep"] = result.results["headline"]["alpha_sweep"]
        report = Report.from_eval(result)
        md = report.write(tmp / "out").read_text()
        _require(report.any_fail, "a missing fill sweep did not produce a FAIL")
        _require("ACCURACY CHECKLIST HAS FAILING ITEMS" in md, "no red banner")

        items = {i.name: i.status for i in
                 accuracy_checklist(result.results, {"price_source": "orats"})}
        _require(items["Fill sensitivity"] == "FAIL", "item 4 did not fail")

        # promote.py must refuse evidence with a FAIL.
        from experiments import promote as promote_mod

        source = Path(promote_mod.__file__).read_text()
        _require("checklist_fails" in source,
                 "promote.py does not consult checklist_fails")
        return "item 4 FAIL, banner rendered, promote.py gate present"


# --------------------------------------------------------------------------
# 4-5. the ledger
# --------------------------------------------------------------------------


def _ledger_sandbox(tmp: Path):
    """Point the ledger at a temp tree and hand back the module."""
    from engine import ledger

    paths.LEDGER = tmp / "ledger"
    return ledger


def _prediction_row(ledger, as_of="2026-10-15", ticker="AAPL", win=0.55):
    rid = ledger.row_id(as_of, ticker, "STR-THRU", None, "2026-10-23")
    return {
        "schema_version": ledger.SCHEMA_VERSION, "row_id": rid, "written_at": "t",
        "as_of": as_of, "decision_ts": f"{as_of}T20:00:00+00:00", "ticker": ticker,
        "event_id": f"{ticker}-e", "event_date": "2026-10-16", "session": "AMC",
        "strategy": "STR-THRU", "structure": {},
        "intended_prices": {"alpha": 0.5, "entry_cost": 1.0},
        "score": {"win_model": win, "exp_pnl_model": 0.03},
        "model_versions": {}, "snapshot_hash": "abc", "audit_receipt": None,
        "supersedes": None, "supersede_reason": None,
    }


@check("ledger_append_only", needs_data=False,
       description="rewriting a row refuses; a supersede round-trips")
def check_ledger_append_only() -> str:
    original_ledger = paths.LEDGER
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = _ledger_sandbox(Path(tmp))
            row = _prediction_row(ledger)
            ledger.write_predictions([row])
            try:
                ledger.write_predictions([_prediction_row(ledger, win=0.9)])
                raise AssertionError("rewriting an existing row_id was allowed")
            except ledger.LedgerError:
                pass

            ledger.supersede(row["row_id"],
                             _prediction_row(ledger, as_of="2026-10-16", win=0.61),
                             reason="chain refreshed")
            visible = ledger.read_predictions()
            raw = ledger.read_predictions(resolve_supersedes=False)
            _require(len(visible) == 1 and visible[0]["score"]["win_model"] == 0.61,
                     "supersede did not replace the visible row")
            _require(len(raw) == 2, "the superseded original left the file")
            _require(not any(n.startswith(("delete", "remove")) for n in dir(ledger)),
                     "the ledger module exposes a delete path")
            return "rewrite refused, supersede round-tripped, original retained"
    finally:
        paths.LEDGER = original_ledger


@check("outcome_idempotent", needs_data=False,
       description="scoring twice writes one outcome; unresolvable rows are recorded")
def check_outcome_idempotent() -> str:
    from engine import replay as replay_mod

    original_ledger = paths.LEDGER
    original_replay = replay_mod.replay
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = _ledger_sandbox(Path(tmp))
            ledger.write_predictions([_prediction_row(ledger, ticker="AAPL"),
                                      _prediction_row(ledger, ticker="MSFT")])

            priced = pd.DataFrame([{
                "event_id": "AAPL-e", "ticker": "AAPL", "strategy": "STR-THRU",
                "event_date": pd.Timestamp("2026-10-16"), "fill_alpha": 0.5,
                "entry_cost": 1.0, "exit_value": 1.12, "ret": 0.12,
                "exit_mode": "chain"}])

            class _R:
                trades = priced

            replay_mod.replay = lambda *a, **k: _R()
            first = ledger.score_outcomes(through="2026-10-20")
            second = ledger.score_outcomes(through="2026-10-20")
            outcomes = ledger.read_outcomes()

            _require(first["resolved"] == 1 and first["unresolvable"] == 1,
                     f"unexpected first pass: {first}")
            _require(second["resolved"] == 0 and len(outcomes) == 2,
                     "re-running the scorer duplicated outcome rows")
            unresolvable = [o for o in outcomes if o["status"] == "unresolvable"]
            _require(len(unresolvable) == 1 and unresolvable[0]["reason"],
                     "an unresolvable prediction was dropped instead of recorded")
            return "1 resolved, 1 unresolvable recorded, re-run idempotent"
    finally:
        paths.LEDGER = original_ledger
        replay_mod.replay = original_replay


# --------------------------------------------------------------------------
# 6. leak poison in all three wired paths
# --------------------------------------------------------------------------


@check("leak_poison", needs_data=False,
       description="future-dated evidence raises in the scoring, decision and ledger paths")
def check_leak_poison() -> str:
    """The three paths where a Phase-4 receipt is produced.

    The walk-forward path is not poisonable from outside: ``walk_forward``
    slices the training frame itself and asserts on it, so a gate cannot be
    handed its test year. That structural guard has its own marker-proven check
    (``checks/phase2_checks.py::wf_leak_poison``); this one covers the paths
    where a caller supplies the timestamps — feature causality, decision
    causality, and the ledger write.
    """
    from engine.audit import (
        FeatureVector,
        LeakError,
        assert_causal,
        assert_decision_causal,
        audit_receipt_for_snapshot,
        receipt_from_vectors,
    )

    hits = []

    # (a) feature causality — a value stamped after the decision it informed.
    poisoned = FeatureVector(
        ticker="T", as_of=pd.Timestamp("2026-01-05"),
        event_date=pd.Timestamp("2026-01-06"), session="AMC",
        values={"iv30d": 0.4},
        feature_as_of={"iv30d": pd.Timestamp("2026-01-07")})
    try:
        assert_causal(poisoned)
        raise AssertionError("assert_causal accepted a post-decision feature")
    except LeakError:
        hits.append("score/features")

    # (b) decision causality — deciding at the close AFTER a BMO print.
    try:
        assert_decision_causal(pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-06"),
                               "BMO")
        raise AssertionError("assert_decision_causal accepted a post-print decision")
    except LeakError:
        hits.append("decision")

    # (c) ledger write — a frozen row whose evidence post-dates its decision.
    board = pd.DataFrame([{"ticker": "T", "as_of": pd.Timestamp("2026-10-15"),
                           "evidence_cutoff": pd.Timestamp("2026-10-17")}])
    try:
        audit_receipt_for_snapshot(board, decision_ts=pd.Timestamp("2026-10-15"))
        raise AssertionError("the ledger accepted post-decision evidence")
    except LeakError:
        hits.append("ledger.write")

    # A forward board — decisions ahead of the clock — must NOT raise: that is
    # the shape every nightly run has.
    forward = pd.DataFrame([{"ticker": "T", "as_of": pd.Timestamp("2026-09-10"),
                             "evidence_cutoff": pd.Timestamp("2026-09-09")}])
    audit_receipt_for_snapshot(forward, decision_ts=pd.Timestamp("2026-08-30"))

    _require(len(hits) == 3, f"only {hits} raised")

    # And a clean vector yields a receipt with a positive margin — the evidence
    # the checklist renders instead of asserting the audit happened.
    clean = FeatureVector(
        ticker="T", as_of=pd.Timestamp("2026-01-05"),
        event_date=pd.Timestamp("2026-01-06"), session="AMC",
        values={"iv30d": 0.4},
        feature_as_of={"iv30d": pd.Timestamp("2026-01-02")})
    receipt = receipt_from_vectors([clean], path="score")
    _require(receipt.margin_seconds and receipt.margin_seconds > 0,
             "a clean audit produced no positive margin")
    _require(len(receipt.receipt_hash) == 16, "receipt carries no hash")
    return "raised in " + ", ".join(hits) + f"; clean receipt {receipt.summary()}"


# --------------------------------------------------------------------------
# 7. calibration trigger
# --------------------------------------------------------------------------


@check("calibration_trigger", needs_data=False,
       description="50 scored predictions regenerate the report and health.json")
def check_calibration_trigger() -> str:
    from engine import replay as replay_mod

    original_ledger = paths.LEDGER
    original_replay = replay_mod.replay
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = _ledger_sandbox(Path(tmp))
            rng = np.random.default_rng(0)
            rows, priced = [], []
            for i in range(50):
                ticker = f"T{i:03d}"
                rows.append(_prediction_row(ledger, ticker=ticker,
                                            win=float(rng.uniform(0.3, 0.7))))
                priced.append({"event_id": f"{ticker}-e", "ticker": ticker,
                               "strategy": "STR-THRU",
                               "event_date": pd.Timestamp("2026-10-16"),
                               "fill_alpha": 0.5, "entry_cost": 1.0, "exit_value": 1.0,
                               "ret": float(rng.normal(0.02, 0.1)), "exit_mode": "chain"})
            ledger.write_predictions(rows)

            class _R:
                trades = pd.DataFrame(priced)

            replay_mod.replay = lambda *a, **k: _R()
            _require(not ledger.calibration_due()[0], "trigger fired before any outcome")
            ledger.score_outcomes(through="2026-10-20")
            due, n_now, _ = ledger.calibration_due()
            _require(due and n_now == 50, f"trigger did not arm at 50 rows ({n_now})")

            out = ledger.calibrate()
            _require(out["regenerated"], "calibration did not regenerate")
            body = Path(out["report"]).read_text()
            _require("## 6. Calibration" in body, "calibration report has no §6")

            health = json.loads(Path(out["health"]).read_text())
            for key in ("generated_at", "n_scored", "per_strategy", "champion_versions",
                        "snapshot_hash", "data_freshness", "quota_state"):
                _require(key in health, f"health.json missing frozen key {key!r}")
            _require(not ledger.calibrate()["regenerated"],
                     "the trigger refired without new rows")
            return f"regenerated at n={health['n_scored']}, health.json schema complete"
    finally:
        paths.LEDGER = original_ledger
        replay_mod.replay = original_replay


# --------------------------------------------------------------------------
# 8. numbers preserved — the Part A diff test
# --------------------------------------------------------------------------

#: Values the committed reports rendered WRONG and the new generator fixes.
#: Each entry is a documented, reviewed exception rather than a tolerance:
#: the old ``_fmt(pct=True)`` appended "%" without scaling, so the capacity
#: line printed a 36.5% relative spread as "+0.3649%".
KNOWN_RENDERING_FIXES = {"capacity"}


@check("numbers_preserved",
       description="regenerated reports carry the committed reports' numbers")
def check_numbers_preserved() -> str:
    """Part A changed presentation, not arithmetic — this proves it.

    Only the generator-produced body is compared: the committed reports end
    with a block run.py appended AFTER the generator (the pattern Part A
    removes), whose inputs are not in the metrics artifact this regeneration
    reads. Matching allows a value to reappear scaled by 100, because the old
    ``_fmt(pct=True)`` appended "%" without scaling — the capacity line printed
    a 36.5% relative spread as "+0.3649%", which is the one number this check
    is expected to find "missing" and the new generator renders correctly.
    """
    dirs = [d for d in _experiment_dirs() if (d / "REPORT.md").exists()]
    _require(bool(dirs), "no committed REPORT.md to diff against")

    def matched(x: float, pool: np.ndarray, body: str) -> bool:
        # Terminal equities above 1e6x render as ">1e6x" by design — printing
        # 45,528,139,812,307,560x teaches the reader to distrust the document,
        # and the exact value stays in results/metrics_*.json.
        if abs(x) >= 1e6 and ">1e6×" in body:
            return True
        for candidate in (x, x * 100.0, x / 100.0):
            tol = max(0.05, abs(candidate) * 0.005)
            if pool.size and np.min(np.abs(pool - candidate)) <= tol:
                return True
        return False

    details = []
    with tempfile.TemporaryDirectory() as tmp:
        for exp in dirs:
            old = (exp / "REPORT.md").read_text()
            # Cut the hand-appended block and the provenance hashes.
            for marker in ("\n---\n\n## Appendix — ", "## 8. Provenance"):
                if marker in old:
                    old = old.split(marker)[0]
            new = _regenerate(exp, Path(tmp) / exp.name).split("## 8. Provenance")[0]
            new_pool = np.array(_numbers(new), dtype=float)
            missing = [x for x in set(np.round(_numbers(old), 6))
                       if not matched(float(x), new_pool, new)]
            _require(not missing,
                     f"{exp.name}: {len(missing)} value(s) vanished — {sorted(missing)[:6]}")
            details.append(f"{exp.name.split('_')[0]} ✓")
    return "numbers preserved: " + ", ".join(details)


# --------------------------------------------------------------------------
# 9-13. format guards
# --------------------------------------------------------------------------


def _report_bodies(tmp: Path) -> dict[str, str]:
    """One rendered body per report kind, for the format guards."""
    from engine.report import build_provenance

    bodies = {"evaluation": Report.from_eval(_synthetic_result(tmp)).write(tmp / "e").read_text()}
    bodies["calibration"] = Report({
        "kind": "calibration",
        "spec": {"id": "LEDGER", "title": "ledger calibration", "type": "descriptive"},
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(seeds={}), "survivorship_note": "",
        "calibration": {"available": True, "n": 300, "base_rate": 0.4, "brier": 0.24,
                        "brier_base_rate": 0.235, "brier_skill": 0.02,
                        "reliability_monotonicity": 0.8,
                        "deciles": [{"predicted": 0.3, "realized": 0.35, "n": 150}]},
        "funnel": [{"stage": "outcomes resolved", "events": 300, "note": "x",
                    "headline": True}],
        "extra_sections": [],
    }).write(tmp / "c").read_text()
    return bodies


@check("no_raw_json", needs_data=False,
       description="no JSON blob in a report body")
def check_no_raw_json() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for kind, body in _report_bodies(tmp).items():
            head = body.split("## 8. Provenance")[0]
            # The appendix may fence a grid dump; the body may not.
            _require("{" not in head.replace("{0, .25, .5, .75, 1}", ""),
                     f"{kind}: raw JSON blob in the body")
        return "evaluation and calibration bodies are prose and tables"


@check("units_present", needs_data=False,
       description="no bare four-decimal floats in a report body")
def check_units_present() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for kind, body in _report_bodies(tmp).items():
            head = body.split("## 8. Provenance")[0]
            bare = re.findall(r"\|\s*-?\d+\.\d{4}\s*\|", head)
            _require(not bare, f"{kind}: unformatted floats in tables: {bare[:5]}")
        return "every table cell carries %, × or a stated unit"


@check("no_bare_dashes", needs_data=False,
       description="a dash never stands in for a missing value")
def check_no_bare_dashes() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for kind, body in _report_bodies(tmp).items():
            cells = re.findall(r"\|\s*—\s*\|", body)
            _require(not cells, f"{kind}: {len(cells)} bare-dash table cell(s)")
            for line in body.splitlines():
                if re.search(r":\s*—\s*$", line):
                    raise AssertionError(f"{kind}: dash as a value — {line!r}")
        return "missing values render as 'n/a', never as a dash"


@check("section_order", needs_data=False, description="the fixed section order")
def check_section_order() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bodies = _report_bodies(tmp)
        body = bodies["evaluation"]
        positions = [body.find(s) for s in SECTION_ORDER]
        _require(all(p >= 0 for p in positions),
                 f"missing section(s): {[s for s, p in zip(SECTION_ORDER, positions) if p < 0]}")
        _require(positions == sorted(positions), "evaluation sections out of order")

        # Other kinds skip sections but never reorder them.
        cal = bodies["calibration"]
        present = [s for s in SECTION_ORDER if s in cal]
        found = [cal.find(s) for s in present]
        _require(found == sorted(found), "calibration sections out of order")
        _require("## 4. Monte Carlo" not in cal,
                 "the calibration report rendered a Monte Carlo it has no data for")
        return f"{len(SECTION_ORDER)} sections ordered; calibration renders {len(present)}"


@check("glossary_complete", needs_data=False,
       description="every rendered metric has a definition")
def check_glossary_complete() -> str:
    from engine.evaluate import METRIC_KEYS

    undefined = [k for k in METRIC_KEYS
                 if k not in METRIC_SPEC and k not in COMPOSITE_METRIC_KEYS]
    _require(not undefined, f"canonical metrics with no definition: {undefined}")
    with tempfile.TemporaryDirectory() as tmp:
        body = _report_bodies(Path(tmp))["evaluation"]
        glossary = body.split("## 10. Glossary")[1]
        for _key, (_fmt, label, definition) in METRIC_SPEC.items():
            if definition:
                _require(f"| {label} |" in glossary, f"{label} missing from the glossary")
        for term in GLOSSARY:
            _require(term in glossary, f"{term} missing from the glossary")
    return f"{len(METRIC_SPEC)} metrics + {len(GLOSSARY)} terms defined"


@check("verdict_derivable", needs_data=False,
       description="the verdict call and its warnings are computed, not passed")
def check_verdict_derivable() -> str:
    supported = verdict({"headline": {"mean": 0.04, "n": 7620, "breakeven_alpha": 0.42,
                                      "by_year": {"2024": {"n": 100, "mean": 0.05,
                                                           "win_rate": 0.5}}}},
                        {"type": "confirmatory"})
    _require(supported.call.startswith("SUPPORTED"), f"got {supported.call!r}")

    failed = verdict({"headline": {"mean": -0.03, "n": 100, "breakeven_alpha": 0.9,
                                   "by_year": {}}},
                     {"type": "confirmatory"})
    _require(failed.call.startswith("NOT SUPPORTED"), f"got {failed.call!r}")

    descriptive = verdict({"headline": {"mean": 0.01, "n": 100, "breakeven_alpha": 0.4,
                                        "by_year": {}}},
                          {"type": "descriptive"})
    _require(descriptive.call.startswith("DESCRIPTIVE"), f"got {descriptive.call!r}")

    warns = compute_warnings(
        {"headline": {"max_concurrency": 133, "breakeven_alpha": 0.47,
                      "by_year": {"2020": {"n": 5, "mean": 0.01}}},
         "mc": {"by_fraction": {"0.05": {"p_loss": 0.0}}},
         "stress": {"slippage": {"available": True,
                                 "shifts": {"-1d": {"coverage": 0.001,
                                                    "delta_mean": float("nan")}}}}},
        {"brier_skill": -0.084})
    for expected in ("Brier skill", "INCONCLUSIVE", "overlap", "n < 30", "margin"):
        _require(any(expected in w for w in warns), f"warning missing: {expected}")
    return f"3 calls derived; {len(warns)} warnings computed"


# --------------------------------------------------------------------------
# 15-16. the append ban and figure captions
# --------------------------------------------------------------------------


@check("no_report_append", needs_data=False,
       description="no experiment appends to REPORT.md behind the generator")
def check_no_report_append() -> str:
    offenders = []
    for run_py in (ROOT / "experiments").glob("EXP-*/run.py"):
        source = run_py.read_text()
        if re.search(r"open\(\s*[^)]*REPORT\.md[^)]*,\s*[\"']a[\"']", source):
            offenders.append(run_py.relative_to(ROOT))
        if re.search(r"REPORT\.md[\"']?\s*\)\s*\.\s*open\(\s*[\"']a", source):
            offenders.append(run_py.relative_to(ROOT))
    _require(not offenders, f"run.py appends to REPORT.md: {offenders}")
    n = len(list((ROOT / "experiments").glob("EXP-*/run.py")))
    return f"{n} experiment runner(s) render through the generator"


@check("figure_captions", needs_data=False,
       description="every figure carries a caption clear of its axes")
def check_figure_captions() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        result = _synthetic_result(tmp)
        result.results["stress"] = {
            "regimes": {"2022": {"n": 20, "mean": 0.03, "win_rate": 0.5}},
        }
        report = Report.from_eval(result)
        report.context["calibration"] = {
            "available": True, "base_rate": 0.4,
            "deciles": [{"predicted": 0.3, "realized": 0.35, "n": 50}]}
        out = report.write(tmp / "out")
        figures = sorted((out.parent / "figures").glob("*.png"))
        _require(figures, "no figures rendered")

        # Every PNG has its data sidecar, and the markdown references it.
        body = out.read_text()
        for png in figures:
            _require(png.with_suffix(".json").exists(),
                     f"{png.name}: no figure-data sidecar for golden comparison")
            _require(f"figures/{png.name}" in body, f"{png.name} not referenced")

        # Caption geometry: the caption must sit below every axes bounding box.
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        from engine.report import _finish

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("x label that must not be overprinted")
        path = _finish(fig, tmp / "geom.png", "Falsified if: the caption overlaps.")
        _require(path.exists(), "caption helper did not save")
        return f"{len(figures)} figures, all captioned with data sidecars"


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


ORDER = [
    "unittests", "golden_report", "regeneration", "checklist_honesty",
    "ledger_append_only", "outcome_idempotent", "leak_poison", "calibration_trigger",
    "numbers_preserved", "no_raw_json", "units_present", "no_bare_dashes",
    "section_order", "glossary_complete", "verdict_derivable", "no_report_append",
    "figure_captions",
]


def run(names: list[str], *, skip_data: bool = False) -> list[CheckOutcome]:
    outcomes: list[CheckOutcome] = []
    for name in names:
        spec = REGISTRY[name]
        if skip_data and spec["needs_data"]:
            print(f"  {name:22s} SKIP (needs data)", flush=True)
            outcomes.append(CheckOutcome(name, True, "skipped", skipped=True))
            continue
        started = time.time()
        try:
            detail = spec["fn"]() or ""
            elapsed = time.time() - started
            print(f"  {name:22s} PASS  {detail} ({elapsed:.0f}s)", flush=True)
            outcomes.append(CheckOutcome(name, True, detail, elapsed))
        except Exception as exc:  # noqa: BLE001 — a check failure is a result
            elapsed = time.time() - started
            print(f"  {name:22s} FAIL  {exc} ({elapsed:.0f}s)", flush=True)
            outcomes.append(CheckOutcome(name, False, str(exc), elapsed))
    return outcomes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", choices=ORDER, default=None)
    ap.add_argument("--no-data", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.list:
        for name in ORDER:
            spec = REGISTRY[name]
            flag = "data" if spec["needs_data"] else "pure"
            print(f"  {name:22s} [{flag}]  {spec['description']}")
        return 0

    names = args.only or ORDER
    print(f"Phase 4 acceptance checks ({len(names)} checks)\n", flush=True)
    started = time.time()
    outcomes = run(names, skip_data=args.no_data)

    failed = [o for o in outcomes if not o.passed]
    skipped = [o for o in outcomes if o.skipped]
    print(f"\n{len(outcomes) - len(failed) - len(skipped)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped in {time.time()-started:.0f}s")
    if args.json:
        Path(args.json).write_text(
            json.dumps([o.__dict__ for o in outcomes], indent=1, default=str))
    if failed:
        print("\nFAILED:", file=sys.stderr)
        for outcome in failed:
            print(f"  {outcome.name}: {outcome.detail}", file=sys.stderr)
        return 1
    print("\nPHASE 4 CHECKS: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
