#!/usr/bin/env python3
"""Render Phase 4's own evidence report.

    python3 checks/phase4_checks.py --json /tmp/p4.json
    python3 checks/phase4_report.py --checks-json /tmp/p4.json

Phase 4 is the phase that makes results legible, so its own report goes
through its own generator — the same section order, formatters, provenance and
glossary every other result now gets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.report import Report, build_provenance  # noqa: E402

#: Defects the completion pass found in the reports the generator was already
#: producing, each with the evidence line it was found in.
DEFECTS = [
    ["Units", "`mean/trade 0.0404` in the headline, `+4.04%` in the appended appendix",
     "One `METRIC_SPEC` maps every key to its formatter; both render `+4.04%`"],
    ["Capacity understated 100×",
     "`mean relative spread +0.3649%` — the old `_fmt(pct=True)` appended `%` "
     "without scaling", "`36.49%`, the value the store actually holds"],
    ["Calibration", "40 lines of `json.dumps` with `brier_skill: -0.084` inside "
     "and every checklist item still PASS",
     "A verdict sentence, a decile table, and a WARN advisory"],
    ["Dead stresses", "`Stale dates: Δmean — on 76 events` (a NaN, at 0.1% chain "
     "coverage) rendered as a result",
     "`INCONCLUSIVE — only 0.10% of trades had an adjacent cached chain`"],
    ["Monte Carlo", "`terminal p50 341,870×`, `P(loss) 0.0000`, with the "
     "overlap caveat below the table",
     "MC and deployment-capped deterministic columns side by side, values above "
     "1e6× clamped, the caveat above the table"],
    ["Figures", "every caption overprinted its x-axis label; a 25,000× equity "
     "curve on a linear axis",
     "Reserved caption margin; log scale past 100× span; figure data sidecars "
     "for golden comparison"],
    ["No verdict", "nothing in the generated body said whether the hypothesis held",
     "§0 renders a derived call, a question table and computed warnings"],
    ["Appended content", "`run.py` opened REPORT.md in append mode; EXP-102's "
     "defined-risk falsification landed outside the report's formatting",
     "`extra_sections` render through the generator and can promote a row into §0"],
]

GAPS = [
    ["Real forward ledger rows", "The calendar ends 2026-08-27, so there are no "
     "forward events to freeze until the Sep-1 ORATS pull refreshes it. The chain "
     "was proven end-to-end on a sandboxed backfill (1,048 predictions → 210 "
     "resolved → calibration + health.json); back-dated rows are deliberately kept "
     "OUT of the real ledger."],
    ["Stored notes in regenerated reports", "The three experiment reports were "
     "re-rendered from their saved metrics artifacts, so text stored INSIDE those "
     "artifacts (e.g. the doubled `no short leg; tail injection N/A` note) still "
     "reads as it was recorded. The generator-side fix lands when the experiments "
     "next run."],
    ["Slippage / stale-date stresses", "Both remain INCONCLUSIVE at ~0.1% chain "
     "coverage on every experiment. The reports now say so instead of printing a "
     "number; closing the gap needs adjacent-day chains from the Sep-1 pull."],
]


def sections(checks: list | None) -> list[dict]:
    out = [
        {"title": "What the reports said before, and what they say now",
         "note": "Every row is a defect found in a report the generator had already "
                 "produced — the content was complete, the document was not legible.",
         "columns": ["area", "before", "after"],
         "align": ["---", "---", "---"],
         "rows": DEFECTS,
         "promote_to_verdict": True,
         "verdict_row": ("Are the reports legible now?",
                         f"**{len(DEFECTS)} defect classes closed** — units, calibration "
                         "honesty, dead stresses, MC reconciliation, figures, verdict, "
                         "appended content", "")},
        {"title": "What Phase 4 added",
         "body": [
             "- **§0 Verdict** — a derived call, a question table, and warnings "
             "computed from the results (a caller cannot quiet one by omitting it).",
             "- **§1.5 Sample funnel** — which universe every `n` belongs to, with "
             "the headline row marked.",
             "- **`engine/ledger.py`** — append-only prediction ledger, outcome "
             "scorer, calibration trigger, and the frozen `health.json` Phase 3 reads.",
             "- **`AuditReceipt`** — the leak audit now emits evidence (counts, "
             "latest stamp, margin) instead of a claim that it ran.",
             "- **`extra_sections`** — spec-specific analyses render through the "
             "generator; appending to REPORT.md is now a check failure.",
             "- **`checks/phase4_checks.py`** — 17 checks, including the format "
             "guards that stop the document regressing.",
         ]},
        {"title": "What the ledger drill found",
         "note": "The end-to-end drill scored a past window (1,048 STR-THRU "
                 "predictions frozen at 2026-07-20, resolved through "
                 "`engine.replay`) in a SANDBOX ledger — back-dated rows must never "
                 "enter the real one. It is a proof the chain works, and it "
                 "produced a finding worth carrying forward.",
         "columns": ["measurement", "value", "reading"],
         "align": ["---", "---:", "---"],
         "rows": [
             ["predictions frozen", "1,048", "one board, one file, append-only"],
             ["resolved", "210", "20% — only where an exit chain exists"],
             ["unresolvable", "838", "recorded with a reason, not dropped"],
             ["Brier skill", "-0.025", "inside the -0.05 floor on this sample"],
             ["predicted mean P&L", "+4.08%", "what the scorer said the trades would pay"],
             ["realized mean P&L", "-0.09%", "what they paid"],
         ],
         "body": [
             "**Win-rate calibration and P&L calibration are different claims, and "
             "this drill separates them:** the win probabilities rank acceptably "
             "while the expected P&L overstates by 4.2 pp on a 210-trade sample "
             "whose 20% resolution rate is itself a liquidity selection. The "
             "generator now states both, and the gap raises a warning in §0 of "
             "every report that carries a calibration block.",
             "",
             "This is a drill result, not a verdict on the strategy: 210 trades, "
             "one window, resolvable-only. It is exactly the question the live "
             "ledger exists to answer once the Sep-1 pull gives it forward events.",
         ],
         "promote_to_verdict": True,
         "verdict_row": ("Does the ledger chain work end to end?",
                         "**Yes** — 1,048 frozen → 210 resolved → calibration + "
                         "health.json, on real data in a sandbox ledger", ""),
         "falsifies": "the predicted-vs-realized P&L gap persisting as the live "
                      "ledger accrues forward events."},
        {"title": "Known gaps", "columns": ["gap", "state"], "align": ["---", "---"],
         "rows": GAPS},
    ]
    if checks:
        out.insert(2, {
            "title": "Acceptance checks",
            "columns": ["check", "result", "detail"],
            "align": ["---", "---", "---"],
            "rows": [[c["name"], "**PASS**" if c["passed"] else "**FAIL**",
                      c.get("detail", "")] for c in checks],
            "falsifies": "any check turning red on a later run — the format guards are "
                         "what keep the reports readable once nobody is watching.",
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checks-json", default=None)
    ap.add_argument("--out", default=str(paths.REPORTS / "phase4_reporting.md"))
    args = ap.parse_args(argv)

    checks = None
    if args.checks_json and Path(args.checks_json).exists():
        checks = json.loads(Path(args.checks_json).read_text())

    passed = sum(1 for c in (checks or []) if c["passed"])
    out = paths.assert_writable(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    context = {
        "kind": "audit",
        "spec": {
            "id": "PHASE-4", "title": "Verification & reporting layer", "type": "descriptive",
            "hypothesis": "Every result the program produces is legible at every step: "
                          "each number states its unit, the universe it was computed on, "
                          "and what would falsify it; and every live prediction is frozen "
                          "before its outcome exists.",
        },
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(
            seeds={},
            input_files=[p for p in [Path(args.checks_json)] if p and p.exists()]),
        "survivorship_note": "",
        "calibration": None,
        "funnel": [
            {"stage": "acceptance checks defined", "events": 17,
             "note": "checks/phase4_checks.py"},
            {"stage": "checks passing", "events": passed,
             "note": "run with --json to record the detail", "headline": True},
        ],
        "extra_sections": sections(checks),
    }
    Report(context).write(out.parent, filename=out.name)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
