#!/usr/bin/env python3
"""Render the rearchitecture phase-0 evidence report.

    python3 checks/rearchitecture_phase0_report.py          # runs the gate, keeps its JSON
    python3 checks/rearchitecture_phase0_report.py --gate-json reports/rearchitecture_phase0_gate.json

Convention 6: exit criteria met, acceptance tests green, and a generated report
documenting the evidence. It goes through ``engine.report`` like every other
result.

**Every measurement below is computed from the gate's JSON** — the claim rows,
both verdicts, the corpus numbers, the seeded controls, the open findings and
the files touched. The first version of this report hard-coded the controls
table, the stage each §11 control landed in and the sentence "all proved
independent", so the report stayed confident while the evidence underneath it
moved. The only hand-written rows now are dispositions, not measurements: what
is carried forward to a later phase and why, and the diagnoses of two
pre-existing suite failures.

The gate JSON the report was rendered from is kept beside it in ``reports/``
(private) and hashed into the provenance block, so the report can be
regenerated from exactly what it read.

Named for the rearchitecture rather than for "phase 0": ``checks/phase0_*.py``
already belong to the DATA phase 0, a different programme at the same number.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.replay_identity import SEEDED_CONTROLS  # noqa: E402
from engine import paths  # noqa: E402
from engine.report import Report, build_provenance  # noqa: E402

GATE_JSON = paths.REPORTS / "rearchitecture_phase0_gate.json"

#: The last commit before phase 0 landed; "legacy untouched" is measured from it.
PHASE0_BASE = "e7a00e1"

#: ``(claim, gate row, how it is answered)``.
GATE_ROWS = (
    ("Every strategy and refusal code has an intact, covering frozen fixture",
     "tier0_corpus", "checks/tier0_corpus.py"),
    ("Every fixture re-scored by the real engine, bound to the current code and "
     "frozen dependencies", "tier1_real_replay", "tools/replay_tier1.py receipt"),
    ("Seeded defects in real engine stages yield their stage-named findings in "
     "one pass", "tier1_seeded_controls", "tools/replay_tier1.py --seed-defects receipt"),
    ("The phase-0 instruments' own negative controls pass",
     "negative_controls", "phase-0 pytest suite"),
    ("The layer check runs green with an empty adapter ledger",
     "import_layers", "checks/import_layers.py --all"),
    ("v2 budgets green at zero exemptions",
     "code_budgets", "checks/code_budgets.py --all"),
    ("Every package README present, consumers matching the graph",
     "package_readmes", "checks/package_readmes.py --all"),
    ("The hook is installed and versioned",
     "pre_commit_hook", "checks/install_hooks.py --check"),
    ("The current baseline package is intact and re-exports byte-identically, "
     "its lock matching the repo", "baseline_package",
     "baseline/CURRENT + tools/baseline_export.py --verify receipt"),
)

#: Where phase 0 stops, and which phase owns the rest. Dispositions, not
#: measurements.
CARRIED_FORWARD = (
    ["Engine-level reordered / batched / restarted parity",
     "Rearchitecture phase 1, acceptance test O30 (slice P1-5). Tier 0 cannot "
     "re-score, and tier 1 re-scores each pair once, in a fresh process, written "
     "and read back. A tier-0 case that only reordered file loading could not "
     "fail and was removed rather than kept as a claim."],
    ["Execution-level stage localization",
     "Findings are localized from record fields along the declared stage graph. "
     "Per-stage input and implementation hashes are phase 1; fine-grained scorer "
     "stages arrive with phase 4 extraction."],
    ["Nightly re-verification over HEAD",
     "Phase 1 (Operations) wires `checks/rearchitecture_phase0_gate.py --json` "
     "into the nightly. Phase 1 §10.4 separates correctness gates from budget "
     "gates and forbids feeding one aggregated phase-0 exit code into publication."],
    ["Transitive dependency pins",
     "`requirements.txt` pins every third-party package the code imports (§6.1). "
     "Transitive pins land with phase 1's pinned linter, in the same lock."],
    ["Module length is a warning, not a failure",
     "§4.3 states it as a soft cap enforced as a warning; the design wins (§2)."],
    ["`features` may not import `models`",
     "§4.1's table gives features \"0-1\" while its prose says features may import "
     "inference; honouring the prose would create a 2 <-> 3 cycle. The table is "
     "enforced; the question is recorded in `checks/layer_map.py`."],
)

#: Pre-existing suite failures found while running the full acceptance suite,
#: recorded rather than fixed — guide §10 forbids legacy edits.
PRE_EXISTING = (
    ["`tests/test_portfolio.py::...::test_a_real_book_carries_everything_declared`",
     "`engine/portfolio.py:211` — `row.get(\"exit_finality\") or {}` does not "
     "catch NaN (NaN is truthy), so a schema-v3 book row whose exit_finality is "
     "NaN crashes `.get`. Exposed by ledger data written 2026-09-11. A separate "
     "decision per §10."],
    ["`tests/test_private_mirror.py::TestCollection::test_it_finds_real_files_and_stays_small`",
     "Mirror size budgets breached by ordinary growth since the 2026-09-09 raise. "
     "The Sep-9 inspection named ARCHIVING old primaries as the next action, not "
     "loosening further — a prune decision, not a budget bump."],
)


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _checks(gate: dict) -> dict:
    return gate.get("checks") or {}


def _state(row: dict) -> str:
    if row.get("ok"):
        return "**met**"
    return "**NOT RUN**" if row.get("skipped") else "**failed**"


def _rows_for_gate(gate: dict) -> list[list[str]]:
    out = []
    for claim, key, how in GATE_ROWS:
        row = _checks(gate).get(key) or {}
        lines = (row.get("detail") or "").strip().splitlines()
        evidence = f"`{how}`" + (f" — {lines[-1][:400]}" if lines else "")
        out.append([claim, _state(row), _cell(evidence)])
    return out


def claims_met(gate: dict) -> int:
    return sum(1 for _, key, _ in GATE_ROWS if (_checks(gate).get(key) or {}).get("ok"))


def _gate_verdict(gate: dict) -> str:
    met = claims_met(gate)
    failed = [key for _, key, _ in GATE_ROWS if not (_checks(gate).get(key) or {}).get("ok")]
    head = "**Yes**" if not failed else "**Not yet**"
    tail = "" if not failed else f"; failing: {', '.join(failed)}"
    return f"{head} — {met}/{len(GATE_ROWS)} claims met{tail}"


def _corpus_rows(gate: dict) -> list[list[str]]:
    t0 = _checks(gate).get("tier0_corpus") or {}
    t1 = _checks(gate).get("tier1_real_replay") or {}
    uncovered = t0.get("uncovered_axes") or []
    cases = t0.get("cases") or {}
    agreeing = sum(1 for verdict in cases.values() if verdict == "agree")
    return [
        ["frozen pairs", str(t0.get("pairs", "—")),
         _cell(f"version `{t0.get('corpus_version')}`; captured through "
               "`Scorer.score`, `dynamic_short_vol` and `replay_one`")],
        ["corpus hash", f"`{(t0.get('corpus_hash') or '—')[:19]}`",
         "content hash of every pair's payload hash"],
        ["required axes uncovered", str(len(uncovered)),
         _cell(", ".join(f"`{a}`" for a in uncovered) or "none")],
        ["tier-0 cases agreeing", f"{agreeing}/{len(cases)}",
         _cell(", ".join(f"{k}: {v}" for k, v in cases.items()) or "—")],
        ["tier-0 runtime", f"{t0.get('seconds', 0):.2f}s",
         "budget 10s, network disabled, no panel load"],
        ["tier-1 pairs re-scored", str(t1.get("pairs_replayed", "—")),
         _cell((t1.get("detail") or "—")[:300])],
        ["code hash", f"`{(gate.get('code_hash') or '—')[:19]}`",
         "what every tier-1 receipt must bind to"],
    ]


def _control_cell(control: dict | None) -> str:
    if not control:
        return "**not run**"
    if control.get("problems"):
        return _cell("**failed** — " + "; ".join(control["problems"]))
    stages = sorted({line.split(":")[0]
                     for lines in (control.get("observed") or {}).values()
                     for line in lines})
    return _cell(f"detected at {', '.join(f'`{s}`' for s in stages)} "
                 f"on `{control.get('target')}`")


def _tier0_controls(gate: dict) -> dict:
    return (_checks(gate).get("tier0_corpus") or {}).get("seeded_controls") or {}


def _control_rows(gate: dict) -> list[list[str]]:
    t0 = _tier0_controls(gate).get("controls") or {}
    t1 = (_checks(gate).get("tier1_seeded_controls") or {}).get("controls") or {}
    rows = [[f"{cause.replace('_', ' ')} (`{spec['commit']}`)",
             _control_cell(t0.get(cause)), _control_cell(t1.get(cause))]
            for cause, spec in SEEDED_CONTROLS.items()]
    mismatches = _tier0_controls(gate).get("field_set_mismatches")
    rows.append([
        "a field dropped from the compared set (`28cf8b1`)",
        _cell("by construction; compared population equals each pair's own leaves"
              + (f" — **{len(mismatches)} mismatches**" if mismatches else "")),
        "by construction; checked on every replayed pair",
    ])
    return rows


def _controls_verdict(gate: dict) -> str:
    t0_ok = ((_checks(gate).get("tier0_corpus") or {}).get("cases") or {}).get(
        "seeded_controls") == "agree"
    t1_ok = bool((_checks(gate).get("tier1_seeded_controls") or {}).get("ok"))
    if t0_ok and t1_ok:
        return ("**Yes** — every seeded cause detected and localized to its stage, "
                "over the real corpus and through real engine stages")
    gaps = [label for ok, label in ((t0_ok, "tier-0 seeded controls did not behave"),
                                    (t1_ok, "tier-1 seeded controls did not behave or did not run"))
            if not ok]
    return "**Not yet** — " + "; ".join(gaps)


def _open_rows(gate: dict) -> list[list[str]]:
    rows = []
    for _, key, _ in GATE_ROWS:
        row = _checks(gate).get(key) or {}
        if not row.get("ok"):
            rows.append([f"`{key}`", _cell((row.get("detail") or "no detail")[:600])])
    for axis in (_checks(gate).get("tier0_corpus") or {}).get("uncovered_axes") or []:
        rows.append([f"uncovered axis `{axis}`",
                     "no captured pair demonstrates it through the real entry points"])
    for finding in (_checks(gate).get("tier1_real_replay") or {}).get("findings") or []:
        rows.append([_cell(f"replay finding `{finding}`"),
                     "the engine no longer reproduces the frozen record here"])
    return rows or [["none", "every claim met"]]


def _git_lines(*args: str) -> list[str]:
    proc = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                          text=True, check=False)
    return [line for line in proc.stdout.splitlines() if line] if proc.returncode == 0 else []


def _touched() -> tuple[list[str], list[str]]:
    """``(legacy engine files changed, every file changed outside engine/v2)``."""
    legacy = sorted(set(_git_lines("diff", "--name-only", PHASE0_BASE, "--",
                                   "engine", ":(exclude)engine/v2")))
    changed = set(_git_lines("diff", "--name-only", PHASE0_BASE, "--", ".",
                             ":(exclude)engine/v2"))
    untracked = {p for p in _git_lines("ls-files", "--others", "--exclude-standard")
                 if not p.startswith("engine/v2/")}
    return legacy, sorted(changed | untracked)


def _not_done_rows(gate: dict) -> list[list[str]]:
    legacy, outside = _touched()
    budgets_ok = (_checks(gate).get("code_budgets") or {}).get("ok")
    shown = ", ".join(f"`{p}`" for p in outside[:40])
    more = f" and {len(outside) - 40} more" if len(outside) > 40 else ""
    return [
        ["No production logic in `engine/v2/`",
         "Every v2 package is `__init__.py` + README except `engine/v2/diagnosis/`, "
         "which §5 requires" + ("" if budgets_ok else " — **v2 budgets currently failing**")],
        ["No edit to legacy `engine/`",
         f"**held** — no file under `engine/` outside `engine/v2/` differs from `{PHASE0_BASE}`"
         if not legacy else _cell("**BROKEN** — " + ", ".join(legacy))],
        ["Files changed outside `engine/v2/`", _cell(f"{len(outside)}: {shown}{more}")],
        ["No defect fixed because the corpus revealed it",
         "Open findings are listed under 'Open, recorded not fixed' (§3.2, §10), "
         "not fixed."],
        ["No strategy, threshold, champion, fill convention or clock touched",
         "Follows from the legacy row: the engine that produces the board is unmodified."],
    ]


def sections(gate: dict) -> list[dict]:
    return [
        {"title": "The phase-0 exit gate",
         "note": "Each claim answered by a check that re-derives its answer now. "
                 "`checks/rearchitecture_phase0_gate.py` runs them all.",
         "columns": ["claim", "state", "evidence"],
         "align": ["---", ":---:", "---"],
         "rows": _rows_for_gate(gate),
         "promote_to_verdict": True,
         "verdict_row": ("Is phase 0's exit gate met?", _gate_verdict(gate), "")},
        {"title": "The tier-0 corpus",
         "note": "The oracle every later phase states its exit gate against. "
                 "Private: it carries licensed quotes.",
         "columns": ["measurement", "value", "reading"],
         "align": ["---", "---:", "---"],
         "rows": _corpus_rows(gate),
         "body": [
             "**Tier 0 checks the oracle; tier 1 checks the engine against it.** "
             "Tier 0 runs in seconds with no panel: manifest membership, "
             "addressing, digests, a serialized round trip, coverage re-derived "
             "from the records, pinned fixtures against their sources, and the "
             "seeded controls over the real corpus. Tier 1 re-scores every "
             "declared pair through the production entry points in a fresh "
             "process, writes and reads each result back, and binds its receipt "
             "to the content hash of the code and the frozen dependencies — a "
             "receipt from different code, a partial run or a skipped pair does "
             "not satisfy the gate.",
         ]},
        {"title": "What the seeded controls showed",
         "note": "A check that has never failed is not known to work. Each "
                 "seedable 2026-09-11 cause is planted into its own distinct "
                 "pair, so one pass is also an ablation.",
         "columns": ["seeded cause", "tier 0 — real corpus", "tier 1 — real engine stages"],
         "align": ["---", "---", "---"],
         "rows": _control_rows(gate),
         "body": [
             "Localization is observed from record fields along the declared "
             "stage graph: it names where two records first disagree, not which "
             "internal computation diverged. Findings in different root stages "
             "are not downstream of one another through that graph; findings "
             "inside one stage may share a cause.",
         ],
         "promote_to_verdict": True,
         "verdict_row": ("Can one pass separate the seeded causes?",
                         _controls_verdict(gate), ""),
         "falsifies": "a future red check that still cannot say how many "
                      "separate causes are behind it."},
        {"title": "Open, recorded not fixed",
         "note": "The gate stays red until each is addressed — by a fix, or by a "
                 "recorded decision and a new baseline version (§3.2).",
         "columns": ["item", "state"],
         "align": ["---", "---"],
         "rows": _open_rows(gate)},
        {"title": "What phase 0 deliberately did not do",
         "note": "§10. Measured from git, not asserted.",
         "columns": ["prohibition", "how it held"],
         "align": ["---", "---"],
         "rows": _not_done_rows(gate)},
        {"title": "Carried forward, with the reason",
         "note": "Where the work belongs to a later phase.",
         "columns": ["item", "disposition"],
         "align": ["---", "---"],
         "rows": [list(row) for row in CARRIED_FORWARD]},
        {"title": "Pre-existing suite failures, recorded not fixed",
         "note": "Found while running the full suite; both predate phase 0.",
         "columns": ["failure", "diagnosis"],
         "align": ["---", "---"],
         "rows": [list(row) for row in PRE_EXISTING]},
    ]


def _load_or_run_gate(gate_json: str | None) -> tuple[dict, Path]:
    if gate_json:
        path = Path(gate_json)
        return json.loads(path.read_text()), path
    from checks.rearchitecture_phase0_gate import gate as run_gate

    previous = json.loads(GATE_JSON.read_text()) if GATE_JSON.is_file() else None
    gate = run_gate(previous)
    path = paths.assert_writable(GATE_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate, indent=2, sort_keys=True, default=str) + "\n")
    return gate, path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate-json", default=None,
                    help="output of `checks/rearchitecture_phase0_gate.py --json`; "
                         "omitted, the gate is run and its JSON kept in reports/")
    ap.add_argument("--out", default=str(paths.REPORTS / "rearchitecture_phase0.md"))
    args = ap.parse_args(argv)

    gate, gate_path = _load_or_run_gate(args.gate_json)
    out = paths.assert_writable(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    context = {
        "kind": "audit",
        "spec": {
            "id": "REARCH-PHASE-0",
            "title": "Rearchitecture phase 0 — baseline",
            "type": "descriptive",
            "hypothesis": (
                "Every later migration phase is checkable: a frozen corpus of what "
                "the current engine does, a replay that proves the engine still "
                "produces it, a comparison receipt that names each seeded cause "
                "in one pass, an enforced layer map, and an empty v2 skeleton."
            ),
        },
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(seeds={}, input_files=[gate_path]),
        "survivorship_note": "",
        "calibration": None,
        "funnel": [
            {"stage": "gate claims defined", "events": len(GATE_ROWS),
             "note": "checks/rearchitecture_phase0_gate.py"},
            {"stage": "gate claims met", "events": claims_met(gate),
             "note": "every unmet claim is listed under 'Open'", "headline": True},
        ],
        "extra_sections": sections(gate),
    }
    Report(context).write(out.parent, filename=out.name)
    print(f"wrote {out} ({out.stat().st_size:,} bytes) from {gate_path}")
    return 0 if gate.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
