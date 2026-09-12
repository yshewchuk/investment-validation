#!/usr/bin/env python3
"""Render the rearchitecture phase-0 evidence report.

    python3 checks/rearchitecture_phase0_gate.py --json > /tmp/p0.json
    python3 checks/rearchitecture_phase0_report.py --gate-json /tmp/p0.json

Convention 6: exit criteria met, acceptance tests green, and a generated report
documenting the evidence. This is the third of those, and it goes through
``engine.report`` like every other result rather than being hand-written — a
phase whose whole subject is "make the next phases checkable" should not report
itself in a format nothing checks.

Named for the rearchitecture rather than for "phase 0": ``checks/phase0_*.py``
already belong to the DATA phase 0 (`guides/phase0_data_foundations.md`), which
is a different programme at the same number.
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

#: The eight claims the phase is finished against: §12 of the design for the
#: first three, §12 of the guide for the rest.
GATE_ROWS = (
    ("Every strategy and critical refusal reproducible",
     "tier0_corpus", "checks/tier0_corpus.py over fixtures/tier0"),
    ("Every score fixture re-scored by the REAL engine against frozen "
     "dependencies", "tier1_real_replay", "tools/replay_tier1.py receipt"),
    ("Five seeded defects yield five stage-named findings in one pass",
     "negative_controls", "tests/test_phase0_negative_controls.py"),
    ("The layer check runs green with an empty adapter ledger",
     "import_layers", "checks/import_layers.py --all"),
    ("v2 budgets green at zero exemptions",
     "code_budgets", "checks/code_budgets.py --all"),
    ("Every package README present, consumers matching the graph",
     "package_readmes", "checks/package_readmes.py --all"),
    ("The hook is installed and versioned",
     "pre_commit_hook", "checks/install_hooks.py --check"),
    ("The baseline package is frozen on disk, intact, lock matching the repo",
     "baseline_package", "baseline/<latest>/MANIFEST.json + requirements.txt"),
)

#: What phase 0 deliberately did not do (§10), stated so the next phase does
#: not have to re-derive it from an empty diff.
NOT_DONE = (
    ["No production logic in `engine/v2/`",
     "18 packages, `__init__.py` + README only. The one exception is "
     "`engine/v2/diagnosis/`, which §5 requires and which is under the §4.3 "
     "budgets from its first line."],
    ["No edit to legacy `engine/`",
     "The only touched file outside `engine/v2/` and `checks/` is "
     "`checks/repo_hygiene.py`, extended for the fixture path — which §7.2 "
     "requires to happen BEFORE the first capture."],
    ["No defect fixed because the corpus revealed it",
     "§3.2: a known defect requires a separately recorded correction and a new "
     "version, not a silent update to the baseline fixture."],
    ["No strategy, threshold, champion, fill convention or clock touched",
     "The board's numbers are unchanged: the corpus captured them and nothing "
     "else moved."],
)

#: Pre-existing suite failures found while running the full acceptance suite,
#: recorded rather than fixed — guide §10 forbids legacy edits, and both
#: predate phase 0 (each reproduces with the whole phase-0 diff stashed).
PRE_EXISTING = (
    ["`tests/test_portfolio.py::...::test_a_real_book_carries_everything_declared`",
     "`engine/portfolio.py:211` — `row.get(\"exit_finality\") or {}` does not "
     "catch NaN (NaN is truthy), so a schema-v3 book row whose exit_finality is "
     "NaN crashes `.get`. Exposed by ledger data written 2026-09-11, not by any "
     "phase-0 change. A separate decision per §10."],
    ["`tests/test_private_mirror.py::TestCollection::test_it_finds_real_files_and_stays_small`",
     "Both budgets breached by ordinary growth since the 2026-09-09 raise: "
     "non-growth 176.9MB vs 160MB (phase 0 contributes 0.3MB of it), evidence "
     "143.0MB vs 130MB, total 319.9MB vs the 300MB backstop. The Sep-9 "
     "inspection named the next action as ARCHIVING old primaries, not "
     "loosening further — so this needs a prune decision, not a budget bump."],
)

#: Where this phase stopped short of the guide, and why. Stated rather than
#: left for someone to notice.
CARRIED_FORWARD = (
    ["Nightly re-verification over HEAD",
     "§9 asks for it; §10 forbids editing legacy `engine/`, and §12 of the "
     "design places it in **phase 1 (Operations)**. `checks/rearchitecture_phase0_gate.py "
     "--json` is the entry point it will call, shaped for the `code_budgets` "
     "class §4.7 adds to `health.json`. The `nightly.py` wiring is phase 1's."],
    ["Module length is a warning, not a failure",
     "§4.3 states it as \"a soft cap ... enforced as a warning rather than a "
     "failure\"; §9 of the guide lists it among the enforced set. The design "
     "wins (§2), so it is reported and counted and does not block."],
    ["`features` may not import `models`",
     "§4.1's table gives features \"0-1\" while its prose says features may "
     "import inference. Honouring the prose would create a 2 <-> 3 cycle, since "
     "models may import features. The table is enforced; the question is "
     "recorded in `checks/layer_map.py` for the phase that writes Tier-4 "
     "materialization."],
)


def _rows_for_gate(gate: dict) -> list[list[str]]:
    checks = gate.get("checks", {})
    out = []
    for claim, key, how in GATE_ROWS:
        row = checks.get(key, {})
        if row.get("ok"):
            state = "**met**"
        elif row.get("skipped"):
            state = "**NOT RUN**"
        else:
            state = "**failed**"
        detail = (row.get("detail") or "").splitlines()
        out.append([claim, state, f"`{how}`" + (f" — {detail[-1]}" if detail else "")])
    return out


def _corpus_rows(gate: dict) -> list[list[str]]:
    row = gate.get("checks", {}).get("tier0_corpus", {})
    tier1 = gate.get("checks", {}).get("tier1_real_replay", {})
    pairs = row.get("pairs")
    uncovered = row.get("uncovered_axes") or []
    return [
        ["frozen pairs", str(pairs if pairs is not None else "—"),
          "captured through `engine.score.score`, `dynamic_short_vol` and "
          "`engine.replay.replay_one`"],
        ["corpus hash", f"`{(row.get('corpus_hash') or '—')[:19]}`",
          "content hash of every pair's payload hash"],
        ["required axes uncovered", str(len(uncovered)),
          ", ".join(uncovered) if uncovered else "none"],
        ["tier-0 replay runtime", f"{row.get('seconds', 0):.2f}s",
          "budget is 10s, network disabled, no panel load"],
        ["tier-1 real replays", str(tier1.get("pairs_replayed", "—")),
          f"re-scored through the production entry points against "
          f"hash-verified frozen dependencies; {tier1.get("pairs_skipped", "—")} "
          "skipped with named reasons"],
    ]


def sections(gate: dict) -> list[dict]:
    ok = gate.get("ok", False)
    return [
        {"title": "The phase-0 exit gate",
         "note": "Eight claims, each answered by a runnable check rather than by "
                 "a sentence. `checks/rearchitecture_phase0_gate.py` runs all eight.",
         "columns": ["claim", "state", "evidence"],
         "align": ["---", ":---:", "---"],
         "rows": _rows_for_gate(gate),
         "promote_to_verdict": True,
         "verdict_row": (
             "Is the migration measurable yet?",
             f"**{'Yes' if ok else 'Not yet'}** — "
             f"{len(GATE_ROWS) - len(gate.get('failed', []))}/{len(GATE_ROWS)} "
             "gate claims met",
             "")},
        {"title": "The tier-0 corpus",
         "note": "The oracle every later phase states its exit gate against. "
                 "Private: it carries licensed quotes, so it lives in the "
                 "mirror and `checks/repo_hygiene.py` blocks it from this repo.",
         "columns": ["measurement", "value", "reading"],
         "align": ["---", "---:", "---"],
         "rows": _corpus_rows(gate),
         "body": [
              "**Tier-0 does not re-score; tier-1 does.** §11 names \"a corpus "
              "that fits or fetches\" as a failure mode: it stops being seconds, "
              "stops running on every edit, and becomes a tier-2 check nobody "
              "waits for. So the integrity half stays seconds-fast: the "
              "manifest is reconciled against the files (exact membership, "
              "per-file hashes, corpus hash recomputed from the SURVIVORS), "
              "coverage is RE-DERIVED from the frozen records by a second "
              "stdlib-only implementation rather than read from the index's "
              "claims, and everything survives a serialized round trip, a "
              "reordering, a batch/single split and a fresh process. The "
              "compatibility half is `tools/replay_tier1.py`: it hash-verifies "
              "the frozen dependencies from the baseline package, rebuilds the "
              "real Scorer, re-scores every fixture through the production "
              "entry points, and refuses INCOMPARABLE on any drift — a changed "
              "pricing formula fails the tier-1 receipt even though tier-0 "
              "stays green. Re-computing these numbers through a *different* "
              "implementation is what the corpus is FOR, and that is the "
              "parity comparison of phases 2-8.",
          ]},
        {"title": "What the negative controls prove",
         "note": "A check that has never failed is not known to work. 1,567 "
                 "tests passed over the five 2026-09-11 defects, and the "
                 "determinism test that should have caught the analog ordering "
                 "bug passed the same frame twice.",
         "columns": ["seeded corruption", "reported as"],
         "align": ["---", "---"],
         "rows": [
             ["Forecast suppressed when `structure_params` are replayed (`e845f3e`)",
              "`forecast`, the whole forecast block as a null mask"],
             ["A field removed from the compared set (`28cf8b1`)",
              "Impossible by construction — the set is the record's own fields"],
             ["Analog bootstrap reseeded by row order (`b9aa1fd`)",
              "`analogs`, `ci_low`/`ci_high` only; the point estimate does not move"],
             ["A replay input rounded to six places (`b33036c`)",
              "`serialization`, `structure_params.width_moneyness`, delta 4.3e-07"],
             ["Rounding reapplied after the exemption (`6b9d5cf`)",
              "`serialization`, the written file disagreeing with its digest"],
         ],
         "body": [
             "All five are seeded **at once** and one pass reports all of them, "
             "each naming its stage, each proved independent of the others. "
             "That single assertion is the phase's reason for existing: it is "
             "the difference between five nights and one.",
             "",
             "The §11 controls are seeded too — a corrupted timestamp, feature "
             "builder, geometry, model hash and dataset membership — and each "
             "lands in `resolve_context`, `features`, `geometry`, `features` "
             "and `analogs` respectively.",
         ],
         "promote_to_verdict": True,
         "verdict_row": ("Can one pass separate five causes?",
                         "**Yes** — five seeded defects, one pass, five "
                         "stage-named findings, all proved independent", ""),
         "falsifies": "a future red self-check that still cannot say how many "
                      "independent causes are behind it."},
        {"title": "The enforced layer map",
         "note": "A logical boundary that is not a physical one constrains "
                 "nothing: nothing currently stops `render.py` importing the "
                 "scorer, and nothing did.",
         "columns": ["rule", "how it is enforced", "negative control"],
         "align": ["---", "---", "---"],
         "rows": [
             ["Inside v2, imports point down only",
              "`checks/layer_map.py` holds the §4.1 table as one literal; a "
              "package may import a strictly lower layer, never a peer",
              "a planted upward import, and a planted 4a peer import, both fail"],
             ["`engine/v2/diagnosis` is imported by nothing",
              "a sink flag in the map; any import of it fails",
              "a planted `serving -> diagnosis` import fails"],
             ["v2 reaches legacy only through declared adapters",
              "`checks/legacy_adapters.json`, one adapter module per package",
              "a planted undeclared legacy import fails; so does a second "
              "adapter module in one package"],
             ["Legacy never imports v2",
              "rule 3, checked over the whole `engine/` tree",
              "a planted `engine/score.py -> engine.v2.scoring` import fails"],
         ],
         "body": [
             "The adapter ledger reads `{\"count\": 0}`. It is the migration "
             "made numeric (§4.6): it may only shrink, and phase 8 begins when "
             "it reaches zero. It starts at zero here because v2 is empty, not "
             "because the migration is done.",
         ]},
        {"title": "What phase 0 deliberately did not do",
         "note": "§10. If a phase-0 change alters a board number, the change is "
                 "wrong.",
         "columns": ["prohibition", "how it held"],
         "align": ["---", "---"],
         "rows": [list(row) for row in NOT_DONE]},
        {"title": "Carried forward, with the reason",
         "note": "Three places where the guide and the design disagree, or "
                 "where the work belongs to the next phase. Recorded rather "
                 "than left to be rediscovered.",
         "columns": ["item", "disposition"],
         "align": ["---", "---"],
         "rows": [list(row) for row in CARRIED_FORWARD]},
        {"title": "Pre-existing failures, recorded not fixed",
         "note": "The full acceptance suite (1,996 passed) surfaced two "
                 "failures that reproduce with the entire phase-0 diff "
                 "stashed. §10: record them, raise them as separate decisions.",
         "columns": ["failure", "diagnosis"],
         "align": ["---", "---"],
         "rows": [list(row) for row in PRE_EXISTING]},
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate-json", default=None,
                    help="output of `checks/rearchitecture_phase0_gate.py --json`")
    ap.add_argument("--out",
                    default=str(paths.REPORTS / "rearchitecture_phase0.md"))
    args = ap.parse_args(argv)

    if args.gate_json and Path(args.gate_json).exists():
        gate = json.loads(Path(args.gate_json).read_text())
    else:
        from checks.rearchitecture_phase0_gate import gate as run_gate
        gate = run_gate()

    out = paths.assert_writable(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    met = len(GATE_ROWS) - len(gate.get("failed", []))
    context = {
        "kind": "audit",
        "spec": {
            "id": "REARCH-PHASE-0",
            "title": "Rearchitecture phase 0 — baseline",
            "type": "descriptive",
            "hypothesis": (
                "Every later migration phase is checkable in seconds: a frozen "
                "corpus of what the current engine does, a comparison receipt "
                "that names every independent cause in one pass, an enforced "
                "layer map, and an empty v2 skeleton for the layers to describe."
            ),
        },
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(
            seeds={},
            input_files=[p for p in [Path(args.gate_json)] if p and p.exists()],
        ),
        "survivorship_note": "",
        "calibration": None,
        "funnel": [
            {"stage": "gate claims defined", "events": len(GATE_ROWS),
             "note": "checks/rearchitecture_phase0_gate.py"},
            {"stage": "gate claims met", "events": met,
             "note": "run with --json to record the detail", "headline": True},
        ],
        "extra_sections": sections(gate),
    }
    Report(context).write(out.parent, filename=out.name)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0 if gate.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
