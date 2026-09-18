#!/usr/bin/env python3
"""Render the Phase 3B incremental-data acceptance evidence report.

    python3 checks/phase3b_report.py
    python3 checks/phase3b_report.py --receipt <run_receipt.json> \
        --evidence <evidence.json> --artifact-root <run_root>

Every number below is read from a run receipt written by
``checks/phase3b_real.py`` and from the gate result that
``checks/rearchitecture_phase3b.py`` computes over the matching
``evidence.json`` — nothing here is hand-typed. Default source is the frozen,
read-only run this closeout is evaluated against
(``/tmp/phase3b-acceptance-final/run-1789640509``); pass ``--receipt``/
``--evidence``/``--artifact-root`` to render a different run.

**The measured scope is a bounded slice, not production acceptance.**
``checks/phase3b_real.py:_rows`` reads at most 64 rows from ONE partition per
table. Real curated tables are far larger (``daily_market`` ~9.1M rows,
``option_chains`` ~28.7M rows). Raising that limit or running a full-partition
pass was explicitly NOT attempted here: the supervisor decided on 2026-09-18
to accept the bounded scope as a recorded limitation rather than requiring a
real-scale run (see ``guides/rearchitecture_phase3b_closeout.md``). This
report therefore never writes ``"production_acceptance": true`` and always
states the fraction of each table actually exercised.

Written through ``engine.report`` like every other phase report, per
Convention 6.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import rearchitecture_phase3b as gate  # noqa: E402
from engine import paths  # noqa: E402
from engine.report import Report, build_provenance, cell  # noqa: E402

DEFAULT_RUN_ROOT = Path("/tmp/phase3b-acceptance-final/run-1789640509")
DEFAULT_RECEIPT = DEFAULT_RUN_ROOT / "run_receipt.json"
DEFAULT_EVIDENCE = DEFAULT_RUN_ROOT / "evidence.json"
OUT_DIR = paths.REPORTS / "phase3b_acceptance"

#: Attributed decision, not a code default: the supervisor's own call,
#: recorded verbatim so a later reader knows this was a judgement, not a
#: measurement.
ACCEPTED_LIMITATION = (
    "Accepted limitation (supervisor decision, 2026-09-18): Phase 3B "
    "acceptance runs `checks/phase3b_real.py`, which reads at most 64 rows "
    "from ONE partition per table (`_rows(path, limit=64)`) rather than the "
    "full curated tables. Real table sizes are far larger — roughly 9.1M "
    "rows for `daily_market` and 28.7M rows for `option_chains` — so the "
    "gate below is bounded-scope evidence, not proof the incremental-data "
    "path works at production scale. The supervisor decided to record this "
    "as an accepted limitation rather than require a real-scale run before "
    "closing this remediation slice; raising the 64-row limit was explicitly "
    "out of scope. See `guides/rearchitecture_phase3b_closeout.md`."
)

#: Hand-written disposition, not a measurement (same discipline as
#: rearchitecture_phase0_report.py's CARRIED_FORWARD/PRE_EXISTING): the
#: acceptance gate above only checks the 8 P3B01-P3B08 subjects, not the
#: wider v2 test suite. Verified 2026-09-18 by the supervisor in the main
#: checkout (real data present): 3 of 4 candidate real-data tests pass and
#: fail only as a worktree/missing-data artifact elsewhere; exactly one test
#: is genuinely red.
KNOWN_RED_TESTS = (
    ("tests/test_v2_ops_nightly_completion.py::"
     "test_action_finality_writes_a_coverage_output_from_monkeypatched_frames",
     "R3B-7 (open, not part of this closeout): `covered_tickers: []` where "
     "`['AAA']` is expected -- a per-ticker finality coverage regression. "
     "Assigned separately; not fixed here."),
)

R3B2_FINDING = (
    "**Finding (R3B-2): the `__whole__` rewrite on `feature_panel` and "
    "`tier4_forecasts` is correct semantics, not a defect.** "
    "`checks/phase3b_real.py:_partition()` returns the literal `\"__whole__\"` "
    "whenever `contract.partition_columns` is empty, and both tables are "
    "declared that way in `engine/v2/data/legacy_mapping.py` "
    "(`_build_feature_panel_contract`/`_build_tier4_contract`, both call "
    "`_finalize(..., partition_columns=(), ...)`). That is a deliberate "
    "Phase 2 decision, not an oversight: `rearchitecture_phase2_data_access.md` "
    "§3.3/§5.1 declares each of these two derived tables as ONE logical "
    "partition, because on disk they are still the legacy single-file "
    "artifacts (`engine.paths.PANEL`, `engine.paths.TIER4`) — unlike the six "
    "Tier-2 tables, which are physically partitioned by `year=` directories "
    "and therefore have a real `partition_columns`. A correction to either "
    "file has no finer physical partition to target, so rewriting the whole "
    "file on every correction is the correct — and only possible — semantics "
    "for these two tables as currently laid out. No fix is needed; this is "
    "recorded as a decision, not left as an open defect."
)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _scope_rows(receipt: dict) -> list[list[str]]:
    rows = []
    for table in sorted(receipt["source_files"]):
        source = receipt["source_files"][table]
        exercised = receipt["table_results"].get(table, {}).get("rows_before", 0)
        total = source["rows_in_source"]
        fraction = f"{exercised / total * 100:.4f}%" if total else "n/a"
        rows.append([f"`{table}`", f"{total:,}", str(exercised), fraction])
    return rows


def _manifest_rows(receipt: dict) -> list[list[str]]:
    rows = []
    for table in sorted(receipt["source_files"]):
        source = receipt["source_files"][table]
        rows.append([f"`{table}`", source["path"], source["content_hash"][:19] + "…"])
    return rows


def _noop_rows(receipt: dict) -> list[list[str]]:
    rows = []
    for table in sorted(receipt["table_results"]):
        result = receipt["table_results"][table]
        rows.append([
            f"`{table}`", str(result["no_op_rewrites"]),
            "yes" if result["no_op_committed"] else "no",
            "yes" if result["retry_idempotent"] else "no",
        ])
    return rows


def _cache_rows(evidence: dict) -> list[list[str]]:
    counts = evidence.get("subjects", {}).get("P3B03", {}).get("counts", {})
    return [[str(name), str(value)] for name, value in sorted(counts.items())]


def _changed_partition_rows(receipt: dict) -> list[list[str]]:
    rows = []
    for table in sorted(receipt["table_results"]):
        parts = receipt["table_results"][table]["changed_partitions"]
        rows.append([f"`{table}`", ", ".join(parts) or "none"])
    return rows


def _equal_cell(result: dict, field: str) -> str:
    if field not in result:
        return "n/a (older receipt schema)"
    return "equal" if result[field] else "**MISMATCH**"


def _rebuild_rows(receipt: dict) -> list[list[str]]:
    rows = []
    for table in sorted(receipt["table_results"]):
        result = receipt["table_results"][table]
        rows.append([
            f"`{table}`",
            _equal_cell(result, "rebuild_equal"),
            _equal_cell(result, "persisted_rows_equal"),
            _equal_cell(result, "downstream_results_equal"),
            str(result.get("clean_rebuild_rows", "n/a")),
            (result["clean_rebuild_artifact_hash"][:19] + "…"
             if "clean_rebuild_artifact_hash" in result else "n/a"),
        ])
    return rows


def _negative_control_rows(gate_result: dict) -> list[list[str]]:
    controls = gate_result.get("negative_controls", {})
    return [[name, "pass" if passed else "**FAIL**"]
            for name, passed in sorted(controls.items())]


def _resource_rows(receipt: dict, evidence: dict) -> list[list[str]]:
    p3b08 = evidence.get("subjects", {}).get("P3B08", {}).get("counts", {})
    peak = receipt["peak_rss_bytes"]
    return [
        ["runtime_ms", f"{receipt['runtime_ms']:,}"],
        ["peak_rss_bytes", f"{peak:,} ({peak / (1 << 30):.3f} GiB)"],
        ["provider_calls", str(p3b08.get("provider_calls", "—"))],
        ["cache_hits", str(p3b08.get("cache_hits", "—"))],
        ["concurrent_provider_processes (contention)",
         str(p3b08.get("concurrent_provider_processes", "—"))],
        ["chain_dependency_count", str(receipt.get("chain_dependency_count", "—"))],
    ]


def _subject_rows(gate_result: dict) -> list[list[str]]:
    rows = []
    for subject_id in sorted(gate_result.get("subjects", {})):
        result = gate_result["subjects"][subject_id]
        rows.append([f"`{subject_id}`", result["name"], result["status"],
                     str(len(result.get("findings", [])))])
    return rows


def _failure_rows(gate_result: dict) -> list[list[str]]:
    findings = gate_result.get("findings") or []
    if findings:
        return [[cell(finding.get("code")), cell(json.dumps(finding, sort_keys=True))]
                for finding in findings]
    return [[
        "none",
        f"{gate_result['counts']['passed_subjects']}/"
        f"{gate_result['counts']['registered_subjects']} subjects passed, "
        f"{gate_result['counts']['negative_controls_passed']}/"
        f"{gate_result['counts']['negative_controls_total']} negative controls "
        "passed. The one fault this run observes (subject P3B07: "
        "`injected_faults`) is a deliberately staged fault used to exercise "
        "atomic recovery, not an unplanned failure — old head retained, "
        "retry classified, zero partial promotions.",
    ]]


def _snapshot_rows(evidence: dict) -> list[list[str]]:
    return [[ref] for ref in evidence.get("retained_snapshot_refs", [])]


def sections(receipt: dict, evidence: dict, gate_result: dict) -> list[dict]:
    total_exercised = sum(r.get("rows_before", 0) for r in receipt["table_results"].values())
    total_source = sum(s["rows_in_source"] for s in receipt["source_files"].values())
    return [
        {"title": "Measured scope — bounded slice, not production scale",
         "note": "Rows in the real curated/derived table vs. rows this run actually "
                 "read and exercised, per `checks/phase3b_real.py:_sources`/`_rows`.",
         "columns": ["table", "rows in source", "rows exercised", "fraction exercised"],
         "align": ["---", "---:", "---:", "---:"],
         "rows": _scope_rows(receipt),
         "body": [ACCEPTED_LIMITATION],
         "promote_to_verdict": True,
         "verdict_row": ("Is this production-scale acceptance?",
                         f"**No** — bounded-scope evidence, {total_exercised:,} of "
                         f"{total_source:,} available rows exercised "
                         f"({total_exercised / total_source * 100:.4f}%). Accepted "
                         "limitation, supervisor decision 2026-09-18.", "")},
        {"title": "Raw sources and manifests used",
         "note": "One frozen curated Parquet partition per table, content-hashed.",
         "columns": ["table", "path", "content hash"],
         "align": ["---", "---", "---"],
         "rows": _manifest_rows(receipt)},
        {"title": "No-op normalization / write counters",
         "note": "A committed no-op replay must rewrite zero partitions and stay "
                 "idempotent on retry.",
         "columns": ["table", "no_op_rewrites", "no_op replay committed", "retry idempotent"],
         "align": ["---", "---:", ":---:", ":---:"],
         "rows": _noop_rows(receipt)},
        {"title": "Cache-first acquisition counters (P3B03)",
         "columns": ["counter", "value"], "align": ["---", "---:"],
         "rows": _cache_rows(evidence)},
        {"title": "Changed partitions", "columns": ["table", "changed partitions"],
         "align": ["---", "---"], "rows": _changed_partition_rows(receipt)},
        {"title": "Whole-file rewrite finding (R3B-2)",
         "columns": ["table", "changed_partitions written"], "align": ["---", "---"],
         "rows": [row for row in _changed_partition_rows(receipt)
                  if row[0] in ("`feature_panel`", "`tier4_forecasts`")],
         "body": [R3B2_FINDING],
         "promote_to_verdict": True,
         "verdict_row": ("Is the whole-file rewrite on `feature_panel`/`tier4_forecasts` "
                         "a bug?", "**No** — correct semantics for a table declared "
                         "with one logical partition (Phase 2 decision).", "")},
        {"title": "Full-rebuild comparisons",
         "note": "Committed table vs. a clean rebuild from the same base + revisions, "
                 "reopened from the store and re-read downstream.",
         "columns": ["table", "rebuild_equal", "persisted_rows_equal",
                     "downstream_results_equal", "clean rebuild rows", "clean rebuild hash"],
         "align": ["---", ":---:", ":---:", ":---:", "---:", "---"],
         "rows": _rebuild_rows(receipt)},
        {"title": "Negative controls",
         "note": f"{gate_result['counts']['negative_controls_passed']}/"
                 f"{gate_result['counts']['negative_controls_total']} passed, from the "
                 "acceptance gate's own evaluation of this run's evidence.",
         "columns": ["subject.control", "result"], "align": ["---", ":---:"],
         "rows": _negative_control_rows(gate_result)},
        {"title": "Runtime / RSS / cache / contention counters",
         "columns": ["counter", "value"], "align": ["---", "---:"],
         "rows": _resource_rows(receipt, evidence)},
        {"title": "Acceptance subjects (P3B01–P3B08)",
         "columns": ["subject", "name", "status", "findings"],
         "align": ["---", "---", ":---:", "---:"],
         "rows": _subject_rows(gate_result),
         "promote_to_verdict": True,
         "verdict_row": ("Does the bounded acceptance gate pass?",
                         f"**{gate_result['status']}** — "
                         f"{gate_result['counts']['passed_subjects']}/"
                         f"{gate_result['counts']['registered_subjects']} subjects, "
                         f"{len(gate_result.get('findings') or [])} findings", "")},
        {"title": "Actual failures observed during this run",
         "columns": ["code", "detail"], "align": ["---", "---"],
         "rows": _failure_rows(gate_result)},
        {"title": "Known red tests in the wider v2 suite at close",
         "note": "This gate checks only the 8 P3B01-P3B08 subjects above. The "
                 "gate PASS and the row(s) below are both true at the same time "
                 "-- the acceptance gate passing does not mean the whole test "
                 "suite is green.",
         "columns": ["test", "reason"], "align": ["---", "---"],
         "rows": [[cell(test), cell(reason)] for test, reason in KNOWN_RED_TESTS],
         "promote_to_verdict": True,
         "verdict_row": ("Is the wider v2 test suite green at close?",
                         f"**No** -- {len(KNOWN_RED_TESTS)} known red test(s), listed "
                         "here by name; the acceptance gate PASS above covers only "
                         "its own 8 subjects, not the suite.", "")},
        {"title": "Retained snapshot refs",
         "columns": ["snapshot ref"], "align": ["---"],
         "rows": _snapshot_rows(evidence)},
    ]


def build_context(receipt: dict, evidence: dict, gate_result: dict,
                  input_files: list[Path]) -> dict:
    total_exercised = sum(r.get("rows_before", 0) for r in receipt["table_results"].values())
    return {
        "kind": "audit",
        "spec": {
            "id": "REARCH-PHASE-3B-ACCEPTANCE",
            "title": "Rearchitecture Phase 3B — incremental EOD data acceptance",
            "type": "descriptive",
            "hypothesis": (
                "descriptive: does the incremental-data acceptance gate "
                "(checks/rearchitecture_phase3b.py) pass over a measured run "
                "receipt, and what scope does that pass actually cover?"
            ),
        },
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(seeds={}, input_files=input_files),
        "survivorship_note": "",
        "calibration": None,
        "funnel": [
            {"stage": "tables in the acceptance registry",
             "events": len(receipt["source_files"]), "note": "checks/phase3b_real.py TABLES"},
            {"stage": "acceptance subjects (P3B01..P3B08)",
             "events": gate_result["counts"]["registered_subjects"],
             "note": "checks/rearchitecture_phase3b.py SUBJECTS"},
            {"stage": "subjects passed",
             "events": gate_result["counts"]["passed_subjects"],
             "note": f"{gate_result['counts']['failed_subjects']} failed", "headline": True},
            {"stage": "negative controls passed",
             "events": gate_result["counts"]["negative_controls_passed"],
             "note": f"of {gate_result['counts']['negative_controls_total']}"},
            {"stage": "logical rows exercised (bounded slice)",
             "events": total_exercised,
             "note": "64-row cap per source partition — NOT the full table"},
        ],
        "extra_sections": sections(receipt, evidence, gate_result),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    ap.add_argument("--artifact-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "report.md")
    args = ap.parse_args(argv)

    receipt = _read_json(args.receipt)
    evidence = _read_json(args.evidence)
    artifact_root = args.artifact_root or args.evidence.parent
    gate_result = gate.evaluate(evidence, artifact_root=artifact_root)

    out = paths.assert_writable(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    context = build_context(receipt, evidence, gate_result,
                            input_files=[args.receipt, args.evidence])
    Report(context).write(out.parent, filename=out.name)
    print(f"wrote {out} ({out.stat().st_size:,} bytes); gate status "
          f"{gate_result['status']}, {len(gate_result['findings'])} findings")
    return 0 if gate_result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
