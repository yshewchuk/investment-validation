#!/usr/bin/env python3
"""Per-module mutation-score ratchet, mirroring
``checks/v2_coverage_ratchet.py``'s pattern: a fixed comparison, a
committed baseline JSON, a regression code that fails the gate, and a
measurement that NEVER rewrites the baseline (``--output`` only ever writes a
fresh measurement for a human to review and commit).

This script never runs mutmut. It reads an already-produced mutation-report
artifact -- a directory holding ``results.jsonl`` and ``summary.json``, the
shape ``tools/mutation_results.py merge`` writes and the ``mutation-report``
CI artifact already is (``--dir``), or a previously written measurement
(``--input``, from this script's own ``--output``).

Why it only compares WEEKLY FULL runs
--------------------------------------
Per-push incremental runs (``.github/workflows/mutation.yml``) restore each
module's cached mutmut state and re-test only functions whose source (or
selected tests) changed since the last run for that module. Different
modules therefore see wildly different fractions of their mutants re-tested
on any given push -- sometimes ~0% (nothing in that module's files changed),
sometimes effectively 100% (a cache miss, e.g. CI run 35458557499's
``data_legacy`` and ``ops_snapshots`` jobs both re-tested every mutant they
own on an ordinary push). A push-to-push score delta over that mix is noise,
not signal: a module that happened to retest little looks artificially
stable, and one that retested a lot can swing on sample composition alone,
independent of any real test-quality change. So ``compare()`` refuses to
gate on anything whose ``mode`` is not ``"full"`` on EITHER side (measured or
baseline) -- ``MUTATION_MEASUREMENT_NOT_FULL`` / ``MUTATION_BASELINE_NOT_FULL``
-- and does not run the per-module checks at all when either fires, unlike
the coverage ratchet's parallel-mode check (which still runs its other
checks, because a parallel measurement's only bias there is a one-directional
overcount). Here the bias can run either way per module, so a mode failure
alone is the only trustworthy thing to report.

Equivalent/low-value mutants
----------------------------
``tools/mutation_triage.toml`` entries are re-applied HERE, against the
CURRENT triage file on disk -- never trusted from whatever was baked into a
row's own ``triage`` field at export time, because a triage entry can be
added (or go stale) after a run without re-running mutmut. A mutant with a
live (non-stale) triage entry is excluded from BOTH sides of a module's
ratio (``checked_effective``), so: (a) triaging a real survivor as
EQUIVALENT/LOW-VALUE can only ever hold or improve the ratio, never look
like a regression; (b) a stale entry (the function changed since the entry
was written) stops excluding its mutant, and that mutant counts as live
again -- exactly ``tests/test_mutation_ci.py``'s existing staleness contract
for the triage file, reused rather than reinvented. mutmut's own mutant
naming (``pkg.mod.x_func__mutmut_N`` / ``xǁClassǁmethod__mutmut_N``) is an
ordinal within one function, not a line number, so it survives unrelated
line moves elsewhere in the file; ``diff_contains`` is the existing pin
against the one remaining risk -- an edit to the MUTATED function itself
renumbering which ordinal a given mutation gets. When that happens the pin
stops matching and the entry is reported stale rather than silently
misapplied to a different mutation (verified during this file's own
authorship: a triage entry written against one CI run's mutant numbering
came up correctly stale against a differently-commit-scoped run's numbering
for the same function).

New modules
-----------
A module in ``measured`` with no ``baseline`` entry never silently passes:
it is ``MUTATION_NEW_MODULE_BASELINE_REQUIRED``, mirroring
``COVERAGE_NEW_PACKAGE_BASELINE_REQUIRED``. A module that only exists in the
baseline (removed from the matrix, e.g. an ``excluded`` module -- see
``data_legacy``, excluded 2026-09-19) is NOT an error: unlike coverage's
fixed architectural ``PACKAGES`` list, mutation modules are an evolving
CI-scope choice, and refusing on removal would fight every future exclusion.
Its baseline entry simply stops being compared against anything.

Both sides of a module's counts are validated the same way
(``MUTATION_COUNTS_INVALID`` / ``MUTATION_BASELINE_COUNTS_INVALID``): missing
or wrong-typed ``killed_effective``/``checked_effective``, or
``killed_effective > checked_effective``, refuse that module rather than
raise or silently compare nonsense. A measurement with an empty ``modules``
map is ``MUTATION_MEASUREMENT_EMPTY`` rather than a vacuous pass: a real full
run always covers every enabled module (``mutation.yml``'s `plan` job skips
`mutate`/`report` entirely when its module list is empty), so an empty
measurement means a broken or truncated artifact, never a legitimately
narrow one.

Backends and policies (mutmut -> pytest-gremlins migration)
-----------------------------------------------------------
A measurement carries ``backend``/``backend_version``/``policy_id`` from the
report's ``summary.json``; schema-1 (mutmut) artifacts carry none, and a
baseline with no ``backend`` is read as historical mutmut. mutmut and
pytest-gremlins measure different things (operator sets, error/pardon
semantics, score denominators), so ``compare()`` short-circuits on
``MUTATION_BACKEND_MISMATCH`` (backend or pinned version differs) and, for two
gremlins sides, ``MUTATION_POLICY_MISMATCH`` (the scoring policy differs --
timeout or score formula; the operator set *observed* in one artifact is data,
not policy, and never triggers it). A new reviewed FULL gremlins baseline is
committed separately: with ``--baseline`` absent, a gremlins measurement reads
``checks/mutation_ratchet_baseline_gremlins.json`` (``MUTATION_BASELINE_MISSING``
until that file exists) and a mutmut one reads the historical
``checks/mutation_ratchet_baseline.json``; the old baseline is never edited or
renamed for this. An incomplete or tool-errored gremlins measurement
(``complete: false`` / ``tool_error: true``, e.g. a nonzero run exit code, a
timeout kill, or raw error results) is refused outright --
``MUTATION_MEASUREMENT_INCOMPLETE`` / ``MUTATION_MEASUREMENT_TOOL_ERROR`` --
never compared or passed. In ``module_counts``, an ``excluded`` (pardoned)
mutant leaves the ratio like a ``skipped`` one does; a ``suspicious`` (error)
mutant stays in ``checked_effective`` and can never count as killed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gremlin_results as gr  # noqa: E402
import mutation_results as mr  # noqa: E402

BASELINE = ROOT / "checks/mutation_ratchet_baseline.json"
BASELINE_GREMLINS = ROOT / "checks/mutation_ratchet_baseline_gremlins.json"
SCHEMA_VERSION = "mutation_ratchet.v1"


def read_report_dir(path: Path) -> tuple[list[dict], dict]:
    rows = mr.read_jsonl(path / "results.jsonl")
    summary = json.loads((path / "summary.json").read_text())
    return rows, summary


def module_counts(rows: list[dict], triage: dict[str, dict]) -> dict[str, dict]:
    """Per-module {total, checked_effective, killed_effective,
    survived_untriaged, triaged}, recomputed against ``triage`` (the CURRENT
    triage file, not each row's own baked-in ``triage`` field)."""
    counts: dict[str, dict] = {}
    for row in rows:
        bucket = counts.setdefault(row["module"], {
            "total": 0, "checked_effective": 0, "killed_effective": 0,
            "survived_untriaged": 0, "triaged": 0})
        bucket["total"] += 1
        if row["status"] in ("skipped", "excluded"):  # never ran / pardoned: out of the ratio
            continue
        current = mr.triage_for(row["mutant_name"], row.get("diff"), triage)
        if current is not None and not current["stale"]:
            bucket["triaged"] += 1
            continue
        bucket["checked_effective"] += 1
        if row["status"] in ("killed", "timeout"):
            bucket["killed_effective"] += 1
        else:
            bucket["survived_untriaged"] += 1
    return counts


def build_measurement(rows: list[dict], summary: dict, *,
                      triage_path: Path = mr.TRIAGE_FILE) -> dict:
    """Per-module effective counts plus the artifact's backend identity. A
    schema-1 (mutmut) summary carries none of the gremlins keys, and they stay
    null here -- ``compare`` reads that as the historical mutmut side."""
    triage = mr.load_triage(triage_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": summary.get("mode"),
        "run_id": summary.get("run_id"),
        "sha": summary.get("sha"),
        "backend": summary.get("backend"),
        "backend_version": summary.get("backend_version"),
        "policy_id": gr.policy_identity(summary["policy"]) if summary.get("policy") else None,
        "complete": summary.get("complete", True),
        "tool_error": bool(summary.get("tool_error", False)),
        "modules": module_counts(rows, triage),
    }


def _valid_counts(doc: dict) -> bool:
    killed, checked = doc.get("killed_effective"), doc.get("checked_effective")
    return type(killed) is int and type(checked) is int and 0 <= killed <= checked


def compare(measured: dict, baseline: dict) -> list[dict]:
    failures = []
    if measured.get("mode") != "full":
        failures.append({"code": "MUTATION_MEASUREMENT_NOT_FULL"})
    if baseline.get("mode") != "full":
        failures.append({"code": "MUTATION_BASELINE_NOT_FULL"})
    if failures:
        # Either side is an incremental (or unlabeled) view: per-module counts
        # would compare different mutant subsets and cannot be trusted either
        # direction. Report only the mode problem, not noise on top of it.
        return failures
    backend_measured = measured.get("backend") or "mutmut"  # schema 1: historical mutmut
    backend_baseline = baseline.get("backend") or "mutmut"
    if backend_measured != backend_baseline:
        return [{"code": "MUTATION_BACKEND_MISMATCH", "measured_backend": backend_measured,
                 "baseline_backend": backend_baseline}]
    ver_m, ver_b = measured.get("backend_version"), baseline.get("backend_version")
    pol_m, pol_b = measured.get("policy_id"), baseline.get("policy_id")
    if backend_measured == gr.BACKEND:
        # A gremlins measurement is only comparable to a gremlins baseline that
        # states the SAME pinned backend_version and scoring policy. Absent or
        # differing identity fields must not pass -- the old "only compare when
        # both present" guard let a gremlins measurement that dropped
        # policy_id/backend_version through against a baseline that carried them.
        if not ver_m or not ver_b or ver_m != ver_b:
            return [{"code": "MUTATION_BACKEND_MISMATCH", "field": "backend_version",
                     "measured": ver_m, "baseline": ver_b}]
        if not pol_m or not pol_b or pol_m != pol_b:
            return [{"code": "MUTATION_POLICY_MISMATCH", "measured": pol_m, "baseline": pol_b}]
    else:
        # Historical mutmut schema 1 carries none of these; their absence stays
        # supported and is only ever compared when present on both sides.
        if ver_m and ver_b and ver_m != ver_b:
            return [{"code": "MUTATION_BACKEND_MISMATCH", "field": "backend_version",
                     "measured": ver_m, "baseline": ver_b}]
        if pol_m and pol_b and pol_m != pol_b:
            return [{"code": "MUTATION_POLICY_MISMATCH", "measured": pol_m, "baseline": pol_b}]
    if measured.get("tool_error"):
        return [{"code": "MUTATION_MEASUREMENT_TOOL_ERROR"}]
    if measured.get("complete") is False:
        return [{"code": "MUTATION_MEASUREMENT_INCOMPLETE"}]
    measured_modules = measured.get("modules") or {}
    if not measured_modules:
        # A real full run always covers every enabled module (mutation.yml's
        # `plan` job skips `mutate`/`report` entirely when its module list is
        # empty), so an empty measurement here means a broken or truncated
        # artifact, not a legitimately narrow run. Never let that read as a
        # vacuous pass.
        return [{"code": "MUTATION_MEASUREMENT_EMPTY"}]
    baseline_modules = baseline.get("modules") or {}
    for name, current in measured_modules.items():
        if not _valid_counts(current):
            failures.append({"code": "MUTATION_COUNTS_INVALID", "module": name})
            continue
        killed, checked = current["killed_effective"], current["checked_effective"]
        previous = baseline_modules.get(name)
        if previous is None:
            failures.append({"code": "MUTATION_NEW_MODULE_BASELINE_REQUIRED", "module": name})
            continue
        if not _valid_counts(previous):
            failures.append({"code": "MUTATION_BASELINE_COUNTS_INVALID", "module": name})
            continue
        prev_killed, prev_checked = previous["killed_effective"], previous["checked_effective"]
        if checked == 0 or prev_checked == 0:
            continue  # nothing measurable on one side (e.g. every mutant triaged)
        if killed * prev_checked < prev_killed * checked:
            failures.append({"code": "MUTATION_REGRESSION", "module": name,
                             "previous": [prev_killed, prev_checked],
                             "current": [killed, checked]})
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path,
                        help="a merged mutation-report directory (results.jsonl + "
                             "summary.json) -- e.g. the mutation-report CI artifact, "
                             "or `tools/mutation_results.py merge` output. Never runs mutmut.")
    parser.add_argument("--input", type=Path,
                        help="a measurement previously written with --output, instead of --dir")
    parser.add_argument("--triage", type=Path, default=mr.TRIAGE_FILE)
    parser.add_argument("--output", type=Path, help="write the measurement (never the baseline)")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="baseline file; by backend: the historical mutmut baseline for "
                             "schema-1 measurements, checks/mutation_ratchet_baseline_gremlins."
                             "json for pytest-gremlins ones (committed separately, by hand, "
                             "after review)")
    args = parser.parse_args(argv)

    if bool(args.dir) == bool(args.input):
        parser.error("exactly one of --dir or --input is required")
    if args.input:
        measured = json.loads(args.input.read_text())
    else:
        rows, summary = read_report_dir(args.dir)
        measured = build_measurement(rows, summary, triage_path=args.triage)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")

    baseline_path = args.baseline or (BASELINE_GREMLINS
                                      if measured.get("backend") == gr.BACKEND else BASELINE)
    if not baseline_path.exists():
        failures = [{"code": "MUTATION_BASELINE_MISSING"}]
    else:
        failures = compare(measured, json.loads(baseline_path.read_text()))
    print(json.dumps({"ok": not failures, "findings": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
