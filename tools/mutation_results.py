"""Queryable mutation-testing results: one row per mutant, plus summaries.

Builds the report files the mutation CI workflow uploads, from mutmut's own
state in a module's work copy (``$MUTATION_PILOT_HOME/<module>/``):

    results.jsonl   one JSON row per mutant (schema below, ``SCHEMA_VERSION``)
    summary.json    counts and scores per module and per file
    summary.md      the GitHub job summary: scores, and survivors in the
                    functions this push changed

Commands::

    python3 tools/mutation_results.py export MODULE --out DIR [--mode M]
            [--changed-base SHA] [--run-exit-code N] [--elapsed S]
    python3 tools/mutation_results.py merge --out DIR [--expected-modules JSON]
            IN_DIR [IN_DIR ...]

With ``--expected-modules`` (the plan job's JSON module list) the merged
artifact must describe EXACTLY that module set: a missing, unexpected or
duplicated module report -- or no module directory at all -- yields an
incomplete diagnostic (``complete: false``, ``tool_error: true``, machine-
readable ``failure_reasons``, score withheld) and a nonzero exit, so a subset
can never be published as the latest completed run. Without the contract the
merge behaves as before (local and historical merges keep working); a module
reported twice is an unusable input set either way and is refused.

Row fields (schema_version 1); the guide (tests/README.md, "Mutation CI")
documents each one:

    schema_version run_id sha ref trigger mode module file function line
    mutant_name status mutmut_status retested_this_run diff triage

``status`` is one of ``STATUSES``. ``triage`` is null or
``{"verdict": "EQUIVALENT"|"LOW-VALUE", "note": str, "stale": bool}`` from
``tools/mutation_triage.toml``. Everything here is code (names, line numbers,
diffs); nothing reads or prints data or model values.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRIAGE_FILE = REPO / "tools" / "mutation_triage.toml"
SCHEMA_VERSION = 1
STATUSES = ("killed", "survived", "no_tests", "timeout", "suspicious", "skipped")
TRIAGE_VERDICTS = ("EQUIVALENT", "LOW-VALUE")
ROW_FIELDS = ("schema_version", "run_id", "sha", "ref", "trigger", "mode", "module",
              "file", "function", "line", "mutant_name", "status", "mutmut_status",
              "retested_this_run", "diff", "triage")
# A pre-run snapshot of mutmut's verdicts, written by mutation_pilot.py run next
# to (not inside) mutants/, so it is never cached: it describes one run only.
SNAPSHOT_NAME = "prerun-snapshot.json"

# mutmut 3.8 stats.status_by_exit_code, restated so rows can be built without
# mutmut importable. Unknown codes are "suspicious", as there.
_MUTMUT_STATUS = {
    1: "killed", 3: "killed", 0: "survived", 5: "no tests", 33: "no tests",
    2: "interrupted", None: "not checked", 34: "skipped", 35: "suspicious",
    36: "timeout", 37: "type check", -24: "timeout", 24: "timeout",
    152: "timeout", 255: "timeout", -11: "segfault", -9: "segfault",
}
# mutmut's status -> the report's six. A type-check rejection is a kill (as in
# mutmut's own score); a segfault or an interrupted worker is "suspicious";
# "not checked" (the run never reached it) is "skipped".
_REPORT_STATUS = {
    "killed": "killed", "type check": "killed", "survived": "survived",
    "no tests": "no_tests", "timeout": "timeout", "suspicious": "suspicious",
    "segfault": "suspicious", "interrupted": "suspicious",
    "not checked": "skipped", "skipped": "skipped",
}
_SEP = "ǁ"  # mutmut's class-name separator in mangled names


def mutmut_status(code: int | None) -> str:
    return _MUTMUT_STATUS.get(code, "suspicious")


def report_status(code: int | None) -> str:
    return _REPORT_STATUS[mutmut_status(code)]


def function_of(mutant_name: str) -> str:
    """``Class.method`` or ``func`` for a mutmut mutant name."""
    mangled = mutant_name.partition("__mutmut_")[0].rpartition(".")[2]
    if _SEP in mangled:
        _, cls, func = mangled.split(_SEP)[:3]
        return f"{cls}.{func}"
    return mangled[2:] if mangled.startswith("x_") else mangled


# -- source positions ---------------------------------------------------------

def function_spans(source: str) -> dict[str, tuple[int, int]]:
    """Top-level functions and class methods (what mutmut mutates) -> line span."""
    spans: dict[str, tuple[int, int]] = {}

    def add(node, name: str) -> None:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        spans[name] = (start, node.end_lineno)

    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(sub, f"{node.name}.{sub.name}")
    return spans


def changed_line(diff: str, source_lines: list[str], span: tuple[int, int] | None) -> int | None:
    """The original-file line of the first removed line in ``diff``."""
    removed = next((ln[1:] for ln in diff.splitlines()
                    if ln.startswith("-") and not ln.startswith("---")), None)
    if span is None:
        return None
    lo, hi = span
    if removed is None:
        return lo
    hits = [i for i in range(lo, hi + 1) if source_lines[i - 1].strip() == removed.strip()]
    return hits[0] if hits else lo


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def changed_new_lines(diff_text: str) -> dict[str, set[int]]:
    """``git diff -U0`` output -> {new-side path: changed line numbers}.

    A pure deletion (``+N,0``) marks line N, the line the deletion sits after,
    so the function it was cut from still counts as changed.
    """
    out: dict[str, set[int]] = {}
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            path = None if target == "/dev/null" else target.removeprefix("b/")
            if path is not None:
                out.setdefault(path, set())
        elif path is not None and (m := _HUNK.match(line)):
            start, count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            out[path].update(range(start, start + count) if count else {max(start, 1)})
    return out


def changed_functions(base: str, head: str, files: list[str], *, repo: Path = REPO) -> set[tuple[str, str]]:
    """(file, function) pairs whose lines differ between ``base`` and ``head``."""
    diff = subprocess.run(["git", "-C", str(repo), "diff", "-U0", base, head, "--", *files],
                          check=True, capture_output=True, text=True).stdout
    result: set[tuple[str, str]] = set()
    for path, lines in changed_new_lines(diff).items():
        shown = subprocess.run(["git", "-C", str(repo), "show", f"{head}:{path}"],
                               capture_output=True, text=True)
        if shown.returncode != 0:
            continue
        for func, (lo, hi) in function_spans(shown.stdout).items():
            if any(lo <= n <= hi for n in lines):
                result.add((path, func))
    return result


# -- triage -------------------------------------------------------------------

def load_triage(path: Path = TRIAGE_FILE) -> dict[str, dict]:
    """``[[triage]]`` entries keyed by mutant name. Validates verdicts."""
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        entries = tomllib.load(fh).get("triage", [])
    out: dict[str, dict] = {}
    for entry in entries:
        name, verdict = entry.get("mutant"), entry.get("verdict")
        if not name or verdict not in TRIAGE_VERDICTS or not entry.get("note"):
            raise ValueError(f"bad triage entry {entry!r}: needs mutant, verdict "
                             f"{TRIAGE_VERDICTS} and note")
        if name in out:
            raise ValueError(f"duplicate triage entry for {name}")
        out[name] = entry
    return out


def triage_for(name: str, diff: str | None, triage: dict[str, dict]) -> dict | None:
    """The row's triage field. ``stale`` when the mutation no longer matches.

    mutmut numbers mutants per function, so an edited function can give an old
    name a different mutation. An entry's optional ``diff_contains`` pins the
    mutation it was written for; if the diff no longer contains it, the entry
    is reported but marked stale (and does not count as triaged).
    """
    entry = triage.get(name)
    if entry is None:
        return None
    pin = entry.get("diff_contains")
    stale = bool(pin) and (diff is None or pin not in diff)
    return {"verdict": entry["verdict"], "note": entry["note"], "stale": stale}


def is_triaged(row: dict) -> bool:
    t = row.get("triage")
    return bool(t) and not t.get("stale")


# -- mutmut state --------------------------------------------------------------

def _meta(work: Path, rel: str) -> dict | None:
    path = work / "mutants" / (rel + ".meta")
    return json.loads(path.read_text()) if path.exists() else None


def snapshot(work: Path, files: list[str]) -> dict:
    """Verdicts and function hashes before a run, to tell what it re-tested."""
    stats_path = work / "mutants" / "mutmut-stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    snap = {"config_fingerprint": stats.get("config_fingerprint"), "files": {}}
    for rel in files:
        meta = _meta(work, rel)
        if meta is not None:
            snap["files"][rel] = {"exit_code_by_key": meta["exit_code_by_key"],
                                  "hash_by_function_name": meta.get("hash_by_function_name", {})}
    return snap


def retested(name: str, code: int | None, func_hash: str | None,
             before_file: dict | None, config_changed: bool) -> bool:
    """Whether this run (re)decided ``name``: new, previously unrun, changed
    function, changed verdict, or a mutmut config change (which resets them)."""
    if code is None:
        return False
    if config_changed or before_file is None:
        return True
    codes = before_file["exit_code_by_key"]
    if name not in codes or codes[name] is None or codes[name] != code:
        return True
    mangled = name.partition("__mutmut_")[0].rpartition(".")[2]
    return before_file["hash_by_function_name"].get(mangled) != func_hash


def _diff_for(work: Path, rel: str, name: str) -> str | None:
    """mutmut's unified diff for one mutant; None when mutmut or the mutated
    file is unavailable (e.g. state restored from cache but never re-run)."""
    try:
        from mutmut.mutation.diff_apply import get_diff_for_mutant
    except ImportError:
        return None
    cwd = os.getcwd()
    os.chdir(work)  # mutmut resolves mutants/ relative to cwd
    try:
        return get_diff_for_mutant(name, path=rel)
    except Exception:  # stale index or missing mutated file: report, don't die
        return None
    finally:
        os.chdir(cwd)


def run_info(mode: str) -> dict:
    """Run identity from the GitHub Actions environment, or the local checkout."""
    env = os.environ
    if env.get("GITHUB_RUN_ID"):
        return {"run_id": env["GITHUB_RUN_ID"], "sha": env.get("GITHUB_SHA"),
                "ref": env.get("GITHUB_REF"), "trigger": env.get("GITHUB_EVENT_NAME"),
                "mode": mode}

    def git(*args: str) -> str | None:
        proc = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return proc.stdout.strip() or None if proc.returncode == 0 else None

    return {"run_id": "local", "sha": git("rev-parse", "HEAD"),
            "ref": git("symbolic-ref", "-q", "HEAD"), "trigger": "local", "mode": mode}


def build_rows(work: Path, module: str, files: list[str], info: dict, *,
               before: dict | None = None, triage: dict[str, dict] | None = None,
               diffs: bool = True) -> list[dict]:
    """One row per mutant in ``files`` of this module's work copy."""
    triage = triage or {}
    stats_path = work / "mutants" / "mutmut-stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    config_changed = before is not None and \
        before.get("config_fingerprint") != stats.get("config_fingerprint")
    rows: list[dict] = []
    for rel in files:
        meta = _meta(work, rel)
        if meta is None:
            continue
        source = (work / rel).read_text() if (work / rel).exists() else ""
        spans = function_spans(source) if source else {}
        lines = source.splitlines()
        hashes = meta.get("hash_by_function_name", {})
        before_file = None if before is None else before["files"].get(rel)
        for name, code in sorted(meta["exit_code_by_key"].items()):
            func = function_of(name)
            diff = _diff_for(work, rel, name) if diffs else None
            mangled = name.partition("__mutmut_")[0].rpartition(".")[2]
            rows.append({
                "schema_version": SCHEMA_VERSION, **{k: info.get(k) for k in
                                                     ("run_id", "sha", "ref", "trigger", "mode")},
                "module": module, "file": rel, "function": func,
                "line": changed_line(diff or "", lines, spans.get(func)),
                "mutant_name": name, "status": report_status(code),
                "mutmut_status": mutmut_status(code),
                "retested_this_run": None if before is None else
                retested(name, code, hashes.get(mangled), before_file, config_changed),
                "diff": diff, "triage": triage_for(name, diff, triage),
            })
    return rows


# -- summaries ------------------------------------------------------------------

def score_block(counts: Counter) -> dict:
    """Counts plus mutmut's score convention: (killed + timeout) / checked,
    where checked excludes skipped (never run). no_tests counts against it."""
    c = {s: counts.get(s, 0) for s in STATUSES}
    total = sum(c.values())
    checked = total - c["skipped"]
    score = round((c["killed"] + c["timeout"]) / checked, 4) if checked else None
    return {"total": total, **c, "checked": checked, "score": score,
            "survived_untriaged": counts.get("survived_untriaged", 0)}


def _counts(rows: list[dict]) -> Counter:
    c = Counter(r["status"] for r in rows)
    c["survived_untriaged"] = sum(1 for r in rows if r["status"] in ("survived", "no_tests")
                                  and not is_triaged(r))
    return c


def summarize(rows: list[dict], module: str, info: dict, files: list[str], **extra) -> dict:
    by_file = {rel: [r for r in rows if r["file"] == rel] for rel in files}
    return {"schema_version": SCHEMA_VERSION, **info, "module": module, **extra,
            **score_block(_counts(rows)),
            "retested_this_run": sum(1 for r in rows if r["retested_this_run"]),
            "files": {rel: score_block(_counts(rs)) for rel, rs in by_file.items()}}


def merge_summaries(summaries: list[dict]) -> dict:
    """The merged artifact's summary.json: every module plus overall totals."""
    first = summaries[0] if summaries else {}
    totals: Counter = Counter()
    for s in summaries:
        totals.update({k: s[k] for k in STATUSES + ("survived_untriaged",)})
    return {"schema_version": SCHEMA_VERSION,
            **{k: first.get(k) for k in ("run_id", "sha", "ref", "trigger", "mode")},
            **score_block(totals),
            "modules": {s["module"]: {k: v for k, v in s.items() if k not in
                                      ("run_id", "sha", "ref", "trigger", "schema_version")}
                        for s in sorted(summaries, key=lambda s: s["module"])}}


def _pct(score: float | None) -> str:
    return "--" if score is None else f"{100 * score:.1f}%"


class MergeError(Exception):
    """An unusable merge input set or a broken module contract: refuse, never crash."""


def null_block() -> dict:
    """A score block where nothing can be honestly stated: every count is null,
    never a fabricated zero over an empty or ambiguous module set."""
    return {"total": None, **{s: None for s in STATUSES}, "checked": None,
            "score": None, "survived_untriaged": None}


def _num(v) -> str:
    return "--" if v is None else str(v)


def parse_expected_modules(raw: str | None) -> list[str] | None:
    """The plan's module contract -> a list of distinct module names, or ``None``
    when no contract was given. Mirrors ``gremlin_results.parse_expected_modules``:
    anything that is not a JSON array of names is a broken contract and a refusal,
    never a silent "expected nothing"."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        raise MergeError("--expected-modules is empty: the plan published no module list")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MergeError(f"--expected-modules is not valid JSON: {exc}") from exc
    if not isinstance(doc, list) or not all(isinstance(m, str) and m for m in doc):
        raise MergeError(f"--expected-modules must be a JSON array of module names, "
                         f"got {text!r}")
    if (dup := [m for m, n in Counter(doc).items() if n > 1]):
        raise MergeError(f"--expected-modules lists duplicate module(s): {sorted(dup)}")
    return doc


def markdown(summary: dict, rows: list[dict], changed: set[tuple[str, str]] | None,
             *, limit: int = 40) -> str:
    """Job summary: score table, then survivors in the changed functions."""
    out = [f"### Mutation testing: `{summary['module']}` ({summary.get('mode')})", "",
           "| file | total | killed | survived | no tests | timeout | other | skipped | score |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, b in [("**module**", summary)] + [(f"`{f}`", b) for f, b in summary["files"].items()]:
        out.append(f"| {label} | {b['total']} | {b['killed']} | {b['survived']} | {b['no_tests']} "
                   f"| {b['timeout']} | {b['suspicious']} | {b['skipped']} | {_pct(b['score'])} |")
    out.append("")
    if summary.get("run_exit_code") not in (None, 0):
        out += [f"**mutmut exited {summary['run_exit_code']}**: results are partial.", ""]
    live = [r for r in rows if r["status"] in ("survived", "no_tests") and not is_triaged(r)]
    if changed is None:
        title = "Untriaged survivors re-tested this run"
        picked = [r for r in live if r["retested_this_run"]]
    else:
        title = "Untriaged survivors in functions this push changed"
        picked = [r for r in live if (r["file"], r["function"]) in changed]
    out.append(f"#### {title}: {len(picked)}")
    for r in picked[:limit]:
        out += ["", f"<details><summary><code>{r['file']}:{r['line']}</code> "
                f"<code>{r['function']}</code> [{r['status']}]</summary>", "",
                "```diff", (r["diff"] or "(diff unavailable)").rstrip(), "```", "</details>"]
    if len(picked) > limit:
        out += ["", f"... and {len(picked) - limit} more in results.jsonl."]
    return "\n".join(out) + "\n"


# -- files ------------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def export_module(work: Path, module: str, files: list[str], out: Path, *, mode: str,
                  changed_base: str | None = None, run_exit_code: int | None = None,
                  elapsed: float | None = None, diffs: bool = True) -> dict:
    info = run_info(mode)
    snap_path = work / SNAPSHOT_NAME
    before = json.loads(snap_path.read_text()) if snap_path.exists() else None
    rows = build_rows(work, module, files, info, before=before, triage=load_triage(), diffs=diffs)
    changed = None
    if changed_base and info.get("sha") and set(changed_base) != {"0"}:
        try:
            changed = changed_functions(changed_base, info["sha"], files)
        except subprocess.CalledProcessError:
            changed = None  # base not fetched: fall back to "re-tested this run"
    summary = summarize(rows, module, info, files, run_exit_code=run_exit_code,
                        elapsed_seconds=elapsed)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "results.jsonl", rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "summary.md").write_text(markdown(summary, rows, changed))
    return summary


def merge_dirs(inputs: list[Path], out: Path,
               expected: list[str] | None = None) -> dict:
    """Concatenate per-module results and merge their summaries into ``out``.

    Without ``expected`` this is the historical/local merge, unchanged -- a
    module reported twice is refused, because there is no honest way to pick one.
    With ``expected`` (the plan job's module list) the merged artifact must
    describe EXACTLY that set: a missing, unexpected or duplicated module report,
    no module directory at all, or inputs from more than one run/SHA/mode make the
    artifact an incomplete diagnostic -- ``complete: false``, ``tool_error: true``,
    machine-readable ``failure_reasons`` and a withheld score -- so a subset can
    never be read as the latest completed full run. Counts stay auditable for what
    really arrived, except when nothing arrived or one module reported twice,
    where every count is null (never a fabricated zero or a silently doubled
    total)."""
    out.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    summaries, rows = [], []
    for d in inputs:
        if (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text())
            name = s["module"]
            if name in seen:
                if expected is None:
                    raise MergeError(f"module {name!r} appears in both {seen[name]} and {d}")
                duplicates.setdefault(name, [seen[name]]).append(d)
            else:
                seen[name] = d
            summaries.append(s)
        if (d / "results.jsonl").exists():
            rows.extend(read_jsonl(d / "results.jsonl"))
    rows.sort(key=lambda r: (r["module"], r["file"], r["mutant_name"]))
    merged = merge_summaries(summaries)
    if expected is not None:
        present = sorted(seen)
        missing = [m for m in expected if m not in seen]
        unexpected = [m for m in present if m not in set(expected)]
        complete_set = bool(summaries) and not (missing or unexpected or duplicates)
        mismatch = []
        for key in ("run_id", "sha", "mode"):
            if len(vals := {s.get(key) for s in summaries}) > 1:
                mismatch.append(f"{key}s {sorted(map(str, vals))}")
        contract_violated = not complete_set or bool(mismatch)
        reasons = []
        if not inputs:
            reasons.append("NO_MODULE_REPORTS: no module artifact directory was downloaded")
        if missing:
            reasons.append(f"MISSING_MODULES: expected {len(expected)} module(s), "
                           f"{len(seen)} reported; no report for {', '.join(missing)}")
        if unexpected:
            reasons.append(f"UNEXPECTED_MODULES: report(s) outside the expected set: "
                           f"{', '.join(unexpected)}")
        for name, dirs in sorted(duplicates.items()):
            reasons.append(f"DUPLICATE_MODULES: {name} reported by "
                           + " + ".join(str(d) for d in sorted(dirs, key=str))
                           + "; no single report to attribute")
        if mismatch:
            reasons.append("RUN_PROVENANCE_MISMATCH: " + "; ".join(mismatch))
        if not summaries or duplicates:
            # Nothing honest to aggregate: null counts, not zeros over an empty
            # set and not a total that silently double-counts.
            merged = {**merged, **null_block()}
        elif contract_violated:
            # Counts cover only what really arrived; no input measurement made
            # this score, so it is withheld rather than recomputed.
            merged = {**merged, "score": None}
        # A duplicated module has no single honest entry: the ambiguity lives in
        # ``module_contract.duplicate``, not resolved to whichever dir read first.
        merged["modules"] = {k: v for k, v in merged["modules"].items()
                             if k not in duplicates}
        merged.update({
            "expected_modules": list(expected),
            "module_contract": {"expected": list(expected), "present": present,
                                "missing": missing, "unexpected": unexpected,
                                "duplicate": {n: sorted(str(d) for d in ds)
                                              for n, ds in sorted(duplicates.items())},
                                "complete_set": complete_set},
            "complete": not contract_violated,
            "tool_error": contract_violated,
            "failure_reasons": reasons,
        })
    write_jsonl(out / "results.jsonl", rows)
    (out / "summary.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    lines = [f"### Mutation testing: all modules (mutmut, {merged.get('mode')})", "",
             "| module | total | killed | survived | no tests | skipped | untriaged | score |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, b in merged["modules"].items():
        lines.append(f"| `{name}` | {_num(b['total'])} | {_num(b['killed'])} | {_num(b['survived'])} | "
                     f"{_num(b['no_tests'])} | {_num(b['skipped'])} | {_num(b['survived_untriaged'])} | "
                     f"{_pct(b['score'])} |")
    lines.append(f"| **all** | {_num(merged['total'])} | {_num(merged['killed'])} | "
                 f"{_num(merged['survived'])} | {_num(merged['no_tests'])} | "
                 f"{_num(merged['skipped'])} | {_num(merged['survived_untriaged'])} | "
                 f"{_pct(merged['score'])} |")
    if expected is not None:
        lines += ["", f"Expected modules ({len(expected)}): {', '.join(expected) or '(none)'}",
                  f"Reported modules ({len(seen)}): {', '.join(present) or '(none)'}"]
        if merged["tool_error"]:
            lines += ["", "**The module reports that arrived are NOT the set the plan "
                      "expected -- this is not a valid full run. Score withheld, counts "
                      "cover only what really arrived, and the artifact is marked "
                      "INCOMPLETE (tool failure) so no consumer can read it as the "
                      "latest completed run.**"]
            lines += [f"- {r}" for r in merged["failure_reasons"]
                      if r.startswith(("NO_MODULE_REPORTS", "MISSING_MODULES",
                                       "UNEXPECTED_MODULES", "DUPLICATE_MODULES",
                                       "RUN_PROVENANCE"))]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return merged


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(REPO / "tools"))
    import mutation_pilot as pilot

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="write results.jsonl, summary.json, summary.md for a module")
    p.add_argument("module")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("full", "incremental", "local"), default="local")
    p.add_argument("--changed-base", default=None, help="sha the push started from")
    p.add_argument("--run-exit-code", type=int, default=None)
    p.add_argument("--elapsed", type=float, default=None, help="run wall time, seconds")
    p.add_argument("--no-diffs", action="store_true")
    p = sub.add_parser("merge", help="merge per-module outputs into one report")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-modules", default=None,
                   help="JSON array of the module names the plan expects; a report set "
                        "that is not exactly this list is an incomplete tool error")
    # nargs="*": the report job calls merge even when the download found no
    # module directory at all, so the expected-module contract can write the
    # incomplete diagnostic instead of the job passing on an empty set.
    p.add_argument("inputs", nargs="*", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "merge":
        try:
            merged = merge_dirs(args.inputs, args.out,
                                parse_expected_modules(args.expected_modules))
        except MergeError as exc:
            print(f"mutation_results: refusing to merge: {exc}", file=sys.stderr)
            return 2
        print(f"merged {len(merged['modules'])} modules, {_num(merged['total'])} mutants "
              f"(score {_pct(merged['score'])}, "
              f"{'TOOL FAILURE' if merged.get('tool_error') else 'clean'}) -> {args.out}")
        for reason in merged.get("failure_reasons", []):
            print(f"mutation_results: {reason}", file=sys.stderr)
        return 1 if merged.get("tool_error") else 0
    cfg = pilot.load_config()
    files = pilot.mutate_files(cfg, args.module)
    summary = export_module(pilot.home() / args.module, args.module, files, args.out,
                            mode=args.mode, changed_base=args.changed_base,
                            run_exit_code=args.run_exit_code, elapsed=args.elapsed,
                            diffs=not args.no_diffs)
    print(f"{args.module}: {summary['total']} mutants, score {_pct(summary['score'])}, "
          f"{summary['survived_untriaged']} untriaged survivors -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
