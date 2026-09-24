"""pytest-gremlins report adapter: raw 1.9.0 JSON -> queryable artifacts (schema 2).

The gremlins CI branch runs ``pytest-gremlins`` per module; one run writes one
raw report (``coverage/gremlins/gremlins.json``). This adapter converts that
raw JSON into the same artifact files ``tools/mutation_results.py`` writes for
mutmut -- ``results.jsonl``, ``summary.json``, ``summary.md`` -- plus an
untouched copy of the raw report (``gremlins.json``) in the module directory
for audit, and merges per-module artifacts into one report::

    python3 tools/gremlin_results.py export MODULE --raw PATH --out DIR
            [--mode M] [--changed-base SHA] [--run-exit-code N] [--elapsed S]
            [--source-root DIR]
    python3 tools/gremlin_results.py merge --out DIR [--expected-modules JSON]
            [IN_DIR ...]

Schema version 2 rows carry the backend explicitly: ``schema_version`` 2,
``backend`` "pytest-gremlins", ``backend_version`` "1.9.0" and a stable
``policy`` fingerprint (operator set as observed in the raw report, the pinned
30 s per-gremlin timeout and the score formula). Legacy field names are kept
where they are honest: ``mutant_name`` is the gremlin id, and ``mutmut_status``,
``retested_this_run``, ``diff`` and ``triage`` are always ``null`` -- gremlins
publishes no mutmut names, diffs or cache/retest provenance, and inventing one
is worse than a null.

Raw status -> report status: zapped->killed, survived->survived,
timeout->timeout, error->suspicious (a tool error: counted as checked, never
as a kill), pardoned->excluded (kept out of the score's denominator). The
gremlins score is ``(zapped + timeout) / (total - pardoned)``.

Honesty rules this file enforces: a missing, empty, malformed, stale or
internally inconsistent raw report is an incomplete artifact and a tool
failure -- it never produces a 100% score or a fabricated empty measurement
(counts are ``null`` when the raw cannot be trusted). A nonzero
``--run-exit-code`` (including the runner's timeout ``-1``) still writes an
honest partial artifact when the raw parses, flagged ``complete: false`` /
``tool_error: true`` with machine-detectable ``failure_reasons``. Survivors
never fail anything; only the tool does. And a merge is only complete when the
module reports that arrived are EXACTLY the set the plan said would run
(``--expected-modules``): a missing, extra or duplicated module report yields an
incomplete tool-error diagnostic naming the gap, never a valid-looking subset
that a downstream reader would take for a full run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]

BACKEND = "pytest-gremlins"
BACKEND_VERSION = "1.9.0"
SCHEMA_VERSION = 2
TIMEOUT_SECONDS = 30
SCORE_FORMULA = "(zapped+timeout)/(total-pardoned)"

# The pinned 1.9.0 raw vocabulary and its mapping to the report's five.
RAW_STATUSES = ("zapped", "survived", "timeout", "error", "pardoned")
STATUS_MAP = {"zapped": "killed", "survived": "survived", "timeout": "timeout",
              "error": "suspicious", "pardoned": "excluded"}
STATUSES = ("killed", "survived", "timeout", "suspicious", "excluded")
SUMMARY_FIELDS = ("total", "zapped", "survived", "timeout", "error", "pardoned", "percentage")

MODULE_LEVEL = "<module>"  # stable marker for a mutation outside any function
DEFAULT_RAW = "coverage/gremlins/gremlins.json"
RAW_COPY = "gremlins.json"
#: A raw report older than the run plus this grace was not written by this run.
STALE_GRACE_SECONDS = 3600.0

ROW_FIELDS = ("schema_version", "backend", "backend_version", "policy", "run_id", "sha",
              "ref", "trigger", "mode", "module", "file", "function", "line",
              "mutant_name", "status", "backend_status", "gremlin_id", "operator",
              "description", "killing_test", "error_output", "execution_time_ms",
              "selected_tests", "mutmut_status", "retested_this_run", "diff", "triage")
#: Gremlins-only columns appended to ``mutation_report.py``'s CSV when rows carry them.
CSV_EXTRA_FIELDS = ("backend", "backend_version", "policy", "gremlin_id", "backend_status",
                    "operator", "description", "killing_test", "error_output",
                    "execution_time_ms", "selected_tests")


class MergeError(Exception):
    """Inputs are not one consistent backend/policy set; merge refuses rather
    than silently compare incompatible artifacts."""


# -- run identity and policy -----------------------------------------------------

def run_info(mode: str) -> dict:
    """Run identity from the GitHub Actions environment. No git fallback: this
    adapter sits beside gremlins' raw output, not the pilot's work copies."""
    env = os.environ
    if env.get("GITHUB_RUN_ID"):
        return {"run_id": env["GITHUB_RUN_ID"], "sha": env.get("GITHUB_SHA"),
                "ref": env.get("GITHUB_REF"), "trigger": env.get("GITHUB_EVENT_NAME"),
                "mode": mode}
    return {"run_id": "local", "sha": env.get("GITHUB_SHA"), "ref": None,
            "trigger": "local", "mode": mode}


def build_policy(operators) -> dict:
    return {"backend": BACKEND, "backend_version": BACKEND_VERSION,
            "timeout_seconds": TIMEOUT_SECONDS, "score": SCORE_FORMULA,
            "operators": sorted(set(operators))}


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def policy_fingerprint(policy: dict) -> str:
    """Stable id of one artifact's full policy, operators included (rows/summary)."""
    return hashlib.sha256(_canonical(policy).encode()).hexdigest()[:16]


def policy_identity(policy: dict | None) -> str | None:
    """Comparison id: the scoring policy WITHOUT the operators observed in one
    artifact. Which operators happened to fire legitimately differs between
    modules of one run and between weeks; a changed timeout or score formula
    is a different policy and must never be compared against."""
    if not isinstance(policy, dict):
        return None
    return policy_fingerprint({k: v for k, v in policy.items() if k != "operators"})


def identity_relabel_risk(policy: dict | None, backend_version) -> str | None:
    """Why an artifact's identity cannot be carried into a merged report under
    this adapter's constants (and would have to be relabelled), or ``None``
    when it is exactly the supported one and can be preserved verbatim."""
    expected = {k: v for k, v in build_policy([]).items() if k != "operators"}
    observed = {k: v for k, v in (policy or {}).items() if k != "operators"}
    if backend_version != observed.get("backend_version"):
        return (f"summary backend_version {backend_version!r} disagrees with its "
                f"policy {observed.get('backend_version')!r}")
    if observed != expected:
        return (f"unsupported identity {sorted(observed.items())} "
                f"!= {sorted(expected.items())}")
    return None


# -- raw validation ----------------------------------------------------------------

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def path_ok(p) -> bool:
    """Repo-relative POSIX path, no traversal, no absolute, no drive letters."""
    if not isinstance(p, str) or not p or p.startswith("/") or "\\" in p or ":" in p:
        return False
    parts = PurePosixPath(p).parts
    return bool(parts) and ".." not in parts


_OPTIONAL_CHECKS = {
    "killing_test": lambda v: v is None or isinstance(v, str),
    "error_output": lambda v: v is None or isinstance(v, str),
    "execution_time_ms": lambda v: v is None or (_is_int(v) or (isinstance(v, float) and not isinstance(v, bool))) and v >= 0,
    "selected_tests": lambda v: v is None or (isinstance(v, list) and all(isinstance(x, str) for x in v)),
}


def item_problems(res) -> list[str]:
    if not isinstance(res, dict):
        return ["result is not an object"]
    bad = []
    if not (isinstance(res.get("gremlin_id"), str) and res["gremlin_id"]):
        bad.append("gremlin_id")
    if not path_ok(res.get("file_path")):
        bad.append("file_path")
    if not (_is_int(res.get("line_number")) and res.get("line_number") >= 1):
        bad.append("line_number")
    if res.get("status") not in RAW_STATUSES:
        bad.append("status")
    for field in ("operator", "description"):
        if not isinstance(res.get(field), str):
            bad.append(field)
    for field, ok in _OPTIONAL_CHECKS.items():
        if field in res and not ok(res[field]):
            bad.append(field)
    return bad


def validate_raw(doc) -> tuple[list | None, dict | None, list[str], list[str]]:
    """Pinned 1.9.0 shape check -> (results, summary, fatal, inconsistent).

    ``fatal`` means nothing here can be trusted (no rows, no counts); the
    shape parses but the declared counts disagree with the results themselves
    (``inconsistent``): rows and counts stay auditable, the score is withheld.
    """
    fatal: list[str] = []
    if not isinstance(doc, dict):
        return None, None, ["raw report is not a JSON object"], []
    missing = [k for k in ("summary", "files", "results") if k not in doc]
    if missing:
        return None, None, [f"raw report lacks top-level key(s): {missing}"], []
    summary, results = doc["summary"], doc["results"]
    if not isinstance(summary, dict):
        fatal.append("summary is not an object")
    elif (gaps := [k for k in SUMMARY_FIELDS if k not in summary]):
        fatal.append(f"summary lacks field(s): {gaps}")
    else:
        for k in SUMMARY_FIELDS[:-1]:
            if not _is_int(summary[k]) or summary[k] < 0:
                fatal.append(f"summary.{k} is not a non-negative integer")
        pct = summary["percentage"]
        if isinstance(pct, bool) or not isinstance(pct, (int, float)):
            fatal.append("summary.percentage is not numeric")
    if not isinstance(results, list):
        fatal.append("results is not a list")
    else:
        for i, res in enumerate(results):
            if (bad := item_problems(res)):
                fatal.append(f"results[{i}] malformed: {', '.join(bad)}")
    if fatal:
        return None, None, fatal, []

    incons: list[str] = []
    by = Counter(r["status"] for r in results)
    if sum(summary[k] for k in RAW_STATUSES) != summary["total"]:
        incons.append("summary statuses do not sum to summary.total")
    if summary["total"] != len(results):
        incons.append(f"summary.total {summary['total']} but results hold {len(results)}")
    for k in RAW_STATUSES:
        if summary[k] != by.get(k, 0):
            incons.append(f"summary.{k}={summary[k]} but results contain {by.get(k, 0)}")
    ids = [r["gremlin_id"] for r in results]
    if len(ids) != len(set(ids)):
        incons.append("duplicate gremlin_id in results")
    denom = summary["total"] - summary["pardoned"]
    expected = round(100.0 * (summary["zapped"] + summary["timeout"]) / denom, 1) if denom else 0.0
    if abs(expected - summary["percentage"]) > 1.0:
        incons.append(f"summary.percentage {summary['percentage']} != recomputed {expected}")
    return results, summary, [], incons


def load_raw(raw_path: Path):
    """Read and parse the raw report -> (doc, raw_bytes, problems)."""
    try:
        raw_bytes = raw_path.read_bytes()
    except OSError as exc:
        return None, None, [f"raw report missing or unreadable at {raw_path} "
                            f"({exc.strerror or exc})"]
    if not raw_bytes.strip():
        return None, raw_bytes, ["raw report is empty"]
    try:
        return json.loads(raw_bytes.decode("utf-8")), raw_bytes, []
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, raw_bytes, [f"raw report is not valid JSON: {exc}"]


# -- source positions ----------------------------------------------------------------

class SourceIndex:
    """Enclosing-function lookup by (file, line) from the checked-out source.

    The function is derived from the AST of the file the run reported -- never
    from the gremlin id, which carries no name gremlins could have fabricated
    mutmut-style. A missing or unparseable file yields ``None``; a line
    outside any function (module-level mutations) yields ``MODULE_LEVEL``.
    """

    def __init__(self, root: Path):
        self.root, self._trees = root, {}

    def function_at(self, rel: str, line: int) -> str | None:
        if rel not in self._trees:
            try:
                self._trees[rel] = ast.parse((self.root / rel).read_text())
            except (OSError, SyntaxError, ValueError):
                self._trees[rel] = None
        tree = self._trees[rel]
        if tree is None:
            return None
        found: list[str] = []
        self._search(tree, "", line, found)
        return found[-1] if found else MODULE_LEVEL

    def _search(self, node, prefix: str, line: int, found: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                if start <= line <= child.end_lineno:
                    name = prefix + child.name
                    found.append(name)
                    self._search(child, name + ".", line, found)
            elif isinstance(child, ast.ClassDef):
                if child.lineno <= line <= child.end_lineno:
                    self._search(child, prefix + child.name + ".", line, found)
            else:
                self._search(child, prefix, line, found)


# -- rows and blocks -----------------------------------------------------------------

def build_rows(results: list[dict], module: str, info: dict, fingerprint: str,
               sources: SourceIndex) -> list[dict]:
    rows = []
    for res in results:
        rel, line = res["file_path"], res["line_number"]
        rows.append({
            "schema_version": SCHEMA_VERSION, "backend": BACKEND,
            "backend_version": BACKEND_VERSION, "policy": fingerprint,
            **{k: info.get(k) for k in ("run_id", "sha", "ref", "trigger", "mode")},
            "module": module, "file": rel, "function": sources.function_at(rel, line),
            "line": line, "mutant_name": res["gremlin_id"],
            "status": STATUS_MAP[res["status"]], "backend_status": res["status"],
            "gremlin_id": res["gremlin_id"], "operator": res["operator"],
            "description": res["description"], "killing_test": res.get("killing_test"),
            "error_output": res.get("error_output"),
            "execution_time_ms": res.get("execution_time_ms"),
            "selected_tests": res.get("selected_tests"),
            # Honest nulls: gremlins 1.9.0 publishes none of these.
            "mutmut_status": None, "retested_this_run": None, "diff": None, "triage": None,
        })
    return rows


def score_block(counts: Counter) -> dict:
    """Gremlins' score convention: (killed + timeout) / (total - excluded).
    Errors sit in the denominator (checked, never killed); pardoned are out."""
    c = {s: counts.get(s, 0) for s in STATUSES}
    total = sum(c.values())
    checked = total - c["excluded"]
    score = round((c["killed"] + c["timeout"]) / checked, 4) if checked else None
    return {"total": total, **c, "checked": checked, "score": score,
            "survived_untriaged": c["survived"]}


def null_block() -> dict:
    return {"total": None, **{s: None for s in STATUSES}, "checked": None,
            "score": None, "survived_untriaged": None}


def backend_counts(raw_counts: Counter) -> dict:
    return {s: raw_counts.get(s, 0) for s in RAW_STATUSES}


# -- summaries and markdown ------------------------------------------------------------

def _pct(score) -> str:
    return "--" if score is None else f"{100 * score:.1f}%"


def _num(v) -> str:
    return "--" if v is None else str(v)


def markdown(summary: dict, rows: list[dict], *, limit: int = 40) -> str:
    head = [f"### Mutation testing: `{summary['module']}` "
            f"(pytest-gremlins {summary['backend_version']}, {summary.get('mode')})", "",
            "| file | total | killed | survived | timeout | error | pardoned | score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    blocks = [("**module**", summary)] + [(f"`{f}`", b) for f, b in summary["files"].items()]
    for label, b in blocks:
        head.append(f"| {label} | {_num(b['total'])} | {_num(b.get('killed'))} "
                    f"| {_num(b.get('survived'))} | {_num(b.get('timeout'))} "
                    f"| {_num(b.get('suspicious'))} | {_num(b.get('excluded'))} "
                    f"| {_pct(b.get('score'))} |")
    head.append("")
    if summary.get("run_exit_code") not in (None, 0):
        rc = summary["run_exit_code"]
        head += [f"**pytest-gremlins exited {rc}"
                 + (" (step timeout)" if rc == -1 else "")
                 + ": results are INCOMPLETE (tool failure).**", ""]
    if not summary["complete"] or summary["tool_error"]:
        status = "TOOL FAILURE" if summary["tool_error"] else "INCOMPLETE"
        head += [f"**Artifact status: {status}** "
                 f"({', '.join(summary['failure_reasons']) or 'see problems'})."]
    if summary["problems"]:
        head += ["", "Raw report problems (nothing here is fabricated):"]
        head += [f"- {p}" for p in summary["problems"][:10]]
    if summary["counts_backend"] and summary["counts_backend"]["error"]:
        head += ["", f"{summary['counts_backend']['error']} gremlin(s) ERRORED: tool errors, "
                      "counted as checked, never as killed."]
    survivors = [r for r in rows if r["status"] == "survived"]
    if summary["total"] is not None:  # an unusable raw has no survivors to claim or deny
        head += ["", f"#### Survivors: {len(survivors)}"]
    for r in survivors[:limit]:
        head.append(f"- `{r['file']}:{r['line']}` `{r['function']}` — `{r['operator']}`: "
                    f"{r['description']}")
    if len(survivors) > limit:
        head.append(f"- ... and {len(survivors) - limit} more in results.jsonl.")
    return "\n".join(head) + "\n"


# -- export -----------------------------------------------------------------------------

def export_module(module: str, raw_path: Path, out: Path, *, mode: str,
                  changed_base: str | None = None, run_exit_code: int | None = None,
                  elapsed: float | None = None, source_root: Path = REPO) -> dict:
    info = run_info(mode)
    doc, raw_bytes, problems = load_raw(raw_path)
    fatal, incons = list(problems), []
    results = summary = None
    age = None
    if doc is not None:
        results, summary, extra_fatal, incons = validate_raw(doc)
        fatal += extra_fatal
        try:
            age = time.time() - raw_path.stat().st_mtime
            if age > (elapsed or 0.0) + STALE_GRACE_SECONDS:
                fatal.append(f"raw report is STALE: last written {age:.0f}s before export "
                             f"(run took {elapsed or 0:.0f}s + {STALE_GRACE_SECONDS:.0f}s grace)")
        except OSError:
            pass
    usable = results is not None and not fatal

    policy = build_policy(r["operator"] for r in results) if usable else build_policy([])
    fingerprint = policy_fingerprint(policy)
    rows = build_rows(results, module, info, fingerprint, SourceIndex(source_root)) if usable else []

    if usable:
        block = score_block(Counter(r["status"] for r in rows))
        counts_b = backend_counts(Counter(r["status"] for r in results))
        if incons:
            block = {**block, "score": None}  # counts auditable, score withheld
        by_file = {rel: score_block(Counter(r["status"] for r in rows if r["file"] == rel))
                   for rel in sorted({r["file"] for r in rows})}
    else:
        block, counts_b, by_file = null_block(), {s: None for s in RAW_STATUSES}, {}

    reasons = []
    if fatal:
        reasons.append("RAW_INVALID")
    # Honest changed-base provenance: the gremlins raw carries no per-function
    # diff, so the base sha is recorded, never used to re-attribute survivors.
    # An all-zeros base (a branch's first push) means "no base to compare to".
    changed_base = changed_base if changed_base and set(changed_base) != {"0"} else None

    if incons:
        reasons.append("RAW_INCONSISTENT")
    if run_exit_code not in (None, 0):
        reasons.append("TIMEOUT_KILL" if run_exit_code == -1 else "RUN_INCOMPLETE")
    if counts_b.get("error"):
        reasons.append("ERROR_RESULTS")
    complete = usable and not incons and run_exit_code in (None, 0)
    tool_error = not complete or bool(counts_b.get("error"))

    summary_doc = {
        "schema_version": SCHEMA_VERSION, "backend": BACKEND,
        "backend_version": BACKEND_VERSION, "policy": policy,
        "policy_fingerprint": fingerprint,
        **info, "module": module,
        "changed_base": changed_base,
        "raw_path": raw_path.as_posix(),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None,
        "raw_age_seconds": round(age, 1) if age is not None and usable else None,
        "run_exit_code": run_exit_code, "elapsed_seconds": elapsed,
        **block, "counts_backend": counts_b, "raw_summary": summary,
        "files": by_file,
        "complete": complete, "tool_error": tool_error,
        "failure_reasons": reasons, "problems": fatal + incons,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "results.jsonl", rows)
    (out / "summary.json").write_text(json.dumps(summary_doc, indent=2, sort_keys=True) + "\n")
    (out / "summary.md").write_text(markdown(summary_doc, rows))
    if raw_bytes is not None:  # audit copy: exactly the bytes the backend wrote
        (out / RAW_COPY).write_bytes(raw_bytes)
    return summary_doc


# -- merge -------------------------------------------------------------------------------

def read_artifact(d: Path) -> tuple[list[dict], dict]:
    sp, rp = d / "summary.json", d / "results.jsonl"
    if not sp.is_file() or not rp.is_file():
        raise MergeError(f"{d} is not a gremlins report directory (needs summary.json "
                         "+ results.jsonl)")
    try:
        summary = json.loads(sp.read_text())
    except json.JSONDecodeError as exc:
        raise MergeError(f"{d}: unreadable summary.json ({exc})") from exc
    rows = read_jsonl(rp)
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("backend") != BACKEND:
        raise MergeError(f"{d}: not a schema-2 {BACKEND} artifact (schema 1 / mutmut "
                         "reports are merged by tools/mutation_results.py, never here)")
    for r in rows:
        if r.get("schema_version") != SCHEMA_VERSION or r.get("backend") != BACKEND:
            raise MergeError(f"{d}: results.jsonl mixes schema 1 and schema 2 rows")
    return rows, summary


def parse_expected_modules(raw: str | None) -> list[str] | None:
    """The plan's module contract -> a list of distinct module names, or ``None``
    when no contract was given.

    The value is exactly the ``plan`` job's ``modules`` output (the JSON array
    ``gremlin_pilot.py matrix`` prints). Anything that is not such an array is a
    broken contract and a refusal, never a silent "expected nothing": an empty
    list would make every arriving report unexpected, and a truncated one would
    invent missing modules out of a shell quoting mistake."""
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


def merge_dirs(inputs: list[Path], out: Path,
               expected: list[str] | None = None) -> dict:
    seen: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    identities: dict[str, None] = {}
    versions: set = set()
    run_ids: set = set()
    shas: set = set()
    modes: set = set()
    rows, summaries = [], []
    for d in inputs:
        rs, s = read_artifact(d)
        name = s["module"]
        if name in seen:
            # Without a contract, two reports for one module is an unusable
            # input set: refuse wholesale. With one, the contract is precisely
            # what catches it -- so the duplicate is recorded and the merged
            # artifact becomes an incomplete diagnostic naming both
            # directories. Silently keeping one and dropping the other would
            # fabricate a module set out of an ambiguous input.
            if expected is None:
                raise MergeError(f"module {name!r} appears in both {seen[name]} and {d}")
            duplicates.setdefault(name, [seen[name]]).append(d)
        else:
            seen[name] = d
        # The merged report is emitted under THIS adapter's identity, so every
        # input must already carry exactly that identity: merging a different
        # backend_version, or a timeout/score that differs from the pinned one,
        # would silently relabel a measurement. Reject rather than rewrite.
        if (risk := identity_relabel_risk(s.get("policy"), s.get("backend_version"))) is not None:
            raise MergeError(f"{d}: {risk}; merge refuses rather than relabel a measurement")
        ident = policy_identity(s.get("policy"))
        if ident is None:
            raise MergeError(f"{d}: artifact carries no policy fingerprint")
        identities.setdefault(ident, None)
        versions.add(s.get("backend_version"))
        run_ids.add(s.get("run_id"))
        shas.add(s.get("sha"))
        modes.add(s.get("mode"))
        rows.extend(rs)
        summaries.append(s)
    if len(identities) > 1:
        raise MergeError(f"mixed scoring policy ({sorted(identities)}); merge refuses")
    if len(versions) > 1:
        raise MergeError(f"mixed backend version {sorted(map(str, versions))}; merge refuses")

    # One merged report describes ONE run of ONE mode: a full artifact merged
    # with an incremental one (or two runs / two SHAs) is not any single
    # measurement, so it is flagged incomplete/tool-error and its score
    # withheld -- never presented as a valid full measurement.
    mismatch = []
    if len(modes) > 1:
        mismatch.append(f"modes {sorted(map(str, modes))}")
    if len(shas) > 1:
        mismatch.append(f"source SHAs {sorted(map(str, shas))}")
    if len(run_ids) > 1:
        mismatch.append(f"run ids {sorted(map(str, run_ids))}")
    provenance_ok = not mismatch

    # The expected-module contract (the plan's module list, published by the
    # workflow's plan job). The merged report claims to describe that run, so a
    # report set that is not exactly the expected set -- a module whose job died
    # before it exported/uploaded, an artifact from a module the plan never
    # scheduled, one module reported twice, or nothing downloaded at all -- is
    # NOT a valid full run. It is written as an incomplete tool-error
    # diagnostic naming the gap, so no consumer can read a subset as latest
    # completed. Without --expected-modules the contract is simply not asserted
    # (historical/local merges keep working).
    present = sorted(seen)
    missing = [m for m in (expected or []) if m not in seen]
    unexpected = [m for m in present if expected is not None and m not in set(expected)]
    complete_set = bool(summaries) and not (missing or unexpected or duplicates)
    contract_violated = (not complete_set) if expected is not None else not summaries

    if not summaries or duplicates:
        # Nothing arrived, or one module arrived twice: there is no honest
        # aggregate to state, so every count is null (never a fabricated zero
        # over an empty set, never a total that silently double-counts).
        merged_block = null_block()
    elif all(s["total"] is not None for s in summaries):
        merged_block = score_block(Counter(r["status"] for r in rows))
        if (any(s["score"] is None and s["checked"] != 0 for s in summaries)
                or not provenance_ok or contract_violated):
            # A module withheld its score (inconsistent raw), the inputs are not
            # one run/mode, or the module set is not the expected one: the
            # merged artifact must not recompute a number no input measurement
            # actually made. Counts stay auditable for what really arrived.
            merged_block = {**merged_block, "score": None}
    else:
        merged_block = null_block()
    complete = (all(s["complete"] for s in summaries) if summaries else False) \
        and provenance_ok and not contract_violated
    reasons = [f"{s['module']}: {r}" for s in summaries for r in s["failure_reasons"]]
    if not provenance_ok:
        reasons.append("RUN_PROVENANCE_MISMATCH: " + "; ".join(mismatch))
    if not inputs:
        reasons.insert(0, "NO_MODULE_REPORTS: no module artifact directory was downloaded")
    for name, dirs in sorted(duplicates.items()):
        reasons.append(f"DUPLICATE_MODULES: {name} reported by "
                       + " + ".join(str(d) for d in sorted(dirs, key=str))
                       + "; no single report to attribute")
    if missing:
        reasons.append(f"MISSING_MODULES: expected {len(expected)} module(s), "
                       f"{len(seen)} reported; no report for {', '.join(missing)}")
    if unexpected:
        reasons.append(f"UNEXPECTED_MODULES: report(s) outside the expected set: "
                       f"{', '.join(unexpected)}")
    merged = {
        "schema_version": SCHEMA_VERSION, "backend": BACKEND,
        "backend_version": BACKEND_VERSION,
        "policy": build_policy(r["operator"] for r in rows),
        "run_id": summaries[0]["run_id"] if summaries else None,
        "sha": summaries[0]["sha"] if summaries else None,
        "ref": summaries[0]["ref"] if summaries else None,
        "trigger": summaries[0]["trigger"] if summaries else None,
        "mode": summaries[0]["mode"] if summaries else None,
        **merged_block,
        "complete": complete,
        "tool_error": any(s["tool_error"] for s in summaries) or not provenance_ok
                      or contract_violated,
        "failure_reasons": reasons,
        "expected_modules": list(expected) if expected is not None else None,
        "module_contract": None if expected is None else {
            "expected": list(expected), "present": present, "missing": missing,
            "unexpected": unexpected,
            "duplicate": {n: sorted(str(d) for d in ds)
                          for n, ds in sorted(duplicates.items())},
            "complete_set": complete_set},
        # A module that reported more than once has no honest entry here: the
        # map describes one report per module, so the ambiguity lives in
        # ``module_contract.duplicate`` and the name is left out rather than
        # resolved to whichever directory happened to be read first.
        "modules": {s["module"]: {k: v for k, v in s.items() if k not in
                                  ("run_id", "sha", "ref", "trigger", "schema_version",
                                   "backend", "backend_version")}
                    for s in sorted(summaries, key=lambda s: s["module"])
                    if s["module"] not in duplicates},
    }
    merged["policy_fingerprint"] = policy_fingerprint(merged["policy"])
    rows.sort(key=lambda r: (r["module"], r["file"], r["mutant_name"]))
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "results.jsonl", rows)
    (out / "summary.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    lines = [f"### Mutation testing: all modules (pytest-gremlins {BACKEND_VERSION}, "
             f"{merged.get('mode')})", "",
             "| module | total | killed | survived | timeout | error | pardoned | score | status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, b in merged["modules"].items():
        state = "ok" if b["complete"] and not b["tool_error"] else \
            "TOOL FAILURE" if b["tool_error"] else "INCOMPLETE"
        lines.append(f"| `{name}` | {_num(b['total'])} | {_num(b['killed'])} "
                     f"| {_num(b['survived'])} | {_num(b['timeout'])} "
                     f"| {_num(b['suspicious'])} | {_num(b['excluded'])} "
                     f"| {_pct(b['score'])} | {state} |")
    lines.append(f"| **all** | {_num(merged['total'])} | {_num(merged['killed'])} "
                 f"| {_num(merged['survived'])} | {_num(merged['timeout'])} "
                 f"| {_num(merged['suspicious'])} | {_num(merged['excluded'])} "
                  f"| {_pct(merged['score'])} | "
                  f"{'ok' if merged['complete'] and not merged['tool_error'] else 'TOOL FAILURE'} |")
    if not provenance_ok:
        lines += ["", "**Inputs are not one run/mode/source SHA -- this is not a single "
                  "measurement. Score withheld, artifact marked INCOMPLETE (tool failure).**"]
    if expected is not None:
        lines += ["", f"Expected modules ({len(expected)}): {', '.join(expected) or '(none)'}",
                  f"Reported modules ({len(seen)}): {', '.join(present) or '(none)'}"]
    if contract_violated:
        lines += ["", "**The module reports that arrived are NOT the set the plan expected "
                  "-- this is not a valid full run. Score withheld, counts cover only what "
                  "really arrived, and the artifact is marked INCOMPLETE (tool failure) so "
                  "no consumer can read it as the latest completed run.**"]
        lines += [f"- {r}" for r in reasons
                  if r.startswith(("NO_MODULE_REPORTS", "MISSING_MODULES",
                                   "UNEXPECTED_MODULES", "DUPLICATE_MODULES"))]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return merged



# -- files --------------------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="convert one raw gremlins report into module artifacts")
    p.add_argument("module")
    p.add_argument("--raw", type=Path, default=Path(DEFAULT_RAW))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("full", "incremental", "local"), default="local")
    p.add_argument("--changed-base", default=None, help="sha the push started from (provenance only)")
    p.add_argument("--run-exit-code", type=int, default=None)
    p.add_argument("--elapsed", type=float, default=None, help="run wall time, seconds")
    p.add_argument("--source-root", type=Path, default=REPO,
                   help="root the raw file_path values are relative to (function lookup)")
    p = sub.add_parser("merge", help="merge per-module gremlins artifacts into one report")
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
            merged = merge_dirs(args.inputs, args.out, parse_expected_modules(
                args.expected_modules))
        except MergeError as exc:
            print(f"gremlin_results: refusing to merge: {exc}", file=sys.stderr)
            return 2
        print(f"merged {len(merged['modules'])} modules, {_num(merged['total'])} gremlins "
              f"(score {_pct(merged['score'])}, "
              f"{'TOOL FAILURE' if merged['tool_error'] else 'clean'}) -> {args.out}")
        for reason in merged["failure_reasons"]:
            if reason.startswith(("NO_MODULE_REPORTS", "MISSING_MODULES", "UNEXPECTED_MODULES",
                                  "DUPLICATE_MODULES")):
                print(f"gremlin_results: {reason}", file=sys.stderr)
        return 1 if merged["tool_error"] else 0

    summary = export_module(args.module, args.raw, args.out, mode=args.mode,
                            changed_base=args.changed_base,
                            run_exit_code=args.run_exit_code, elapsed=args.elapsed,
                            source_root=args.source_root)
    state = "TOOL FAILURE" if summary["tool_error"] else (
        "complete" if summary["complete"] else "INCOMPLETE")
    print(f"{args.module}: {_num(summary['total'])} gremlins, score {_pct(summary['score'])}, "
          f"{state} -> {args.out} (backend {BACKEND} {BACKEND_VERSION})")
    return 1 if summary["tool_error"] else 0


if __name__ == "__main__":
    sys.exit(main())
