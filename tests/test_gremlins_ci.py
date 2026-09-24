"""pytest-gremlins report adapter: tools/gremlin_results.py (schema 2), its merge
expected-module contract, plus the schema-2 awareness added to
tools/mutation_report.py and checks/mutation_ratchet.py, and the plan->merge
wiring in the gremlins workflow candidate.

All synthetic: the raw JSON is written in the pinned 1.9.0 shape and the source
tree is a string. No gremlins or mutmut execution, no git, no data/. The
candidate workflow is parsed as YAML only.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import time
import types
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gremlin_pilot as gp  # noqa: E402
import gremlin_results as gr  # noqa: E402
import mutation_report as rep  # noqa: E402
import mutation_results as mr  # noqa: E402
from checks import mutation_ratchet as ratchet  # noqa: E402

SOURCE = '''\
CONSTANT = 1


def add(a, b):
    return a + b


class Box:
    value = 0

    def size(self, n):
        if n > 1:
            return n
        return 0

    def wrap(self):
        def inner():
            return 1
        return inner
'''
REL = "engine/v2/toy.py"


def gremlin(gid, status, line, **extra):
    res = {"gremlin_id": gid, "file_path": REL, "line_number": line, "status": status,
           "operator": "flip-add", "description": f"{gid}: a + b became a - b"}
    res.update(extra)
    return res


def raw_doc(results, **summary_over):
    c = Counter(r["status"] for r in results)
    total = len(results)
    denom = total - c["pardoned"]
    summary = {"total": total, "zapped": c["zapped"], "survived": c["survived"],
               "timeout": c["timeout"], "error": c["error"], "pardoned": c["pardoned"],
               "percentage": round(100 * (c["zapped"] + c["timeout"]) / denom, 1) if denom else 0.0}
    summary.update(summary_over)
    return {"summary": summary, "files": {REL: {"zapped": 0}}, "results": results}


@pytest.fixture
def ci_env(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")


def export(tmp_path, doc=None, *, rc=0, elapsed=60, mode="full", name="toy",
           raw_text=None, no_raw=False, mtime_age=None, changed_base=None):
    """Synthetic run: checkout with SOURCE, raw report, export call -> (rc, out)."""
    source_root = tmp_path / "checkout"
    (source_root / "engine" / "v2").mkdir(parents=True, exist_ok=True)
    (source_root / "engine" / "v2" / "toy.py").write_text(SOURCE)
    raw = tmp_path / "coverage" / name / "gremlins.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not no_raw:
        raw.write_text(raw_text if raw_text is not None else json.dumps(doc))
        if mtime_age is not None:
            t = time.time() - mtime_age
            os.utime(raw, (t, t))
    out = tmp_path / "report" / name
    argv = ["export", name, "--raw", str(raw), "--out", str(out), "--mode", mode,
            "--run-exit-code", str(rc), "--elapsed", str(elapsed),
            "--source-root", str(source_root)]
    if changed_base is not None:  # exactly the flag the push workflow appends
        argv += ["--changed-base", changed_base]
    return gr.main(argv), out


def summary_of(out: Path) -> dict:
    return json.loads((out / "summary.json").read_text())


def legacy_row(module, status, name, **over):
    row = {"schema_version": 1, "run_id": "7", "sha": "abc", "ref": "r", "trigger": "push",
           "mode": "full", "module": module, "file": "a.py", "function": "f",
           "line": 3, "mutant_name": name, "status": status, "mutmut_status": status,
           "retested_this_run": True, "diff": "-a\n+b\n", "triage": None}
    row.update(over)
    return row


# -- valid conversion ------------------------------------------------------------------

def test_valid_raw_report_converts_to_schema2_rows(tmp_path, ci_env):
    doc = raw_doc([
        gremlin("g1", "zapped", 5, killing_test="tests/test_toy.py::t_add",
                execution_time_ms=12, selected_tests=["tests/test_toy.py"]),
        gremlin("g2", "survived", 12, operator="swap-comparison"),
        gremlin("g3", "timeout", 5),
        gremlin("g4", "pardoned", 1, operator="const-removal"),
    ])
    code, out = export(tmp_path, doc)
    assert code == 0
    rows = {r["mutant_name"]: r for r in gr.read_jsonl(out / "results.jsonl")}
    assert set(rows) == {"g1", "g2", "g3", "g4"}
    assert set(rows["g1"]) == set(gr.ROW_FIELDS)
    g1 = rows["g1"]
    assert (g1["schema_version"], g1["backend"], g1["backend_version"]) == \
        (2, "pytest-gremlins", "1.9.0")
    assert (g1["status"], g1["backend_status"]) == ("killed", "zapped")
    assert g1["function"] == "add" and g1["line"] == 5 and g1["operator"] == "flip-add"
    assert g1["killing_test"] == "tests/test_toy.py::t_add" and g1["execution_time_ms"] == 12
    assert g1["selected_tests"] == ["tests/test_toy.py"]
    assert (rows["g2"]["status"], rows["g2"]["function"]) == ("survived", "Box.size")
    assert rows["g3"]["status"] == "timeout"
    assert (rows["g4"]["status"], rows["g4"]["function"]) == ("excluded", "<module>")
    assert all(r["mutmut_status"] is None and r["retested_this_run"] is None
               and r["diff"] is None and r["triage"] is None for r in rows.values())
    assert all(r["run_id"] == "77" and r["sha"] == "deadbeef" and r["mode"] == "full"
               and r["trigger"] == "schedule" for r in rows.values())
    s = summary_of(out)
    assert (s["total"], s["checked"], s["score"]) == (4, 3, round(2 / 3, 4))  # pardoned out
    assert (s["killed"], s["survived"], s["timeout"], s["suspicious"], s["excluded"]) == \
        (1, 1, 1, 0, 1)
    assert s["counts_backend"] == {"zapped": 1, "survived": 1, "timeout": 1, "error": 0,
                                   "pardoned": 1}
    assert s["complete"] is True and s["tool_error"] is False and s["failure_reasons"] == []
    assert s["policy"]["timeout_seconds"] == 30
    assert s["policy"]["score"] == "(zapped+timeout)/(total-pardoned)"
    assert s["policy"]["operators"] == ["const-removal", "flip-add", "swap-comparison"]
    assert all(r["policy"] == s["policy_fingerprint"] for r in rows.values())
    raw_bytes = (tmp_path / "coverage" / "toy" / "gremlins.json").read_bytes()
    assert (out / "gremlins.json").read_bytes() == raw_bytes  # untouched audit copy
    assert s["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert s["raw_summary"] == doc["summary"]
    md = (out / "summary.md").read_text()
    assert "pytest-gremlins 1.9.0" in md and "Survivors: 1" in md and f"{REL}:12" in md


# -- plugin file-path normalization (finding 2: absolute paths on the runner) ---------------

def test_normalize_file_path_accepts_in_root_absolute_and_refuses_the_rest(tmp_path):
    """The runner's raw writes an absolute file_path under the checkout. An
    absolute path is repo-relative-izable only when it RESOLVES beneath the
    root; every other refusal (outside the root, traversal that escapes, wrong
    drive, non-string) and the strict relative rules must hold exactly as before."""
    root = tmp_path / "checkout"
    (root / "engine" / "v2").mkdir(parents=True)
    assert gr.normalize_file_path(str(root / REL), root) == REL           # in-root absolute
    assert gr.normalize_file_path(str(root / "top.py"), root) == "top.py"
    assert gr.normalize_file_path(REL, root) == REL                       # relative, verbatim
    assert gr.normalize_file_path("./x.py", root) == "./x.py"            # path_ok's own rules
    assert gr.normalize_file_path("/etc/passwd", root) is None            # absolute, outside
    assert gr.normalize_file_path(str(tmp_path / "elsewhere.py"), root) is None
    assert gr.normalize_file_path(str(root / "engine" / ".." / ".." / "secret.py"), root) is None
    assert gr.normalize_file_path("C:\\evil\\x.py", root) is None         # wrong drive
    assert gr.normalize_file_path("", root) is None
    assert gr.normalize_file_path(None, root) is None


def test_export_turns_the_plugin_absolute_path_into_a_valid_score(tmp_path, ci_env):
    """The real plugin shape (CI run 36062806372): an absolute file_path under
    the checkout. Export must normalize it repo-relative, resolve the function
    from source and score -- not fail RAW_INVALID -- while leaving the raw bytes
    and SHA provenance untouched."""
    abs_toy = str(tmp_path / "checkout" / REL)  # exactly what gremlins 1.9.0 writes
    doc = raw_doc([
        gremlin("g1", "zapped", 5, file_path=abs_toy, killing_test="tests/test_toy.py::t_add"),
        gremlin("g2", "survived", 12, file_path=abs_toy, operator="swap-comparison"),
    ])
    code, out = export(tmp_path, doc)
    assert code == 0
    s = summary_of(out)
    assert s["complete"] is True and s["tool_error"] is False and s["failure_reasons"] == []
    assert (s["total"], s["checked"], s["score"]) == (2, 2, 0.5)  # a real measurement
    rows = {r["mutant_name"]: r for r in gr.read_jsonl(out / "results.jsonl")}
    assert rows["g1"]["file"] == REL and rows["g1"]["function"] == "add"
    assert rows["g2"]["file"] == REL and rows["g2"]["function"] == "Box.size"
    assert list(s["files"]) == [REL]  # per-file block keyed repo-relative
    # provenance preserved: the audit copy still carries the absolute path bytes
    assert json.loads((out / "gremlins.json").read_bytes())["results"][0]["file_path"] == abs_toy
    assert s["raw_summary"] == doc["summary"]


def test_export_still_refuses_an_absolute_path_outside_the_source_root(tmp_path, ci_env):
    """Normalization is a convenience for in-checkout paths, not a licence to
    read outside it: an absolute path under a different root stays
    ``malformed: file_path`` / RAW_INVALID, with no score fabricated."""
    outside = str(tmp_path / "elsewhere" / REL)
    code, out = export(tmp_path, raw_doc([gremlin("g1", "zapped", 5, file_path=outside)]))
    assert code == 1
    s = summary_of(out)
    assert s["complete"] is False and s["tool_error"] is True
    assert "RAW_INVALID" in s["failure_reasons"]
    assert any("malformed: file_path" in p for p in s["problems"])
    assert s["total"] is None and s["score"] is None


# -- zero mutants and survivors -----------------------------------------------------------

def test_zero_mutants_is_an_honest_undefined_score(tmp_path, ci_env):
    code, out = export(tmp_path, raw_doc([]))
    assert code == 0
    s = summary_of(out)
    assert s["complete"] is True and s["tool_error"] is False
    assert s["total"] == 0 and s["checked"] == 0 and s["score"] is None  # undefined, not 100%
    assert gr.read_jsonl(out / "results.jsonl") == []
    assert "--" in (out / "summary.md").read_text()


def test_survivors_only_is_a_success_and_scores_zero(tmp_path, ci_env):
    code, out = export(tmp_path, raw_doc([gremlin("g1", "survived", 5),
                                          gremlin("g2", "survived", 12)]))
    assert code == 0  # scores never fail jobs; survivors are reported, not punished
    s = summary_of(out)
    assert s["score"] == 0.0 and s["survived_untriaged"] == 2
    assert s["complete"] is True and s["tool_error"] is False


# -- status semantics ----------------------------------------------------------------------

def test_timeout_scores_like_a_kill_and_error_and_pardon_never(tmp_path, ci_env):
    c1, o1 = export(tmp_path, raw_doc([gremlin("t", "timeout", 5),
                                       gremlin("s", "survived", 12)]), name="a")
    s1 = summary_of(o1)
    assert (c1, s1["score"], s1["timeout"]) == (0, 0.5, 1)  # (zapped+timeout)/checked
    c2, o2 = export(tmp_path, raw_doc([gremlin("e", "error", 5, error_output="boom"),
                                       gremlin("s", "survived", 12),
                                       gremlin("p", "pardoned", 1)]), name="b")
    s2 = summary_of(o2)
    assert s2["suspicious"] == 1 and s2["excluded"] == 1
    assert s2["checked"] == 2  # the error is checked; the pardon leaves the denominator
    assert s2["score"] == 0.0  # an error is never a kill
    assert c2 == 1 and s2["tool_error"] is True and "ERROR_RESULTS" in s2["failure_reasons"]
    row = {r["mutant_name"]: r for r in gr.read_jsonl(o2 / "results.jsonl")}["e"]
    assert (row["status"], row["backend_status"], row["error_output"]) == \
        ("suspicious", "error", "boom")


# -- refusal of bad raw reports -------------------------------------------------------------

def test_missing_empty_or_malformed_raw_is_a_tool_failure_never_a_measurement(tmp_path, ci_env):
    bad_status = raw_doc([gremlin("x", "ghost", 5)])
    bad_path = raw_doc([gremlin("x", "zapped", 5, file_path="/etc/passwd")])
    cases = {"missing": {"no_raw": True}, "empty": {"raw_text": "  "},
             "garbage": {"raw_text": "{not json"}, "bad-status": {"doc": bad_status},
             "bad-path": {"doc": bad_path}}
    for name, kwargs in cases.items():
        code, out = export(tmp_path, name=name, **kwargs)
        assert code == 1, name
        s = summary_of(out)
        assert s["complete"] is False and s["tool_error"] is True, name
        assert s["total"] is None and s["score"] is None, name  # no 100%, no fabricated zero
        assert "RAW_INVALID" in s["failure_reasons"] and s["problems"], name
        assert gr.read_jsonl(out / "results.jsonl") == [], name
        assert s["backend"] == "pytest-gremlins", name  # backend stated even on failure
        assert (out / "gremlins.json").exists() is (name != "missing"), name


def test_inconsistent_counts_stay_auditable_but_withhold_the_score(tmp_path, ci_env):
    doc = raw_doc([gremlin("g1", "zapped", 5), gremlin("g2", "survived", 12)])
    doc["summary"]["zapped"] = 99  # the raw summary now lies about its own results
    code, out = export(tmp_path, doc)
    assert code == 1
    s = summary_of(out)
    assert len(gr.read_jsonl(out / "results.jsonl")) == 2  # rows are still published
    assert s["total"] == 2 and s["killed"] == 1  # counts recomputed from the results
    assert s["score"] is None and s["complete"] is False
    assert "RAW_INCONSISTENT" in s["failure_reasons"] and s["tool_error"] is True


def test_stale_raw_is_refused(tmp_path, ci_env):
    code, out = export(tmp_path, raw_doc([gremlin("g1", "zapped", 5)]),
                       elapsed=30, mtime_age=3 * 24 * 3600)
    assert code == 1
    s = summary_of(out)
    assert s["total"] is None and s["score"] is None and s["tool_error"] is True
    assert any("STALE" in p for p in s["problems"])


def test_nonzero_run_exit_writes_an_honest_partial_artifact(tmp_path, ci_env):
    code, out = export(tmp_path, raw_doc([gremlin("g1", "zapped", 5),
                                          gremlin("g2", "timeout", 12)]), rc=-1)
    assert code == 1
    s = summary_of(out)
    assert s["total"] == 2 and s["score"] == 1.0  # honest score of what actually ran...
    assert s["complete"] is False and s["tool_error"] is True  # ...flagged partial
    assert s["run_exit_code"] == -1 and "TIMEOUT_KILL" in s["failure_reasons"]
    md = (out / "summary.md").read_text()
    assert "INCOMPLETE" in md and "-1" in md


# -- changed-base provenance (finding 1) ----------------------------------------------------

def test_export_accepts_the_workflow_style_changed_base_on_push(tmp_path, ci_env):
    """The gremlins push job runs ``export MODULE --raw ... --out ... --mode
    incremental --changed-base "$BEFORE" --run-exit-code ... --elapsed ...``
    (mutation.yml's export step, ``${BEFORE:+--changed-base "$BEFORE"}``). The
    adapter must accept the argument and record it verbatim -- the parser used
    to reject it -- and never use it to re-attribute a survivor or change a
    score: the gremlins raw carries no per-function diff, so it is provenance."""
    base = "0123456789abcdef0123456789abcdef01234567"
    code, out = export(tmp_path, raw_doc([gremlin("g1", "zapped", 5),
                                          gremlin("g2", "survived", 12)]),
                       mode="incremental", changed_base=base)
    assert code == 0  # the exact push invocation is not an argparse error
    s = summary_of(out)
    assert s["changed_base"] == base and s["mode"] == "incremental"
    assert s["complete"] is True and s["score"] == 0.5  # provenance never touches the score
    assert [r["status"] for r in gr.read_jsonl(out / "results.jsonl")] == ["killed", "survived"]


@pytest.mark.parametrize("base", [None, "", "0" * 40])
def test_export_records_no_changed_base_when_absent_empty_or_first_push(tmp_path, ci_env, base):
    """An omitted flag, an empty value, and a branch's all-zeros first-push base
    all mean "no base to compare to" -- recorded as None, never a fabricated sha."""
    code, out = export(tmp_path, raw_doc([gremlin("g1", "zapped", 5)]), changed_base=base)
    assert code == 0
    assert summary_of(out)["changed_base"] is None


# -- source function lookup ------------------------------------------------------------------

@pytest.mark.parametrize("line,expected", [(5, "add"), (12, "Box.size"), (13, "Box.size"),
                                           (18, "Box.wrap.inner"), (1, gr.MODULE_LEVEL),
                                           (9, gr.MODULE_LEVEL), (8, gr.MODULE_LEVEL)])
def test_enclosing_function_from_source(tmp_path, line, expected):
    (tmp_path / "engine" / "v2").mkdir(parents=True)
    (tmp_path / "engine" / "v2" / "toy.py").write_text(SOURCE)
    assert gr.SourceIndex(tmp_path).function_at(REL, line) == expected


def test_decorated_functions_resolve_to_the_function(tmp_path):
    (tmp_path / "dec.py").write_text("import functools\n\n\n@functools.lru_cache\n"
                                     "def cached(x):\n    return x\n")
    assert gr.SourceIndex(tmp_path).function_at("dec.py", 4) == "cached"


def test_function_lookup_is_honest_when_the_source_is_gone(tmp_path):
    assert gr.SourceIndex(tmp_path).function_at("engine/gone.py", 5) is None
    (tmp_path / "broken.py").write_text("def (:\n")
    assert gr.SourceIndex(tmp_path).function_at("broken.py", 1) is None


def test_export_reports_function_null_when_source_is_missing(tmp_path, ci_env):
    doc = raw_doc([gremlin("g1", "zapped", 5, file_path="engine/v2/deleted.py")])
    _, out = export(tmp_path, doc)
    assert gr.read_jsonl(out / "results.jsonl")[0]["function"] is None


# -- policy fingerprint ------------------------------------------------------------------------

def test_policy_identity_tracks_scoring_not_which_operators_fired():
    assert gr.policy_identity(gr.build_policy(["a"])) == \
        gr.policy_identity(gr.build_policy(["a", "b"]))  # observed set is data
    assert gr.policy_fingerprint(gr.build_policy(["a"])) != \
        gr.policy_fingerprint(gr.build_policy(["a", "b"]))
    changed = gr.build_policy(["a"])
    changed["timeout_seconds"] = 60
    assert gr.policy_identity(changed) != gr.policy_identity(gr.build_policy(["a"]))


# -- merge ---------------------------------------------------------------------------------------

def test_merge_combines_module_artifacts(tmp_path, ci_env):
    c1, o1 = export(tmp_path, raw_doc([gremlin("a1", "zapped", 5),
                                       gremlin("a2", "pardoned", 1)]), name="alpha")
    c2, o2 = export(tmp_path, raw_doc([gremlin("b1", "survived", 12),
                                       gremlin("b2", "timeout", 5)]), name="beta")
    assert (c1, c2) == (0, 0)
    merged = tmp_path / "merged"
    assert gr.main(["merge", "--out", str(merged), str(o1), str(o2)]) == 0
    m = summary_of(merged)
    assert (m["total"], m["killed"], m["timeout"], m["excluded"]) == (4, 1, 1, 1)
    assert m["checked"] == 3 and m["score"] == round(2 / 3, 4)
    assert set(m["modules"]) == {"alpha", "beta"}
    assert m["schema_version"] == 2 and m["backend"] == "pytest-gremlins"
    assert m["policy"]["operators"] == ["flip-add"]  # union of what the modules observed
    assert len(gr.read_jsonl(merged / "results.jsonl")) == 4
    assert "| **all** | 4 |" in (merged / "summary.md").read_text()


def test_merge_rejects_mixed_backends_policies_and_versions(tmp_path, ci_env):
    _, o1 = export(tmp_path, raw_doc([gremlin("a1", "zapped", 5)]), name="alpha")
    _, o2 = export(tmp_path, raw_doc([gremlin("b1", "zapped", 5)]), name="beta")
    legacy = tmp_path / "legacy"  # a schema-1 mutmut artifact from mutation_results.py
    legacy.mkdir()
    rows = [legacy_row("oldmut", "killed", "m1")]
    mr.write_jsonl(legacy / "results.jsonl", rows)
    info = {"run_id": "7", "sha": "abc", "ref": "r", "trigger": "push", "mode": "full"}
    (legacy / "summary.json").write_text(json.dumps(mr.summarize(rows, "oldmut", info, ["a.py"])))
    assert gr.main(["merge", "--out", str(tmp_path / "m1"), str(o1), str(legacy)]) == 2
    assert not (tmp_path / "m1").exists()  # refused wholesale, never a partial merge

    twin = tmp_path / "twin"
    shutil.copytree(o2, twin)

    def tamper(d, fn):
        s = summary_of(d)
        fn(s)
        (d / "summary.json").write_text(json.dumps(s))

    tamper(twin, lambda s: s.__setitem__("backend_version", "1.10.0"))
    assert gr.main(["merge", "--out", str(tmp_path / "m2"), str(o1), str(twin)]) == 2
    tamper(twin, lambda s: s.__setitem__("backend_version", "1.9.0"))
    tamper(twin, lambda s: s["policy"].__setitem__("timeout_seconds", 60))
    assert gr.main(["merge", "--out", str(tmp_path / "m3"), str(o1), str(twin)]) == 2
    tamper(twin, lambda s: s["policy"].__setitem__("timeout_seconds", 30))
    assert gr.main(["merge", "--out", str(tmp_path / "m4"), str(o1), str(twin)]) == 0  # sane again

    srow = tmp_path / "srow"
    shutil.copytree(o1, srow)
    tamper(srow, lambda s: s.__setitem__("module", "srow"))
    mr.write_jsonl(srow / "results.jsonl", [legacy_row("srow", "killed", "old1")])
    assert gr.main(["merge", "--out", str(tmp_path / "m5"), str(o1), str(srow)]) == 2  # schema mix
    assert gr.main(["merge", "--out", str(tmp_path / "m6"), str(o1), str(o1)]) == 2  # dup module
    assert gr.main(["merge", "--out", str(tmp_path / "m7"), str(tmp_path / "nodir")]) == 2


def test_merge_does_not_resurrect_a_withheld_score(tmp_path, ci_env):
    _, o1 = export(tmp_path, raw_doc([gremlin("a1", "zapped", 5)]), name="alpha")
    bad = raw_doc([gremlin("b1", "zapped", 5), gremlin("b2", "survived", 12)])
    bad["summary"]["zapped"] = 7  # beta's raw lies; its export withholds the score
    _, o2 = export(tmp_path, bad, name="beta")
    merged = tmp_path / "merged"
    assert gr.main(["merge", "--out", str(merged), str(o1), str(o2)]) == 1
    m = summary_of(merged)
    assert m["total"] == 3 and m["killed"] == 2  # counts stay auditable...
    assert m["score"] is None  # ...but the withheld score is not recomputed
    assert m["complete"] is False and m["tool_error"] is True
    assert any("RAW_INCONSISTENT" in r for r in m["failure_reasons"])


# -- merge run-provenance consistency (finding 3) ---------------------------------------------

def test_merge_full_plus_incremental_is_never_a_valid_full_measurement(tmp_path, ci_env):
    """Reproduced: a full and an incremental artifact merged and reported as one
    complete full run. They are not one measurement -- so the merged report is
    flagged incomplete/tool-error, its score withheld, and merge exits nonzero;
    the counts stay auditable but nothing reads as a valid full measurement."""
    _, full = export(tmp_path, raw_doc([gremlin("a1", "zapped", 5),
                                        gremlin("a2", "zapped", 12)]), name="alpha", mode="full")
    _, inc = export(tmp_path, raw_doc([gremlin("b1", "zapped", 5)]), name="beta", mode="incremental")
    merged = tmp_path / "merged"
    assert gr.main(["merge", "--out", str(merged), str(full), str(inc)]) == 1  # nonzero exit
    m = summary_of(merged)
    assert m["complete"] is False and m["tool_error"] is True
    assert m["score"] is None  # not a valid full measurement, even though mode reads "full"
    assert m["total"] == 3 and m["killed"] == 3  # counts stay auditable
    assert any(r.startswith("RUN_PROVENANCE_MISMATCH") and "modes" in r for r in m["failure_reasons"])
    assert "not a single measurement" in (merged / "summary.md").read_text().lower()


@pytest.mark.parametrize("field,new_value,label", [
    ("mode", "incremental", "modes"),
    ("sha", "cafebabe", "source SHAs"),
    ("run_id", "99", "run ids"),
])
def test_merge_requires_all_inputs_to_agree_on_run_provenance(tmp_path, ci_env, field, new_value,
                                                              label):
    """Every module input must agree on mode, source SHA and run identity; a
    single disagreeing field makes the merged artifact an incomplete tool-error
    report (with its score withheld), never a valid measurement."""
    _, a = export(tmp_path, raw_doc([gremlin("a1", "zapped", 5)]), name="alpha")
    _, b = export(tmp_path, raw_doc([gremlin("b1", "zapped", 5)]), name="beta")
    s = summary_of(b)
    s[field] = new_value  # one artifact from a different run / commit / mode
    (b / "summary.json").write_text(json.dumps(s))
    merged = tmp_path / "merged"
    assert gr.main(["merge", "--out", str(merged), str(a), str(b)]) == 1
    m = summary_of(merged)
    assert m["complete"] is False and m["tool_error"] is True and m["score"] is None
    assert m["total"] == 2  # rows still merged and auditable
    reason = next(r for r in m["failure_reasons"] if r.startswith("RUN_PROVENANCE_MISMATCH"))
    assert label in reason


def test_merge_all_agreeing_inputs_still_produce_a_valid_full_measurement(tmp_path, ci_env):
    """The other side of finding 3: when mode, sha and run id all agree, merge is
    a clean full measurement (exit 0, complete, scored) -- the provenance check
    withholds only genuine mismatches."""
    _, a = export(tmp_path, raw_doc([gremlin("a1", "zapped", 5)]), name="alpha")
    _, b = export(tmp_path, raw_doc([gremlin("b1", "zapped", 5)]), name="beta")
    merged = tmp_path / "merged"
    assert gr.main(["merge", "--out", str(merged), str(a), str(b)]) == 0
    m = summary_of(merged)
    assert m["complete"] is True and m["tool_error"] is False and m["score"] == 1.0


# -- merge: the plan's expected-module contract -------------------------------------------------

CANDIDATE = ROOT / "tools" / "gremlin_workflow_candidate.yml"


def alpha_beta(tmp_path, *, beta_mode="full"):
    """Two exported modules, in exactly the shape the matrix jobs upload."""
    _, a = export(tmp_path, raw_doc([gremlin("a1", "zapped", 5),
                                     gremlin("a2", "survived", 12)]), name="alpha")
    _, b = export(tmp_path, raw_doc([gremlin("b1", "timeout", 5)]), name="beta", mode=beta_mode)
    return a, b


def merge(tmp_path, dirs, *, expected=None, name="merged"):
    """The report job's merge call: ``--out DIR`` (plus the plan's module list,
    as the string the plan job publishes) over the dirs that arrived -- none."""
    out = tmp_path / name
    argv = ["merge", "--out", str(out)]
    if expected is not None:
        argv += ["--expected-modules", expected]
    return gr.main(argv + [str(d) for d in dirs]), out


def contract(out: Path) -> dict:
    return summary_of(out)["module_contract"]


def test_merge_with_the_exact_expected_module_set_stays_a_valid_full_run(tmp_path, ci_env):
    a, b = alpha_beta(tmp_path)
    code, out = merge(tmp_path, [a, b], expected=json.dumps(["alpha", "beta"]))
    assert code == 0
    m = summary_of(out)
    assert m["complete"] is True and m["tool_error"] is False and m["failure_reasons"] == []
    assert (m["total"], m["checked"], m["score"]) == (3, 3, round(2 / 3, 4))
    assert set(m["modules"]) == {"alpha", "beta"}
    assert m["expected_modules"] == ["alpha", "beta"]
    assert contract(out) == {"expected": ["alpha", "beta"], "present": ["alpha", "beta"],
                             "missing": [], "unexpected": [], "duplicate": {},
                             "complete_set": True}
    md = (out / "summary.md").read_text()
    assert "Expected modules (2): alpha, beta" in md and "Reported modules (2)" in md
    # a set, not a sequence: the plan's order does not matter
    assert merge(tmp_path, [b, a], expected=json.dumps(["beta", "alpha"]))[0] == 0
    # and without a contract the historical behaviour is untouched
    assert merge(tmp_path, [a, b], name="legacy")[0] == 0
    assert summary_of(tmp_path / "legacy")["module_contract"] is None
    assert summary_of(tmp_path / "legacy")["expected_modules"] is None


def test_merge_missing_expected_module_is_never_a_valid_full_run(tmp_path, ci_env):
    """The blocker: one matrix job died before its export/upload, so only a
    subset arrived. The merged artifact used to claim a complete clean run; it
    must instead say which module is absent and fail."""
    a, b = alpha_beta(tmp_path)  # beta's job uploaded nothing
    code, out = merge(tmp_path, [a], expected=json.dumps(["alpha", "beta"]))
    assert code == 1  # the report job fails on this rc
    m = summary_of(out)
    assert m["complete"] is False and m["tool_error"] is True
    assert m["score"] is None  # a subset of the plan is not the plan's score
    assert (m["total"], m["checked"], m["killed"]) == (2, 2, 1)  # what arrived, auditable
    assert len(gr.read_jsonl(out / "results.jsonl")) == 2
    assert set(m["modules"]) == {"alpha"}
    assert contract(out) == {"expected": ["alpha", "beta"], "present": ["alpha"],
                             "missing": ["beta"], "unexpected": [], "duplicate": {},
                             "complete_set": False}
    assert any(r.startswith("MISSING_MODULES") and "beta" in r for r in m["failure_reasons"])
    md = (out / "summary.md").read_text()
    assert "MISSING_MODULES" in md and "not a valid full run" in md
    # the honest partial rows are still there for triage
    assert [r["mutant_name"] for r in gr.read_jsonl(out / "results.jsonl")] == ["a1", "a2"]
    # ...and no consumer can read it as the latest completed measurement
    _, good = merge(tmp_path, [a, b], expected=json.dumps(["alpha", "beta"]), name="good")
    triage = tmp_path / "none.toml"
    measured = ratchet.build_measurement(*ratchet.read_report_dir(out), triage_path=triage)
    baseline = ratchet.build_measurement(*ratchet.read_report_dir(good), triage_path=triage)
    assert ratchet.compare(measured, baseline)[0]["code"] == "MUTATION_MEASUREMENT_TOOL_ERROR"


def test_merge_reports_a_module_the_plan_never_scheduled(tmp_path, ci_env):
    a, b = alpha_beta(tmp_path)  # an artifact from another plan/week slipped in
    code, out = merge(tmp_path, [a, b], expected=json.dumps(["alpha", "gamma"]))
    assert code == 1
    m = summary_of(out)
    assert m["complete"] is False and m["tool_error"] is True and m["score"] is None
    assert contract(out) == {"expected": ["alpha", "gamma"], "present": ["alpha", "beta"],
                             "missing": ["gamma"], "unexpected": ["beta"], "duplicate": {},
                             "complete_set": False}
    reasons = " ".join(m["failure_reasons"])
    assert "MISSING_MODULES" in reasons and "gamma" in reasons
    assert "UNEXPECTED_MODULES" in reasons and "beta" in reasons
    # an empty contract means "nothing was scheduled": every report is unexpected
    code, out = merge(tmp_path, [a], expected=json.dumps([]), name="empty-contract")
    assert code == 1
    assert "UNEXPECTED_MODULES" in " ".join(summary_of(out)["failure_reasons"])


def test_merge_duplicate_module_report_against_the_contract_is_a_diagnostic(tmp_path, ci_env):
    """Two directories claiming one module is not a module set: neither is
    silently preferred, no aggregate is stated, and every row stays published."""
    a, b = alpha_beta(tmp_path)
    twin = tmp_path / "beta-again"
    shutil.copytree(b, twin)
    code, out = merge(tmp_path, [a, b, twin], expected=json.dumps(["alpha", "beta"]))
    assert code == 1
    m = summary_of(out)
    assert m["complete"] is False and m["tool_error"] is True
    assert m["total"] is None and m["checked"] is None and m["score"] is None
    assert len(gr.read_jsonl(out / "results.jsonl")) == 4  # honest rows, all of them
    assert set(m["modules"]) == {"alpha"}  # beta double-reported: resolved to neither dir
    assert contract(out)["duplicate"] == {"beta": sorted([str(b), str(twin)])}
    assert any(r.startswith("DUPLICATE_MODULES") and "beta" in r for r in m["failure_reasons"])
    assert "DUPLICATE_MODULES" in (out / "summary.md").read_text()
    # without the contract the same input is still the old wholesale refusal
    assert gr.main(["merge", "--out", str(tmp_path / "refused"), str(a), str(b),
                    str(twin)]) == 2
    assert not (tmp_path / "refused").exists()


def test_merge_with_no_module_directories_still_writes_an_incomplete_diagnostic(tmp_path, ci_env):
    """Every matrix job died before upload: zero directories is not "nothing to
    consolidate, pass" -- it is a full run with nothing measured."""
    code, out = merge(tmp_path, [], expected=json.dumps(["alpha", "beta"]))
    assert code == 1
    for name in ("summary.json", "summary.md", "results.jsonl"):
        assert (out / name).is_file(), name  # a diagnostic artifact exists for triage
    m = summary_of(out)
    assert m["complete"] is False and m["tool_error"] is True
    assert m["total"] is None and m["checked"] is None and m["score"] is None  # never a 0% run
    assert m["modules"] == {} and m["run_id"] is None and m["mode"] is None
    assert gr.read_jsonl(out / "results.jsonl") == []
    assert contract(out)["missing"] == ["alpha", "beta"]
    assert contract(out)["present"] == [] and contract(out)["complete_set"] is False
    reasons = m["failure_reasons"]
    assert any(r.startswith("NO_MODULE_REPORTS") for r in reasons)
    assert any(r.startswith("MISSING_MODULES") for r in reasons)
    md = (out / "summary.md").read_text()
    assert "NO_MODULE_REPORTS" in md and "MISSING_MODULES" in md and "| **all** |" in md
    # even with no contract at all, an empty input set is not a pass
    assert merge(tmp_path, [], name="no-contract")[0] == 1
    assert summary_of(tmp_path / "no-contract")["tool_error"] is True


@pytest.mark.parametrize("raw", ["", "   ", "alpha,beta", '{"m": ["alpha"]}', "[1]", '"alpha"',
                                 '["alpha", "alpha"]', '["alpha", ""]'],
                         ids=["empty", "blank", "csv", "object", "ints", "string", "dup",
                              "empty-name"])
def test_a_malformed_expected_module_list_is_refused_not_treated_as_no_contract(tmp_path,
                                                                                ci_env, raw):
    """A broken contract value is a broken plan, not "expect nothing": accepting
    it would make every arriving report unexpected (or an empty list make a
    subset look complete). Refuse, write nothing, exit nonzero."""
    a, _ = alpha_beta(tmp_path)
    code, out = merge(tmp_path, [a], expected=raw)
    assert code == 2 and not out.exists()


def test_the_contract_never_overrides_provenance_or_identity_refusals(tmp_path, ci_env):
    """Regression guard: --expected-modules ADDS a gate, it does not soften the
    existing ones."""
    a, b = alpha_beta(tmp_path, beta_mode="incremental")
    code, out = merge(tmp_path, [a, b], expected=json.dumps(["alpha", "beta"]))
    assert code == 1
    m = summary_of(out)
    assert m["complete"] is False and m["tool_error"] is True and m["score"] is None
    assert any(r.startswith("RUN_PROVENANCE_MISMATCH") and "modes" in r
               for r in m["failure_reasons"])
    assert contract(out)["complete_set"] is True  # the set is right; the provenance is not

    tampered = tmp_path / "tampered"
    shutil.copytree(b, tampered)
    s = summary_of(tampered)
    s["policy"]["timeout_seconds"] = 60  # a different scoring policy
    (tampered / "summary.json").write_text(json.dumps(s))
    assert gr.main(["merge", "--out", str(tmp_path / "refused"), str(a), str(tampered),
                    "--expected-modules", json.dumps(["alpha", "beta"])]) == 2
    assert not (tmp_path / "refused").exists()

    # a module that came back partial is a tool error even when the set matches
    _, c = export(tmp_path, raw_doc([gremlin("c1", "zapped", 5)]), name="gamma", rc=-1)
    code, out = merge(tmp_path, [c], expected=json.dumps(["gamma"]), name="partial")
    assert code == 1
    m = summary_of(out)
    assert m["tool_error"] is True and contract(out)["complete_set"] is True
    assert any(r.startswith("gamma:") and "TIMEOUT_KILL" in r for r in m["failure_reasons"])


# -- plan job -> report job wiring ---------------------------------------------------------------

def test_candidate_workflow_hands_the_plan_module_list_to_merge():
    wf = yaml.safe_load(CANDIDATE.read_text())
    plan = next(s for s in wf["jobs"]["plan"]["steps"] if s.get("id") == "plan")["run"]
    assert 'modules=$(python3 tools/gremlin_pilot.py matrix --only "$ONLY")' in plan
    assert 'echo "modules=$modules"' in plan
    assert wf["jobs"]["plan"]["outputs"]["modules"] == "${{ steps.plan.outputs.modules }}"
    report = wf["jobs"]["report"]
    assert report["env"]["EXPECTED_MODULES"] == "${{ needs.plan.outputs.modules }}"
    merge_run = next(s for s in report["steps"]
                     if s.get("name") == "Merge module reports")["run"]
    assert '--expected-modules "$EXPECTED_MODULES"' in merge_run   # quoted, never interpolated
    assert '--expected-modules "${{' not in merge_run              # not injected into the script
    assert '"${dirs[@]}"' in merge_run
    assert "exit 0" not in merge_run                               # no early pass on an empty set
    assert 'echo 0 > "$RUNNER_TEMP/gremlin-merge-rc"' not in merge_run
    gate = next(s for s in report["steps"]
                if str(s.get("name", "")).startswith("Fail only on a merge error"))["run"]
    assert 'rc" != 0' in gate
    upload = next(s for s in report["steps"]
                  if str(s.get("uses", "")).startswith("actions/upload-artifact"))
    assert upload["if"] == "always()" and upload["with"]["name"] == "mutation-report"


def test_the_plan_matrix_output_is_exactly_the_string_merge_consumes(tmp_path, ci_env, monkeypatch,
                                                                    capsys):
    """Producer and consumer of the contract, round-tripped: the plan job
    publishes ``gremlin_pilot.py matrix``'s stdout and the report job feeds that
    same string to ``merge --expected-modules``. A matrix job that never
    uploaded must surface as MISSING_MODULES, not as a smaller valid run."""
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["chooser", "serving"])
    assert gp.cmd_matrix({}, types.SimpleNamespace(only="")) == 0
    planned = capsys.readouterr().out.strip()
    assert gr.parse_expected_modules(planned) == ["chooser", "serving"]
    _, c = export(tmp_path, raw_doc([gremlin("c1", "zapped", 5)]), name="chooser")
    _, s = export(tmp_path, raw_doc([gremlin("s1", "survived", 12)]), name="serving")
    code, out = merge(tmp_path, [c, s], expected=planned)
    assert code == 0 and summary_of(out)["complete"] is True
    code, out = merge(tmp_path, [c], expected=planned, name="partial-plan")
    assert code == 1
    m = summary_of(out)
    assert m["complete"] is False and m["tool_error"] is True and m["score"] is None
    assert m["module_contract"]["missing"] == ["serving"]


# -- mutation_report.py over both schemas ---------------------------------------------------------

def test_query_reads_historical_schema1_rows(tmp_path, capsys):
    rows = [legacy_row("old", "killed", "m1"), legacy_row("old", "survived", "m2", triage=None)]
    mr.write_jsonl(tmp_path / "results.jsonl", rows)
    assert rep.main(["--dir", str(tmp_path), "--format", "jsonl"]) == 0
    assert [json.loads(x) for x in capsys.readouterr().out.splitlines()] == rows
    assert rep.main(["--dir", str(tmp_path), "--untriaged"]) == 0
    assert "m2" in capsys.readouterr().out and "#3" not in capsys.readouterr().out


def test_query_filters_and_formats_gremlins_rows(tmp_path, ci_env):
    _, out = export(tmp_path, raw_doc([gremlin("g1", "zapped", 5),
                                       gremlin("g2", "survived", 12, operator="swap-comparison"),
                                       gremlin("g3", "timeout", 5),
                                       gremlin("g4", "error", 5, error_output="x"),
                                       gremlin("g5", "pardoned", 1)]), name="toy")
    rows = gr.read_jsonl(out / "results.jsonl")
    assert [r["mutant_name"] for r in rep.filter_rows(rows, statuses=["killed"])] == ["g1"]
    assert [r["mutant_name"] for r in rep.filter_rows(rows, statuses=["excluded"])] == ["g5"]
    assert [r["mutant_name"] for r in rep.filter_rows(rows, statuses=["suspicious",
                                                                      "timeout"])] == ["g3", "g4"]
    assert [r["mutant_name"] for r in rep.filter_rows(rows, untriaged=True)] == ["g2"]
    assert [r["mutant_name"] for r in rep.filter_rows(rows, functions=["Box.*"])] == ["g2"]
    with pytest.raises(ValueError):
        rep.filter_rows(rows, statuses=["dead"])
    buf = io.StringIO()
    rep.emit(rows, "csv", show_diff=False, history_rows=False, out=buf)
    header = buf.getvalue().splitlines()[0].split(",")
    assert "operator" in header and "gremlin_id" in header and "backend_status" in header
    buf = io.StringIO()
    rep.emit(rows[:2], "table", show_diff=True, history_rows=False, out=buf)
    text = buf.getvalue()
    assert "flip-add [zapped]" in text and "-- 2 mutants" in text


def gremlins_block(score, *, operators=("flip-add",), timeout=30):
    """A realistic merged pytest-gremlins ALL block: it states its identity, the
    way tools/gremlin_results.py merge writes summary.json (backend, pinned
    version and the scoring policy)."""
    return {"sha": "s2", "mode": "full", "backend": "pytest-gremlins", "backend_version": "1.9.0",
            "policy": gr.build_policy(operators) | {"timeout_seconds": timeout},
            "total": 4, "killed": 2, "survived": 1, "timeout": 1, "suspicious": 0,
            "excluded": 0, "checked": 4, "survived_untriaged": 1, "score": score, "modules": {}}


def mutmut_block(score):
    """A schema-1 mutmut ALL block: it states no backend identity at all."""
    return {"sha": "s1", "mode": "full", "total": 10, "killed": 5, "survived": 5,
            "no_tests": 0, "skipped": 0, "survived_untriaged": 5, "score": score, "modules": {}}


def test_history_reads_gremlins_blocks_without_fabricating_columns():
    runs = [({"databaseId": 2, "createdAt": "2026-09-20T00:00:00Z"}, gremlins_block(0.75)),
            ({"databaseId": 1, "createdAt": "2026-09-13T00:00:00Z"}, mutmut_block(0.5))]
    rows = rep.merge_history(runs)
    m, g = rows
    assert (m["backend"], m["backend_version"], m["policy_id"]) == (None, None, None)
    assert g["backend"] == "pytest-gremlins" and g["backend_version"] == "1.9.0"
    assert g["policy_id"] == gr.policy_identity(gremlins_block(0.75)["policy"])
    assert g["no_tests"] is None and g["skipped"] is None  # mutmut columns, honestly absent
    buf = io.StringIO()
    rep.emit(rows, "table", show_diff=False, history_rows=True, out=buf)
    assert "--" in buf.getvalue()  # renders, does not crash on the None cells


def test_history_shows_no_numeric_delta_across_incompatible_identities():
    """Finding 5: a mutmut 50% followed by a gremlins 80% is not a +30pp win --
    the two measure different things, so the delta reads as None (``--``) even
    though the number rose, while the identity stays visible on every row."""
    runs = [({"databaseId": 2, "createdAt": "2026-09-20T00:00:00Z"}, gremlins_block(0.8)),
            ({"databaseId": 1, "createdAt": "2026-09-13T00:00:00Z"}, mutmut_block(0.5))]
    m, g = rep.merge_history(runs)
    assert (m["score"], m["delta"]) == (0.5, None)   # first run: nothing to compare
    assert (g["score"], g["delta"]) == (0.8, None)   # gremlins after mutmut: no +30pp
    assert g["backend"] == "pytest-gremlins"          # ...but the change is on display
    buf = io.StringIO()
    rep.emit([m, g], "table", show_diff=False, history_rows=True, out=buf)
    text = buf.getvalue()
    assert "+30" not in text and "pytest-gremlins" in text


def test_history_delta_survives_across_same_identity_but_not_a_policy_change():
    """Same backend/version/policy still trends (delta computed), including two
    gremlins runs that merely observed different operators; a changed scoring
    policy (timeout) is a different measurement and breaks the delta."""
    same = [({"databaseId": 1, "createdAt": "1"}, gremlins_block(0.5)),
            ({"databaseId": 2, "createdAt": "2"}, gremlins_block(0.8, operators=["flip-add", "const-removal"])),
            ({"databaseId": 3, "createdAt": "3"}, gremlins_block(0.9))]
    a, b, c = rep.merge_history(same)
    assert (a["delta"], b["delta"], c["delta"]) == (None, 0.3, 0.1)
    # a timeout change is a different policy: the next run shows no delta across it
    changed = same + [({"databaseId": 4, "createdAt": "4"}, gremlins_block(0.95, timeout=60))]
    d = rep.merge_history(changed)[-1]
    assert (d["policy_id"], d["delta"]) != (c["policy_id"], 0.05) and d["delta"] is None


# -- ratchet: backend/policy guards -----------------------------------------------------------------

def gmeas(**over):
    base = {"schema_version": ratchet.SCHEMA_VERSION, "mode": "full",
            "backend": "pytest-gremlins", "backend_version": "1.9.0", "policy_id": "p1",
            "complete": True, "tool_error": False,
            "modules": {"m": {"killed_effective": 8, "checked_effective": 10}}}
    base.update(over)
    return base


def test_ratchet_refuses_cross_backend_comparison():
    mutmut_measured = {"schema_version": ratchet.SCHEMA_VERSION, "mode": "full",
                       "modules": {"m": {"killed_effective": 8, "checked_effective": 10}}}
    assert ratchet.compare(gmeas(), mutmut_measured) == [  # old baseline: no backend = mutmut
        {"code": "MUTATION_BACKEND_MISMATCH", "measured_backend": "pytest-gremlins",
         "baseline_backend": "mutmut"}]
    assert ratchet.compare(mutmut_measured, gmeas())[0]["code"] == "MUTATION_BACKEND_MISMATCH"
    assert ratchet.compare(gmeas(backend_version="2.0.0"), gmeas())[0]["code"] == \
        "MUTATION_BACKEND_MISMATCH"
    assert ratchet.compare(gmeas(policy_id="p2"), gmeas())[0]["code"] == "MUTATION_POLICY_MISMATCH"
    assert ratchet.compare(gmeas(), gmeas()) == []


def test_ratchet_gremlins_measurement_cannot_omit_identity_against_a_full_baseline():
    """Finding 4: a gremlins measurement that drops backend_version or policy_id
    must not pass against a baseline that carries them. The old guard compared
    only when BOTH sides had a field, so an artifact missing its identity slipped
    through; for schema-2 gremlins an absent identity is now a hard mismatch."""
    baseline = gmeas()
    for drop, code in (("backend_version", "MUTATION_BACKEND_MISMATCH"),
                       ("policy_id", "MUTATION_POLICY_MISMATCH")):
        measured = gmeas()
        measured.pop(drop)  # a gremlins artifact that carried no identity at all
        assert ratchet.compare(measured, baseline)[0]["code"] == code
        measured[drop] = None  # present but null: the same refusal
        assert ratchet.compare(measured, baseline)[0]["code"] == code


def test_ratchet_mutmut_historical_identity_absence_stays_supported():
    """Finding 4's carve-out: schema-1 mutmut states no backend_version/policy_id
    and that absence must remain comparable (both mutmut sides), so historical
    mutmut measurements and baselines keep working -- only gremlins requires it."""
    side = {"schema_version": ratchet.SCHEMA_VERSION, "mode": "full",
            "modules": {"m": {"killed_effective": 8, "checked_effective": 10}}}
    assert ratchet.compare(dict(side), dict(side)) == []
    # a mutmut side naming one version is still comparable to a silent one
    # (mutmut fields are only compared when present on both sides).
    assert ratchet.compare({**side, "backend_version": "3.8"}, side) == []


def test_ratchet_refuses_incomplete_or_tool_errored_gremlins_measurements():
    assert ratchet.compare(gmeas(complete=False), gmeas())[0]["code"] == \
        "MUTATION_MEASUREMENT_INCOMPLETE"
    assert ratchet.compare(gmeas(tool_error=True), gmeas())[0]["code"] == \
        "MUTATION_MEASUREMENT_TOOL_ERROR"


def test_ratchet_module_counts_follow_gremlins_semantics():
    rows = [legacy_row("m", st, st, diff=None) for st in
            ("killed", "timeout", "suspicious", "survived", "excluded")]
    c = ratchet.module_counts(rows, {})
    assert c["m"] == {"total": 5, "checked_effective": 4, "killed_effective": 2,
                      "survived_untriaged": 2, "triaged": 0}  # pardoned out, errors never killed


def test_build_measurement_carries_the_gremlins_identity(tmp_path, ci_env):
    _, out = export(tmp_path, raw_doc([gremlin("g1", "zapped", 5),
                                       gremlin("g2", "survived", 12)]))
    rows, s = ratchet.read_report_dir(out)
    measured = ratchet.build_measurement(rows, s, triage_path=tmp_path / "none.toml")
    assert measured["backend"] == "pytest-gremlins" and measured["backend_version"] == "1.9.0"
    assert measured["policy_id"] == gr.policy_identity(s["policy"])
    assert measured["complete"] is True and measured["tool_error"] is False
    assert measured["modules"]["toy"] == {"total": 2, "checked_effective": 2,
                                          "killed_effective": 1, "survived_untriaged": 1,
                                          "triaged": 0}


def test_gremlins_measurements_default_to_their_own_baseline(tmp_path, ci_env, monkeypatch, capsys):
    _, out = export(tmp_path, raw_doc([gremlin("g1", "zapped", 5)]))
    capsys.readouterr()  # drop the export's stdout line; we want the ratchet's verdict alone
    monkeypatch.setattr(ratchet, "BASELINE_GREMLINS", tmp_path / "no_gremlins_baseline.json")
    assert ratchet.main(["--dir", str(out)]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["findings"] == [{"code": "MUTATION_BASELINE_MISSING"}]  # not compared to mutmut


def test_gremlins_ratchet_end_to_end_against_its_own_baseline(tmp_path, ci_env):
    _, out = export(tmp_path, raw_doc([gremlin("g1", "zapped", 5),
                                       gremlin("g2", "survived", 12)]))
    s = summary_of(out)
    baseline = tmp_path / "gremlins_baseline.json"
    baseline.write_text(json.dumps({
        "schema_version": ratchet.SCHEMA_VERSION, "mode": "full",
        "backend": "pytest-gremlins", "backend_version": "1.9.0",
        "policy_id": gr.policy_identity(s["policy"]),
        "modules": {"toy": {"killed_effective": 1, "checked_effective": 2}}}))
    assert ratchet.main(["--dir", str(out), "--baseline", str(baseline),
                         "--triage", str(tmp_path / "none.toml")]) == 0
    # a regression (8 of 10 -> 7 of 10 equivalent) is still caught with backend/policy matching
    baseline.write_text(json.dumps(json.loads(baseline.read_text()) | {
        "modules": {"toy": {"killed_effective": 8, "checked_effective": 10}}}))
    assert ratchet.main(["--dir", str(out), "--baseline", str(baseline),
                         "--triage", str(tmp_path / "none.toml")]) == 1
