"""Execute the gremlins workflow wiring instead of matching substrings.

* The ``if:`` conditions and the ``DESC`` expression of the status jobs are EVALUATED by a
  tiny GitHub-expression evaluator (``==``, ``!=``, ``&&``, ``||``, parentheses, string
  literals, ``always()``, ``format()``, dotted contexts) for every input combination and
  compared with an independent oracle (the decision table in tests/README.md).
* The ``run:`` scripts of the classify step, ``report_status`` and ``report_neutral`` are RUN
  under bash with ``git`` / ``gh`` / ``python`` stubbed as functions that log their argv.

# packages: foundation, ops
"""
from __future__ import annotations

import itertools
import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text())["jobs"]

_TOKEN = re.compile(r"\s*(?:(&&|\|\||==|!=|\(|\)|,)|'((?:[^']|'')*)'|([A-Za-z_][\w.]*))")


def _unwrap(src: str) -> str:
    m = re.match(r"^\s*\$\{\{(.*)\}\}\s*$", src, re.DOTALL)
    return m.group(1) if m else src


def _tokens(src: str):
    src = src.strip()
    pos, out = 0, []
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        assert m, f"cannot tokenize {src[pos:]!r}"
        if m.group(1):
            out.append(("op", m.group(1)))
        elif m.group(2) is not None:
            out.append(("str", m.group(2).replace("''", "'")))
        else:
            out.append(("name", m.group(3)))
        pos = m.end()
    return out


def evaluate(src: str, ctx: dict):
    """Evaluate a GitHub Actions expression (the subset the workflow uses)."""
    toks = _tokens(_unwrap(src))
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else (None, None)

    def take():
        pos[0] += 1
        return toks[pos[0] - 1]

    def expect(op):
        assert take() == ("op", op)

    def atom():
        kind, val = take()
        if kind == "str":
            return val
        if kind == "op" and val == "(":
            v = or_()
            expect(")")
            return v
        assert kind == "name", (kind, val)
        if peek() == ("op", "("):
            take()
            args = []
            if peek() != ("op", ")"):
                args.append(or_())
                while peek() == ("op", ","):
                    take()
                    args.append(or_())
            expect(")")
            if val == "always":
                return True
            if val == "format":
                text = args[0]
                for i, a in enumerate(args[1:]):
                    text = text.replace("{%d}" % i, str(a))
                return text
            raise AssertionError(f"unsupported function {val}")
        return ctx[val]  # KeyError: the condition reads a context the test does not model

    def cmp_():
        left = atom()
        if peek() in (("op", "=="), ("op", "!=")):
            op = take()[1]
            right = atom()
            return (left == right) if op == "==" else (left != right)
        return left

    def and_():
        left = cmp_()
        while peek() == ("op", "&&"):
            take()
            right = cmp_()
            left = right if left else left
        return left

    def or_():
        left = and_()
        while peek() == ("op", "||"):
            take()
            right = and_()
            left = left if left else right
        return left

    value = or_()
    assert pos[0] == len(toks), "trailing tokens"
    return value


def test_evaluator_semantics():
    ctx = {"a": "x", "b": ""}
    assert evaluate("a == 'x' && b == ''", ctx) is True
    assert evaluate("a != 'x' || b != ''", ctx) is False
    assert evaluate("b || 'fallback'", ctx) == "fallback"
    assert evaluate("a && 'second'", ctx) == "second"
    assert evaluate("format('{0}-{1}', a, 'y')", ctx) == "x-y"
    assert evaluate("always() && (a == 'x' || b == 'z')", ctx) is True
    assert evaluate("${{ 'it''s' }}", ctx) == "it's"


def expected_posted(modules, skipped, classified, report):
    """Independent oracle: the decision table of tests/README.md (skipped '' means 0)."""
    skipped = skipped or "0"
    if modules == "[]":
        return "success"
    if skipped != "0":
        return "failure"
    if classified == "failure":
        return "failure"
    if classified == "neutral":
        return "neutral"
    if classified == "success" and report == "success":
        return "success"
    if report != "success":
        return "neutral"
    return "success"


def _ctx(pr, modules, state, skipped, result, desc="classified text"):
    return {
        "needs.plan.outputs.pr_number": pr,
        "needs.plan.outputs.modules": modules,
        "needs.report.outputs.state": state,
        "needs.report.outputs.description": desc,
        "needs.report.outputs.skipped_count": skipped,
        "needs.report.result": result,
        "needs.plan.result": "success",
    }


def test_report_neutral_if_matches_the_decision_table_for_every_input():
    cond = JOBS["report_neutral"]["if"]
    combos = itertools.product(["", "7"], ["[]", '["a"]', ""], ["", "success", "failure", "neutral"],
                               ["0", "", "2"], ["success", "failure", "cancelled", "skipped"])
    runs = 0
    for pr, modules, state, skipped, result in combos:
        runs_job = bool(evaluate(cond, _ctx(pr, modules, state, skipped, result)))
        want = pr != "" and expected_posted(modules, skipped, state, result) == "neutral"
        assert runs_job == want, (pr, modules, state, skipped, result)
        runs += runs_job
    assert runs > 0


def test_report_neutral_desc_expression_for_every_input():
    expr = JOBS["report_neutral"]["steps"][0]["env"]["DESC"]
    for state, desc, result in itertools.product(["", "success", "neutral"], ["", "grey text"],
                                                 ["failure", "cancelled", "skipped"]):
        got = evaluate(expr, _ctx("7", '["a"]', state, "0", result, desc))
        if state == "neutral" and desc:
            assert got == desc
        else:
            assert got == (f"gremlins: report job {result} before classification -- infra; "
                           "not a pass")


def test_classify_step_and_status_jobs_run_only_for_a_resolved_pr():
    for pr in ("", "7"):
        ctx = _ctx(pr, '["a"]', "", "0", "success")
        classify = next(s for s in JOBS["report"]["steps"] if s.get("id") == "classify")
        assert bool(evaluate(classify["if"], ctx)) == (pr != "")
        assert bool(evaluate(JOBS["report_status"]["if"], ctx)) == (pr != "")


GH_STUB = r'''
gh() {
  echo "gh $*" >> "$LOG"
  if [ "$1 $2" = "run list" ]; then
    case "$*" in
      *"--branch main"*"--event push"*) printf '%s\n' $FAKE_RUNS ;;
      *) printf '%s\n' 999 ;;
    esac
    return 0
  fi
  if [ "$1 $2" = "run download" ]; then
    mkdir -p baseline
    case " $ARTIFACT_IDS " in *" $3 "*) echo "$3" > baseline/summary.json; return 0 ;; esac
    return 1
  fi
  return 0
}
git() {
  echo "git $*" >> "$LOG"
  if [ -n "${GIT_FAIL:-}" ]; then return 1; fi
  if [ "$1" = "-c" ]; then echo "+changed line"; fi
  return 0
}
python() { echo "python $*" >> "$LOG"; }
'''


def _run(run, tmp_path, **env):
    log = tmp_path / "calls.log"
    log.write_text("")
    full = dict(os.environ, LOG=str(log), REPO="o/r", FAKE_RUNS="", ARTIFACT_IDS="")
    full.update(env)
    res = subprocess.run(["bash", "-c", GH_STUB + "\n" + run], capture_output=True, text=True,
                         env=full, cwd=tmp_path)
    return res, log.read_text().splitlines()


def _step(job, predicate):
    return next(s for s in JOBS[job]["steps"] if predicate(s))


def test_classify_step_runs_git_gh_and_the_tool_with_the_documented_arguments(tmp_path):
    run = _step("report", lambda s: s.get("id") == "classify")["run"]
    res, calls = _run(run, tmp_path, PR="7", BASE_SHA="base1", HEAD_SHA="head1",
                      FAKE_RUNS="30 20 10", ARTIFACT_IDS="20 10 999")
    assert res.returncode == 0, res.stderr
    assert calls == [
        "git fetch -q --no-tags origin +refs/pull/7/head:refs/remotes/pr/head",
        "git -c core.quotepath=off diff -U0 --no-color --no-ext-diff base1...head1",
        "gh run list --repo o/r --workflow mutation.yml --branch main --event push "
        "--status completed --limit 10 --json databaseId --jq .[].databaseId",
        "gh run download 30 --repo o/r -n mutation-report -D baseline",
        "gh run download 20 --repo o/r -n mutation-report -D baseline",
        "python tools/mutation_pr_status.py --merged-dir merged --diff-file pr.diff "
        "--baseline-summary baseline/summary.json",
    ]
    assert (tmp_path / "pr.diff").read_text() == "+changed line\n"


def test_classify_step_without_a_diff_or_baseline_warns_and_still_runs_the_tool(tmp_path):
    run = _step("report", lambda s: s.get("id") == "classify")["run"]
    res, calls = _run(run, tmp_path, PR="7", BASE_SHA="b", HEAD_SHA="h", GIT_FAIL="1",
                      FAKE_RUNS="30", ARTIFACT_IDS="")
    assert res.returncode == 0
    assert "::warning::could not compute the PR diff" in res.stdout
    assert ("::warning::no recent main push run has a mutation-report artifact; "
            "no failure is excused as pre-existing") in res.stdout
    assert calls[-1] == "python tools/mutation_pr_status.py --merged-dir merged"
    assert not any(c.startswith("git -c") for c in calls)  # diff skipped after the fetch failed


def test_report_status_script_posts_exactly_the_table_state(tmp_path):
    run = _step("report_status", lambda s: str(s.get("name", "")).startswith("Post mutation"))["run"]
    combos = itertools.product(["[]", '["a","b"]', ""], ["0", "2"], ["", "success", "failure", "neutral"],
                               ["success", "failure", "cancelled", "skipped"])
    for modules, skipped, classified, report in combos:
        res, calls = _run(run, tmp_path, SHA="sha1", RUN_URL="https://x/run/5", MODULES=modules,
                          SKIPPED_COUNT=skipped, CLASSIFIED_STATE=classified,
                          CLASSIFIED_DESC="classified text", REPORT_RESULT=report)
        assert res.returncode == 0, (modules, skipped, classified, report, res.stderr)
        want = expected_posted(modules, skipped, classified, report)
        key = (modules, skipped, classified, report)
        if want == "neutral":
            assert calls == [], key  # grey: no commit status at all
            assert "::notice::nothing measured" in res.stdout
        else:
            assert len(calls) == 1, key
            prefix = f"gh api repos/o/r/statuses/sha1 -f state={want} -f context=mutation/pr -f description="
            assert calls[0].startswith(prefix), (key, calls[0])
            assert calls[0].endswith(" -f target_url=https://x/run/5"), key


def test_report_neutral_script_posts_a_completed_neutral_check_run(tmp_path):
    step = JOBS["report_neutral"]["steps"][0]
    res, calls = _run(step["run"], tmp_path, SHA="sha1", RUN_URL="https://x/run/5", DESC="grey text")
    assert res.returncode == 0, res.stderr
    assert calls == [
        "gh api repos/o/r/check-runs -f name=mutation/pr -f head_sha=sha1 -f status=completed "
        "-f conclusion=neutral -f details_url=https://x/run/5 "
        "-f output[title]=Mutation testing: nothing measured -f output[summary]=grey text"]
    assert JOBS["report_neutral"]["permissions"] == {"checks": "write"}
