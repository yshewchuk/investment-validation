#!/usr/bin/env python3
"""oc-check: the only shell command opencode (GLM) may run in the pilot.

Usage (from the agent worktree root):
  python3 tools/oc_check.py                 tests affected by the diff vs origin/main (changed tests + importers)
  python3 tools/oc_check.py tests/test_a.py [tests/test_b.py::test_x ...]   explicit targets

Test selection (see changed_tests): changed test files, test files that import a
changed engine/checks/tools module, and -- whenever any engine/checks/tools module
changed -- every tests/test_*.py carrying a line equal to ALWAYS_RUN_MARKER
("# land: always-run"), for tests that scan source as text and so never name the
module they check.

Runs, in order, against the WORKING TREE of the current agent worktree:
  1. the pre-commit gates (hygiene, import layers, code budgets, package READMEs, v2 lint)
  2. the named test files, serially, under bounded_run (1.5 GB cap, 1 GB box floor),
     CI marker expression, no xdist.
Output is trimmed so a long log cannot flood the model. Exit 0 only when all pass.

Every run writes .oc_logs/oc_check_report.json: verdict, per-step results, targets,
pytest tail and the git tree id of the working tree at the end of the run.
  python3 tools/oc_check.py --verify    VERIFIED only if the report exists, is ALL GREEN and
                                        matches the CURRENT tree (so a claim "I ran oc-check"
                                        is checkable, and any edit after the run shows as STALE).
"""
import json
import tempfile
import time
import os
import re
import subprocess
import sys
from pathlib import Path

MAIN = Path("/root/investing-plan")
WORKTREES = MAIN / ".claude" / "worktrees"
ARG_RE = re.compile(r"^tests/[A-Za-z0-9_/]+\.py(::[A-Za-z0-9_\[\]\-.]+)*$")
MARKERS = "not needs_data and not needs_corpus and not heavy_host and not browser"
TAIL = 60
ALWAYS_RUN_MARKER = "# land: always-run"  # test files carrying this line always run when any engine/checks/tools file changed


def refuse(msg):
    print(f"oc-check: REFUSED: {msg}")
    sys.exit(2)


def tail(text):
    lines = text.splitlines()
    return "\n".join(lines[-TAIL:])


STEPS = []
REPORT = Path(".oc_logs") / "oc_check_report.json"


def tree_id(root):
    """Git tree id of the working tree incl. untracked, non-ignored files (.oc_logs is ignored)."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git = ["git", "-C", str(root)]
        subprocess.run(git + ["read-tree", "HEAD"], env=env, check=True, capture_output=True)
        subprocess.run(git + ["add", "-A"], env=env, check=True, capture_output=True)
        return subprocess.run(git + ["write-tree"], env=env, check=True,
                              capture_output=True, text=True).stdout.strip()


def run(label, cmd, env, timeout):
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"== {label}: TIMEOUT after {timeout}s")
        STEPS.append({"step": label, "result": "TIMEOUT"})
        return False
    ok = p.returncode == 0
    print(f"== {label}: {'PASS' if ok else f'FAIL (rc={p.returncode})'}")
    out = tail(p.stdout + p.stderr) if (not ok or label == "pytest") else ""
    if out:
        print(out)
    STEPS.append({"step": label, "result": "PASS" if ok else f"FAIL rc={p.returncode}", "tail": out})
    return ok


def verify(root):
    if not REPORT.is_file():
        print("oc-check --verify: MISSING (oc-check never ran in this worktree)")
        sys.exit(3)
    rep = json.loads(REPORT.read_text())
    now = tree_id(root)
    if rep.get("tree") != now:
        print(f"oc-check --verify: STALE (report tree {rep.get('tree')}, current {now}; "
              f"files changed after the last run at {rep.get('finished')})")
        sys.exit(4)
    if rep.get("verdict") != "ALL GREEN":
        print(f"oc-check --verify: NOT GREEN at current tree ({rep.get('verdict')})")
        sys.exit(1)
    print(f"oc-check --verify: VERIFIED ALL GREEN at tree {now} "
          f"({len(rep.get('targets', []))} test targets, finished {rep.get('finished')})")
    sys.exit(0)


def changed_tests(root):
    """Test files affected by the diff against origin/main: changed test files, plus
    test files that import a changed engine/checks/tools module, plus (whenever any
    engine/checks/tools module changed) every tests/test_*.py carrying an
    ALWAYS_RUN_MARKER line -- for tests that scan source as text (e.g. every
    fail("CODE") literal in a package) and so never name the module they check."""
    git = ["git", "-C", str(root)]
    base = subprocess.run(git + ["merge-base", "HEAD", "origin/main"], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(git + ["diff", "--name-only", base or "HEAD"], capture_output=True, text=True).stdout.split()
    new = subprocess.run(git + ["ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout.split()
    changed = [f for f in dict.fromkeys(diff + new) if f.endswith(".py") and (root / f).is_file()]
    picked = [f for f in changed if f.startswith("tests/test_")]
    mods = [f[:-3].replace("/", ".") for f in changed if f.split("/")[0] in ("engine", "checks", "tools")]
    mods = [m[:-9] if m.endswith(".__init__") else m for m in mods]
    if mods:
        pat = re.compile(r"\b(" + "|".join(re.escape(m) for m in mods) + r")\b")
        for t in sorted((root / "tests").glob("test_*.py")):
            rel = f"tests/{t.name}"
            if rel not in picked and pat.search(t.read_text(errors="ignore")):
                picked.append(rel)
        for t in sorted((root / "tests").glob("test_*.py")):
            rel = f"tests/{t.name}"
            if rel in picked:
                continue
            lines = (t.read_text(errors="ignore")).splitlines()
            if any(line.strip() == ALWAYS_RUN_MARKER for line in lines):
                picked.append(rel)
    return picked


def main():
    root = Path.cwd().resolve()
    if WORKTREES not in root.parents:
        refuse(f"must run from an agent worktree under {WORKTREES}, not {root}")
    if not (root / "checks" / "repo_hygiene.py").is_file():
        refuse("run from the worktree root")
    args = sys.argv[1:]
    if args == ["--verify"]:
        verify(root)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not args:
        args = changed_tests(root)
        print(f"oc-check: tests affected by the diff vs origin/main: {len(args)}")
        if len(args) > 12:
            print("oc-check: more than 12 affected test files; running the first 12. Name targets explicitly to choose.")
            args = args[:12]
    elif len(args) > 12:
        refuse("at most 12 test targets")
    for a in args:
        if not ARG_RE.match(a) or ".." in a:
            refuse(f"bad test target {a!r}; use tests/<file>.py[::test]")
        if not (root / a.split("::")[0]).is_file():
            refuse(f"no such test file {a.split('::')[0]}")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENROUTER", "ANTHROPIC", "GITHUB", "GH_"))}
    env["INVESTING_PLAN_ROOT"] = str(root)
    py = sys.executable
    gates = [
        ("hygiene", [py, "checks/repo_hygiene.py", "--all", "--quiet"]),
        ("import_layers", [py, "checks/import_layers.py", "--all", "--quiet"]),
        ("code_budgets", [py, "checks/code_budgets.py", "--all", "--quiet"]),
        ("package_readmes", [py, "checks/package_readmes.py", "--all", "--quiet"]),
        ("v2_lint", [py, "checks/rearchitecture_phase1_lint.py", "--worktree", "--quiet"]),
    ]
    ok = all([run(label, cmd, env, 300) for label, cmd in gates])
    if args:
        cmd = [py, str(root / "tools" / "bounded_run.py"), "--max-rss-gb", "1.5", "--min-free-gb", "1.0",
               "--cores", "4", "--", py, "-m", "pytest", "-q", "-p", "no:xdist", "-p", "no:cacheprovider",
               "-m", MARKERS, "--tb=short", "-rfE", *args]
        ok = run("pytest", cmd, env, 900) and ok
    verdict = "ALL GREEN" if ok else "NOT GREEN"
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({
        "verdict": verdict, "tree": tree_id(root), "targets": args, "steps": STEPS,
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    print(f"oc-check: {verdict}  (report: {REPORT})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
