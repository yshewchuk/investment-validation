"""Mutation-testing pilot: run mutmut per critical module and report the result.

Config: ``tools/mutation_pilot.toml``. Guide: ``tests/README.md``,
"Mutation-testing pilot".

mutmut 3 always writes its state to ``./mutants/`` under the directory it runs
in, so running it from the repo would put a mutated copy of ``engine/`` (whose
``.py`` files the allowlist .gitignore lets back in) inside the checkout. This
tool instead runs every module in its own work copy OUTSIDE the repo:

    $MUTATION_PILOT_HOME/<module>/          (default ~/.cache/investing-plan-mutation-pilot)
        <every tracked file>                synced from the repo (a clean clone, minus .git)
        setup.cfg                           generated [mutmut] section for this module
        mutants/                            mutmut's own state and results

Nothing from ``data/`` is copied, so a test that silently needs real data fails
the clean run instead of reading it.

Commands::

    python3 tools/mutation_pilot.py list
    python3 tools/mutation_pilot.py matrix [--only a,b]            # CI matrix (JSON)
    python3 tools/mutation_pilot.py count [MODULE ...]          # mutant counts, no tests run
    python3 tools/mutation_pilot.py run MODULE [--max-children N] [--fresh] [GLOB ...]
    python3 tools/mutation_pilot.py report [MODULE ...] [--no-diffs]

``run`` is the heavy step: launch it under ``tools/bounded_run.py``. ``GLOB``
restricts the run to matching mutant names (mutmut's own syntax, e.g.
``"engine.models.no_fit.x_forbid_fitting*"``). ``report`` prints code only
(counts, file:line and the mutation diff), never data or model values.
Before mutmut starts, ``run`` resets survived/no-tests verdicts if the module's
tests changed since the last run, and writes ``prerun-snapshot.json`` so
``tools/mutation_results.py export`` can tell which mutants this run re-tested.
"""
from __future__ import annotations

import argparse
import filecmp
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
CONFIG = REPO / "tools" / "mutation_pilot.toml"

# mutmut 3.8 stats.status_by_exit_code, restated so the report does not need
# mutmut importable just to count. Unknown codes are "suspicious", as there.
_STATUS = {
    1: "killed", 3: "killed", 0: "survived", 5: "no tests", 33: "no tests",
    2: "interrupted", None: "not checked", 34: "skipped", 35: "suspicious",
    36: "timeout", 37: "type check", -24: "timeout", 24: "timeout",
    152: "timeout", 255: "timeout", -11: "segfault", -9: "segfault",
}


def status_of(code: int | None) -> str:
    return _STATUS.get(code, "suspicious")


def load_config() -> dict:
    with CONFIG.open("rb") as fh:
        return tomllib.load(fh)


def home() -> Path:
    raw = os.environ.get("MUTATION_PILOT_HOME")
    base = Path(raw) if raw else Path.home() / ".cache" / "investing-plan-mutation-pilot"
    base = base.expanduser().resolve()
    # The main checkout too, when this runs from a worktree under it. That also
    # keeps state out of data/, which only exists inside a checkout.
    for forbidden in (REPO, Path("/root/investing-plan").resolve()):
        if base == forbidden or forbidden in base.parents:
            sys.exit(f"MUTATION_PILOT_HOME must be outside the repo ({forbidden}); got {base}")
    return base


def module_cfg(cfg: dict, name: str) -> dict:
    modules = cfg["modules"]
    if name not in modules:
        sys.exit(f"unknown module {name!r}; known: {', '.join(modules)}")
    return modules[name]


def enabled_modules(cfg: dict) -> list[str]:
    """Modules that run. One with an ``excluded`` reason is listed, never run."""
    return [name for name, mod in cfg["modules"].items() if not mod.get("excluded")]


def expand(patterns: list[str], tracked: list[str], *, skip: list[str] = ()) -> list[str]:
    """Tracked paths matching ``patterns`` (fnmatch; ``*`` spans ``/``), in
    pattern order, minus ``skip``. A pattern that matches nothing is an error:
    a renamed file must not silently drop out of the matrix."""
    out: list[str] = []
    for pat in patterns:
        hits = [p for p in tracked if fnmatch.fnmatchcase(p, pat)]
        if not hits:
            sys.exit(f"pattern matches no tracked file: {pat}")
        out.extend(h for h in hits if h not in out)
    return [p for p in out if not any(fnmatch.fnmatchcase(p, s) for s in skip)]


def mutate_files(cfg: dict, name: str, tracked: list[str] | None = None) -> list[str]:
    mod = module_cfg(cfg, name)
    tracked = tracked if tracked is not None else _tracked(["engine"])
    return expand(mod["mutate"], tracked, skip=mod.get("skip", []))


def test_files(cfg: dict, name: str, tracked: list[str] | None = None) -> list[str]:
    tracked = tracked if tracked is not None else _tracked(["tests"])
    return expand(module_cfg(cfg, name).get("tests", []), tracked)


# -- work copy ---------------------------------------------------------------

def _tracked(paths: list[str]) -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--", *paths],
                         check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def sync_workdir(name: str, cfg: dict, *, fresh: bool) -> Path:
    """Mirror the tracked ``copy`` trees into the module's work copy.

    Unchanged files are left alone (their mtime is what mutmut uses to keep
    earlier results); changed files are rewritten, so mutmut regenerates and
    re-tests exactly the functions whose source hash moved.
    """
    defaults, mod = cfg["defaults"], module_cfg(cfg, name)
    work = home() / name
    if fresh and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    if mod.get("excluded"):
        sys.exit(f"{name} is excluded: {mod['excluded']}")
    tracked_list = _tracked(defaults["copy"])
    tracked = set(tracked_list)
    # Top-level entries of the copy: what mutmut must also copy into mutants/.
    tops = sorted({rel.split("/", 1)[0] for rel in tracked_list})
    mutate, tests = mutate_files(cfg, name, tracked_list), test_files(cfg, name, tracked_list)
    for rel in sorted(tracked):
        src, dst = REPO / rel, work / rel
        if not src.is_file():
            continue  # deleted in the working tree but still in the index
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for root in (t for t in tops if (work / t).is_dir()):
        for path in (work / root).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts \
                    and str(path.relative_to(work)) not in tracked:
                path.unlink()

    def lines(key: str, values: list[str]) -> str:
        return f"{key} =\n" + "".join(f"    {v}\n" for v in values)

    setup = ("[mutmut]\n"
             + lines("source_paths", ["engine"])
             + lines("only_mutate", mutate)
             # mutmut copies source_paths and tests/ into mutants/ by itself;
             # every other copied tree must be listed to be importable there.
             + lines("also_copy", [t for t in tops if t not in ("engine", "tests")] or ["tests"])
             + lines("pytest_add_cli_args_test_selection", tests)
             + lines("pytest_add_cli_args", defaults["pytest_args"]
                     + [f"--deselect={d}" for d in defaults.get("deselect", [])])
             + f"timeout_constant = {float(defaults['timeout_constant'])}\n"
             + f"timeout_multiplier = {float(defaults['timeout_multiplier'])}\n"
             # The work copy is not a git checkout; source hashes decide staleness.
             + "use_git_change_detection = false\n"
             # mutmut's default "fork" forks each mutant from the process that
             # already ran the whole test selection, so module state the tests
             # left behind (no_fit's thread-local flag, caches) leaks into every
             # mutant: the smoke run scored a killable mutant as survived.
             # "forkserver" forks from a process that has only collected tests.
             + "process_isolation = forkserver\n")
    (work / "setup.cfg").write_text(setup)
    return work


# -- commands ----------------------------------------------------------------

def cmd_list(cfg: dict, _args) -> int:
    for name, mod in cfg["modules"].items():
        state = f"EXCLUDED: {mod['excluded']}" if mod.get("excluded") else f"why: {mod['why']}"
        print(f"{name}\n  mutate: {' '.join(mod['mutate'])}\n  tests:  "
              f"{' '.join(mod.get('tests', []))}\n  {state}")
    return 0


def cmd_matrix(cfg: dict, args) -> int:
    """JSON list of the modules a CI run covers: every enabled module, or the
    comma-separated ``--only`` subset (an unknown or excluded name is an error)."""
    names = enabled_modules(cfg)
    if args.only and args.only.strip():
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        bad = [n for n in wanted if n not in names]
        if bad:
            sys.exit(f"not enabled modules: {bad}; enabled: {', '.join(names)}")
        names = [n for n in names if n in wanted]
    print(json.dumps(names))
    return 0


# Test-side changes mutmut cannot see: it re-tests a mutant only when the
# mutated function's own source changes. A new or stronger test therefore
# leaves an old "survived" verdict standing until a full run. The driver keeps
# a digest of the module's test files plus the shared tests/ helpers, and when
# it moves, resets survived and no-tests verdicts so this run re-tests them.
# Killed verdicts are kept; the weekly full run re-derives everything.
CI_STATE = "mutation-ci-state.json"
_RETEST_ON_TEST_CHANGE = {0, 5, 33}


def tests_digest(work: Path, tests: list[str]) -> str:
    helpers = sorted(p.relative_to(work).as_posix() for p in (work / "tests").glob("*.py")
                     if not p.name.startswith("test_"))
    h = hashlib.sha256()
    for rel in sorted(set(tests) | set(helpers)):
        h.update(rel.encode() + b"\0" + (work / rel).read_bytes() + b"\0")
    return h.hexdigest()


def reset_on_test_change(work: Path, files: list[str], digest: str) -> int:
    """Reset survived/no-tests verdicts when the digest moved; record it."""
    state_path = work / "mutants" / CI_STATE
    old = json.loads(state_path.read_text()).get("tests_digest") if state_path.exists() else None
    reset = 0
    if old is not None and old != digest:
        for rel in files:
            meta_path = work / "mutants" / (rel + ".meta")
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            codes = meta["exit_code_by_key"]
            for key, code in codes.items():
                if code in _RETEST_ON_TEST_CHANGE:
                    codes[key] = None
                    reset += 1
            meta_path.write_text(json.dumps(meta, indent=4))
    if (work / "mutants").is_dir():
        state_path.write_text(json.dumps({"tests_digest": digest}) + "\n")
    return reset


def cmd_count(cfg: dict, args) -> int:
    """Mutants mutmut would generate per file. Parses only; runs no tests."""
    names = args.modules or list(cfg["modules"])
    for name in names:
        work = sync_workdir(name, cfg, fresh=False)
        # mutate_file_contents, not create_mutations: the latter also counts
        # mutations mutmut then drops (e.g. inside decorated functions).
        code = ("import sys\nfrom mutmut.mutation.file_mutation import mutate_file_contents\n"
                "for rel in sys.argv[1:]:\n"
                "    print(len(mutate_file_contents(rel, open(rel).read()).mutant_names), rel)\n")
        out = subprocess.run([sys.executable, "-c", code, *mutate_files(cfg, name)],
                             cwd=work, check=True, capture_output=True, text=True).stdout
        total = sum(int(line.split()[0]) for line in out.splitlines())
        print(f"{name}: {total} mutants")
        for line in out.splitlines():
            print(f"  {line}")
    return 0


def cmd_run(cfg: dict, args) -> int:
    from mutation_results import SNAPSHOT_NAME, snapshot

    work = sync_workdir(args.module, cfg, fresh=args.fresh)
    files = mutate_files(cfg, args.module)
    digest = tests_digest(work, test_files(cfg, args.module))
    reset = reset_on_test_change(work, files, digest)
    if reset:
        print(f"[mutation_pilot] tests changed: {reset} survived/no-tests verdicts reset", flush=True)
    # The verdicts before this run, so the export can mark what it re-tested.
    (work / SNAPSHOT_NAME).write_text(json.dumps(snapshot(work, files)))
    children = args.max_children or int(cfg["defaults"]["max_children"])
    cmd = [sys.executable, "-u", "-m", "mutmut", "run", "--max-children", str(children), *args.globs]
    print(f"[mutation_pilot] {args.module}: {' '.join(cmd[2:])}  (cwd {work})", flush=True)
    # mutmut forks one worker per mutant from a process that has already run
    # the tests. A BLAS/OpenMP thread pool started before the fork deadlocks
    # the child, which then reads as a false "timeout" (seen in the smoke run
    # on tests that import sklearn). Single-threaded native libraries avoid it,
    # and --max-children is the parallelism anyway.
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    rc = subprocess.run(cmd, cwd=work, env=env).returncode
    # On a first run mutants/ did not exist before mutmut; record the digest now
    # so the next run compares against this one.
    reset_on_test_change(work, [], digest)
    if rc != 0:
        diagnose(work, cfg, args.module, env)
    return rc


def diagnose(work: Path, cfg: dict, name: str, env: dict) -> int:
    """After a failed mutmut run, rerun the module's tests in plain pytest.

    mutmut swallows pytest's output when its clean/stats run fails ("failed to
    collect stats. runner returned 1" and nothing else). This reruns the same
    selection with the same pytest args, from the same directory (mutants/,
    trampolined code, unmutated), without -x, and prints the failing test ids
    with short tracebacks. If it passes, the failure only happens inside mutmut
    (state it leaves in the process), and the message says so.
    """
    defaults = cfg["defaults"]
    cwd = work / "mutants" if (work / "mutants").is_dir() else work
    cmd = [sys.executable, "-m", "pytest", "--rootdir=.", "-q", "-rfE", "--tb=short",
           "-p", "no:randomly", "-p", "no:random-order", *defaults["pytest_args"],
           *[f"--deselect={d}" for d in defaults.get("deselect", [])],
           *test_files(cfg, name)]
    denv = dict(env, MUTANT_UNDER_TEST="")
    print(f"\n[mutation_pilot] diagnose {name}: plain pytest of the selected tests in "
          f"{cwd}", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=denv, capture_output=True, text=True,
                              timeout=1500)
    except subprocess.TimeoutExpired:
        print("[mutation_pilot] diagnose: pytest did not finish in 1500 s", flush=True)
        return -1
    lines = (proc.stdout + proc.stderr).splitlines()
    print("\n".join(lines[-250:]), flush=True)
    verdict = ("tests fail outside mutmut too: see above" if proc.returncode else
               "tests PASS outside mutmut: the failure depends on mutmut's in-process run")
    print(f"[mutation_pilot] diagnose {name}: pytest exited {proc.returncode}; {verdict}",
          flush=True)
    return proc.returncode


def _function_span(source: str, func: str, cls: str | None) -> tuple[int, int] | None:
    import ast
    tree = ast.parse(source)
    body = tree.body
    if cls:
        body = next((n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), [])
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return start, node.end_lineno
    return None


def _changed_line(diff: str, source_lines: list[str], span: tuple[int, int] | None) -> int | None:
    """The original-file line of the first removed line in ``diff``."""
    removed = next((ln[1:] for ln in diff.splitlines()
                    if ln.startswith("-") and not ln.startswith("---")), None)
    if removed is None or span is None:
        return span[0] if span else None
    lo, hi = span
    hits = [i for i in range(lo, hi + 1) if source_lines[i - 1].strip() == removed.strip()]
    return hits[0] if hits else lo


def cmd_report(cfg: dict, args) -> int:
    names = args.modules or list(cfg["modules"])
    base = home()
    print(f"{'module / file':58s} {'total':>6s} {'killed':>6s} {'surv':>6s} {'t/out':>6s} "
          f"{'notest':>6s} {'other':>6s} {'unrun':>6s} {'score':>7s}")
    survivors: list[tuple[str, str, str, str]] = []
    for name in names:
        work = base / name
        per_file: dict[str, Counter] = {}
        for rel in mutate_files(cfg, name):
            meta = work / "mutants" / (rel + ".meta")
            counts: Counter = Counter()
            if meta.exists():
                codes = json.loads(meta.read_text())["exit_code_by_key"]
                for key, code in sorted(codes.items()):
                    status = status_of(code)
                    counts[status] += 1
                    if status in ("survived", "no tests"):
                        survivors.append((name, rel, key, status))
            per_file[rel] = counts
        total = sum(per_file.values(), Counter())
        for label, c in [(name, total)] + [(f"  {rel}", c) for rel, c in per_file.items()]:
            killed = c["killed"] + c["type check"]
            other = c["suspicious"] + c["segfault"] + c["interrupted"]
            unrun = c["not checked"] + c["skipped"]
            n = sum(c.values())
            checked = n - unrun
            # mutmut's own convention: a timeout counts as a kill. "no tests"
            # (no targeted test executes the function) counts against the score.
            score = f"{100 * (killed + c['timeout']) / checked:6.1f}%" if checked else "     --"
            print(f"{label:58s} {n:6d} {killed:6d} {c['survived']:6d} {c['timeout']:6d} "
                  f"{c['no tests']:6d} {other:6d} {unrun:6d} {score}")
    if args.no_diffs or not survivors:
        return 0
    print("\nSurvivors (file:line, status, mutant, diff):")
    by_module: dict[str, list] = {}
    for item in survivors:
        by_module.setdefault(item[0], []).append(item)
    for name, items in by_module.items():
        work = base / name
        cwd = os.getcwd()
        os.chdir(work)  # mutmut's diff helpers resolve mutants/ relative to cwd
        try:
            from mutmut.mutation.diff_apply import get_diff_for_mutant
            from mutmut.utils.format_utils import orig_function_and_class_names_from_key
            for _, rel, key, status in items:
                source = (work / rel).read_text()
                func, cls = orig_function_and_class_names_from_key(key)
                span = _function_span(source, func.rpartition(".")[2], cls)
                try:
                    diff = get_diff_for_mutant(key, path=rel)
                except Exception as exc:  # stale index: report, don't die
                    diff = f"(diff unavailable: {type(exc).__name__})"
                line = _changed_line(diff, source.splitlines(), span)
                print(f"\n{rel}:{line if line else '?'}  [{status}]  {key}")
                body = [ln for ln in diff.splitlines() if not ln.startswith(("---", "+++"))]
                print("\n".join(f"    {ln}" for ln in body))
        finally:
            os.chdir(cwd)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("matrix", help="JSON list of enabled modules, for the CI matrix")
    p.add_argument("--only", default="", help="comma-separated subset")
    p = sub.add_parser("count")
    p.add_argument("modules", nargs="*")
    p = sub.add_parser("run")
    p.add_argument("module")
    p.add_argument("globs", nargs="*", help="mutant-name globs (default: all)")
    p.add_argument("--max-children", type=int, default=None)
    p.add_argument("--fresh", action="store_true", help="delete the work copy and its results first")
    p = sub.add_parser("report")
    p.add_argument("modules", nargs="*")
    p.add_argument("--no-diffs", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config()
    return {"list": cmd_list, "matrix": cmd_matrix, "count": cmd_count, "run": cmd_run,
            "report": cmd_report}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
