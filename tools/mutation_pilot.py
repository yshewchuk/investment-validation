"""Mutation-testing pilot: run mutmut per critical module and report the result.

Config: ``tools/mutation_pilot.toml``. Guide: ``tests/README.md``,
"Mutation-testing pilot".

mutmut 3 always writes its state to ``./mutants/`` under the directory it runs
in, so running it from the repo would put a mutated copy of ``engine/`` (whose
``.py`` files the allowlist .gitignore lets back in) inside the checkout. This
tool instead runs every module in its own work copy OUTSIDE the repo:

    $MUTATION_PILOT_HOME/<module>/          (default ~/.cache/investing-plan-mutation-pilot)
        engine/ tests/ checks/ tools/       tracked files only, synced from the repo
        setup.cfg                           generated [mutmut] section for this module
        mutants/                            mutmut's own state and results

Nothing from ``data/`` is copied, so a test that silently needs real data fails
the clean run instead of reading it.

Commands::

    python3 tools/mutation_pilot.py list
    python3 tools/mutation_pilot.py count [MODULE ...]          # mutant counts, no tests run
    python3 tools/mutation_pilot.py run MODULE [--max-children N] [--fresh] [GLOB ...]
    python3 tools/mutation_pilot.py report [MODULE ...] [--no-diffs]

``run`` is the heavy step: launch it under ``tools/bounded_run.py``. ``GLOB``
restricts the run to matching mutant names (mutmut's own syntax, e.g.
``"engine.models.no_fit.x_forbid_fitting*"``). ``report`` prints code only
(counts, file:line and the mutation diff), never data or model values.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
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

    tracked = set(_tracked(defaults["copy"]))
    missing = [p for p in mod["mutate"] + mod["tests"] if p not in tracked]
    if missing:
        sys.exit(f"{name}: not tracked in the repo: {missing}")
    for rel in sorted(tracked):
        src, dst = REPO / rel, work / rel
        if not src.is_file():
            continue  # deleted in the working tree but still in the index
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for root in defaults["copy"]:
        for path in (work / root).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts \
                    and str(path.relative_to(work)) not in tracked:
                path.unlink()

    def lines(key: str, values: list[str]) -> str:
        return f"{key} =\n" + "".join(f"    {v}\n" for v in values)

    setup = ("[mutmut]\n"
             + lines("source_paths", ["engine"])
             + lines("only_mutate", mod["mutate"])
             # mutmut copies source_paths and tests/ into mutants/ by itself;
             # every other copied tree must be listed to be importable there.
             + lines("also_copy", [c for c in defaults["copy"] if c not in ("engine", "tests")] or ["tests"])
             + lines("pytest_add_cli_args_test_selection", mod["tests"])
             + lines("pytest_add_cli_args", defaults["pytest_args"])
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
        print(f"{name}\n  mutate: {' '.join(mod['mutate'])}\n  tests:  {' '.join(mod['tests'])}\n  why:    {mod['why']}")
    return 0


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
        out = subprocess.run([sys.executable, "-c", code, *module_cfg(cfg, name)["mutate"]],
                             cwd=work, check=True, capture_output=True, text=True).stdout
        total = sum(int(line.split()[0]) for line in out.splitlines())
        print(f"{name}: {total} mutants")
        for line in out.splitlines():
            print(f"  {line}")
    return 0


def cmd_run(cfg: dict, args) -> int:
    work = sync_workdir(args.module, cfg, fresh=args.fresh)
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
    return subprocess.run(cmd, cwd=work, env=env).returncode


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
        for rel in module_cfg(cfg, name)["mutate"]:
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
    return {"list": cmd_list, "count": cmd_count, "run": cmd_run, "report": cmd_report}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
