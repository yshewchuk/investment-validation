"""Mutation-testing pilot, pytest-gremlins backend: run one module's tests with
parallel gremlin workers and report the result.

This is the runner that replaces mutmut in the active CI workflow. It reuses the
*module partition* from ``tools/mutation_pilot.toml`` unchanged -- the same
enabled/excluded modules, the same ``mutate``/``skip``/``tests`` fnmatch
expansion (via ``mutation_pilot``'s reliable ``load_config``, ``expand``,
``mutate_files``, ``test_files`` and ``_tracked`` helpers) and the same
``[defaults] deselect`` list. Only the execution backend differs, so
``tests/test_mutation_ci.py``'s exhaustive "every engine/v2 file sits in exactly
one module" ownership test governs both runners.

Unlike mutmut 3 (which forks a mutated copy of ``engine/`` into a work copy OUTSIDE
the repo), pytest-gremlins is a pytest plugin that mutates bytecode in memory -- its
default worker pool runs each mutant in its own subprocess of the checkout -- and
never rewrites files on disk, so it runs directly in the repo checkout. There is no
work copy, no ``$MUTATION_PILOT_HOME``, no sync step.

Commands::

    python3 tools/gremlin_pilot.py matrix [--only a,b]      # CI matrix (JSON)
    python3 tools/gremlin_pilot.py fingerprint MODULE       # cache namespace (hex)
    python3 tools/gremlin_pilot.py run MODULE [--workers N] [--fresh]

``fingerprint`` prints the outer invalidation digest the CI cache key namespaces
each module's ``.gremlins_cache`` with; see ``cache_fingerprint`` below for what
it covers (every tracked input) and why it is deliberately broader than
pytest-gremlins' own keys.

``run`` builds and executes::

    python -m pytest --gremlins \\
        --gremlin-targets=<comma-separated expanded source files> \\
        --gremlin-workers=N --gremlin-cache --gremlin-report=json \\
        -p no:cacheprovider --deselect=... <expanded test files>

Rules this tool encodes, all pinned by ``tests/test_gremlin_ci.py``:

* Workers are explicit and capped. Default 2 locally; under CI (``GITHUB_ACTIONS``)
  ``min(nproc, 4)``. A ``--workers`` request is never honored above that ceiling.
  ``--gremlin-workers`` is what makes gremlins parallel; there is no separate
  parallelism switch and never a second runner (pytest-xdist) stacked on top of
  it -- we simply never pass pytest's ``-n``, so xdist is loaded but has nothing
  to run. We must NOT disable it either: pytest-gremlins 1.9.0 implements
  xdist's hooks (``pytest_configure_node``), so ``-p no:xdist`` makes pluggy
  abort collection with PluginValidationError: unknown hook (CI run
  36059422920 -- pytest exit 3, no raw report, every module a tool failure).
* No batch mode. pytest-gremlins' batch mode unions the test pools of the files
  that cover a mutant, which runs unrelated tests against it and so manufactures
  false timeouts. We never pass ``--gremlin-batch``.
* The per-mutant timeout is hardcoded to 30 s in 1.9.0 and is not configurable;
  we record it in the run metadata so a slow suite reads as an expectation, not a
  mystery.
* BLAS/OpenMP thread pools are pinned to 1 for the whole run, exactly as the
  mutmut driver does: pytest-gremlins' default worker pool runs each mutant in
  its own subprocess and every child inherits this environment, so a native
  thread pool the parent would hand down is the classic false-timeout source.
* The CI cache namespace is ``fingerprint``, not pytest-gremlins' own keys. The
  plugin caches a gremlin by its own source hash plus the hashes of the test
  files it directly selected, which does not see ``tests/conftest.py``, a shared
  test helper, a production module or experiment fixture those imports pull in,
  the dependency locks or the mutation config. ``cache_fingerprint`` digests
  EVERY tracked repository input (``git ls-files``) rather than a hand list that
  can drift, so any of those pushes starts a new namespace instead of reusing a
  stale mutation outcome; unchanged inputs keep the identical namespace.
* ``--fresh`` starts clean: it clears the gremlins cache directory
  (``.gremlins_cache``) so nothing is reused. Every run -- fresh or incremental
  -- also deletes any old raw ``coverage/gremlins/gremlins.json`` first, so a
  stale report can never look like the result of this run. The raw JSON is the
  only source the report adapter reads, and "current" must mean "written now".
* The exit code is the pytest/gremlins process's own, unchanged. Scores and
  survivors NEVER fail a run: this is report-only mutation testing. Missing or
  malformed current-run results are diagnosed (and surfaced through the report
  adapter's return code), never silenced here, and there is no ``|| true``.
* Long runs stay visible: the child's output is forwarded live, and a heartbeat
  line is emitted at least once a minute while the subprocess runs, so a silent
  stretch is not mistaken for a hang.

The result files themselves are built by ``tools/gremlin_results.py`` (export /
merge), a separate adapter that reads the raw JSON this run writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import mutation_pilot as pilot  # noqa: E402  (reuses load_config + module partition)

# pytest-gremlins 1.9.0 facts this tool is written against: Python 3.14 is
# supported; the raw report is written to coverage/gremlins/gremlins.json; the
# cache directory is .gremlins_cache and 1.9.0 does have --gremlin-clear-cache;
# the default worker pool runs mutants in subprocesses; the per-mutant timeout is
# hardcoded to 30s and has no configuration option; it implements pytest-xdist's
# hooks (pytest_configure_node), so `-p no:xdist` aborts collection with a pluggy
# PluginValidationError (CI run 36059422920) and xdist must stay loaded (never
# passed a `-n`; parallelism is --gremlin-workers only). None of these are guessed.
RAW_REL = Path("coverage") / "gremlins" / "gremlins.json"
CACHE_DIRNAME = ".gremlins_cache"
PER_MUTANT_TIMEOUT_SECONDS = 30  # hardcoded in 1.9.0, not configurable
HEARTBEAT_SECONDS = 60.0
DEFAULT_LOCAL_WORKERS = 2
CI_WORKERS_CAP = 4
# The same native-thread pin the mutmut driver applies: pytest-gremlins' default
# pool runs each mutant in a subprocess that inherits this environment, so every
# child starts with its BLAS/OpenMP pools at one thread.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# -- worker sizing -----------------------------------------------------------

def is_ci(env: dict[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get("GITHUB_ACTIONS", "").strip().lower() == "true"


def cpu_count() -> int:
    """Cores this process may actually use (respects a cpuset / taskset)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def resolve_workers(explicit: int | None = None, *, ci: bool = False,
                    nproc: int = 1) -> int:
    """Explicit or default worker count, capped. Locally default 2, under CI
    ``min(nproc, 4)``. An explicit ``--workers`` is never honored above the
    ceiling, so a bad value can never oversubscribe the runner.
    """
    nproc = max(1, int(nproc))
    ceiling = min(nproc, CI_WORKERS_CAP) if ci else nproc
    if explicit is not None:
        return max(1, min(int(explicit), ceiling))
    if ci:
        return ceiling
    return min(DEFAULT_LOCAL_WORKERS, ceiling)


# -- environment -------------------------------------------------------------

def thread_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of ``base`` (default the live environment) with every BLAS/OpenMP
    thread pool pinned to 1. The child's own harness variables (MUTANT_UNDER_TEST,
    PYTEST_CURRENT_TEST) are left exactly as they are -- gremlins sets what it
    needs and there is no supported reason to strip the parent's."""
    env = dict(os.environ if base is None else base)
    for var in _THREAD_VARS:
        env[var] = "1"
    return env


# -- report / cache paths (all relative to the run's working directory) ------

def raw_report_path(cwd: Path) -> Path:
    return Path(cwd) / RAW_REL


def cache_dir(cwd: Path) -> Path:
    return Path(cwd) / CACHE_DIRNAME


def clear_stale_report(cwd: Path) -> bool:
    """Delete any old raw ``coverage/gremlins/gremlins.json`` before a run, so a
    report from a previous run can never be mistaken for this one's. Returns
    whether a file was removed."""
    path = raw_report_path(cwd)
    if path.exists():
        path.unlink()
        return True
    return False


def clear_cache_dir(cwd: Path) -> bool:
    """``--fresh`` support: remove the gremlins cache directory so nothing is
    reused. pytest-gremlins 1.9.0 DOES have ``--gremlin-clear-cache``, and passing
    it would also work; this tool clears ``.gremlins_cache`` -- the documented
    cache directory -- itself instead, so the pytest argv stays identical whether
    or not the run is fresh (one command shape, one thing for the tests to pin)
    and a fresh run cannot be silently downgraded by a dropped flag. Both paths
    clear exactly the same state. Returns whether a directory was removed."""
    path = cache_dir(cwd)
    if path.is_dir():
        shutil.rmtree(path)
        return True
    return False


def describe_raw_report(cwd: Path) -> str:
    """Classify the current-run raw report as ``present`` / ``empty`` /
    ``missing``. Diagnostic only: gremlins' return code is what this run reports;
    whether the results are structurally sound or contain ERROR gremlins is the
    report adapter's job, decided from this same file."""
    path = raw_report_path(cwd)
    if not path.exists():
        return "missing"
    return "present" if path.stat().st_size else "empty"


# -- cache invalidation fingerprint ------------------------------------------

# pytest-gremlins 1.9.0 caches a gremlin by its OWN source hash plus the hashes of
# the test files it directly selected. That is narrower than what a module's
# outcome actually depends on: ``tests/conftest.py``, the shared ``tests/`` helpers
# the selected tests import, the production modules those imports pull in (not
# just the mutation tools), the experiment fixtures a selected test reads, the
# dependency locks and the mutation/workflow config can all change whether a
# mutant is killed, survives or times out without moving any hash the plugin looks
# at. Rather than maintain a hand list of every one of those -- an incomplete list
# IS the cache-correctness bug (an unseen edit reuses a stale mutant outcome) --
# the fingerprint covers EVERY tracked repository input, read from the CI
# checkout's index via ``git ls-files``. The tracked tree is small (~1000 files),
# so this is cheap, and over-invalidation is the accepted cost: a needless restart
# only re-runs a module, while a stale hit publishes a wrong mutation outcome. Two
# consequences, both deliberate -- a docs-only push may now restart a module (a
# ``.md`` edit cannot move a kill or a survive, but keeping the rule complete
# matters more than keeping that one push cached), and an unrelated tool edit is no
# longer special-cased out. Untracked generated cache/log paths (``.gremlins_cache``,
# ``.oc_logs``, ``__pycache__``, ``coverage/``) are not in the index, so they can
# never perturb the digest -- and the key is computed before any cache is restored
# anyway. The module name is folded in, so the per-module matrix namespaces stay
# separate (and the key keeps ``matrix.module`` in its own right). The scheme string
# is folded into the digest, so bumping ``FINGERPRINT_SCHEME`` on a recipe change
# moves every namespace and can never restore an entry the old algorithm wrote;
# v1 -> v2 is exactly that transition (narrow hand list -> all tracked inputs).
FINGERPRINT_SCHEME = "gremlins-cache-v2"
_MISSING_INPUT = b"<missing>"


def fingerprint_inputs(cwd: Path | None = None) -> list[str]:
    """Every tracked repository path, sorted and POSIX-relative, exactly as the CI
    checkout's index lists it (``git ls-files -z``). Listing tracked files rather
    than walking the filesystem is what keeps untracked/generated paths out and
    makes "the whole tracked tree" mean the whole tracked tree -- there is nothing
    left to forget. A missing git or a non-repo directory raises: the
    ``fingerprint`` step must fail loudly rather than fall back to a narrow key and
    silently reuse a stale cache."""
    root = REPO if cwd is None else Path(cwd)
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                         check=True, stdout=subprocess.PIPE).stdout
    return sorted(p for p in out.decode().split("\0") if p)


def cache_fingerprint(module: str = "", cwd: Path | None = None) -> str:
    """The outer invalidation digest for one module's cache namespace: a hash over
    every tracked input plus the module and the scheme, so no unseen tracked edit
    can reuse a stale mutation outcome (see the block comment above)."""
    root = REPO if cwd is None else Path(cwd)
    h = hashlib.sha256()
    h.update(f"{FINGERPRINT_SCHEME}\0module={module}\0".encode())
    for rel in fingerprint_inputs(root):
        path = root / rel
        blob = path.read_bytes() if path.is_file() else _MISSING_INPUT
        h.update(rel.encode() + b"\0" + blob + b"\0")
    return h.hexdigest()


# -- command construction ----------------------------------------------------

def build_pytest_command(targets: list[str], tests: list[str], workers: int, *,
                         fresh: bool, deselect: list[str],
                         pytest_args: list[str],
                         python: str | None = None) -> list[str]:
    """The ``python -m pytest --gremlins ...`` argv for one module.

    ``--gremlin-cache`` is always passed: it both reads ``.gremlins_cache`` (an
    incremental run reuses it) and rebuilds it (a ``--fresh`` run has had the
    directory cleared first, so this run writes a clean one). ``--gremlin-workers``
    is the parallelism switch; ``--gremlin-report=json`` makes the raw
    ``coverage/gremlins/gremlins.json`` the adapter reads. ``-p no:cacheprovider``
    comes from ``pytest_args`` (the toml's default). pytest-xdist is neither
    disabled nor invoked: no pytest ``-n`` (gremlins' own workers do the running)
    and no ``-p no:xdist`` either -- 1.9.0 implements xdist's hooks, so
    disabling the plugin fails pluggy validation during collection. No batch
    flag ever appears.
    """
    python = python or sys.executable
    cmd = [python, "-m", "pytest", "--gremlins",
           f"--gremlin-targets={','.join(targets)}",
           f"--gremlin-workers={workers}",
           "--gremlin-cache", "--gremlin-report=json"]
    cmd += list(pytest_args)
    cmd += [f"--deselect={d}" for d in deselect]
    cmd += list(tests)
    return cmd


# -- subprocess streaming ----------------------------------------------------

def beat_line(elapsed: float) -> str:
    """One heartbeat line for an elapsed run (no trailing newline)."""
    return f"[gremlin_pilot] still running... {int(elapsed)}s elapsed"


def stream_and_heartbeat(proc: subprocess.Popen[str], *, interval: float = HEARTBEAT_SECONDS,
                         write=None, clock=time.monotonic) -> int:
    """Forward the child's merged stdout/stderr live and emit a heartbeat line at
    least every ``interval`` seconds while it runs, then return its exit code.

    The reader runs in the calling thread (so output order is preserved and there
    is no pipe to deadlock); the heartbeat runs on a daemon thread that wakes on
    the child finishing, so it never lingers past the subprocess.
    """
    write = write or (lambda text: sys.stdout.write(text))
    start = clock()
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(interval):
            write(beat_line(clock() - start) + "\n")

    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        for line in proc.stdout or ():
            write(line)
    finally:
        stop.set()
        th.join(timeout=1.0)
    return proc.wait()


# -- metadata ----------------------------------------------------------------

def log_run_metadata(module: str, *, targets: list[str], tests: list[str], workers: int,
                     fresh: bool, cmd: list[str], write=None) -> None:
    """Print what this run is and the 1.9.0 constraints that shape it. Diagnostic
    only; nothing here changes the exit code the job gates on."""
    write = write or (lambda text: print(text, flush=True))
    write(f"[gremlin_pilot] {module}: {len(targets)} source file(s), {len(tests)} test file(s), "
          f"workers={workers}, fresh={fresh}")
    write(f"[gremlin_pilot] per-mutant timeout {PER_MUTANT_TIMEOUT_SECONDS}s is hardcoded in "
          f"pytest-gremlins 1.9.0 (not configurable); a slow suite expects it")
    write("[gremlin_pilot] batch mode disabled: it unions test pools across files and "
          "manufactures false timeouts")
    write(f"[gremlin_pilot] cmd: {' '.join(cmd)}")


# -- commands ----------------------------------------------------------------

def select_modules(cfg: dict, only: str = "") -> list[str]:
    """Every enabled module (a module with an ``excluded`` reason is listed in the
    toml but never run), or the comma-separated ``--only`` subset. An unknown,
    excluded or empty name is an error, so a renamed module cannot silently drop
    out of the matrix. A value that names NOTHING (``--only ,``) is an error too,
    never an empty matrix: only an absent/blank ``--only`` means "every enabled
    module", so a mistyped subset cannot quietly reduce the run to zero modules
    (GitHub renders a dispatched empty input as the empty string)."""
    names = pilot.enabled_modules(cfg)
    if only.strip():
        wanted = [n.strip() for n in only.split(",") if n.strip()]
        if not wanted:
            sys.exit(f"--only {only!r} names no module; pass an empty --only for "
                     f"every enabled module ({', '.join(names)})")
        bad = [n for n in wanted if n not in names]
        if bad:
            sys.exit(f"not enabled modules: {bad}; enabled: {', '.join(names)}")
        names = [n for n in names if n in wanted]
    return names


def cmd_matrix(cfg: dict, args) -> int:
    print(json.dumps(select_modules(cfg, getattr(args, "only", ""))))
    return 0


def cmd_fingerprint(cfg: dict, args) -> int:
    """Print one module's outer cache namespace: the digest the workflow's ``key``
    step puts in front of every ``.gremlins_cache`` key, so any change to a tracked
    input -- a shared helper, a fixture, an imported production module or experiment
    ``run.py``, a dependency lock or the config -- lands in a NEW namespace instead
    of reusing stale mutation outcomes. Validation reuses ``select_modules``, so a blank,
    unknown or excluded name is an error exactly as it is for ``matrix``/``run``,
    and a comma list is refused: one name per namespace, so a renamed module can
    never quietly inherit another module's cached results."""
    module = args.module.strip()
    if not module:
        sys.exit("fingerprint needs a module name: the module is part of the namespace")
    names = select_modules(cfg, module)
    if len(names) != 1 or names[0] != module:
        sys.exit(f"fingerprint takes exactly one enabled module, got {args.module!r}")
    cwd = Path(args.cwd) if getattr(args, "cwd", None) else REPO
    print(cache_fingerprint(module, cwd))
    return 0


def cmd_run(cfg: dict, args) -> int:
    """Run one module under pytest-gremlins in the repo checkout. Return the
    subprocess's own exit code, unchanged. Scores/survivors never fail the run."""
    module = args.module
    mod = pilot.module_cfg(cfg, module)
    if mod.get("excluded"):
        sys.exit(f"{module} is excluded: {mod['excluded']}")
    cwd = Path(args.cwd) if getattr(args, "cwd", None) else REPO
    targets = pilot.mutate_files(cfg, module)
    tests = pilot.test_files(cfg, module)
    if not targets or not tests:
        sys.exit(f"{module}: nothing to run (targets or tests empty)")
    workers = resolve_workers(args.workers, ci=is_ci(), nproc=args.nproc or cpu_count())
    deselect = list(cfg["defaults"].get("deselect", []))
    pytest_args = list(cfg["defaults"].get("pytest_args", ["-p", "no:cacheprovider"]))

    clear_stale_report(cwd)  # every run: a stale raw report must not look current
    if args.fresh:
        clear_cache_dir(cwd)
    cmd = build_pytest_command(targets, tests, workers, fresh=args.fresh,
                               deselect=deselect, pytest_args=pytest_args)
    log_run_metadata(module, targets=targets, tests=tests, workers=workers,
                     fresh=args.fresh, cmd=cmd)
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=thread_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    rc = stream_and_heartbeat(proc)
    status = describe_raw_report(cwd)
    if status != "present":
        print(f"[gremlin_pilot] {module}: raw report {status} "
              f"({raw_report_path(cwd)}); the report adapter will treat this run as "
              f"a tool error, not a score", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("matrix", help="JSON list of enabled modules, for the CI matrix")
    p.add_argument("--only", default="", help="comma-separated subset")
    p = sub.add_parser("fingerprint",
                       help="outer cache-namespace digest of one module over every tracked input "
                            "(hex, via git ls-files; no tests run)")
    p.add_argument("module", help="the module whose cache namespace to print")
    p.add_argument("--cwd", type=Path, default=None, help="tree root to digest (default: repo checkout)")
    p = sub.add_parser("run", help="run one module's tests under pytest-gremlins")
    p.add_argument("module")
    p.add_argument("--workers", type=int, default=None,
                   help="gremlin workers (default: 2 local, min(nproc,4) under CI; capped)")
    p.add_argument("--fresh", action="store_true",
                   help="clear the gremlins cache so nothing is reused")
    p.add_argument("--nproc", type=int, default=None, help="override for worker sizing")
    p.add_argument("--cwd", type=Path, default=None, help="run directory (default: repo checkout)")
    args = parser.parse_args(argv)
    cfg = pilot.load_config()
    return {"matrix": cmd_matrix, "fingerprint": cmd_fingerprint, "run": cmd_run}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
