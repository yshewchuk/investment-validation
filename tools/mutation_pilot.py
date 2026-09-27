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
import ast
import filecmp
import fnmatch
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
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


# mutmut's supported ``debug = true`` setting is full-run verbosity, not a
# stats-only hook: with it on, mutmut echoes every mutant's child pytest output
# for the whole run instead of swallowing it (CI run 36001042208 logged only
# "failed to collect stats. runner returned 1" and hid the traceback). That echo
# is what makes the failure diagnosable, but it floods the log for every mutant,
# so it is not free and it never reruns anything or changes the exit code the job
# gates on. The driver turns it on only where it is warranted: when the
# environment asks explicitly, or on the one CI shard whose stats step is known
# to fail. It is read here, not inside sync_workdir, so the run command and the
# CI job share one source of truth.
# ops_catalog_state now shares that CI-only default: its clean/stats step
# failed the same way in CI run 36025664817.
def stats_debug_enabled(module: str, env: dict[str, str] | None = None) -> bool:
    """Should mutmut's debug (full-run verbosity) be on for this module's run?

    An explicit ``MUTATION_PILOT_DEBUG`` always wins -- ``1/true/yes/on`` turn
    it on, anything else (``0/false/off``, even empty) turns it off. With no
    explicit value it is on only for the ``ops_legacy`` shard under CI
    (``GITHUB_ACTIONS=true``), the one run whose clean/stats step is known to
    fail; every other CI shard and every local run stays quiet.
    ``ops_catalog_state`` and ``ops_runtime`` now share that CI-only default.
    """
    source = os.environ if env is None else env
    if "MUTATION_PILOT_DEBUG" in source:
        return source["MUTATION_PILOT_DEBUG"].strip().lower() in ("1", "true", "yes", "on")
    if (source.get("GITHUB_ACTIONS", "").strip().lower() == "true"
            and module == "ops_catalog_state"):
        return True
    if (source.get("GITHUB_ACTIONS", "").strip().lower() == "true"
            and module == "ops_runtime"):
        return True
    return (source.get("GITHUB_ACTIONS", "").strip().lower() == "true"
            and module == "ops_legacy")


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


# -- PR module selection: module ownership, transitive dependency, else the --
# -- inert allowlist, else every enabled module (never zero for an          --
# -- unrecognized path) -------------------------------------------------------
#
# `changed_modules` decides, per pull_request run, which enabled modules a
# changed-file list can affect. The rule is "select ALL unless proven safe to
# skip", never the reverse: a changed path that doesn't match a currently-
# listed shape is not evidence it cannot affect a module, so it must never
# quietly select zero modules. Per changed path, in order:
#   1. some ENABLED module's own `mutate`/`tests` patterns match it
#      (`module_owns_changed_path`, checked against the bare path string, so
#      a deleted/renamed-away owned file still matches), OR the path is in
#      that module's DEPENDENCY SET (`module_dependency_closure`: the
#      transitive closure, over a static `ast` import graph of every tracked
#      `.py` file (including `conftest.py` closure roots and literal
#      `importlib.import_module`/`__import__` strings), of the imports
#      reachable from the module's own `mutate` files plus its `tests` files) -> that
#      module is selected. A module that OWNS a changed path is not
#      necessarily the only one whose DEPENDENCY SET contains it: e.g.
#      engine/v2/foundation/artifacts.py is owned by `foundation`, but every
#      module whose tests or sources import `engine.v2.foundation`
#      (transitively, through that package's `__init__.py`, which re-exports
#      names from `artifacts.py`) has it in their dependency set too, and is
#      selected alongside `foundation`.
#   2. else it is on the small, docs-only inert allowlist
#      (`tools/mutation_pilot.toml`'s `[pr_selection] inert`, minus
#      `[pr_selection] inert_skip`) -> it selects nothing.
#   3. else (unrecognized: no ENABLED module owns or transitively depends on
#      it, and it is not inert) -> every requested module is selected,
#      immediately, for the whole changed-file list.
# An EXCLUDED module's ownership (e.g. `contracts` owning
# `engine/v2/__init__.py`) does NOT count as "some module owns it" in step 1
# -- an excluded module never runs, so a path only an excluded module claims
# is exactly as unexplained as one no module claims, and must still fall
# through to "select everything". Excluded modules are never given a
# dependency set either (`module_dependency_closure` is only computed for
# `enabled_modules(cfg)`).
#
# Fail safe: if the import graph cannot be built (any tracked `.py` file
# fails to parse -- including a changed file with a syntax error, since the
# graph is built from the current on-disk tree -- or any
# other exception while building it), `changed_modules` selects every
# requested module immediately and prints why, rather than silently falling
# back to ownership-only selection.
#
# This governs both backends (gremlin_pilot.select_modules delegates to this
# function unchanged) and, by the same "unrecognized -> everything" rule,
# selects every enabled module when a PR changes the selector itself:
# tools/mutation_pilot.py, tools/gremlin_pilot.py, tools/mutation_results.py,
# tools/mutation_pilot.toml, or either mutation workflow file are none of
# them module-owned, dependency-owned, or on the inert allowlist.

def is_inert_changed_path(cfg: dict, path: str) -> bool:
    """True if `path` matches the PR-selection inert allowlist
    (`tools/mutation_pilot.toml`'s `[pr_selection] inert`, fnmatch patterns
    where `*` also matches `/`, same convention as `mutate`/`tests`/`skip`)
    and does NOT match `[pr_selection] inert_skip` (patterns carved back out
    of the allowlist: e.g. a `.md` file under `engine/dashboard/static/` would
    be a fingerprinted code asset, not documentation, if a future `inert`
    pattern ever widened to reach it again). Kept small and docs-only on
    purpose: anything the
    mutation-tested code (engine/v2) might read at runtime, or a test might
    load as a fixture, must NOT be on this list, or a real defect could hide
    behind a skipped run. Checked against the bare path string, so a
    deleted/renamed-away inert file still matches."""
    section = cfg.get("pr_selection", {})
    if any(fnmatch.fnmatchcase(path, pat) for pat in section.get("inert_skip", [])):
        return False
    return any(fnmatch.fnmatchcase(path, pat) for pat in section.get("inert", []))


def read_changed_files(path: str) -> list[str]:
    """NUL-delimited changed-file list (``git diff -z --name-only`` output),
    empty tokens dropped. An empty/blank ``path`` returns ``[]`` -- "no
    --changed-files given" is a deliberate no-op the caller decides the
    meaning of. A NON-BLANK path that is not an existing file (missing, or a
    directory) is refused with a clear error, never silently ``[]``: that
    shape is an operator/workflow bug (a bad --changed-files argument), and
    ``changed_modules([])`` treats an empty list as "select nothing", so
    swallowing the bad path would silently produce an empty CI matrix and
    skip every mutation job without ever failing.

    NUL-delimited, not newline-delimited: ``-z`` disables git's C-style
    quoting of paths with unusual bytes, so a non-ASCII or otherwise unusual
    path comes through as its literal bytes instead of a quoted escape
    sequence a newline-based reader would have to un-escape. Tokens are not
    otherwise stripped: only an empty token (e.g. from a trailing NUL) is
    dropped."""
    if not path or not path.strip():
        return []
    p = Path(path)
    if not p.is_file():
        sys.exit(f"--changed-files {path!r} is not a file (missing, or a "
                 f"directory); refusing rather than silently selecting no modules")
    raw = p.read_bytes().decode("utf-8", "surrogateescape")
    return [tok for tok in raw.split("\0") if tok]


def module_owns_changed_path(cfg: dict, name: str, path: str) -> bool:
    """True if ``path`` matches module ``name``'s configured ``tests`` or
    ``mutate`` (minus ``skip``) glob patterns, checked directly against the
    path string -- independent of whether ``path`` is currently tracked.
    ``mutate_files``/``test_files`` only return currently-tracked paths (via
    ``expand``'s ``tracked`` list), so a DELETED source or test file that
    still matches its module's own pattern would otherwise never select that
    module, even though ``git diff --name-only`` reports the deletion."""
    mod = module_cfg(cfg, name)
    if any(fnmatch.fnmatchcase(path, pat) for pat in mod.get("tests", [])):
        return True
    if any(fnmatch.fnmatchcase(path, pat) for pat in mod["mutate"]) and \
            not any(fnmatch.fnmatchcase(path, pat) for pat in mod.get("skip", [])):
        return True
    return False


# -- reverse import closure: what could a changed file affect? ---------------
#
# `build_import_graph` parses (never executes) EVERY git-tracked `.py` file
# in the repo with `ast` and resolves each `import`/`from ... import`
# statement, and each `importlib.import_module("x.y")`/`__import__("x.y")`
# call whose argument is a string literal, to a repo file: `import x.y` and
# `from x.y import z` resolve the dotted name `x.y` to `x/y.py`, or, if that
# is a package, to `x/y/__init__.py`; a relative import (`from . import z` /
# `from ..pkg import z`) resolves against the importing file's own package
# first. A dotted name only resolves to a graph node when its own top-level
# component is a tracked top-level package (`_tracked_roots`, computed fresh
# from the tracked file list every run -- `engine`, `tests`, `checks`,
# `tools`, `experiments`, `dashboard` today, and any future top-level
# directory automatically, with no hand-kept allowlist to fall out of date).
# A stdlib or third-party import's top-level name is never a tracked
# directory, so it is simply never a candidate.
def _tracked_roots(tracked_set: set[str]) -> set[str]:
    """Every top-level package/module name present in `tracked_set` -- the
    first path segment of each tracked `.py` file (or, for a tracked file
    directly at the repo root with no `/`, its name minus `.py`). A dotted
    import only resolves to a graph node when its own top-level component is
    one of these, so a stdlib or third-party import (whose top-level name is
    never a tracked directory) is never mistaken for a repo file. Computed
    fresh from the tracked file list every time `build_import_graph` runs --
    there is NO hand-kept allowlist, so a new top-level package (a future
    `dashboard/`, `scripts/`, ...) is picked up automatically and can never
    be silently missed the way `_GRAPH_ROOTS` (removed by this change) could
    be."""
    roots: set[str] = set()
    for p in tracked_set:
        parts = p.split("/", 1)
        top = parts[0]
        if len(parts) == 1:
            top = top[:-3] if top.endswith(".py") else top
        roots.add(top)
    return roots


def _resolve_dotted(dotted: str, tracked_set: set[str], roots: set[str]) -> str | None:
    """`dotted` (e.g. "engine.v2.foundation" or "checks.phase4_frozen_bridge")
    resolved to a tracked repo file, or None. A package name resolves to its
    `__init__.py`; a dotted name whose top-level component is not in `roots`
    (not a tracked top-level package) is out of scope."""
    if not dotted:
        return None
    parts = dotted.split(".")
    if parts[0] not in roots:
        return None
    as_module = "/".join(parts) + ".py"
    if as_module in tracked_set:
        return as_module
    as_package = "/".join(parts) + "/__init__.py"
    if as_package in tracked_set:
        return as_package
    return None


def _relative_base(rel: str, level: int) -> str | None:
    """The dotted package name `level` steps up from the package containing
    `rel` (`ast.ImportFrom.level`: 1 means "this package", matching Python's
    own relative-import semantics -- the package containing a plain module OR
    a package's own `__init__.py` is the dotted name of its parent directory
    either way). None if `level` climbs above the tracked tree's own root
    (an import this graph cannot resolve)."""
    dir_parts = rel.split("/")[:-1]
    climb = level - 1
    if climb > len(dir_parts):
        return None
    return ".".join(dir_parts[: len(dir_parts) - climb])


def _join_dotted(base: str, tail: str | None) -> str:
    if not tail:
        return base
    return f"{base}.{tail}" if base else tail


def _ancestor_package_inits(dotted: str, tracked_set: set[str]) -> set[str]:
    """Every tracked `__init__.py` of `dotted`'s STRICT ancestor packages
    (excluding `dotted` itself). Python always runs a package's `__init__.py`
    before any of its submodules, for both `import a.b.c` and
    `from a.b import c` -- `build_import_graph` must add those edges too, or
    a change to `a/__init__.py` looks unrelated to code that only ever
    imports `a.b.c` directly."""
    parts = dotted.split(".")
    out: set[str] = set()
    for i in range(1, len(parts)):
        init = "/".join(parts[:i]) + "/__init__.py"
        if init in tracked_set:
            out.add(init)
    return out


def _string_import_target(node: ast.Call) -> str | None:
    """The dotted module name of an `importlib.import_module("x.y")` or
    `__import__("x.y")` call (however `import_module` itself was imported:
    `importlib.import_module(...)`, a bare `import_module(...)` after
    `from importlib import import_module`, or `__import__(...)`), when the
    first positional argument is a string literal. None for every other
    call, including one whose argument is a variable, an f-string, or any
    other computed expression -- those are dynamic and this static graph
    cannot see them (the existing "failed to parse -> select all" fail-safe
    protects the overall selection; this function does not try to)."""
    func = node.func
    is_import_call = (
        (isinstance(func, ast.Attribute) and func.attr == "import_module")
        or (isinstance(func, ast.Name) and func.id in ("import_module", "__import__"))
    )
    if not is_import_call or not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


_FILE_PATH_LOADERS = {
    # call name -> (positional arg index, keyword arg name) of the file path
    "spec_from_file_location": (1, "location"),
    "SourceFileLoader": (1, "path"),
    "run_path": (0, "path_name"),
}


def _call_name(func: ast.expr) -> str | None:
    """The bare called name of a `Call.func` node: `f.attr` for `x.y.f(...)`,
    `f.id` for a bare `f(...)`, else None (e.g. the callee is itself a call
    result or subscript). Used to recognize a loader call regardless of how
    its module was imported (`importlib.util.spec_from_file_location(...)`
    or a bare `spec_from_file_location(...)` after a `from` import)."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _loader_call_path(node: ast.Call) -> tuple[bool, str | None]:
    """(True, literal_path) if `node` calls one of `_FILE_PATH_LOADERS`
    (`importlib.util.spec_from_file_location`, `importlib.machinery.
    SourceFileLoader`, or `runpy.run_path`, however imported) and its file
    path argument is a string literal; (True, None) if it calls one of them
    with a NON-literal (dynamic) or missing path argument -- the caller must
    treat that as the file-path-loader fail-safe; (False, None) if `node`
    calls none of them at all."""
    name = _call_name(node.func)
    if name not in _FILE_PATH_LOADERS:
        return (False, None)
    idx, kw = _FILE_PATH_LOADERS[name]
    arg = node.args[idx] if len(node.args) > idx else None
    if arg is None:
        for keyword in node.keywords:
            if keyword.arg == kw:
                arg = keyword.value
                break
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return (True, arg.value)
    return (True, None)


def _resolve_literal_path(lit: str, tracked_set: set[str]) -> str | None:
    """A literal file-path string (from a loader call's path argument)
    resolved to a tracked file, or None. Only an exact, already-repo-relative
    literal resolves (an optional leading "./" is stripped first) -- this
    does not walk the filesystem or evaluate a computed expression like
    `os.path.join(...)` or `Path(__file__).parent / "x.py"` (those are not
    string literals and never reach this function; `_loader_call_path`
    returns `None` for them, which the fail-safe handles)."""
    norm = lit[2:] if lit.startswith("./") else lit
    return norm if norm in tracked_set else None


def _mutates_sys_path(node: ast.AST) -> bool:
    """True if `node` is a call to `sys.path.insert(...)`,
    `sys.path.append(...)`, or `sys.path.extend(...)`, or an assignment
    (`Assign`/`AugAssign`) whose target is `sys.path` itself or a subscript
    of it (`sys.path[0] = ...`, `sys.path[:0] = [...]`, `sys.path += [...]`).
    Any of these can put an arbitrary, unresolvable directory on the import
    path, after which a plain `import x` elsewhere in the same process may
    resolve to a file this static graph cannot predict -- the caller treats
    this the same as a dynamic loader-call path: the whole file is marked as
    depending on everything."""
    def is_sys_path(expr: ast.expr) -> bool:
        return (isinstance(expr, ast.Attribute) and expr.attr == "path"
                and isinstance(expr.value, ast.Name) and expr.value.id == "sys")

    if isinstance(node, ast.Call):
        func = node.func
        return (isinstance(func, ast.Attribute)
                and func.attr in ("insert", "append", "extend")
                and is_sys_path(func.value))
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    else:
        return False
    for target in targets:
        if is_sys_path(target):
            return True
        if isinstance(target, ast.Subscript) and is_sys_path(target.value):
            return True
    return False


def build_import_graph(tracked: list[str] | None = None) -> dict[str, set[str]]:
    """Static import graph over EVERY git-tracked `.py` file in the repo (no
    hand-kept root allowlist -- see `_tracked_roots`): maps each file to the
    set of tracked files it imports, via `import`/`from ... import`
    (resolved by `_resolve_dotted`) AND via a string literal passed to
    `importlib.import_module(...)` or `__import__(...)` (resolved by
    `_string_import_target` then `_resolve_dotted`), PLUS the tracked
    `__init__.py` of every strict ancestor package of each resolved import
    (`_ancestor_package_inits`) -- Python always runs a package's `__init__.py`
    before any of its submodules, so `import engine.pkg.inner` depends on
    `engine/pkg/__init__.py` even when neither the import statement nor
    `engine/pkg/__init__.py` itself ever names `engine.pkg.inner`. Every
    tracked file is a key, even one with no resolvable imports (an empty
    set), so `module_dependency_closure` can always look it up. Raises
    `SyntaxError` (via `ast.parse`) on the first file that fails to parse --
    a real syntax error in the current tree, never swallowed into a
    silently partial graph; `changed_modules` treats that as "select every
    module"."""
    tracked = tracked if tracked is not None else [
        p for p in _tracked(["."]) if p.endswith(".py")]
    tracked_set = set(tracked)
    roots = _tracked_roots(tracked_set)
    graph: dict[str, set[str]] = {rel: set() for rel in tracked}
    for rel in tracked:
        source = (REPO / rel).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            raise SyntaxError(f"{rel}: {exc}") from exc
        edges = graph[rel]
        dynamic = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = _resolve_dotted(alias.name, tracked_set, roots)
                    if target:
                        edges.add(target)
                    edges |= _ancestor_package_inits(alias.name, tracked_set)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = _relative_base(rel, node.level)
                    dotted = _join_dotted(base, node.module) if base is not None else None
                else:
                    dotted = node.module or ""
                if dotted:
                    target = _resolve_dotted(dotted, tracked_set, roots)
                    if target:
                        edges.add(target)
                    edges |= _ancestor_package_inits(dotted, tracked_set)
                    for alias in node.names:
                        sub = _resolve_dotted(_join_dotted(dotted, alias.name), tracked_set, roots)
                        if sub:
                            edges.add(sub)
            elif isinstance(node, ast.Call):
                dotted_str = _string_import_target(node)
                if dotted_str:
                    target = _resolve_dotted(dotted_str, tracked_set, roots)
                    if target:
                        edges.add(target)
                    edges |= _ancestor_package_inits(dotted_str, tracked_set)
                    continue
                is_loader, literal = _loader_call_path(node)
                if is_loader:
                    if literal is None:
                        dynamic = True
                    else:
                        resolved = _resolve_literal_path(literal, tracked_set)
                        if resolved:
                            edges.add(resolved)
                elif _mutates_sys_path(node):
                    dynamic = True
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                if _mutates_sys_path(node):
                    dynamic = True
        if dynamic:
            edges |= tracked_set - {rel}
    return graph


def _conftest_ancestors(rel: str, tracked_set: set[str]) -> set[str]:
    """Every tracked `conftest.py` in `rel`'s own directory or any ancestor
    directory up to the repo root -- pytest applies every one of these to a
    test file, so a change reachable only through a conftest.py's OWN
    imports (not the test file's) can still affect that test's behavior.
    `rel` is expected to be a test file path; a `conftest.py` at the repo
    root itself is included when tracked."""
    parts = rel.split("/")[:-1]
    out: set[str] = set()
    for i in range(len(parts), -1, -1):
        candidate = "/".join(parts[:i] + ["conftest.py"]) if parts[:i] else "conftest.py"
        if candidate in tracked_set:
            out.add(candidate)
    return out


def _closure_roots(mod: dict, tracked_set: set[str]) -> set[str]:
    """Module `mod`'s own `tests`/`mutate` (minus `skip`) glob patterns,
    expanded against `tracked_set`, PLUS every tracked `conftest.py` that
    applies to one of its `tests` files (`_conftest_ancestors`) -- a
    conftest.py's own imports become part of the closure even though the
    conftest file itself never matches a `mutate`/`tests` pattern. Unlike
    `expand`, a pattern matching nothing here is NOT an error: this powers
    dependency-closure roots, which must tolerate a `tracked_set` that does
    not happen to contain one of the module's configured files (e.g. a
    unit-test fixture, or a graph built from a narrower tree) without
    raising -- ownership (`module_owns_changed_path`) is unaffected either
    way, since it never consults a tracked list."""
    test_pats, mutate_pats = mod.get("tests", []), mod["mutate"]
    skip_pats = mod.get("skip", [])
    out: set[str] = set()
    test_hits: set[str] = set()
    for p in tracked_set:
        if any(fnmatch.fnmatchcase(p, pat) for pat in test_pats):
            out.add(p)
            test_hits.add(p)
            continue
        if any(fnmatch.fnmatchcase(p, pat) for pat in mutate_pats) and \
                not any(fnmatch.fnmatchcase(p, pat) for pat in skip_pats):
            out.add(p)
    for t in test_hits:
        out |= _conftest_ancestors(t, tracked_set)
    return out


def module_dependency_closure(cfg: dict, name: str, graph: dict[str, set[str]],
                              tracked_set: set[str] | None = None) -> set[str]:
    """Every file module `name` transitively imports: the closure of
    `graph`'s edges starting from `name`'s own `mutate` files plus `tests`
    files (`_closure_roots`). This is a FORWARD dependency set -- what
    `name` relies on -- and `changed_modules` uses it in REVERSE: a changed
    path in this set means code `name`'s own tests exercise has changed, so
    `name`'s cached mutation verdict may now be stale even though `name`
    does not OWN that path."""
    tracked_set = tracked_set if tracked_set is not None else set(graph)
    seen: set[str] = set()
    stack = list(_closure_roots(module_cfg(cfg, name), tracked_set))
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        stack.extend(graph.get(f, ()))
    return seen


def changed_modules(cfg: dict, names: list[str], changed: list[str], *,
                    graph: dict[str, set[str]] | None = None) -> list[str]:
    """The subset of ``names`` (already ``--only``-filtered) a changed-file
    list selects, under the "select ALL unless proven safe to skip" rule
    documented above. An empty ``changed`` selects nothing -- a PR with no
    diff is not "select everything"; that is the one intentional
    zero-selection default. Any other unrecognized path selects every name
    in ``names``, never zero. ``graph`` is normally left ``None`` (built
    fresh via `build_import_graph`); tests pass a small synthetic graph to
    exercise the closure rule without depending on this repo's real files."""
    changed_set = set(changed)
    if not changed_set:
        return []
    enabled = enabled_modules(cfg)
    if graph is None:
        try:
            graph = build_import_graph()
        except Exception as exc:
            print(f"[mutation_pilot] import graph build failed "
                  f"({type(exc).__name__}: {exc}); selecting every enabled "
                  f"module for the whole changed-file list", flush=True)
            return list(names)
    tracked_set = set(graph)
    dep_sets = {n: module_dependency_closure(cfg, n, graph, tracked_set) for n in enabled}
    owned: set[str] = set()
    for path in changed_set:
        selectors = {n for n in enabled
                     if module_owns_changed_path(cfg, n, path) or path in dep_sets[n]}
        if selectors:
            owned.update(selectors)
            continue
        if is_inert_changed_path(cfg, path):
            continue
        return list(names)
    return [n for n in names if n in owned]


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

    also_copy = [t for t in tops if t not in ("engine", "tests")] or ["tests"]
    setup = mutmut_config_text(defaults, mutate, tests, also_copy,
                               debug=stats_debug_enabled(name))
    (work / "setup.cfg").write_text(setup)
    return work


def mutmut_config_text(defaults: dict, mutate: list[str], tests: list[str],
                       also_copy: list[str], *, debug: bool = False) -> str:
    """The generated ``[mutmut]`` section for a module's work copy.

    ``debug`` is mutmut's supported setting, and it means full-run verbosity:
    when true, mutmut echoes every mutant's child pytest output for the whole
    run, not just the clean/stats collection step. It is expensive, so it stays
    off unless ``stats_debug_enabled`` opts the run in (an explicit environment
    value, or the ops_legacy CI shard).
    ops_catalog_state now shares that CI-only default.
    """
    def lines(key: str, values: list[str]) -> str:
        return f"{key} =\n" + "".join(f"    {v}\n" for v in values)

    setup = ("[mutmut]\n"
             + lines("source_paths", ["engine"])
             + lines("only_mutate", mutate)
             # mutmut copies source_paths and tests/ into mutants/ by itself;
             # every other copied tree must be listed to be importable there.
             + lines("also_copy", also_copy)
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
    if debug:
        setup += "debug = true\n"
    return setup


# -- commands ----------------------------------------------------------------

def cmd_list(cfg: dict, _args) -> int:
    for name, mod in cfg["modules"].items():
        state = f"EXCLUDED: {mod['excluded']}" if mod.get("excluded") else f"why: {mod['why']}"
        print(f"{name}\n  mutate: {' '.join(mod['mutate'])}\n  tests:  "
              f"{' '.join(mod.get('tests', []))}\n  {state}")
    return 0


def cmd_matrix(cfg: dict, args) -> int:
    """JSON list of the modules a CI run covers: every enabled module, or the
    comma-separated ``--only`` subset (an unknown or excluded name is an error).
    A value that names NOTHING (``--only ,``) is an error too, never an empty
    matrix: only an absent/blank ``--only`` means "every enabled module", so a
    mistyped subset cannot quietly reduce the run to zero modules (GitHub
    renders a dispatched empty input as the empty string)."""
    names = enabled_modules(cfg)
    if args.only and args.only.strip():
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        if not wanted:
            sys.exit(f"--only {args.only!r} names no module; pass an empty --only for "
                     f"every enabled module ({', '.join(names)})")
        bad = [n for n in wanted if n not in names]
        if bad:
            sys.exit(f"not enabled modules: {bad}; enabled: {', '.join(names)}")
        names = [n for n in names if n in wanted]
    changed_files = getattr(args, "changed_files", "")
    if changed_files and changed_files.strip():
        names = changed_modules(cfg, names, read_changed_files(changed_files))
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


# Sentinel `cmd_run` reports for a run THIS SCRIPT stopped on purpose because
# its time budget ran out -- the SAME value the CI export step's own fallback
# (`cat mutation-rc 2>/dev/null || echo -1`) already writes when GitHub's step
# timeout kills the whole step before the "echo $?" line can run. Reusing -1
# 124, not -1: the exit code is written to a shell file and re-read as text
# (mutation-rc in the workflow), where -1 is already claimed for "no rc file at
# all" (a missing or GitHub-killed step, via `|| echo -1`) -- an ambiguity a
# clean stop must not share. 124 matches coreutils `timeout`'s own convention
# for "the command was still running when the time budget expired" and gives
# mutation_results.py's TIME_BUDGET_STOP handling (partial results, a withheld
# score, the module named in `failed_run_modules`, exempted from the gate only
# in incremental mode) an unambiguous code of its own to key off.
TIME_BUDGET_STOP_RC = 124
# How long to let mutmut's own KeyboardInterrupt unwind (stop_all_workers +
# shutdown/drain of the fork server, per mutmut/__main__.py's `run` command)
# after SIGINT before concluding it is stuck and escalating to SIGKILL.
SIGINT_GRACE_SECONDS = 180


def _run_with_time_budget(cmd: list[str], cwd: Path, env: dict, budget_s: float) -> int:
    """Run ``cmd`` for at most ``budget_s`` seconds; return its exit code if it
    finishes in time, or ``TIME_BUDGET_STOP_RC`` if this function had to stop it.

    mutmut's ``run`` command catches ``KeyboardInterrupt`` (SIGINT) and does a
    clean shutdown: it stops in-flight workers, drains what already finished
    and returns normally (verified in mutmut 3.8's ``mutmut/__main__.py``,
    ``except KeyboardInterrupt: ... runner.stop_all_workers() / finally:
    runner.shutdown()``). Mutants already decided before the stop were already
    flushed to their ``.meta`` file the moment each one finished
    (``_register_mutant_result`` calls ``mutation_data.save()`` per result, not
    only at the end), so a clean stop loses at most the handful of mutants that
    were still in flight. SIGTERM is NOT used for this: mutmut installs no
    handler for it, so Python's default SIGTERM action kills the process
    immediately with no unwind -- indistinguishable from a hard kill, and
    exactly what relying on GitHub's own step timeout would deliver instead of
    a clean stop.
    """
    start = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)
    try:
        return proc.wait(timeout=budget_s)
    except subprocess.TimeoutExpired:
        pass
    elapsed = time.monotonic() - start
    print(f"[mutation_pilot] time budget of {budget_s:.0f}s reached after {elapsed:.0f}s; "
          f"sending SIGINT for a clean stop (mutants already decided stay on disk)", flush=True)
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        real_rc = proc.wait(timeout=SIGINT_GRACE_SECONDS)
        print(f"[mutation_pilot] mutmut stopped cleanly after SIGINT (its own exit code "
              f"{real_rc}); reporting {TIME_BUDGET_STOP_RC} (TIME_BUDGET_STOP, not a tool error)",
              flush=True)
    except subprocess.TimeoutExpired:
        print(f"[mutation_pilot] mutmut did not exit within {SIGINT_GRACE_SECONDS}s of SIGINT; "
              f"escalating to SIGKILL", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    return TIME_BUDGET_STOP_RC


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
    # and --max-children is the parallelism anyway. The environment mutmut's
    # pytest child inherits is left as-is: mutmut sets MUTANT_UNDER_TEST itself,
    # and pytest sets PYTEST_CURRENT_TEST, so filtering the parent here would
    # not stop a child from inheriting them -- and doing so has no supported
    # basis. When the ops_legacy CI shard's stats step fails, the driver has
    # already enabled mutmut's debug setting (see stats_debug_enabled), which
    # echoes that swallowed pytest trace for the whole run; set
    # MUTATION_PILOT_DEBUG=1 anywhere else to do the same. Either way it is
    # verbosity, not a rerun: the exit code the job gates on is unchanged.
    # ops_catalog_state now shares that CI-only default, so the same echo
    # covers its shard too.
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    if args.time_budget_seconds:
        rc = _run_with_time_budget(cmd, work, env, args.time_budget_seconds)
    else:
        rc = subprocess.run(cmd, cwd=work, env=env).returncode
    # On a first run mutants/ did not exist before mutmut; record the digest now
    # so the next run compares against this one.
    reset_on_test_change(work, [], digest)
    return rc


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
    p.add_argument("--changed-files", default="", metavar="PATH",
                   help="path to a NUL-delimited changed-file list (git diff -z --no-renames "
                        "--name-only); when given, a path selects every enabled module that "
                        "owns it OR transitively depends on it (a static ast import-graph "
                        "closure), a path on the inert allowlist selects nothing, and any "
                        "other path selects every enabled module (never zero on an "
                        "unrecognized change, and never zero if the import graph itself "
                        "cannot be built); a path that is not an existing file is a hard "
                        "failure, not a silent empty selection. Omitted/blank: unchanged "
                        "behavior.")
    p = sub.add_parser("count")
    p.add_argument("modules", nargs="*")
    p = sub.add_parser("run")
    p.add_argument("module")
    p.add_argument("globs", nargs="*", help="mutant-name globs (default: all)")
    p.add_argument("--max-children", type=int, default=None)
    p.add_argument("--time-budget-seconds", type=float, default=None,
                   help="stop mutmut cleanly (SIGINT) after this many seconds instead of "
                        "letting it run unbounded; returns TIME_BUDGET_STOP_RC (124)")
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
