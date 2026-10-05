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


def config_hash(cfg: dict, name: str) -> str:
    """sha256 hex digest of the canonical JSON of exactly what affects module
    ``name``'s mutation run: ``cfg["defaults"]`` plus that module's own
    ``[modules.<name>]`` section (via the existing ``module_cfg`` lookup, which
    already exits non-zero with a message for an unknown module -- reused
    here, not re-parsed by hand). Canonical = ``json.dumps(sort_keys=True,
    separators=(",", ":"))``, so key order never perturbs the digest.

    This is the CI cache-key ingredient (see cmd_config_hash / the
    ``config-hash`` subcommand): it changes only when THIS module's own
    section or the shared ``[defaults]`` table changes, never when a
    different module's section changes or a new module is added, so one
    module's cache is never busted by another module's edit.
    """
    mod = module_cfg(cfg, name)
    resolved = {"defaults": cfg["defaults"], "module": mod}
    blob = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def cmd_config_hash(cfg: dict, args) -> int:
    print(config_hash(cfg, args.module))
    return 0


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
#      `.py` file (including `conftest.py` closure roots and the one literal
#      `importlib.import_module("x.y")` shape -- `__import__`, in any form,
#      is always dynamic now), of the imports
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
# in the repo with `ast`, resolves each file's REAL import edges, and adds a
# file-wide catch-all (`tracked_set - {rel}`) ONLY for a file with one of
# the unresolved constructs its three classifiers track:
#   - an unresolved import/load attempt by
#     `_has_unresolved_import_attempt` (non-literal/aliased `importlib`,
#     runpy, `__import__`, loader constructs, bare exec/eval, a conftest.py's
#     non-literal or annotated top-level `pytest_plugins`, ...);
#   - an import-path mutation to an unproven target by
#     `_has_unresolved_sys_path_mutation`: a `sys.path` write (`insert`,
#     `append`, or `extend`; a rebinding of the list or a write to one of
#     its subscripts) -- however `sys` is bound, `from sys import path`
#     included -- or a `site.addsitedir`/`monkeypatch.syspath_prepend`,
#     unless the target is provably the file's own repository root;
#   - a process launch that `_has_unresolved_process_launch` catches on AST
#     shape alone (an `os.system`/`os.exec*`/`os.spawn*` call or mere
#     reference, `subprocess.getoutput`/`getstatusoutput`, or a supported
#     `subprocess` exec function used as a value) or a `subprocess` launch
#     `_subprocess_targets` cannot prove is either a non-Python command or a
#     tracked Python script/`-m` module target (variable argv or
#     interpreter, `-c`, `-m pytest`, an untracked script or module, or an
#     unsupported interpreter flag).
#
# All three categories contribute both to the whole-file catch-all and to
# `unresolved_import_files`'s taint set, so a test that reaches one of these
# files only through real (precise) import edges stays in the #155 failsafe.
# Every other file -- including one that only READS `sys.path`, imports
# `site` without calling it, mentions `PYTHONPATH`, imports `subprocess`
# only for a provably literal NON-Python command like `git`, or touches
# `pkgutil` or `compile` -- keeps just its real ast-resolved edges: a plain
# `import x.y [as z]`/`from x.y import z` (including a relative import),
# resolved to a repo file the same way as before (`import x.y`/`from x.y
# import z` resolve the dotted name `x.y` to `x/y.py`, or, if that is a
# package, to `x/y/__init__.py`; a relative import resolves against the
# importing file's own package first; a dotted name only resolves to a graph
# node when its own top-level component is a tracked top-level package --
# `_tracked_roots`, computed fresh from the tracked file list every run,
# with no hand-kept allowlist to fall out of date -- so a stdlib or
# third-party import's top-level name is simply never a candidate); the
# single literal call `importlib.import_module("<absolute.name>")`
# (`_allowed_import_module_call`); or a plain, unannotated
# `pytest_plugins = [...]` assignment of string literals, at a conftest.py's
# top level only (`_pytest_plugins_targets`). `_is_dynamic_file` remains the
# older, broader ALLOWLIST classifier (and the exact list issue #42 is
# about), but `build_import_graph` no longer calls it: the narrower
# unresolved-construct triggers above replaced its catch-all condition.
#
# This is conservative for the known unresolved constructs above, NOT sound
# in general: see https://github.com/yshewchuk/investment-validation/issues/42
# for constructs it does not recognize at all (string-target
# monkeypatch.setattr/mock.patch, pytest.importorskip, getattr-based
# imports of importlib/sys, __import__ via globals()/builtins, asyncio
# subprocess-exec calls, __path__/sys.meta_path edits, pytest_plugins
# outside a conftest.py or under an `if`, and `from pkg import *`
# re-exports). A test file that depends on repository code only through one
# of those, or any other runtime loading this analysis doesn't track, may be
# omitted from a PR's narrowed selection, with the full suite on every push
# to `main` and the weekly scheduled mutation run as the backstop.
# `tests/conftest.py` is a closure root for every test file
# (`_conftest_ancestors`), so a PR that changes `tests/conftest.py` itself
# still selects broadly -- deliberate, since conftest.py is a genuinely
# shared input. `module_dependency_closure` walks a file's REAL edges
# (`build_import_graph`'s `.precise`) rather than treating its catch-all as
# something to expand further, so reaching an unresolved-import file no
# longer cascades into "every enabled module, for every unrelated
# single-module change" the way it used to -- see that function's own
# docstring for the mechanism.
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


_DYNAMIC_MODULES = {"site", "runpy", "subprocess", "multiprocessing", "pkgutil"}
# "importlib" and "sys" are handled separately below: importing them plainly
# is fine (needed for the one allowed import_module shape, and `import sys`
# by itself is inert on its own) -- only specific attributes/names of theirs
# are banned.

_OS_DYNAMIC_ATTRS = {
    "system", "popen",
    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
}

_DYNAMIC_NAMES = {
    "__import__", "syspath_prepend", "addsitedir", "PYTHONPATH",
    "exec", "eval", "compile", "spec_from_file_location", "SourceFileLoader",
}


def _bound_aliases(tree: ast.Module, module: str) -> set[str]:
    """Every name this file binds to the top-level module `module` via a
    plain `import module` or `import module as X`, wherever it appears (not
    just at the top level) -- so `import sys as s` is recognized as binding
    `sys` to `s`, and a later `s.path` reference is caught exactly like
    `sys.path` would be. Does not follow `from module import x`; callers
    check those `ast.ImportFrom` nodes directly instead."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _looks_like_import_module_call(func: ast.expr) -> bool:
    """True if `func` (a `Call.func`) is an attribute access named
    `import_module` (`importlib.import_module`, an aliased base, ...) or a
    bare name `import_module` (after any `from ... import import_module`).
    `_is_dynamic_file` uses this to recognize "this call is an ATTEMPT at
    the one allowed shape" so it can fail the whole file safe on every
    attempt that isn't EXACTLY that shape, rather than silently treating it
    like an unrelated function call."""
    return (isinstance(func, ast.Attribute) and func.attr == "import_module") \
        or (isinstance(func, ast.Name) and func.id == "import_module")


def _allowed_import_module_call(node: ast.Call) -> str | None:
    """The literal absolute dotted target of `node`, if `node` is EXACTLY
    `importlib.import_module("<literal>")`: an unaliased attribute access on
    a bare `importlib` name, a single positional string-literal argument
    with no leading dot (an absolute name), and no other positional or
    keyword argument at all -- a `package=` keyword, or a second positional
    argument however spelled, means the literal could be package-relative,
    which this never attempts to resolve. None for every other shape,
    including one that merely LOOKS like this call (an aliased `importlib`,
    a bare `import_module` after a `from` import, `importlib.__import__`,
    ...) -- `_is_dynamic_file` is what turns a None here into a whole-file
    fail-safe, not this function."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "import_module"
            and isinstance(func.value, ast.Name) and func.value.id == "importlib"):
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
            and not arg.value.startswith("."):
        return arg.value
    return None


def _is_dynamic_file(tree: ast.Module) -> bool:
    """True if `tree` contains ANY construct outside the narrow allowlist
    `build_import_graph` resolves exactly (a plain import, the one literal
    `import_module` shape, or a conftest.py's plain `pytest_plugins`
    assignment -- checked separately). This tests AST node types and names
    DIRECTLY, never a call-shape pattern match, so it trips the moment the
    file references, in ANY form -- an import, an alias, an attribute
    access, or a bare name -- any of:
      - `sys.path`, however `sys` got bound (`_bound_aliases`), including
        `from sys import path`;
      - the modules `site`, `runpy`, `subprocess`, `multiprocessing`,
        `pkgutil` (importing one AT ALL is enough, used or not);
      - a dynamic-exec `os` function (`os.system`, `os.exec*`, `os.spawn*`,
        `os.popen`), however `os` got bound;
      - `importlib` used any way OTHER than the one literal `import_module`
        shape (`_allowed_import_module_call`) -- including an aliased
        `importlib` import, any `from importlib import ...`, and any other
        `importlib.*` attribute (`importlib.util`, `importlib.reload`,
        `importlib.__import__`, ...);
      - the standalone names `__import__`, `syspath_prepend`, `addsitedir`,
        `PYTHONPATH`, `exec`, `eval`, `compile`, `spec_from_file_location`,
        `SourceFileLoader` -- as an import, an attribute, or a bare name.
    Also true for an ANNOTATED `pytest_plugins` assignment ANYWHERE
    (`ast.AnnAssign`) -- only a plain, unannotated one, at a conftest.py's
    top level, is the allowed shape (checked by `_pytest_plugins_targets`,
    called only for conftest.py).

    `subprocess` fails the whole file safe HERE with no narrower attempt:
    this broad allowlist never resolves argv. `build_import_graph` does NOT
    call this function; it uses `_subprocess_targets`, which resolves a
    proved-Python launch of a tracked script or `-m` module to its precise
    target edge(s), clears a literal provably non-Python command, and keeps
    the catch-all for everything else. Loader constructs
    (`spec_from_file_location`, `SourceFileLoader`, `runpy`) get the same
    treatment in both: no static path evaluation is attempted for any of
    them, literal or not."""
    sys_aliases = _bound_aliases(tree, "sys") | {"sys"}
    os_aliases = _bound_aliases(tree, "os") | {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _DYNAMIC_MODULES:
                    return True
                if top == "importlib" and alias.asname is not None:
                    return True  # an aliased `importlib` can't reach the one allowed shape
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0] if module else ""
            if top in _DYNAMIC_MODULES or top == "importlib":
                return True  # every from-importlib import, no exceptions
            if top == "sys" and any(a.name == "path" for a in node.names):
                return True
            if top == "os" and any(a.name in _OS_DYNAMIC_ATTRS for a in node.names):
                return True
            if any(a.name in _DYNAMIC_NAMES for a in node.names):
                return True
        elif isinstance(node, ast.Attribute):
            if node.attr in _DYNAMIC_NAMES or node.attr in _DYNAMIC_MODULES:
                return True
            if node.attr == "path" and isinstance(node.value, ast.Name) \
                    and node.value.id in sys_aliases:
                return True
            if node.attr in _OS_DYNAMIC_ATTRS and isinstance(node.value, ast.Name) \
                    and node.value.id in os_aliases:
                return True
            if isinstance(node.value, ast.Name) and node.value.id == "importlib" \
                    and node.attr != "import_module":
                return True
        elif isinstance(node, ast.Name):
            if node.id in _DYNAMIC_NAMES or node.id in _DYNAMIC_MODULES:
                return True
        elif isinstance(node, ast.Constant) and node.value == "PYTHONPATH":
            return True
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "pytest_plugins":
                return True
        elif isinstance(node, ast.Call) and _looks_like_import_module_call(node.func):
            if _allowed_import_module_call(node) is None:
                return True
    return False


def _pytest_plugins_targets(tree: ast.Module) -> tuple[bool, list[str] | None]:
    """(found, literal_dotted_names) for a module-level, UNANNOTATED
    `pytest_plugins = ...` assignment -- pytest imports every name in this
    list as a plugin BEFORE collecting or running any test, an execution
    path `build_import_graph`'s ordinary import/call walk never sees on its
    own. `found` is False if `tree`'s top level has no such assignment
    (including only an augmented one, `pytest_plugins += [...]`, which this
    never resolves -- see below). When found, `literal_dotted_names` is the
    assigned names IF the value is a literal string or a literal list/tuple
    of literal strings; None if it is anything else (a variable, a list
    containing anything non-literal, a computed expression, an `AugAssign`,
    ...) -- `build_import_graph` marks the WHOLE FILE dynamic for a None
    here, the same as any other unresolvable construct. Only `tree`'s
    TOP-LEVEL statements are scanned -- a `pytest_plugins` assigned inside a
    function or an `if` block is not pytest's own collection hook either.
    An ANNOTATED assignment (`pytest_plugins: list[str] = [...]`) is not
    handled here at all -- it is an `ast.AnnAssign`, not an `ast.Assign`,
    and `_is_dynamic_file` catches it separately, unconditionally, since it
    is outside the allowlist's exact shape regardless of literal-ness."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytest_plugins" for t in node.targets):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return (True, [value.value])
            if isinstance(value, (ast.List, ast.Tuple)):
                names: list[str] = []
                for elt in value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        names.append(elt.value)
                    else:
                        return (True, None)
                return (True, names)
            return (True, None)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "pytest_plugins":
            return (True, None)
    return (False, None)


_SUBPROCESS_FUNCS = {"run", "call", "Popen", "check_call", "check_output"}
# `getoutput`/`getstatusoutput` always run their command through a shell, so
# unlike the five above their argv is never resolvable; the two sets together
# are every `subprocess` name `_has_unresolved_process_launch` tracks.
_SUBPROCESS_SHELL_ATTRS = frozenset({"getoutput", "getstatusoutput"})
_SUBPROCESS_LAUNCH_ATTRS = frozenset(_SUBPROCESS_FUNCS) | _SUBPROCESS_SHELL_ATTRS


def _is_python_executable_base(name: str) -> bool:
    """True for a literal `python`/`python3`/`python3.11` basename (`git`,
    `bash` and `python-config` are False: provably non-Python commands)."""
    if not name.startswith("python"):
        return False
    rest = name[len("python"):]
    return rest == "" or all(c.isdigit() or c == "." for c in rest)


def _subprocess_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """(module aliases, directly-imported function names) for the `subprocess`
    module: `import subprocess [as sp]` binds `subprocess`/`sp` for
    `sp.run(...)`-style attribute calls; `from subprocess import run as
    run_process` binds the bare name `run_process` for direct calls. Only the
    five exec functions are followed."""
    aliases: set[str] = set()
    funcs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and not node.level \
                and (node.module or "") == "subprocess":
            for alias in node.names:
                if alias.name in _SUBPROCESS_FUNCS:
                    funcs.add(alias.asname or alias.name)
    return aliases, funcs


_PYTHON_NO_VALUE_FLAGS = frozenset({
    "-b", "-B", "-d", "-E", "-h", "-?", "-i", "-I", "-O", "-OO", "-P",
    "-q", "-R", "-s", "-S", "-u", "-v", "-V", "-x",
    "--help", "--help-env", "--help-xoptions", "--help-all", "--version",
})
_PYTHON_VALUE_FLAGS = frozenset({"-W", "-X", "-Q", "--check-hash-based-pycs"})
_PYTHON_VALUE_PREFIXES = ("-W", "-X", "-Q")


def _is_python_interpreter_expr(node: ast.expr, sys_aliases: set[str]) -> bool:
    """True for an argv[0] this scan can PROVE launches Python: a literal
    `python`/`python3`/`python3.11` basename (with or without a directory),
    or an attribute access `<sys alias>.executable` (the form the real
    `engine/v2/ops/executor.py` launch uses)."""
    if isinstance(node, ast.Attribute) and node.attr == "executable" \
            and isinstance(node.value, ast.Name) and node.value.id in sys_aliases:
        return True
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and _is_python_executable_base(node.value.rsplit("/", 1)[-1]))


def _literal_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _resolve_script_target(text: str, tracked_set: set[str]) -> str | None:
    if text in tracked_set:
        return text
    normalized = os.path.normpath(text)
    return normalized if normalized in tracked_set else None


_SAFE_NON_PYTHON_COMMANDS = frozenset({"git", "npm", "curl", "free"})


def _module_run_targets(dotted: str, tracked_set: set[str],
                        roots: set[str]) -> set[str] | None:
    """Edges for a proved-Python `-m dotted` launch, or None if unresolved.

    A regular module target behaves exactly as `_resolve_dotted` plus
    `_ancestor_package_inits` always did: its own file plus every tracked
    strict ancestor package `__init__.py`. A PACKAGE target runs
    `__init__.py` and then `__main__.py`, so both are edges when
    `__main__.py` is tracked; a package with no tracked `__main__.py`
    cannot be resolved (`python -m` would find no `__main__` to execute),
    so the launch is unresolved rather than silently missing whatever
    `__main__` would have run. A dotted name that resolves to nothing at
    all is unresolved, as before."""
    target = _resolve_dotted(dotted, tracked_set, roots)
    if target is None:
        return None
    edges = _ancestor_package_inits(dotted, tracked_set)
    if not target.endswith("/__init__.py"):
        return edges | {target}
    main = target[: -len("__init__.py")] + "__main__.py"
    if main not in tracked_set:
        return None
    return edges | {target, main}


def _python_argv_targets(argv: ast.expr | None, tracked_set: set[str],
                         roots: set[str], sys_aliases: set[str]) -> tuple[set[str], bool]:
    """(precise edges, unresolved) for one subprocess argv expression.

    A literal argv whose first element is on the explicit
    `_SAFE_NON_PYTHON_COMMANDS` allowlist (`git`, `npm`, `curl`, `free`,
    the direct non-Python commands this repo launches) contributes no edge
    and is not unresolved. Any other non-Python first element is
    unresolved, including an unknown literal command and a shell
    interpreter (`sh`, `bash`) whose own command line is never analyzed. A
    proved-Python first element (`python`/`python3`/versioned basename or
    `<sys alias>.executable`) resolves exactly one literal script, or one
    literal `-m dotted.name` target through `_module_run_targets` (a
    regular module file, or a package's `__init__.py` plus `__main__.py`,
    each plus tracked ancestor package `__init__.py` files), through the
    interpreter's supported flags (`-u`, `-X utf8`, `-W ignore`, `--`,
    ...). Everything else is unresolved: a non-literal argv or first
    element, `-c`, `-m` of an untracked module or a package without a
    tracked `__main__.py`, a script that is not tracked, or an interpreter
    flag this scan does not support. An unresolved launch keeps the
    whole-file catch-all."""
    if not (isinstance(argv, ast.List) and argv.elts):
        return set(), True
    first = argv.elts[0]
    if not _is_python_interpreter_expr(first, sys_aliases):
        if _literal_str(first) in _SAFE_NON_PYTHON_COMMANDS:
            return set(), False
        return set(), True
    elts = argv.elts
    i = 1
    while i < len(elts):
        text = _literal_str(elts[i])
        if text is None:
            return set(), True
        if text == "--":
            i += 1
            break
        if text == "-" or text == "-c" or text.startswith("-c"):
            return set(), True
        if text == "-m" or text.startswith("-m"):
            if text == "-m":
                i += 1
                if i >= len(elts):
                    return set(), True
                dotted = _literal_str(elts[i])
                if dotted is None:
                    return set(), True
            else:
                dotted = text[2:]
            edges = _module_run_targets(dotted, tracked_set, roots)
            if edges is None:
                return set(), True
            return edges, False
        if text in _PYTHON_VALUE_FLAGS:
            if i + 1 >= len(elts) or _literal_str(elts[i + 1]) is None:
                return set(), True
            i += 2
            continue
        if text in _PYTHON_NO_VALUE_FLAGS:
            i += 1
            continue
        if any(text.startswith(p) and len(text) > len(p)
               for p in _PYTHON_VALUE_PREFIXES) \
                or text.startswith("--check-hash-based-pycs="):
            i += 1
            continue
        if text.startswith("-"):
            return set(), True
        target = _resolve_script_target(text, tracked_set)
        if target is None:
            return set(), True
        return {target}, False
    if i < len(elts):
        text = _literal_str(elts[i])
        if text is None:
            return set(), True
        target = _resolve_script_target(text, tracked_set)
        if target is not None:
            return {target}, False
    return set(), True


def _subprocess_launch_unresolved(node: ast.Call) -> bool:
    """True if a supported `subprocess` call's own keywords hide or redirect
    what it launches, before any argv analysis: an `executable=` override
    replaces the program `_python_argv_targets` would have proved, an active
    or non-literal `shell=` hands the command to a shell instead of
    executing argv directly, and a `**kwargs` expansion could carry either.
    An explicit literal `shell=False` is the one accepted shell spelling;
    absent keywords are fine."""
    for kw in node.keywords:
        if kw.arg is None or kw.arg == "executable":
            return True
        if kw.arg == "shell" and not (
                isinstance(kw.value, ast.Constant) and kw.value.value is False):
            return True
    return False


def _subprocess_targets(tree: ast.Module, tracked_set: set[str],
                        roots: set[str]) -> tuple[set[str], bool]:
    """(precise target edges, unresolved) for every `subprocess` launch in
    `tree` (the five exec functions, however bound -- `import subprocess [as
    sp]` or `from subprocess import run as launch`), each judged by
    `_python_argv_targets` once `_subprocess_launch_unresolved` clears its
    keywords. A call with an `executable=` override, an active or
    non-literal `shell=`, or a `**kwargs` that could carry either is
    unresolved and contributes no edge: its argv no longer proves what
    runs. An explicit `shell=False` keeps the normal argv analysis, as does
    an absent keyword. `unresolved` is True if ANY launch on its own cannot
    be proven safe, in which case `build_import_graph` adds the whole-file
    catch-all on top of whatever precise edges were found."""
    aliases, funcs = _subprocess_bindings(tree)
    sys_aliases = _bound_aliases(tree, "sys") | {"sys"}
    edges: set[str] = set()
    unresolved = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr not in _SUBPROCESS_FUNCS or not isinstance(func.value, ast.Name) \
                    or func.value.id not in aliases:
                continue
        elif not (isinstance(func, ast.Name) and func.id in funcs):
            continue
        if _subprocess_launch_unresolved(node):
            unresolved = True
            continue
        argv = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg in ("args", "cmd")), None)
        call_edges, call_unresolved = _python_argv_targets(
            argv, tracked_set, roots, sys_aliases)
        edges |= call_edges
        unresolved = unresolved or call_unresolved
    return edges, unresolved


def _launch_module_aliases(tree: ast.Module, module: str) -> set[str]:
    """Every local name that can reach the top-level `module` (`os` or
    `subprocess`): the module name itself, `import module as alias`, and --
    because `import os.path` with no `as` binds the top-level name `os` --
    a bare dotted import of one of its submodules. A dotted import WITH an
    `as` binds only the submodule, which cannot reach `module`'s own
    attributes, so it is not included."""
    aliases = {module}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    aliases.add(alias.asname or alias.name)
                elif alias.asname is None and alias.name.startswith(module + "."):
                    aliases.add(module)
    return aliases


def _has_unresolved_process_launch(tree: ast.Module) -> bool:
    """True if `tree` launches a process in a shape this scan cannot prove
    safe, judged on AST shape alone (never on argv):

      - `os.system`, `os.popen`, and every `os.exec*`/`os.spawn*` API
        (`_OS_DYNAMIC_ATTRS`), whether called directly -- their argv is
        never analyzed here -- or merely REFERENCED (`alias = os.system`,
        `handlers.append(os.execv)`), however `os` is bound;
      - `subprocess.getoutput`/`getstatusoutput`, direct call or reference
        (both always run through a shell);
      - any of the five supported direct `subprocess` calls (`run`, `call`,
        `Popen`, `check_call`, `check_output`) used as a VALUE rather than
        called: passed as an argument, assigned to another name, returned,
        ... `_subprocess_targets` can only judge a direct call's argv, so a
        reference is a launch this scan must treat as unresolved.

    A direct call to one of the five is deliberately NOT flagged here:
    `_subprocess_targets` decides it from argv plus its `shell=`/
    `executable=` keywords, so an ordinary proved-safe
    `subprocess.run([...])` stays resolved. Module aliases (`import os as
    o`, `import subprocess as sp`, `import os.path`) and `from os import
    ...`/`from subprocess import ...` aliases are followed; a wildcard
    from-import of either module is unresolved. An attribute on any other
    receiver (an unrelated `thing.system()`, `app.run`) is never flagged.
    One of `build_import_graph`'s catch-all triggers and one of
    `unresolved_import_files`'s taint sources: a file with any of these
    shapes depends on every other tracked file, and a test whose real-edge
    closure reaches it stays in the #155 failsafe set.
    Only these two modules are in scope: `multiprocessing`, `asyncio`
    subprocess calls and `os.posix_spawn` remain issue #42 territory, as
    do `getattr`-built references and module rebinding (`sp =
    subprocess`)."""
    os_aliases = _launch_module_aliases(tree, "os")
    subprocess_aliases = _launch_module_aliases(tree, "subprocess")
    from_os: dict[str, str] = {}
    from_subprocess: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        module = node.module or ""
        if module == "os":
            for alias in node.names:
                if alias.name == "*":
                    return True
                if alias.name in _OS_DYNAMIC_ATTRS:
                    from_os[alias.asname or alias.name] = alias.name
        elif module == "subprocess":
            for alias in node.names:
                if alias.name == "*":
                    return True
                if alias.name in _SUBPROCESS_LAUNCH_ATTRS:
                    from_subprocess[alias.asname or alias.name] = alias.name
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr in _OS_DYNAMIC_ATTRS and node.value.id in os_aliases:
                launch = node.attr
            elif node.attr in _SUBPROCESS_LAUNCH_ATTRS \
                    and node.value.id in subprocess_aliases:
                launch = node.attr
            else:
                continue
        elif isinstance(node, ast.Name):
            launch = from_os.get(node.id) or from_subprocess.get(node.id)
            if launch is None:
                continue
        else:
            continue
        parent = parents.get(id(node))
        if not (isinstance(parent, ast.Call) and parent.func is node):
            return True
        if launch in _OS_DYNAMIC_ATTRS or launch in _SUBPROCESS_SHELL_ATTRS:
            return True
    return False


class _ImportGraph(dict):
    """`build_import_graph`'s return value: behaves as a plain
    `dict[str, set[str]]` everywhere (subscripting, `.get`, `in`, `len`,
    `set(...)`, `.items()`, equality against a plain dict/set of the same
    contents) -- every existing caller that treats it as exactly that type is
    unaffected. Carries one extra attribute, `precise`: a
    `dict[str, set[str]]` of each tracked file's REAL, ast-resolved edges
    only, computed unconditionally, before an unresolved-import file's
    catch-all (`tracked_set - {rel}`) is unioned into its entry in `self`. A
    file's own real edges (if any) are always a subset of its catch-all, so
    `self[rel]`'s VALUE is unchanged by tracking them separately -- `precise`
    exists purely so `module_dependency_closure` can walk real edges through
    a file that ALSO carries a catch-all, instead of losing that
    information the moment the file fails safe. A caller that never asks
    for `.precise` (every caller before this round) sees no difference at
    all."""


def build_import_graph(tracked: list[str] | None = None) -> dict[str, set[str]]:
    """Static import graph over EVERY git-tracked `.py` file in the repo (no
    hand-kept root allowlist -- see `_tracked_roots`): maps each file to the
    set of tracked files it imports, via `import`/`from ... import`
    (resolved by `_resolve_dotted`) AND via the one allowed
    `importlib.import_module("x.y")` shape (`_allowed_import_module_call`
    then `_resolve_dotted`), PLUS the tracked `__init__.py` of every strict
    ancestor package of each resolved import (`_ancestor_package_inits`) --
    Python always runs a package's `__init__.py` before any of its
    submodules, so `import engine.pkg.inner` depends on
    `engine/pkg/__init__.py` even when neither the import statement nor
    `engine/pkg/__init__.py` itself ever names `engine.pkg.inner`. A
    `subprocess` launch of a proved-Python interpreter (`python`/`python3`/
    versioned basename or `<sys alias>.executable`) that names a tracked
    literal script or `-m dotted.name` module adds that target (plus the
    module's tracked ancestor package `__init__.py` files) as an edge the
    same way (`_subprocess_targets`).

        A file with a genuinely unresolved import/load attempt by
    `_has_unresolved_import_attempt` (including a non-literal or annotated
    top-level conftest.py `pytest_plugins`), an import-path mutation to a
    target that is not provably the file's own repository root by
    `_has_unresolved_sys_path_mutation` (`sys.path` insert/append/extend,
    a rebinding or subscript write of the path list, `site.addsitedir`,
    `monkeypatch.syspath_prepend`), a process launch
    `_has_unresolved_process_launch` catches on AST shape alone (an
    `os.system`/`os.exec*`/`os.spawn*` call or reference, a
    `subprocess.getoutput`/`getstatusoutput`, or a supported `subprocess`
    exec function used as a value), OR a `subprocess` launch
    `_subprocess_targets` cannot prove is either a non-Python command or a
    tracked Python script/module target,
    still gets this precise resolution (see `.precise` below), but ALSO
    depends on EVERY
    OTHER TRACKED FILE in its main entry (`edges |= tracked_set - {rel}`),
    never a narrower guess for that added catch-all. This classification is
    deliberately narrower than the older broad `_is_dynamic_file`
    allowlist: a `sys.path` READ or alias with no mutation, a bare
    `PYTHONPATH` string or assignment, `import site` alone, a `subprocess`
    import or direct call whose argv is provably NON-Python (e.g. `git`)
    or whose Python target resolved to a tracked file, and
    non-import constructs such as `pkgutil` or a bare `compile` reference
    do not add the catch-all; the remaining gap is tracked in
    https://github.com/yshewchuk/investment-validation/issues/42 and issue
    #155, with push-to-main and the weekly scheduled mutation run as the
    backstop. Today the real `tests/conftest.py` repository-root
    `sys.path.insert` is a PROVEN root insertion (`REPO_ROOT =
    Path(__file__).resolve().parents[1]` + `str(REPO_ROOT)`), so it is not
    DYNAMIC and neither that insertion nor its provably non-Python `npm`
    subprocess calls put it in `unresolved_import_files`; it remains a
    closure root for every test through `_conftest_ancestors` regardless.
 Every tracked file is a key, even one with no
    resolvable imports (an empty set), so `module_dependency_closure` can
    always look it up. Raises `SyntaxError` (via `ast.parse`) on the first
    file that fails to parse -- a real syntax error in the current tree,
    never swallowed into a silently partial graph; `changed_modules` treats
    that as "select every module".

    The returned `_ImportGraph` also carries `.precise`: each file's REAL
    ast-resolved edges alone, computed UNCONDITIONALLY (a DYNAMIC file's own
    ordinary, statically-resolvable imports are never skipped just because
    it also fails safe elsewhere), before a DYNAMIC file's catch-all is
    unioned into its entry in `self`. Since a DYNAMIC file's real edges are
    always a subset of its own catch-all, `self[rel]` is IDENTICAL to what it
    would be without `.precise` -- every caller that only ever subscripted
    the graph is unaffected. `.precise` exists for `module_dependency_closure`,
    which needs a DYNAMIC file's real edges without inheriting its catch-all
    as something to expand further (see that function's own docstring)."""
    tracked = tracked if tracked is not None else [
        p for p in _tracked(["."]) if p.endswith(".py")]
    tracked_set = set(tracked)
    roots = _tracked_roots(tracked_set)
    graph = _ImportGraph({rel: set() for rel in tracked})
    precise: dict[str, set[str]] = {}
    for rel in tracked:
        source = (REPO / rel).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            raise SyntaxError(f"{rel}: {exc}") from exc
        edges = graph[rel]
        is_conftest = rel.rsplit("/", 1)[-1] == "conftest.py"
        if is_conftest:
            found, dotted_names = _pytest_plugins_targets(tree)
            if found and dotted_names is not None:
                for dotted in dotted_names:
                    target = _resolve_dotted(dotted, tracked_set, roots)
                    if target:
                        edges.add(target)
                    edges |= _ancestor_package_inits(dotted, tracked_set)
        # Always resolve precise imports now, unresolved or not -- the
        # catch-all check below only decides whether the catch-all is ALSO
        # unioned in, never whether real edges are computed at all (see
        # `_ImportGraph`).
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
                target = _allowed_import_module_call(node)
                if target:
                    resolved = _resolve_dotted(target, tracked_set, roots)
                    if resolved:
                        edges.add(resolved)
                    edges |= _ancestor_package_inits(target, tracked_set)
        subprocess_edges, subprocess_unresolved = _subprocess_targets(
            tree, tracked_set, roots)
        edges |= subprocess_edges
        precise[rel] = set(edges)
        if (_has_unresolved_import_attempt(tree, is_conftest)
                or _has_unresolved_sys_path_mutation(tree, rel)
                or _has_unresolved_process_launch(tree)
                or subprocess_unresolved):
            edges |= tracked_set - {rel}

    graph.precise = precise
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
    does not OWN that path.

    The walk follows REAL edges only -- `graph.precise` when `graph` is a
    real `_ImportGraph` from `build_import_graph` (every production call
    site). This recovers a DYNAMIC file's genuine, statically-resolvable
    imports (e.g. `tests/test_v2_ops_foundation.py` is individually DYNAMIC
    yet has a plain `from engine.v2 import foundation` -- the closure must
    still reach `engine.v2.foundation` through it) while never treating a
    DYNAMIC file's catch-all edge (`tracked_set - {rel}`, "this file might
    import literally anything") as something to expand further: that catch-
    all is a conservative fact about the file ITSELF (reaching it, or
    changing it, still selects every module whose closure reaches it -- see
    `_closure_roots`/`_conftest_ancestors` for `tests/conftest.py`, which is
    a closure root for every module), not a real transitive edge to relay
    (measured: relaying it collapsed every enabled module's dependency set
    to the whole ~932-file tracked tree, because `tests/conftest.py` is a
    closure root for every module and was DYNAMIC via its own
    `sys.path.insert` before that insertion was proven to be the repo
    root).

    A synthetic test may instead pass a graph with an explicit `.dynamic`
    attribute (a `set[str]` of file names whose catch-all edge must not be
    expanded further) -- there is no other reliable way for such a test to
    express "this node is DYNAMIC": inferring it from edge-set shape
    (`edges == tracked_set - {f}`) is unsound, because a small synthetic
    graph's ORDINARY real edge can coincidentally equal "every other tracked
    file" (e.g. a 2-file graph where the only file imports the other). A
    plain `dict` with no `.dynamic` attribute gets an empty
    `dynamic_boundary` -- ordinary full-edge traversal, no cutoff -- so
    existing synthetic graphs that never intended to exercise DYNAMIC
    catch-all behavior are unaffected."""
    tracked_set = tracked_set if tracked_set is not None else set(graph)
    precise = getattr(graph, "precise", None)
    dynamic_boundary = (
        None if precise is not None
        else getattr(graph, "dynamic", set())
    )
    seen: set[str] = set()
    stack = list(_closure_roots(module_cfg(cfg, name), tracked_set))
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        if precise is not None:
            stack.extend(precise.get(f, ()))
        elif f in dynamic_boundary:
            continue
        else:
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


# -- PR test selection: every test file's own real-edge closure -------------
#
# select_pr_tests generalizes changed_modules above from the hand-configured
# [modules.<name>] partition (which covers only engine/v2/* + ops, not the
# whole tests/ tree) to every tracked tests/test_*.py file's own closure, so
# the `test` CI job's pull_request runs can narrow WHICH TEST FILES pytest
# collects instead of always running all of them. It reuses
# build_import_graph, .precise, _conftest_ancestors and
# is_inert_changed_path unchanged; it does not touch changed_modules or
# module_dependency_closure.

def dynamic_files(graph: dict[str, set[str]]) -> set[str]:
    """Tracked files build_import_graph classified DYNAMIC: their resolved
    edge set (self) is a strict superset of their real ast-resolved edges
    alone (.precise), because it also carries the catch-all
    (tracked_set - {rel}). A graph with no .precise (e.g. a hand-built
    plain dict) reports none."""
    precise = getattr(graph, "precise", None)
    if precise is None:
        return set()
    return {f for f, edges in graph.items() if edges - precise.get(f, set())}


def pytest_test_files(tracked_set: set[str]) -> list[str]:
    """Every tracked file the `test` CI job's pytest run collects: that
    job's positional argument is the tests/ directory, and pytest's default
    collection pattern is test_*.py."""
    return sorted(p for p in tracked_set
                  if p.startswith("tests/") and p.rsplit("/", 1)[-1].startswith("test_")
                  and p.endswith(".py"))


def forces_full_suite(cfg: dict, path: str) -> bool:
    """True if `path` matches tools/mutation_pilot.toml's [pr_selection]
    full_suite allowlist. Same fnmatch convention as is_inert_changed_path."""
    return any(fnmatch.fnmatchcase(path, pat)
               for pat in cfg.get("pr_selection", {}).get("full_suite", []))


def _has_unresolved_import_attempt(tree: ast.Module, is_conftest: bool) -> bool:
    """True if `tree` attempts to load some OTHER module by a construct
    whose target cannot be statically resolved: `importlib` used any way
    other than the one literal `importlib.import_module("<literal>")`
    shape, a bare `importlib.reload(...)` call, or an access through the
    unaliased `importlib.metadata` namespace (the sole non-loading
    `importlib` spelling allowed; its version/entry-point queries cannot
    reach a module). `reload` is allowed
    unconditionally (unlike `import_module`, there is no literal-argument
    shape to check): it only re-executes a module that was already
    obtained some other way, so it cannot by itself introduce a new,
    otherwise-invisible dependency -- whatever obtained the module in the
    first place is what would need checking, and a dynamic way of
    obtaining it is already caught by the other rules here. Every other
    `importlib` use is flagged: an aliased import, any `from importlib
    import ...`, any other `importlib.*` attribute, or a non-literal
    `import_module(...)` call;
    the standalone name `__import__`, however bound;
    `spec_from_file_location` or `SourceFileLoader` as a bare name or an
    attribute's `.attr`; any `import runpy` or `from runpy import ...`; a
    bare `ast.Name` `exec` or `eval` used as the direct `func` of an
    `ast.Call` (an `eval`/`exec` name in any other position -- an
    annotation, a store target, a call argument, an unrelated attribute --
    is not a dynamic-load attempt); or, for a conftest.py only, an
    ANNOTATED or non-literal top-level `pytest_plugins` assignment. The
    qualified `builtins.exec`/`builtins.eval` form is also caught, since
    it is the exact same risk under a different spelling.
    Deliberately narrower than `_is_dynamic_file`: this function checks
    only for import-statement-shaped dynamic loading (the forms listed
    above). The other runtime-loading mechanisms this scan DOES track are
    judged by the adjacent classifiers, not here: import-path mutations by
    `_has_unresolved_sys_path_mutation` and process launches by
    `_has_unresolved_process_launch`. Each of the three contributes to
    `build_import_graph`'s catch-all edge and to
    `unresolved_import_files`'s taint set; none of them is a sound
    analysis of everything a file can run. What remains genuinely
    untracked -- `pkgutil`, bare `compile`, `PYTHONPATH`,
    `multiprocessing`/`asyncio` subprocess calls, `getattr`-built
    references, module rebinding, ... -- is the deliberate scope boundary
    tracked in issue #42, under the project's documented best-effort
    contract (see ARCHITECTURE.md): a test that depends on repository code
    only through one of those constructs, or any other runtime loading
    this analysis doesn't track, may be omitted from a PR's narrowed
    selection, and the full suite on every push to `main` is the
    backstop. Used by select_pr_tests's taint rule AND by
    `build_import_graph` as one of the catch-all triggers (import-path
    mutations and process launches are judged next door, by
    `_has_unresolved_sys_path_mutation` and
    `_has_unresolved_process_launch`)."""
    if is_conftest:
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) \
                    and isinstance(node.target, ast.Name) \
                    and node.target.id == "pytest_plugins":
                return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "runpy":
                    return True
                if alias.name.split(".")[0] == "importlib" and alias.asname is not None:
                    return True
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0] if node.module else ""
            if top == "runpy":
                return True
            if top == "importlib":
                return True
        elif isinstance(node, ast.Attribute):
            if node.attr in ("__import__", "spec_from_file_location", "SourceFileLoader"):
                return True
            if isinstance(node.value, ast.Name) and node.value.id == "builtins" \
                    and node.attr in ("exec", "eval"):
                return True
            if isinstance(node.value, ast.Name) and node.value.id == "importlib" \
                    and node.attr not in ("import_module", "reload", "metadata"):
                return True
        elif isinstance(node, ast.Name):
            if node.id in ("__import__", "spec_from_file_location",
                           "SourceFileLoader"):
                return True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval"):
                return True
            if _looks_like_import_module_call(node.func) and _allowed_import_module_call(node) is None:
                return True
    if is_conftest:
        found, dotted_names = _pytest_plugins_targets(tree)
        if found and dotted_names is None:
            return True
    return False


def _has_unresolved_sys_path_mutation(tree: ast.Module, rel: str) -> bool:
    """True if the file at tracked path `rel` writes an import search path
    in a shape whose effect this scan cannot prove safe:

      - a `sys.path.insert`/`append` call whose path argument is not
        provably the repository root for `rel` (the file's own
        `Path(__file__).resolve().parents[<depth>]` expression, directly,
        through a `str(...)` call, or via a plain top-level name assigned
        that expression), however `sys` is bound -- a `sys` alias,
        `from sys import path`, or an alias of the from-imported name;
      - any `sys.path.extend(...)` call on such a receiver: unlike
        insert/append the argument is an iterable, so no single target can
        be proved to be the repository root and every extend is unresolved;
      - a `site.addsitedir` or `monkeypatch.syspath_prepend` call whose
        target is not provably the repository root (whatever object the
        attribute hangs off -- `monkeypatch` is only a parameter name to
        this static scan);
      - an assignment (`Assign`/`AnnAssign`/`AugAssign`) whose target IS
        the recognized path receiver (`sys.path`/an alias, or a name bound
        by `from sys import path`) or a subscript of it (`sys.path[0] =
        ...`, `path += ...`): rebinding the list or writing an element
        mutates the search path whatever the right-hand side is.

    Ordinary READS never count: `p = sys.path`, `if ROOT not in sys.path`,
    and `first = sys.path[0]` are not mutations and are not flagged. One of
    `build_import_graph`'s catch-all triggers and one of
    `unresolved_import_files`'s taint sources; the real `tests/conftest.py`
    repository-root insertion IS provably the root, so it is neither."""
    depth = rel.count("/")
    root_texts = {f"Path(__file__).resolve().parents[{depth}]",
                  f"pathlib.Path(__file__).resolve().parents[{depth}]"}
    root_names = {n.targets[0].id for n in tree.body
                  if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name) and ast.unparse(n.value) in root_texts}
    sysn, siten = _bound_aliases(tree, "sys") | {"sys"}, _bound_aliases(tree, "site")
    pairs = {(n.module, a.asname or a.name, a.name) for n in ast.walk(tree)
             if isinstance(n, ast.ImportFrom) and not n.level for a in n.names}
    paths = {x for m, x, a in pairs if (m, a) == ("sys", "path")}
    sitdirs = {x for m, x, a in pairs if (m, a) == ("site", "addsitedir")}
    path_attrs = {f"{s}.path" for s in sysn}
    path_recvs = path_attrs | paths

    def is_path_receiver(expr: ast.expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in paths
        return isinstance(expr, ast.Attribute) and ast.unparse(expr) in path_attrs

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            attr = getattr(f, "attr", None)
            recv = ast.unparse(f.value) if attr else None
            if attr == "extend" and recv in path_recvs:
                return True
            ok = (isinstance(f, ast.Name) and f.id in sitdirs) or attr == "syspath_prepend" or (
                attr == "addsitedir" and recv in siten) or (
                attr in ("insert", "append") and recv in path_recvs)
            x = (n.args + [None, None])[1 if attr == "insert" else 0]
            if isinstance(x, ast.Call) and not x.keywords and len(x.args) == 1 \
                    and isinstance(x.func, ast.Name) and x.func.id == "str":
                x = x.args[0]
            if ok and not (getattr(x, "id", None) in root_names
                           or (x is not None and ast.unparse(x) in root_texts)):
                return True
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for target in targets:
                if is_path_receiver(target):
                    return True
                if isinstance(target, ast.Subscript) and is_path_receiver(target.value):
                    return True
    return False


def unresolved_import_files(tracked: list[str]) -> set[str]:
    """The subset of `tracked` that triggers ANY of the catch-all-producing
    unresolved constructs `build_import_graph` checks per file -- the union
    of `_has_unresolved_import_attempt`, `_has_unresolved_sys_path_mutation`,
    `_has_unresolved_process_launch`, and `_subprocess_targets(...)[1]`.
    The tracked set and roots are derived exactly as graph building derives
    them (`_tracked_roots`), so a `subprocess` literal script or `-m` target
    resolves the same way in both scans. Re-parses each file ONCE (a second
    AST pass beyond `build_import_graph`'s), since this classification is
    not otherwise exposed by the graph. Used only by select_pr_tests's taint
    rule (via `_closure_from_roots`), never dynamic_files' broader leaf
    rule.

    Every category above taints deliberately: a test whose real-edge
    closure reaches one of these files cannot trust that closure, because
    the reached file can import, add to the import path, or launch code
    this static analysis cannot see. The real `tests/conftest.py` is NOT in
    this set: its known repository-root `sys.path.insert` is a PROVEN root
    insertion and its `npm` launches are provably non-Python, so it taints
    nothing."""
    tracked_set = set(tracked)
    roots = _tracked_roots(tracked_set)
    out: set[str] = set()
    for rel in tracked:
        source = (REPO / rel).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel)
        is_conftest = rel.rsplit("/", 1)[-1] == "conftest.py"
        if (_has_unresolved_import_attempt(tree, is_conftest)
                or _has_unresolved_sys_path_mutation(tree, rel)
                or _has_unresolved_process_launch(tree)
                or _subprocess_targets(tree, tracked_set, roots)[1]):
            out.add(rel)
    return out


def _closure_from_roots(roots: set[str], graph: dict[str, set[str]],
                        dyn: set[str], *, taint_exempt: set[str]) -> tuple[set[str], bool]:
    """BFS over graph's REAL edges only (graph.precise when present, else
    graph itself), starting from `roots`. Returns (closure, tainted):
    `tainted` is True iff some file reached via a real import edge -- never
    one of `taint_exempt` -- is itself in `dyn` (unresolved_import_files'
    output). Such a file's own further edges are unresolvable (an
    unresolved dynamic import could load literally anything), so anything
    that transitively imports it cannot trust its own closure either.

    `taint_exempt` is normally just `{t}` (the test file itself): its OWN
    narrow-unresolved status is handled by the separate, broader leaf rule
    in `select_pr_tests` (`t in dyn`, where `dyn` is the broad
    `dynamic_files` set, a superset of `unresolved_import_files`), so
    re-tainting it here would be redundant. It is NOT the test file's
    conftest ancestors: a conftest ancestor with a genuine unresolved
    import SHOULD taint its dependents, because a test using one of that
    conftest's fixtures can have a real, invisible runtime dependency on
    whatever the fixture loads -- fixture injection is a runtime name
    lookup, not a static import edge, so the test's own closure never sees
    that dependency (the gate's reproduced scenario: a conftest.py fixture
    calls `importlib.import_module(name)` with a non-literal `name`, and
    the test using the fixture has no static edge at all to the loaded
    module).

    `tests/conftest.py` never taints anything even though it is passed in
    `roots` as every test's ancestor: this function is only ever called
    with the NARROW `unresolved_import_files` set as `dyn`, never the
    broad `dynamic_files` set, and that narrow set contains only files
    with a genuinely unresolved construct. The real `tests/conftest.py`
    has none -- its known repository-root `sys.path.insert` is a PROVEN
    root insertion and its `npm` launches are provably non-Python -- so it
    is not in the narrow set and needs no separate exemption here."""
    precise = getattr(graph, "precise", None)
    seen: set[str] = set()
    tainted = False
    stack = list(roots)
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        if f in dyn and f not in taint_exempt:
            tainted = True
        stack.extend((precise if precise is not None else graph).get(f, ()))
    return seen, tainted


def select_pr_tests(cfg: dict, changed: list[str], *,
                    graph: dict[str, set[str]] | None = None) -> list[str] | None:
    """The pytest test files (tests/test_*.py) a pull_request `test` CI run
    should collect. Returns None for "run the full suite" (a path on the
    full_suite allowlist, an unrecognized/unreached path, or any failure
    building the graph or scanning for unresolved imports -- never a
    silent narrow selection on an error).
    An empty `changed` returns [] (no diff -> nothing to run), matching
    changed_modules; a docs-only diff no test reads returns [] too.

    Fan-out limits: the #155 fail-safe set (DYNAMIC or tainted tests) is
    added only when the diff touches a non-test python file; a collected
    test file is a leaf (selects itself and its static importers), and a doc
    path (`is_doc_changed_path`) selects only the tests whose closure names
    that doc (`_doc_reader_tests`). A diff of only those selects a narrow set.

    #155 (unresolved dynamic loading) handling: a test file that is ITSELF
    classified DYNAMIC (_is_dynamic_file) is selected whenever the diff
    touches a non-test python file, and so is a
    test file that reaches, via a real import edge, some OTHER file in
    `unresolved_import_files` -- one with a genuine unresolved import
    attempt, import-path mutation, or process launch (a "helper" that can
    load or run code this scan cannot see) -- both via _closure_from_roots's
    `tainted` return.
    Only the test file's OWN narrow-unresolved status is exempted here
    (handled separately by the `t in dyn` leaf rule, using the broader
    dynamic_files set); a conftest ANCESTOR with a genuine unresolved
    construct DOES taint its dependents, because a test using one of its
    fixtures can have a real, invisible runtime dependency on whatever
    that fixture loads -- see _closure_from_roots's own docstring."""
    changed_set = set(changed)
    if not changed_set:
        return []
    if graph is None:
        try:
            graph = build_import_graph()
        except Exception as exc:
            print(f"[mutation_pilot] import graph build failed "
                  f"({type(exc).__name__}: {exc}); selecting the full test suite",
                  file=sys.stderr, flush=True)
            return None
    if any(forces_full_suite(cfg, p) for p in changed_set):
        return None
    tracked_set = set(graph)
    tests = pytest_test_files(tracked_set)
    dyn = dynamic_files(graph)
    try:
        unresolved = unresolved_import_files(sorted(tracked_set))
    except Exception as exc:
        print(f"[mutation_pilot] unresolved-import scan failed "
              f"({type(exc).__name__}: {exc}); selecting the full test suite",
              file=sys.stderr, flush=True)
        return None
    closures: dict[str, set[str]] = {}
    failsafe: set[str] = set()
    for t in tests:
        closure, tainted = _closure_from_roots(
            {t} | _conftest_ancestors(t, tracked_set), graph, unresolved,
            taint_exempt={t})
        closures[t] = closure
        if t in dyn or tainted:
            failsafe.add(t)
    tests_set = set(tests)
    selected: set[str] = set()
    docs: list[str] = []
    needs_failsafe = False
    for path in sorted(changed_set):
        hit = {t for t in tests if path in closures[t]}
        selected |= hit
        if path in tests_set:
            continue  # a collected test file is a leaf: it affects only itself + its importers
        if hit:
            needs_failsafe = True
        elif is_doc_changed_path(cfg, path):
            docs.append(path)
        else:
            return None
    if needs_failsafe:
        selected |= failsafe
    if docs:
        selected |= _doc_reader_tests(docs, tests, closures)
    return sorted(selected)


def is_doc_changed_path(cfg: dict, path: str) -> bool:
    """True if `path` is documentation for test selection: on the `inert`
    allowlist, or any `*.md` outside `inert_skip`. Wider than
    is_inert_changed_path on purpose: that one gates mutation-module skipping,
    this one only picks which tests read the doc (`_doc_reader_tests`)."""
    if any(fnmatch.fnmatchcase(path, pat)
           for pat in cfg.get("pr_selection", {}).get("inert_skip", [])):
        return False
    return is_inert_changed_path(cfg, path) or path.endswith(".md")


def _string_literals(source: str) -> list[str]:
    """Every string constant in `source` except docstrings: a path a file
    reads at runtime is a string literal, while a docstring or comment merely
    mentioning a doc is not a read."""
    tree = ast.parse(source)
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip]


def _doc_reader_tests(docs: list[str], tests: list[str],
                      closures: dict[str, set[str]]) -> set[str]:
    """Tests that can read a changed doc: those whose import closure (python
    files only) holds a non-docstring string literal naming the doc's file
    name. An unreadable or unparsable closure file counts as a match, never a
    silent skip."""
    names = {d.rsplit("/", 1)[-1] for d in docs}
    named: dict[str, bool] = {}

    def names_a_doc(rel: str) -> bool:
        if rel not in named:
            try:
                src = (REPO / rel).read_text(encoding="utf-8")
                named[rel] = any(n in lit for lit in _string_literals(src) for n in names)
            except (OSError, UnicodeDecodeError, SyntaxError):
                named[rel] = True
        return named[rel]

    return {t for t in tests if any(names_a_doc(rel) for rel in closures[t])}


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

def cmd_select_tests(cfg: dict, args) -> int:
    """Print __ALL__ (run the full test suite) or the selected pytest test
    file paths, one per line (possibly zero lines, never a trailing blank
    line), for the `test` CI job's pull_request runs."""
    changed = read_changed_files(args.changed_files)
    selected = select_pr_tests(cfg, changed)
    if selected is None:
        print("__ALL__")
    else:
        for t in selected:
            print(t)
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
    p = sub.add_parser("select-tests",
                       help="test files (or __ALL__) a pull_request `test` CI run should run")
    p.add_argument("--changed-files", required=True, metavar="PATH",
                   help="path to a NUL-delimited changed-file list (git diff -z --no-renames "
                        "--name-only); see select_pr_tests's docstring for the selection rule")
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
    p = sub.add_parser("config-hash",
                       help="sha256 of [defaults] + this module's own section, for the CI cache key")
    p.add_argument("module")
    args = parser.parse_args(argv)
    cfg = load_config()
    return {"list": cmd_list, "matrix": cmd_matrix, "select-tests": cmd_select_tests,
            "count": cmd_count, "run": cmd_run,
            "report": cmd_report, "config-hash": cmd_config_hash}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
