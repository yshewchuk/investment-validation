#!/usr/bin/env python3
"""A per-file line budget for every ``ARCHITECTURE.md``.

AGENTS.md "Small PRs" (user decision 2026-09-28): a component ``ARCHITECTURE.md``
is contract level -- purpose, interfaces, dependencies, invariants, and failure
semantics as a short condition -> outcome table. It carries no step-by-step
procedure restating the code, no retry-key derivations, no per-branch refusal
walks, and no history; that detail lives in code, tests and the PR body. A doc
that keeps growing past that shape is the mechanical symptom this check
catches: every ``ARCHITECTURE.md`` (root or component) must stay at or under
BUDGET lines.

Five docs already exceed BUDGET and predate this check. Rewriting them to
contract level is real work this PR does not do by fiat, so each is pinned in
EXEMPT at its line count when this check was added -- a cap, not a new
allowance: none of the five may grow even one line past that number (see
``_effective_cap``).

Reads every tracked path's staged content by default (what would actually be
committed); ``--all`` reads worktree content instead. Either read failing
raises rather than being scored as an empty, in-budget doc -- a budget check
that can silently pass on a read error is not a check (see
``_read_worktree_strict`` / ``_read_staged_strict``).

Usage::

    python3 checks/architecture_doc_budgets.py          # tracked + staged
    python3 checks/architecture_doc_budgets.py --all    # worktree
"""
from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checks.repo_hygiene import tracked_paths  # noqa: E402

#: Environment variables that redirect git to a different repository than
#: the one named by ``-C``; an inherited value from the caller's shell must
#: not silently retarget a git call this module makes.
_GIT_ENV_LEAK = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY")


def _clean_git_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _GIT_ENV_LEAK}


@contextlib.contextmanager
def _clean_git_process_env():
    """Temporarily strips ``_GIT_ENV_LEAK`` from the process environment, so
    every git call inside the block -- including ``tracked_paths``, which
    shells out with the ambient environment rather than an explicit
    ``env=`` -- inspects the requested root. Without this, an inherited
    ``GIT_DIR``/``GIT_WORK_TREE`` can silently retarget ``tracked_paths`` to
    an unrelated repository, so ``_sources`` sees zero files and the whole
    budget check passes with ``docs=0`` instead of failing or erroring."""
    saved = {k: os.environ.pop(k, None) for k in _GIT_ENV_LEAK}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

__all__ = ["BUDGET", "EXEMPT", "Violation", "Report", "check_files", "main"]

#: Contract-level docs measured on this tree fit comfortably under this; see
#: the module docstring for what stays out of a component ARCHITECTURE.md.
BUDGET = 500

#: path -> line count when this check was added. Each of these predates the
#: contract-level shape and is far enough over BUDGET that shrinking it is a
#: rewrite this PR does not make; each is capped at its own size instead of
#: being free to keep growing.
EXEMPT: dict[str, int] = {
    "engine/v2/ops/ARCHITECTURE.md": 5314,
    "engine/v2/data/ARCHITECTURE.md": 1048,
    "engine/v2/scoring/ARCHITECTURE.md": 1017,
    "engine/v2/models/ARCHITECTURE.md": 863,
    "ARCHITECTURE.md": 531,
}


@dataclass(frozen=True)
class Violation:
    path: str
    lines: int
    budget: int

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"  {self.path}: {self.lines} lines, budget {self.budget}"


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    docs: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations


def _is_architecture_doc(path: str) -> bool:
    return path == "ARCHITECTURE.md" or path.endswith("/ARCHITECTURE.md")


def _effective_cap(path: str) -> int:
    """An exempt path's cap is its pinned size; everything else uses BUDGET."""
    return EXEMPT[path] if path in EXEMPT else BUDGET


def check_files(files: dict[str, bytes]) -> Report:
    """The pure core the CLI and the tests both drive."""
    report = Report()
    for path, blob in files.items():
        if not _is_architecture_doc(path):
            continue
        report.docs += 1
        lines = len(blob.splitlines())
        cap = _effective_cap(path)
        if lines > cap:
            report.violations.append(Violation(path, lines, cap))
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _read_worktree_strict(root: Path, rel: str) -> bytes:
    """Raises OSError on a read failure -- never scored as an empty doc."""
    return (root / rel).read_bytes()


def _read_staged_strict(root: Path, rel: str) -> bytes:
    """Raises CalledProcessError if ``rel`` cannot be read from the index --
    see ``_read_worktree_strict``."""
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f":{rel}"],
        capture_output=True, check=True, env=_clean_git_env(),
    )
    return proc.stdout


def _sources(root: Path, use_worktree: bool) -> dict[str, bytes]:
    with _clean_git_process_env():
        paths = [rel for rel in tracked_paths(root) if _is_architecture_doc(rel)]
        if use_worktree:
            return {rel: _read_worktree_strict(root, rel) for rel in paths}
        return {rel: _read_staged_strict(root, rel) for rel in paths}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--all", action="store_true",
                    help="read worktree content instead of staged content")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    report = check_files(_sources(root, args.all))

    if not args.quiet:
        print(f"ARCHITECTURE.md budgets: {report.docs} doc(s), budget {BUDGET} lines "
              f"({len(EXEMPT)} exempt at a pinned cap)")
    if report.ok:
        if not args.quiet:
            print("ARCHITECTURE BUDGETS OK")
        return 0
    print(f"\nARCHITECTURE BUDGETS FAILED -- {len(report.violations)} violation(s):",
          file=sys.stderr)
    for v in report.violations:
        print(str(v), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())