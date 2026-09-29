#!/usr/bin/env python3
"""A per-PR growth ceiling for every ``ARCHITECTURE.md``.

User decision 2026-09-29 (replaces the 2026-09-28 fixed-budget/EXEMPT
model): below CEILING lines, a PR may add at most MAX_PR_GROWTH net lines to
any one ARCHITECTURE.md, measured against that doc's size on the base
branch -- a doc creeps up over many small PRs, never jumps in one. At or
over CEILING, a PR may shrink the doc but not grow it at all: it needs a
dedicated compression PR, or a code refactor, before it can take more
content. An owner is never responsible for trimming a doc's unrelated
sections to make room for their own change (AGENTS.md "Small PRs").

"The PR's base" is the base branch's current tip (``origin/<branch>``), not
a true merge-base: CI's checkout is shallow and lacks the history a real
merge-base needs, and the tip is the fallback the user's rule names. This
can over/undercount a PR's own growth by whatever the base moved meanwhile;
see ``_resolve_base_ref``, which refreshes a missing local ref rather than
silently reading 0 lines.

Reads every tracked ARCHITECTURE.md's staged content by default (what would
actually be committed); ``--all`` reads worktree content instead. Any
relevant read failing raises -- a check that can silently pass on a read
error is not a check -- except a doc missing from the base branch, which is
a doc new in this PR and scores 0 base lines.

Usage::

    python3 checks/architecture_doc_budgets.py          # tracked + staged
    python3 checks/architecture_doc_budgets.py --all    # worktree
    python3 checks/architecture_doc_budgets.py --base-ref origin/main
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

#: These redirect git to a different repository than the one named by -C.
#: GIT_INDEX_FILE is deliberately excluded -- it is git's own mechanism for
#: pointing a hook at a temporary commit index (e.g. `git commit --only`),
#: and that is exactly the content "what would actually be committed" must
#: read.
_GIT_ENV_LEAK = ("GIT_DIR", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY")


def _clean_git_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _GIT_ENV_LEAK}


@contextlib.contextmanager
def _clean_git_process_env():
    """Strips ``_GIT_ENV_LEAK`` for the block, so every git call inside it
    -- including ``tracked_paths``, which shells out with the ambient
    environment -- inspects the requested root, not an inherited
    GIT_DIR/GIT_WORK_TREE target."""
    saved = {k: os.environ.pop(k, None) for k in _GIT_ENV_LEAK}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


__all__ = ["CEILING", "MAX_PR_GROWTH", "Violation", "Report", "check_files", "main"]

#: A doc at or over this many lines may only shrink; see the module docstring.
CEILING = 1000

#: Net lines any one PR may add to a doc still under CEILING, measured
#: against that doc's size on the base branch.
MAX_PR_GROWTH = 50


@dataclass(frozen=True)
class Violation:
    path: str
    base_lines: int
    new_lines: int
    reason: str  # "ceiling" or "growth"

    @property
    def growth(self) -> int:
        return self.new_lines - self.base_lines

    def __str__(self) -> str:  # pragma: no cover - formatting only
        if self.reason == "ceiling":
            return (f"  {self.path}: {self.base_lines} lines, at/over the "
                     f"{CEILING}-line ceiling and grew to {self.new_lines} "
                     f"-- doc at ceiling, needs a compression PR")
        return (f"  {self.path}: {self.base_lines} -> {self.new_lines} lines "
                 f"-- PR adds {self.growth}>{MAX_PR_GROWTH} lines, split it "
                 f"or move detail to the PR body")


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    docs: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations


def _is_architecture_doc(path: str) -> bool:
    return path == "ARCHITECTURE.md" or path.endswith("/ARCHITECTURE.md")


def check_files(base: dict[str, bytes], new: dict[str, bytes]) -> Report:
    """The pure core the CLI and the tests both drive. ``base``/``new`` map
    path -> content; a path absent from ``base`` is new in this PR (0 base
    lines), one absent from ``new`` was removed (not scored)."""
    report = Report()
    for path in sorted(p for p in new if _is_architecture_doc(p)):
        new_lines = len(new[path].splitlines())
        base_blob = base.get(path)
        base_lines = len(base_blob.splitlines()) if base_blob is not None else 0
        report.docs += 1
        if base_lines >= CEILING:
            if new_lines > base_lines:
                report.violations.append(Violation(path, base_lines, new_lines, "ceiling"))
            continue
        if new_lines - base_lines > MAX_PR_GROWTH:
            report.violations.append(Violation(path, base_lines, new_lines, "growth"))
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _read_worktree_strict(root: Path, rel: str) -> bytes:
    """Raises OSError on a read failure -- never scored as an empty doc."""
    return (root / rel).read_bytes()


def _read_staged_strict(root: Path, rel: str) -> bytes:
    """Raises CalledProcessError if ``rel`` can't be read from the index."""
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f":{rel}"],
        capture_output=True, check=True, env=_clean_git_env(),
    )
    return proc.stdout


_MISSING_AT_REF_MARKERS = ("does not exist in", "exists on disk, but not in")


def _read_base(root: Path, rel: str, base_ref: str) -> bytes | None:
    """``rel``'s content at ``base_ref``, or None if it simply doesn't
    exist there (new in this PR). Any other git failure (a bad ref, a
    corrupt object) raises rather than reading as 0 lines, which would
    silently exempt a doc from the growth check."""
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f"{base_ref}:{rel}"],
        capture_output=True, env=_clean_git_env(),
    )
    if proc.returncode == 0:
        return proc.stdout
    stderr = proc.stderr.decode("utf-8", "replace")
    if any(m in stderr for m in _MISSING_AT_REF_MARKERS):
        return None
    raise subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)


def _resolve_base_ref(root: Path, branch: str) -> str:
    """An already-present local ``origin/<branch>`` is used as-is (no
    network needed -- true for a normal checkout or a worktree cut from
    ``origin/main``). CI's shallow PR checkout has no such ref, so it's
    fetched at depth 1: enough to read the file, not a real merge-base."""
    ref = f"origin/{branch}"
    check = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "-q", ref],
        capture_output=True, env=_clean_git_env(),
    )
    if check.returncode == 0:
        return ref
    fetch = subprocess.run(
        ["git", "-C", str(root), "fetch", "--depth", "1", "origin",
         f"+{branch}:refs/remotes/origin/{branch}"],
        capture_output=True, env=_clean_git_env(),
    )
    if fetch.returncode != 0:
        raise RuntimeError(
            f"cannot resolve base ref {ref}: "
            f"{fetch.stderr.decode('utf-8', 'replace').strip()}"
        )
    return ref


def _base_branch() -> str:
    """CI sets GITHUB_BASE_REF to the PR's target branch for
    pull_request-triggered workflows; every PR here targets main, so that's
    also the default outside CI."""
    return os.environ.get("GITHUB_BASE_REF") or "main"


def _sources(
    root: Path, use_worktree: bool, base_ref: str | None,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    with _clean_git_process_env():
        paths = [rel for rel in tracked_paths(root) if _is_architecture_doc(rel)]
        if use_worktree:
            new = {rel: _read_worktree_strict(root, rel) for rel in paths}
        else:
            new = {rel: _read_staged_strict(root, rel) for rel in paths}
        ref = base_ref or _resolve_base_ref(root, _base_branch())
        base = {}
        for rel in paths:
            blob = _read_base(root, rel, ref)
            if blob is not None:
                base[rel] = blob
        return base, new


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--all", action="store_true",
                    help="read worktree content instead of staged content")
    ap.add_argument("--base-ref", default=None,
                    help="git ref to diff against (default: auto-detected origin/<branch>)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    base, new = _sources(root, args.all, args.base_ref)
    report = check_files(base, new)

    if not args.quiet:
        print(f"ARCHITECTURE.md growth: {report.docs} doc(s), "
              f"ceiling {CEILING} lines, {MAX_PR_GROWTH} net lines/PR below it")
    if report.ok:
        if not args.quiet:
            print("ARCHITECTURE DOC GROWTH OK")
        return 0
    print(f"\nARCHITECTURE DOC GROWTH FAILED -- {len(report.violations)} violation(s):",
          file=sys.stderr)
    for v in report.violations:
        print(str(v), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
