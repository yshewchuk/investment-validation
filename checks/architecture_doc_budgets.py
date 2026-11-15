#!/usr/bin/env python3
"""A per-PR growth ceiling for every ``ARCHITECTURE.md``.

User decision 2026-09-29 (replaces the 2026-09-28 fixed-budget/EXEMPT
model): below CEILING words, a PR may add at most MAX_PR_GROWTH net words
to any one ARCHITECTURE.md, measured against that doc's size on the base
branch -- a doc creeps up over many small PRs, never jumps in one. At or
over CEILING, a PR may shrink the doc but not grow it at all: it needs a
dedicated compression PR, or a code refactor, before it can take more
content. An owner is never responsible for trimming a doc's unrelated
sections to make room for their own change (AGENTS.md "Small PRs").

Measure: words (whitespace-split tokens), not lines, because a line count
is satisfiable by reflowing (#2439 joined hard-wrapped paragraphs, 1007 ->
486 lines, same bytes, and passed). Words are unchanged by any
whitespace/layout change, so a reflow neither shrinks nor grows a doc,
and a real deletion always shrinks it.

Calibration (user decision 2026-11-15, option A): the old limits were
1000 lines and +50 lines. Rule: CEILING = the median word count of the
tracked ARCHITECTURE.md docs that sat within 5% of the old 1000-line
ceiling on 2026-11-15 (5 docs: 9,248; 9,888; 10,454; 15,417; 23,580 words
-> median 10,454, rounded to the nearest 500 = 10,500); MAX_PR_GROWTH =
CEILING / 20 (the old 50/1000 ratio) = 525, rounded down to 500. Never
tuned to a particular PR.

"The PR's base" is the base branch's current tip (``origin/<branch>``), not
a true merge-base: CI's checkout is shallow and lacks the history a real
merge-base needs, and the tip is the fallback the user's rule names. This
can over/undercount a PR's own growth by whatever the base moved meanwhile;
see ``_resolve_base_ref``, which refreshes a missing local ref rather than
silently reading 0 words.

Reads every tracked ARCHITECTURE.md's staged content by default (what would
actually be committed); ``--all`` reads worktree content instead. Any
relevant read failing raises -- a check that can silently pass on a read
error is not a check -- except a doc missing from the base branch, which is
a doc new in this PR and scores 0 base words.

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


__all__ = ["CEILING", "MAX_PR_GROWTH", "MAX_DOC_BYTES", "Violation", "Report", "check_files", "main"]

#: A doc at or over this many words may only shrink; see the module docstring.
CEILING = 10_500

#: Net words any one PR may add to a doc still under CEILING, measured
#: against that doc's size on the base branch.
MAX_PR_GROWTH = 500

#: Largest ARCHITECTURE.md (bytes) the check will tokenise; a bigger blob is refused, never silently scored.
MAX_DOC_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Violation:
    path: str
    base_words: int
    new_words: int
    reason: str  # "ceiling" or "growth"

    @property
    def growth(self) -> int:
        return self.new_words - self.base_words

    def __str__(self) -> str:  # pragma: no cover - formatting only
        if self.reason == "ceiling":
            return (f"  {self.path}: {self.base_words} words, at/over the "
                     f"{CEILING}-word ceiling and grew to {self.new_words} "
                     f"-- doc at ceiling, needs a compression PR")
        return (f"  {self.path}: {self.base_words} -> {self.new_words} words "
                 f"-- PR adds {self.growth}>{MAX_PR_GROWTH} words, split it "
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


def _size(blob: bytes) -> int:
    if len(blob) > MAX_DOC_BYTES:
        raise ValueError(f"ARCHITECTURE.md blob of {len(blob)} bytes exceeds MAX_DOC_BYTES={MAX_DOC_BYTES}")
    return len(blob.decode("utf-8", "replace").split())


def check_files(base: dict[str, bytes], new: dict[str, bytes]) -> Report:
    """The pure core the CLI and the tests both drive. ``base``/``new`` map
    path -> content; a path absent from ``base`` is new in this PR (0 base
    words), one absent from ``new`` was removed (not scored)."""
    report = Report()
    for path in sorted(p for p in new if _is_architecture_doc(p)):
        new_words = _size(new[path])
        base_blob = base.get(path)
        base_words = _size(base_blob) if base_blob is not None else 0
        report.docs += 1
        if base_words >= CEILING:
            if new_words > base_words:
                report.violations.append(Violation(path, base_words, new_words, "ceiling"))
            continue
        if new_words - base_words > MAX_PR_GROWTH:
            report.violations.append(Violation(path, base_words, new_words, "growth"))
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _run_bounded(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    """``subprocess.run`` capturing output, but refuses (ValueError) stdout
    over MAX_DOC_BYTES while reading it, so an oversized blob is never fully
    buffered. stderr is only read after stdout is drained (git errors are tiny)."""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) as proc:
        out = proc.stdout.read(MAX_DOC_BYTES + 1)
        if len(out) > MAX_DOC_BYTES:
            proc.kill()
            raise ValueError(f"{cmd[-1]} exceeds MAX_DOC_BYTES={MAX_DOC_BYTES}")
        err = proc.stderr.read()
        rc = proc.wait()
    return subprocess.CompletedProcess(cmd, rc, out, err)


def _read_worktree_strict(root: Path, rel: str) -> bytes:
    """Raises OSError on a read failure, ValueError over MAX_DOC_BYTES --
    never scored as an empty doc."""
    with (root / rel).open("rb") as f:
        data = f.read(MAX_DOC_BYTES + 1)
    if len(data) > MAX_DOC_BYTES:
        raise ValueError(f"{rel} exceeds MAX_DOC_BYTES={MAX_DOC_BYTES}")
    return data


def _read_staged_strict(root: Path, rel: str) -> bytes:
    """Raises CalledProcessError if ``rel`` can't be read from the index,
    ValueError over MAX_DOC_BYTES."""
    proc = _run_bounded(
        ["git", "-C", str(root), "show", f":{rel}"], _clean_git_env(),
    )
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)
    return proc.stdout


_MISSING_AT_REF_MARKERS = ("does not exist in", "exists on disk, but not in")


def _read_base(root: Path, rel: str, base_ref: str) -> bytes | None:
    """``rel``'s content at ``base_ref``, or None if it simply doesn't
    exist there (new in this PR) -- detected from git's English stderr text,
    so the subprocess is forced to the C locale regardless of the caller's
    LANG/LC_ALL. Any other git failure (a bad ref, a
    corrupt object) raises rather than reading as 0 words, which would
    silently exempt a doc from the growth check."""
    proc = _run_bounded(
        ["git", "-C", str(root), "show", f"{base_ref}:{rel}"],
        {**_clean_git_env(), "LC_ALL": "C"},
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
              f"ceiling {CEILING} words, {MAX_PR_GROWTH} net words/PR below it")
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
