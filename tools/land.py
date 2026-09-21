#!/usr/bin/env python3
"""land: merge a sub-agent branch into main, gate it, and push.

Replaces the manual sequence a supervisor used to run by hand:

  1. refuse unless the working tree is clean and the branch exists;
  2. fetch ``origin/main`` and detach at it;
  3. ``git merge --no-ff <branch>``;
  4. deletion guard: whitespace-insensitive lines the merge removes, in two
     tiers -- tier 1 (hard, never overridable) refuses if any removed line
     was itself added to ``origin/main`` within the last
     ``RECENT_ADDITION_WINDOW`` commits (a stale-checkout revert of
     just-landed work); tier 2 (soft) refuses any other removed lines unless
     ``--allow-deletions N`` names the exact count;
  5. run repo hygiene with the real ``.env`` -- fail CLOSED when zero secret
     patterns were loaded (a hygiene check with an inactive value-grep must
     never wave a push through);
  6. run the tests selected by the merge diff, via tools/oc_check.py's own
     ``changed_tests``;
  7. ``git push origin HEAD:main`` (unless ``--dry-run``/``--no-push``).

Usage::

    python3 tools/land.py <branch> -m "<merge message>" [--dry-run] [--no-push]
    python3 tools/land.py <branch> -m "<merge message>" --allow-deletions 3

Exit codes: 0 success; 1 working tree not clean; 2 branch does not exist;
3 merge conflict (aborted, HEAD restored); 4 ``.env`` missing/unreadable;
5 refusing to PUSH (0 secret patterns loaded); 6 unparsable hygiene output;
7 hygiene violations; 8 tests failed (HEAD restored); 9 push failed
(HEAD restored); 10 deletion guard tier 1 -- merge reverts lines
origin/main gained recently (HEAD restored, never overridable); 11 deletion
guard tier 2 -- merge removes other lines that exist on main and either no
``--allow-deletions`` was given or it did not match the exact count (HEAD
restored).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

#: The real checkout's .env, never a relative path and never the worktree's own.
DEFAULT_ENV_PATH = Path("/root/investing-plan/.env")

SECRET_COUNT_RE = re.compile(r"(\d+) secret pattern\(s\) loaded from")
TEST_TAIL_LINES = 150

#: Off the heavy-job cores (bounded_run --cpu-set 0-5). Established in this
#: repo: ops suites stall under bounded_run --cores <6 (need 5 worker CPUs)
#: and should run --cores 8; this box (12 cores) only has 6 cores free of
#: the heavy-job set, so 6-11 is the exact non-overlapping range, measured
#: (443 passed / 2 failed) on land.py's own 22-file selection. Overridable
#: with --cpu-set.
DEFAULT_TEST_CPU_SET = "6-11"

#: Deletion-guard tier 1 lookback: a merge that removes a line origin/main
#: gained within this many of its own most recent commits is a
#: stale-checkout revert of just-landed work, refused unconditionally (see
#: module docstring). Clamped to the root commit on shorter histories.
RECENT_ADDITION_WINDOW = 50

#: Deletion-guard noise floor: ignore lines shorter than this (after
#: stripping) when matching removed lines against recently-added ones, or
#: closing braces / bare "else:" / blank lines generate false hits.
MIN_DELETION_GUARD_LINE_LEN = 8

#: Cap on how many removed lines the guard prints per refusal.
MAX_DELETION_LINES_SHOWN = 40


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def git(root: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc


def repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if proc.returncode != 0:
        eprint("land: refusing: not inside a git repository")
        sys.exit(1)
    return Path(proc.stdout.strip())


def save_head(root: Path) -> tuple[str, str]:
    proc = git(root, "symbolic-ref", "-q", "--short", "HEAD")
    if proc.returncode == 0 and proc.stdout.strip():
        return ("branch", proc.stdout.strip())
    proc = git(root, "rev-parse", "HEAD", check=True)
    return ("commit", proc.stdout.strip())


def restore_head(root: Path, saved: tuple[str, str]) -> None:
    kind, value = saved
    if kind == "branch":
        git(root, "checkout", value)
    else:
        git(root, "checkout", "--detach", value)


def _load_oc_check(root: Path):
    spec = importlib.util.spec_from_file_location(
        "oc_check", str(root / "tools" / "oc_check.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def changed_tests(root: Path) -> list[str]:
    return _load_oc_check(root).changed_tests(root)


def _diff_marked_lines(root: Path, base_sha: str, head_sha: str,
                        marker: str) -> list[tuple[str, str]]:
    """``[(file, raw_line)]`` for ``marker``-prefixed lines of
    ``git diff -w base_sha head_sha``, excluding the file-header lines
    (``---``/``+++``), in diff order. ``raw_line`` keeps the leading marker
    character (e.g. ``"-    foo = 1"``).
    """
    diff = git(root, "diff", "-w", base_sha, head_sha, check=True).stdout
    current_file = "?"
    header = marker * 3
    out: list[tuple[str, str]] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" ")
            current_file = parts[-1][2:] if parts[-1].startswith("b/") else parts[-1]
            continue
        if line.startswith(header):
            continue
        if line.startswith(marker):
            out.append((current_file, line))
    return out


def _normalized(raw_line: str) -> str:
    """Strip the leading diff marker and surrounding whitespace."""
    return raw_line[1:].strip()


def _recent_base_sha(root: Path, sha: str, n: int) -> str:
    """``sha~n``, clamped to the root commit when history is shorter."""
    proc = git(root, "rev-parse", f"{sha}~{n}")
    if proc.returncode == 0:
        return proc.stdout.strip()
    roots = git(root, "rev-list", "--max-parents=0", sha, check=True).stdout.split()
    return roots[0] if roots else sha


def _attribute_recent_commit(root: Path, recent_base: str, origin_main_sha: str,
                              file: str, content: str) -> str:
    """Short SHA + subject of the ``origin/main`` commit in
    ``(recent_base, origin_main_sha]`` that added ``content`` to ``file``.
    A bounded pickaxe search -- only run for the small number of actual
    tier-1 hits, never per line of the full diff.
    """
    proc = git(root, "log", "--reverse", "--format=%h\t%s",
               f"-S{content}", f"{recent_base}..{origin_main_sha}", "--", file)
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip().splitlines()[0]
    return "<unknown commit>"


def _run(root: Path, args: argparse.Namespace, saved: tuple[str, str],
         env_path: Path) -> int:
    branch = args.branch

    git(root, "fetch", "origin", "main", check=True)
    git(root, "checkout", "--detach", "FETCH_HEAD", check=True)
    origin_main_sha = git(root, "rev-parse", "HEAD", check=True).stdout.strip()

    merge = git(root, "merge", "--no-ff", branch, "-m", args.message)
    if merge.returncode != 0:
        conflicted = git(root, "diff", "--name-only", "--diff-filter=U").stdout.split()
        git(root, "merge", "--abort")
        restore_head(root, saved)
        detail = f": {', '.join(conflicted)}" if conflicted else ""
        eprint(f"land: refusing: merge conflict on {branch}{detail}")
        return 3
    merged_sha = git(root, "rev-parse", "HEAD", check=True).stdout.strip()

    # --- Deletion guard (two tiers) -----------------------------------
    # Tier 1 (hard, never overridable): the merge removes a line that
    # origin/main itself gained within its last RECENT_ADDITION_WINDOW
    # commits -- the signature of a stale-checkout revert of work that just
    # landed (the incident this guard exists for: a branch deleted a call
    # and its imports that a merge shortly before had added to main).
    # Tier 2 (soft): any other whitespace-insensitive removed lines, only
    # passable with --allow-deletions N matching the exact count.
    recent_base = _recent_base_sha(root, origin_main_sha, RECENT_ADDITION_WINDOW)
    added_recent = _diff_marked_lines(root, recent_base, origin_main_sha, "+")
    added_recent_norm = {
        n for _, line in added_recent
        if len(n := _normalized(line)) >= MIN_DELETION_GUARD_LINE_LEN
    }
    removed = _diff_marked_lines(root, origin_main_sha, merged_sha, "-")

    tier1_hits = [
        (file, line) for file, line in removed
        if len(_normalized(line)) >= MIN_DELETION_GUARD_LINE_LEN
        and _normalized(line) in added_recent_norm
    ]
    if tier1_hits:
        for file, line in tier1_hits[:MAX_DELETION_LINES_SHOWN]:
            commit = _attribute_recent_commit(
                root, recent_base, origin_main_sha, file, _normalized(line))
            eprint(f"land: deletion guard [tier 1]: {file}: {line}  <- added by {commit}")
        if len(tier1_hits) > MAX_DELETION_LINES_SHOWN:
            eprint(f"land: deletion guard [tier 1]: ... and "
                   f"{len(tier1_hits) - MAX_DELETION_LINES_SHOWN} more")
        restore_head(root, saved)
        eprint(
            f"land: refusing: this merge reverts recently-landed work -- it removes "
            f"{len(tier1_hits)} line(s) that origin/main gained within its last "
            f"{RECENT_ADDITION_WINDOW} commits (since {recent_base[:12]}). This is "
            "NOT overridable by --allow-deletions; land the revert as its own "
            "commit with its own message if that is truly intended.")
        return 10

    removed_by_file: dict[str, int] = {}
    for file, _line in removed:
        removed_by_file[file] = removed_by_file.get(file, 0) + 1
    removed_count = len(removed)

    if args.allow_deletions is not None and args.allow_deletions != removed_count:
        for file, count in sorted(removed_by_file.items()):
            eprint(f"land: deletion guard [tier 2]:   {file}: {count} line(s) removed")
        for file, line in removed[:MAX_DELETION_LINES_SHOWN]:
            eprint(f"land: deletion guard [tier 2]:   {file}: {line}")
        restore_head(root, saved)
        eprint(
            f"land: refusing: --allow-deletions {args.allow_deletions} does not "
            f"match the actual removed-line count {removed_count}")
        return 11

    if removed_count == 0:
        print("land: deletion guard: 0 line(s) removed by this merge "
              "(whitespace-insensitive) - OK")
    elif args.allow_deletions == removed_count:
        print(f"land: deletion guard [tier 2]: {removed_count} line(s) removed, "
              f"matches --allow-deletions {removed_count} - OK")
    else:
        eprint(f"land: deletion guard [tier 2]: {removed_count} line(s) removed by "
               "this merge (whitespace-insensitive), and this merge removes lines "
               "that exist on main:")
        for file, count in sorted(removed_by_file.items()):
            eprint(f"land: deletion guard [tier 2]:   {file}: {count} line(s) removed")
        for file, line in removed[:MAX_DELETION_LINES_SHOWN]:
            eprint(f"land: deletion guard [tier 2]:   {file}: {line}")
        restore_head(root, saved)
        eprint(
            "land: refusing: this merge removes lines that exist on main; pass "
            f"--allow-deletions N (N = the exact expected removed-line count, "
            f"{removed_count} here) to allow it")
        return 11

    # Computed once, up front, so hygiene scans exactly what THIS merge
    # changed (not the whole tree, and never a silent 0-file scan): a
    # --paths list built from an empty diff would be an argparse error, so
    # an empty (no-op) merge falls back to --all rather than passing no
    # paths at all -- either way hygiene always reports a real, nonzero
    # scope, never "0 file(s) ... HYGIENE OK".
    files_changed_list = git(
        root, "diff", "--name-only", origin_main_sha, merged_sha,
        check=True).stdout.split()
    hygiene_scope = ["--paths", *files_changed_list] if files_changed_list else ["--all"]

    hygiene = subprocess.run(
        [sys.executable, str(root / "checks" / "repo_hygiene.py"),
         *hygiene_scope, "--env", str(env_path)],
        cwd=str(root), capture_output=True, text=True)
    match = SECRET_COUNT_RE.search(hygiene.stdout)
    if match is None:
        restore_head(root, saved)
        eprint("land: refusing: could not parse the secret-pattern count out of "
               "hygiene output")
        eprint((hygiene.stdout + hygiene.stderr).strip())
        return 6
    secret_count = int(match.group(1))
    if secret_count == 0:
        restore_head(root, saved)
        eprint(f"land: refusing to PUSH: 0 secret pattern(s) loaded from "
               f"{env_path} - refusing to push")
        return 5
    if hygiene.returncode != 0:
        restore_head(root, saved)
        eprint("land: refusing: hygiene reported violations")
        eprint((hygiene.stdout + hygiene.stderr).strip())
        return 7

    oc = _load_oc_check(root)
    targets = oc.changed_tests(root)
    if not targets:
        tests_summary = "no tests selected by the merge diff"
    else:
        # oc.MARKERS is the same "not needs_data and not needs_corpus and
        # not heavy_host and not browser" expression CI and oc_check.py's
        # own pytest step use. needs_data/needs_corpus mark exactly the
        # tests that read the untracked data/ and fixtures/ trees, which a
        # worktree (land.py's usual root) never has -- see tests/README.md
        # "CI tests" and tests/conftest.py LOCAL_ONLY_MARKERS. Applying the
        # same deselect here means land.py can never fail a merge because
        # data/ is absent (those tests never run under it), and it never
        # claims to have exercised them either: the summary below names the
        # exclusion instead of folding it into "passed".
        pytest_run = subprocess.run(
            [sys.executable, str(root / "tools" / "bounded_run.py"),
             "--max-rss-gb", "2", "--cpu-set", args.cpu_set, "--",
             sys.executable, "-m", "pytest", "-v", "-m", oc.MARKERS, *targets],
            cwd=str(root), capture_output=True, text=True)
        # pytest's own exit code 5 means "no tests were collected" -- here,
        # every test in the selected file(s) was deselected by -m (the
        # whole-file needs_corpus case this defect was measured on). That
        # is an honest, expected outcome of the CI-marker filter, not a
        # failure: 0 vs 5 vs any other code distinguishes "ran clean",
        # "selected nothing to run", and "actually failed" without
        # collapsing the middle case into either of the outer two.
        if pytest_run.returncode not in (0, 5):
            tail = "\n".join(
                (pytest_run.stdout + pytest_run.stderr).splitlines()[-TEST_TAIL_LINES:])
            restore_head(root, saved)
            eprint(f"land: refusing: tests failed (exit {pytest_run.returncode})")
            eprint(tail)
            return 8
        marker_note = "CI-marker filter: not needs_data/needs_corpus/heavy_host/browser"
        if pytest_run.returncode == 5:
            tests_summary = (
                f"{len(targets)} test file(s) selected, 0 test(s) ran "
                f"under the {marker_note}")
        else:
            tests_summary = f"{len(targets)} test file(s) passed ({marker_note})"

    if args.dry_run:
        push_summary = "skipped (--dry-run)"
    elif args.no_push:
        push_summary = "skipped (--no-push)"
    else:
        push = git(root, "push", "origin", "HEAD:main")
        if push.returncode != 0:
            restore_head(root, saved)
            eprint("land: refusing: push failed")
            eprint((push.stdout + push.stderr).strip())
            return 9
        push_summary = "pushed"

    files_changed = len(files_changed_list)

    if args.dry_run:
        restore_head(root, saved)

    print(f"land: merged {branch} -> {merged_sha[:12]}")
    print(f"land: files changed: {files_changed}")
    print(f"land: hygiene: {secret_count} secret pattern(s) loaded, OK")
    print(f"land: tests: {tests_summary}")
    print(f"land: push: {push_summary}")
    return 0


def main(argv: list[str] | None = None, *, env_path: Path = DEFAULT_ENV_PATH,
         root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge a branch into main, gate the merge, and push.")
    parser.add_argument("branch", help="branch to merge")
    parser.add_argument("-m", "--message", required=True,
                        help="merge commit message")
    parser.add_argument("--dry-run", action="store_true",
                        help="merge, gate and test, but never push and restore "
                             "the original HEAD at the end")
    parser.add_argument("--no-push", action="store_true",
                        help="merge, gate and test, but never push; on success "
                             "leave HEAD at the new merged commit")
    parser.add_argument("--allow-deletions", type=int, default=None, metavar="N",
                        help="acknowledge tier-2 deletions: N must equal the "
                             "exact whitespace-insensitive removed-line count "
                             "for this merge, or land refuses (exit 11). Never "
                             "overrides tier 1 (exit 10), which reverts lines "
                             "origin/main gained recently.")
    parser.add_argument("--cpu-set", default=DEFAULT_TEST_CPU_SET,
                        help="taskset CPU list passed to bounded_run.py --cpu-set "
                             f"for the gated test run (default: {DEFAULT_TEST_CPU_SET}, "
                             "off the heavy-job cores 0-5)")
    args = parser.parse_args(argv)

    if root is None:
        root = repo_root()

    status = git(root, "status", "--porcelain")
    if status.stdout.strip():
        eprint("land: refusing: working tree not clean")
        eprint(status.stdout.rstrip())
        return 1

    if git(root, "rev-parse", "--verify", "--quiet",
           f"refs/heads/{args.branch}").returncode != 0:
        eprint(f"land: refusing: branch does not exist: {args.branch}")
        return 2

    if not env_path.is_file() or not os.access(env_path, os.R_OK):
        eprint(f"land: refusing: .env missing or unreadable: {env_path}")
        return 4

    saved = save_head(root)

    try:
        return _run(root, args, saved, env_path)
    except BaseException:
        git(root, "merge", "--abort")
        restore_head(root, saved)
        raise


if __name__ == "__main__":
    raise SystemExit(main())