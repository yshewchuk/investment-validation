#!/usr/bin/env python3
"""land: merge a sub-agent branch into main, gate it, and push.

Replaces the manual sequence a supervisor used to run by hand:

  1. refuse unless the working tree is clean and the branch exists;
  2. fetch ``origin/main`` and detach at it;
  3. ``git merge --no-ff <branch>``;
  4. run repo hygiene with the real ``.env`` -- fail CLOSED when zero secret
     patterns were loaded (a hygiene check with an inactive value-grep must
     never wave a push through);
  5. run the tests selected by the merge diff, via tools/oc_check.py's own
     ``changed_tests``;
  6. ``git push origin HEAD:main`` (unless ``--dry-run``/``--no-push``).

Usage::

    python3 tools/land.py <branch> -m "<merge message>" [--dry-run] [--no-push]

Exit codes: 0 success; 1 working tree not clean; 2 branch does not exist;
3 merge conflict (aborted, HEAD restored); 4 ``.env`` missing/unreadable;
5 refusing to PUSH (0 secret patterns loaded); 6 unparsable hygiene output;
7 hygiene violations; 8 tests failed (HEAD restored); 9 push failed
(HEAD restored).
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
TEST_TAIL_LINES = 40


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


def changed_tests(root: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "oc_check", str(root / "tools" / "oc_check.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.changed_tests(root)


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

    hygiene = subprocess.run(
        [sys.executable, str(root / "checks" / "repo_hygiene.py"),
         "--all", "--env", str(env_path)],
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

    targets = changed_tests(root)
    if not targets:
        tests_summary = "no tests selected by the merge diff"
    else:
        pytest_run = subprocess.run(
            [sys.executable, str(root / "tools" / "bounded_run.py"),
             "--max-rss-gb", "2", "--cores", "4", "--",
             sys.executable, "-m", "pytest", "-q", *targets],
            cwd=str(root), capture_output=True, text=True)
        if pytest_run.returncode != 0:
            tail = "\n".join(
                (pytest_run.stdout + pytest_run.stderr).splitlines()[-TEST_TAIL_LINES:])
            restore_head(root, saved)
            eprint(f"land: refusing: tests failed (exit {pytest_run.returncode})")
            eprint(tail)
            return 8
        tests_summary = f"{len(targets)} test file(s) passed"

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

    files_changed = len(git(
        root, "diff", "--name-only", origin_main_sha, merged_sha,
        check=True).stdout.split())

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