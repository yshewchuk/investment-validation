#!/usr/bin/env python3
"""The test-layout ratchet (R1-R7): root tests move into ``tests/v2``, never back.

``checks/test_layout_budget.txt`` records the exact root-level ``tests/test_*.py``
count. Run directly as a CLI. See ``guides/test_selection_by_layer.md``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks import test_selection as ts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = "checks/test_layout_budget.txt"
V2 = ts.TESTS


def _git_env():
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(name, None)
    return env


@dataclass
class Report:
    findings: list[str] = field(default_factory=list)

    @property
    def ok(self):
        return not self.findings


def is_test(path):
    name = path.rsplit("/", 1)[-1]
    return (path.startswith("tests/") and path.endswith(".py")
            and (name.startswith("test_") or name.endswith("_test.py")))


def _rooted(path):
    return (path.count("/") == 1 and path.startswith("tests/")
            and path.endswith(".py")
            and path.rsplit("/", 1)[-1].startswith("test_"))


def _owned(path):
    return None if path.startswith(ts.INTEGRATION) else ts.package_for(path, V2)


def check_layout(base_paths, head_paths, base_budget, head_budget, root=ROOT,
                 modified_paths=()):
    """The pure ratchet core: R1-R7 over base/head path sets and budgets.

    ``modified_paths`` are tracked paths whose content changed in place (git
    status M, renames disabled); a move or deletion never appears there.
    """
    base, head = set(base_paths), set(head_paths)
    total = sum(1 for path in head if _rooted(path))
    base_total = sum(1 for path in base if _rooted(path))
    found = []
    for path in sorted(head - base):  # R1: a new test outside the layout
        if is_test(path) and not (path.startswith(ts.INTEGRATION) or _owned(path)):
            found.append(f"{path}: new test outside tests/v2/<package>/ or {ts.INTEGRATION}")
    for path in sorted(set(modified_paths)):  # R7: a root test edited in place
        if _rooted(path) and path in head:
            found.append(f"{path}: root-level test modified in place; git mv it to "
                         "tests/v2/<package>/ (or the package test dir) and decrease "
                         f"{BUDGET_PATH} by one")
    if total > base_total:  # R2: the unmoved set grew
        found.append(f"root test set grew from {base_total} to {total}")
    if head_budget != total:  # R3: stale budget
        found.append(f"budget {head_budget} != root test count {total}")
    if head_budget > base_budget:  # R3: budget increased
        found.append(f"budget grew from {base_budget} to {head_budget}")
    if base_budget - head_budget != base_total - total:  # R3/R5: move accounting
        found.append("budget change does not match the root test count change")
    for path in sorted(base | head):
        if (path in head and path.startswith(V2) and not path.startswith(ts.INTEGRATION)
                and path.endswith(".py") and _owned(path) is None):  # R4: unknown package
            found.append(f"{path}: names no package in checks/layer_map.py")
        if path not in head or not is_test(path):  # R4: declaration strictness
            continue
        if path.startswith(ts.INTEGRATION):
            _, error = ts.declaration(root, path, ts.NAMES)
        elif path.startswith(V2) and ts._DECL.search(ts.read(root, path) or ""):
            error = "'# packages:' declaration is misplaced"
        else:
            continue
        if error is not None:
            found.append(f"{path}: {error}")
    return Report(found)


def _paths(root, *args):
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          env=_git_env())
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "git failed")
    return [path for path in proc.stdout.split("\0") if path]


def main(argv=None):
    parser = argparse.ArgumentParser(description="test-layout ratchet")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    try:
        base = _paths(root, "ls-tree", "-r", "--name-only", "-z", args.base_ref)
        proc = subprocess.run(["git", "-C", str(root), "show",
                               f"{args.base_ref}:{BUDGET_PATH}"], capture_output=True,
                              env=_git_env())
        base_budget = (int(proc.stdout.decode().strip()) if proc.returncode == 0
                       else sum(1 for path in base if _rooted(path)))
        budget = int((root / BUDGET_PATH).read_text().strip())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"test-layout ratchet: cannot read base or budget: {exc}", file=sys.stderr)
        return 1
    head = _paths(root, "ls-files", "-z") + _paths(
        root, "ls-files", "--others", "--exclude-standard", "-z")
    try:
        modified = _paths(root, "diff", "--name-only", "--no-renames",
                          "--diff-filter=M", "-z", args.base_ref, "--")
    except RuntimeError as exc:
        print(f"test-layout ratchet: cannot diff against base: {exc}", file=sys.stderr)
        return 1
    report = check_layout(base, head, base_budget, budget, root, modified)
    if report.ok:
        if not args.quiet:
            print("TEST LAYOUT OK")
        return 0
    print("test-layout ratchet FAILED:", file=sys.stderr)
    for finding in report.findings:
        print(f"  {finding}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
