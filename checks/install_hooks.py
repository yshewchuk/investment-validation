#!/usr/bin/env python3
"""Install the versioned pre-commit hook, and report whether it is installed.

    python3 checks/install_hooks.py            # install / update
    python3 checks/install_hooks.py --check    # report state, change nothing
    python3 checks/install_hooks.py --json     # for the nightly

``.git/hooks`` is not versioned. A fresh clone therefore has no hook at all,
and the hook a working clone does have can drift from the one in the repo
without anything saying so. Both failures look exactly like "the checks
passed", which is the worst shape a control can fail in.

So the hook lives at ``checks/hooks/pre-commit``, in the repo, and this script
copies it into place and can be asked whether the copy is current.
:mod:`checks.rearchitecture_phase0_gate` calls :func:`state` so the nightly can report it —
§4.6: a missing hook is a budget failure on the same footing, because a control
that cannot detect its own absence is not a control.

An existing hook that is neither ours nor a known predecessor is **not**
overwritten without ``--force``: somebody put it there on purpose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "checks" / "hooks" / "pre-commit"

#: The hook this replaces: the hygiene-only gate installed 2026-08-29. Known by
#: content so upgrading it is silent and safe, rather than needing --force for
#: a file we wrote ourselves.
_PREDECESSOR = "exec python3 \"$(git rev-parse --show-toplevel)/checks/repo_hygiene.py\""


def hooks_dir(root: Path = ROOT) -> Path:
    proc = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-path", "hooks"],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return root / ".git" / "hooks"
    return (root / proc.stdout.strip()).resolve()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def state(root: Path = ROOT) -> dict:
    """Whether the installed hook exists, is current, and is executable."""
    source = (root / "checks" / "hooks" / "pre-commit")
    target = hooks_dir(root) / "pre-commit"
    wanted = source.read_text() if source.exists() else ""
    if not target.exists():
        return {"installed": False, "current": False, "executable": False,
                "reason": "no pre-commit hook in .git/hooks",
                "path": str(target)}
    found = target.read_text()
    return {
        "installed": True,
        "current": _digest(found) == _digest(wanted),
        "executable": os.access(target, os.X_OK),
        "reason": ("" if _digest(found) == _digest(wanted)
                   else "installed hook differs from checks/hooks/pre-commit"),
        "path": str(target),
        "source_sha256": _digest(wanted),
        "installed_sha256": _digest(found),
    }


def install(root: Path = ROOT, *, force: bool = False) -> dict:
    source = root / "checks" / "hooks" / "pre-commit"
    if not source.exists():
        raise FileNotFoundError(f"{source} is missing")
    target = hooks_dir(root) / "pre-commit"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        found = target.read_text()
        known = _PREDECESSOR in found or _digest(found) == _digest(source.read_text())
        if not known:
            return {**state(root), "written": False,
                    "reason": "an unrecognized pre-commit hook is installed; "
                              "inspect it, then re-run with --force"}
    target.write_text(source.read_text())
    target.chmod(0o755)
    return {"written": True, **state(root)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-root", default=str(ROOT))
    ap.add_argument("--check", action="store_true", help="report, change nothing")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    result = state(root) if args.check else install(root, force=args.force)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        verb = "state" if args.check else ("installed" if result.get("written")
                                           else "NOT installed")
        print(f"pre-commit hook {verb}: {result['path']}")
        if result.get("reason"):
            print(f"  {result['reason']}")
    ok = result["installed"] and result["current"] and result["executable"]
    if not ok and not args.check:
        print("hook install FAILED", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
