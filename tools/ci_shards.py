#!/usr/bin/env python3
"""CI test shards: print the workflow matrix or one shard's test files.

Declaration: ``tests/shards.toml`` (ordered ``[[shard]]`` tables; the first
shard whose fnmatch pattern matches a file owns it; the last shard is the
remainder and owns every unclaimed file). Checked by
``tests/v2/integration/test_ci_shards.py``.

    python3 tools/ci_shards.py matrix
    python3 tools/ci_shards.py files SHARD [--selection FILE]

``matrix`` prints compact JSON ``{"include": [{"shard": NAME}, ...]}`` for a
GitHub Actions ``fromJSON`` matrix. ``files`` prints SHARD's test files, one
per line. With ``--selection FILE`` (the PR test-selection list from
``tools/mutation_pilot.py select-tests``) it prints only the selected files
that SHARD owns: ``__ALL__`` as first line selects every file, an empty file
selects none. Comment lines (``#``) are ignored. An empty result prints
nothing and exits 0 (the shard skips).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("INVESTING_PLAN_ROOT") or Path(__file__).resolve().parents[1])
SHARDS_TOML = ROOT / "tests" / "shards.toml"
ALL = "__ALL__"


@dataclass(frozen=True)
class Shard:
    name: str
    patterns: tuple[str, ...]
    remainder: bool


def load_shards(path: Path = SHARDS_TOML) -> list[Shard]:
    """Parse and validate the declaration; raise ValueError on a bad one."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh).get("shard", [])
    if not isinstance(raw, list):
        raise ValueError("`shard` must be an array of [[shard]] tables")
    shards: list[Shard] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"shard #{i} must be a table")
        unknown = set(entry) - {"name", "patterns", "remainder"}
        if unknown:
            raise ValueError(f"shard #{i} has unknown keys: {sorted(unknown)}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"shard #{i} has no name")
        remainder = entry.get("remainder", False)
        if not isinstance(remainder, bool):
            raise ValueError(f"shard {name!r}: remainder must be a boolean")
        raw_patterns = entry.get("patterns", [])
        if not isinstance(raw_patterns, list) or not all(isinstance(p, str) for p in raw_patterns):
            raise ValueError(f"shard {name!r}: patterns must be a list of strings")
        patterns = tuple(raw_patterns)
        if remainder and patterns:
            raise ValueError(f"remainder shard {name!r} must not list patterns")
        if not remainder and not patterns:
            raise ValueError(f"shard {name!r} has no patterns and is not the remainder")
        shards.append(Shard(name, patterns, remainder))
    names = [s.name for s in shards]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate shard names: {names}")
    if [s.remainder for s in shards].count(True) != 1 or not shards[-1].remainder:
        raise ValueError("exactly one remainder shard is required, and it must be last")
    return shards


def discover_test_files(root: Path = ROOT) -> list[str]:
    """Every pytest test file under tests/, as sorted posix paths from root."""
    tests = root / "tests"
    found = {p.relative_to(root).as_posix()
             for pattern in ("test_*.py", "*_test.py") for p in tests.rglob(pattern)}
    return sorted(f for f in found if "__pycache__" not in f)


def owner(path: str, shards: list[Shard]) -> str:
    """The shard that owns ``path``: first pattern match, else the remainder."""
    for shard in shards:
        if any(fnmatch.fnmatchcase(path, pat) for pat in shard.patterns):
            return shard.name
    return shards[-1].name


def assign(files: list[str], shards: list[Shard]) -> dict[str, list[str]]:
    """Partition ``files`` over the shards (every shard present, maybe empty)."""
    out: dict[str, list[str]] = {s.name: [] for s in shards}
    for path in files:
        out[owner(path, shards)].append(path)
    return out


def read_selection(path: Path) -> list[str] | None:
    """The selected files, or None when the selection is ``__ALL__``.

    Lines starting with ``#`` are comments and are ignored before the
    ``__ALL__``-first-line rule is applied (``cmd_select_tests`` appends
    reason comments after the file lines)."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("#")]
    if lines and lines[0] == ALL:
        return None
    return lines


def shard_files(name: str, selection: Path | None = None) -> list[str]:
    shards = load_shards()
    if name not in {s.name for s in shards}:
        raise KeyError(name)
    files = discover_test_files()
    if selection is not None:
        chosen = read_selection(selection)
        if chosen is not None:
            files = sorted(set(chosen) & set(files))
    return assign(files, shards)[name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("matrix")
    files_p = sub.add_parser("files")
    files_p.add_argument("shard")
    files_p.add_argument("--selection", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "matrix":
        print(json.dumps({"include": [{"shard": s.name} for s in load_shards()]},
                         separators=(",", ":")))
        return 0
    try:
        files = shard_files(args.shard, args.selection)
    except KeyError:
        print(f"unknown shard {args.shard!r}", file=sys.stderr)
        return 2
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
