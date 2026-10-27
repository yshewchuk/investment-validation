#!/usr/bin/env python3
"""Direct live/experiment import isolation guard (D7).

Production training/retrain/release modules (``engine/v2/models``, top-level
``tools/phase5_*.py``) may not import top-level ``experiments`` modules,
``engine.v2.research``, ``engine.v2.foundation.experiment_holdouts`` or
``engine.v2.foundation.final_holdout``. Direct ``import``/``from import``
statements only: no transitive graph walking, alias or binding resolution,
re-exports, path-root rules or the opposite import direction. A regular run
compares today's inventory against the committed baseline and fails on any
difference; the baseline is shrink-only and ``--write-baseline`` is the only
writer.
"""
import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "checks/live_experiment_isolation_baseline.json"

FORBIDDEN_ROOTS = (
    "experiments",
    "engine.v2.research",
    "engine.v2.foundation.experiment_holdouts",
    "engine.v2.foundation.final_holdout",
)


def live_files(root: Path = ROOT) -> list[str]:
    root = Path(root)
    paths = [p.relative_to(root).as_posix()
             for p in (root / "engine/v2/models").rglob("*.py")]
    paths += [p.relative_to(root).as_posix()
              for p in (root / "tools").glob("phase5_*.py")]
    return sorted(paths)


def _forbidden(target: str) -> bool:
    return any(target == r or target.startswith(r + ".") for r in FORBIDDEN_ROOTS)


def _identity(v: dict) -> tuple:
    return (v["path"], v["line"], v["target"])


def violations(root: Path = ROOT) -> list[dict]:
    found = []
    for rel in live_files(root):
        tree = ast.parse((Path(root) / rel).read_text(), filename=rel)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names if _forbidden(a.name)]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                if _forbidden(node.module):
                    targets = [node.module]
                else:
                    targets = [f"{node.module}.{a.name}" for a in node.names
                               if _forbidden(f"{node.module}.{a.name}")]
            for target in targets:
                found.append({"path": rel, "line": node.lineno, "target": target})
    return sorted(found, key=_identity)


def compare(current: list[dict], baseline: list[dict]) -> tuple[list[dict], list[dict]]:
    cur = Counter(_identity(v) for v in current)
    base = Counter(_identity(v) for v in baseline)
    cur_by_id = {_identity(v): v for v in current}
    base_by_id = {_identity(v): v for v in baseline}
    added = sorted(
        (cur_by_id[k] for k, n in (cur - base).items() for _ in range(n)),
        key=_identity,
    )
    missing = sorted(
        (base_by_id[k] for k, n in (base - cur).items() for _ in range(n)),
        key=_identity,
    )
    return added, missing


def _line(v: dict) -> str:
    return f"{v['path']}:{v['line']}: {v['target']}"


def _print_category(name: str, items: list[dict]) -> None:
    print(f"{name} ({len(items)}):", file=sys.stderr)
    for v in items:
        print(f"  {_line(v)}", file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-baseline", action="store_true",
                        help="regenerate the baseline from today's source "
                             "(shrink-only when one already exists)")
    args = parser.parse_args(argv)
    current = violations()
    if args.write_baseline:
        baseline = (json.loads(BASELINE_PATH.read_text())
                    if BASELINE_PATH.exists() else None)
        if baseline is not None:
            added, _missing = compare(current, baseline)
            if added:
                print(f"refusing to write {BASELINE_PATH.name}: proposed "
                      f"inventory adds {len(added)} entry/entries "
                      "(baseline is shrink-only)", file=sys.stderr)
                _print_category("added", added)
                return 1
        BASELINE_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n")
        return 0
    if not BASELINE_PATH.exists():
        print(f"baseline missing: {BASELINE_PATH}", file=sys.stderr)
        return 1
    added, missing = compare(current, json.loads(BASELINE_PATH.read_text()))
    if added or missing:
        _print_category("added", added)
        _print_category("missing", missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())