#!/usr/bin/env python3
"""Every v2 package carries the §4.5 README, and two of its sections are true.

`system_rearchitecture.md` §4.5 gives each package a ``README.md`` with seven
sections. A file of seven headings rots into decoration unless something reads
it, so two of them are machine-checked against the import graph
:mod:`checks.import_layers` already parses:

* **Consumers** — a README claiming a consumer that does not import it, or
  omitting one that does, is a failure rather than a stale sentence;
* **Public interface** — importing a name absent from that list is an
  upward-equivalent violation, caught by the same pass.

Both sections carry a machine-readable directive so there is no prose parsing
and no ambiguity about what was claimed::

    <!-- consumers: engine.v2.scoring, engine.v2.serving -->
    <!-- public-interface: score_event, score_many -->

``none`` is the explicit empty value. An absent directive is a failure, because
"I forgot to declare it" and "nothing imports it" must not look the same.

The graph is built from **tracked files with the staged versions layered over
them**, never from the staged set alone: a partial graph would report every
unstaged consumer as a broken claim, and a check that fails on correct commits
is a check that gets bypassed.

Usage::

    python3 checks/package_readmes.py          # graph = tracked + staged
    python3 checks/package_readmes.py --all    # graph = worktree
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checks.layer_map import PACKAGES, README_SECTIONS, package_of  # noqa: E402
from checks.import_layers import build_graph  # noqa: E402
from checks.repo_hygiene import (  # noqa: E402
    read_staged_blob,
    read_worktree_blob,
    staged_paths,
    tracked_paths,
)

__all__ = ["Violation", "Report", "directive", "check", "main"]

_DIRECTIVE = "<!--\\s*{key}\\s*:(?P<value>[^>]*)-->"
_NONE = {"none", "(none)", ""}


@dataclass
class Violation:
    package: str
    rule: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"  [{self.rule}] {self.package}\n      {self.detail}"


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    packages: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, package: str, rule: str, detail: str) -> None:
        self.violations.append(Violation(package, rule, detail))


# --------------------------------------------------------------------------
# reading a README
# --------------------------------------------------------------------------


def directive(text: str, key: str) -> set[str] | None:
    """The declared value of a ``<!-- key: a, b -->`` directive.

    Returns an empty set for the explicit ``none``, and ``None`` when the
    directive is absent — which is a failure, not an empty declaration.
    """
    match = re.search(_DIRECTIVE.format(key=re.escape(key)), text)
    if match is None:
        return None
    raw = match.group("value").strip()
    if raw.lower() in _NONE:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def missing_sections(text: str) -> list[str]:
    headings = set(re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    return [s for s in README_SECTIONS if s not in headings]


# --------------------------------------------------------------------------
# the import graph, as consumers and used names
# --------------------------------------------------------------------------


def _package_modules(files: dict[str, bytes]) -> set[str]:
    """Every dotted module that exists as a file, for submodule/name telling."""
    out = set()
    for rel in files:
        if rel.endswith(".py"):
            parts = rel[:-3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            out.add(".".join(parts))
    return out


def observed(files: dict[str, bytes]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """``({package: consumers}, {package: names imported from outside})``."""
    modules = _package_modules(files)
    consumers: dict[str, set[str]] = {p.dotted: set() for p in PACKAGES}
    names: dict[str, set[str]] = {p.dotted: set() for p in PACKAGES}
    for edge in build_graph(files).edges:
        target = package_of(edge.imported)
        source = package_of(edge.importer)
        if target is None or source is target:
            continue
        if source is not None:
            consumers[target.dotted].add(source.dotted)
        tail = edge.imported[len(target.dotted):].lstrip(".")
        head = tail.split(".", 1)[0] if tail else ""
        if head and f"{target.dotted}.{head}" not in modules:
            names[target.dotted].add(head)
    return consumers, names


# --------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------


def _check_one(pkg, text: str | None, consumers: set[str], names: set[str],
               report: Report) -> None:
    if text is None:
        report.add(pkg.path, "missing-readme",
                   "§4.5 requires a README.md in every v2 package")
        return
    gaps = missing_sections(text)
    if gaps:
        report.add(pkg.path, "missing-section",
                   f"missing the §4.5 section(s): {', '.join(gaps)}")

    claimed = directive(text, "consumers")
    if claimed is None:
        report.add(pkg.path, "undeclared-consumers",
                   "no `<!-- consumers: ... -->` directive; write `none` to "
                   "declare that nothing imports this package")
    else:
        for extra in sorted(claimed - consumers):
            report.add(pkg.path, "consumer-not-observed",
                       f"README claims {extra} imports this package; nothing "
                       "in the tree does")
        for missing in sorted(consumers - claimed):
            report.add(pkg.path, "consumer-not-declared",
                       f"{missing} imports this package and the README does "
                       "not say so")

    interface = directive(text, "public-interface")
    if interface is None:
        report.add(pkg.path, "undeclared-interface",
                   "no `<!-- public-interface: ... -->` directive; write "
                   "`none` while the package is empty")
        return
    for name in sorted(names - interface):
        report.add(pkg.path, "private-name-imported",
                   f"{name!r} is imported from outside this package and is not "
                   "in its declared public interface — everything else is "
                   "internal regardless of underscore convention")


def check(files: dict[str, bytes], readmes: dict[str, str | None]) -> Report:
    """The pure core the CLI and the tests both drive."""
    report = Report()
    consumers, names = observed(files)
    for pkg in PACKAGES:
        report.packages += 1
        _check_one(pkg, readmes.get(pkg.dotted), consumers[pkg.dotted],
                   names[pkg.dotted], report)
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _sources(root: Path, use_worktree: bool) -> tuple[dict[str, bytes], dict[str, str | None]]:
    blobs: dict[str, bytes] = {}
    for rel in tracked_paths(root):
        blobs[rel] = read_worktree_blob(root, rel)
    if not use_worktree:
        for rel in staged_paths(root):
            blobs[rel] = read_staged_blob(root, rel)
    files = {rel: blob for rel, blob in blobs.items() if rel.endswith(".py")}
    readmes: dict[str, str | None] = {}
    for pkg in PACKAGES:
        blob = blobs.get(f"{pkg.path}/README.md")
        if blob is None:
            path = root / pkg.path / "README.md"
            blob = path.read_bytes() if path.exists() else None
        readmes[pkg.dotted] = None if blob is None else blob.decode("utf-8", "replace")
    return files, readmes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--all", action="store_true",
                    help="read the worktree rather than layering staged blobs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    report = check(*_sources(root, args.all))

    if not args.quiet:
        print(f"package READMEs: {report.packages} v2 package(s), "
              f"{len(README_SECTIONS)} required section(s) each")
    if report.ok:
        if not args.quiet:
            print("READMES OK")
        return 0
    print(f"\nREADMES FAILED — {len(report.violations)} violation(s):", file=sys.stderr)
    for v in report.violations:
        print(str(v), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
