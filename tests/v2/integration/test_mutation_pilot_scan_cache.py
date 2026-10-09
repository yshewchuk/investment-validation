"""Regression: mutation_pilot's bounded in-process full-repository scan caches.

Synthetic only: `REPO` and the tracked-path source are redirected to a single
temporary Python file and `ast.parse` is counted, so neither the default
`build_import_graph()` nor `unresolved_import_files()` ever scans the real
checkout.

# packages: ops
"""
from __future__ import annotations

import os
from pathlib import Path

from tools import mutation_pilot as pilot

REL = "scan_target.py"


def _install_parse_spy(monkeypatch):
    real_parse = pilot.ast.parse
    counts: dict[str, int] = {}

    def spy(source, filename="<unknown>", *args, **kwargs):
        counts[filename] = counts.get(filename, 0) + 1
        return real_parse(source, filename, *args, **kwargs)

    monkeypatch.setattr(pilot.ast, "parse", spy)
    return counts


def _redirect(monkeypatch, tmp_path: Path, text: str = "import os\n") -> Path:
    target = tmp_path / REL
    target.write_text(text)
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    monkeypatch.setattr(pilot, "_tracked", lambda paths: [REL])
    return target


def test_default_graph_and_repeated_unresolved_scan_reuse_cached_results(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    counts = _install_parse_spy(monkeypatch)

    graph = pilot.build_import_graph()
    graph[REL].add("sentinel")
    again = pilot.build_import_graph()
    assert "sentinel" not in again[REL]  # defensive copy, never the cached object
    assert counts[REL] == 1  # the repeat never reparsed

    first = pilot.unresolved_import_files([REL])
    first.add("sentinel")
    second = pilot.unresolved_import_files([REL])
    assert "sentinel" not in second
    assert counts[REL] == 2  # graph once + unresolved once; the repeats cached


def test_rewriting_the_source_with_a_new_size_invalidates_both_caches(tmp_path, monkeypatch):
    target = _redirect(monkeypatch, tmp_path)
    counts = _install_parse_spy(monkeypatch)

    pilot.build_import_graph()
    pilot.unresolved_import_files([REL])
    assert counts[REL] == 2

    target.write_text("import os\nimport sys\n")  # same path, different st_size
    pilot.build_import_graph()
    assert counts[REL] == 3  # graph cache key (size) moved, so it reparsed
    pilot.unresolved_import_files([REL])
    assert counts[REL] == 4  # unresolved cache moved too, so it reparsed


def test_same_size_rewrite_with_restored_mtime_invalidates_both_caches(tmp_path, monkeypatch):
    target = _redirect(monkeypatch, tmp_path)
    counts = _install_parse_spy(monkeypatch)

    pilot.build_import_graph()
    pilot.unresolved_import_files([REL])
    assert counts[REL] == 2

    before = target.stat()
    target.write_text("import re\n")  # same path, exactly the same byte length
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = target.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns  # the write moved ctime

    pilot.build_import_graph()
    assert counts[REL] == 3  # graph cache key moved (ctime), so it reparsed
    pilot.unresolved_import_files([REL])
    assert counts[REL] == 4  # unresolved cache moved too, so it reparsed


def test_explicit_tracked_builds_stay_uncached(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    counts = _install_parse_spy(monkeypatch)

    pilot.build_import_graph([REL])
    pilot.build_import_graph([REL])
    assert counts[REL] == 2  # an explicit tracked list is parsed every call
