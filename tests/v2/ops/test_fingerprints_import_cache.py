"""The import-name memo in ``fingerprints._import_names``: key, hit path and bound (#791)."""
import ast

from engine.v2.ops import fingerprints


def test_identical_text_under_two_packages_resolves_relative_imports_differently():
    fingerprints._IMPORT_NAMES.clear()
    text = "from . import x\n"
    assert fingerprints._import_names(text, ["a", "b"]) == ("a.b", "a.b.x")
    assert fingerprints._import_names(text, ["c"]) == ("c", "c.x")


def test_repeat_call_hits_the_cache_without_reparsing(monkeypatch):
    fingerprints._IMPORT_NAMES.clear()
    calls = []
    real = ast.parse
    monkeypatch.setattr(fingerprints.ast, "parse", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = fingerprints._import_names("import os\n", ["p"])
    assert fingerprints._import_names("import os\n", ["p"]) == first == ("os",)
    assert len(calls) == 1
    fingerprints._import_names("import os\n", ["q"])
    fingerprints._import_names("import sys\n", ["p"])
    assert len(calls) == 3


def test_cache_is_cleared_at_the_limit_and_names_stay_correct(monkeypatch):
    fingerprints._IMPORT_NAMES.clear()
    monkeypatch.setattr(fingerprints, "_IMPORT_NAMES_LIMIT", 2)
    for name in ("a", "b", "c"):
        assert fingerprints._import_names(f"import {name}\n", []) == (name,)
    assert len(fingerprints._IMPORT_NAMES) == 1
    assert fingerprints._import_names("import a\n", []) == ("a",)
