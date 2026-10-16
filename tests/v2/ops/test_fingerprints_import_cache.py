"""The import-name memo in ``fingerprints._import_names``: key, hit path and bound (#791)."""
import ast
import hashlib

from engine.v2.ops import fingerprints


def test_identical_text_under_two_packages_resolves_relative_imports_differently():
    fingerprints._IMPORT_NAMES.clear()
    text = "from . import x\n"
    assert fingerprints._import_names(text, ["a", "b"]) == ("a.b", "a.b.x")
    assert fingerprints._import_names(text, ["c"]) == ("c", "c.x")
    assert fingerprints._import_names(text, ["a", "b"]) == ("a.b", "a.b.x")
    assert fingerprints._import_names(text, ["a", "c"]) == ("a.c", "a.c.x")


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
    monkeypatch.setattr(fingerprints, "_IMPORT_NAMES_LIMIT", 3)
    calls = []
    real = ast.parse
    monkeypatch.setattr(fingerprints.ast, "parse", lambda *a, **k: calls.append(1) or real(*a, **k))
    for name in ("a", "b", "c"):
        assert fingerprints._import_names(f"import {name}\n", []) == (name,)
    assert len(calls) == 3 and len(fingerprints._IMPORT_NAMES) == 3
    for name in ("a", "b", "c"):
        assert fingerprints._import_names(f"import {name}\n", []) == (name,)
    assert len(calls) == 3  # all three still hit: the cache is not cleared before it is full
    assert fingerprints._import_names("import d\n", []) == ("d",)
    assert len(calls) == 4 and len(fingerprints._IMPORT_NAMES) == 1
    assert fingerprints._import_names("import a\n", []) == ("a",)
    assert len(calls) == 5


def test_default_bound_is_twenty_thousand_entries():
    assert fingerprints._IMPORT_NAMES_LIMIT == 20000


def test_import_free_source_is_cached_too(monkeypatch):
    fingerprints._IMPORT_NAMES.clear()
    calls = []
    real = ast.parse
    monkeypatch.setattr(fingerprints.ast, "parse", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert fingerprints._import_names("x = 1\n", []) == ()
    assert fingerprints._import_names("x = 1\n", []) == ()
    assert len(calls) == 1


def test_cache_stores_a_digest_key_not_the_source_text():
    fingerprints._IMPORT_NAMES.clear()
    fingerprints._import_names("import os\n", ["p"])
    (digest, package), names = next(iter(fingerprints._IMPORT_NAMES.items()))
    assert digest == hashlib.sha256(b"import os\n").digest()
    assert package == ("p",) and names == ("os",)


def test_source_closure_derives_the_package_from_the_file_path(tmp_path):
    for rel, text in {
        "pkg/__init__.py": "",
        "pkg/sub/__init__.py": "",
        "pkg/sub/mod.py": "from . import sibling\nfrom .. import top\n",
        "pkg/sub/sibling.py": "",
        "pkg/top.py": "",
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    closure = fingerprints.source_closure(tmp_path, ["pkg/sub/mod.py"])
    assert set(closure) == {"pkg/__init__.py", "pkg/sub/__init__.py", "pkg/sub/mod.py",
                            "pkg/sub/sibling.py", "pkg/top.py"}
