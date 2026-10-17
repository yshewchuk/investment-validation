"""The import-name memo in ``fingerprints._import_names``: key, hit path and bound (#791)."""
import ast
import hashlib

import pytest

from engine.v2.ops import fingerprints
from engine.v2.ops.errors import OpsError


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


def test_source_closure_digest_matches_the_file_bytes_and_a_tiny_cap_refuses(tmp_path, monkeypatch):
    (tmp_path / "m.py").write_bytes(b"import os\n")
    assert fingerprints.source_closure(tmp_path, ["m.py"]) == {
        "m.py": "sha256:" + hashlib.sha256(b"import os\n").hexdigest()}
    monkeypatch.setattr(fingerprints, "_MAX_SOURCE_BYTES", 4)
    with pytest.raises(OpsError) as caught:
        fingerprints.source_closure(tmp_path, ["m.py"])
    assert caught.value.code == "INPUT_CHANGED"


def test_source_size_cap_is_eight_mebibytes_and_its_boundary_is_exact(tmp_path):
    assert fingerprints._MAX_SOURCE_BYTES == 8 * 1024 * 1024
    limit = fingerprints._MAX_SOURCE_BYTES
    (tmp_path / "ok.py").write_bytes(b"#" * (limit - 1) + b"\n")
    assert list(fingerprints.source_closure(tmp_path, ["ok.py"])) == ["ok.py"]
    (tmp_path / "big.py").write_bytes(b"#" * (limit + 1))
    with pytest.raises(OpsError) as caught:
        fingerprints.source_closure(tmp_path, ["big.py"])
    assert caught.value.code == "INPUT_CHANGED"
    assert "exceeds the size bound" in str(caught.value)


def test_source_closure_reads_each_py_file_once_for_digest_and_imports(tmp_path, monkeypatch):
    (tmp_path / "m.py").write_bytes(b"import first_module\n")
    (tmp_path / "first_module.py").write_bytes(b"")
    (tmp_path / "second_module.py").write_bytes(b"")
    real_open = fingerprints.Path.open
    opens = []

    def open_then_swap(self, *args, **kwargs):
        if self.name == "m.py" and args[:1] == ("rb",):
            opens.append(1)
            if len(opens) == 1:
                stream = real_open(self, *args, **kwargs)
                # Swap the file after the first read so a second read would see different bytes.
                self.unlink()
                self.write_bytes(b"import second_module\n")
                return stream
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(fingerprints.Path, "open", open_then_swap)
    closure = fingerprints.source_closure(tmp_path, ["m.py"])
    assert len(opens) == 1
    assert closure["m.py"] == "sha256:" + hashlib.sha256(b"import first_module\n").hexdigest()
    assert set(closure) == {"m.py", "first_module.py"}


def test_source_closure_hashes_a_non_py_entry_without_import_expansion(tmp_path):
    (tmp_path / "data.txt").write_bytes(b"import os\n")
    (tmp_path / "os.py").write_bytes(b"")
    assert fingerprints.source_closure(tmp_path, ["data.txt"]) == {
        "data.txt": "sha256:" + hashlib.sha256(b"import os\n").hexdigest()}


def test_unresolved_import_is_omitted_without_refusal(tmp_path):
    (tmp_path / "m.py").write_bytes(b"import not_in_the_repo\n")
    assert list(fingerprints.source_closure(tmp_path, ["m.py"])) == ["m.py"]


def test_undecodable_py_lets_the_builtin_error_escape(tmp_path):
    (tmp_path / "m.py").write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        fingerprints.source_closure(tmp_path, ["m.py"])


def test_absolute_from_import_names_the_module_and_each_imported_name():
    fingerprints._IMPORT_NAMES.clear()
    assert fingerprints._import_names("from pkg.mod import a, b\n", ["x"]) == ("pkg.mod", "pkg.mod.a", "pkg.mod.b")


def test_source_closure_resolves_each_dotted_name_at_most_once_per_call(tmp_path, monkeypatch):
    for rel, text in {
        "a.py": "import pkg.x\nimport pkg.y\n",
        "b.py": "import pkg.x\nimport pkg.y\n",
        "pkg/__init__.py": "",
        "pkg/x.py": "",
        "pkg/y.py": "",
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    counts: dict[str, int] = {}
    real = fingerprints._module_files

    def counting(root, name):
        counts[name] = counts.get(name, 0) + 1
        return real(root, name)

    monkeypatch.setattr(fingerprints, "_module_files", counting)
    closure = fingerprints.source_closure(tmp_path, ["a.py", "b.py"])
    assert set(counts) == {"pkg.x", "pkg.y"}
    assert all(count == 1 for count in counts.values())
    assert sum(counts.values()) == len(set(counts))
    assert set(closure) == {"a.py", "b.py", "pkg/__init__.py", "pkg/x.py", "pkg/y.py"}
