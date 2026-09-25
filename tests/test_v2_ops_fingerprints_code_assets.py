"""The plan's code snapshot must cover non-``.py`` files legacy/v2 code reads
relative to its own code location, not just ``.py`` files -- same defect
class as ``tests/test_v2_ops_fingerprints_model_scan.py``'s champion-pickle
gap, this time for ``Path(__file__).resolve().parent / "..."`` reads.

Real shadow attempt 15, ``legacy_render`` (``att_4b53399913701df33932746a4ecaab9e``,
``att_ca847a89bd184778f870648132c06c07``; ``staging/diagnostics/worker.stderr``):
``FileNotFoundError: dashboard static templates missing at
/root/phase2-shadow-ops/code/<impl_hash>/engine/dashboard/static``, raised at
``engine/dashboard/render.py:1090`` (``_copy_static``) from the path built at
``render.py:1079`` (``Path(__file__).resolve().parent / "static"``).
``engine/v2/ops/fingerprints.py::worker_source_manifest`` only ever walked
``.py`` files via static import analysis (``source_closure``/``_imports``),
so nothing about ``engine/dashboard/static/`` was ever in the manifest, and
``snapshot_code`` (driven entirely by the manifest's keys) never copied it.

Current design, mirroring :data:`MODEL_PICKLE_MODULES`:

* ``fingerprints.CODE_ASSET_FILES``/``CODE_ASSET_DIRS`` are declared,
  checked-in tuples. ``worker_source_manifest`` always includes them
  (``_code_asset_entries``), content-hashed exactly like a ``.py`` entry, so
  they are part of ``implementation_ref`` and ``snapshot_code`` copies them.
* A missing declared file or directory refuses ``INPUT_CHANGED`` at plan time
  (``_code_asset_entries``/``source_closure``), the same shape as
  ``_declared_module_entries`` for a missing model-pickle module.
* ``_assert_code_asset_path_safe`` refuses any declared path resolving under
  the repo-root ``data/`` store or named ``.env`` -- the code-asset closure
  must never carry data or secrets.
* This file's own regex guard (``_scan_file_relative_reads``) keeps the
  declared tuples honest: it statically scans ``engine/`` for
  ``Path(__file__).resolve().parent / "literal"`` reads and fails if one
  is not covered by ``CODE_ASSET_FILES``/``CODE_ASSET_DIRS`` (or does not
  end in ``.py``, a different, already-declared-elsewhere concern).
"""
from __future__ import annotations
# land: always-run

import re
from pathlib import Path

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import (
    CODE_ASSET_DIRS,
    CODE_ASSET_FILES,
    _assert_code_asset_path_safe,
    file_hash,
    snapshot_code,
    worker_source_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Mirrors _BASE_FILES in test_v2_ops_fingerprints_model_scan.py: the
#: trivial stand-ins that make source_closure's own static walk a no-op
#: beyond the model-pickle/code-asset declarations these tests exercise.
_BASE_FILES = {
    "engine/__init__.py": "",
    "engine/v2/__init__.py": "",
    "engine/v2/ops/__init__.py": "",
    "engine/v2/ops/worker.py": "# stub entrypoint, no imports of interest\n",
    "engine/models/__init__.py": "",
    "engine/models/ensemble.py": "",
    "engine/models/registry.py": "",
    "engine/models/training/__init__.py": "",
    "engine/models/training/common.py": "",
    "engine/models/training/runup_move.py": "",
}


def _write(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _seed_base(root: Path) -> None:
    for relative, content in _BASE_FILES.items():
        _write(root, relative, content)


def _seed_code_assets(root: Path) -> None:
    for relative in CODE_ASSET_FILES:
        _write(root, relative, "{}")
    for relative in CODE_ASSET_DIRS:
        _write(root, f"{relative}/index.html", "<html></html>")
        _write(root, f"{relative}/assets/app.css", "body{}")
        _write(root, f"{relative}/assets/app.js", "console.log(1)")


# --------------------------------------------------------------------------
# worker_source_manifest: declared code assets, unconditional on data/
# --------------------------------------------------------------------------


def test_code_assets_join_the_snapshot_without_any_data_dir(tmp_path):
    _seed_base(tmp_path)
    _seed_code_assets(tmp_path)
    assert not (tmp_path / "data").exists()

    manifest = worker_source_manifest(tmp_path)

    for relative in CODE_ASSET_FILES:
        assert relative in manifest, f"{relative} missing from manifest"
        assert manifest[relative] == file_hash(tmp_path / relative)
    for relative in CODE_ASSET_DIRS:
        directory = tmp_path / relative
        expected_files = [p for p in directory.rglob("*") if p.is_file()]
        assert expected_files, "test fixture must seed at least one file per declared dir"
        for path in expected_files:
            rel = path.relative_to(tmp_path).as_posix()
            assert rel in manifest, f"{rel} missing from manifest"
            assert manifest[rel] == file_hash(path)


def test_declared_code_asset_file_missing_refuses_at_plan_time(tmp_path):
    _seed_base(tmp_path)
    for relative in CODE_ASSET_DIRS:
        _write(tmp_path, f"{relative}/index.html", "<html></html>")
    # No CODE_ASSET_FILES written.

    with pytest.raises(OpsError) as excinfo:
        worker_source_manifest(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"


def test_declared_code_asset_dir_missing_refuses_at_plan_time(tmp_path):
    _seed_base(tmp_path)
    for relative in CODE_ASSET_FILES:
        _write(tmp_path, relative, "{}")
    # No CODE_ASSET_DIRS written at all -- must not silently expand to zero files.

    with pytest.raises(OpsError) as excinfo:
        worker_source_manifest(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.details["path"] in CODE_ASSET_DIRS


def test_manifest_identical_with_and_without_a_data_dir(tmp_path):
    """implementation_ref must not depend on whether data/ happens to sit
    next to the code -- the same property test_v2_ops_fingerprints_model_scan.py
    pins for MODEL_PICKLE_MODULES, now for the code-asset closure."""
    _seed_base(tmp_path)
    _seed_code_assets(tmp_path)
    without_data = worker_source_manifest(tmp_path)

    _write(tmp_path, "data/features/panel.parquet", "not real parquet")
    with_data = worker_source_manifest(tmp_path)

    assert without_data == with_data


# --------------------------------------------------------------------------
# safety: no data/ store paths, no secrets, ever
# --------------------------------------------------------------------------


def test_code_asset_path_under_repo_root_data_is_refused():
    with pytest.raises(OpsError) as excinfo:
        _assert_code_asset_path_safe("data/features/panel.parquet")
    assert excinfo.value.code == "INPUT_CHANGED"


def test_code_asset_path_named_dotenv_is_refused():
    with pytest.raises(OpsError) as excinfo:
        _assert_code_asset_path_safe(".env")
    assert excinfo.value.code == "INPUT_CHANGED"


def test_code_asset_path_under_a_code_package_literally_named_data_is_allowed():
    """engine/v2/data/ is a code PACKAGE, not the Tier-1/2/3 store -- only
    the leading path segment is checked against the repo-root data/ store."""
    _assert_code_asset_path_safe("engine/v2/data/legacy_annotations.json")  # must not raise


# --------------------------------------------------------------------------
# snapshot_code + render: the real static dir, copied and served from a
# snapshot directory that is NOT the repo checkout.
# --------------------------------------------------------------------------


def test_real_snapshot_carries_dashboard_static_with_matching_hashes(tmp_path):
    manifest = worker_source_manifest(REPO_ROOT)
    destination = tmp_path / "snapshot"
    snapshot_code(REPO_ROOT, destination, manifest)

    static_dir = REPO_ROOT / "engine" / "dashboard" / "static"
    real_files = [p for p in static_dir.rglob("*") if p.is_file()]
    assert real_files, "engine/dashboard/static must carry real files for this test to mean anything"
    for path in real_files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        snapshot_path = destination / rel
        assert snapshot_path.is_file(), f"{rel} missing from the snapshot"
        assert file_hash(snapshot_path) == file_hash(path)

    annotations_rel = "engine/v2/data/legacy_annotations.json"
    assert (destination / annotations_rel).is_file()
    assert file_hash(destination / annotations_rel) == file_hash(REPO_ROOT / annotations_rel)


def test_render_copy_static_runs_from_the_snapshot_directory_not_the_repo(tmp_path):
    """The exact failure this fixes: legacy_render's _copy_static must find
    engine/dashboard/static under a SNAPSHOT directory, never falling back to
    (or needing) the live repo checkout."""
    from engine.dashboard.render import _copy_static

    manifest = worker_source_manifest(REPO_ROOT)
    destination = tmp_path / "snapshot"
    snapshot_code(REPO_ROOT, destination, manifest)

    snapshot_static = destination / "engine" / "dashboard" / "static"
    assert snapshot_static.is_dir()

    out = tmp_path / "bundle"
    out.mkdir()
    files = _copy_static(out, static_dir=snapshot_static)
    assert files
    assert (out / "index.html").is_file()


# --------------------------------------------------------------------------
# guard: no new Path(__file__)-relative non-.py read is left undeclared
# --------------------------------------------------------------------------

_FILE_RELATIVE_READ = re.compile(
    r"""Path\(__file__\)\.resolve\(\)\.parent\s*/\s*(['"])([^'"]+)\1"""
)


def _scan_file_relative_reads(root: Path) -> list[tuple[str, str]]:
    """``(declaring .py file, literal)`` for every
    ``Path(__file__).resolve().parent / "literal"`` under ``engine/``. A
    ``.py`` literal is a different, already-handled concern (a code module
    reached only by dynamic loading, the same class :data:`MODEL_PICKLE_MODULES`
    exists for) -- this guard is scoped to non-``.py`` runtime assets."""
    hits = []
    for path in sorted((root / "engine").rglob("*.py")):
        text = path.read_text()
        for match in _FILE_RELATIVE_READ.finditer(text):
            literal = match.group(2)
            if literal.endswith(".py"):
                continue
            hits.append((path.relative_to(root).as_posix(), literal))
    return hits


def _covered(declaring_file: str, literal: str) -> bool:
    candidate = (Path(declaring_file).parent / literal).as_posix()
    return candidate in CODE_ASSET_FILES or candidate in CODE_ASSET_DIRS


def test_no_undeclared_file_relative_non_py_read_in_the_real_tree():
    hits = _scan_file_relative_reads(REPO_ROOT)
    assert hits, "the scan itself must find the two known real hits, or it is not scanning"
    undeclared = [(f, literal) for f, literal in hits if not _covered(f, literal)]
    assert not undeclared, (
        "undeclared Path(__file__)-relative non-.py read(s) found -- add each to "
        f"CODE_ASSET_FILES or CODE_ASSET_DIRS in engine/v2/ops/fingerprints.py: {undeclared}"
    )


def test_scan_flags_a_synthetic_undeclared_read(tmp_path):
    _write(tmp_path, "engine/__init__.py", "")
    _write(tmp_path, "engine/widget/__init__.py", "")
    _write(
        tmp_path, "engine/widget/thing.py",
        'from pathlib import Path\n'
        'ASSET = Path(__file__).resolve().parent / "widget_template.html"\n',
    )
    # widget_template.html is never written and never declared.

    hits = _scan_file_relative_reads(tmp_path)
    assert ("engine/widget/thing.py", "widget_template.html") in hits
    assert not _covered("engine/widget/thing.py", "widget_template.html")
