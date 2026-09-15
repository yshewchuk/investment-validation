"""The plan's code snapshot must cover what pinned model artifacts load, not
just what ``.py`` files statically import.

Real shadow nightly attempt 13 (root ``/root/phase2-shadow-ops``, job
``legacy_score``): the DYN-SV chooser champion's joblib pickle references
``engine.models.ensemble`` via a bare STACK_GLOBAL opcode. Nothing in this
repo's ``.py`` files imports that module by name (only ``engine.score``'s
static import of ``engine.models.registry`` reaches the registry, never the
custom ensemble class a pickle instantiates), so the static AST closure in
``engine/v2/ops/fingerprints.py::source_closure`` never picked it up and the
worker's ``joblib.load`` hit ``ModuleNotFoundError`` in a snapshot that had
copied only the modules something imports.

``engine/v2/ops/fingerprints.py::worker_source_manifest`` now also scans
every champion artifact's pickle (``_model_module_entries``) and either adds
the referenced ``engine.*`` modules (with their own static closure) or
refuses ``INPUT_CHANGED`` when a referenced module does not exist anywhere
in the source tree.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import joblib
import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import file_hash, worker_source_manifest

#: Every worker_source_manifest entry the fixed worker entrypoint needs,
#: mirrored as trivial stubs so source_closure's own static walk is a no-op
#: beyond them (this test is only about the model-artifact addition).
_BASE_FILES = {
    "engine/__init__.py": "",
    "engine/v2/__init__.py": "",
    "engine/v2/ops/__init__.py": "",
    "engine/v2/ops/worker.py": "# stub entrypoint, no imports of interest\n",
    "engine/models/__init__.py": "",
}


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_registry(root: Path, *, model_id: str, artifact_relative: str,
                    artifact_bytes: bytes) -> None:
    registry = {"version": 1, "models": [
        {"id": model_id, "champion": True, "produces": None,
         "artifact": artifact_relative,
         "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest()},
    ]}
    _write(root, "engine/models/registry.json", json.dumps(registry))


def _dump_pickle_referencing(root: Path, artifact_relative: str, *, dotted_module: str,
                             class_source: str = "class Thing:\n    def __init__(self):\n        self.x = 1\n") -> bytes:
    """A real joblib artifact whose pickle GLOBALs ``dotted_module``: a
    throwaway module is installed into ``sys.modules`` under that exact
    dotted name so ``type(obj).__module__`` (what pickle actually writes) is
    genuinely ``dotted_module``, then removed once dumped."""
    module = types.ModuleType(dotted_module)
    exec(compile(class_source, dotted_module, "exec"), module.__dict__)  # noqa: S102
    sys.modules[dotted_module] = module
    try:
        instance = module.Thing()
        path = root / artifact_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(instance, path, compress=3)
    finally:
        del sys.modules[dotted_module]
    return path.read_bytes()


def _seed_base(root: Path) -> None:
    for relative, content in _BASE_FILES.items():
        _write(root, relative, content)


def test_model_module_outside_static_closure_joins_the_snapshot(tmp_path):
    _seed_base(tmp_path)
    # engine.models._scan_fixture_thing exists in the source tree but is
    # imported by nothing -- mirrors engine/models/ensemble.py exactly:
    # present, real, but unreachable from any `import` statement.
    _write(tmp_path, "engine/models/_scan_fixture_thing.py",
          "class Thing:\n    def __init__(self):\n        self.x = 1\n")
    artifact_bytes = _dump_pickle_referencing(
        tmp_path, "data/models/thing.joblib",
        dotted_module="engine.models._scan_fixture_thing")
    _write_registry(tmp_path, model_id="thing_v1",
                    artifact_relative="data/models/thing.joblib",
                    artifact_bytes=artifact_bytes)

    manifest = worker_source_manifest(tmp_path)

    assert "engine/models/_scan_fixture_thing.py" in manifest
    # Hashed from the real source file, like every other closure entry --
    # the model scan only decides WHICH files join the manifest, never how
    # they're hashed.
    assert manifest["engine/models/_scan_fixture_thing.py"] == file_hash(
        tmp_path / "engine/models/_scan_fixture_thing.py")


def test_model_module_missing_from_source_tree_refuses_at_plan_time(tmp_path):
    _seed_base(tmp_path)
    # No engine/models/_scan_fixture_missing.py anywhere under tmp_path.
    artifact_bytes = _dump_pickle_referencing(
        tmp_path, "data/models/thing.joblib",
        dotted_module="engine.models._scan_fixture_missing")
    _write_registry(tmp_path, model_id="thing_v1",
                    artifact_relative="data/models/thing.joblib",
                    artifact_bytes=artifact_bytes)

    with pytest.raises(OpsError) as excinfo:
        worker_source_manifest(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.details["module"] == "engine.models._scan_fixture_missing"


def test_no_champion_artifacts_present_is_a_graceful_noop(tmp_path):
    """No ``data/`` at all (every non-production worktree, incl. this repo's
    own) must not turn every ``worker_source_manifest`` call into a hard
    failure -- a champion artifact's bytes not being materialized here is a
    separate, already-handled concern (``Registry.load``,
    ``reference_inputs.resolve_reference_files``)."""
    _seed_base(tmp_path)
    _write_registry(tmp_path, model_id="thing_v1",
                    artifact_relative="data/models/thing.joblib",
                    artifact_bytes=b"never written")

    manifest = worker_source_manifest(tmp_path)
    assert "engine/v2/ops/worker.py" in manifest


def test_real_repo_manifest_unaffected_without_materialized_data():
    """Sanity check against the real worktree used to run this suite: no
    ``data/`` directory here (gitignored), so the model scan must not raise
    and the manifest must still be exactly the static closure."""
    root = Path(__file__).resolve().parents[1]
    assert not (root / "data").exists()
    manifest = worker_source_manifest(root)
    assert "engine/v2/ops/worker.py" in manifest
