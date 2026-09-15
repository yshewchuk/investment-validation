"""The plan's code snapshot must cover what pinned model artifacts load, not
just what ``.py`` files statically import -- and it must do so from a code
root that never carries ``data/``, since that is how a real nightly plans.

Real shadow nightly attempt 13 (root ``/root/phase2-shadow-ops``, job
``legacy_score``): the DYN-SV chooser champion's joblib pickle references
``engine.models.ensemble`` via a bare STACK_GLOBAL opcode. Nothing in this
repo's ``.py`` files imports that module by name (only ``engine.score``'s
static import of ``engine.models.registry`` reaches the registry, never the
custom ensemble class a pickle instantiates), so the static AST closure in
``engine/v2/ops/fingerprints.py::source_closure`` never picked it up.

First fix attempt (send-back): scanned champion artifacts under the CODE
root passed to ``worker_source_manifest``. Wrong: a real nightly plans and
snapshots its code from a frozen worktree (e.g. ``/root/phase2-heavy-
<commit>``) that never has ``data/`` next to it -- only the live host passed
to ``capture-inputs --source-root`` does. Scanning the code root made the
fix a silent no-op in the real flow.

Current design:

* ``fingerprints.MODEL_PICKLE_MODULES`` is a declared, checked-in tuple.
  ``worker_source_manifest`` always includes it and its own import closure,
  independent of whether ``data/`` exists under ``root`` -- a deterministic
  ``implementation_ref``, and a missing declared module now refuses
  ``INPUT_CHANGED`` (``_declared_module_entries``) instead of being silently
  dropped the way a *discovered* static import would be.
* ``fingerprints.verify_pinned_model_modules(root)`` opcode-scans the real
  pinned champion artifacts against a root that DOES carry ``data/`` (called
  from ``engine.v2.ops.capture_inputs._resolve_reference_bundle``, against
  ``--source-root``) and refuses ``INPUT_CHANGED`` if any references an
  ``engine.*`` module outside the declared set, or if a pinned artifact is
  missing. This is what keeps the declared set honest.
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
from engine.v2.ops.fingerprints import (
    CODE_ASSET_DIRS,
    CODE_ASSET_FILES,
    MODEL_PICKLE_MODULES,
    file_hash,
    verify_pinned_model_modules,
    worker_source_manifest,
)

#: Every worker_source_manifest entry the fixed worker entrypoint needs,
#: mirrored as trivial stubs so source_closure's own static walk is a no-op
#: beyond them (these tests are only about the declared model-pickle set).
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
    # worker_source_manifest also requires the declared non-.py code assets
    # (tests/test_v2_ops_fingerprints_code_assets.py covers those directly;
    # this file is only about MODEL_PICKLE_MODULES, so seed trivial stand-ins).
    for relative in CODE_ASSET_FILES:
        _write(root, relative, "{}")
    for relative in CODE_ASSET_DIRS:
        _write(root, f"{relative}/index.html", "<html></html>")


# --------------------------------------------------------------------------
# worker_source_manifest: declared set, unconditional on data/
# --------------------------------------------------------------------------


def test_declared_model_modules_join_the_snapshot_without_any_data_dir(tmp_path):
    """No ``data/`` anywhere under ``tmp_path`` (the real frozen-worktree
    shape) -- the declared modules must still join the manifest."""
    _seed_base(tmp_path)
    _write(tmp_path, "engine/models/ensemble.py", "class MeanEnsemble:\n    pass\n")
    _write(tmp_path, "engine/models/registry.py", "")
    _write(tmp_path, "engine/models/training/__init__.py", "")
    _write(tmp_path, "engine/models/training/common.py", "")
    _write(tmp_path, "engine/models/training/runup_move.py", "")
    assert not (tmp_path / "data").exists()

    manifest = worker_source_manifest(tmp_path)

    for module in MODEL_PICKLE_MODULES:
        rel = module.replace(".", "/") + ".py"
        assert rel in manifest, f"{rel} missing from manifest"
        assert manifest[rel] == file_hash(tmp_path / rel)


def test_declared_model_module_missing_from_source_tree_refuses_at_plan_time(tmp_path):
    _seed_base(tmp_path)
    _write(tmp_path, "engine/models/registry.py", "")
    _write(tmp_path, "engine/models/training/__init__.py", "")
    _write(tmp_path, "engine/models/training/common.py", "")
    _write(tmp_path, "engine/models/training/runup_move.py", "")
    # No engine/models/ensemble.py: a declared module absent from the tree.

    with pytest.raises(OpsError) as excinfo:
        worker_source_manifest(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.details["module"] == "engine.models.ensemble"


def test_manifest_identical_with_and_without_a_data_dir(tmp_path):
    """implementation_ref must not depend on whether data/ happens to sit
    next to the code -- the exact silent-no-op failure mode this design
    replaces."""
    _seed_base(tmp_path)
    for module in MODEL_PICKLE_MODULES:
        _write(tmp_path, module.replace(".", "/") + ".py", "")
    without_data = worker_source_manifest(tmp_path)

    artifact_bytes = _dump_pickle_referencing(
        tmp_path, "data/models/thing.joblib", dotted_module="engine.models.ensemble")
    _write_registry(tmp_path, model_id="thing_v1", artifact_relative="data/models/thing.joblib",
                    artifact_bytes=artifact_bytes)
    with_data = worker_source_manifest(tmp_path)

    assert without_data == with_data


# --------------------------------------------------------------------------
# verify_pinned_model_modules: opcode-scan against a data-bearing root
# --------------------------------------------------------------------------


def test_verify_accepts_a_champion_artifact_inside_the_declared_set(tmp_path):
    artifact_bytes = _dump_pickle_referencing(
        tmp_path, "data/models/thing.joblib", dotted_module="engine.models.ensemble")
    _write_registry(tmp_path, model_id="thing_v1", artifact_relative="data/models/thing.joblib",
                    artifact_bytes=artifact_bytes)

    verify_pinned_model_modules(tmp_path)  # must not raise


def test_verify_refuses_a_champion_artifact_outside_the_declared_set(tmp_path):
    artifact_bytes = _dump_pickle_referencing(
        tmp_path, "data/models/thing.joblib", dotted_module="engine.models._scan_fixture_undeclared")
    _write_registry(tmp_path, model_id="thing_v1", artifact_relative="data/models/thing.joblib",
                    artifact_bytes=artifact_bytes)

    with pytest.raises(OpsError) as excinfo:
        verify_pinned_model_modules(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.details["modules"] == ["engine.models._scan_fixture_undeclared"]


def test_verify_refuses_a_missing_pinned_artifact(tmp_path):
    _write_registry(tmp_path, model_id="thing_v1", artifact_relative="data/models/thing.joblib",
                    artifact_bytes=b"never written")

    with pytest.raises(OpsError) as excinfo:
        verify_pinned_model_modules(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"


def test_verify_refuses_a_missing_registry(tmp_path):
    with pytest.raises(OpsError) as excinfo:
        verify_pinned_model_modules(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"


# --------------------------------------------------------------------------
# guard: the declared set stays a superset of what real champions reference
# --------------------------------------------------------------------------


def test_declared_set_covers_the_real_champion_artifacts_when_data_is_present():
    """Scans the REAL champion artifacts under ``/root/investing-plan`` when
    that host is reachable from this box (skips cleanly in CI/worktrees
    without a materialized ``data/``, exactly like the production frozen
    worktree). Keeps MODEL_PICKLE_MODULES honest against drift: a future
    champion needing a new module should fail THIS test, loudly, in review --
    not surface as a ModuleNotFoundError in a worker."""
    live_root = Path("/root/investing-plan")
    if not (live_root / "data" / "models").is_dir():
        pytest.skip("no materialized data/models/ next to this box's source tree")

    verify_pinned_model_modules(live_root)  # exercises the same assertion
