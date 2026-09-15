"""Declared transitive source closures and explicit change plans."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import platform
import shutil
from pathlib import Path

import joblib.numpy_pickle as _joblib_numpy_pickle
from joblib.numpy_pickle_utils import _validate_fileobject_and_memmap

from engine.v2.foundation import ensure_directory, safe_relative_path
from engine.v2.ops.errors import fail


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_closure(root, entries):
    """Resolve local static imports; callers also declare runtime/configuration edges."""
    pending, found = list(entries), {}
    while pending:
        rel = pending.pop()
        safe_relative_path(rel)
        if rel in found:
            continue
        path = root / rel
        if not path.is_file():
            raise fail("INPUT_CHANGED", "declared implementation file is missing")
        found[rel] = file_hash(path)
        if path.suffix == ".py":
            pending.extend(_imports(root, rel, path.read_text()))
    return dict(sorted(found.items()))


def worker_source_manifest(root):
    """Package initializers imported before the fixed worker module, plus
    every ``engine.*`` module a currently-materialized champion model
    artifact's pickle references but no ``.py`` file statically imports.

    A static AST closure (below) only sees code reachable by ``import``. A
    joblib-pickled model artifact can reference a class (e.g. a custom
    ensemble wrapper) that nothing ever imports by name — the model registry
    only imports :mod:`engine.models.registry`, and the pickle bytes carry
    the rest. Real evidence: the DYN-SV chooser champion's pickle GLOBALs
    ``engine.models.ensemble``, which no ``.py`` file in this tree imports.
    """
    root = Path(root)
    entries = [
        "engine/__init__.py", "engine/v2/__init__.py",
        "engine/v2/ops/__init__.py", "engine/v2/ops/worker.py",
    ]
    entries += _model_module_entries(root)
    return source_closure(root, entries)


def _module_files(root, name):
    """``a/b/c.py``/``a/b/__init__.py`` candidates existing under ``root``
    for every prefix of a dotted module name, longest prefix last."""
    parts = name.split(".")
    found = []
    for index in range(1, len(parts) + 1):
        stem = "/".join(parts[:index])
        for candidate in (stem + ".py", stem + "/__init__.py"):
            if (root / candidate).is_file():
                found.append(candidate)
    return found


def _imports(root, rel, source):
    package = rel.removesuffix(".py").replace("/", ".").split(".")[:-1]
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[:len(package) - node.level + 1] if node.level else []
            base = ".".join(prefix + ([node.module] if node.module else []))
            names.extend([base, *(base + "." + alias.name for alias in node.names)])
    paths = set()
    for name in names:
        paths.update(_module_files(root, name))
    return sorted(paths)


# --------------------------------------------------------------------------
# model artifacts: what a pinned pickle needs that no import statement shows
# --------------------------------------------------------------------------
#
# joblib writes raw ndarray bytes directly into the pickle stream, outside
# normal opcodes (``NumpyArrayWrapper.write_array``/``.read_array`` in
# ``joblib/numpy_pickle.py``) — a plain ``pickletools.genops`` walk desyncs
# the instant it reaches one. Correctly skipping those bytes needs the
# wrapper's own (trusted, side-effect-free) length/alignment logic, so this
# reuses joblib's ``NumpyUnpickler`` for the walk, but overrides
# ``find_class`` to never resolve or construct anything the pickle names
# except numpy's own types and joblib's own array wrapper — the same narrow
# trust boundary ``joblib.load`` itself relies on for array reconstruction.
# Every other class/function (the artifact's real payload: sklearn/lightgbm
# estimators, this repo's ``engine.*`` wrapper classes) is replaced by
# :class:`_Inert` before it is ever called, so no artifact-controlled code
# executes — only its *names* are recorded. This is not ``pickle.load``.

#: Modules :class:`_ScanningUnpickler` resolves for real, because their own
#: reconstruction (allocate an array, build a dtype) is trusted and — for the
#: joblib pair — is what lets the walk skip past embedded raw array bytes.
_TRUSTED_PICKLE_MODULES = ("numpy", "joblib.numpy_pickle")


class _Inert:
    """Stands in for every pickled class outside ``_TRUSTED_PICKLE_MODULES``.
    Constructing, calling or setting state on it never runs artifact code."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Inert()

    def __setstate__(self, state):
        pass

    def __reduce__(self):
        return (_Inert, ())


def _pickle_trusted(module):
    return any(module == m or module.startswith(m + ".") for m in _TRUSTED_PICKLE_MODULES)


class _ScanningUnpickler(_joblib_numpy_pickle.NumpyUnpickler):
    """Records every module a joblib artifact's pickle stream GLOBAL/
    STACK_GLOBAL-references; resolves (and therefore executes) only
    :data:`_TRUSTED_PICKLE_MODULES`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.referenced_modules: set[str] = set()

    def find_class(self, module, name):
        self.referenced_modules.add(module)
        if _pickle_trusted(module):
            return super().find_class(module, name)
        return _Inert

    def load_build(self):
        top = self.stack[-1] if self.stack else None
        if (isinstance(top, _joblib_numpy_pickle.NumpyArrayWrapper)
                and top.dtype is not None and top.dtype.hasobject):
            # The real path here recurses into a second, fully-trusting
            # pickle.load(unpickler.file_handle) — refuse instead of ever
            # calling that on artifact-controlled bytes.
            raise ValueError("object-dtype array: cannot scan without a nested pickle.load")
        super().load_build()


def _pickled_modules(path):
    """Every module a joblib artifact at ``path`` references, scanned
    without executing any of its own classes (see the section docstring)."""
    with path.open("rb") as fh, _validate_fileobject_and_memmap(fh, str(path), None) as (fobj, _mode):
        unpickler = _ScanningUnpickler(str(path), fobj, True)
        try:
            unpickler.load()
        except Exception as exc:
            raise fail("INPUT_CHANGED", "pinned model artifact could not be scanned for its "
                      "module references", details={"path": str(path), "error": str(exc)}) from exc
    return unpickler.referenced_modules


def _pinned_champion_artifacts(root):
    """Root-relative paths of champion artifacts that are actually present
    under ``root`` right now. A champion the registry declares but whose
    bytes are not materialized here (every non-production worktree: ``data/``
    is gitignored) is not this function's concern — a missing artifact is
    refused separately, where something actually needs to read it
    (``Registry.load``, ``reference_inputs.resolve_reference_files``). This
    only adds source modules for artifacts it can actually open."""
    from engine.v2.data.reference_inputs import REGISTRY_PATH, champion_artifact_paths

    registry_path = root / REGISTRY_PATH
    if not registry_path.is_file():
        return ()
    return tuple(rel for rel in champion_artifact_paths(registry_path) if (root / rel).is_file())


def _model_module_entries(root):
    """Extra :func:`source_closure` entries for every ``engine.*`` module a
    pinned, materialized champion artifact's pickle references. Refuses
    (``INPUT_CHANGED``) when a referenced module does not exist anywhere in
    the source tree — a plan must not silently ship a code snapshot its own
    pinned models cannot load."""
    modules = set()
    for rel in _pinned_champion_artifacts(root):
        modules |= {m for m in _pickled_modules(root / rel)
                   if m == "engine" or m.startswith("engine.")}
    entries = set()
    for module in sorted(modules):
        files = _module_files(root, module)
        leaf = module.replace(".", "/")
        if not any(f in (leaf + ".py", leaf + "/__init__.py") for f in files):
            raise fail("INPUT_CHANGED", "a pinned model artifact references an engine module "
                      "absent from the source tree", details={"module": module,
                                                               "artifact_paths": list(
                                                                   _pinned_champion_artifacts(root))})
        entries.update(files)
    return sorted(entries)


def environment_identity(thread_count=1):
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), machine=platform.machine(),
                libraries=versions, numerical_threads=thread_count)


def snapshot_code(root, destination, manifest):
    """Private code copy; a worker never imports from an editable checkout."""
    ensure_directory(destination)
    for rel, expected in manifest.items():
        safe_relative_path(rel)
        source, target = root / rel, destination / rel
        if file_hash(source) != expected:
            raise fail("INPUT_CHANGED", "source changed during code pinning")
        ensure_directory(target.parent)
        if not target.exists():
            shutil.copyfile(source, target)
            target.chmod(0o444)
        if file_hash(target) != expected:
            raise fail("INTEGRITY_FAILED", "pinned source content differs")


def rerun_plan(stages, previous):
    """Compare a DAG in topological order and invalidate only its changed closure."""
    decisions = {}
    pending = dict(stages)
    while pending:
        ready = [name for name, stage in pending.items()
                 if all(parent in decisions for parent in stage["dependencies"])]
        if not ready:
            raise fail("INVALID_REQUEST", "stage dependency graph is cyclic or incomplete")
        for name in ready:
            stage = pending.pop(name)
            changed = [key for key in ("inputs", "implementation", "parameters", "environment", "schema")
                       if previous.get(name, {}).get(key) != stage.get(key)]
            parents = [p for p in stage["dependencies"] if decisions[p]["action"] == "rerun"]
            decisions[name] = {"action": "rerun" if changed or parents else "reuse",
                               "reasons": changed + ["dependency:" + p for p in parents],
                               "resource_class": stage.get("resource_class")}
    return decisions
