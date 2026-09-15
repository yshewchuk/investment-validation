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


#: Every ``engine.*`` module a champion model artifact's pickle is known to
#: reference but that no ``.py`` file statically imports (real evidence: the
#: DYN-SV chooser champion's pickle GLOBALs ``engine.models.ensemble``, which
#: nothing imports — the registry only imports :mod:`engine.models.registry`,
#: and the pickle bytes carry the rest; ``engine.models.training.common``/
#: ``.runup_move`` are reached the same way by ``size_v1_4``/
#: ``runup_move_d14_v1_gbm``, though today they also happen to be reachable
#: through ``engine.models.training.train_all``'s own static imports).
#:
#: Declared here, checked into source control, rather than discovered by
#: scanning ``data/models/*.joblib`` under the CODE root: a real nightly
#: plans and snapshots its code from a frozen worktree (e.g.
#: ``/root/phase2-heavy-<commit>``) that never carries ``data/`` — only the
#: live host passed as ``--source-root`` to ``capture-inputs`` does. Scanning
#: the code root made ``worker_source_manifest`` a silent no-op in the real
#: flow and ``implementation_ref`` depend on whether ``data/`` happened to
#: sit next to the code. :func:`verify_pinned_model_modules` (called from
#: ``engine.v2.ops.capture_inputs``, against that live host) keeps this set
#: honest by refusing when a pinned artifact needs a module NOT in it.
MODEL_PICKLE_MODULES = (
    "engine.models.ensemble",
    "engine.models.registry",
    "engine.models.training.common",
    "engine.models.training.runup_move",
)

#: Non-``.py`` files legacy/v2 code reads as a ``__file__``-relative sibling
#: of its own module, rather than through ``engine.paths`` — declared here,
#: the same way :data:`MODEL_PICKLE_MODULES` is, because the AST-based
#: :func:`source_closure` walk only follows ``import``/``from`` statements
#: and a file-relative sibling read leaves no import for it to find.
#:
#: Real evidence: shadow attempt 15, ``legacy_render`` — ``_copy_static``
#: (``engine/dashboard/render.py:1090``) raised ``FileNotFoundError`` at
#: ``/root/phase2-shadow-ops/code/<impl_hash>/engine/dashboard/static``
#: because ``worker_source_manifest`` never packaged it.
#:
#: A single file goes in :data:`CODE_ASSET_FILES`; a whole directory (like
#: the dashboard's static client) goes in :data:`CODE_ASSET_DIRS` and every
#: file under it is packaged, content-hashed like a ``.py`` entry. Both are
#: git-tracked code, never anything under ``data/`` or a secret — enforced by
#: :func:`_assert_code_asset_path_safe`, not just convention.
CODE_ASSET_FILES: tuple[str, ...] = (
    "engine/v2/data/legacy_annotations.json",
)

CODE_ASSET_DIRS: tuple[str, ...] = (
    "engine/dashboard/static",
)


def worker_source_manifest(root):
    """Package initializers imported before the fixed worker module, plus
    :data:`MODEL_PICKLE_MODULES` and :data:`CODE_ASSET_FILES`/
    :data:`CODE_ASSET_DIRS`, each with their own closure — unconditional,
    independent of whether ``data/`` exists under ``root``."""
    root = Path(root)
    entries = [
        "engine/__init__.py", "engine/v2/__init__.py",
        "engine/v2/ops/__init__.py", "engine/v2/ops/worker.py",
    ]
    entries += _declared_module_entries(root, MODEL_PICKLE_MODULES)
    entries += _code_asset_entries(root)
    return source_closure(root, entries)


def _assert_code_asset_path_safe(rel):
    """Refuse a declared code asset that resolves under the repo-root
    ``data/`` store (:data:`engine.paths.DATA`) or names a secrets file —
    the code-asset closure must never carry either. Only the LEADING path
    segment is checked against ``data``: ``engine/v2/data/...`` is a code
    package, not the Tier-1/2/3 store, and must not be rejected."""
    parts = Path(rel).parts
    if parts[0] == "data" or Path(rel).name == ".env":
        raise fail("INPUT_CHANGED", "a declared code asset resolves under a data/secret path",
                  details={"path": rel})


def _code_asset_entries(root):
    """Entries for :data:`CODE_ASSET_FILES` plus every file under
    :data:`CODE_ASSET_DIRS`, refusing (``INPUT_CHANGED``) if a declared
    directory is entirely missing — a directory that merely expands to zero
    files would otherwise pass silently, unlike :func:`source_closure`'s own
    per-file check, which still covers each expanded file below."""
    entries = []
    for rel in CODE_ASSET_FILES:
        safe_relative_path(rel)
        _assert_code_asset_path_safe(rel)
        entries.append(rel)
    for rel in CODE_ASSET_DIRS:
        safe_relative_path(rel)
        _assert_code_asset_path_safe(rel)
        directory = root / rel
        if not directory.is_dir():
            raise fail("INPUT_CHANGED", "a declared code-asset directory is missing from the "
                      "source tree", details={"path": rel})
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                found = path.relative_to(root).as_posix()
                _assert_code_asset_path_safe(found)
                entries.append(found)
    return entries


def _declared_module_entries(root, modules):
    """Entries for every module in ``modules``, refusing (``INPUT_CHANGED``)
    if one is missing from the source tree — unlike :func:`_module_files`,
    which silently omits what it cannot find (right for a *discovered*
    static import, wrong for a *declared* dependency this function's caller
    cannot ship without)."""
    entries = []
    for module in modules:
        files = _module_files(root, module)
        leaf = module.replace(".", "/")
        if not any(f in (leaf + ".py", leaf + "/__init__.py") for f in files):
            raise fail("INPUT_CHANGED", "a declared model-pickle module is missing from the "
                      "source tree", details={"module": module})
        entries.extend(files)
    return entries


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


def verify_pinned_model_modules(root):
    """Refuse (``INPUT_CHANGED``) unless every champion model artifact's
    pickle references only ``engine.*`` modules already in
    :data:`MODEL_PICKLE_MODULES`.

    Called from ``engine.v2.ops.capture_inputs`` where the plan already
    resolves pinned legacy reference inputs, against the live data host
    (``--source-root``, e.g. ``/root/investing-plan``) — never against the
    frozen code-snapshot root ``worker_source_manifest`` builds from, which
    never carries ``data/models/*.joblib``. Keeps the declared set honest: a
    newly promoted champion that needs an undeclared module refuses here
    instead of reaching a worker as ``ModuleNotFoundError``.

    A pinned champion artifact this refuses to find or read is a refusal
    too — no silent skip: unlike :func:`worker_source_manifest`'s callers,
    this function only ever runs where ``data/`` is expected to exist.
    """
    from engine.v2.data.reference_inputs import REGISTRY_PATH, champion_artifact_paths

    root = Path(root)
    registry_path = root / REGISTRY_PATH
    if not registry_path.is_file():
        raise fail("INPUT_CHANGED", "model registry is missing", details={"path": REGISTRY_PATH})
    declared = set(MODEL_PICKLE_MODULES)
    for rel in champion_artifact_paths(registry_path):
        path = root / rel
        if not path.is_file():
            raise fail("INPUT_CHANGED", "a pinned champion artifact is missing",
                      details={"artifact": rel})
        found = {m for m in _pickled_modules(path) if m == "engine" or m.startswith("engine.")}
        missing = found - declared
        if missing:
            raise fail("INPUT_CHANGED", "a pinned model artifact references an engine module "
                      "outside the declared model-pickle closure",
                      details={"artifact": rel, "modules": sorted(missing)})


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
