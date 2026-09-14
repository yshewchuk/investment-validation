"""Pinned legacy reference inputs for snapshot-backed scoring — guide §14, D13/D14.

Guide §14: "Exact legacy snapshot metadata and model/registry artifacts remain
separately pinned compatibility inputs." :data:`LEGACY_REFERENCE_INPUTS_V1`
names every such input, as a path or a pattern relative to
``engine.paths.ROOT``. Every path comes from a ``legacy_adapter`` accessor
over the legacy constant that defines it; no path string is typed here.

Resolution at import planning (:func:`resolve_reference_files`, called by
``import_snapshot.plan_import`` so every file is leased and pinned):

* ``exact`` kinds: the one file at that path.
* ``champion_artifact``: every registry entry with ``champion: true`` names
  its own ``artifact`` path and ``artifact_sha256``. The path must be relative
  and sit inside the models directory, outside the Tier-4 serving directory.
  The file's sha256 must equal the recorded digest.
* ``tier4_serving_cache``: only files directly inside the serving directory
  whose name ends ``_<sha256(panel.parquet)[:12]>.joblib``, for the panel
  bytes this same import pins.

Refusal codes:

* ``INPUT_CHANGED``: a required file is missing or is a symlink. This covers
  an exact input and a registry-referenced champion artifact, and matches how
  ``plan_import`` already treats a missing table file.
* ``CONTRACT_MISMATCH``: the registry is not the reviewed shape, an artifact
  path escapes the models directory, or an artifact's bytes differ from its
  recorded sha256.
* ``TIER4_CACHE_STALE``: a champion Tier-4 feature model (``produces`` set)
  has zero serving caches for this panel hash.

The coordinator publishes these files outside the commit transaction
(:func:`publish_reference_inputs`), and ``catalog.commit_snapshot`` records
them per import receipt (``reference_catalog``). They never enter
``import_snapshot.request_hash``, so they never enter snapshot identity.

Write survey (static trace; no behaviour change)
------------------------------------------------
Every write reachable from ``engine.features.FeatureContext.load``,
``engine.score.Scorer.__init__``/``.score`` and ``engine.score.score_calendar``
when they run against a read-only materialized root. The trace covers every
``to_parquet``/``joblib.dump``/``write_text``/``write_bytes``/``open(w|a)``/
``mkdir``/``paths.assert_writable`` site in the modules those entry points
import (score, features, calendar, replay, analogs, audit, entry_rules, fills,
forecast_sizing, payoff, structures, structure_registry, models.registry,
models.training.*, data.store, data.manifest, data.fetch, data.throttle,
data.features.panel/tier4).

**No write is unconditional on the normal scoring path.** One write is
reachable, and its trigger is ordinary:

1. ``engine/data/features/tier4.py:1320`` ``paths.assert_writable(SERVING_DIR)
   .mkdir(parents=True, exist_ok=True)``, then ``:1321`` ``joblib.dump(...)``
   to ``data/models/tier4/<model_id>_<fold:%Y%m>_<panel_sha256[:12]>.joblib``.
   Path: ``Scorer._serving`` (``engine/score.py:1967``) calls
   ``tier4.serving_model(fold, panel=..., model=...)`` with the default
   ``cache=True``, once per (fold, producer) the board touches. It fires on a
   serving-cache MISS: no file for that exact (model_id, fold, panel hash), or
   a file whose stored model_id/fold_start/tier3_snapshot/features disagree.
   ``assert_writable`` passes (``data/`` is not grandfathered), so on a
   read-only root the failure is a ``PermissionError`` from ``mkdir`` (if the
   directory was never materialized) or from ``joblib.dump``. Pinning every
   cache for the panel hash, and refusing an import with none for a champion
   model, narrows this. It does not close it: a board whose serving fold has
   no cache file still writes.

Present but not reachable from those entry points:

2. ``engine/data/fetch.py:415-447`` (Tier-1 body/meta write, fetch-log
   append) and ``engine/data/throttle.py:238,248`` (Polygon lock file): only
   inside a network fetch or ``Throttle.acquire``. ``engine/data/features/
   panel.py:456`` constructs a ``Fetcher`` only to compute a cached body path
   (read-only; ``Fetcher.__init__``/``Throttle.__init__`` write nothing), and
   only in the panel build.
3. ``engine/data/store.py:99-109`` (``_write_frame``/partition ``mkdir``) and
   ``:295`` (``rmtree``): ``write_table``/table delete only. Scoring calls
   ``read_table``/``iter_table``/``file_sha256``.
4. ``engine/data/manifest.py:62-64`` (``write_snapshot``) and ``:138-188``
   (``write_manifest``): rebuild only. Scoring calls ``read_snapshot``
   (``engine/score.py:2928``).
5. ``engine/data/features/tier4.py:1031-1037`` (``write_forecasts``) and
   ``:1140`` (CLI report): the Tier-4 build only.
6. ``engine/models/registry.py:236-238`` (``ModelArtifact.save``) and
   ``:623-624``/``:671`` (``Registry.save``/registration): training and
   promotion only. Scoring calls ``load_registry``/``load_champion``.
7. ``engine/structure_registry.py:201-202``: champion promotion only.
8. ``engine/paths.py:196`` (``ensure_dirs``): called only by
   ``engine/data/rebuild.py:399``.
9. ``engine/ledger.py`` (every write): not imported by any of the four entry
   points. ``score_calendar`` returns a frame, and its callers write the ledger.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from engine.v2.contracts import LegacyFileRef
from engine.v2.foundation import CONTENT_HASH_PREFIX

from . import errors, legacy_adapter, objects
from . import reference_catalog as catalog_rows

__all__ = [
    "DATA_DIR",
    "LEGACY_REFERENCE_INPUTS_V1",
    "LEGACY_SNAPSHOT_PATH",
    "TIER4_SERVING_DIR",
    "kind_for_path",
    "manifest_pins",
    "publish_reference_inputs",
    "resolve_reference_files",
]

DATA_DIR = legacy_adapter.legacy_data_dir()
LEGACY_SNAPSHOT_PATH = legacy_adapter.legacy_snapshot_path()
TIER4_SERVING_DIR = legacy_adapter.legacy_tier4_serving_dir()
_MODELS_DIR = legacy_adapter.legacy_models_dir()
_REGISTRY_PATH = legacy_adapter.legacy_registry_path()

#: kind -> where the input lives. ``exact`` entries name one path;
#: the other two name a directory plus the rule that selects files in it.
LEGACY_REFERENCE_INPUTS_V1: dict[str, object] = {
    "schema_version": "legacy_reference_inputs.v1",
    "root": "engine.paths.ROOT",
    "inputs": {
        catalog_rows.CALENDAR_KIND: {
            "resolution": "exact", "path": legacy_adapter.legacy_calendar_path()},
        "model_registry": {"resolution": "exact", "path": _REGISTRY_PATH},
        "structure_champions": {
            "resolution": "exact", "path": legacy_adapter.legacy_structures_path()},
        "chooser_analog_pool": {
            "resolution": "exact", "path": legacy_adapter.legacy_chooser_pool_path()},
        catalog_rows.LEGACY_SNAPSHOT_KIND: {"resolution": "exact", "path": LEGACY_SNAPSHOT_PATH},
        "champion_artifact": {
            "resolution": "registry_champion_artifacts", "directory": _MODELS_DIR},
        "tier4_serving_cache": {
            "resolution": "panel_hash_suffix", "directory": TIER4_SERVING_DIR,
            "pattern": "<model_id>_<fold:%Y%m>_<sha256(panel.parquet)[:12]>.joblib"},
    },
}

_INPUTS: dict[str, dict] = LEGACY_REFERENCE_INPUTS_V1["inputs"]  # type: ignore[assignment]
_EXACT = {spec["path"]: kind for kind, spec in _INPUTS.items() if spec["resolution"] == "exact"}
_CACHE_NAME = re.compile(r"^[^/]+_[0-9a-f]{12}\.joblib$")


def kind_for_path(path: str) -> str | None:
    """The reference kind a legacy-relative path belongs to, or ``None``."""
    if path in _EXACT:
        return _EXACT[path]
    parent, _, name = path.rpartition("/")
    if parent == TIER4_SERVING_DIR and _CACHE_NAME.match(name):
        return "tier4_serving_cache"
    if path.startswith(_MODELS_DIR + "/") and not path.startswith(TIER4_SERVING_DIR + "/"):
        return "champion_artifact"
    return None


# --------------------------------------------------------------------------
# plan time: resolve every reference file under a pinned legacy root
# --------------------------------------------------------------------------


def resolve_reference_files(root: Path, *, panel_content_hash: str,
                            file_ref) -> tuple[LegacyFileRef, ...]:
    """Every reference file under ``root``, sorted by path.

    ``file_ref(root, relative)`` is ``import_snapshot``'s own hashing
    enumerator: it refuses a missing or symlinked file with ``INPUT_CHANGED``.
    """
    refs = {path: file_ref(root, path) for path in _EXACT}
    champions = _champion_entries(root / _REGISTRY_PATH)
    for entry in champions:
        path = _artifact_path(entry)
        ref = file_ref(root, path)
        if ref.content_hash != CONTENT_HASH_PREFIX + str(entry.get("artifact_sha256")):
            raise errors.fail("CONTRACT_MISMATCH", "champion artifact bytes differ from the "
                              "registry's recorded sha256", details={"path": path})
        refs[path] = ref
    for path in _tier4_caches(root, champions, panel_content_hash):
        refs[path] = file_ref(root, path)
    return tuple(refs[path] for path in sorted(refs))


def _champion_entries(registry_path: Path) -> list[dict]:
    try:
        document = json.loads(registry_path.read_text())
        models = document["models"]
        if not isinstance(models, list) or not all(isinstance(m, dict) for m in models):
            raise TypeError("models is not a list of objects")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise errors.fail("CONTRACT_MISMATCH", "legacy model registry is not the reviewed shape",
                          details={"path": _REGISTRY_PATH}) from exc
    return [entry for entry in models if entry.get("champion") is True]


def _artifact_path(entry: dict) -> str:
    raw = entry.get("artifact")
    candidate = PurePosixPath(raw) if isinstance(raw, str) and raw else None
    path = candidate.as_posix() if candidate is not None else ""
    inside = (candidate is not None and not candidate.is_absolute() and ".." not in candidate.parts
              and path.startswith(_MODELS_DIR + "/") and kind_for_path(path) == "champion_artifact")
    if not inside:
        raise errors.fail("CONTRACT_MISMATCH", "champion artifact path is not inside the models "
                          "directory", details={"model_id": entry.get("id"), "artifact": raw})
    return path


def _tier4_caches(root: Path, champions: list[dict], panel_content_hash: str) -> list[str]:
    suffix = "_" + panel_content_hash.removeprefix(CONTENT_HASH_PREFIX)[:12] + ".joblib"
    directory = root / TIER4_SERVING_DIR
    if directory.is_symlink():
        raise errors.fail("INPUT_CHANGED", "Tier-4 serving directory is a symlink",
                          details={"path": TIER4_SERVING_DIR})
    names = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    matching = [name for name in names if name.endswith(suffix)]
    for entry in champions:
        if entry.get("produces") is None:
            continue
        prefix = f"{entry.get('id')}_"
        if not any(name.startswith(prefix) and re.fullmatch(r"\d{6}", name[len(prefix):-len(suffix)])
                   for name in matching):
            raise errors.fail("TIER4_CACHE_STALE", "a champion Tier-4 model has no serving cache "
                              "for this panel", details={"model_id": entry.get("id"),
                                                         "suffix": suffix})
    return [f"{TIER4_SERVING_DIR}/{name}" for name in matching]


def manifest_pins(refs) -> tuple[tuple[str, ...], str | None]:
    """``LegacyInputManifest.registry_and_model_refs`` and ``.calendar_ref`` for ``refs``."""
    from .legacy_materialization import format_pinned_ref

    registry, calendar = [], None
    for ref in refs:
        kind = kind_for_path(ref.path)
        if kind == catalog_rows.CALENDAR_KIND:
            calendar = format_pinned_ref(ref.path, ref.content_hash)
        elif kind not in (None, catalog_rows.LEGACY_SNAPSHOT_KIND):
            registry.append(format_pinned_ref(ref.path, ref.content_hash))
    return tuple(registry), calendar


# --------------------------------------------------------------------------
# coordinator: publish what the import pinned, before the commit transaction
# --------------------------------------------------------------------------


def publish_reference_inputs(store, attempt_id: str, legacy_root, file_refs, *,
                             keepalive=None) -> tuple[catalog_rows.ReferenceInput, ...]:
    """Publish every reference file in ``file_refs`` from the staged legacy root.

    Refuses with ``CONTRACT_MISMATCH`` if any ``exact`` input is absent: a
    manifest that was not built by :func:`resolve_reference_files`.
    ``keepalive``, when given, is called before each file is published.
    """
    keepalive = keepalive or (lambda: None)
    published = []
    for ref in file_refs:
        kind = kind_for_path(ref.path)
        if kind is None:
            continue
        keepalive()
        obj = objects.publish_legacy_file(store, attempt_id, legacy_root, ref)
        published.append(catalog_rows.ReferenceInput(
            kind=kind, legacy_path=ref.path, object_id=obj.object_id,
            content_hash=obj.content_hash, byte_size=obj.byte_size))
    missing = sorted(set(_EXACT) - {item.legacy_path for item in published})
    if missing:
        raise errors.fail("CONTRACT_MISMATCH", "import manifest lacks required reference inputs",
                          details={"paths": missing})
    return tuple(published)
