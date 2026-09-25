"""Per-input parse cache for the incremental Tier-2 rebuild.

The legacy nightly's rebuild is parse-bound: it re-reads and re-parses every
raw file on every run although almost none of them changed. This module keeps
the parsed result of each raw input, keyed on a manifest of
``(path, size, mtime_ns, sha256)``, so a later run can skip the parse of an
input whose bytes are unchanged.

Only per-input work is cached. Anything that depends on the whole universe
(cross-source dedupe, the market-wide daily index, validation of a ticker's
combined series, the partition writes themselves) is recomputed by the caller
every run.

Safety rules, each of which makes a run behave as a full rebuild (every input
re-parsed, the cache rewritten):

* no manifest, an unreadable or malformed manifest, or a manifest written by a
  different :func:`code_version` (the parsing code, the pickle-relevant library
  versions, or :data:`CACHE_SCHEMA`);
* ``INVESTING_PLAN_REBUILD_FULL=1`` in the environment, or ``force=True``.

A single entry that fails to load, or whose header does not match the manifest,
is treated as a miss and re-parsed. Every write is atomic (tmp + rename).

The cache lives under ``data/cache/rebuild/<table>/`` (gitignored with the rest
of ``data/``). It is opt-in: :func:`engine.data.rebuild.rebuild` only uses it
when called with ``incremental=True``.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine import paths

__all__ = [
    "CACHE_SCHEMA",
    "FORCE_ENV",
    "CODE_FILES",
    "InputCache",
    "Signature",
    "code_version",
    "cache_root",
]

#: Bump to invalidate every cache written by an earlier layout.
CACHE_SCHEMA = 1

#: Set to ``1`` to ignore the cache (a full rebuild that rewrites it).
FORCE_ENV = "INVESTING_PLAN_REBUILD_FULL"

#: Source files whose content decides what a cached parse holds. A change to
#: any of them invalidates the whole cache.
CODE_FILES = (
    "engine/data/rebuild.py",
    "engine/data/rebuild_cache.py",
    "engine/data/rebuild_incremental.py",
    "engine/data/normalize/n_daily.py",
    "engine/data/normalize/n_chains.py",
    "engine/data/normalize/common.py",
    "engine/data/normalize/fetch_store.py",
    "engine/data/fetch.py",
    "engine/data/validate.py",
    "engine/data/schemas.py",
    "engine/data/store.py",
)

#: Errors a damaged or foreign pickle can raise on load. Any of them makes the
#: entry a miss; nothing else is swallowed.
_LOAD_ERRORS = (
    OSError,
    EOFError,
    pickle.UnpicklingError,
    ValueError,
    TypeError,
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
)

_MANIFEST = "manifest.json"


def cache_root() -> Path:
    return paths.DATA / "cache" / "rebuild"


def _library_versions() -> list[str]:
    import numpy
    import pandas

    versions = [f"python={platform.python_version()}", f"numpy={numpy.__version__}",
                f"pandas={pandas.__version__}"]
    try:
        import pyarrow

        versions.append(f"pyarrow={pyarrow.__version__}")
    except ImportError:  # pragma: no cover - environment probe
        versions.append("pyarrow=absent")
    return versions


def code_version() -> str:
    """Hash of the cache schema, library versions and the parsing code."""
    code_root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256(f"schema={CACHE_SCHEMA}".encode())
    for item in _library_versions():
        h.update(item.encode())
    for rel in CODE_FILES:
        h.update(rel.encode())
        h.update((code_root / rel).read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Signature:
    """What the manifest records about one raw input."""

    size: int
    mtime_ns: int
    sha256: str

    def as_dict(self) -> dict:
        return {"size": self.size, "mtime_ns": self.mtime_ns, "sha256": self.sha256}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


class InputCache:
    """Manifest plus one pickled parse per raw input, for one table."""

    def __init__(self, table: str, *, root: Path | None = None, force: bool = False):
        self.table = table
        self.dir = paths.assert_writable(Path(root) if root is not None else cache_root() / table)
        self.version = code_version()
        self.stats = {"reused": 0, "parsed": 0, "hashed": 0, "uncached": 0, "removed": 0}
        self.fallback_reason: str | None = None
        forced = force or os.environ.get(FORCE_ENV, "") not in ("", "0")
        self._old = {} if forced else self._load_manifest()
        if forced:
            self.fallback_reason = "forced"
        self._new: dict[str, dict] = {}

    # -- manifest ----------------------------------------------------------

    def _load_manifest(self) -> dict:
        path = self.dir / _MANIFEST
        if not path.exists():
            self.fallback_reason = "no manifest"
            return {}
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            self.fallback_reason = f"unreadable manifest: {type(exc).__name__}"
            return {}
        if not isinstance(doc, dict) or not isinstance(doc.get("inputs"), dict):
            self.fallback_reason = "malformed manifest"
            return {}
        if doc.get("version") != self.version:
            self.fallback_reason = "code version changed"
            return {}
        return doc["inputs"]

    @property
    def full(self) -> bool:
        """True when this run cannot reuse anything (a full rebuild)."""
        return self.fallback_reason is not None

    def _entry_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.dir / "entries" / digest[:2] / f"{digest}.pkl"

    # -- per input ---------------------------------------------------------

    def signature(self, path: Path) -> Signature:
        """The input's signature, hashing its bytes only if size/mtime moved."""
        st = path.stat()
        rec = self._old.get(str(path))
        if rec and rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns:
            return Signature(st.st_size, st.st_mtime_ns, str(rec.get("sha256")))
        self.stats["hashed"] += 1
        return Signature(st.st_size, st.st_mtime_ns, _sha256(path))

    def lookup(self, path: Path, source_id: str) -> tuple[bool, Any, Signature]:
        """``(hit, value, signature)`` for one raw input."""
        sig = self.signature(path)
        key = str(path)
        rec = self._old.get(key)
        if not rec or rec.get("sha256") != sig.sha256 or rec.get("source_id") != source_id:
            return False, None, sig
        entry = self._load_entry(key)
        if (
            not isinstance(entry, dict)
            or entry.get("version") != self.version
            or entry.get("sha256") != sig.sha256
            or entry.get("source_id") != source_id
            or "value" not in entry
        ):
            return False, None, sig
        self._new[key] = {**sig.as_dict(), "source_id": source_id}
        self.stats["reused"] += 1
        return True, entry["value"], sig

    def _load_entry(self, key: str) -> Any:
        try:
            with open(self._entry_path(key), "rb") as fh:
                return pickle.load(fh)
        except _LOAD_ERRORS:
            return None

    def store(self, path: Path, sig: Signature, source_id: str, value: Any) -> None:
        """Record a fresh parse of ``path``."""
        key = str(path)
        entry = {"version": self.version, "sha256": sig.sha256,
                 "source_id": source_id, "value": value}
        _atomic_write(self._entry_path(key), pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
        self._new[key] = {**sig.as_dict(), "source_id": source_id}
        self.stats["parsed"] += 1

    def note_uncached(self) -> None:
        """Count an input parsed this run but deliberately not cached."""
        self.stats["uncached"] += 1

    # -- end of run --------------------------------------------------------

    def commit(self) -> None:
        """Write the manifest for this run and drop entries for vanished inputs."""
        stale = [k for k in self._old if k not in self._new]
        for key in stale:
            self._entry_path(key).unlink(missing_ok=True)
        self.stats["removed"] = len(stale)
        doc = {"version": self.version, "table": self.table, "inputs": self._new}
        _atomic_write(self.dir / _MANIFEST, json.dumps(doc, sort_keys=True).encode())

    def summary(self) -> str:
        mode = f"full ({self.fallback_reason})" if self.full else "incremental"
        s = self.stats
        return (f"cache {self.table}: {mode}; reused {s['reused']:,}, parsed {s['parsed']:,}, "
                f"uncached {s['uncached']:,}, hashed {s['hashed']:,}, removed {s['removed']:,}")

