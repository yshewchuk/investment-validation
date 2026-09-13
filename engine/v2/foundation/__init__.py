"""paths, env, canonical JSON, session/calendar arithmetic, causality primitives

Layer 0 of `system_rearchitecture.md` §4.1 (0.5 in ``checks/layer_map.py``:
foundation imports contracts, so the two cannot be peers). Replaces
`paths.py`, `env.py`, `jsonio.py`, `audit.py`, `session arithmetic from
calendar.py`.

Rearchitecture phase 1 writes the first production code here: the canonical
hash (moved from diagnosis, §3.2), strict typed documents, clocks, and durable
content-addressed artifacts. See ``README.md`` for the public interface.
"""
from __future__ import annotations

from engine.v2.foundation.artifacts import (
    ArtifactError,
    ArtifactStore,
    ensure_directory,
    fsync_directory,
    safe_relative_path,
)
from engine.v2.foundation.canonical import (
    CONTENT_HASH_PREFIX,
    canonical_json,
    content_hash,
)
from engine.v2.foundation.clock import (
    Clock,
    SystemClock,
    format_timestamp,
    parse_timestamp,
)
from engine.v2.foundation.typed import (
    DocumentError,
    from_document,
    parse_schema_version,
    to_document,
)

__all__ = [
    "CONTENT_HASH_PREFIX",
    "ArtifactError",
    "ArtifactStore",
    "Clock",
    "DocumentError",
    "SystemClock",
    "canonical_json",
    "content_hash",
    "ensure_directory",
    "format_timestamp",
    "from_document",
    "fsync_directory",
    "parse_schema_version",
    "parse_timestamp",
    "safe_relative_path",
    "to_document",
]
