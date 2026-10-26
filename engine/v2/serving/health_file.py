"""Bounded, single-open read of a health artifact (no TOCTOU, no symlink, no FIFO)."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

#: Maximum bytes a health artifact may hold; one more byte is read so an
#: oversize artifact is refused instead of streamed whole into memory.
HEALTH_MAX_BYTES = 1_048_576


def read_document(path: Path) -> dict:
    """Open once with O_NOFOLLOW|O_NONBLOCK, fstat the descriptor, read cap plus one byte, parse once."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError("health artifact is indirect") from exc
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("health artifact is indirect")
        raw = handle.read(HEALTH_MAX_BYTES + 1)
    if len(raw) > HEALTH_MAX_BYTES:
        raise ValueError("health artifact exceeds the size limit")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("unsupported health artifact")
    return document
