"""Admission of the chooser analog pool's parquet source (runtime-shared).

Both ``tools/phase5_prepare_release.py`` (staging the frozen pool into a
release) and ``tools/phase5_training_job.py`` (the ``--state
chooser_analog_pool`` job) read the same
``data/features/chooser_analog_pool.parquet`` source, so its admission lives
here: a runtime module with NO ``checks`` import, which the training job's
worker path may read. ``CHOOSER_POOL_MAX_BYTES`` is this module's constant
and is read at call time by :func:`read_chooser_pool_frame`.
"""
from __future__ import annotations

import errno
import io
import os
import stat
from pathlib import Path
from typing import Iterable

import pandas as pd

from engine.v2.models.chooser_analog_pool import CHOOSER_ANALOG_DIMS

#: The chooser analog pool's id in a release (its source file, by name).
CHOOSER_POOL_ID = "features.chooser_analog_pool"

#: Read bound: the real pool is ~2.5 MB; refuse a source grown past this
#: instead of reading an unbounded frame into memory.
CHOOSER_POOL_MAX_BYTES = 256 * 1024 * 1024


def require_chooser_pool_columns(columns: Iterable[str]) -> None:
    """Refuse a pool source missing any column the artifact builder reads.

    ``build_chooser_analog_pool_artifact`` filters via ``row.get``, so a
    missing column would silently drop EVERY row and freeze an empty pool.
    """
    names = set(columns)
    if not all(name in names for name in ("strategy", "exit_date", "pnl",
                                          *CHOOSER_ANALOG_DIMS)):
        raise ValueError("chooser analog pool source lacks required columns")


def read_chooser_pool_frame(path: Path) -> pd.DataFrame:
    """Bounded single-handle read of the chooser analog pool parquet source.

    Opens the fd once (``O_NOFOLLOW``: a symlinked source is refused, never
    followed; ``O_NONBLOCK``: a FIFO cannot wedge the read), checks the fd's
    own ``fstat`` so the size bound and the read describe the same object the
    type check saw, and parses at most ``st_size + 1`` bytes -- more arriving
    means the file grew since the check, so it is refused as too large. A
    plainly missing file raises ``FileNotFoundError``; a non-regular or
    oversized source raises ``ValueError``. The cap is the module constant,
    read at call time.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.ELOOP:
            raise ValueError("chooser analog pool source is not a regular file") from None
        raise
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("chooser analog pool source is not a regular file")
        if info.st_size > CHOOSER_POOL_MAX_BYTES:
            raise ValueError("chooser analog pool source is too large")
        limit = info.st_size + 1
        chunks, read = [], 0
        while read < limit:
            block = os.read(fd, limit - read)
            if not block:
                break
            chunks.append(block)
            read += len(block)
    finally:
        os.close(fd)
    if read > info.st_size:
        raise ValueError("chooser analog pool source is too large")
    return pd.read_parquet(io.BytesIO(b"".join(chunks)))
