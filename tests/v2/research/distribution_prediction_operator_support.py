"""Shared synthetic prediction request and read-only WAL snapshot fixtures."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import subprocess
import sys


def request():
    return {"schema_version": "distribution_prediction_request.v1", "max_events": 50,
        "limits": {"parent": {"memory_bytes": 4 << 30, "cpu_seconds": 60, "wall_seconds": 60},
            "job_wall_seconds": 60},
        "uncertainty": {"seed": 0, "resamples": 4, "block_days": 7,
            "block_origin": "2023-01-01", "confidence": .95}}


def catalog_bytes(path):
    """Inspect bytes outside the process that owns SQLite's POSIX locks.

    Even a raw read/close before the pin can drop an existing connection's
    locks without updating SQLite's bookkeeping on older SQLite versions.
    """
    return subprocess.run([sys.executable, "-c",
        "import pathlib, sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())",
        str(path)], stdout=subprocess.PIPE, check=True, timeout=10).stdout


@contextmanager
def pinned_wal_snapshot(catalog):
    """Keep a real read snapshot so new commits cannot checkpoint into main."""
    reader = sqlite3.connect(Path(catalog).resolve().as_uri() + "?mode=ro",
        uri=True, isolation_level=None)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM sqlite_schema").fetchone()
        yield reader
    finally:
        reader.rollback()
        reader.close()
