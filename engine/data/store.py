"""Tier-2 storage: partitioned, idempotent, content-hashed.

Tables live at ``data/curated/{table}/year=YYYY/part.parquet``. Partitioning by
year is what makes a rebuild affordable: a normalizer that only touched 2024
rewrites one partition instead of six million rows, and a consumer that only
needs 2018–2020 reads three files.

Writes are idempotent by construction — a partition is written to a temporary
name and moved into place, so a rebuild interrupted halfway leaves either the
old partition or the new one, never a half-written file. Re-running a rebuild
produces byte-identical partitions, which is what the determinism check asserts.

Parquet via pyarrow is the format; ``csv.gz`` is the fallback for an
environment where pyarrow cannot be installed. Both satisfy the same contracts,
and :func:`table_format` reports which is in use so a report's provenance block
can record it.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.data.schemas import SCHEMAS, assert_schema, coerce, empty_frame

__all__ = [
    "HAVE_PARQUET",
    "table_format",
    "write_table",
    "write_partition",
    "PartitionedWriter",
    "read_table",
    "iter_table",
    "table_years",
    "table_stats",
    "TableStats",
    "drop_table",
    "file_sha256",
]

try:  # pragma: no cover - environment probe
    import pyarrow  # noqa: F401

    HAVE_PARQUET = True
except ImportError:  # pragma: no cover
    HAVE_PARQUET = False

SUFFIX = ".parquet" if HAVE_PARQUET else ".csv.gz"


def table_format() -> str:
    return "parquet" if HAVE_PARQUET else "csv.gz"


def _part_name(part: int) -> str:
    return f"part-{part:04d}{SUFFIX}"


def _partition_file(part_dir: Path, part: int = 0) -> Path:
    return part_dir / _part_name(part)


def _partition_files(part_dir: Path) -> list[Path]:
    """Every part file in a partition, in deterministic order."""
    if not part_dir.exists():
        return []
    return sorted(
        p for p in part_dir.iterdir() if p.name.startswith("part-") and p.name.endswith(SUFFIX)
    )


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if HAVE_PARQUET:
        # Repeated writes of identical data must be byte-identical (the
        # determinism check relies on it), which pyarrow's defaults give us as
        # long as nothing here injects a timestamp or a row-order dependency.
        df.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    else:
        with gzip.open(tmp, "wt", newline="") as fh:
            df.to_csv(fh, index=False)
    os.replace(tmp, path)


def write_partition(df: pd.DataFrame, name: str, year: int, part: int = 0) -> Path:
    """Write one part file of one year partition, replacing any existing one."""
    part_dir = paths.assert_writable(paths.curated_partition(name, year))
    part_dir.mkdir(parents=True, exist_ok=True)
    path = _partition_file(part_dir, part)
    _write_frame(df.reset_index(drop=True), path)
    return path


class PartitionedWriter:
    """Streaming writer for tables too large to hold in memory at once.

    ``daily_market`` is ~9.4M rows and ``option_chains`` ~6.5M, against ~6 GB of
    usable RAM: materializing either as one frame is not an option. Batches are
    accumulated per year and flushed as numbered part files, so the writer's
    peak memory is one batch rather than one table.

    Determinism holds as long as the caller feeds batches in a fixed order —
    the rebuild iterates sorted source files, so it does. Each batch is sorted
    on the table's primary key before it is written.

    Use as a context manager; ``__exit__`` flushes whatever is buffered.
    """

    #: Total buffered rows across *all* years before a flush. The cap has to be
    #: global: a per-year threshold never trips when input arrives ticker by
    #: ticker, because each ticker spreads ~160 rows across 20 years, so the
    #: whole 9.4M-row table would be resident before any single year hit its
    #: limit.
    MAX_BUFFERED_ROWS = 500_000

    def __init__(
        self,
        name: str,
        *,
        validate: bool = True,
        replace: bool = True,
        max_buffered_rows: int | None = None,
    ):
        self.name = name
        self.schema = SCHEMAS[name]
        self.validate = validate
        self.max_buffered_rows = max_buffered_rows or self.MAX_BUFFERED_ROWS
        self.rows_written = 0
        self.flushes = 0
        self._buffers: dict[int, list[pd.DataFrame]] = {}
        self._buffered_rows: dict[int, int] = {}
        self._parts: dict[int, int] = {}
        self._touched: set[int] = set()
        if replace:
            drop_table(name)

    @property
    def buffered_rows(self) -> int:
        return sum(self._buffered_rows.values())

    def add(self, df: pd.DataFrame) -> None:
        """Buffer a frame, flushing every year once the global cap is reached."""
        if df is None or len(df) == 0:
            return
        out = coerce(df, self.name)
        part_col = self.schema.partition_by
        for year, chunk in out.groupby(part_col, sort=True):
            year = int(year)
            self._buffers.setdefault(year, []).append(chunk)
            self._buffered_rows[year] = self._buffered_rows.get(year, 0) + len(chunk)
        if self.buffered_rows >= self.max_buffered_rows:
            self.flush()

    def flush(self) -> None:
        """Write every buffered year out as the next part file."""
        for year in sorted(list(self._buffers)):
            self._flush_year(year)
        self.flushes += 1

    def _flush_year(self, year: int) -> None:
        chunks = self._buffers.pop(year, None)
        self._buffered_rows.pop(year, None)
        if not chunks:
            return
        frame = pd.concat(chunks, ignore_index=True)
        frame = frame.sort_values(list(self.schema.primary_key), kind="stable")
        if self.validate:
            # Key uniqueness cannot be asserted per part file: the same key can
            # legitimately arrive in two batches when two source payloads
            # overlap. `finalize(dedupe=True)` resolves it across the whole
            # table once every batch has been written.
            assert_schema(frame, self.name, check_keys=False)
        part = self._parts.get(year, 0)
        write_partition(frame, self.name, year, part)
        self._parts[year] = part + 1
        self._touched.add(year)
        self.rows_written += len(frame)

    def close(self) -> None:
        self.flush()

    def finalize(self, *, dedupe: bool = True) -> int:
        """Compact each year into one part file, optionally deduplicating.

        A streamed build cannot enforce primary-key uniqueness as it goes: two
        source payloads can legitimately carry the same contract (entry-date and
        calendar pulls overlap on a trade date), and they may land in different
        batches. Uniqueness is therefore a whole-table property, resolved here
        in a second pass — one year at a time, so peak memory stays at one
        partition rather than one table.

        The first occurrence wins, and batches are fed in sorted source order,
        so which row survives is a function of the source set rather than of
        scheduling. Returns the number of rows removed.

        This also compacts the numbered part files a streamed write leaves
        behind, which makes later reads cheaper and the content hash stable
        against changes in flush timing.
        """
        self.flush()
        removed = 0
        for year in sorted(self._touched):
            part_dir = paths.curated_partition(self.name, year)
            parts = _partition_files(part_dir)
            if not parts:
                continue
            frame = pd.concat([_read_part(p, None) for p in parts], ignore_index=True)
            before = len(frame)
            if dedupe and self.schema.primary_key:
                frame = frame.drop_duplicates(
                    subset=list(self.schema.primary_key), keep="first"
                )
            frame = frame.sort_values(list(self.schema.primary_key), kind="stable")
            removed += before - len(frame)
            for stale in parts:
                stale.unlink(missing_ok=True)
            write_partition(frame, self.name, year, 0)
            self._parts[year] = 1
        self.rows_written -= removed
        return removed

    def __enter__(self) -> "PartitionedWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def years(self) -> list[int]:
        return sorted(self._touched)


def write_table(
    df: pd.DataFrame,
    name: str,
    *,
    validate: bool = True,
    replace: bool = True,
) -> dict[int, Path]:
    """Coerce, validate, and write ``df`` as year partitions.

    ``replace=True`` drops partitions that the new frame does not cover, so a
    rebuild cannot leave a stale year behind. That is the difference between a
    table that is *rebuilt* and one that is merely *appended to*.
    """
    schema = SCHEMAS[name]
    out = coerce(df, name)
    if validate:
        assert_schema(out, name)

    if schema.partition_by is None:
        raise ValueError(f"{name} is not partitioned")
    part_col = schema.partition_by
    if part_col not in out.columns:
        raise ValueError(f"{name}: partition column {part_col!r} absent")

    if replace:
        # Drop first rather than diffing: a previous streamed write may have
        # left several part files in a year this frame now covers with one, and
        # a stale part-0001 would silently double-count rows on the next read.
        drop_table(name)

    written: dict[int, Path] = {}
    if len(out):
        # Sorting by primary key makes partition bytes a function of content
        # only — not of the order rows happened to arrive in.
        out = out.sort_values(list(schema.primary_key), kind="stable")
        for year, chunk in out.groupby(part_col, sort=True):
            written[int(year)] = write_partition(chunk, name, int(year))
    return written


def drop_table(name: str) -> None:
    shutil.rmtree(paths.assert_writable(paths.curated_table(name)), ignore_errors=True)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def table_years(name: str) -> list[int]:
    root = paths.curated_table(name)
    if not root.exists():
        return []
    years = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("year="):
            try:
                years.append(int(child.name.split("=", 1)[1]))
            except ValueError:
                continue
    return sorted(years)


def _read_part(path: Path, columns: Sequence[str] | None) -> pd.DataFrame:
    """Read one partition, tolerating a partition older than the schema.

    A column added to a Tier-2 schema does not exist in partitions written
    before it — the option_chains liquidity fields (volume, open interest, bid
    and ask size) are the first case, present only from the 2026-09 pull. A
    reader asking for one used to get ``ArrowInvalid`` from every earlier
    partition, which would have forced a full rebuild of a 15M-row table to
    read a column that is NaN there anyway. Missing columns are filled with
    NaN instead, so "this partition predates the field" reads the same as
    "this row has no value" — which is exactly what it means.
    """
    wanted = list(columns) if columns else None
    if HAVE_PARQUET:
        if wanted is None:
            return pd.read_parquet(path)
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
        present = [c for c in wanted if c in available]
        frame = pd.read_parquet(path, columns=present)
    else:
        with gzip.open(path, "rt") as fh:
            frame = pd.read_csv(fh)
        if wanted is None:
            return frame
        present = [c for c in wanted if c in frame.columns]
        frame = frame[present]
    for missing in [c for c in wanted if c not in frame.columns]:
        frame[missing] = np.nan
    return frame[wanted]


def iter_table(
    name: str,
    *,
    years: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
):
    """Yield ``(year, frame)`` one partition at a time.

    The way to touch a table too big to hold in memory. Consumers that only
    need a rolling window or a per-year aggregate should use this rather than
    :func:`read_table`.
    """
    if name not in SCHEMAS:
        raise KeyError(f"unknown table {name!r}; known: {sorted(SCHEMAS)}")
    wanted = sorted(set(years)) if years is not None else table_years(name)
    for year in wanted:
        parts = _partition_files(paths.curated_partition(name, year))
        if not parts:
            continue
        frame = pd.concat([_read_part(p, columns) for p in parts], ignore_index=True)
        yield year, frame


def read_table(
    name: str,
    *,
    years: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a Tier-2 table, optionally restricted to years and columns."""
    frames = [frame for _, frame in iter_table(name, years=years, columns=columns)]
    if not frames:
        base = empty_frame(name)
        return base[list(columns)] if columns else base
    out = pd.concat(frames, ignore_index=True)
    return coerce(out, name, only=list(columns) if columns else None)


# --------------------------------------------------------------------------
# stats / manifest input
# --------------------------------------------------------------------------


@dataclass
class TableStats:
    name: str
    rows: int
    years: list[int]
    files: int
    bytes: int
    content_hash: str
    fmt: str

    def as_dict(self) -> dict:
        return {
            "table": self.name,
            "rows": self.rows,
            "years": f"{min(self.years)}–{max(self.years)}" if self.years else "—",
            "partitions": len(self.years),
            "files": self.files,
            "bytes": self.bytes,
            "content_hash": self.content_hash,
            "format": self.fmt,
        }


def table_stats(name: str) -> TableStats:
    """Row counts, coverage, and a content hash for the manifest.

    The content hash is over the *partition file digests*, not over a
    concatenated frame: it is cheap on a six-million-row table and it changes
    if and only if some partition's bytes changed.
    """
    years = table_years(name)
    digests: list[str] = []
    total_bytes = 0
    rows = 0
    files = 0
    for year in years:
        for path in _partition_files(paths.curated_partition(name, year)):
            files += 1
            total_bytes += path.stat().st_size
            digests.append(f"{year}/{path.name}:{file_sha256(path)}")
            if HAVE_PARQUET:
                import pyarrow.parquet as pq

                rows += pq.ParquetFile(path).metadata.num_rows
            else:
                with gzip.open(path, "rt") as fh:
                    rows += max(0, sum(1 for _ in fh) - 1)
    content_hash = hashlib.sha256("|".join(digests).encode()).hexdigest()
    return TableStats(
        name=name,
        rows=rows,
        years=years,
        files=files,
        bytes=total_bytes,
        content_hash=content_hash,
        fmt=table_format(),
    )
